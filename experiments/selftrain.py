from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from TetrisGym_updated import TetrisGym
from diffusion.build_diffusion_sequences import build_diffusion_sequences
from diffusion.diffusion_model_updated import PlanDenoiser
from diffusion.diffusion_planner_updated import DQNCritic, DiffusionMPCPlanner, PlannerCfg
from diffusion.diffusion_utils_updated import board_feature_summary
from diffusion.filter_expert_dataset import filter_expert_dataset
from diffusion.run_diffusion_mpc_updated import run_eval as run_diffusion_eval
from diffusion.train_diffusion_updated import run_train
from baselines.run_beam_search import run_eval as run_beam_eval
from experiments.config_utils import parse_with_config
from experiments.hashing import sha256_file, write_json
from experiments.repro import prepare_run_artifacts


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ensure_ckpt(path: str, width: int, height: int, horizon: int, d_model: int, layers: int, heads: int) -> str:
    if path and os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model = PlanDenoiser(
        board_h=height,
        board_w=width,
        horizon=horizon,
        d_model=d_model,
        n_layers=layers,
        n_heads=heads,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "width": width,
            "height": height,
            "horizon": horizon,
            "d_model": d_model,
            "layers": layers,
            "heads": heads,
        },
        path,
    )
    return path


def _save_transitions_npz(rows: List[dict], out_path: str, episodes: int, width: int, height: int, max_steps_per_episode: int) -> str:
    if not rows:
        raise RuntimeError("No rollout transitions were collected.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    data = {
        "episode_id": np.array([r["episode_id"] for r in rows], dtype=np.int64),
        "t": np.array([r["t"] for r in rows], dtype=np.int64),
        "board": np.stack([r["board"] for r in rows]).astype(np.uint8),
        "curr_id": np.array([r["curr_id"] for r in rows], dtype=np.int64),
        "next_id": np.array([r["next_id"] for r in rows], dtype=np.int64),
        "action_id": np.array([r["action_id"] for r in rows], dtype=np.int64),
        "action_rot": np.array([r["action_rot"] for r in rows], dtype=np.int64),
        "action_x": np.array([r["action_x"] for r in rows], dtype=np.int64),
        "reward": np.array([r["reward"] for r in rows], dtype=np.float32),
        "done": np.array([r["done"] for r in rows], dtype=bool),
        "holes_before": np.array([r["holes_before"] for r in rows], dtype=np.float32),
        "holes_after": np.array([r["holes_after"] for r in rows], dtype=np.float32),
        "bumpiness_before": np.array([r["bumpiness_before"] for r in rows], dtype=np.float32),
        "bumpiness_after": np.array([r["bumpiness_after"] for r in rows], dtype=np.float32),
        "max_height_before": np.array([r["max_height_before"] for r in rows], dtype=np.float32),
        "max_height_after": np.array([r["max_height_after"] for r in rows], dtype=np.float32),
        "lines_before": np.array([r["lines_before"] for r in rows], dtype=np.float32),
        "lines_after": np.array([r["lines_after"] for r in rows], dtype=np.float32),
        "lines_cleared": np.array([r["lines_cleared"] for r in rows], dtype=np.float32),
        "q_sa": np.array([r["q_sa"] for r in rows], dtype=np.float32),
        "q_max": np.array([r["q_max"] for r in rows], dtype=np.float32),
        "advantage": np.array([r["advantage"] for r in rows], dtype=np.float32),
        "invalid_fallback": np.array([r["invalid_fallback"] for r in rows], dtype=bool),
        "episode_return_so_far": np.array([r["episode_return_so_far"] for r in rows], dtype=np.float32),
        "teacher": np.array(["diffusion_policy"]),
        "width": np.array([width], dtype=np.int64),
        "height": np.array([height], dtype=np.int64),
        "episodes": np.array([episodes], dtype=np.int64),
        "max_steps_per_episode": np.array([max_steps_per_episode], dtype=np.int64),
    }
    np.savez_compressed(out_path, **data)
    return out_path


def rollout_diffusion_policy_transitions(
    ckpt_path: str,
    out_path: str,
    episodes: int,
    seed: int,
    width: int,
    height: int,
    max_steps_per_episode: int,
    device: str,
    horizon: int,
    num_candidates: int,
    sample_steps: int,
    temperature: float,
    sampling_constraints: str,
    rerank_mode: str,
    critic_ckpt: str,
    invalid_handling: str,
    invalid_penalty: float,
) -> str:
    _set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    env = TetrisGym(width=width, height=height, max_steps=max_steps_per_episode, render_mode="skip", seed=seed)

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = PlanDenoiser(
        board_h=height,
        board_w=width,
        horizon=horizon,
        d_model=int(ckpt.get("d_model", 128)),
        n_layers=int(ckpt.get("layers", 4)),
        n_heads=int(ckpt.get("heads", 4)),
    ).to(dev)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    critic = DQNCritic(critic_ckpt, board_h=height, board_w=width, device=dev) if os.path.exists(critic_ckpt) else None
    planner = DiffusionMPCPlanner(
        model=model,
        cfg=PlannerCfg(
            horizon=horizon,
            num_candidates=num_candidates,
            sample_steps=sample_steps,
            temperature=temperature,
            sampling_constraints=sampling_constraints,
            rerank_mode=rerank_mode,
            invalid_handling=invalid_handling,
            invalid_penalty=invalid_penalty,
        ),
        device=dev,
        critic=critic if rerank_mode == "dqn" else None,
    )

    rows: List[dict] = []
    for ep in tqdm(range(episodes), desc="SelfTrain rollout"):
        obs = env.reset()
        done = False
        t = 0
        ep_return = 0.0
        prev_score = 0.0

        while not done and t < max_steps_per_episode:
            valid = env.get_valid_action_ids()
            if not valid:
                break

            board = obs.board.copy()
            curr_id = int(obs.curr_id)
            next_id = int(obs.next_id)
            before = board_feature_summary(board)

            aid, _, _ = planner.plan(env, (obs.board, obs.curr_id, obs.next_id))
            invalid_fallback = False
            if aid not in valid:
                invalid_fallback = True
                aid = valid[0]

            q_sa = np.nan
            q_max = np.nan
            advantage = np.nan
            if critic is not None:
                q_vals = critic.q_values(board, curr_id=curr_id, next_id=next_id)
                q_sa = float(q_vals[aid].item())
                q_max = float(q_vals[valid].max().item())
                advantage = q_sa - q_max

            rot_idx, xpos = env.id_to_action[int(aid)]
            next_obs, _, done, info = env.step(int(aid))
            score_now = float(info.get("score", prev_score))
            reward = score_now - prev_score
            prev_score = score_now
            ep_return += reward

            after = board_feature_summary(next_obs.board)
            rows.append(
                {
                    "episode_id": int(ep),
                    "t": int(t),
                    "board": board.astype(np.uint8),
                    "curr_id": curr_id,
                    "next_id": next_id,
                    "action_id": int(aid),
                    "action_rot": int(rot_idx),
                    "action_x": int(xpos),
                    "reward": float(reward),
                    "done": bool(done),
                    "holes_before": float(before["holes"]),
                    "holes_after": float(after["holes"]),
                    "bumpiness_before": float(before["bumpiness"]),
                    "bumpiness_after": float(after["bumpiness"]),
                    "max_height_before": float(before["max_height"]),
                    "max_height_after": float(after["max_height"]),
                    "lines_before": float(before["lines"]),
                    "lines_after": float(after["lines"]),
                    "lines_cleared": float(info.get("lines_cleared", 0)),
                    "q_sa": float(q_sa),
                    "q_max": float(q_max),
                    "advantage": float(advantage),
                    "invalid_fallback": bool(invalid_fallback),
                    "episode_return_so_far": float(ep_return),
                }
            )

            obs = next_obs
            t += 1

    return _save_transitions_npz(rows, out_path, episodes, width, height, max_steps_per_episode)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Iterative self-training for diffusion planner.")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--output_dir", type=str, default="runs/selftrain")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--episodes_per_iter", type=int, default=2000)
    p.add_argument("--max_steps_per_episode", type=int, default=1000)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--critic_ckpt", type=str, default="dqn_updated.pt")
    p.add_argument("--init_ckpt", type=str, default="checkpoints/plan_denoiser.pt")
    p.add_argument("--from_scratch", action="store_true")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--top_episode_pct", type=float, default=0.10)
    p.add_argument("--advantage_quantile", type=float, default=0.80)
    p.add_argument("--board_health", action="store_true")
    p.add_argument("--min_steps_per_episode", type=int, default=1)
    p.add_argument("--rerank_mode", type=str, default="dqn", choices=["heuristic", "dqn"])
    p.add_argument("--sampling_constraints", type=str, default="mask_logits", choices=["none", "mask_logits"])
    p.add_argument("--invalid_handling", type=str, default="penalize", choices=["none", "penalize", "resample"])
    p.add_argument("--invalid_penalty", type=float, default=1e6)
    p.add_argument("--num_candidates", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--train_epochs", type=int, default=3)
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--train_lr", type=float, default=3e-4)
    p.add_argument("--train_mask_prob", type=float, default=0.5)
    p.add_argument("--train_d_model", type=int, default=128)
    p.add_argument("--train_layers", type=int, default=4)
    p.add_argument("--train_heads", type=int, default=4)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--eval_max_steps", type=int, default=2000)
    p.add_argument("--beam_eval_episodes", type=int, default=50)
    p.add_argument("--beam_eval_max_steps", type=int, default=2000)
    p.add_argument("--beam_width", type=int, default=16)
    p.add_argument("--beam_horizon", type=int, default=3)
    p.add_argument("--diversity_enable", action="store_true")
    p.add_argument("--diversity_buckets", type=int, default=4)
    p.add_argument("--diversity_per_bucket_cap", type=int, default=10)
    p.add_argument("--smoke", action="store_true")
    return p


def run_selftrain(args: argparse.Namespace) -> str:
    prepare_run_artifacts(args.output_dir, vars(args).copy())
    _set_seed(int(args.seed))

    if args.smoke:
        args.iterations = 1
        args.episodes_per_iter = min(int(args.episodes_per_iter), 10)
        args.train_epochs = 1
        args.eval_episodes = min(int(args.eval_episodes), 5)
        args.beam_eval_episodes = min(int(args.beam_eval_episodes), 3)
        args.eval_max_steps = min(int(args.eval_max_steps), 200)
        args.beam_eval_max_steps = min(int(args.beam_eval_max_steps), 100)
        args.beam_width = min(int(args.beam_width), 4)
        args.beam_horizon = min(int(args.beam_horizon), 2)

    current_ckpt = args.init_ckpt
    if args.from_scratch or (not current_ckpt) or (not os.path.exists(current_ckpt)):
        bootstrap = os.path.join(args.output_dir, "iter_0", "train", "checkpoints", "ckpt.pt")
        current_ckpt = _ensure_ckpt(
            bootstrap,
            width=int(args.width),
            height=int(args.height),
            horizon=int(args.horizon),
            d_model=int(args.train_d_model),
            layers=int(args.train_layers),
            heads=int(args.train_heads),
        )

    progress_rows: List[dict] = []
    progress_csv = os.path.join(args.output_dir, "progress.csv")

    for i in range(int(args.iterations)):
        iter_dir = os.path.join(args.output_dir, f"iter_{i}")
        rollout_dir = os.path.join(iter_dir, "rollout")
        dataset_dir = os.path.join(iter_dir, "dataset")
        next_dir = os.path.join(args.output_dir, f"iter_{i+1}")
        train_dir = os.path.join(next_dir, "train")
        eval_dir = os.path.join(next_dir, "eval")
        beam_dir = os.path.join(next_dir, "beam_eval")
        os.makedirs(rollout_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(beam_dir, exist_ok=True)

        raw_path = os.path.join(rollout_dir, "raw_transitions.npz")
        filtered_path = os.path.join(rollout_dir, "filtered_transitions.npz")
        seq_path = os.path.join(dataset_dir, f"sequences_H{int(args.horizon)}.npz")
        ckpt_out = os.path.join(train_dir, "checkpoints", "ckpt.pt")

        rollout_diffusion_policy_transitions(
            ckpt_path=current_ckpt,
            out_path=raw_path,
            episodes=int(args.episodes_per_iter),
            seed=int(args.seed) + i,
            width=int(args.width),
            height=int(args.height),
            max_steps_per_episode=int(args.max_steps_per_episode),
            device=str(args.device),
            horizon=int(args.horizon),
            num_candidates=int(args.num_candidates),
            sample_steps=int(args.sample_steps),
            temperature=float(args.temperature),
            sampling_constraints=str(args.sampling_constraints),
            rerank_mode=str(args.rerank_mode),
            critic_ckpt=str(args.critic_ckpt),
            invalid_handling=str(args.invalid_handling),
            invalid_penalty=float(args.invalid_penalty),
        )

        filter_expert_dataset(
            in_path=raw_path,
            out_path=filtered_path,
            top_episode_pct=float(args.top_episode_pct),
            advantage_quantile=float(args.advantage_quantile),
            board_health=bool(args.board_health),
            min_steps_per_episode=int(args.min_steps_per_episode),
            diversity_enable=bool(args.diversity_enable),
            diversity_buckets=int(args.diversity_buckets),
            diversity_per_bucket_cap=int(args.diversity_per_bucket_cap),
        )

        build_diffusion_sequences(
            in_path=filtered_path,
            out_path=seq_path,
            horizon=int(args.horizon),
            stride=int(args.stride),
        )
        dataset_hash = sha256_file(seq_path)
        write_json(
            os.path.join(dataset_dir, "meta.json"),
            {
                "dataset_path": seq_path,
                "dataset_hash": dataset_hash,
                "iter": i,
            },
        )

        train_summary = run_train(
            SimpleNamespace(
                config="",
                data="",
                dataset_path=seq_path,
                horizon=int(args.horizon),
                epochs=int(args.train_epochs),
                batch_size=int(args.train_batch_size),
                lr=float(args.train_lr),
                mask_prob=float(args.train_mask_prob),
                d_model=int(args.train_d_model),
                layers=int(args.train_layers),
                heads=int(args.train_heads),
                device=str(args.device),
                output_dir=train_dir,
                save=ckpt_out,
                init_ckpt="" if bool(args.from_scratch) else current_ckpt,
                seed=int(args.seed) + i,
            )
        )

        diff_summary = run_diffusion_eval(
            SimpleNamespace(
                config="",
                ckpt=ckpt_out,
                episodes=int(args.eval_episodes),
                max_steps=int(args.eval_max_steps),
                width=int(args.width),
                height=int(args.height),
                horizon=int(args.horizon),
                num_candidates=int(args.num_candidates),
                sample_steps=int(args.sample_steps),
                temperature=float(args.temperature),
                sampling_constraints=str(args.sampling_constraints),
                rerank_mode=str(args.rerank_mode),
                critic_ckpt=str(args.critic_ckpt),
                invalid_handling=str(args.invalid_handling),
                invalid_penalty=float(args.invalid_penalty),
                resample_retries=3,
                seed=int(args.seed) + i,
                device=str(args.device),
                output_dir=eval_dir,
                run_name=f"iter_{i+1}_diffusion",
                bootstrap_ci=0,
                bootstrap_samples=1000,
            )
        )

        beam_summary = run_beam_eval(
            SimpleNamespace(
                config="",
                episodes=int(args.beam_eval_episodes),
                max_steps=int(args.beam_eval_max_steps),
                width=int(args.width),
                height=int(args.height),
                beam_width=int(args.beam_width),
                horizon=int(args.beam_horizon),
                seed=int(args.seed) + i,
                run_name=f"iter_{i+1}_beam",
                output_dir=beam_dir,
                bootstrap_ci=0,
                bootstrap_samples=1000,
            )
        )

        filtered = np.load(filtered_path)
        seq_data = np.load(seq_path)
        progress_rows.append(
            {
                "iter": i + 1,
                "raw_transitions": int(np.load(raw_path)["episode_id"].shape[0]),
                "filtered_transitions": int(filtered["episode_id"].shape[0]),
                "sequences_built": int(seq_data["board"].shape[0]),
                "dataset_hash": dataset_hash,
                "diffusion_mean_score": float(diff_summary["mean_score"]),
                "diffusion_invalid_rate": float(diff_summary["invalid_rate"]),
                "diffusion_runtime_ms": float(diff_summary["mean_decision_ms"]),
                "diffusion_masked_fraction": float(diff_summary.get("mean_masked_fraction", 0.0)),
                "beam_mean_score": float(beam_summary["mean_score"]),
                "train_final_loss": float(train_summary["final_mean_loss"]),
            }
        )

        with open(progress_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(progress_rows[0].keys()))
            writer.writeheader()
            writer.writerows(progress_rows)

        current_ckpt = ckpt_out

    print(f"Self-training completed. Progress: {progress_csv}")
    return progress_csv


def main() -> None:
    parser = build_parser()
    args, _ = parse_with_config(parser)
    run_selftrain(args)


if __name__ == "__main__":
    main()
