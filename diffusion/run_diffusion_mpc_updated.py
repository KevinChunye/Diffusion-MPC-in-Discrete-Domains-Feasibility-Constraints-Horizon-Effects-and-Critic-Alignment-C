"""run_diffusion_mpc_updated.py

Evaluate a trained PlanDenoiser using diffusion-style MPC.

Example:
  python run_diffusion_mpc_updated.py \
    --ckpt checkpoints/plan_denoiser.pt \
    --episodes 100 --max_steps 2000 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
from tqdm import tqdm

import torch

# Allow running as `python diffusion/run_diffusion_mpc_updated.py` from repo root.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (THIS_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from TetrisGym_updated import TetrisGym
from diffusion_model_updated import PlanDenoiser
from diffusion_planner_updated import DiffusionMPCPlanner, DQNCritic, PlannerCfg
from experiments.config_utils import parse_with_config
from experiments.manifest import write_manifest
from experiments.metrics import MetricsLogger
from experiments.repro import prepare_run_artifacts
from experiments.video_utils import save_video, select_episode_indices


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--ckpt", type=str, default="checkpoints/plan_denoiser.pt")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--num_candidates", type=int, default=64)
    p.add_argument("--sample_steps", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sampling_constraints", type=str, default="none", choices=["none", "mask_logits"])
    p.add_argument("--rerank_mode", type=str, default="heuristic", choices=["heuristic", "dqn"])
    p.add_argument("--critic_ckpt", type=str, default="dqn_updated.pt")
    p.add_argument("--invalid_handling", type=str, default="none", choices=["none", "penalize", "resample"])
    p.add_argument("--invalid_penalty", type=float, default=1e6)
    p.add_argument("--resample_retries", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="results/diffusion_mpc")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--bootstrap_ci", type=int, default=0)
    p.add_argument("--bootstrap_samples", type=int, default=1000)
    p.add_argument("--record_video", type=str, nargs="?", const="true", default="false")
    p.add_argument("--video_episodes", type=int, default=1)
    p.add_argument("--video_max_steps", type=int, default=200)
    p.add_argument("--video_format", type=str, default="gif", choices=["gif", "mp4"])
    p.add_argument("--video_select", type=str, default="first", choices=["first", "best", "median", "worst"])
    return p


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return bool(v)


def _record_frame(env, frames, info, max_frames: int) -> None:
    if max_frames <= 0 or len(frames) >= max_frames:
        return
    frame = env.render(info=info, mode="rgb_array")
    if frame is not None:
        frames.append(frame)


def _run_episode(env, planner, max_steps: int, run_name: str, ep_num: int, decision_rows: list | None, record_frames: bool, video_max_steps: int):
    obs = env.reset()
    done = False
    steps = 0
    ep_decision_ms = []
    ep_masked_fracs = []
    ep_regrets = []
    invalid_count = 0
    decision_count = 0
    frames = []
    if record_frames:
        _record_frame(env, frames, info=None, max_frames=video_max_steps)

    while not done and steps < max_steps:
        valid = env.get_valid_action_ids()
        if not valid:
            break
        t0 = time.perf_counter()
        aid, _, _ = planner.plan(env, (obs.board, obs.curr_id, obs.next_id))
        t1 = time.perf_counter()
        decision_ms = 1000.0 * (t1 - t0)
        ep_decision_ms.append(decision_ms)
        ep_masked_fracs.append(float(planner.last_plan_stats.get("masked_fraction", 0.0)))
        best_roll = float(planner.last_plan_stats.get("best_candidate_rollout_score", 0.0))
        chosen_roll = float(planner.last_plan_stats.get("chosen_rollout_score", 0.0))
        regret = float(planner.last_plan_stats.get("regret", max(0.0, best_roll - chosen_roll)))
        ep_regrets.append(regret)
        if decision_rows is not None:
            decision_rows.append(
                {
                    "run_name": run_name,
                    "episode": ep_num,
                    "step": steps + 1,
                    "best_candidate_rollout_score": best_roll,
                    "chosen_rollout_score": chosen_roll,
                    "regret": regret,
                }
            )
        decision_count += 1

        if aid not in valid:
            invalid_count += 1
            aid = valid[0]
        obs, _, done, info = env.step(aid)
        steps += 1
        if record_frames:
            _record_frame(env, frames, info=info, max_frames=video_max_steps)

    return {
        "score": float(env.game.score),
        "steps": int(steps),
        "decision_ms": ep_decision_ms,
        "invalid_count": int(invalid_count),
        "decision_count": int(decision_count),
        "masked_fraction": float(np.mean(ep_masked_fracs)) if ep_masked_fracs else 0.0,
        "regret": float(np.mean(ep_regrets)) if ep_regrets else 0.0,
        "regrets": ep_regrets,
        "frames": frames,
    }


def run_eval(args: argparse.Namespace) -> dict:
    if not args.ckpt:
        raise ValueError("--ckpt is required")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    artifacts = prepare_run_artifacts(args.output_dir, vars(args).copy())

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # build model
    model = PlanDenoiser(board_h=args.height, board_w=args.width, horizon=args.horizon).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    critic = None
    if args.rerank_mode == "dqn":
        critic = DQNCritic(
            ckpt_path=args.critic_ckpt,
            board_h=args.height,
            board_w=args.width,
            device=device,
        )

    cfg = PlannerCfg(
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        sample_steps=args.sample_steps,
        temperature=args.temperature,
        sampling_constraints=args.sampling_constraints,
        rerank_mode=args.rerank_mode,
        invalid_handling=args.invalid_handling,
        invalid_penalty=args.invalid_penalty,
        resample_retries=args.resample_retries,
    )
    planner = DiffusionMPCPlanner(model, cfg, device=device, critic=critic)

    env = TetrisGym(width=args.width, height=args.height, max_steps=args.max_steps, render_mode="skip")

    run_name = args.run_name or (
        f"{args.rerank_mode}_{args.invalid_handling}"
        f"_K{args.num_candidates}_H{args.horizon}_S{args.sample_steps}_T{args.temperature:g}"
    )
    logger = MetricsLogger(run_name=run_name)
    eval_start = time.perf_counter()
    decision_rows = []
    all_regrets = []

    record_video = _as_bool(args.record_video)
    video_count = min(int(args.video_episodes), int(args.episodes))
    selected_episode_1based = []
    selected_scores = []
    video_files = []
    videos_dir = os.path.join(args.output_dir, "videos")
    if record_video:
        os.makedirs(videos_dir, exist_ok=True)

    ep_scores = []
    first_targets = set(select_episode_indices(range(args.episodes), video_count, "first")) if record_video else set()

    for ep in tqdm(range(args.episodes), desc="Diffusion-MPC"):
        rec_now = bool(record_video and args.video_select == "first" and ep in first_targets)
        ep_result = _run_episode(
            env=env,
            planner=planner,
            max_steps=int(args.max_steps),
            run_name=run_name,
            ep_num=ep + 1,
            decision_rows=decision_rows,
            record_frames=rec_now,
            video_max_steps=int(args.video_max_steps),
        )
        ep_scores.append(float(ep_result["score"]))
        all_regrets.extend(ep_result["regrets"])
        logger.add_episode(
            episode=ep + 1,
            score=float(ep_result["score"]),
            steps=int(ep_result["steps"]),
            decision_ms=ep_result["decision_ms"],
            invalid_count=int(ep_result["invalid_count"]),
            decision_count=int(ep_result["decision_count"]),
            extra={
                "masked_fraction": float(ep_result["masked_fraction"]),
                "regret": float(ep_result["regret"]),
            },
        )
        if rec_now and ep_result["frames"]:
            out_file = os.path.join(videos_dir, f"{run_name}_ep{ep+1:04d}.{args.video_format}")
            saved = save_video(ep_result["frames"], out_file, fmt=args.video_format, fps=5)
            video_files.append(saved)
            selected_episode_1based.append(ep + 1)
            selected_scores.append(float(ep_result["score"]))

    if record_video and args.video_select != "first" and video_count > 0:
        replay_targets = select_episode_indices(ep_scores, video_count, args.video_select)
        replay_set = set(replay_targets)
        if replay_set:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            env_replay = TetrisGym(
                width=args.width,
                height=args.height,
                max_steps=args.max_steps,
                render_mode="skip",
                seed=int(args.seed),
            )
            replay_planner = DiffusionMPCPlanner(model, cfg, device=device, critic=critic)
            max_target = max(replay_set)
            for ep in range(max_target + 1):
                rec_now = ep in replay_set
                ep_result = _run_episode(
                    env=env_replay,
                    planner=replay_planner,
                    max_steps=int(args.max_steps),
                    run_name=run_name,
                    ep_num=ep + 1,
                    decision_rows=None,
                    record_frames=rec_now,
                    video_max_steps=int(args.video_max_steps),
                )
                if rec_now and ep_result["frames"]:
                    out_file = os.path.join(videos_dir, f"{run_name}_{args.video_select}_ep{ep+1:04d}.{args.video_format}")
                    saved = save_video(ep_result["frames"], out_file, fmt=args.video_format, fps=5)
                    video_files.append(saved)
                    selected_episode_1based.append(ep + 1)
                    selected_scores.append(float(ep_result["score"]))

    video_files_s = ";".join(video_files)
    selected_ep_s = ";".join(str(x) for x in selected_episode_1based)
    selected_score_s = ";".join(f"{x:.3f}" for x in selected_scores)

    os.makedirs(args.output_dir, exist_ok=True)
    episodes_csv_name = f"{run_name}_episodes.csv"
    summary = logger.write(
        args.output_dir,
        summary_extra={
            "rerank_mode": args.rerank_mode,
            "invalid_handling": args.invalid_handling,
            "num_candidates": args.num_candidates,
            "horizon": args.horizon,
            "sample_steps": args.sample_steps,
            "temperature": args.temperature,
            "sampling_constraints": args.sampling_constraints,
            "episodes_csv": os.path.join(args.output_dir, episodes_csv_name),
            "total_eval_walltime_sec": float(time.perf_counter() - eval_start),
            "seed": int(args.seed),
            "mean_regret": float(np.mean(all_regrets)) if all_regrets else 0.0,
            "p90_regret": float(np.percentile(np.array(all_regrets, dtype=np.float32), 90)) if all_regrets else 0.0,
            "video_files": video_files_s,
            "selected_episode": selected_ep_s,
            "final_score": selected_score_s,
        },
        append_summary=True,
        metrics_filename=episodes_csv_name,
        bootstrap_ci=bool(int(args.bootstrap_ci)),
        bootstrap_samples=int(args.bootstrap_samples),
    )
    shutil.copyfile(
        os.path.join(args.output_dir, episodes_csv_name),
        os.path.join(args.output_dir, "metrics.csv"),
    )
    decision_csv = os.path.join(args.output_dir, f"{run_name}_decisions.csv")
    with open(decision_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "episode",
                "step",
                "best_candidate_rollout_score",
                "chosen_rollout_score",
                "regret",
            ],
        )
        writer.writeheader()
        writer.writerows(decision_rows)

    print("\nResults")
    print(f"Run: {run_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean score: {float(summary['mean_score']):.2f}")
    print(f"Median score: {float(summary['median_score']):.2f}")
    print(f"Mean steps: {float(summary['mean_steps']):.1f}")
    print(f"% score>0: {100.0 * float(summary['pct_score_gt0']):.1f}%")
    print(f"Runtime per decision step (ms): {float(summary['mean_decision_ms']):.2f}")
    print(f"Invalid rate: {100.0 * float(summary['invalid_rate']):.2f}%")
    print(f"Masked fraction: {float(summary.get('mean_masked_fraction', 0.0)):.3f}")
    print(f"Mean regret: {float(summary.get('mean_regret', 0.0)):.4f}")
    print(f"P90 regret: {float(summary.get('p90_regret', 0.0)):.4f}")
    print(f"Episodes CSV: {summary['episodes_csv']}")
    print(f"Decision CSV: {decision_csv}")
    if video_files:
        print(f"Videos: {len(video_files)} saved under {videos_dir}")
    print(f"Summary CSV: {os.path.join(args.output_dir, 'summary.csv')}")
    write_manifest(
        output_dir=args.output_dir,
        command="python diffusion/run_diffusion_mpc_updated.py ...",
        resolved_config=vars(args).copy(),
        git_commit=artifacts.get("git_commit", ""),
        config_hash=artifacts.get("config_hash", ""),
        model_hash="",
        dataset_hash="",
        extra_fields={
            "video_files": video_files_s,
            "selected_episode": selected_ep_s,
            "final_score": selected_score_s,
        },
    )
    return summary


def main():
    parser = build_parser()
    args, _ = parse_with_config(parser)
    run_eval(args)


if __name__ == "__main__":
    main()
