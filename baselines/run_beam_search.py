from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TetrisGym_updated import TetrisGym
from baselines.beam_search_planner import BeamCfg, BeamSearchPlanner
from experiments.config_utils import parse_with_config
from experiments.manifest import write_manifest
from experiments.metrics import MetricsLogger
from experiments.repro import prepare_run_artifacts
from experiments.video_utils import save_video, select_episode_indices


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


def _run_episode(env, planner, max_steps: int, record_frames: bool, video_max_steps: int):
    obs = env.reset()
    done = False
    steps = 0
    ep_decision_ms = []
    invalid_count = 0
    decision_count = 0
    frames = []
    if record_frames:
        _record_frame(env, frames, info=None, max_frames=video_max_steps)

    while not done and steps < int(max_steps):
        valid = env.get_valid_action_ids()
        if not valid:
            break
        t0 = time.perf_counter()
        aid, _ = planner.plan(env)
        t1 = time.perf_counter()
        decision_ms = 1000.0 * (t1 - t0)
        ep_decision_ms.append(decision_ms)
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
        "frames": frames,
    }


def run_eval(args: argparse.Namespace) -> dict:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    resolved = vars(args).copy()
    artifacts = prepare_run_artifacts(args.output_dir, resolved)

    env = TetrisGym(
        width=int(args.width),
        height=int(args.height),
        max_steps=int(args.max_steps),
        render_mode="skip",
        seed=int(args.seed),
    )
    planner = BeamSearchPlanner(BeamCfg(horizon=int(args.horizon), beam_width=int(args.beam_width)))
    logger = MetricsLogger(run_name=str(args.run_name))
    eval_start = time.perf_counter()
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

    for ep in tqdm(range(int(args.episodes)), desc="BeamSearch"):
        rec_now = bool(record_video and args.video_select == "first" and ep in first_targets)
        ep_result = _run_episode(
            env=env,
            planner=planner,
            max_steps=int(args.max_steps),
            record_frames=rec_now,
            video_max_steps=int(args.video_max_steps),
        )
        ep_scores.append(float(ep_result["score"]))

        logger.add_episode(
            episode=ep + 1,
            score=float(ep_result["score"]),
            steps=int(ep_result["steps"]),
            decision_ms=ep_result["decision_ms"],
            invalid_count=int(ep_result["invalid_count"]),
            decision_count=int(ep_result["decision_count"]),
        )
        if rec_now and ep_result["frames"]:
            out_file = os.path.join(videos_dir, f"{args.run_name}_ep{ep+1:04d}.{args.video_format}")
            saved = save_video(ep_result["frames"], out_file, fmt=args.video_format, fps=5)
            video_files.append(saved)
            selected_episode_1based.append(ep + 1)
            selected_scores.append(float(ep_result["score"]))

    if record_video and args.video_select != "first" and video_count > 0:
        replay_targets = select_episode_indices(ep_scores, video_count, args.video_select)
        replay_set = set(replay_targets)
        if replay_set:
            np.random.seed(int(args.seed))
            random.seed(int(args.seed))
            torch.manual_seed(int(args.seed))
            env_replay = TetrisGym(
                width=int(args.width),
                height=int(args.height),
                max_steps=int(args.max_steps),
                render_mode="skip",
                seed=int(args.seed),
            )
            replay_planner = BeamSearchPlanner(BeamCfg(horizon=int(args.horizon), beam_width=int(args.beam_width)))
            max_target = max(replay_set)
            for ep in range(max_target + 1):
                rec_now = ep in replay_set
                ep_result = _run_episode(
                    env=env_replay,
                    planner=replay_planner,
                    max_steps=int(args.max_steps),
                    record_frames=rec_now,
                    video_max_steps=int(args.video_max_steps),
                )
                if rec_now and ep_result["frames"]:
                    out_file = os.path.join(videos_dir, f"{args.run_name}_{args.video_select}_ep{ep+1:04d}.{args.video_format}")
                    saved = save_video(ep_result["frames"], out_file, fmt=args.video_format, fps=5)
                    video_files.append(saved)
                    selected_episode_1based.append(ep + 1)
                    selected_scores.append(float(ep_result["score"]))

    video_files_s = ";".join(video_files)
    selected_ep_s = ";".join(str(x) for x in selected_episode_1based)
    selected_score_s = ";".join(f"{x:.3f}" for x in selected_scores)

    summary = logger.write(
        args.output_dir,
        summary_extra={
            "planner": "beam_search",
            "beam_width": int(args.beam_width),
            "horizon": int(args.horizon),
            "episodes": int(args.episodes),
            "total_eval_walltime_sec": float(time.perf_counter() - eval_start),
            "seed": int(args.seed),
            "video_files": video_files_s,
            "selected_episode": selected_ep_s,
            "final_score": selected_score_s,
        },
        bootstrap_ci=bool(int(args.bootstrap_ci)),
        bootstrap_samples=int(args.bootstrap_samples),
    )
    print("\nResults")
    print(f"Run: {args.run_name}")
    print(f"Mean score: {float(summary['mean_score']):.2f}")
    print(f"Median score: {float(summary['median_score']):.2f}")
    print(f"Mean steps: {float(summary['mean_steps']):.1f}")
    print(f"% score>0: {100.0 * float(summary['pct_score_gt0']):.1f}%")
    print(f"Runtime per decision step (ms): {float(summary['mean_decision_ms']):.2f}")
    print(f"Invalid rate: {100.0 * float(summary['invalid_rate']):.2f}%")
    if video_files:
        print(f"Videos: {len(video_files)} saved under {videos_dir}")
    print(f"metrics.csv: {os.path.join(args.output_dir, 'metrics.csv')}")
    print(f"summary.csv: {os.path.join(args.output_dir, 'summary.csv')}")
    write_manifest(
        output_dir=args.output_dir,
        command="python baselines/run_beam_search.py ...",
        resolved_config=vars(args).copy(),
        git_commit=artifacts.get("git_commit", ""),
        config_hash=artifacts.get("config_hash", ""),
        dataset_hash="",
        model_hash="",
        extra_fields={
            "video_files": video_files_s,
            "selected_episode": selected_ep_s,
            "final_score": selected_score_s,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate heuristic beam-search baseline.")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--beam_width", type=int, default=16)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default="beam_search_eval")
    p.add_argument("--output_dir", type=str, default="runs/beam_search_eval")
    p.add_argument("--bootstrap_ci", type=int, default=0)
    p.add_argument("--bootstrap_samples", type=int, default=1000)
    p.add_argument("--record_video", type=str, nargs="?", const="true", default="false")
    p.add_argument("--video_episodes", type=int, default=1)
    p.add_argument("--video_max_steps", type=int, default=200)
    p.add_argument("--video_format", type=str, default="gif", choices=["gif", "mp4"])
    p.add_argument("--video_select", type=str, default="first", choices=["first", "best", "median", "worst"])
    return p


def main() -> None:
    parser = build_parser()
    args, _ = parse_with_config(parser)
    run_eval(args)


if __name__ == "__main__":
    main()
