from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from diffusion.build_diffusion_sequences import build_diffusion_sequences
from diffusion.filter_expert_dataset import filter_expert_dataset
from diffusion.generate_teacher_rollouts import generate_teacher_rollouts
from experiments.config_utils import parse_with_config
from experiments.hashing import sha256_file, write_json
from experiments.repro import prepare_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end expert dataset pipeline.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="datasets/expert_v1")
    parser.add_argument("--output_dir", type=str, default="runs/dataset_gen")
    parser.add_argument("--teacher", type=str, default="dqn", choices=["dqn", "heuristic"])
    parser.add_argument("--dqn_ckpt", type=str, default="dqn_updated.pt")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--top_episode_pct", type=float, default=0.10)
    parser.add_argument("--advantage_quantile", type=float, default=0.80)
    parser.add_argument("--board_health", action="store_true")
    parser.add_argument("--min_steps_per_episode", type=int, default=1)

    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    return parser


def run_pipeline(args: argparse.Namespace) -> dict:
    artifacts = prepare_run_artifacts(args.output_dir, vars(args).copy())

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "raw_transitions.npz")
    filtered_path = os.path.join(args.out_dir, "filtered_transitions.npz")
    seq_path = os.path.join(args.out_dir, f"sequences_H{args.horizon}.npz")

    print("[1/3] Generating teacher rollouts")
    generate_teacher_rollouts(
        episodes=args.episodes,
        seed=args.seed,
        out_path=raw_path,
        teacher=args.teacher,
        dqn_ckpt=args.dqn_ckpt,
        max_steps_per_episode=args.max_steps_per_episode,
        width=args.width,
        height=args.height,
        device=args.device,
    )

    print("[2/3] Filtering transitions")
    filter_expert_dataset(
        in_path=raw_path,
        out_path=filtered_path,
        top_episode_pct=args.top_episode_pct,
        advantage_quantile=args.advantage_quantile,
        board_health=args.board_health,
        min_steps_per_episode=args.min_steps_per_episode,
    )

    print("[3/3] Building diffusion sequences")
    build_diffusion_sequences(
        in_path=filtered_path,
        out_path=seq_path,
        horizon=args.horizon,
        stride=args.stride,
    )

    raw = np.load(raw_path)
    filt = np.load(filtered_path)
    seq = np.load(seq_path)
    dataset_hash = sha256_file(seq_path)
    dataset_meta = {
        "dataset_path": seq_path,
        "dataset_hash": dataset_hash,
        "config_hash_used": artifacts.get("config_hash", ""),
        "raw_path": raw_path,
        "filtered_path": filtered_path,
    }
    write_json(os.path.join(args.out_dir, "meta.json"), dataset_meta)
    write_json(os.path.join(args.output_dir, "meta.json"), dataset_meta)
    summary = {
        "run_name": "dataset_pipeline",
        "episodes_requested": int(args.episodes),
        "episodes_raw": int(np.unique(raw["episode_id"]).size),
        "episodes_kept": int(filt["meta_kept_num_episodes"][0]) if "meta_kept_num_episodes" in filt else -1,
        "transitions_raw": int(raw["episode_id"].shape[0]),
        "transitions_kept": int(filt["episode_id"].shape[0]),
        "sequences_built": int(seq["board"].shape[0]),
        "horizon": int(args.horizon),
        "stride": int(args.stride),
        "raw_path": raw_path,
        "filtered_path": filtered_path,
        "sequence_path": seq_path,
        "dataset_hash": dataset_hash,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\nDataset pipeline complete.")
    print(f"Raw transitions: {raw_path}")
    print(f"Filtered transitions: {filtered_path}")
    print(f"Sequence dataset: {seq_path}")
    print(f"metrics.csv: {metrics_path}")
    print(f"summary.csv: {summary_path}")
    return summary


def main() -> None:
    parser = build_parser()
    args, _ = parse_with_config(parser)
    run_pipeline(args)


if __name__ == "__main__":
    main()
