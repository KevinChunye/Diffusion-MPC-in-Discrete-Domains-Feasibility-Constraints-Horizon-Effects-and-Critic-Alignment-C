"""build_diffusion_sequences.py

Convert filtered transitions into diffusion training sequences.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (THIS_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.config_utils import parse_with_config


def _compute_rtg_per_episode(episode_id: np.ndarray, reward: np.ndarray, t: np.ndarray) -> np.ndarray:
    rtg = np.zeros_like(reward, dtype=np.float32)
    for ep in np.unique(episode_id):
        idx = np.where(episode_id == ep)[0]
        idx = idx[np.argsort(t[idx])]
        ep_rewards = reward[idx].astype(np.float32)
        ep_rtg = np.cumsum(ep_rewards[::-1])[::-1]
        rtg[idx] = ep_rtg
    return rtg


def build_diffusion_sequences(
    in_path: str,
    out_path: str,
    horizon: int,
    stride: int,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    data = np.load(in_path)
    episode_id = data["episode_id"].astype(np.int64)
    t = data["t"].astype(np.int64)
    board = data["board"].astype(np.uint8)  # (N,H,W)
    curr_id = data["curr_id"].astype(np.int64)
    next_id = data["next_id"].astype(np.int64)
    action_rot = data["action_rot"].astype(np.int64)
    action_x = data["action_x"].astype(np.int64)
    reward = data["reward"].astype(np.float32)

    rtg = _compute_rtg_per_episode(episode_id, reward, t)
    width = int(data["width"][0]) if "width" in data else int(np.max(action_x) + 1)
    height = int(data["height"][0]) if "height" in data else int(board.shape[1])

    boards: List[np.ndarray] = []
    curr_ids: List[int] = []
    next_ids: List[int] = []
    rot_seqs: List[np.ndarray] = []
    x_seqs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    rtg_start: List[float] = []
    seq_episode_id: List[int] = []
    seq_t_start: List[int] = []

    uniq_eps = np.unique(episode_id)
    for ep in uniq_eps:
        idx = np.where(episode_id == ep)[0]
        idx = idx[np.argsort(t[idx])]
        n = idx.size
        if n == 0:
            continue

        for s in range(0, n, stride):
            e = min(s + horizon, n)
            length = e - s
            if length <= 0:
                continue

            start_i = idx[s]
            chunk = idx[s:e]

            rot = np.zeros((horizon,), dtype=np.int64)
            xs = np.zeros((horizon,), dtype=np.int64)
            valid = np.zeros((horizon,), dtype=bool)
            rot[:length] = action_rot[chunk]
            xs[:length] = action_x[chunk]
            valid[:length] = True

            boards.append(board[start_i][None, ...])  # (1,H,W)
            curr_ids.append(int(curr_id[start_i]))
            next_ids.append(int(next_id[start_i]))
            rot_seqs.append(rot)
            x_seqs.append(xs)
            masks.append(valid)
            rtg_start.append(float(rtg[start_i]))
            seq_episode_id.append(int(ep))
            seq_t_start.append(int(t[start_i]))

    if not boards:
        raise RuntimeError("No sequences were built. Check filtered transitions/horizon.")

    out = {
        "board": np.stack(boards).astype(np.uint8),  # (N,1,H,W)
        "curr_id": np.array(curr_ids, dtype=np.int64),
        "next_id": np.array(next_ids, dtype=np.int64),
        "rot_seq": np.stack(rot_seqs).astype(np.int64),
        "x_seq": np.stack(x_seqs).astype(np.int64),
        "valid_mask": np.stack(masks).astype(bool),
        "rtg": np.array(rtg_start, dtype=np.float32),
        "episode_id": np.array(seq_episode_id, dtype=np.int64),
        "t_start": np.array(seq_t_start, dtype=np.int64),
        "width": np.array([width], dtype=np.int64),
        "height": np.array([height], dtype=np.int64),
        "horizon": np.array([horizon], dtype=np.int64),
        "stride": np.array([stride], dtype=np.int64),
        "meta_raw_num_episodes": data["meta_raw_num_episodes"] if "meta_raw_num_episodes" in data else np.array([len(uniq_eps)], dtype=np.int64),
        "meta_kept_num_episodes": data["meta_kept_num_episodes"] if "meta_kept_num_episodes" in data else np.array([len(uniq_eps)], dtype=np.int64),
        "meta_kept_num_transitions": data["meta_kept_num_transitions"] if "meta_kept_num_transitions" in data else np.array([episode_id.shape[0]], dtype=np.int64),
        "meta_sequences_built": np.array([len(boards)], dtype=np.int64),
    }
    np.savez_compressed(out_path, **out)

    print(f"Saved sequence dataset: {out_path}")
    print(f"#episodes raw={int(out['meta_raw_num_episodes'][0])}, #episodes kept={int(out['meta_kept_num_episodes'][0])}")
    print(f"#transitions kept={int(out['meta_kept_num_transitions'][0])}, #sequences built={len(boards)}")
    print(f"RTG stats: min={float(np.min(out['rtg'])):.3f}, mean={float(np.mean(out['rtg'])):.3f}, max={float(np.max(out['rtg'])):.3f}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    args, _ = parse_with_config(parser)

    build_diffusion_sequences(
        in_path=args.in_path,
        out_path=args.out_path,
        horizon=args.horizon,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
