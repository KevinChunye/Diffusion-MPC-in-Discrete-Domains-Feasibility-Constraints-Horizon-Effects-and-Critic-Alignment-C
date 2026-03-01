"""filter_expert_dataset.py

Filter teacher transitions into a higher-quality expert subset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (THIS_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.config_utils import parse_with_config


def _episode_returns(episode_id: np.ndarray, reward: np.ndarray) -> Dict[int, float]:
    returns: Dict[int, float] = {}
    for ep in np.unique(episode_id):
        m = episode_id == ep
        returns[int(ep)] = float(np.sum(reward[m]))
    return returns


def _final_episode_feature(
    episode_id: np.ndarray,
    t: np.ndarray,
    values: np.ndarray,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for ep in np.unique(episode_id):
        idx = np.where(episode_id == ep)[0]
        if idx.size == 0:
            continue
        final_i = idx[np.argmax(t[idx])]
        out[int(ep)] = float(values[final_i])
    return out


def _apply_diversity_selection(
    eps_sorted: list[int],
    ep_returns: Dict[int, float],
    final_max_height: Dict[int, float],
    final_holes: Dict[int, float],
    buckets: int,
    per_bucket_cap: int,
) -> list[int]:
    if not eps_sorted:
        return eps_sorted
    b = max(1, int(buckets))
    cap = max(1, int(per_bucket_cap))

    hs = np.array([final_max_height.get(ep, 0.0) for ep in eps_sorted], dtype=np.float32)
    holes = np.array([final_holes.get(ep, 0.0) for ep in eps_sorted], dtype=np.float32)
    h_edges = np.quantile(hs, np.linspace(0.0, 1.0, b + 1))
    hole_edges = np.quantile(holes, np.linspace(0.0, 1.0, b + 1))

    def _bin(v: float, edges: np.ndarray) -> int:
        if edges.size <= 2:
            return 0
        return int(np.searchsorted(edges[1:-1], v, side="right"))

    bucket_counts: Dict[tuple[int, int], int] = {}
    kept: list[int] = []
    for ep in sorted(eps_sorted, key=lambda e: ep_returns[e], reverse=True):
        key = (_bin(final_max_height.get(ep, 0.0), h_edges), _bin(final_holes.get(ep, 0.0), hole_edges))
        if bucket_counts.get(key, 0) >= cap:
            continue
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
        kept.append(ep)
    return kept


def filter_expert_dataset(
    in_path: str,
    out_path: str,
    top_episode_pct: float,
    advantage_quantile: float,
    board_health: bool,
    min_steps_per_episode: int,
    diversity_enable: bool = False,
    diversity_buckets: int = 4,
    diversity_per_bucket_cap: int = 10,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    raw = np.load(in_path)

    episode_id = raw["episode_id"].astype(np.int64)
    reward = raw["reward"].astype(np.float32)
    advantage = raw["advantage"].astype(np.float32) if "advantage" in raw else np.full_like(reward, np.nan)

    uniq_eps = np.unique(episode_id)
    raw_num_episodes = int(uniq_eps.size)
    raw_num_transitions = int(episode_id.shape[0])

    ep_lengths = {int(ep): int(np.sum(episode_id == ep)) for ep in uniq_eps}
    eligible_eps = [int(ep) for ep in uniq_eps if ep_lengths[int(ep)] >= min_steps_per_episode]
    if not eligible_eps:
        raise RuntimeError("No episodes satisfy min_steps_per_episode.")

    ep_returns = _episode_returns(episode_id, reward)
    eligible_eps = sorted(eligible_eps, key=lambda ep: ep_returns[ep], reverse=True)

    keep_n = max(1, int(np.ceil(float(top_episode_pct) * len(eligible_eps))))
    top_eps = eligible_eps[:keep_n]
    if diversity_enable:
        if not {"max_height_after", "holes_after", "t"}.issubset(set(raw.files)):
            raise KeyError("diversity enabled but required fields (max_height_after, holes_after, t) are missing")
        final_max_h = _final_episode_feature(episode_id, raw["t"].astype(np.int64), raw["max_height_after"].astype(np.float32))
        final_holes = _final_episode_feature(episode_id, raw["t"].astype(np.int64), raw["holes_after"].astype(np.float32))
        top_eps = _apply_diversity_selection(
            top_eps,
            ep_returns=ep_returns,
            final_max_height=final_max_h,
            final_holes=final_holes,
            buckets=diversity_buckets,
            per_bucket_cap=diversity_per_bucket_cap,
        )
    keep_eps = set(top_eps)
    mask = np.isin(episode_id, list(keep_eps))

    adv_vals = advantage[mask]
    finite_adv = np.isfinite(adv_vals)
    if finite_adv.any():
        threshold = float(np.quantile(adv_vals[finite_adv], float(advantage_quantile)))
        keep_adv_mask = (~np.isfinite(advantage)) | (advantage >= threshold)
    else:
        threshold = float("nan")
        keep_adv_mask = np.ones_like(mask, dtype=bool)
    mask = mask & keep_adv_mask

    if board_health:
        if "holes_before" not in raw or "holes_after" not in raw:
            raise KeyError("board_health filter requested but holes_before/holes_after missing.")
        mask = mask & (raw["holes_after"] <= raw["holes_before"])

    kept_indices = np.where(mask)[0]
    if kept_indices.size == 0:
        raise RuntimeError("Filtering removed all transitions. Relax thresholds.")

    out = {k: raw[k][kept_indices] if raw[k].shape[:1] == (raw_num_transitions,) else raw[k] for k in raw.files}

    out["meta_raw_num_episodes"] = np.array([raw_num_episodes], dtype=np.int64)
    out["meta_raw_num_transitions"] = np.array([raw_num_transitions], dtype=np.int64)
    out["meta_kept_num_episodes"] = np.array([len(keep_eps)], dtype=np.int64)
    out["meta_kept_num_transitions"] = np.array([kept_indices.size], dtype=np.int64)
    out["meta_top_episode_pct"] = np.array([top_episode_pct], dtype=np.float32)
    out["meta_advantage_quantile"] = np.array([advantage_quantile], dtype=np.float32)
    out["meta_advantage_threshold"] = np.array([threshold], dtype=np.float32)
    out["meta_board_health"] = np.array([bool(board_health)])
    out["meta_min_steps_per_episode"] = np.array([min_steps_per_episode], dtype=np.int64)
    out["meta_diversity_enable"] = np.array([bool(diversity_enable)])
    out["meta_diversity_buckets"] = np.array([int(diversity_buckets)], dtype=np.int64)
    out["meta_diversity_per_bucket_cap"] = np.array([int(diversity_per_bucket_cap)], dtype=np.int64)

    np.savez_compressed(out_path, **out)
    print(f"Saved filtered transitions: {out_path}")
    print(f"#episodes raw={raw_num_episodes}, kept={len(keep_eps)}")
    print(f"#transitions raw={raw_num_transitions}, kept={kept_indices.size}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--top_episode_pct", type=float, default=0.10)
    parser.add_argument("--advantage_quantile", type=float, default=0.80)
    parser.add_argument("--board_health", action="store_true")
    parser.add_argument("--min_steps_per_episode", type=int, default=1)
    parser.add_argument("--diversity_enable", action="store_true")
    parser.add_argument("--diversity_buckets", type=int, default=4)
    parser.add_argument("--diversity_per_bucket_cap", type=int, default=10)
    args, _ = parse_with_config(parser)

    filter_expert_dataset(
        in_path=args.in_path,
        out_path=args.out_path,
        top_episode_pct=args.top_episode_pct,
        advantage_quantile=args.advantage_quantile,
        board_health=args.board_health,
        min_steps_per_episode=args.min_steps_per_episode,
        diversity_enable=args.diversity_enable,
        diversity_buckets=args.diversity_buckets,
        diversity_per_bucket_cap=args.diversity_per_bucket_cap,
    )


if __name__ == "__main__":
    main()
