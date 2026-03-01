from __future__ import annotations

import os
from typing import Iterable, List

import imageio.v2 as imageio
import numpy as np


def select_episode_indices(scores: Iterable[float], count: int, mode: str) -> List[int]:
    vals = np.array(list(scores), dtype=np.float32)
    n = int(vals.shape[0])
    if n == 0 or count <= 0:
        return []
    k = min(int(count), n)
    mode = str(mode).strip().lower()

    if mode == "first":
        return list(range(k))

    order = np.argsort(vals)
    if mode == "best":
        chosen = order[::-1][:k]
    elif mode == "worst":
        chosen = order[:k]
    elif mode == "median":
        med = float(np.median(vals))
        by_dist = np.argsort(np.abs(vals - med))
        chosen = by_dist[:k]
    else:
        raise ValueError(f"Unknown video_select={mode}")

    # Keep chronological ordering for replay simplicity.
    out = sorted(int(i) for i in chosen.tolist())
    return out


def save_video(frames: List[np.ndarray], out_path: str, fmt: str = "gif", fps: int = 5) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fmt = str(fmt).lower()
    if not frames:
        return ""
    arrs = [np.asarray(f, dtype=np.uint8) for f in frames]

    if fmt == "gif":
        imageio.mimsave(out_path, arrs, fps=max(1, int(fps)))
        return out_path

    if fmt == "mp4":
        # imageio ffmpeg backend may be unavailable on some environments; fall back to GIF.
        try:
            with imageio.get_writer(out_path, fps=max(1, int(fps)), format="FFMPEG") as writer:
                for a in arrs:
                    writer.append_data(a)
            return out_path
        except Exception:
            fallback = os.path.splitext(out_path)[0] + ".gif"
            imageio.mimsave(fallback, arrs, fps=max(1, int(fps)))
            return fallback

    raise ValueError(f"Unsupported video format: {fmt}")

