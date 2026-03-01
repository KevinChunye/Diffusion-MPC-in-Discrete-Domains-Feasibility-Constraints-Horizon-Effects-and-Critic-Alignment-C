from __future__ import annotations

import csv
import datetime as dt
import glob
import os
from pathlib import Path
from typing import Dict, List

from experiments.hashing import sha256_dict, sha256_file
import yaml


INDEX_COLUMNS = [
    "timestamp",
    "run_id",
    "stage",
    "method",
    "variant",
    "git_commit",
    "config_path",
    "config_hash",
    "output_dir",
    "seed",
    "episodes",
    "horizon",
    "K",
    "sampling_steps",
    "temp",
    "sampling_constraints",
    "rerank_mode",
    "dataset_path",
    "dataset_hash",
    "ckpt_path",
    "model_hash",
    "mean_score",
    "median_score",
    "p10_score",
    "p90_score",
    "std_score",
    "cvar10_score",
    "mean_steps",
    "invalid_rate",
    "mean_runtime_ms",
    "mean_masked_fraction",
    "notes",
]


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _last_summary_row(summary_csv: str) -> Dict[str, str]:
    rows = _read_csv_rows(summary_csv)
    return rows[-1] if rows else {}


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _safe_float(v, default=""):
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=""):
    try:
        return int(float(v))
    except Exception:
        return default


def _infer_method(stage: str, summary: Dict[str, str]) -> str:
    if summary.get("planner") == "beam_search" or "beam" in stage:
        return "beam_search"
    if "dataset" in stage:
        return "dataset_pipeline"
    if "train" in stage:
        return "diffusion_train"
    return "diffusion"


def _infer_variant(summary: Dict[str, str]) -> str:
    sc = str(summary.get("sampling_constraints", "")).strip()
    rr = str(summary.get("rerank_mode", "")).strip()
    if sc and rr:
        return f"{sc}_{rr}"
    return sc or rr or ""


def _canonical_row(row: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for c in INDEX_COLUMNS:
        out[c] = row.get(c, "")
    return out


def upsert_index_row(row: Dict[str, object], index_path: str = "runs/index.csv") -> str:
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    rows = _read_csv_rows(index_path)
    by_id = {r.get("run_id", ""): r for r in rows if r.get("run_id", "")}

    run_id = str(row.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("row must include run_id")
    by_id[run_id] = {k: str(v) for k, v in _canonical_row(row).items()}
    merged = list(by_id.values())
    merged.sort(key=lambda r: (r.get("timestamp", ""), r.get("run_id", "")))

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)
    return index_path


def register_stage(
    output_dir: str,
    stage: str,
    method: str = "",
    variant: str = "",
    notes: str = "",
    index_path: str = "runs/index.csv",
) -> str:
    summary_csv = os.path.join(output_dir, "summary.csv")
    summary = _last_summary_row(summary_csv)
    if not summary:
        return index_path

    config_path = os.path.join(output_dir, "config.yaml")
    git_commit_path = os.path.join(output_dir, "git_commit.txt")
    config_hash = sha256_file(config_path)
    git_commit = _read_text(git_commit_path)
    cfg = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

    dataset_path = summary.get("dataset_path") or summary.get("sequence_path") or cfg.get("dataset_path") or ""
    dataset_hash = summary.get("dataset_hash") or (sha256_file(dataset_path) if dataset_path else "")
    ckpt_path = summary.get("checkpoint_path") or summary.get("ckpt_path") or summary.get("ckpt") or cfg.get("ckpt") or cfg.get("save") or ""
    model_hash = summary.get("model_hash") or (sha256_file(ckpt_path) if ckpt_path else "")

    seed = summary.get("seed", cfg.get("seed", ""))
    episodes = summary.get("episodes", cfg.get("episodes", ""))
    horizon = summary.get("horizon", cfg.get("horizon", ""))
    K = summary.get("num_candidates", cfg.get("num_candidates", summary.get("beam_width", cfg.get("beam_width", ""))))
    sampling_steps = summary.get("sample_steps", cfg.get("sample_steps", ""))
    temp = summary.get("temperature", cfg.get("temperature", ""))
    sampling_constraints = summary.get("sampling_constraints", cfg.get("sampling_constraints", ""))
    rerank_mode = summary.get("rerank_mode", cfg.get("rerank_mode", ""))

    mth = method or _infer_method(stage, summary)
    var = variant or _infer_variant(summary)

    run_id_src = {
        "stage": stage,
        "method": mth,
        "variant": var,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "model_hash": model_hash,
        "seed": seed,
    }
    run_id = sha256_dict(run_id_src)[:16]

    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "stage": stage,
        "method": mth,
        "variant": var,
        "git_commit": git_commit,
        "config_path": config_path if os.path.exists(config_path) else "",
        "config_hash": config_hash,
        "output_dir": output_dir,
        "seed": _safe_int(seed),
        "episodes": _safe_int(episodes),
        "horizon": _safe_int(horizon),
        "K": _safe_int(K),
        "sampling_steps": _safe_int(sampling_steps),
        "temp": _safe_float(temp),
        "sampling_constraints": sampling_constraints,
        "rerank_mode": rerank_mode,
        "dataset_path": dataset_path,
        "dataset_hash": dataset_hash,
        "ckpt_path": ckpt_path,
        "model_hash": model_hash,
        "mean_score": _safe_float(summary.get("mean_score", "")),
        "median_score": _safe_float(summary.get("median_score", "")),
        "p10_score": _safe_float(summary.get("p10_score", "")),
        "p90_score": _safe_float(summary.get("p90_score", "")),
        "std_score": _safe_float(summary.get("std_score", "")),
        "cvar10_score": _safe_float(summary.get("cvar10_score", "")),
        "mean_steps": _safe_float(summary.get("mean_steps", "")),
        "invalid_rate": _safe_float(summary.get("invalid_rate", "")),
        "mean_runtime_ms": _safe_float(summary.get("mean_decision_ms", "")),
        "mean_masked_fraction": _safe_float(summary.get("mean_masked_fraction", "")),
        "notes": notes,
    }
    return upsert_index_row(row, index_path=index_path)


def register_directory_summaries(root_dir: str, stage_prefix: str = "selftrain", index_path: str = "runs/index.csv") -> str:
    summary_paths = sorted(glob.glob(os.path.join(root_dir, "**", "summary.csv"), recursive=True))
    for p in summary_paths:
        out_dir = str(Path(p).parent)
        rel = os.path.relpath(out_dir, root_dir)
        stage = f"{stage_prefix}:{rel.replace(os.sep, '/')}"
        method = "beam_search" if "beam_eval" in rel else ""
        register_stage(out_dir, stage=stage, method=method, index_path=index_path)
    return index_path
