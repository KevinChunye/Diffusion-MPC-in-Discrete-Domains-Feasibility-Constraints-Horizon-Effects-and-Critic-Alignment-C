from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import yaml

from experiments.hashing import read_json


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _last_row(path: str) -> Dict[str, str]:
    rows = _read_csv(path)
    return rows[-1] if rows else {}


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _infer_method_variant(summary: Dict[str, str], config: Dict) -> tuple[str, str]:
    if summary.get("planner") == "beam_search":
        method = "beam_search"
        variant = f"beam_w{summary.get('beam_width', config.get('beam_width', ''))}_h{summary.get('horizon', config.get('horizon', ''))}"
        return method, variant
    method = "diffusion"
    sc = summary.get("sampling_constraints", config.get("sampling_constraints", ""))
    rr = summary.get("rerank_mode", config.get("rerank_mode", ""))
    variant = f"{sc}_{rr}".strip("_")
    return method, variant


def _historical_best(index_rows: List[Dict[str, str]], method: str, variant: str) -> Dict[str, str]:
    cand = [r for r in index_rows if r.get("method", "") == method]
    if variant:
        cand = [r for r in cand if r.get("variant", "") == variant]
    if not cand:
        return {}
    cand.sort(
        key=lambda r: (
            -_to_float(r.get("mean_score", -1e9), -1e9),
            _to_float(r.get("invalid_rate", 1e9), 1e9),
            _to_float(r.get("mean_runtime_ms", 1e9), 1e9),
        )
    )
    return cand[0]


def _recommendation(summary: Dict[str, str], hist_best: Dict[str, str]) -> tuple[str, str]:
    mean_score = _to_float(summary.get("mean_score", 0.0), 0.0)
    invalid = _to_float(summary.get("invalid_rate", 0.0), 0.0)
    mean_regret = _to_float(summary.get("mean_regret", summary.get("mean_mean_regret", 0.0)), 0.0)
    episodes = int(_to_float(summary.get("episodes", 0), 0))
    best_score = _to_float(hist_best.get("mean_score", mean_score), mean_score)

    if invalid > 0.02:
        bottleneck = "Constraint handling bottleneck"
        rec = "Enable `sampling_constraints=mask_logits` or tighten invalid handling (`penalize/resample`)."
    elif mean_regret > max(0.5, 0.15 * abs(mean_score) + 0.1):
        bottleneck = "Reranking bottleneck"
        rec = "Improve reranker quality (use/upgrade DQN critic) before increasing sampling compute."
    else:
        bottleneck = "Sampling bottleneck"
        rec = "Increase candidate quality/diversity: tune `K`, `sample_steps`, `horizon`, and temperature."

    if episodes < 20:
        rec += " Then run multi-seed confirm (>=3 seeds)."
    elif mean_score >= 0.95 * best_score:
        rec += " This is near historical best; prioritize multi-seed confirm or self-training."
    return bottleneck, rec


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze a run directory against historical index.")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--index_path", type=str, default="runs/index.csv")
    p.add_argument("--out_path", type=str, default="")
    args = p.parse_args()

    run_dir = args.run_dir
    summary = _last_row(os.path.join(run_dir, "summary.csv"))
    metrics_rows = _read_csv(os.path.join(run_dir, "metrics.csv"))
    cfg_path = os.path.join(run_dir, "config.yaml")
    config = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    ckpt_meta = {}
    for cand in [
        os.path.join(run_dir, "checkpoints", "ckpt_meta.json"),
        os.path.join(run_dir, "ckpt_meta.json"),
    ]:
        if os.path.exists(cand):
            ckpt_meta = read_json(cand)
            break

    method, variant = _infer_method_variant(summary, config)
    index_rows = _read_csv(args.index_path)
    hist_best = _historical_best(index_rows, method, variant)
    bottleneck, rec = _recommendation(summary, hist_best)

    ci_low = summary.get("ci_low", "")
    ci_high = summary.get("ci_high", "")
    ci_note = "Bootstrap CI unavailable."
    try:
        lo = float(ci_low)
        hi = float(ci_high)
        ci_note = f"Bootstrap CI for mean score: [{lo:.3f}, {hi:.3f}]"
    except Exception:
        pass

    mean_score = _to_float(summary.get("mean_score", 0.0), 0.0)
    best_score = _to_float(hist_best.get("mean_score", 0.0), 0.0)
    delta = mean_score - best_score

    out_path = args.out_path or os.path.join(run_dir, "analysis.md")
    lines = []
    lines.append("# Run Analysis")
    lines.append("")
    lines.append("## Key Metrics")
    for k in [
        "run_name",
        "episodes",
        "mean_score",
        "median_score",
        "p10_score",
        "p90_score",
        "std_score",
        "cvar10_score",
        "mean_steps",
        "invalid_rate",
        "mean_decision_ms",
        "mean_masked_fraction",
        "mean_regret",
        "p90_regret",
    ]:
        if k in summary:
            lines.append(f"- `{k}`: {summary[k]}")
    lines.append("")
    lines.append("## Historical Comparison")
    if hist_best:
        lines.append(f"- Method/variant: `{method}` / `{variant}`")
        lines.append(f"- Historical best mean_score: `{best_score:.4f}`")
        lines.append(f"- This run mean_score: `{mean_score:.4f}` (delta `{delta:+.4f}`)")
        lines.append(f"- Best run_id: `{hist_best.get('run_id','')}`")
    else:
        lines.append("- No historical runs found for same method/variant.")
    lines.append("")
    lines.append("## Confidence")
    lines.append(f"- {ci_note}")
    lines.append("")
    lines.append("## Bottleneck Diagnosis")
    lines.append(f"- {bottleneck}")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- {rec}")
    if ckpt_meta:
        lines.append("")
        lines.append("## Checkpoint Metadata")
        for k in ["model_hash", "dataset_hash_used", "param_count", "best_loss"]:
            if k in ckpt_meta:
                lines.append(f"- `{k}`: {ckpt_meta[k]}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Analysis written: {out_path}")


if __name__ == "__main__":
    main()

