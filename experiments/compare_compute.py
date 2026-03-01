from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple


def _read_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_int(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _is_eval_stage(stage: str) -> bool:
    s = (stage or "").lower()
    return ("eval" in s) or (s == "")


def _diffusion_budget(row: Dict[str, str]) -> float:
    k = max(1, _to_float(row.get("K", 0), 0.0))
    h = max(1, _to_float(row.get("horizon", 0), 0.0))
    return float(k * h)


def _beam_budget(row: Dict[str, str]) -> float:
    b = max(1, _to_int(row.get("K", 0), 1))
    h = max(1, _to_int(row.get("horizon", 0), 1))
    # Approximate full-width branching compute.
    return float(b ** h)


def _score(row: Dict[str, str]) -> float:
    # Higher is better; used for sorting candidates.
    return _to_float(row.get("mean_score", 0.0), 0.0)


def _match_rows(
    diffusion_rows: List[Dict[str, str]],
    beam_rows: List[Dict[str, str]],
    top_k: int,
    max_rel_gap: float,
) -> List[Dict[str, object]]:
    beam_pool = sorted(beam_rows, key=_score, reverse=True)
    used_beam_ids = set()
    out: List[Dict[str, object]] = []

    for d in sorted(diffusion_rows, key=_score, reverse=True):
        d_budget = _diffusion_budget(d)
        best = None
        best_gap = 1e18
        for b in beam_pool:
            bid = b.get("run_id", "")
            if bid and bid in used_beam_ids:
                continue
            b_budget = _beam_budget(b)
            gap = abs(d_budget - b_budget) / max(d_budget, b_budget, 1.0)
            if gap < best_gap:
                best_gap = gap
                best = b
        if best is None or best_gap > max_rel_gap:
            continue

        if best.get("run_id", ""):
            used_beam_ids.add(best["run_id"])

        out.append(
            {
                "diffusion_run_id": d.get("run_id", ""),
                "beam_run_id": best.get("run_id", ""),
                "diffusion_output_dir": d.get("output_dir", ""),
                "beam_output_dir": best.get("output_dir", ""),
                "diffusion_variant": d.get("variant", ""),
                "beam_variant": best.get("variant", ""),
                "diffusion_budget": d_budget,
                "beam_budget": _beam_budget(best),
                "rel_budget_gap": best_gap,
                "diffusion_mean_score": _to_float(d.get("mean_score", 0.0), 0.0),
                "beam_mean_score": _to_float(best.get("mean_score", 0.0), 0.0),
                "score_delta_diff_minus_beam": _to_float(d.get("mean_score", 0.0), 0.0)
                - _to_float(best.get("mean_score", 0.0), 0.0),
                "diffusion_runtime_ms": _to_float(d.get("mean_runtime_ms", 0.0), 0.0),
                "beam_runtime_ms": _to_float(best.get("mean_runtime_ms", 0.0), 0.0),
                "diffusion_invalid_rate": _to_float(d.get("invalid_rate", 0.0), 0.0),
                "beam_invalid_rate": _to_float(best.get("invalid_rate", 0.0), 0.0),
            }
        )
        if len(out) >= int(top_k):
            break
    return out


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "diffusion_run_id",
        "beam_run_id",
        "diffusion_output_dir",
        "beam_output_dir",
        "diffusion_variant",
        "beam_variant",
        "diffusion_budget",
        "beam_budget",
        "rel_budget_gap",
        "diffusion_mean_score",
        "beam_mean_score",
        "score_delta_diff_minus_beam",
        "diffusion_runtime_ms",
        "beam_runtime_ms",
        "diffusion_invalid_rate",
        "beam_invalid_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    lines.append("# Compute-Matched Comparison")
    lines.append("")
    if not rows:
        lines.append("No matched runs found.")
    else:
        lines.append("| Diffusion run | Beam run | Diff budget (K*H) | Beam budget (B^H) | Rel gap | Diff score | Beam score | Delta |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                "| {dr} | {br} | {db:.1f} | {bb:.1f} | {gap:.3f} | {ds:.3f} | {bs:.3f} | {delta:+.3f} |".format(
                    dr=r["diffusion_run_id"] or "-",
                    br=r["beam_run_id"] or "-",
                    db=float(r["diffusion_budget"]),
                    bb=float(r["beam_budget"]),
                    gap=float(r["rel_budget_gap"]),
                    ds=float(r["diffusion_mean_score"]),
                    bs=float(r["beam_mean_score"]),
                    delta=float(r["score_delta_diff_minus_beam"]),
                )
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute-matched diffusion vs beam run comparison.")
    p.add_argument("--index_path", type=str, default="runs/index.csv")
    p.add_argument("--diffusion_method", type=str, default="diffusion")
    p.add_argument("--diffusion_variant", type=str, default="")
    p.add_argument("--beam_method", type=str, default="beam_search")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--max_rel_gap", type=float, default=0.50)
    p.add_argument("--output_csv", type=str, default="runs/compute_match/results.csv")
    p.add_argument("--output_md", type=str, default="runs/compute_match/results.md")
    args = p.parse_args()

    rows = _read_rows(args.index_path)
    diffusion_rows = [
        r
        for r in rows
        if r.get("method", "") == args.diffusion_method and _is_eval_stage(r.get("stage", ""))
    ]
    if args.diffusion_variant:
        diffusion_rows = [r for r in diffusion_rows if r.get("variant", "") == args.diffusion_variant]

    beam_rows = [r for r in rows if r.get("method", "") == args.beam_method and _is_eval_stage(r.get("stage", ""))]
    matched = _match_rows(
        diffusion_rows=diffusion_rows,
        beam_rows=beam_rows,
        top_k=int(args.top_k),
        max_rel_gap=float(args.max_rel_gap),
    )
    _write_csv(args.output_csv, matched)
    _write_md(args.output_md, matched)

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_md}")
    if not matched:
        print("No matches found. Try increasing --max_rel_gap or removing --diffusion_variant.")
        return
    for r in matched[: min(5, len(matched))]:
        print(
            "diff={dr} beam={br} gap={gap:.3f} delta={delta:+.3f}".format(
                dr=r.get("diffusion_run_id", ""),
                br=r.get("beam_run_id", ""),
                gap=float(r.get("rel_budget_gap", 0.0)),
                delta=float(r.get("score_delta_diff_minus_beam", 0.0)),
            )
        )


if __name__ == "__main__":
    main()
