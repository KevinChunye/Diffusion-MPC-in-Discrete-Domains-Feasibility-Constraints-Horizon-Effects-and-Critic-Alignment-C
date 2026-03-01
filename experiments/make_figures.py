from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def _selftrain_plots(progress_files: List[str], output_dir: str) -> None:
    all_points = []
    for pth in progress_files:
        run_name = Path(pth).parent.name
        rows = _read_csv(pth)
        for r in rows:
            all_points.append(
                {
                    "run": run_name,
                    "iter": int(float(r["iter"])),
                    "score": _to_float(r.get("diffusion_mean_score", 0.0), 0.0),
                    "invalid": _to_float(r.get("diffusion_invalid_rate", 0.0), 0.0),
                    "runtime": _to_float(r.get("diffusion_runtime_ms", 0.0), 0.0),
                }
            )
    if not all_points:
        return

    plt.figure(figsize=(6, 4))
    for run in sorted({p["run"] for p in all_points}):
        pts = sorted([p for p in all_points if p["run"] == run], key=lambda x: x["iter"])
        plt.plot([p["iter"] for p in pts], [p["score"] for p in pts], marker="o", label=run)
    plt.xlabel("Iteration")
    plt.ylabel("Diffusion Mean Score")
    plt.title("Score vs Iteration")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_vs_iteration.png"))
    plt.savefig(os.path.join(output_dir, "score_vs_iteration.pdf"))
    plt.close()

    plt.figure(figsize=(6, 4))
    for run in sorted({p["run"] for p in all_points}):
        pts = sorted([p for p in all_points if p["run"] == run], key=lambda x: x["iter"])
        plt.plot([p["iter"] for p in pts], [p["invalid"] for p in pts], marker="o", label=run)
    plt.xlabel("Iteration")
    plt.ylabel("Diffusion Invalid Rate")
    plt.title("Invalid Rate vs Iteration")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "invalid_rate_vs_iteration.png"))
    plt.savefig(os.path.join(output_dir, "invalid_rate_vs_iteration.pdf"))
    plt.close()

    plt.figure(figsize=(6, 4))
    runs = sorted({p["run"] for p in all_points})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(runs))))
    for c, run in zip(colors, runs):
        pts = [p for p in all_points if p["run"] == run]
        plt.scatter([p["runtime"] for p in pts], [p["score"] for p in pts], label=run, color=c)
    plt.xlabel("Runtime per decision (ms)")
    plt.ylabel("Diffusion Mean Score")
    plt.title("Score vs Runtime")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_vs_runtime.png"))
    plt.savefig(os.path.join(output_dir, "score_vs_runtime.pdf"))
    plt.close()


def _best_rows_per_method(index_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    candidates = []
    for r in index_rows:
        ms = _to_float(r.get("mean_score", ""), np.nan)
        inv = _to_float(r.get("invalid_rate", ""), np.nan)
        rt = _to_float(r.get("mean_runtime_ms", ""), np.nan)
        if np.isnan(ms):
            continue
        method = (r.get("method") or "").strip()
        if not method:
            continue
        out = r.copy()
        out["_mean_score"] = ms
        out["_invalid_rate"] = inv if not np.isnan(inv) else 1e9
        out["_runtime"] = rt if not np.isnan(rt) else 1e9
        candidates.append(out)

    best: Dict[tuple[str, str], Dict[str, str]] = {}
    for r in candidates:
        key = (r.get("method", ""), r.get("variant", ""))
        if key not in best:
            best[key] = r
            continue
        a = best[key]
        if (r["_mean_score"], -r["_invalid_rate"], -r["_runtime"]) > (a["_mean_score"], -a["_invalid_rate"], -a["_runtime"]):
            best[key] = r
    return list(best.values())


def _write_tables(best_rows: List[Dict[str, str]], out_tables_dir: str) -> None:
    os.makedirs(out_tables_dir, exist_ok=True)
    headers = [
        "method",
        "variant",
        "run_id",
        "seed",
        "episodes",
        "mean_score",
        "median_score",
        "p10_score",
        "p90_score",
        "std_score",
        "cvar10_score",
        "invalid_rate",
        "mean_runtime_ms",
        "sampling_constraints",
        "rerank_mode",
        "output_dir",
    ]
    rows = [{h: r.get(h, "") for h in headers} for r in best_rows]

    csv_path = os.path.join(out_tables_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(out_tables_dir, "results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")

    tex_path = os.path.join(out_tables_dir, "results.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{" + "l" * len(headers) + "}\n")
        f.write(" \\hline\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write(" \\hline\n")
        for r in rows:
            vals = [str(r[h]).replace("_", "\\_") for h in headers]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write(" \\hline\n")
        f.write("\\end{tabular}\n")


def _index_plots(index_rows: List[Dict[str, str]], output_dir: str) -> None:
    eval_rows = []
    for r in index_rows:
        ms = _to_float(r.get("mean_score", ""), np.nan)
        rt = _to_float(r.get("mean_runtime_ms", ""), np.nan)
        inv = _to_float(r.get("invalid_rate", ""), np.nan)
        if np.isnan(ms) or np.isnan(rt):
            continue
        eval_rows.append(
            {
                "method": r.get("method", ""),
                "variant": r.get("variant", ""),
                "mean_score": ms,
                "runtime": rt,
                "invalid_rate": inv if not np.isnan(inv) else 0.0,
                "seed": r.get("seed", ""),
                "episodes": r.get("episodes", ""),
                "output_dir": r.get("output_dir", ""),
            }
        )
    if not eval_rows:
        return

    # score vs runtime scatter
    plt.figure(figsize=(7, 5))
    methods = sorted({r["method"] for r in eval_rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(methods))))
    for c, m in zip(colors, methods):
        pts = [r for r in eval_rows if r["method"] == m]
        plt.scatter([r["runtime"] for r in pts], [r["mean_score"] for r in pts], label=m, color=c)
    seeds = sorted({str(r["seed"]) for r in eval_rows if str(r["seed"])})
    n_eps = sorted({str(r["episodes"]) for r in eval_rows if str(r["episodes"])})
    plt.xlabel("Mean Runtime per Decision (ms)")
    plt.ylabel("Mean Score")
    plt.title(f"Score vs Runtime (episodes={','.join(n_eps)}; seeds={','.join(seeds)})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "index_score_vs_runtime.png"))
    plt.savefig(os.path.join(output_dir, "index_score_vs_runtime.pdf"))
    plt.close()

    # invalid_rate vs method bar
    method_inv = []
    for m in methods:
        vals = [r["invalid_rate"] for r in eval_rows if r["method"] == m]
        method_inv.append((m, float(np.mean(vals)) if vals else 0.0))
    plt.figure(figsize=(7, 4))
    plt.bar([m for m, _ in method_inv], [v for _, v in method_inv])
    plt.ylabel("Mean Invalid Rate")
    plt.title("Invalid Rate by Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "index_invalid_rate_by_method.png"))
    plt.savefig(os.path.join(output_dir, "index_invalid_rate_by_method.pdf"))
    plt.close()

    # score distribution boxplot per method (from per-episode CSV when available)
    score_groups: Dict[str, List[float]] = {m: [] for m in methods}
    for r in eval_rows:
        out_dir = r["output_dir"]
        metrics_path = os.path.join(out_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue
        mrows = _read_csv(metrics_path)
        for mr in mrows:
            s = _to_float(mr.get("score", ""), np.nan)
            if not np.isnan(s):
                score_groups[r["method"]].append(s)
    labels = [m for m in methods if score_groups[m]]
    if labels:
        plt.figure(figsize=(7, 4))
        plt.boxplot([score_groups[m] for m in labels], tick_labels=labels)
        total_n = sum(len(score_groups[m]) for m in labels)
        plt.ylabel("Episode Score")
        plt.title(f"Score Distribution by Method (N={total_n})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "index_score_boxplot_by_method.png"))
        plt.savefig(os.path.join(output_dir, "index_score_boxplot_by_method.pdf"))
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Generate publication-oriented figures/tables from historical runs.")
    p.add_argument("--runs_glob", type=str, default="runs/selftrain/*")
    p.add_argument("--index_path", type=str, default="runs/index.csv")
    p.add_argument("--output_dir", type=str, default="runs/figures")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Keep existing selftrain progress plots.
    progress_files = sorted(glob.glob(os.path.join(args.runs_glob, "progress.csv")))
    if progress_files:
        _selftrain_plots(progress_files, args.output_dir)

    # New historical index-driven tables/plots.
    if os.path.exists(args.index_path):
        rows = _read_csv(args.index_path)
        if rows:
            best = _best_rows_per_method(rows)
            _write_tables(best, os.path.join(args.output_dir, "tables"))
            _index_plots(rows, args.output_dir)
    print(f"Figures/tables written under: {args.output_dir}")


if __name__ == "__main__":
    main()
