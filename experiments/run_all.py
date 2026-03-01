from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _run_one(
    script_path: Path,
    ckpt: str,
    critic_ckpt: str,
    output_dir: str,
    run_name: str,
    episodes: int,
    max_steps: int,
    width: int,
    height: int,
    horizon: int,
    num_candidates: int,
    sample_steps: int,
    temperature: float,
    rerank_mode: str,
    invalid_handling: str,
    invalid_penalty: float,
    resample_retries: int,
    seed: int,
    device: str,
) -> None:
    cmd = [
        sys.executable,
        str(script_path),
        "--ckpt", ckpt,
        "--critic_ckpt", critic_ckpt,
        "--episodes", str(episodes),
        "--max_steps", str(max_steps),
        "--width", str(width),
        "--height", str(height),
        "--horizon", str(horizon),
        "--num_candidates", str(num_candidates),
        "--sample_steps", str(sample_steps),
        "--temperature", str(temperature),
        "--rerank_mode", rerank_mode,
        "--invalid_handling", invalid_handling,
        "--invalid_penalty", str(invalid_penalty),
        "--resample_retries", str(resample_retries),
        "--seed", str(seed),
        "--device", device,
        "--output_dir", output_dir,
        "--run_name", run_name,
    ]
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _plot_summary(rows: list[dict[str, str]], out_dir: str) -> None:
    run_names = [r["run_name"] for r in rows]
    mean_scores = [float(r["mean_score"]) for r in rows]
    mean_ms = [float(r["mean_decision_ms"]) for r in rows]

    plt.figure(figsize=(max(8, len(run_names) * 0.8), 4))
    plt.bar(run_names, mean_scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Score")
    plt.title("Mean Score by Run")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_score_by_run.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(mean_ms, mean_scores)
    for x, y, name in zip(mean_ms, mean_scores, run_names):
        plt.annotate(name, (x, y), fontsize=8)
    plt.xlabel("Runtime per decision step (ms)")
    plt.ylabel("Mean Score")
    plt.title("Latency vs Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_vs_score.png"))
    plt.close()


def _write_markdown_report(rows: list[dict[str, str]], out_dir: str) -> None:
    rows_sorted = sorted(rows, key=lambda r: float(r["mean_score"]), reverse=True)

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Diffusion-MPC Experiment Report\n\n")
        f.write("## Setup\n")
        f.write("- Experiment 1: DQN value reranking (`rerank_mode=dqn`, `invalid_handling=none`)\n")
        f.write("- Experiment 2: DQN value reranking + invalid-action penalty (`invalid_handling=penalize`)\n")
        f.write("- Ablations: candidates `K`, horizon `H`, sampling steps `S`, temperature `T`\n\n")

        f.write("## Results Table\n\n")
        f.write("| run_name | rerank_mode | invalid_handling | K | H | S | T | mean_score | median_score | mean_steps | %score>0 | decision_ms |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_sorted:
            f.write(
                f"| {r['run_name']} | {r['rerank_mode']} | {r['invalid_handling']} | "
                f"{r['num_candidates']} | {r['horizon']} | {r['sample_steps']} | {float(r['temperature']):.3g} | "
                f"{float(r['mean_score']):.3f} | {float(r['median_score']):.3f} | {float(r['mean_steps']):.3f} | "
                f"{100.0 * float(r['pct_score_gt0']):.1f}% | {float(r['mean_decision_ms']):.2f} |\n"
            )

        if rows_sorted:
            best = rows_sorted[0]
            f.write("\n## Best Run\n\n")
            f.write(
                f"- `{best['run_name']}` with mean score `{float(best['mean_score']):.3f}` and "
                f"runtime `{float(best['mean_decision_ms']):.2f} ms` per decision step.\n"
            )

        f.write("\n## Plots\n\n")
        f.write("![Mean score by run](mean_score_by_run.png)\n\n")
        f.write("![Latency vs score](latency_vs_score.png)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp1/Exp2 with ablations and write aggregate report.")
    parser.add_argument("--ckpt", type=str, required=True, help="PlanDenoiser checkpoint")
    parser.add_argument("--critic_ckpt", type=str, default="dqn_updated.pt", help="DQN checkpoint for reranking")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--k_values", type=str, default="32,64")
    parser.add_argument("--h_values", type=str, default="5")
    parser.add_argument("--s_values", type=str, default="8")
    parser.add_argument("--temp_values", type=str, default="0.8,1.0")
    parser.add_argument("--invalid_penalty", type=float, default=1e6)
    parser.add_argument("--resample_retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_root", type=str, default="results/experiments")
    args = parser.parse_args()

    k_values = _parse_int_list(args.k_values)
    h_values = _parse_int_list(args.h_values)
    s_values = _parse_int_list(args.s_values)
    temp_values = _parse_float_list(args.temp_values)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    eval_script = repo_root / "diffusion" / "run_diffusion_mpc_updated.py"

    combos = list(itertools.product(k_values, h_values, s_values, temp_values))
    for idx, (k, h, s, t) in enumerate(combos, start=1):
        exp1_name = f"exp1_k{k}_h{h}_s{s}_t{t:g}"
        exp2_name = f"exp2_k{k}_h{h}_s{s}_t{t:g}"

        print(f"\n=== Ablation {idx}/{len(combos)}: K={k}, H={h}, S={s}, T={t:g} ===")
        _run_one(
            script_path=eval_script,
            ckpt=args.ckpt,
            critic_ckpt=args.critic_ckpt,
            output_dir=out_dir,
            run_name=exp1_name,
            episodes=args.episodes,
            max_steps=args.max_steps,
            width=args.width,
            height=args.height,
            horizon=h,
            num_candidates=k,
            sample_steps=s,
            temperature=t,
            rerank_mode="dqn",
            invalid_handling="none",
            invalid_penalty=args.invalid_penalty,
            resample_retries=args.resample_retries,
            seed=args.seed,
            device=args.device,
        )
        _run_one(
            script_path=eval_script,
            ckpt=args.ckpt,
            critic_ckpt=args.critic_ckpt,
            output_dir=out_dir,
            run_name=exp2_name,
            episodes=args.episodes,
            max_steps=args.max_steps,
            width=args.width,
            height=args.height,
            horizon=h,
            num_candidates=k,
            sample_steps=s,
            temperature=t,
            rerank_mode="dqn",
            invalid_handling="penalize",
            invalid_penalty=args.invalid_penalty,
            resample_retries=args.resample_retries,
            seed=args.seed,
            device=args.device,
        )

    summary_csv = os.path.join(out_dir, "summary.csv")
    with open(summary_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    _plot_summary(rows, out_dir)
    _write_markdown_report(rows, out_dir)

    print("\nCompleted all runs.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Report: {os.path.join(out_dir, 'report.md')}")


if __name__ == "__main__":
    main()

