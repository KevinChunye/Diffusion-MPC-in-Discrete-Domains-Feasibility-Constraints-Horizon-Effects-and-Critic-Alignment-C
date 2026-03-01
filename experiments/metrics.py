from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeRecord:
    episode: int
    score: float
    steps: int
    mean_decision_ms: float
    invalid_count: int
    decision_count: int
    invalid_rate: float
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetricsLogger:
    run_name: str
    records: List[EpisodeRecord] = field(default_factory=list)
    total_invalid: int = 0
    total_decisions: int = 0
    extra_keys: set[str] = field(default_factory=set)

    def add_episode(
        self,
        episode: int,
        score: float,
        steps: int,
        decision_ms: List[float],
        invalid_count: int = 0,
        decision_count: int = 0,
        extra: Dict[str, float] | None = None,
    ) -> None:
        mean_ms = float(np.mean(decision_ms)) if decision_ms else 0.0
        denom = max(1, int(decision_count))
        invalid_rate = float(invalid_count) / float(denom)
        extra = extra or {}
        self.extra_keys.update(extra.keys())
        self.records.append(
            EpisodeRecord(
                episode=int(episode),
                score=float(score),
                steps=int(steps),
                mean_decision_ms=mean_ms,
                invalid_count=int(invalid_count),
                decision_count=int(decision_count),
                invalid_rate=invalid_rate,
                extra={k: float(v) for k, v in extra.items()},
            )
        )
        self.total_invalid += int(invalid_count)
        self.total_decisions += int(decision_count)

    def write(
        self,
        output_dir: str,
        summary_extra: Dict[str, object],
        append_summary: bool = False,
        metrics_filename: str = "metrics.csv",
        bootstrap_ci: bool = False,
        bootstrap_samples: int = 1000,
    ) -> Dict[str, object]:
        os.makedirs(output_dir, exist_ok=True)
        metrics_csv = os.path.join(output_dir, metrics_filename)
        summary_csv = os.path.join(output_dir, "summary.csv")

        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            fields = [
                "run_name",
                "episode",
                "score",
                "steps",
                "mean_decision_ms",
                "invalid_count",
                "decision_count",
                "invalid_rate",
            ] + sorted(self.extra_keys)
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )
            writer.writeheader()
            for r in self.records:
                row = {
                    "run_name": self.run_name,
                    "episode": r.episode,
                    "score": r.score,
                    "steps": r.steps,
                    "mean_decision_ms": r.mean_decision_ms,
                    "invalid_count": r.invalid_count,
                    "decision_count": r.decision_count,
                    "invalid_rate": r.invalid_rate,
                }
                row.update(r.extra)
                writer.writerow(row)

        scores = np.array([r.score for r in self.records], dtype=np.float32)
        steps = np.array([r.steps for r in self.records], dtype=np.float32)
        mean_decision_ms = float(np.mean([r.mean_decision_ms for r in self.records])) if self.records else 0.0
        invalid_rate = float(self.total_invalid) / float(max(1, self.total_decisions))

        p10 = float(np.percentile(scores, 10)) if scores.size else 0.0
        p90 = float(np.percentile(scores, 90)) if scores.size else 0.0
        std_score = float(np.std(scores)) if scores.size else 0.0
        if scores.size:
            n_worst = max(1, int(np.ceil(0.1 * scores.size)))
            cvar10 = float(np.mean(np.sort(scores)[:n_worst]))
        else:
            cvar10 = 0.0

        mean_score = float(np.mean(scores)) if scores.size else 0.0
        total_eval_walltime_sec = float(summary_extra.get("total_eval_walltime_sec", 0.0) or 0.0)
        if total_eval_walltime_sec <= 0.0:
            total_eval_walltime_sec = (float(self.total_decisions) * mean_decision_ms) / 1000.0
        decisions_per_sec = float(self.total_decisions) / total_eval_walltime_sec if total_eval_walltime_sec > 0 else 0.0
        score_per_second = mean_score / total_eval_walltime_sec if total_eval_walltime_sec > 0 else 0.0

        ci_low: float | str = ""
        ci_high: float | str = ""
        if bootstrap_ci and scores.size:
            rng = np.random.default_rng(0)
            n = scores.size
            boot = np.empty((max(1, int(bootstrap_samples)),), dtype=np.float32)
            for i in range(boot.shape[0]):
                sample = rng.choice(scores, size=n, replace=True)
                boot[i] = float(np.mean(sample))
            ci_low = float(np.percentile(boot, 2.5))
            ci_high = float(np.percentile(boot, 97.5))

        summary: Dict[str, object] = {
            "run_name": self.run_name,
            "episodes": len(self.records),
            "mean_score": mean_score,
            "median_score": float(np.median(scores)) if scores.size else 0.0,
            "p10_score": p10,
            "p90_score": p90,
            "std_score": std_score,
            "cvar10_score": cvar10,
            "mean_steps": float(np.mean(steps)) if steps.size else 0.0,
            "pct_score_gt0": float(np.mean(scores > 0.0)) if scores.size else 0.0,
            "mean_decision_ms": mean_decision_ms,
            "invalid_rate": invalid_rate,
            "total_eval_walltime_sec": total_eval_walltime_sec,
            "decisions_per_sec": decisions_per_sec,
            "score_per_second": score_per_second,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "metrics_csv": metrics_csv,
        }
        for k in sorted(self.extra_keys):
            vals = [r.extra.get(k, 0.0) for r in self.records]
            summary[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
        summary.update(summary_extra)

        mode = "a" if append_summary else "w"
        write_header = (not append_summary) or (not os.path.exists(summary_csv))
        with open(summary_csv, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)

        return summary
