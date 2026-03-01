from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from experiments.hashing import sha256_dict, sha256_text
from experiments.lightning_env import default_artifact_root


def _read_rows(path: str) -> List[Dict[str, str]]:
    import csv

    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_eval(expr: str, variables: Dict[str, float]) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Name,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Mod,
    )
    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if not isinstance(n, allowed_nodes):
            raise ValueError(f"Unsupported token in objective: {type(n).__name__}")
        if isinstance(n, ast.Name) and n.id not in variables:
            variables[n.id] = 0.0
    return float(eval(compile(tree, "<objective>", "eval"), {"__builtins__": {}}, variables))


def _normalize_cfg_hash(config: Dict[str, object]) -> str:
    txt = yaml.safe_dump(config, sort_keys=True)
    return sha256_text(txt)


def _extract_params(row: Dict[str, str]) -> Dict[str, float]:
    return {
        "horizon": _to_float(row.get("horizon", 5), 5),
        "K": _to_float(row.get("K", 64), 64),
        "sampling_steps": _to_float(row.get("sampling_steps", 8), 8),
        "temp": _to_float(row.get("temp", 1.0), 1.0),
        "seed": _to_float(row.get("seed", 0), 0),
        "invalid_rate": _to_float(row.get("invalid_rate", 0.0), 0.0),
        "mean_runtime_ms": _to_float(row.get("mean_runtime_ms", 0.0), 0.0),
        "mean_score": _to_float(row.get("mean_score", 0.0), 0.0),
        "cvar10_score": _to_float(row.get("cvar10_score", 0.0), 0.0),
    }


def _signature(c: Dict[str, object]) -> Tuple:
    return (
        int(c.get("horizon", 0)),
        int(c.get("num_candidates", 0)),
        int(c.get("sample_steps", 0)),
        float(c.get("temperature", 0.0)),
        str(c.get("sampling_constraints", "")),
        str(c.get("rerank_mode", "")),
        int(c.get("seed", 0)),
    )


def _candidate_grid(base: Dict[str, object], stage: str) -> List[Dict[str, object]]:
    K = int(base.get("num_candidates", 64))
    H = int(base.get("horizon", 5))
    S = int(base.get("sample_steps", 8))
    T = float(base.get("temperature", 1.0))

    if stage == "coarse":
        Ks = sorted({max(8, K // 2), K, int(round(K * 1.5)), K * 2})
        Hs = sorted({max(2, H - 1), H, H + 1, H + 2})
        Ss = sorted({max(2, S // 2), S, S + 2, S + 4})
        Ts = sorted({max(0.5, round(T - 0.2, 2)), round(T, 2), round(T + 0.2, 2)})
    elif stage == "refine":
        Ks = sorted({max(8, K - 16), K, K + 16})
        Hs = sorted({max(2, H - 1), H, H + 1})
        Ss = sorted({max(2, S - 2), S, S + 2})
        Ts = sorted({max(0.5, round(T - 0.1, 2)), round(T, 2), round(T + 0.1, 2)})
    else:  # confirm
        Ks = [K]
        Hs = [H]
        Ss = [S]
        Ts = [T]

    seeds = [int(base.get("seed", 0))]
    if stage == "confirm":
        seeds = sorted({int(base.get("seed", 0)), 1, 2, 3})

    out = []
    for k in Ks:
        for h in Hs:
            for s in Ss:
                for t in Ts:
                    for seed in seeds:
                        c = dict(base)
                        c.update(
                            {
                                "num_candidates": int(k),
                                "horizon": int(h),
                                "sample_steps": int(s),
                                "temperature": float(t),
                                "seed": int(seed),
                            }
                        )
                        out.append(c)
    return out


def _knn_predict(candidate: Dict[str, object], hist_rows: List[Dict[str, str]], objective: str) -> float:
    if not hist_rows:
        return 0.0

    c = {
        "horizon": float(candidate.get("horizon", 5)),
        "K": float(candidate.get("num_candidates", 64)),
        "sampling_steps": float(candidate.get("sample_steps", 8)),
        "temp": float(candidate.get("temperature", 1.0)),
    }
    vals = []
    for r in hist_rows:
        p = _extract_params(r)
        dist = (
            abs(c["horizon"] - p["horizon"]) / max(1.0, c["horizon"])
            + abs(c["K"] - p["K"]) / max(1.0, c["K"])
            + abs(c["sampling_steps"] - p["sampling_steps"]) / max(1.0, c["sampling_steps"])
            + abs(c["temp"] - p["temp"]) / max(0.1, c["temp"])
        )
        w = 1.0 / (1.0 + dist)
        obj = _safe_eval(objective, {k: _to_float(r.get(k, 0.0), 0.0) for k in r.keys()})
        vals.append((w, obj))
    num = sum(w * o for w, o in vals)
    den = sum(w for w, _ in vals) + 1e-8
    return float(num / den)


def _novelty(candidate: Dict[str, object], tried_sigs: set[Tuple]) -> float:
    sig = _signature(candidate)
    return 0.0 if sig in tried_sigs else 1.0


def _score_candidate(candidate: Dict[str, object], hist_rows: List[Dict[str, str]], tried_sigs: set[Tuple], objective: str, stage: str) -> float:
    pred = _knn_predict(candidate, hist_rows, objective)
    nov = _novelty(candidate, tried_sigs)
    if stage == "coarse":
        return pred + 0.5 * nov
    if stage == "refine":
        return pred + 0.15 * nov
    return pred - 0.1 * nov  # confirm prefers near-known configs


def _resolve_plan_output_root(artifact_root: str) -> str:
    if artifact_root.strip():
        return str((Path(artifact_root).expanduser() / "runs" / "planned").resolve())
    env_root = os.environ.get("TETRIS_ARTIFACT_ROOT", "").strip()
    if env_root:
        return str((Path(env_root).expanduser() / "runs" / "planned").resolve())
    if str(default_artifact_root()):
        root = default_artifact_root()
        # For local runs (artifact root defaults to repo/runs), keep backward-compatible output path.
        if root.name == "runs":
            return "runs/planned"
        return str((root / "runs" / "planned").resolve())
    return "runs/planned"


def _build_command(method: str, config: str, out_root: str, idx: int, c: Dict[str, object]) -> str:
    run_dir = os.path.join(out_root, f"proposal_{idx:03d}")
    if method == "selftrain":
        cmd = f"python -m experiments.pipeline selftrain --config {config} --output_dir {run_dir}"
        keys = ["horizon", "num_candidates", "sample_steps", "temperature", "sampling_constraints", "rerank_mode", "seed"]
    elif method == "diffusion_train":
        cmd = f"python -m experiments.pipeline train --config {config} --output_dir {run_dir}"
        keys = ["horizon", "seed"]
    else:
        cmd = f"python -m experiments.pipeline eval --config {config} --output_dir {run_dir}"
        keys = ["horizon", "num_candidates", "sample_steps", "temperature", "sampling_constraints", "rerank_mode", "seed"]
    for k in keys:
        if k in c:
            cmd += f" --{k} {c[k]}"
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Adaptive tuner from runs/index.csv (no brute-force grid).")
    p.add_argument("--index_path", type=str, default="runs/index.csv")
    p.add_argument("--method", type=str, default="diffusion")
    p.add_argument("--variant", type=str, default="")
    p.add_argument("--objective", type=str, default="mean_score + 0.25*cvar10_score - 0.01*mean_runtime_ms - 10*invalid_rate")
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--stage", type=str, default="coarse", choices=["coarse", "refine", "confirm"])
    p.add_argument("--config", type=str, default="configs/eval_run.yaml")
    p.add_argument("--plans_dir", type=str, default="runs/run_plans")
    p.add_argument("--artifact_root", type=str, default="", help="Optional artifact root (commands write to <root>/runs/planned).")
    p.add_argument("--seed", type=int, default=0, help="Fallback seed if no historical runs.")
    args = p.parse_args()

    rows = _read_rows(args.index_path)
    hist = [r for r in rows if r.get("method", "") == args.method]
    if args.variant:
        hist = [r for r in hist if r.get("variant", "") == args.variant]

    if hist:
        hist_scored = []
        for r in hist:
            vars_map = {k: _to_float(r.get(k, 0.0), 0.0) for k in r.keys()}
            hist_scored.append((_safe_eval(args.objective, vars_map), r))
        hist_scored.sort(key=lambda x: x[0], reverse=True)
        best = hist_scored[0][1]
        base = {
            "horizon": int(_to_float(best.get("horizon", 5), 5)),
            "num_candidates": int(_to_float(best.get("K", 64), 64)),
            "sample_steps": int(_to_float(best.get("sampling_steps", 8), 8)),
            "temperature": float(_to_float(best.get("temp", 1.0), 1.0)),
            "sampling_constraints": best.get("sampling_constraints", "none") or "none",
            "rerank_mode": best.get("rerank_mode", "heuristic") or "heuristic",
            "seed": int(_to_float(best.get("seed", args.seed), args.seed)),
        }
    else:
        base = {
            "horizon": 5,
            "num_candidates": 64,
            "sample_steps": 8,
            "temperature": 1.0,
            "sampling_constraints": "none",
            "rerank_mode": "heuristic",
            "seed": int(args.seed),
        }

    existing_cfg_seed = {(r.get("config_hash", ""), str(r.get("seed", ""))) for r in rows}
    tried_sigs = {
        (
            int(_to_float(r.get("horizon", 0), 0)),
            int(_to_float(r.get("K", 0), 0)),
            int(_to_float(r.get("sampling_steps", 0), 0)),
            float(_to_float(r.get("temp", 0.0), 0.0)),
            str(r.get("sampling_constraints", "")),
            str(r.get("rerank_mode", "")),
            int(_to_float(r.get("seed", 0), 0)),
        )
        for r in rows
    }

    grid = _candidate_grid(base, stage=args.stage)
    scored = []
    for c in grid:
        cfg_for_hash = dict(c)
        cfg_for_hash["method"] = args.method
        cfg_hash = _normalize_cfg_hash(cfg_for_hash)
        seed_s = str(int(c.get("seed", 0)))
        if (cfg_hash, seed_s) in existing_cfg_seed:
            continue
        if _signature(c) in tried_sigs:
            continue
        s = _score_candidate(c, hist, tried_sigs, args.objective, args.stage)
        scored.append((s, c, cfg_hash))
    scored.sort(key=lambda x: x[0], reverse=True)

    proposals = []
    seen = set()
    for rank, (score, c, cfg_hash) in enumerate(scored, start=1):
        sig = _signature(c)
        if sig in seen:
            continue
        seen.add(sig)
        proposals.append(
            {
                "rank": rank,
                "acq_score": float(score),
                "config_hash": cfg_hash,
                "seed": int(c["seed"]),
                "overrides": c,
            }
        )
        if len(proposals) >= int(args.budget):
            break

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.plans_dir, exist_ok=True)
    plan_path = os.path.join(args.plans_dir, f"{ts}_plan.yaml")
    plan = {
        "timestamp": ts,
        "method": args.method,
        "variant": args.variant,
        "objective": args.objective,
        "budget": int(args.budget),
        "stage": args.stage,
        "index_path": args.index_path,
        "config": args.config,
        "artifact_root": args.artifact_root,
        "proposals": [],
    }
    out_root = _resolve_plan_output_root(args.artifact_root)
    for i, pinfo in enumerate(proposals, start=1):
        cmd = _build_command(args.method, args.config, out_root, i, pinfo["overrides"])
        entry = dict(pinfo)
        entry["command"] = cmd
        plan["proposals"].append(entry)

    with open(plan_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(plan, f, sort_keys=False)

    print(f"Plan saved: {plan_path}")
    for pinfo in plan["proposals"]:
        print(pinfo["command"])


if __name__ == "__main__":
    main()
