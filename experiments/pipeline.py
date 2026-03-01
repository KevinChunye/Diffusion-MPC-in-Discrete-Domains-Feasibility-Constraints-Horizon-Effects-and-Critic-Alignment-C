from __future__ import annotations

import argparse
import csv
import glob
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from experiments.config_utils import load_yaml_config, parse_unknown_overrides
from experiments.hashing import read_json, sha256_file
from experiments.lightning_env import default_artifact_root, is_lightning_runtime
from experiments.manifest import write_manifest
from experiments.registry import register_directory_summaries, register_stage


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def _parse_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s == "":
        return default
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _artifact_root_for_paths() -> Path | None:
    explicit = os.environ.get("TETRIS_ARTIFACT_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if is_lightning_runtime():
        return default_artifact_root()
    return None


def _resolve_path(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    base = _artifact_root_for_paths() or Path(_repo_root())
    return str((base / p).resolve())


def _index_path() -> str:
    explicit = os.environ.get("TETRIS_INDEX_PATH", "").strip()
    if explicit:
        return str(Path(explicit).expanduser())
    base = _artifact_root_for_paths()
    if base is not None:
        return str((base / "runs" / "index.csv").resolve())
    return "runs/index.csv"


def _to_cli_tokens(overrides: Dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                out.append(flag)
            else:
                out.extend([flag, "false"])
        else:
            out.extend([flag, str(v)])
    return out


def _run_cmd(cmd: list[str]) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=_repo_root())


def _build_cmd(script_module_or_path: str, config: str, output_dir: str, smoke: bool, extra: list[str]) -> list[str]:
    python = sys.executable
    if script_module_or_path.startswith("module:"):
        mod = script_module_or_path.split(":", 1)[1]
        cmd = [python, "-m", mod]
    else:
        cmd = [python, script_module_or_path]
    cmd += ["--config", config, "--output_dir", output_dir]
    cmd += extra
    if smoke:
        # Stage-specific smoke defaults.
        if "make_dataset" in script_module_or_path:
            cmd += ["--episodes", "5", "--max_steps_per_episode", "100"]
        elif "train_diffusion_updated.py" in script_module_or_path:
            cmd += ["--epochs", "1", "--batch_size", "32"]
        elif "run_diffusion_mpc_updated.py" in script_module_or_path:
            cmd += ["--episodes", "5", "--max_steps", "200"]
    return cmd


def _last_csv_row(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _resolve_resume(explicit_resume: str, config_path: str) -> bool:
    v = _parse_bool(explicit_resume, default=None)
    if v is not None:
        return bool(v)
    try:
        cfg = load_yaml_config(config_path)
        cv = _parse_bool(cfg.get("resume", None), default=None)
        if cv is not None:
            return bool(cv)
    except Exception:
        pass
    return is_lightning_runtime()


def _dataset_ready(output_dir: str, config: Dict[str, Any]) -> bool:
    dataset_dir = _resolve_path(str(config.get("out_dir", output_dir)))
    seq_ok = bool(glob.glob(os.path.join(dataset_dir, "sequences_H*.npz")))
    meta_ok = os.path.exists(os.path.join(dataset_dir, "meta.json")) or os.path.exists(os.path.join(output_dir, "meta.json"))
    return seq_ok and meta_ok


def _train_ready(output_dir: str, config: Dict[str, Any]) -> bool:
    ckpt_candidates = [
        os.path.join(output_dir, "checkpoints", "ckpt.pt"),
        os.path.join(output_dir, "checkpoints", "plan_denoiser.pt"),
    ]
    save_path = str(config.get("save", "")).strip()
    if save_path:
        ckpt_candidates.append(_resolve_path(save_path))
        ckpt_meta_candidate = os.path.join(os.path.dirname(_resolve_path(save_path)), "ckpt_meta.json")
    else:
        ckpt_meta_candidate = ""
    ckpt_ok = any(os.path.exists(p) for p in ckpt_candidates)
    ckpt_meta_ok = os.path.exists(os.path.join(output_dir, "checkpoints", "ckpt_meta.json")) or (
        bool(ckpt_meta_candidate) and os.path.exists(ckpt_meta_candidate)
    )
    train_metrics_ok = os.path.exists(os.path.join(output_dir, "train_metrics.csv"))
    return ckpt_ok and ckpt_meta_ok and train_metrics_ok


def _eval_ready(output_dir: str, _config: Dict[str, Any]) -> bool:
    summary_ok = os.path.exists(os.path.join(output_dir, "summary.csv"))
    metrics_ok = os.path.exists(os.path.join(output_dir, "metrics.csv"))
    if not (summary_ok and metrics_ok):
        return False
    summary = _last_csv_row(os.path.join(output_dir, "summary.csv"))
    planner = str(summary.get("planner", "")).strip().lower()
    if planner == "beam_search":
        return True
    return bool(glob.glob(os.path.join(output_dir, "*_decisions.csv")))


def _selftrain_ready(output_dir: str, _config: Dict[str, Any]) -> bool:
    progress_ok = os.path.exists(os.path.join(output_dir, "progress.csv"))
    iter_eval_ok = bool(glob.glob(os.path.join(output_dir, "iter_*", "eval", "summary.csv")))
    return progress_ok and iter_eval_ok


def _load_resolved_config(output_dir: str, config_path: str, overrides: Dict[str, Any], forced: Dict[str, Any]) -> Dict[str, Any]:
    cfg_path = os.path.join(output_dir, "config.yaml")
    if os.path.exists(cfg_path):
        return load_yaml_config(cfg_path)
    cfg: Dict[str, Any] = {}
    try:
        cfg = load_yaml_config(config_path)
    except Exception:
        cfg = {}
    cfg.update(overrides)
    cfg.update(forced)
    return cfg


def _collect_manifest_hashes(output_dir: str) -> tuple[str, str, str, str]:
    cfg_path = os.path.join(output_dir, "config.yaml")
    config_hash = sha256_file(cfg_path)
    git_commit = ""
    git_path = os.path.join(output_dir, "git_commit.txt")
    if os.path.exists(git_path):
        with open(git_path, "r", encoding="utf-8") as f:
            git_commit = f.read().strip()

    summary = _last_csv_row(os.path.join(output_dir, "summary.csv"))
    dataset_hash = str(summary.get("dataset_hash", "")).strip()
    model_hash = str(summary.get("model_hash", "")).strip()

    if not dataset_hash:
        meta = read_json(os.path.join(output_dir, "meta.json"))
        dataset_hash = str(meta.get("dataset_hash", "")).strip()
    if not model_hash:
        ckpt_meta = read_json(os.path.join(output_dir, "checkpoints", "ckpt_meta.json"))
        model_hash = str(ckpt_meta.get("model_hash", "")).strip()
    return git_commit, config_hash, dataset_hash, model_hash


def _write_stage_manifest(output_dir: str, command: list[str], config: Dict[str, Any]) -> None:
    git_commit, config_hash, dataset_hash, model_hash = _collect_manifest_hashes(output_dir)
    summary = _last_csv_row(os.path.join(output_dir, "summary.csv"))
    extra_fields: Dict[str, Any] = {}
    for k in ["video_files", "selected_episode", "final_score"]:
        if k in summary and str(summary.get(k, "")).strip():
            extra_fields[k] = summary[k]
    manifest_path = write_manifest(
        output_dir=output_dir,
        command=shlex.join(command),
        resolved_config=config,
        git_commit=git_commit,
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        model_hash=model_hash,
        extra_fields=extra_fields,
    )
    print(f"[manifest] {manifest_path}")


def _dataset(args: argparse.Namespace, passthrough: list[str], overrides: Dict[str, Any], index_path: str) -> None:
    out = _resolve_path(args.output_dir)
    os.makedirs(out, exist_ok=True)

    cfg_preview = load_yaml_config(args.config)
    cfg_preview.update(overrides)
    forced = {"output_dir": out}
    dataset_out_dir = str(cfg_preview.get("out_dir", ""))
    if dataset_out_dir:
        forced["out_dir"] = _resolve_path(dataset_out_dir)
    effective_cfg = dict(cfg_preview)
    effective_cfg.update(forced)

    resume = _resolve_resume(args.resume, args.config)
    if resume and _dataset_ready(out, effective_cfg):
        print(f"[skip] dataset stage: artifacts already exist at {out}")
        register_stage(out, stage="dataset", method="dataset_pipeline", index_path=index_path)
        _write_stage_manifest(
            out,
            ["python", "-m", "experiments.make_dataset", "--config", args.config, "--output_dir", out, "[SKIPPED]"],
            _load_resolved_config(out, args.config, overrides, forced),
        )
        return

    forced_cli = {}
    if "out_dir" in forced:
        forced_cli["out_dir"] = forced["out_dir"]
    extra = passthrough + _to_cli_tokens(forced_cli)
    cmd = _build_cmd("module:experiments.make_dataset", args.config, out, args.smoke, extra)
    _run_cmd(cmd)
    register_stage(out, stage="dataset", method="dataset_pipeline", index_path=index_path)
    _write_stage_manifest(out, cmd, _load_resolved_config(out, args.config, overrides, forced))


def _train(args: argparse.Namespace, passthrough: list[str], overrides: Dict[str, Any], index_path: str) -> None:
    out = _resolve_path(args.output_dir)
    os.makedirs(out, exist_ok=True)
    forced = {"output_dir": out}
    effective_cfg = load_yaml_config(args.config)
    effective_cfg.update(overrides)
    effective_cfg.update(forced)

    resume = _resolve_resume(args.resume, args.config)
    if resume and _train_ready(out, effective_cfg):
        print(f"[skip] train stage: artifacts already exist at {out}")
        register_stage(out, stage="train", method="diffusion_train", index_path=index_path)
        _write_stage_manifest(
            out,
            ["python", "diffusion/train_diffusion_updated.py", "--config", args.config, "--output_dir", out, "[SKIPPED]"],
            _load_resolved_config(out, args.config, overrides, forced),
        )
        return

    extra = passthrough
    cmd = _build_cmd("diffusion/train_diffusion_updated.py", args.config, out, args.smoke, extra)
    _run_cmd(cmd)
    register_stage(out, stage="train", method="diffusion_train", index_path=index_path)
    _write_stage_manifest(out, cmd, _load_resolved_config(out, args.config, overrides, forced))


def _eval(args: argparse.Namespace, passthrough: list[str], overrides: Dict[str, Any], index_path: str) -> None:
    out = _resolve_path(args.output_dir)
    os.makedirs(out, exist_ok=True)
    forced = {"output_dir": out}
    effective_cfg = load_yaml_config(args.config)
    effective_cfg.update(overrides)
    effective_cfg.update(forced)

    resume = _resolve_resume(args.resume, args.config)
    if resume and _eval_ready(out, effective_cfg):
        print(f"[skip] eval stage: artifacts already exist at {out}")
        register_stage(out, stage="eval", method="diffusion", index_path=index_path)
        _write_stage_manifest(
            out,
            ["python", "diffusion/run_diffusion_mpc_updated.py", "--config", args.config, "--output_dir", out, "[SKIPPED]"],
            _load_resolved_config(out, args.config, overrides, forced),
        )
        return

    extra = passthrough
    cmd = _build_cmd("diffusion/run_diffusion_mpc_updated.py", args.config, out, args.smoke, extra)
    _run_cmd(cmd)
    register_stage(out, stage="eval", method="diffusion", index_path=index_path)
    _write_stage_manifest(out, cmd, _load_resolved_config(out, args.config, overrides, forced))


def _selftrain(args: argparse.Namespace, passthrough: list[str], overrides: Dict[str, Any], index_path: str) -> None:
    out = _resolve_path(args.output_dir)
    os.makedirs(out, exist_ok=True)
    forced = {"output_dir": out}
    effective_cfg = load_yaml_config(args.config)
    effective_cfg.update(overrides)
    effective_cfg.update(forced)

    resume = _resolve_resume(args.resume, args.config)
    if resume and _selftrain_ready(out, effective_cfg):
        print(f"[skip] selftrain stage: artifacts already exist at {out}")
        register_directory_summaries(out, stage_prefix="selftrain", index_path=index_path)
        _write_stage_manifest(
            out,
            ["python", "-m", "experiments.selftrain", "--config", args.config, "--output_dir", out, "[SKIPPED]"],
            _load_resolved_config(out, args.config, overrides, forced),
        )
        return

    extra = passthrough
    cmd = _build_cmd("module:experiments.selftrain", args.config, out, args.smoke, extra)
    if args.smoke and "--smoke" not in cmd:
        cmd.append("--smoke")
    _run_cmd(cmd)
    register_directory_summaries(out, stage_prefix="selftrain", index_path=index_path)
    _write_stage_manifest(out, cmd, _load_resolved_config(out, args.config, overrides, forced))


def _full(args: argparse.Namespace, passthrough: list[str], overrides: Dict[str, Any], index_path: str) -> None:
    root_out = _resolve_path(args.output_dir)
    os.makedirs(root_out, exist_ok=True)

    ds_out = os.path.join(root_out, "dataset")
    tr_out = os.path.join(root_out, "train")
    ev_out = os.path.join(root_out, "eval")

    _dataset(
        argparse.Namespace(config=args.dataset_config, output_dir=ds_out, smoke=args.smoke, resume=args.resume),
        passthrough,
        {**overrides, "out_dir": ds_out},
        index_path=index_path,
    )

    ds_cfg = load_yaml_config(args.dataset_config)
    seq_candidates = sorted(glob.glob(os.path.join(ds_out, "sequences_H*.npz")))
    if seq_candidates:
        seq_path = seq_candidates[-1]
        horizon = int(Path(seq_path).stem.split("H")[-1])
    else:
        horizon = int(ds_cfg.get("horizon", 8))
        seq_path = os.path.join(ds_out, f"sequences_H{horizon}.npz")

    _train(
        argparse.Namespace(config=args.train_config, output_dir=tr_out, smoke=args.smoke, resume=args.resume),
        passthrough + ["--dataset_path", seq_path],
        overrides,
        index_path=index_path,
    )

    ckpt_path = os.path.join(tr_out, "checkpoints", "plan_denoiser.pt")
    _eval(
        argparse.Namespace(config=args.eval_config, output_dir=ev_out, smoke=args.smoke, resume=args.resume),
        passthrough + ["--ckpt", ckpt_path, "--horizon", str(horizon)],
        overrides,
        index_path=index_path,
    )

    _write_stage_manifest(
        root_out,
        ["python", "-m", "experiments.pipeline", "full", "--output_dir", root_out],
        {"dataset_output": ds_out, "train_output": tr_out, "eval_output": ev_out, "resume": _resolve_resume(args.resume, args.eval_config)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightning-ready pipeline runner.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ds = sub.add_parser("dataset")
    p_ds.add_argument("--config", type=str, default="configs/dataset_gen.yaml")
    p_ds.add_argument("--output_dir", type=str, default="runs/pipeline_dataset")
    p_ds.add_argument("--resume", type=str, default="")
    p_ds.add_argument("--smoke", action="store_true")

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--config", type=str, default="configs/diffusion_train.yaml")
    p_tr.add_argument("--output_dir", type=str, default="runs/pipeline_train")
    p_tr.add_argument("--resume", type=str, default="")
    p_tr.add_argument("--smoke", action="store_true")

    p_ev = sub.add_parser("eval")
    p_ev.add_argument("--config", type=str, default="configs/eval_run.yaml")
    p_ev.add_argument("--output_dir", type=str, default="runs/pipeline_eval")
    p_ev.add_argument("--resume", type=str, default="")
    p_ev.add_argument("--smoke", action="store_true")

    p_full = sub.add_parser("full")
    p_full.add_argument("--dataset_config", type=str, default="configs/dataset_gen.yaml")
    p_full.add_argument("--train_config", type=str, default="configs/diffusion_train.yaml")
    p_full.add_argument("--eval_config", type=str, default="configs/eval_run.yaml")
    p_full.add_argument("--output_dir", type=str, default="runs/pipeline_full")
    p_full.add_argument("--resume", type=str, default="")
    p_full.add_argument("--smoke", action="store_true")

    p_self = sub.add_parser("selftrain")
    p_self.add_argument("--config", type=str, default="configs/selftrain.yaml")
    p_self.add_argument("--output_dir", type=str, default="runs/pipeline_selftrain")
    p_self.add_argument("--resume", type=str, default="")
    p_self.add_argument("--smoke", action="store_true")

    args, unknown = parser.parse_known_args()
    passthrough = []
    overrides = parse_unknown_overrides(unknown)
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool) and v:
            passthrough.append(flag)
        else:
            passthrough.extend([flag, str(v)])

    index_path = _index_path()
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

    if args.cmd == "dataset":
        _dataset(args, passthrough, overrides, index_path=index_path)
    elif args.cmd == "train":
        _train(args, passthrough, overrides, index_path=index_path)
    elif args.cmd == "eval":
        _eval(args, passthrough, overrides, index_path=index_path)
    elif args.cmd == "full":
        _full(args, passthrough, overrides, index_path=index_path)
    elif args.cmd == "selftrain":
        _selftrain(args, passthrough, overrides, index_path=index_path)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
