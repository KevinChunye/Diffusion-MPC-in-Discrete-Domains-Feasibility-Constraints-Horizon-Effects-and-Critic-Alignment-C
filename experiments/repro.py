from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from experiments.hashing import sha256_file


def _try_cmd(cmd: list[str], cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"unavailable ({exc})"


def prepare_run_artifacts(output_dir: str, resolved_config: Dict[str, Any]) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=True)
    config_hash = sha256_file(config_path)

    repo_root = str(Path(__file__).resolve().parents[1])
    git_commit = _try_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    with open(os.path.join(output_dir, "git_commit.txt"), "w", encoding="utf-8") as f:
        f.write(git_commit + "\n")

    pip_freeze = _try_cmd([sys.executable, "-m", "pip", "freeze"])
    with open(os.path.join(output_dir, "pip_freeze.txt"), "w", encoding="utf-8") as f:
        f.write(pip_freeze + "\n")

    return {
        "config_path": config_path,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "checkpoints_dir": ckpt_dir,
        "git_commit_path": os.path.join(output_dir, "git_commit.txt"),
        "pip_freeze_path": os.path.join(output_dir, "pip_freeze.txt"),
    }
