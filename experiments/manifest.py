from __future__ import annotations

import datetime as dt
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from experiments.hashing import write_json


def _try_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"unavailable ({exc})"


def _collect_env(prefixes: tuple[str, ...] = ("LIGHTNING_", "CUDA_", "TETRIS_")) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in os.environ.items():
        if any(k.startswith(pref) for pref in prefixes):
            out[k] = v
    return dict(sorted(out.items()))


def write_manifest(
    output_dir: str,
    command: str,
    resolved_config: Dict[str, Any],
    git_commit: str = "",
    config_hash: str = "",
    dataset_hash: str = "",
    model_hash: str = "",
    extra_fields: Dict[str, Any] | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = str(Path(output_dir) / "manifest.json")

    manifest = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_dir": str(Path(output_dir).resolve()),
        "command": command,
        "resolved_config": resolved_config,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "model_hash": model_hash,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "env": _collect_env(),
        "pip_freeze": _try_cmd([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    if extra_fields:
        manifest.update(extra_fields)
    write_json(manifest_path, manifest)
    return manifest_path
