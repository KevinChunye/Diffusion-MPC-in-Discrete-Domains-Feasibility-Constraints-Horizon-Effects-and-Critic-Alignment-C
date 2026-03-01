from __future__ import annotations

import argparse
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _existing_dir(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path).expanduser()
    return p if p.exists() and p.is_dir() else None


def is_lightning_runtime() -> bool:
    if any(k.startswith("LIGHTNING_") for k in os.environ):
        return True
    if Path("/teamspace").exists():
        return True
    cwd = str(Path.cwd())
    return cwd.startswith("/teamspace/")


def detect_persistent_root() -> Path:
    for key in [
        "TETRIS_PERSISTENT_ROOT",
        "LIGHTNING_PERSISTENT_DIR",
        "LIGHTNING_HOME",
        "LIGHTNING_CLOUD_HOME",
        "TEAMSPACE_ROOT",
    ]:
        p = _existing_dir(os.environ.get(key, ""))
        if p is not None:
            return p

    cwd = Path.cwd().resolve()
    parts = cwd.parts
    if "teamspace" in parts:
        idx = parts.index("teamspace")
        base = Path(*parts[: idx + 1])
        # Prefer /teamspace/studios when possible.
        studios = base / "studios"
        if studios.exists():
            return studios
        return base

    p = _existing_dir("/teamspace/studios")
    if p is not None:
        return p
    p = _existing_dir("/teamspace")
    if p is not None:
        return p

    # Local fallback: keep everything repo-local by default.
    return _repo_root()


def default_project_root() -> Path:
    p = _existing_dir(os.environ.get("TETRIS_PROJECT_ROOT", ""))
    if p is not None:
        return p
    return _repo_root()


def default_artifact_root() -> Path:
    explicit = os.environ.get("TETRIS_ARTIFACT_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if is_lightning_runtime():
        return detect_persistent_root() / "tetris_artifacts"
    return default_project_root() / "runs"


def main() -> None:
    p = argparse.ArgumentParser(description="Lightning path detection helpers.")
    p.add_argument("--print", dest="do_print", action="store_true", help="Print detected paths and exports.")
    args = p.parse_args()

    if not args.do_print:
        return

    persistent = detect_persistent_root()
    project = default_project_root()
    artifact = default_artifact_root()

    print(f"lightning_runtime={str(is_lightning_runtime()).lower()}")
    print(f"persistent_root={persistent}")
    print(f"project_root={project}")
    print(f"artifact_root={artifact}")
    print(f"recommended_export=export TETRIS_ARTIFACT_ROOT={artifact}")


if __name__ == "__main__":
    main()
