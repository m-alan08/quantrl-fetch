from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "torch",
    "numpy",
    "mujoco",
    "gymnasium",
    "gymnasium-robotics",
    "imageio",
    "matplotlib",
    "pandas",
)


def _git_output(arguments: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    directory = cwd or Path(__file__).resolve().parent
    commit = _git_output(["rev-parse", "HEAD"], directory)
    status = _git_output(["status", "--porcelain", "--untracked-files=no"], directory)
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def collect_run_metadata(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "git": git_state(),
        "packages": package_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
