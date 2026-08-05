from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys


def _git_head(repo_root: Path, relative_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(relative_path), "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_ros_environment_uses_python_3_8() -> None:
    assert sys.version_info[:2] == (3, 8)


def test_diffusiondrive_v2_submodule_is_pinned(repo_root: Path) -> None:
    assert _git_head(repo_root, Path("third_party/DiffusionDriveV2")) == (
        "1cd12a1e155c34dcc471261835444c8d5587580b"
    )


def test_validator_submodule_is_pinned(repo_root: Path) -> None:
    assert _git_head(repo_root, Path("third_party/E2E-OpenLoop-ROS-Validator")) == (
        "09a3b88e19ae3354903ed676a35e02416284cb58"
    )


def test_sensor_preset_is_read_only_user_input(repo_root: Path) -> None:
    preset = repo_root / "sensor_data/0725demo_ros.json"
    assert preset.is_file()
    assert hashlib.sha256(preset.read_bytes()).hexdigest() == (
        "b7e69c70cc0b2948c3a5b352eee022be04f8884a1df135697f89bb4bb88263fb"
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(preset)], cwd=repo_root, check=False
    )
    assert ignored.returncode == 0
