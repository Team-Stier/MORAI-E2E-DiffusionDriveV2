from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict

import pytest


DDV2_COMMIT = "1cd12a1e155c34dcc471261835444c8d5587580b"
VALIDATOR_COMMIT = "09a3b88e19ae3354903ed676a35e02416284cb58"
EXPECTED_ASSET_HASHES = {
    "checkpoint": "452c4e0dd328c49b981f3fd5ed13eb1c6a2b0f3a9486adf43b40a5645ed543b1",
    "image_backbone": "f6c1d7785b5d0e796a83f9643fe2cb952ac95740daa9fb2d5c8c6514fdadaeb3",
    "trajectory_anchor": "2d341e525d7882ee5e916c67d37bea571e4c793cf801f0fff3eff59d38b20d98",
}
DERIVATIVE_RELATIVE_PATH = Path(
    "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel_inference.py"
)
TRAINING_REQUIREMENTS_BYTES = (
    b"-r ../third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/"
    b"validator_inference/requirements/diffusiondrive_v2.lock\n"
    b"h5py==3.10.0\n"
    b"mypy==1.14.1\n"
    b"mypy_extensions==1.1.0\n"
    b"rfc8785==0.1.4\n"
    b"ruff==0.12.7\n"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _selected_profile(profile_path: Path) -> Dict[str, Any]:
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    matches = [
        item for item in payload["profiles"] if item["id"] == "diffusiondrive_v2"
    ]
    assert len(matches) == 1
    return matches[0]


def test_training_runtime_versions() -> None:
    import cv2
    import diffusers
    import h5py
    import mypy_extensions
    import numpy
    import scipy
    import timm
    import torch
    import torchvision
    import yaml

    assert sys.version_info[:3] == (3, 9, 25)
    assert torch.__version__ == "2.0.1+cu118"
    assert torch.version.cuda == "11.8"
    assert torchvision.__version__ == "0.15.2+cu118"
    assert diffusers.__version__ == "0.35.2"
    assert numpy.__version__ == "1.23.4"
    assert scipy.__version__ == "1.13.1"
    assert timm.__version__ == "1.0.28"
    assert cv2.__version__ == "4.9.0"
    assert yaml.__version__ == "6.0.3"
    assert h5py.__version__ == "3.10.0"
    expected = {
        "torch": "2.0.1+cu118",
        "torchvision": "0.15.2+cu118",
        "diffusers": "0.35.2",
        "numpy": "1.23.4",
        "scipy": "1.13.1",
        "timm": "1.0.28",
        "opencv-python": "4.9.0.80",
        "PyYAML": "6.0.3",
        "pytest": "8.4.2",
        "h5py": "3.10.0",
        "rfc8785": "0.1.4",
        "ruff": "0.12.7",
        "mypy": "1.14.1",
        "mypy-extensions": "1.1.0",
    }
    observed = {
        distribution: importlib.metadata.version(distribution)
        for distribution in expected
    }
    assert observed == expected
    assert mypy_extensions.__name__ == "mypy_extensions"


def test_training_sync_input_and_bootstrap_contract(repo_root: Path) -> None:
    requirements_path = repo_root / "requirements/training.txt"
    assert requirements_path.read_bytes() == TRAINING_REQUIREMENTS_BYTES

    bootstrap = (repo_root / "scripts/bootstrap_train_env.sh").read_text(
        encoding="utf-8"
    )
    backend = "export UV_TORCH_BACKEND=cu118"
    sanitize_pythonpath = "unset PYTHONPATH"
    install_python = "/home/stier/.local/bin/uv python install 3.9.25"
    create_venv = "/home/stier/.local/bin/uv venv --python 3.9.25 .venv-train"
    existing_venv_guard = "if [[ ! -x .venv-train/bin/python ]]; then"
    validate_python = (
        ".venv-train/bin/python -c 'from pathlib import Path; import sys; "
        "assert sys.version_info[:3] == (3, 9, 25); assert "
        "Path(sys.prefix).resolve() == (Path.cwd() / \".venv-train\").resolve()'"
    )
    sync = (
        "/home/stier/.local/bin/uv pip sync --python .venv-train/bin/python "
        "requirements/training.txt"
    )
    assert existing_venv_guard in bootstrap
    assert create_venv in bootstrap
    assert validate_python in bootstrap
    assert backend in bootstrap
    assert sync in bootstrap
    assert sanitize_pythonpath in bootstrap
    assert bootstrap.index(sanitize_pythonpath) < bootstrap.index(install_python)
    assert bootstrap.index(existing_venv_guard) < bootstrap.index(create_venv)
    assert bootstrap.index(create_venv) < bootstrap.index(validate_python)
    assert bootstrap.index(backend) < bootstrap.index(sync)


def test_documented_training_commands_sanitize_pythonpath(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    expected_commands = (
        "env -u PYTHONPATH bash scripts/bootstrap_train_env.sh",
        "env -u PYTHONPATH bash scripts/fetch_ddv2_assets.sh",
        "env -u PYTHONPATH .venv-train/bin/python -m pytest \\",
    )
    for command in expected_commands:
        assert command in readme


def test_training_runtime_does_not_mix_ros_system_packages() -> None:
    forbidden_fragments = (
        "/usr/lib/python3/dist-packages",
        "/opt/ros/",
        "/catkin_ws/",
    )
    resolved_entries = [Path(entry).resolve().as_posix() for entry in sys.path if entry]
    for entry in resolved_entries:
        assert all(fragment not in entry for fragment in forbidden_fragments)

    for module_name in ("rospy", "rosbag", "cv_bridge"):
        assert importlib.util.find_spec(module_name) is None


def _prepare_fake_asset_repository(tmp_path: Path, repo_root: Path) -> Path:
    fake_root = tmp_path / "repository"
    script = fake_root / "scripts/fetch_ddv2_assets.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(repo_root / "scripts/fetch_ddv2_assets.sh", script)

    python = fake_root / ".venv-train/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve())

    validator = fake_root / "third_party/E2E-OpenLoop-ROS-Validator"
    fetcher = (
        validator
        / "src/Validator_pkg/validator_inference/scripts/fetch_model_assets.py"
    )
    fetcher.parent.mkdir(parents=True)
    fetcher.write_text(
        """from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


expected = [
    "--model-id", "diffusiondrive_v2",
    "--profile-file",
    "third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json",
    "--project-root", "third_party/E2E-OpenLoop-ROS-Validator",
]
if sys.argv[1:] != expected:
    raise SystemExit(f"unexpected official fetch arguments: {sys.argv[1:]!r}")

parser = argparse.ArgumentParser()
parser.add_argument("--model-id")
parser.add_argument("--profile-file")
parser.add_argument("--project-root")
arguments = parser.parse_args()
manifest = (
    Path(arguments.project_root)
    / "models/diffusiondrive_v2/asset-manifest.json"
)
manifest.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "model_id": "diffusiondrive_v2",
    "source": {"observed_commit": "1cd12a1"},
    "assets": [{"role": "checkpoint", "observed_sha256": "fresh"}],
}
if os.environ.get("FAKE_STALE_MANIFEST") == "1":
    payload["assets"][0]["observed_sha256"] = "stale"
manifest.write_text(json.dumps(payload), encoding="utf-8")
""",
        encoding="utf-8",
    )

    package = (
        validator / "src/Validator_pkg/validator_inference/src/validator_inference"
    )
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "assets.py").write_text(
        """from __future__ import annotations


def verify_runtime_assets(profile, project_root):
    return {
        "source": {"observed_commit": "1cd12a1"},
        "assets": [{"role": "checkpoint", "observed_sha256": "fresh"}],
    }
""",
        encoding="utf-8",
    )
    (package / "profile_store.py").write_text(
        """from __future__ import annotations


class SelectedProfile:
    data = {"id": "diffusiondrive_v2"}


def load_selected_profile(profile_file, model_id):
    if model_id != "diffusiondrive_v2":
        raise ValueError(model_id)
    return SelectedProfile()
""",
        encoding="utf-8",
    )
    profile = validator / "src/Validator_pkg/config/profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("{}\n", encoding="utf-8")
    source = validator / "models/sources/diffusiondrive_v2"
    source.mkdir(parents=True)
    (source / "README.md").write_text("fake source\n", encoding="utf-8")
    ddv2 = fake_root / "third_party/DiffusionDriveV2"
    ddv2.mkdir(parents=True)
    (ddv2 / "README.md").write_text("fake source\n", encoding="utf-8")
    _init_git_repository(source)
    _init_git_repository(ddv2)
    _init_git_repository(validator)
    return fake_root


def _init_git_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "MORAI test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True
    )


def _run_fake_asset_fetch(
    fake_root: Path, *, stale: bool = False
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "/opt/ros/noetic/lib/python3/dist-packages"
    if stale:
        environment["FAKE_STALE_MANIFEST"] = "1"
    return subprocess.run(
        ["bash", str(fake_root / "scripts/fetch_ddv2_assets.sh")],
        cwd=fake_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def test_asset_fetch_script_is_idempotent_and_freshly_verifies_manifest(
    tmp_path: Path, repo_root: Path
) -> None:
    fake_root = _prepare_fake_asset_repository(tmp_path, repo_root)

    first = _run_fake_asset_fetch(fake_root)
    assert first.returncode == 0, first.stderr
    manifest = (
        fake_root
        / "third_party/E2E-OpenLoop-ROS-Validator/models/diffusiondrive_v2/asset-manifest.json"
    )
    first_bytes = manifest.read_bytes()

    second = _run_fake_asset_fetch(fake_root)
    assert second.returncode == 0, second.stderr
    assert manifest.read_bytes() == first_bytes


def test_asset_fetch_script_propagates_fresh_manifest_mismatch(
    tmp_path: Path, repo_root: Path
) -> None:
    fake_root = _prepare_fake_asset_repository(tmp_path, repo_root)

    result = _run_fake_asset_fetch(fake_root, stale=True)

    assert result.returncode != 0
    assert "does not match fresh Validator verification" in result.stderr


def test_asset_fetch_script_rejects_tracked_source_changes(
    tmp_path: Path, repo_root: Path
) -> None:
    fake_root = _prepare_fake_asset_repository(tmp_path, repo_root)
    profile = (
        fake_root
        / "third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json"
    )
    profile.write_text("{\"changed\":true}\n", encoding="utf-8")

    result = _run_fake_asset_fetch(fake_root)

    assert result.returncode != 0
    assert "tracked checkout is not clean" in result.stderr


def test_pinned_source_checkouts_match_gitlinks(repo_root: Path) -> None:
    expected = {
        "third_party/DiffusionDriveV2": DDV2_COMMIT,
        "third_party/E2E-OpenLoop-ROS-Validator": VALIDATOR_COMMIT,
    }
    for relative_path, commit in expected.items():
        checkout = repo_root / relative_path
        assert _git_output(checkout, "rev-parse", "HEAD") == commit
        assert _git_output(checkout, "status", "--porcelain", "--untracked-files=no") == ""
        gitlink = subprocess.run(
            ["git", "-C", str(repo_root), "ls-tree", "HEAD", relative_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.split()
        assert gitlink == ["160000", "commit", commit, relative_path]


def test_official_ddv2_assets_and_generated_derivative_are_verified(
    repo_root: Path,
) -> None:
    validator_root = repo_root / "third_party/E2E-OpenLoop-ROS-Validator"
    profile_path = validator_root / "src/Validator_pkg/config/profile.json"
    profile = _selected_profile(profile_path)
    assert profile["source"]["commit"] == DDV2_COMMIT

    declarations = {item["role"]: item for item in profile["runtime_assets"]}
    assert set(declarations) == set(EXPECTED_ASSET_HASHES)
    for role, expected_hash in EXPECTED_ASSET_HASHES.items():
        declaration = declarations[role]
        assert declaration["sha256"] == expected_hash
        asset_path = validator_root / declaration["path"]
        assert asset_path.is_file()
        assert _sha256(asset_path) == expected_hash

    manifest_path = validator_root / "models/diffusiondrive_v2/asset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == "diffusiondrive_v2"
    assert manifest["source"]["commit"] == DDV2_COMMIT
    assert manifest["source"]["observed_commit"] == DDV2_COMMIT
    manifest_assets = {item["role"]: item for item in manifest["assets"]}
    assert set(manifest_assets) == set(EXPECTED_ASSET_HASHES)
    for role, expected_hash in EXPECTED_ASSET_HASHES.items():
        assert manifest_assets[role]["declared_sha256"] == expected_hash
        assert manifest_assets[role]["observed_sha256"] == expected_hash

    source_checkout = validator_root / "models/sources/diffusiondrive_v2"
    assert _git_output(source_checkout, "rev-parse", "HEAD") == DDV2_COMMIT
    assert _git_output(
        source_checkout, "status", "--porcelain", "--untracked-files=no"
    ) == ""
    derivative = source_checkout / DERIVATIVE_RELATIVE_PATH
    assert derivative.is_file()
    derivative_metadata = manifest["source"]["local_derivative"]
    assert Path(derivative_metadata["path"]) == DERIVATIVE_RELATIVE_PATH
    assert _sha256(derivative) == derivative_metadata["sha256"]


@pytest.mark.gpu
def test_cuda_training_preflight() -> None:
    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_properties(0).total_memory > 0
