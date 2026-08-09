from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

import morai_ddv2.cli.train as train_cli
from morai_ddv2.errors import IntegrityError
from morai_ddv2.training.dataset import TrainingConfig


_REQUIRED = [
    "--manifest",
    "manifest.jsonl",
    "--resolved-domain",
    "resolved-domain.json",
    "--base-profile",
    "profile.json",
    "--config",
    "training.yaml",
    "--run-dir",
    "run",
]


@pytest.mark.parametrize(
    "flag", ["--max-epochs", "--max-train-batches", "--max-validation-samples"]
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_train_parser_rejects_nonpositive_smoke_bounds(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        train_cli._parser().parse_args(_REQUIRED + [flag, value])


def test_resolved_run_config_records_bounds_without_mutating_training_config() -> None:
    config = TrainingConfig()
    identities = {
        "training_manifest_sha256": "1" * 64,
        "resolved_domain_sha256": "2" * 64,
        "base_checkpoint_sha256": "3" * 64,
        "training_source_commit": "4" * 40,
        "preprocessing_source_commit": "5" * 40,
    }

    unbounded = train_cli.build_resolved_run_config(
        config=config,
        identities=identities,
        resume_checkpoint_sha256=None,
        max_epochs=None,
        max_train_batches=None,
        max_validation_samples=None,
    )
    bounded = train_cli.build_resolved_run_config(
        config=config,
        identities=identities,
        resume_checkpoint_sha256="6" * 64,
        max_epochs=2,
        max_train_batches=3,
        max_validation_samples=4,
    )

    assert unbounded["training"] == asdict(TrainingConfig())
    assert unbounded["smoke_bounds"] == {
        "max_epochs": None,
        "max_train_batches": None,
        "max_validation_samples": None,
    }
    assert bounded["training"]["epochs"] == 30
    assert bounded["smoke_bounds"] == {
        "max_epochs": 2,
        "max_train_batches": 3,
        "max_validation_samples": 4,
    }
    assert "resume_checkpoint_sha256" not in bounded
    assert config == TrainingConfig()


@pytest.mark.parametrize("name", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_runtime_preflight_requires_each_offline_variable_exactly_one(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv(name, "true")

    with pytest.raises(IntegrityError, match=name):
        train_cli.assert_runtime_contract(require_cuda=False)


def test_runtime_preflight_rejects_wrong_python_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_cli.sys, "version_info", (3, 9, 24))

    with pytest.raises(IntegrityError, match="3.9.25"):
        train_cli.assert_runtime_contract(require_cuda=False)


def test_training_runtime_preflight_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(train_cli.torch.cuda, "is_available", lambda: False)

    with pytest.raises(IntegrityError, match="CUDA"):
        train_cli.assert_runtime_contract(require_cuda=True)


@pytest.mark.parametrize(
    "reason",
    [
        "schema",
        "incomplete-schema-2",
        "resolved-domain",
        "front-identity",
        "raw-tf",
        "ego-epoch",
        "diagnostics",
        "mixed-axis",
        "dirty-cache",
        "source-commit",
        "assets",
        "python-torch",
        "offline",
        "cuda",
    ],
)
def test_train_preflight_failure_happens_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    constructed = []

    def fail_preflight(_arguments):
        raise IntegrityError(reason)

    monkeypatch.setattr(train_cli, "preflight_training", fail_preflight)
    monkeypatch.setattr(
        train_cli,
        "load_pdm_free_sel_network",
        lambda *args, **kwargs: constructed.append((args, kwargs)),
    )

    with pytest.raises(IntegrityError, match=reason):
        train_cli.main(_REQUIRED)

    assert constructed == []


def test_script_train_wrapper_exists(repo_root: Path) -> None:
    wrapper = (repo_root / "scripts/train.py").read_text(encoding="utf-8")
    assert "from morai_ddv2.cli.train import main" in wrapper
