from __future__ import annotations

from dataclasses import fields, replace

import pytest
import yaml

from morai_ddv2.training.dataset import TrainingConfig


EXPECTED_CONFIG = {
    "seed": 20260806,
    "epochs": 30,
    "batch_size": 8,
    "num_workers": 4,
    "learning_rate": 1.0e-5,
    "weight_decay": 1.0e-4,
    "amp": True,
    "grad_clip_norm": 1.0,
    "trajectory_weight": 12.0,
    "preservation_weight": 1.0e-3,
    "validation_batch_size": 1,
    "validation_run_id": "morai-ddv2-v1-oracle-command-validation-v1",
    "checkpoint_interval_steps": 100,
}


def test_training_yaml_and_dataclass_encode_only_the_exact_defaults(repo_root) -> None:
    path = repo_root / "configs/training/morai-ddv2-encoder-fusion-v1.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw == EXPECTED_CONFIG
    assert tuple(raw) == tuple(EXPECTED_CONFIG)
    assert [field.name for field in fields(TrainingConfig)] == list(EXPECTED_CONFIG)
    assert TrainingConfig(**raw) == TrainingConfig()


@pytest.mark.parametrize(
    "changes",
    (
        {"trajectory_weight": 11.999},
        {"trajectory_weight": 12.001},
        {"preservation_weight": 0.0},
        {"preservation_weight": 1.1e-3},
        {"validation_batch_size": 2},
        {"validation_run_id": "another-validation-run"},
    ),
)
def test_training_config_rejects_fixed_contract_drift(changes) -> None:
    with pytest.raises(ValueError):
        replace(TrainingConfig(), **changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"seed": -1},
        {"epochs": 0},
        {"batch_size": 0},
        {"num_workers": -1},
        {"learning_rate": 0.0},
        {"weight_decay": -1.0},
        {"amp": 1},
        {"grad_clip_norm": 0.0},
        {"checkpoint_interval_steps": 0},
    ),
)
def test_training_config_rejects_invalid_operational_values(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(TrainingConfig(), **changes)
