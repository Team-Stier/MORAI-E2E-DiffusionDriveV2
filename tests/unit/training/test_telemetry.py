from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import pytest
import rfc8785

from morai_ddv2.errors import IntegrityError
from morai_ddv2.training.contracts import (
    EpochMetrics,
    EpochTelemetry,
    ProductionRunAttestation,
    TrainingArtifactHashes,
)
from morai_ddv2.training.telemetry import (
    publish_epoch_telemetry,
    publish_production_run_attestation,
    rebuild_training_artifacts,
)
from morai_ddv2.training.validation import ValidationLossSummary


def _epoch_telemetry(epoch: int = 0) -> EpochTelemetry:
    train = EpochMetrics(
        epoch=epoch,
        global_step=(epoch + 1) * 3,
        completed_batches=3,
        completed_samples=5,
        mean_total_loss=4.0 - epoch,
        mean_trajectory_loss=0.3 - epoch * 0.01,
        mean_preservation_loss=0.4 + epoch * 0.01,
    )
    return EpochTelemetry(
        schema_version=1,
        epoch=epoch,
        global_step=train.global_step,
        train=train,
        validation_loss=ValidationLossSummary(
            count=2,
            total=3.0 - epoch,
            trajectory=0.2 - epoch * 0.01,
            preservation=0.3 + epoch * 0.01,
        ),
        validation={
            "selected": {
                "count": 2,
                "ade": 1.5 - epoch * 0.1,
                "fde": 2.5 - epoch * 0.1,
                "heading_mae": 0.25 - epoch * 0.01,
            },
            "bag_macro_ade": 1.75 - epoch * 0.1,
            "min_ade_32": 0.75 - epoch * 0.1,
            "min_fde_32": 1.25 - epoch * 0.1,
        },
        validation_report_sha256="a" * 64,
        identities={
            "training_manifest_sha256": "b" * 64,
            "resolved_domain_sha256": "c" * 64,
        },
        best_epoch=epoch,
        best_selected_micro_ade=1.5 - epoch * 0.1,
        is_best=True,
    )


def test_publish_epoch_telemetry_is_canonical_atomic_and_idempotent(
    tmp_path: Path,
) -> None:
    telemetry = _epoch_telemetry()

    path, digest = publish_epoch_telemetry(tmp_path, telemetry)

    assert path == tmp_path / "metrics/epochs/epoch-0000.json"
    payload = path.read_bytes()
    assert payload == rfc8785.dumps(json.loads(payload))
    assert digest == hashlib.sha256(payload).hexdigest()
    assert not path.with_name(path.name + ".partial").exists()
    assert publish_epoch_telemetry(tmp_path, telemetry) == (path, digest)
    assert path.read_bytes() == payload


def test_publish_epoch_telemetry_rejects_conflicting_existing_epoch(
    tmp_path: Path,
) -> None:
    telemetry = _epoch_telemetry()
    path, _ = publish_epoch_telemetry(tmp_path, telemetry)
    original = path.read_bytes()

    with pytest.raises(IntegrityError, match="conflicts"):
        publish_epoch_telemetry(
            tmp_path,
            replace(telemetry, validation_report_sha256="d" * 64),
        )

    assert path.read_bytes() == original


@pytest.mark.parametrize("field", ["records", "seed", "timestamp_ns", "delay_ns"])
def test_epoch_telemetry_rejects_per_sample_seed_and_nanosecond_fields(
    field: str,
) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        replace(_epoch_telemetry(), validation={field: 1})


def test_epoch_telemetry_rejects_nonfinite_train_and_empty_validation_loss() -> None:
    telemetry = _epoch_telemetry()

    with pytest.raises(ValueError, match="train.*finite"):
        replace(
            telemetry,
            train=replace(telemetry.train, mean_total_loss=float("nan")),
        )
    with pytest.raises(ValueError, match="validation loss.*positive"):
        replace(
            telemetry,
            validation_loss=ValidationLossSummary(0, None, None, None),
        )


def test_rebuild_training_artifacts_replaces_sorted_history_csv_and_png(
    tmp_path: Path,
) -> None:
    epoch_one_path, _ = publish_epoch_telemetry(tmp_path, _epoch_telemetry(1))
    epoch_zero_path, _ = publish_epoch_telemetry(tmp_path, _epoch_telemetry(0))

    hashes = rebuild_training_artifacts(tmp_path)

    assert isinstance(hashes, TrainingArtifactHashes)
    history_path = tmp_path / "metrics/history.jsonl"
    csv_path = tmp_path / "metrics/epoch_metrics.csv"
    plot_path = tmp_path / "plots/training-curves.png"
    expected_history = (
        epoch_zero_path.read_bytes() + b"\n" + epoch_one_path.read_bytes() + b"\n"
    )
    assert history_path.read_bytes() == expected_history
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    assert list(rows[0]) == [
        "epoch",
        "global_step",
        "train_total_loss",
        "train_trajectory_loss",
        "train_preservation_loss",
        "validation_total_loss",
        "validation_trajectory_loss",
        "validation_preservation_loss",
        "selected_ade",
        "selected_fde",
        "selected_heading_mae",
        "min_ade_32",
        "min_fde_32",
        "is_best",
    ]
    assert [row["epoch"] for row in rows] == ["0", "1"]
    assert rows[1]["selected_ade"] == "1.4"
    png = plot_path.read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1_000
    assert hashes.metrics_history_sha256 == hashlib.sha256(expected_history).hexdigest()
    assert (
        hashes.epoch_metrics_csv_sha256
        == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    )
    assert hashes.training_curves_sha256 == hashlib.sha256(png).hexdigest()
    assert not tuple(tmp_path.rglob("*.partial"))

    assert rebuild_training_artifacts(tmp_path) == hashes
    assert plot_path.read_bytes() == png


def test_publish_production_attestation_is_canonical_atomic_and_idempotent(
    tmp_path: Path,
) -> None:
    attestation = ProductionRunAttestation(
        training_config_sha256="1" * 64,
        training_manifest_sha256="2" * 64,
        resolved_domain_sha256="3" * 64,
        cache_schema_version=2,
        clock_policy_version=1,
        base_checkpoint_sha256="4" * 64,
        completed_epochs=2,
        best_checkpoint_sha256="5" * 64,
        best_micro_ade=1.25,
        validation_run_id="validation-v1",
        epoch_telemetry_count=2,
        metrics_history_sha256="6" * 64,
        epoch_metrics_csv_sha256="7" * 64,
        training_curves_sha256="8" * 64,
    )

    path = publish_production_run_attestation(tmp_path, attestation)

    assert path == tmp_path / "run-complete.json"
    payload = path.read_bytes()
    assert payload == rfc8785.dumps(json.loads(payload))
    assert publish_production_run_attestation(tmp_path, attestation) == path
    assert path.read_bytes() == payload
