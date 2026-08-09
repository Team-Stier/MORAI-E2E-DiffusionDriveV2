from __future__ import annotations

import csv
from dataclasses import fields, is_dataclass
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Sequence

from morai_ddv2.atomic import atomic_write_bytes
from morai_ddv2.config import canonical_json_bytes
from morai_ddv2.errors import IntegrityError

from .contracts import (
    EpochMetrics,
    EpochTelemetry,
    ProductionRunAttestation,
    TrainingArtifactHashes,
)
from .validation import ValidationLossSummary


_EPOCH_NAME = re.compile(r"epoch-([0-9]{4})\.json\Z")
_CSV_COLUMNS = (
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
)


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _epoch_payload(telemetry: EpochTelemetry) -> bytes:
    if type(telemetry) is not EpochTelemetry:
        raise TypeError("telemetry must be EpochTelemetry")
    return canonical_json_bytes(_plain(telemetry))


def _publish_idempotent(path: Path, payload: bytes, *, label: str) -> None:
    path = Path(path)
    if path.exists():
        if not path.is_file():
            raise IntegrityError(f"existing {label} is not a file")
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise IntegrityError(f"unable to read existing {label}") from error
        if existing != payload:
            raise IntegrityError(f"existing {label} conflicts with requested bytes")
        return
    atomic_write_bytes(path, payload)


def publish_epoch_telemetry(
    run_dir: Path, telemetry: EpochTelemetry
) -> tuple[Path, str]:
    payload = _epoch_payload(telemetry)
    path = Path(run_dir) / "metrics" / "epochs" / (f"epoch-{telemetry.epoch:04d}.json")
    _publish_idempotent(path, payload, label="epoch telemetry")
    return path, hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _validated_epoch_file(
    path: Path, expected_epoch: int
) -> tuple[bytes, Dict[str, Any]]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IntegrityError(f"invalid epoch telemetry file: {path}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise IntegrityError(f"epoch telemetry is not canonical: {path}")
    try:
        telemetry = EpochTelemetry(
            schema_version=value["schema_version"],
            epoch=value["epoch"],
            global_step=value["global_step"],
            train=EpochMetrics(**value["train"]),
            validation_loss=ValidationLossSummary(**value["validation_loss"]),
            validation=value["validation"],
            validation_report_sha256=value["validation_report_sha256"],
            identities=value["identities"],
            best_epoch=value["best_epoch"],
            best_selected_micro_ade=value["best_selected_micro_ade"],
            is_best=value["is_best"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError(f"epoch telemetry schema is invalid: {path}") from error
    if telemetry.epoch != expected_epoch:
        raise IntegrityError("epoch telemetry filename disagrees with its epoch")
    return payload, value


def _load_sorted_epochs(run_dir: Path) -> tuple[tuple[bytes, Dict[str, Any]], ...]:
    epoch_dir = Path(run_dir) / "metrics" / "epochs"
    if not epoch_dir.is_dir():
        raise IntegrityError("epoch telemetry directory does not exist")
    paths = sorted(epoch_dir.glob("epoch-*.json"), key=lambda path: path.name)
    if not paths:
        raise IntegrityError("no completed epoch telemetry exists")
    result = []
    for expected_epoch, path in enumerate(paths):
        match = _EPOCH_NAME.fullmatch(path.name)
        if match is None or int(match.group(1)) != expected_epoch:
            raise IntegrityError("epoch telemetry files must be consecutive from zero")
        result.append(_validated_epoch_file(path, expected_epoch))
    return tuple(result)


def _csv_payload(values: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for value in values:
        train = value["train"]
        validation_loss = value["validation_loss"]
        validation = value["validation"]
        selected = validation["selected"]
        writer.writerow(
            {
                "epoch": value["epoch"],
                "global_step": value["global_step"],
                "train_total_loss": train["mean_total_loss"],
                "train_trajectory_loss": train["mean_trajectory_loss"],
                "train_preservation_loss": train["mean_preservation_loss"],
                "validation_total_loss": validation_loss["total"],
                "validation_trajectory_loss": validation_loss["trajectory"],
                "validation_preservation_loss": validation_loss["preservation"],
                "selected_ade": selected["ade"],
                "selected_fde": selected["fde"],
                "selected_heading_mae": selected["heading_mae"],
                "min_ade_32": validation["min_ade_32"],
                "min_fde_32": validation["min_fde_32"],
                "is_best": value["is_best"],
            }
        )
    return stream.getvalue().encode("utf-8")


def _plot_payload(values: Sequence[Mapping[str, Any]]) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    epochs = [value["epoch"] for value in values]
    figure = Figure(figsize=(15.0, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3, squeeze=False)[0]

    axes[0].plot(
        epochs,
        [value["train"]["mean_total_loss"] for value in values],
        marker="o",
        label="train total",
    )
    axes[0].plot(
        epochs,
        [value["validation_loss"]["total"] for value in values],
        marker="o",
        label="validation total",
    )
    axes[0].set_title("Train / validation total loss")
    axes[0].set_ylabel("Loss")

    axes[1].plot(
        epochs,
        [value["validation"]["selected"]["ade"] for value in values],
        marker="o",
        label="selected ADE",
    )
    axes[1].plot(
        epochs,
        [value["validation"]["selected"]["fde"] for value in values],
        marker="o",
        label="selected FDE",
    )
    axes[1].set_title("Selected ADE / FDE")
    axes[1].set_ylabel("Meters")

    axes[2].plot(
        epochs,
        [value["validation"]["min_ade_32"] for value in values],
        marker="o",
        label="minADE@32",
    )
    axes[2].plot(
        epochs,
        [value["validation"]["min_fde_32"] for value in values],
        marker="o",
        label="minFDE@32",
    )
    axes[2].set_title("minADE@32 / minFDE@32")
    axes[2].set_ylabel("Meters")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
        axis.legend()
    stream = io.BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=100,
        metadata={"Software": "morai-ddv2"},
    )
    return stream.getvalue()


def rebuild_training_artifacts(run_dir: Path) -> TrainingArtifactHashes:
    epochs = _load_sorted_epochs(run_dir)
    values = tuple(value for _, value in epochs)
    history = b"".join(payload + b"\n" for payload, _ in epochs)
    csv_payload = _csv_payload(values)
    plot_payload = _plot_payload(values)
    run_dir = Path(run_dir)
    history_path = run_dir / "metrics" / "history.jsonl"
    csv_path = run_dir / "metrics" / "epoch_metrics.csv"
    plot_path = run_dir / "plots" / "training-curves.png"
    atomic_write_bytes(history_path, history)
    atomic_write_bytes(csv_path, csv_payload)
    atomic_write_bytes(plot_path, plot_payload)
    return TrainingArtifactHashes(
        metrics_history_sha256=hashlib.sha256(history).hexdigest(),
        epoch_metrics_csv_sha256=hashlib.sha256(csv_payload).hexdigest(),
        training_curves_sha256=hashlib.sha256(plot_payload).hexdigest(),
    )


def publish_production_run_attestation(
    run_dir: Path, attestation: ProductionRunAttestation
) -> Path:
    if type(attestation) is not ProductionRunAttestation:
        raise TypeError("attestation must be ProductionRunAttestation")
    path = Path(run_dir) / "run-complete.json"
    _publish_idempotent(
        path,
        canonical_json_bytes(_plain(attestation)),
        label="production run attestation",
    )
    return path


__all__ = (
    "publish_epoch_telemetry",
    "publish_production_run_attestation",
    "rebuild_training_artifacts",
)
