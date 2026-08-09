from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from morai_ddv2.training.metrics import (
    COMMAND_NAMES,
    VALIDATION_RUN_ID,
    MetricSummary,
    PredictionRecord,
    bezier_headings,
    compute_best_of_32_metrics,
    compute_openloop_metrics,
    reset_torch_seed,
    selector_seed,
    validation_report_json_bytes,
)


TIMESTAMP_NS = 1_722_500_000_123_456_789
EXPECTED_DIGEST = "a9e2a2b3aef37e1993e5de61d7dc84581a4acd8cf938d0296b43c8fb54aec4b0"
EXPECTED_SEED = 12_241_525_629_764_468_249


def _record(
    sample_id: str,
    bag_sha256: str,
    command_name: str,
    xy_error: float,
    heading_prediction: float,
    heading_target: float,
    *,
    min_ade_32: float,
    min_fde_32: float,
) -> PredictionRecord:
    prediction = np.zeros((8, 3), dtype=np.float32)
    prediction[:, 0] = np.float32(xy_error)
    prediction[:, 2] = np.float32(heading_prediction)
    target = np.zeros((8, 3), dtype=np.float32)
    target[:, 2] = np.float32(heading_target)
    return PredictionRecord(
        sample_id=sample_id,
        bag_sha256=bag_sha256,
        command_name=command_name,
        timestamp_ns=TIMESTAMP_NS + len(sample_id),
        seed=selector_seed(TIMESTAMP_NS + len(sample_id)),
        prediction=prediction,
        target=target,
        candidate_source="coarse_top32_post_bezier",
        candidate_count=32,
        min_ade_32=min_ade_32,
        min_fde_32=min_fde_32,
        argmin_ade_rank=0,
        argmin_fde_rank=2,
    )


def test_selector_seed_matches_fixed_sha256_vector() -> None:
    payload = f"diffusiondrive_v2:{TIMESTAMP_NS}:{VALIDATION_RUN_ID}".encode()

    assert hashlib.sha256(payload).hexdigest() == EXPECTED_DIGEST
    assert selector_seed(TIMESTAMP_NS) == EXPECTED_SEED


def test_reset_torch_seed_resets_cpu_and_every_cuda_generator(monkeypatch) -> None:
    import torch

    cpu_seeds = []
    cuda_seeds = []
    monkeypatch.setattr(torch, "manual_seed", cpu_seeds.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seeds.append)

    reset_torch_seed(EXPECTED_SEED)

    assert cpu_seeds == [EXPECTED_SEED]
    assert cuda_seeds == [EXPECTED_SEED]


def test_metrics_module_import_does_not_import_torch(repo_root) -> None:
    code = """
import json
import sys
import morai_ddv2.training.metrics
print(json.dumps({"torch_loaded": "torch" in sys.modules}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-B",
            "-c",
            code,
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"torch_loaded": False}


def test_training_package_preserves_lazy_public_api(repo_root) -> None:
    code = """
import json
import sys
import morai_ddv2.training as training
before = "torch" in sys.modules
manifest_type = training.FreezeManifest
after_contract = "torch" in sys.modules
configure = training.configure_training_boundary
after_freeze = "torch" in sys.modules
print(json.dumps({
    "all": training.__all__,
    "import_helper_public": hasattr(training, "import_module"),
    "before": before,
    "after_contract": after_contract,
    "after_freeze": after_freeze,
    "manifest_module": manifest_type.__module__,
    "configure_module": configure.__module__,
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)

    assert observed == {
        "all": [
            "TRAINABLE_PREFIXES",
            "FreezeManifest",
            "TrainingBoundaryError",
            "assert_training_boundary",
            "build_optimizer",
            "configure_training_boundary",
            "extract_mutable_state",
        ],
        "import_helper_public": False,
        "before": False,
        "after_contract": False,
        "after_freeze": True,
        "manifest_module": "morai_ddv2.training.contracts",
        "configure_module": "morai_ddv2.training.freeze",
    }


def test_degree_eight_bezier_headings_use_derivative_at_k_over_eight() -> None:
    selected_xy = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 0.5],
            [3.0, 1.5],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 7.5],
            [7.0, 10.5],
            [8.0, 14.0],
        ],
        dtype=np.float64,
    )
    control = np.concatenate((np.zeros((1, 2)), selected_xy), axis=0)
    expected = []
    for sample_index in range(1, 9):
        t = sample_index / 8.0
        derivative = np.zeros(2, dtype=np.float64)
        for control_index in range(8):
            weight = (
                8
                * math.comb(7, control_index)
                * t**control_index
                * (1.0 - t) ** (7 - control_index)
            )
            derivative += weight * (control[control_index + 1] - control[control_index])
        expected.append(math.atan2(derivative[1], derivative[0]))

    actual = bezier_headings(selected_xy)

    assert actual.shape == (8,)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1.0e-14)


def test_best_of_32_uses_independent_argmins_keeps_duplicates_and_ignores_heading() -> (
    None
):
    target = np.zeros((8, 3), dtype=np.float64)
    candidates = np.full((32, 8, 3), 10.0, dtype=np.float64)
    candidates[..., 2] = np.linspace(-1000.0, 1000.0, 32)[:, None]
    candidates[0, :, :2] = 0.0
    candidates[0, -1, 0] = 8.0
    candidates[1] = candidates[0]
    candidates[2, :, :2] = 0.0
    candidates[2, :7, 0] = 2.0
    candidates[2, :, 2] = 999999.0

    result = compute_best_of_32_metrics(candidates, target)

    assert result.min_ade_32 == 1.0
    assert result.min_fde_32 == 0.0
    assert result.argmin_ade_rank == 0
    assert result.argmin_fde_rank == 2
    assert 0 <= result.argmin_ade_rank < 32
    assert 0 <= result.argmin_fde_rank < 32


def test_openloop_metrics_are_micro_circular_bag_macro_and_per_command() -> None:
    records = (
        _record(
            "sample-c",
            "b" * 64,
            "LEFT",
            5.0,
            0.0,
            math.pi,
            min_ade_32=0.5,
            min_fde_32=0.25,
        ),
        _record(
            "sample-a",
            "a" * 64,
            "LEFT",
            1.0,
            math.pi - 0.1,
            -math.pi + 0.1,
            min_ade_32=1.5,
            min_fde_32=1.25,
        ),
        _record(
            "sample-b",
            "a" * 64,
            "RIGHT",
            3.0,
            0.0,
            0.0,
            min_ade_32=2.5,
            min_fde_32=2.25,
        ),
    )

    report = compute_openloop_metrics(records)

    assert report.micro.count == 3
    assert report.micro.ade == 3.0
    assert report.micro.fde == 3.0
    assert report.micro.heading_mae == pytest.approx((0.2 + math.pi) / 3.0)
    assert report.bag_macro_ade == 3.5
    assert report.oracle_min_ade_32 == 1.5
    assert report.oracle_min_fde_32 == 1.25
    assert report.oracle_candidate_source == "coarse_top32_post_bezier"
    assert report.oracle_candidate_count == 32
    assert tuple(report.per_command) == COMMAND_NAMES
    assert report.per_command["LEFT"].count == 2
    assert report.per_command["LEFT"].ade == 3.0
    assert report.per_command["LEFT"].heading_mae == pytest.approx(
        (0.2 + math.pi) / 2.0
    )
    assert report.per_command["RIGHT"] == MetricSummary(1, 3.0, 3.0, 0.0)
    assert report.per_command["STRAIGHT"] == MetricSummary(0, None, None, None)
    assert report.per_command["UNKNOWN"] == MetricSummary(0, None, None, None)
    assert tuple(record.sample_id for record in report.records) == (
        "sample-a",
        "sample-b",
        "sample-c",
    )


def test_empty_metrics_are_null_never_nan() -> None:
    report = compute_openloop_metrics(())

    assert report.micro == MetricSummary(0, None, None, None)
    assert report.oracle_min_ade_32 is None
    assert report.oracle_min_fde_32 is None
    assert report.bag_macro_ade is None
    assert all(
        summary == MetricSummary(0, None, None, None)
        for summary in report.per_command.values()
    )
    assert b"NaN" not in validation_report_json_bytes(report)


def test_report_json_is_strict_sorted_and_order_invariant() -> None:
    records = (
        _record(
            "z-sample",
            "b" * 64,
            "UNKNOWN",
            2.0,
            0.0,
            0.0,
            min_ade_32=1.0,
            min_fde_32=2.0,
        ),
        _record(
            "a-sample",
            "a" * 64,
            "STRAIGHT",
            1.0,
            0.0,
            0.0,
            min_ade_32=3.0,
            min_fde_32=4.0,
        ),
    )
    expected_keys = {
        "domain_id",
        "command_source",
        "run_id",
        "micro",
        "oracle_min_ade_32",
        "oracle_min_fde_32",
        "oracle_candidate_source",
        "oracle_candidate_count",
        "bag_macro_ade",
        "per_command",
        "records",
        "drop_statistics",
        "sync_statistics",
        "identities",
    }

    forward = validation_report_json_bytes(compute_openloop_metrics(records))
    reverse = validation_report_json_bytes(
        compute_openloop_metrics(tuple(reversed(records)))
    )
    decoded = json.loads(forward)

    assert forward == reverse
    assert forward == json.dumps(
        decoded,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert set(decoded) == expected_keys
    assert [record["sample_id"] for record in decoded["records"]] == [
        "a-sample",
        "z-sample",
    ]
    assert isinstance(decoded["records"][0]["timestamp_ns"], int)
    assert isinstance(decoded["records"][0]["prediction"], list)
    assert "generated_at" not in decoded
    assert "traversal" not in decoded
