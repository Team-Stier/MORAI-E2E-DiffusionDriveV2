from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import morai_ddv2.cli.evaluate as evaluate_cli
from morai_ddv2.errors import IntegrityError
from morai_ddv2.training.metrics import PredictionRecord, compute_openloop_metrics


_REQUIRED = [
    "offline",
    "--manifest",
    "manifest.jsonl",
    "--resolved-domain",
    "resolved-domain.json",
    "--base-profile",
    "profile.json",
    "--training-checkpoint",
    "best.pt",
    "--report",
    "report.json",
]


def _report(identities):
    zeros = np.zeros((8, 3), dtype=np.float64)
    return compute_openloop_metrics(
        (
            PredictionRecord(
                sample_id="A",
                bag_sha256="f" * 64,
                command_name="LEFT",
                timestamp_ns=1,
                seed=1,
                prediction=zeros,
                target=zeros,
                candidate_source="coarse_top32_post_bezier",
                candidate_count=32,
                min_ade_32=0.0,
                min_fde_32=0.0,
                argmin_ade_rank=0,
                argmin_fde_rank=0,
            ),
        ),
        identities=identities,
    )


def test_evaluate_parser_rejects_unknown_sample_order() -> None:
    with pytest.raises(SystemExit):
        evaluate_cli._parser().parse_args(_REQUIRED + ["--sample-order", "random"])


def test_reverse_changes_traversal_only_and_report_bytes_remain_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = (
        {"metadata": {"sample_id": "B"}},
        {"metadata": {"sample_id": "A"}},
    )
    traversals = []

    def validate(_model, ordered, **kwargs):
        traversals.append(tuple(item["metadata"]["sample_id"] for item in ordered))
        return _report(kwargs["identities"])

    monkeypatch.setattr(evaluate_cli, "validate_selector_b1", validate)
    manifest_report = tmp_path / "manifest.json"
    reverse_report = tmp_path / "reverse.json"

    evaluate_cli.evaluate_offline(
        model=object(),
        samples=samples,
        report_path=manifest_report,
        identities={"manifest": "1" * 64},
        run_id="validation-v1",
        sample_order="manifest",
    )
    evaluate_cli.evaluate_offline(
        model=object(),
        samples=samples,
        report_path=reverse_report,
        identities={"manifest": "1" * 64},
        run_id="validation-v1",
        sample_order="reverse",
    )

    assert traversals == [("B", "A"), ("A", "B")]
    assert manifest_report.read_bytes() == reverse_report.read_bytes()
    assert b'"sample_id":"A"' in manifest_report.read_bytes()


def test_evaluate_preflight_failure_happens_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = []

    def fail_preflight(_arguments):
        raise IntegrityError("cache preflight")

    monkeypatch.setattr(evaluate_cli, "preflight_evaluation", fail_preflight)
    monkeypatch.setattr(
        evaluate_cli,
        "load_pdm_free_sel_network",
        lambda *args, **kwargs: constructed.append((args, kwargs)),
    )

    with pytest.raises(IntegrityError, match="cache preflight"):
        evaluate_cli.main(_REQUIRED)

    assert constructed == []


def test_script_evaluate_wrapper_exists(repo_root: Path) -> None:
    wrapper = (repo_root / "scripts/evaluate.py").read_text(encoding="utf-8")
    assert "from morai_ddv2.cli.evaluate import main" in wrapper
