from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import morai_ddv2.training.validation as validation_module
from morai_ddv2.training.metrics import (
    VALIDATION_RUN_ID,
    selector_seed,
    validation_report_json_bytes,
)
from morai_ddv2.training.validation import (
    ValidationLossSummary,
    capture_coarse_top32_post_bezier,
    validate_selector_b1,
    validate_supervised_loss_b1,
    validation_loss_seed,
)


class _SelectorHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bezier_calls = 0
        self.select_calls = 0
        self.last_bezier_result = None
        self.last_select_result = None

    def bezier_xyyaw(self, xy8: torch.Tensor) -> torch.Tensor:
        self.bezier_calls += 1
        heading = torch.atan2(xy8[..., 1], xy8[..., 0]).unsqueeze(-1)
        result = torch.cat((xy8, heading), dim=-1)
        self.last_bezier_result = result
        return result

    def _select_topk(
        self,
        final_coarse_reward: torch.Tensor,
        topk: int,
        traj_feature: torch.Tensor,
        noisy_traj_points_xy: torch.Tensor,
        sub_rewards_group,
    ):
        self.select_calls += 1
        topk_val, topk_idx = torch.topk(
            final_coarse_reward, topk, dim=-1, largest=True, sorted=True
        )
        feature_index = topk_idx.unsqueeze(-1).expand(-1, -1, traj_feature.shape[-1])
        point_index = topk_idx[:, :, None, None].expand(-1, -1, 8, 2)
        result = (
            torch.gather(traj_feature, 1, feature_index),
            torch.gather(noisy_traj_points_xy, 1, point_index),
            None,
            topk_idx,
            topk_val,
        )
        self.last_select_result = result
        return result


class _LookupCountingHead(_SelectorHead):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(
            self, "lookup_counts", {"bezier_xyyaw": 0, "_select_topk": 0}
        )

    def __getattribute__(self, name: str):
        if name in ("bezier_xyyaw", "_select_topk"):
            counts = object.__getattribute__(self, "lookup_counts")
            counts[name] += 1
        return super().__getattribute__(name)


class _SelectorModel(nn.Module):
    def __init__(self, *, select_outside_shortlist: bool = False) -> None:
        super().__init__()
        self._trajectory_head = _SelectorHead()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.select_outside_shortlist = select_outside_shortlist
        self.forward_calls = 0
        self.grad_enabled = []
        self.forward_seeds = []

    def forward(self, features):
        self.forward_calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        self.forward_seeds.append(torch.initial_seed())
        token = features["token"].reshape(1, 1, 1, 1)
        xy = torch.rand(1, 800, 8, 2, device=token.device) + token
        pool = self._trajectory_head.bezier_xyyaw(xy)
        rewards = torch.arange(800, dtype=xy.dtype, device=xy.device).unsqueeze(0)
        features_for_score = torch.zeros(1, 800, 4, device=xy.device)
        selection = self._trajectory_head._select_topk(
            rewards,
            32,
            features_for_score,
            pool[..., :2],
            None,
        )
        selected_index = 0 if self.select_outside_shortlist else selection[3][0, -1]
        selected = pool[:, selected_index]
        result = {"trajectory": selected}
        self.last_public_result = result
        self.last_bezier_object = pool
        self.last_select_object = selection
        return result


class _LossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.boundary_child = nn.Sequential(nn.Identity())


def _install_bound_callables_as_exact_instance_objects(head: _SelectorHead):
    original_bezier = head.bezier_xyyaw
    original_select = head._select_topk
    head.bezier_xyyaw = original_bezier
    head._select_topk = original_select
    return original_bezier, original_select


def _selector_sample(sample_id: str, timestamp_ns: int, token: float):
    return {
        "features": {"token": torch.tensor([[token]], dtype=torch.float32)},
        "trajectory": torch.zeros(1, 8, 3, dtype=torch.float32),
        "metadata": {
            "sample_id": sample_id,
            "source_bag_sha256": (sample_id[0].lower() * 64),
            "command_name": "LEFT" if token < 2.0 else "RIGHT",
            "timestamp_ns": timestamp_ns,
        },
    }


def _loss_sample(sample_id: str, timestamp_ns: int, token: float):
    return {
        "features": {"token": torch.tensor([[token]], dtype=torch.float32)},
        "trajectory": torch.zeros(1, 8, 3, dtype=torch.float32),
        "metadata": {"sample_id": sample_id, "timestamp_ns": timestamp_ns},
    }


def _module_modes(model: nn.Module):
    return tuple((module, module.training) for module in model.modules())


def test_capture_calls_originals_once_returns_exact_objects_and_detaches_clones() -> (
    None
):
    model = _SelectorModel()
    head = model._trajectory_head
    original_bezier, original_select = (
        _install_bound_callables_as_exact_instance_objects(head)
    )
    xy = torch.arange(1 * 800 * 8 * 2, dtype=torch.float32).view(1, 800, 8, 2)
    rewards = torch.arange(800, dtype=torch.float32).unsqueeze(0)
    features = torch.zeros(1, 800, 4)

    with capture_coarse_top32_post_bezier(model) as capture:
        bezier_result = head.bezier_xyyaw(xy)
        select_result = head._select_topk(
            rewards, 32, features, bezier_result[..., :2], None
        )
        unchanged_pool = bezier_result.detach().clone()
        unchanged_indices = select_result[3].detach().clone()
        assert bezier_result is head.last_bezier_result
        assert select_result is head.last_select_result

    assert head.bezier_calls == 1
    assert head.select_calls == 1
    assert head.bezier_xyyaw is original_bezier
    assert head._select_topk is original_select
    assert capture.post_bezier_pool is not bezier_result
    assert capture.topk_indices is not select_result[3]
    assert not capture.post_bezier_pool.requires_grad
    assert not capture.topk_indices.requires_grad
    assert capture.post_bezier_pool.shape == (1, 800, 8, 3)
    assert capture.topk_indices.shape == (1, 32)
    assert capture.candidates.shape == (1, 32, 8, 3)
    torch.testing.assert_close(capture.post_bezier_pool, unchanged_pool, rtol=0, atol=0)
    torch.testing.assert_close(capture.topk_indices, unchanged_indices, rtol=0, atol=0)
    expected = unchanged_pool[:, unchanged_indices[0]]
    torch.testing.assert_close(capture.candidates, expected, rtol=0, atol=0)

    bezier_result.add_(10000.0)
    select_result[3].zero_()
    torch.testing.assert_close(capture.post_bezier_pool, unchanged_pool, rtol=0, atol=0)
    torch.testing.assert_close(capture.topk_indices, unchanged_indices, rtol=0, atol=0)


def test_capture_reads_each_original_callable_exactly_once() -> None:
    model = _SelectorModel()
    head = _LookupCountingHead()
    model._trajectory_head = head

    with capture_coarse_top32_post_bezier(model):
        pass

    assert head.lookup_counts == {"bezier_xyyaw": 1, "_select_topk": 1}


def test_capture_restores_exact_callables_on_exception() -> None:
    model = _SelectorModel()
    head = model._trajectory_head
    original_bezier, original_select = (
        _install_bound_callables_as_exact_instance_objects(head)
    )

    with pytest.raises(RuntimeError, match="sentinel"):
        with capture_coarse_top32_post_bezier(model):
            raise RuntimeError("sentinel")

    assert head.bezier_xyyaw is original_bezier
    assert head._select_topk is original_select


@pytest.mark.parametrize("raise_inside", [False, True])
def test_capture_preserves_class_descriptor_lookup_topology(raise_inside: bool) -> None:
    model = _SelectorModel()
    head = model._trajectory_head
    names = ("bezier_xyyaw", "_select_topk")
    descriptors = {name: vars(type(head))[name] for name in names}

    assert all(name not in vars(head) for name in names)

    if raise_inside:
        with pytest.raises(RuntimeError, match="sentinel"):
            with capture_coarse_top32_post_bezier(model):
                raise RuntimeError("sentinel")
    else:
        with capture_coarse_top32_post_bezier(model):
            pass

    assert all(name not in vars(head) for name in names)
    assert head.bezier_xyyaw.__func__ is descriptors["bezier_xyyaw"]
    assert head._select_topk.__func__ is descriptors["_select_topk"]


def test_capture_does_not_change_public_output_or_expose_candidates() -> None:
    model = _SelectorModel()
    features = {"token": torch.tensor([[1.0]])}
    torch.manual_seed(314159)
    plain = model(features)
    torch.manual_seed(314159)

    with capture_coarse_top32_post_bezier(model) as capture:
        captured = model(features)

    assert captured is model.last_public_result
    assert tuple(captured) == ("trajectory",)
    assert torch.equal(plain["trajectory"], captured["trajectory"])
    assert any(
        torch.equal(captured["trajectory"][0], row) for row in capture.candidates[0]
    )


def test_selector_validation_is_seeded_b1_same_forward_and_order_invariant() -> None:
    first_timestamp = 1_722_500_000_123_456_789
    samples = (
        _selector_sample("A-sample", first_timestamp, 1.0),
        _selector_sample("B-sample", first_timestamp + 10, 3.0),
    )
    context = {
        "drop_statistics": {"CAMERA_SYNC_MISS": 2, "NONFINITE_VALUE": 0},
        "sync_statistics": {
            "side_header_offset_abs_max_ns": 30_000_000,
            "sensor_receipt_offset_abs_max_ns": 100_000_000,
            "raw_tf_anchor_receipt_offset_abs_max_ns": 50_000_000,
            "tf_epoch_count": 2,
            "ego_epoch_count": 3,
        },
        "identities": {
            "cache_dataset_schema_sha256": "c" * 64,
            "manifest_provenance_sha256": "d" * 64,
        },
    }
    model = _SelectorModel()
    model.eval()
    model._trajectory_head.train()
    modes_before = _module_modes(model)

    forward = validate_selector_b1(model, samples, **context)
    modes_after_forward = _module_modes(model)
    reverse = validate_selector_b1(model, reversed(samples), **context)

    assert model.forward_calls == 4
    assert model.grad_enabled == [False, False, False, False]
    assert model.forward_seeds == [
        selector_seed(first_timestamp),
        selector_seed(first_timestamp + 10),
        selector_seed(first_timestamp + 10),
        selector_seed(first_timestamp),
    ]
    assert modes_after_forward == modes_before
    assert _module_modes(model) == modes_before
    assert tuple(record.sample_id for record in forward.records) == (
        "A-sample",
        "B-sample",
    )
    assert all(record.prediction.shape == (8, 3) for record in forward.records)
    assert [record.seed for record in forward.records] == [
        selector_seed(first_timestamp),
        selector_seed(first_timestamp + 10),
    ]
    assert all(record.candidate_count == 32 for record in forward.records)
    assert all(
        record.candidate_source == "coarse_top32_post_bezier"
        for record in forward.records
    )
    assert forward.drop_statistics == context["drop_statistics"]
    assert forward.sync_statistics == context["sync_statistics"]
    assert forward.identities == context["identities"]
    assert validation_report_json_bytes(forward) == validation_report_json_bytes(
        reverse
    )
    for left, right in zip(forward.records, reverse.records):
        assert left.sample_id == right.sample_id
        assert left.seed == right.seed
        assert np.array_equal(left.prediction, right.prediction)


def test_selector_validation_rejects_selected_trajectory_outside_same_shortlist() -> (
    None
):
    model = _SelectorModel(select_outside_shortlist=True)
    model.train()
    model._trajectory_head.eval()
    modes_before = _module_modes(model)

    with pytest.raises(RuntimeError, match="shortlist"):
        validate_selector_b1(model, (_selector_sample("A-sample", 10, 1.0),))

    assert model.forward_calls == 1
    assert _module_modes(model) == modes_before


def test_validation_loss_seed_matches_supervised_namespace() -> None:
    timestamp_ns = 1_722_500_000_123_456_789
    payload = (
        f"diffusiondrive_v2_supervised_loss:{timestamp_ns}:{VALIDATION_RUN_ID}"
    ).encode()
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    assert validation_loss_seed(timestamp_ns) == expected


def test_supervised_loss_validation_is_b1_no_grad_weighted_and_order_invariant(
    monkeypatch,
) -> None:
    samples = (
        _loss_sample("A-sample", 100, 1.0),
        _loss_sample("B-sample", 200, 3.0),
    )
    model = _LossModel()
    model.train()
    model.boundary_child.eval()
    modes_before = _module_modes(model)
    reference = object()
    manifest = object()
    events = []
    compose_calls = []

    def reset(seed: int) -> None:
        events.append(("seed", seed))
        torch.manual_seed(seed)

    def supervised(current_model, features, target):
        events.append(
            (
                "adapter",
                torch.initial_seed(),
                torch.is_grad_enabled(),
                tuple(module.training for module in current_model.modules()),
            )
        )
        assert target.shape == (1, 8, 3)
        return SimpleNamespace(trajectory_loss=features["token"].reshape(()))

    def compose(
        trajectory_loss,
        current_model,
        current_reference,
        current_manifest,
        *,
        trajectory_weight,
        preservation_weight,
    ):
        compose_calls.append(
            (
                current_model,
                current_reference,
                current_manifest,
                trajectory_weight,
                preservation_weight,
            )
        )
        preservation = torch.tensor(2.0)
        return SimpleNamespace(
            total=12.0 * trajectory_loss + 1.0e-3 * preservation,
            trajectory=trajectory_loss,
            preservation=preservation,
        )

    monkeypatch.setattr(validation_module, "reset_torch_seed", reset)
    monkeypatch.setattr(validation_module, "forward_trajectory_supervised", supervised)
    monkeypatch.setattr(validation_module, "compose_finetune_loss", compose)

    forward = validate_supervised_loss_b1(
        model,
        samples,
        reference=reference,
        manifest=manifest,
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
    )
    modes_after_forward = _module_modes(model)
    reverse = validate_supervised_loss_b1(
        model,
        reversed(samples),
        reference=reference,
        manifest=manifest,
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
    )

    assert forward == reverse
    assert forward.count == 2
    assert forward.trajectory == 2.0
    assert forward.preservation == 2.0
    assert forward.total == pytest.approx(24.002)
    assert all(
        math.isfinite(value)
        for value in (forward.total, forward.trajectory, forward.preservation)
    )
    assert modes_after_forward == modes_before
    assert _module_modes(model) == modes_before
    assert len(compose_calls) == 4
    assert all(
        call == (model, reference, manifest, 12.0, 1.0e-3) for call in compose_calls
    )
    expected_seed_order = [
        validation_loss_seed(100),
        validation_loss_seed(200),
        validation_loss_seed(200),
        validation_loss_seed(100),
    ]
    assert events[::2] == [("seed", seed) for seed in expected_seed_order]
    for event, expected_seed in zip(events[1::2], expected_seed_order):
        assert event[0] == "adapter"
        assert event[1] == expected_seed
        assert event[2] is False
        assert not any(event[3])


def test_supervised_loss_empty_and_exception_restore_exact_boundary(
    monkeypatch,
) -> None:
    model = _LossModel()
    model.eval()
    model.encoder.train()
    modes_before = _module_modes(model)
    empty = validate_supervised_loss_b1(
        model,
        (),
        reference=object(),
        manifest=object(),
        trajectory_weight=12.0,
        preservation_weight=1.0e-3,
    )
    assert empty == ValidationLossSummary(0, None, None, None)
    assert _module_modes(model) == modes_before

    def fail(*_args, **_kwargs):
        raise RuntimeError("adapter failed")

    monkeypatch.setattr(validation_module, "forward_trajectory_supervised", fail)
    with pytest.raises(RuntimeError, match="adapter failed"):
        validate_supervised_loss_b1(
            model,
            (_loss_sample("A-sample", 100, 1.0),),
            reference=object(),
            manifest=object(),
            trajectory_weight=12.0,
            preservation_weight=1.0e-3,
        )
    assert _module_modes(model) == modes_before


def test_selector_and_loss_validation_reject_non_b1_before_forward(monkeypatch) -> None:
    selector_model = _SelectorModel()
    sample = _selector_sample("A-sample", 10, 1.0)
    sample["trajectory"] = torch.zeros(2, 8, 3)
    sample["features"]["token"] = torch.ones(2, 1)

    with pytest.raises(ValueError, match="B=1"):
        validate_selector_b1(selector_model, (sample,))
    assert selector_model.forward_calls == 0

    loss_model = _LossModel()
    adapter_calls = []
    monkeypatch.setattr(
        validation_module,
        "forward_trajectory_supervised",
        lambda *_args, **_kwargs: adapter_calls.append(True),
    )
    with pytest.raises(ValueError, match="B=1"):
        validate_supervised_loss_b1(
            loss_model,
            (sample,),
            reference=object(),
            manifest=object(),
            trajectory_weight=12.0,
            preservation_weight=1.0e-3,
        )
    assert adapter_calls == []
