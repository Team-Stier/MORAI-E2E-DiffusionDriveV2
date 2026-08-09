from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from morai_ddv2.training.contracts import FreezeManifest
from morai_ddv2.upstream.ddv2_sel import SelSchemaError


PREFIX = "agent._transfuser_model."
IMAGE_WEIGHT = "_backbone.image_encoder.layer.weight"
IMAGE_RUNNING_MEAN = "_backbone.image_encoder.norm.running_mean"
FROZEN_WEIGHT = "_trajectory_head.weight"
VOCAB = "_trajectory_head.vocab"


def _raw_state() -> OrderedDict[str, torch.Tensor]:
    named = [
        (PREFIX + IMAGE_WEIGHT, torch.tensor([1.0, 2.0], dtype=torch.float32)),
        (PREFIX + IMAGE_RUNNING_MEAN, torch.tensor([0.0], dtype=torch.float32)),
        (PREFIX + FROZEN_WEIGHT, torch.tensor([[3.0]], dtype=torch.float32)),
    ]
    named.extend(
        (f"{PREFIX}layers.layer_{index:03d}.weight", torch.tensor([float(index)]))
        for index in range(968)
    )
    named.append((PREFIX + VOCAB, torch.tensor([4.0], dtype=torch.float32)))
    state = OrderedDict(named)
    assert len(state) == 972
    return state


def _manifest() -> FreezeManifest:
    return FreezeManifest(
        trainable_parameters=(IMAGE_WEIGHT,),
        frozen_parameters=(FROZEN_WEIGHT,),
        mutable_state_entries=(IMAGE_WEIGHT, IMAGE_RUNNING_MEAN),
    )


def test_validate_raw_schema_reports_exact_972_to_971_mapping() -> None:
    from morai_ddv2.export.schema import SelSchemaReport, validate_raw_schema

    raw = _raw_state()

    report = validate_raw_schema(raw, raw)

    assert isinstance(report, SelSchemaReport)
    assert report.raw_key_count == 972
    assert report.runtime_key_count == 971
    assert list(report.raw_shapes) == list(raw)
    assert report.raw_shapes[PREFIX + FROZEN_WEIGHT] == (1, 1)
    assert VOCAB not in report.normalized_to_raw
    assert len(report.normalized_to_raw) == 971
    assert report.normalized_to_raw[IMAGE_WEIGHT] == PREFIX + IMAGE_WEIGHT


def test_validate_raw_schema_compares_raw_key_set_before_count() -> None:
    from morai_ddv2.export.schema import validate_raw_schema

    base = _raw_state()
    candidate = OrderedDict(base)
    candidate.pop(next(iter(candidate)))

    with pytest.raises(SelSchemaError, match="raw key set mismatch"):
        validate_raw_schema(candidate, base)


def test_validate_raw_schema_rejects_same_count_renamed_or_swapped_key() -> None:
    from morai_ddv2.export.schema import validate_raw_schema

    base = _raw_state()
    renamed = OrderedDict(base)
    tensor = renamed.pop(PREFIX + FROZEN_WEIGHT)
    renamed[PREFIX + "renamed.weight"] = tensor

    with pytest.raises(SelSchemaError, match="raw key set mismatch"):
        validate_raw_schema(renamed, base)


def test_validate_raw_schema_compares_shapes_before_normalization() -> None:
    from morai_ddv2.export.schema import validate_raw_schema

    base = _raw_state()
    duplicate_raw_key = "module." + PREFIX + IMAGE_WEIGHT
    base[duplicate_raw_key] = base.pop(PREFIX + "layers.layer_000.weight")
    candidate = OrderedDict(base)
    candidate[duplicate_raw_key] = torch.zeros(2)

    with pytest.raises(SelSchemaError, match="shape mismatch"):
        validate_raw_schema(candidate, base)


def test_validate_raw_schema_rejects_duplicate_normalized_key() -> None:
    from morai_ddv2.export.schema import validate_raw_schema

    base = _raw_state()
    duplicate_raw_key = "module." + PREFIX + IMAGE_WEIGHT
    base[duplicate_raw_key] = base.pop(PREFIX + "layers.layer_000.weight")

    with pytest.raises(SelSchemaError, match="duplicate normalized key"):
        validate_raw_schema(base, base)


def test_merge_mutable_state_preserves_raw_keys_order_dtype_and_shapes() -> None:
    from morai_ddv2.export.schema import merge_mutable_state

    raw = _raw_state()
    mutable = OrderedDict(
        [
            (IMAGE_WEIGHT, torch.tensor([8.0, 9.0], dtype=torch.float32)),
            (IMAGE_RUNNING_MEAN, torch.tensor([7.0], dtype=torch.float32)),
        ]
    )

    merged = merge_mutable_state(raw, mutable, _manifest())

    assert isinstance(merged, OrderedDict)
    assert list(merged) == list(raw)
    assert torch.equal(merged[PREFIX + IMAGE_WEIGHT], mutable[IMAGE_WEIGHT])
    assert torch.equal(merged[PREFIX + IMAGE_RUNNING_MEAN], mutable[IMAGE_RUNNING_MEAN])
    assert merged[PREFIX + IMAGE_WEIGHT].device.type == "cpu"
    assert merged[PREFIX + IMAGE_WEIGHT].dtype == raw[PREFIX + IMAGE_WEIGHT].dtype
    assert merged[PREFIX + IMAGE_WEIGHT] is not mutable[IMAGE_WEIGHT]
    assert merged[PREFIX + FROZEN_WEIGHT] is raw[PREFIX + FROZEN_WEIGHT]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unauthorized", "mutable state keys do not match manifest"),
        ("missing", "mutable state keys do not match manifest"),
        ("shape", "shape mismatch"),
        ("dtype", "dtype mismatch"),
    ],
)
def test_merge_mutable_state_rejects_untrusted_replacements(
    mutation: str, message: str
) -> None:
    from morai_ddv2.export.schema import merge_mutable_state

    mutable = OrderedDict(
        [
            (IMAGE_WEIGHT, torch.tensor([8.0, 9.0], dtype=torch.float32)),
            (IMAGE_RUNNING_MEAN, torch.tensor([7.0], dtype=torch.float32)),
        ]
    )
    if mutation == "unauthorized":
        mutable[FROZEN_WEIGHT] = torch.tensor([[10.0]])
    elif mutation == "missing":
        mutable.pop(IMAGE_RUNNING_MEAN)
    elif mutation == "shape":
        mutable[IMAGE_WEIGHT] = torch.zeros(3)
    elif mutation == "dtype":
        mutable[IMAGE_WEIGHT] = torch.zeros(2, dtype=torch.float16)

    with pytest.raises(SelSchemaError, match=message):
        merge_mutable_state(_raw_state(), mutable, _manifest())


def test_assert_only_allowed_changed_accepts_mutable_and_rejects_frozen_change() -> (
    None
):
    from morai_ddv2.export.schema import assert_only_allowed_changed

    base = _raw_state()
    exported = OrderedDict(base)
    exported[PREFIX + IMAGE_WEIGHT] = torch.tensor([8.0, 9.0])
    exported[PREFIX + IMAGE_RUNNING_MEAN] = torch.tensor([7.0])
    assert_only_allowed_changed(base, exported, _manifest())

    exported[PREFIX + FROZEN_WEIGHT] = torch.tensor([[11.0]])
    with pytest.raises(SelSchemaError, match="unauthorized state change"):
        assert_only_allowed_changed(base, exported, _manifest())


def test_assert_only_allowed_changed_rejects_schema_drift_first() -> None:
    from morai_ddv2.export.schema import assert_only_allowed_changed

    base = _raw_state()
    exported = OrderedDict(base)
    exported[PREFIX + FROZEN_WEIGHT] = torch.zeros(2)

    with pytest.raises(SelSchemaError, match="shape mismatch"):
        assert_only_allowed_changed(base, exported, _manifest())
