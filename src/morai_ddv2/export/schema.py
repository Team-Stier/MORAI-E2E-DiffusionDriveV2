from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from morai_ddv2.training.contracts import (
    TRAINABLE_PREFIXES,
    FreezeManifest,
)
from morai_ddv2.upstream.ddv2_sel import (
    PDM_REMOVED_CHECKPOINT_KEY,
    RAW_SEL_KEY_COUNT,
    RUNTIME_SEL_KEY_COUNT,
    SelSchemaError,
    normalize_sel_key,
    normalize_sel_runtime_state,
)

from .state_hash import _require_supported_tensor, tensor_data_sha256


_BATCHNORM_BUFFER_SUFFIXES = (
    ".running_mean",
    ".running_var",
    ".num_batches_tracked",
)


@dataclass(frozen=True)
class SelSchemaReport:
    raw_key_count: int
    runtime_key_count: int
    raw_shapes: Mapping[str, tuple[int, ...]]
    normalized_to_raw: Mapping[str, str]


def _raw_key_tuple(state: Mapping[str, torch.Tensor], *, name: str) -> tuple[str, ...]:
    if not isinstance(state, Mapping):
        raise SelSchemaError(f"{name} must be a mapping")
    keys = []
    for key in state:
        if type(key) is not str:
            raise SelSchemaError(f"{name} keys must be strings")
        try:
            key.encode("utf-8")
        except UnicodeEncodeError as error:
            raise SelSchemaError(f"{name} key is not valid UTF-8: {key!r}") from error
        keys.append(key)
    return tuple(keys)


def _key_set_mismatch(
    candidate_keys: tuple[str, ...], expected_keys: tuple[str, ...]
) -> SelSchemaError:
    candidate = set(candidate_keys)
    expected = set(expected_keys)
    missing = sorted(expected - candidate, key=lambda item: item.encode("utf-8"))
    unexpected = sorted(candidate - expected, key=lambda item: item.encode("utf-8"))
    return SelSchemaError(
        f"raw key set mismatch: missing={missing!r} unexpected={unexpected!r}"
    )


def validate_raw_schema(
    raw_state: Mapping[str, torch.Tensor],
    expected_raw_state: Mapping[str, torch.Tensor],
) -> SelSchemaReport:
    raw_keys = _raw_key_tuple(raw_state, name="raw state")
    expected_keys = _raw_key_tuple(expected_raw_state, name="expected raw state")

    if set(raw_keys) != set(expected_keys):
        raise _key_set_mismatch(raw_keys, expected_keys)
    if len(raw_keys) != RAW_SEL_KEY_COUNT:
        raise SelSchemaError(
            f"expected {RAW_SEL_KEY_COUNT} raw checkpoint keys, got {len(raw_keys)}"
        )

    raw_shapes: OrderedDict[str, tuple[int, ...]] = OrderedDict()
    for raw_key in raw_keys:
        candidate = _require_supported_tensor(
            raw_state[raw_key], name=f"raw state[{raw_key!r}]"
        )
        expected = _require_supported_tensor(
            expected_raw_state[raw_key], name=f"expected raw state[{raw_key!r}]"
        )
        candidate_shape = tuple(candidate.shape)
        expected_shape = tuple(expected.shape)
        if candidate_shape != expected_shape:
            raise SelSchemaError(
                f"shape mismatch for raw key {raw_key!r}: "
                f"expected {expected_shape}, got {candidate_shape}"
            )
        raw_shapes[raw_key] = candidate_shape

    runtime = normalize_sel_runtime_state(raw_state)
    if len(runtime) != RUNTIME_SEL_KEY_COUNT:
        raise SelSchemaError(
            f"expected {RUNTIME_SEL_KEY_COUNT} runtime keys, got {len(runtime)}"
        )
    normalized_to_raw: OrderedDict[str, str] = OrderedDict()
    for raw_key in raw_keys:
        normalized = normalize_sel_key(raw_key)
        if normalized == PDM_REMOVED_CHECKPOINT_KEY:
            continue
        normalized_to_raw[normalized] = raw_key
    return SelSchemaReport(
        raw_key_count=len(raw_keys),
        runtime_key_count=len(runtime),
        raw_shapes=raw_shapes,
        normalized_to_raw=normalized_to_raw,
    )


def _validated_mutable_names(manifest: FreezeManifest) -> tuple[str, ...]:
    if not isinstance(manifest, FreezeManifest):
        raise SelSchemaError("manifest must be FreezeManifest")
    trainable = manifest.trainable_parameters
    frozen = manifest.frozen_parameters
    mutable = manifest.mutable_state_entries
    for label, names in (
        ("trainable", trainable),
        ("frozen", frozen),
        ("mutable", mutable),
    ):
        if not isinstance(names, tuple) or any(
            type(name) is not str or not name for name in names
        ):
            raise SelSchemaError(f"manifest {label} names must be nonempty strings")
        if len(names) != len(set(names)):
            raise SelSchemaError(f"manifest {label} names must be unique")
    if set(trainable).intersection(frozen):
        raise SelSchemaError("manifest trainable and frozen parameters overlap")
    if not set(trainable).issubset(mutable):
        raise SelSchemaError("manifest mutable state must include every trainable name")
    trainable_set = set(trainable)
    for name in mutable:
        if not name.startswith(TRAINABLE_PREFIXES):
            raise SelSchemaError(f"manifest contains unauthorized mutable name: {name}")
        if name not in trainable_set and not name.endswith(_BATCHNORM_BUFFER_SUFFIXES):
            raise SelSchemaError(f"manifest contains unauthorized mutable name: {name}")
    return mutable


def _require_exact_mutable_state(
    state: Mapping[str, torch.Tensor], mutable_names: tuple[str, ...]
) -> None:
    keys = _raw_key_tuple(state, name="mutable state")
    if set(keys) != set(mutable_names) or len(keys) != len(mutable_names):
        missing = sorted(set(mutable_names) - set(keys))
        unexpected = sorted(set(keys) - set(mutable_names))
        raise SelSchemaError(
            "mutable state keys do not match manifest: "
            f"missing={missing!r} unexpected={unexpected!r}"
        )


def merge_mutable_state(
    raw_state: Mapping[str, torch.Tensor],
    mutable_normalized_state: Mapping[str, torch.Tensor],
    manifest: FreezeManifest,
) -> OrderedDict[str, torch.Tensor]:
    report = validate_raw_schema(raw_state, raw_state)
    mutable_names = _validated_mutable_names(manifest)
    _require_exact_mutable_state(mutable_normalized_state, mutable_names)
    missing_raw = sorted(set(mutable_names) - set(report.normalized_to_raw))
    if missing_raw:
        raise SelSchemaError(
            f"manifest mutable names are absent from raw state: {missing_raw!r}"
        )

    merged: OrderedDict[str, torch.Tensor] = OrderedDict(raw_state.items())
    for normalized_name in mutable_names:
        raw_key = report.normalized_to_raw[normalized_name]
        original = _require_supported_tensor(
            raw_state[raw_key], name=f"raw state[{raw_key!r}]"
        )
        replacement = _require_supported_tensor(
            mutable_normalized_state[normalized_name],
            name=f"mutable state[{normalized_name!r}]",
        )
        if tuple(replacement.shape) != tuple(original.shape):
            raise SelSchemaError(
                f"shape mismatch for mutable state {normalized_name!r}: "
                f"expected {tuple(original.shape)}, got {tuple(replacement.shape)}"
            )
        if replacement.dtype != original.dtype:
            raise SelSchemaError(
                f"dtype mismatch for mutable state {normalized_name!r}: "
                f"expected {original.dtype}, got {replacement.dtype}"
            )
        merged[raw_key] = replacement.detach().to(device="cpu").contiguous().clone()
    return merged


def assert_only_allowed_changed(
    base_raw: Mapping[str, torch.Tensor],
    exported_raw: Mapping[str, torch.Tensor],
    manifest: FreezeManifest,
) -> None:
    report = validate_raw_schema(exported_raw, base_raw)
    mutable_names = _validated_mutable_names(manifest)
    missing_raw = sorted(set(mutable_names) - set(report.normalized_to_raw))
    if missing_raw:
        raise SelSchemaError(
            f"manifest mutable names are absent from raw state: {missing_raw!r}"
        )
    allowed_raw = {report.normalized_to_raw[name] for name in mutable_names}

    for raw_key, base_tensor in base_raw.items():
        exported_tensor = exported_raw[raw_key]
        if exported_tensor.dtype != base_tensor.dtype:
            raise SelSchemaError(
                f"dtype mismatch for raw key {raw_key!r}: "
                f"expected {base_tensor.dtype}, got {exported_tensor.dtype}"
            )
        if raw_key in allowed_raw:
            continue
        if tensor_data_sha256(base_tensor) != tensor_data_sha256(exported_tensor):
            raise SelSchemaError(f"unauthorized state change: {raw_key}")
