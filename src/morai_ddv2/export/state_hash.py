from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import rfc8785
import torch

from morai_ddv2.config import sha256_bytes
from morai_ddv2.upstream.ddv2_sel import SelSchemaError


_SUPPORTED_DTYPES = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int64: "int64",
}


@dataclass(frozen=True)
class TensorSchemaEntry:
    key: str
    shape: tuple[int, ...]


def _require_supported_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SelSchemaError(f"{name} must be torch.Tensor")
    if value.layout != torch.strided or value.is_quantized:
        raise SelSchemaError(f"unsupported tensor layout for {name}: {value.layout}")
    if value.device.type == "meta":
        raise SelSchemaError(f"unsupported tensor device for {name}: meta")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise SelSchemaError(f"unsupported tensor dtype for {name}: {value.dtype}")
    if value.is_conj() or value.is_neg():
        raise SelSchemaError(f"unsupported tensor view bits for {name}")
    return value


def _sorted_state_items(
    state: Mapping[str, torch.Tensor],
) -> list[tuple[str, torch.Tensor]]:
    if not isinstance(state, Mapping):
        raise SelSchemaError("state must be a mapping")
    items = []
    for key, value in state.items():
        if type(key) is not str:
            raise SelSchemaError("state keys must be strings")
        try:
            key.encode("utf-8")
        except UnicodeEncodeError as error:
            raise SelSchemaError(f"state key is not valid UTF-8: {key!r}") from error
        items.append((key, _require_supported_tensor(value, name=f"state[{key!r}]")))
    return sorted(items, key=lambda item: item[0].encode("utf-8"))


def _little_endian_c_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().to(device="cpu").contiguous()
    array = cpu.numpy()
    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return little_endian.tobytes(order="C")


def tensor_data_sha256(tensor: torch.Tensor) -> str:
    checked = _require_supported_tensor(tensor, name="tensor")
    return sha256_bytes(_little_endian_c_bytes(checked))


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    entries: list[dict[str, Any]] = []
    for key, tensor in _sorted_state_items(state):
        entries.append(
            {
                "key": key,
                "dtype": _SUPPORTED_DTYPES[tensor.dtype],
                "shape": list(tensor.shape),
                "data_sha256": tensor_data_sha256(tensor),
            }
        )
    return sha256_bytes(rfc8785.dumps(entries))


def state_schema_sha256(state: Mapping[str, torch.Tensor]) -> str:
    entries = [
        {"key": key, "shape": list(tensor.shape)}
        for key, tensor in _sorted_state_items(state)
    ]
    return sha256_bytes(rfc8785.dumps(entries))
