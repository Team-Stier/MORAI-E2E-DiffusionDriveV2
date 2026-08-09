from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from morai_ddv2.upstream.ddv2_sel import SelSchemaError


FLOAT32_DATA_SHA256 = "685b7ddd99d5c6487e584654c8e84ceb0d9b47eea2d0eaa7cd4c1aed901ca064"
FLOAT16_DATA_SHA256 = "155ec98b6bf9c10eac3af03356739efcd8c80353a4544d13fafdec7dbbabf620"
INT64_DATA_SHA256 = "5b37e025ca82f258a25fd27cd00ddc0a9a7c200ab8945d6f7cf798115092ae2d"
NONCONTIGUOUS_DATA_SHA256 = (
    "0c9d0bb54e4f5a0121543129f106617549c7ff2b34c6842c5a2e19186c5a7914"
)
STATE_DICT_SHA256 = "ef5ed4dbab0457096a43c3cca8588cabee83953b81a0cd7ac5f5b17cd31cef5e"
STATE_SCHEMA_SHA256 = "ae2cbf076a20651c54190ee58131a391b7d579fcb5e73cf8c22d30c0947e70e2"


def _hashing_state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        [
            (
                "\u4e2d",
                torch.tensor([0, -1, 2**40 + 3], dtype=torch.int64),
            ),
            (
                "a",
                torch.tensor(
                    [1.0, -2.5, 0.0, 3.25],
                    dtype=torch.float32,
                    requires_grad=True,
                ),
            ),
            (
                "\u00e9",
                torch.tensor([1.0, -2.0, 0.5, 3.5], dtype=torch.float16),
            ),
        ]
    )


def test_tensor_data_sha256_has_fixed_little_endian_vectors() -> None:
    from morai_ddv2.export.state_hash import tensor_data_sha256

    float32 = torch.tensor(
        [1.0, -2.5, 0.0, 3.25], dtype=torch.float32, requires_grad=True
    )
    float16 = torch.tensor([1.0, -2.0, 0.5, 3.5], dtype=torch.float16)
    int64 = torch.tensor([0, -1, 2**40 + 3], dtype=torch.int64)

    assert tensor_data_sha256(float32) == FLOAT32_DATA_SHA256
    assert tensor_data_sha256(float16) == FLOAT16_DATA_SHA256
    assert tensor_data_sha256(int64) == INT64_DATA_SHA256


def test_tensor_data_sha256_hashes_noncontiguous_tensor_in_c_order() -> None:
    from morai_ddv2.export.state_hash import tensor_data_sha256

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    assert not tensor.is_contiguous()

    assert tensor_data_sha256(tensor) == NONCONTIGUOUS_DATA_SHA256


def test_state_hashes_are_fixed_and_ignore_mapping_insertion_order() -> None:
    from morai_ddv2.export.state_hash import (
        state_dict_sha256,
        state_schema_sha256,
    )

    first = _hashing_state()
    second = OrderedDict(reversed(tuple(first.items())))

    assert state_dict_sha256(first) == STATE_DICT_SHA256
    assert state_dict_sha256(second) == STATE_DICT_SHA256
    assert state_schema_sha256(first) == STATE_SCHEMA_SHA256
    assert state_schema_sha256(second) == STATE_SCHEMA_SHA256


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([True], dtype=torch.bool),
        torch.tensor([1.0], dtype=torch.bfloat16),
        torch.sparse_coo_tensor([[0]], [1.0], size=(1,)),
    ],
    ids=["float64", "int32", "bool", "bfloat16", "sparse"],
)
def test_tensor_data_sha256_rejects_unsupported_or_unsafe_tensor_types(
    tensor: torch.Tensor,
) -> None:
    from morai_ddv2.export.state_hash import tensor_data_sha256

    with pytest.raises(SelSchemaError, match="unsupported tensor"):
        tensor_data_sha256(tensor)


def test_state_hashes_reject_non_string_keys_and_non_tensor_values() -> None:
    from morai_ddv2.export.state_hash import (
        state_dict_sha256,
        state_schema_sha256,
    )

    with pytest.raises(SelSchemaError, match="keys must be strings"):
        state_schema_sha256({1: torch.zeros(1)})  # type: ignore[dict-item]
    with pytest.raises(SelSchemaError, match="must be torch.Tensor"):
        state_dict_sha256({"weight": [1.0]})  # type: ignore[dict-item]
