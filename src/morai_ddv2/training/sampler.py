from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler


class ResumeContractError(RuntimeError):
    """Raised when persisted sampler progress cannot be resumed exactly."""


@dataclass(frozen=True)
class SamplerState:
    dataset_size: int
    epoch: int
    permutation: torch.Tensor
    committed_cursor: int
    generator_state: torch.Tensor


class StatefulPermutationSampler(Sampler[int]):
    """Random permutation sampler whose cursor advances only after a train step."""

    def __init__(self, data_source: Sized, seed: int) -> None:
        if not isinstance(data_source, Sized):
            raise TypeError("data_source must be sized")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._size = len(data_source)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self._epoch = 0
        self._permutation = torch.randperm(
            self._size, generator=self._generator, device="cpu"
        )
        self._committed_cursor = 0

    def __iter__(self) -> Iterator[int]:
        start = self._committed_cursor
        for position in range(start, self._size):
            yield int(self._permutation[position])

    def __len__(self) -> int:
        return self._size - self._committed_cursor

    def mark_consumed(self, count: int) -> None:
        if (
            type(count) is not int
            or count <= 0
            or self._committed_cursor + count > self._size
        ):
            raise ResumeContractError("invalid consumed sample count")
        self._committed_cursor += count

    def finish_epoch(self) -> None:
        if self._committed_cursor != self._size:
            raise ResumeContractError("cannot finish an incomplete sampler epoch")
        self._epoch += 1
        self._permutation = torch.randperm(
            self._size, generator=self._generator, device="cpu"
        )
        self._committed_cursor = 0

    def state_dict(self) -> SamplerState:
        return SamplerState(
            dataset_size=self._size,
            epoch=self._epoch,
            permutation=self._permutation.clone(),
            committed_cursor=self._committed_cursor,
            generator_state=self._generator.get_state().clone(),
        )

    def load_state_dict(self, state: SamplerState) -> None:
        if type(state) is not SamplerState:
            raise TypeError("state must be a SamplerState")
        if type(state.dataset_size) is not int or state.dataset_size != self._size:
            raise ResumeContractError("sampler dataset size changed")
        if (
            type(state.epoch) is not int
            or state.epoch < 0
            or type(state.committed_cursor) is not int
            or not 0 <= state.committed_cursor <= self._size
        ):
            raise ResumeContractError("invalid sampler epoch or cursor")
        if (
            not isinstance(state.permutation, torch.Tensor)
            or state.permutation.device.type != "cpu"
            or state.permutation.dtype != torch.long
            or state.permutation.shape != (self._size,)
        ):
            raise ResumeContractError("invalid sampler permutation")
        permutation = state.permutation.detach().clone()
        expected = torch.arange(self._size, dtype=torch.long, device="cpu")
        if not torch.equal(permutation.sort().values, expected):
            raise ResumeContractError("sampler state is not a permutation")
        if (
            not isinstance(state.generator_state, torch.Tensor)
            or state.generator_state.device.type != "cpu"
            or state.generator_state.dtype != torch.uint8
            or state.generator_state.ndim != 1
        ):
            raise ResumeContractError("invalid sampler generator state")
        generator_state = state.generator_state.detach().clone()
        previous_generator_state = self._generator.get_state()
        try:
            self._generator.set_state(generator_state)
        except (RuntimeError, TypeError) as error:
            self._generator.set_state(previous_generator_state)
            raise ResumeContractError("invalid sampler generator state") from error
        self._epoch = state.epoch
        self._permutation = permutation
        self._committed_cursor = state.committed_cursor


__all__ = (
    "ResumeContractError",
    "SamplerState",
    "StatefulPermutationSampler",
)
