from __future__ import annotations

from collections.abc import Iterator
import copy
import random
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import RngState
from .sampler import ResumeContractError


def _require_cpu_generator(generator: torch.Generator) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("dataloader_generator must be a torch.Generator")
    if generator.device.type != "cpu":
        raise ResumeContractError("DataLoader generator must be on CPU")


def _owned_byte_state(value: torch.Tensor, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
    ):
        raise ResumeContractError(f"{label} must be a CPU byte RNG state")
    return value.detach().clone()


def _validated_cpu_generator_state(value: torch.Tensor, label: str) -> torch.Tensor:
    owned = _owned_byte_state(value, label)
    probe = torch.Generator(device="cpu")
    try:
        probe.set_state(owned)
    except (RuntimeError, TypeError) as error:
        raise ResumeContractError(f"{label} is invalid") from error
    return owned


def _owned_numpy_state(value: tuple[Any, ...]) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise ResumeContractError("NumPy RNG state must be a tuple")
    current = np.random.get_state()
    try:
        np.random.set_state(value)
        owned = np.random.get_state()
    except (TypeError, ValueError) as error:
        raise ResumeContractError("NumPy RNG state is invalid") from error
    finally:
        np.random.set_state(current)
    return (
        owned[0],
        owned[1].copy(),
        owned[2],
        owned[3],
        owned[4],
    )


def _owned_python_state(value: object) -> object:
    current = random.getstate()
    try:
        random.setstate(value)
        owned = copy.deepcopy(random.getstate())
    except (TypeError, ValueError) as error:
        raise ResumeContractError("Python RNG state is invalid") from error
    finally:
        random.setstate(current)
    return owned


def _owned_cuda_states(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if not isinstance(values, tuple):
        raise ResumeContractError("CUDA RNG states must be a tuple")
    if not values and not torch.cuda.is_initialized():
        return ()
    device_count = torch.cuda.device_count()
    if len(values) != device_count:
        raise ResumeContractError("CUDA RNG state count does not match visible devices")
    owned = []
    for index, value in enumerate(values):
        state = _owned_byte_state(value, f"CUDA RNG state {index}")
        try:
            probe = torch.Generator(device=torch.device("cuda", index))
            probe.set_state(state)
        except (RuntimeError, TypeError) as error:
            raise ResumeContractError(f"CUDA RNG state {index} is invalid") from error
        owned.append(state)
    return tuple(owned)


def clone_rng_state(state: RngState) -> RngState:
    if type(state) is not RngState:
        raise TypeError("state must be an RngState")
    return RngState(
        python=_owned_python_state(state.python),
        numpy=_owned_numpy_state(state.numpy),
        torch_cpu=_validated_cpu_generator_state(
            state.torch_cpu, "Torch CPU RNG state"
        ),
        torch_cuda=_owned_cuda_states(state.torch_cuda),
        dataloader_cpu=_validated_cpu_generator_state(
            state.dataloader_cpu, "DataLoader RNG state"
        ),
        dataloader_iterator_base_cpu=_validated_cpu_generator_state(
            state.dataloader_iterator_base_cpu,
            "DataLoader iterator-base RNG state",
        ),
    )


def capture_rng_state(
    dataloader_generator: torch.Generator,
    *,
    dataloader_iterator_base_cpu: torch.Tensor,
) -> RngState:
    _require_cpu_generator(dataloader_generator)
    iterator_base = _validated_cpu_generator_state(
        dataloader_iterator_base_cpu,
        "DataLoader iterator-base RNG state",
    )
    numpy_state = np.random.get_state()
    return RngState(
        python=copy.deepcopy(random.getstate()),
        numpy=(
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=tuple(state.clone() for state in torch.cuda.get_rng_state_all()),
        dataloader_cpu=dataloader_generator.get_state().clone(),
        dataloader_iterator_base_cpu=iterator_base,
    )


def restore_rng_state(
    state: RngState,
    dataloader_generator: torch.Generator,
) -> None:
    _require_cpu_generator(dataloader_generator)
    owned = clone_rng_state(state)
    random.setstate(owned.python)
    np.random.set_state(owned.numpy)
    torch.set_rng_state(owned.torch_cpu)
    if owned.torch_cuda:
        torch.cuda.set_rng_state_all(list(owned.torch_cuda))
    dataloader_generator.set_state(owned.dataloader_cpu)


def create_dataloader_iterator(
    dataloader: DataLoader,
    dataloader_generator: torch.Generator,
    *,
    iterator_base_state: Optional[torch.Tensor] = None,
    current_generator_state: Optional[torch.Tensor] = None,
) -> tuple[Iterator[Any], torch.Tensor]:
    if not isinstance(dataloader, DataLoader):
        raise TypeError("dataloader must be a DataLoader")
    _require_cpu_generator(dataloader_generator)
    if dataloader.generator is not dataloader_generator:
        raise ResumeContractError(
            "DataLoader must use the supplied CPU generator by identity"
        )
    if (iterator_base_state is None) != (current_generator_state is None):
        raise ResumeContractError(
            "iterator-base and current generator states must be supplied together"
        )

    if iterator_base_state is None:
        base = dataloader_generator.get_state().clone()
        try:
            iterator = iter(dataloader)
        except Exception:
            dataloader_generator.set_state(base)
            raise
        return iterator, base

    base = _validated_cpu_generator_state(
        iterator_base_state,
        "DataLoader iterator-base RNG state",
    )
    current = _validated_cpu_generator_state(
        current_generator_state,
        "DataLoader current RNG state",
    )
    original = dataloader_generator.get_state().clone()
    try:
        dataloader_generator.set_state(base)
        iterator = iter(dataloader)
        dataloader_generator.set_state(current)
    except Exception:
        dataloader_generator.set_state(original)
        raise
    return iterator, base.clone()


__all__ = (
    "capture_rng_state",
    "clone_rng_state",
    "create_dataloader_iterator",
    "restore_rng_state",
)
