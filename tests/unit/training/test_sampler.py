from __future__ import annotations

from dataclasses import fields, replace
from itertools import islice

import pytest
import torch

from morai_ddv2.training.sampler import (
    ResumeContractError,
    SamplerState,
    StatefulPermutationSampler,
)


def _sampler(size: int = 9, seed: int = 20260806) -> StatefulPermutationSampler:
    return StatefulPermutationSampler(list(range(size)), seed=seed)


def test_sampler_state_public_contract_and_private_cpu_generator() -> None:
    sampler = _sampler(size=7)

    assert [field.name for field in fields(SamplerState)] == [
        "dataset_size",
        "epoch",
        "permutation",
        "committed_cursor",
        "generator_state",
    ]
    assert sampler._generator.device.type == "cpu"
    assert list(sampler) == list(sampler)
    assert len(sampler) == 7
    assert sampler.state_dict().committed_cursor == 0


def test_prefetch_does_not_commit_and_resume_restarts_at_first_uncommitted() -> None:
    sampler = _sampler(size=9)
    expected = list(sampler)
    prefetched = list(islice(iter(sampler), 6))

    sampler.mark_consumed(3)
    saved = sampler.state_dict()
    restored = _sampler(size=9)
    restored.load_state_dict(saved)

    assert prefetched == expected[:6]
    assert saved.committed_cursor == 3
    assert list(restored) == expected[3:]
    assert len(restored) == 6


def test_save_restore_stream_matches_two_uninterrupted_epochs() -> None:
    uninterrupted = _sampler(size=9)
    uninterrupted_epoch_zero = list(uninterrupted)
    uninterrupted.mark_consumed(9)
    uninterrupted.finish_epoch()
    uninterrupted_epoch_one = list(uninterrupted)

    interrupted = _sampler(size=9)
    consumed = list(islice(iter(interrupted), 5))[:3]
    interrupted.mark_consumed(len(consumed))
    saved = interrupted.state_dict()

    restored = _sampler(size=9)
    restored.load_state_dict(saved)
    remaining = list(restored)
    restored.mark_consumed(len(remaining))
    restored.finish_epoch()

    assert consumed + remaining == uninterrupted_epoch_zero
    assert list(restored) == uninterrupted_epoch_one
    assert restored.state_dict().epoch == 1


def test_sampler_requires_committed_completion_and_valid_consumption() -> None:
    sampler = _sampler(size=4)

    for count in (0, -1, 5):
        with pytest.raises(ResumeContractError, match="consumed"):
            sampler.mark_consumed(count)
    with pytest.raises(ResumeContractError, match="incomplete"):
        sampler.finish_epoch()

    sampler.mark_consumed(4)
    sampler.finish_epoch()

    assert sampler.state_dict().epoch == 1
    assert sampler.state_dict().committed_cursor == 0
    assert len(sampler) == 4


def test_sampler_state_is_owned_and_load_validation_is_exact() -> None:
    sampler = _sampler(size=5)
    saved = sampler.state_dict()
    original = list(sampler)

    saved.permutation[0] = saved.permutation[1]
    saved.generator_state.zero_()

    assert list(sampler) == original

    valid = sampler.state_dict()
    invalid_states = (
        replace(valid, dataset_size=6),
        replace(valid, epoch=-1),
        replace(valid, committed_cursor=-1),
        replace(valid, committed_cursor=6),
        replace(valid, permutation=torch.tensor([0, 1, 2, 3, 3])),
        replace(valid, permutation=torch.arange(4)),
        replace(valid, permutation=torch.arange(5, dtype=torch.float32)),
        replace(valid, generator_state=torch.zeros(1, dtype=torch.uint8)),
    )
    for state in invalid_states:
        with pytest.raises(ResumeContractError):
            _sampler(size=5).load_state_dict(state)


def test_load_state_clones_caller_owned_tensors() -> None:
    source = _sampler(size=6)
    source.mark_consumed(2)
    state = source.state_dict()
    restored = _sampler(size=6)
    restored.load_state_dict(state)
    expected = list(restored)

    state.permutation.fill_(0)
    state.generator_state.zero_()

    assert list(restored) == expected
    assert restored.state_dict().committed_cursor == 2
