from __future__ import annotations

import pytest
import torch
from torch import nn

from morai_ddv2.training.supervised_adapter import sample_noisy_plan_anchors


class _RecordingScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append((original_samples.clone(), noise.clone(), timesteps.clone()))
        return original_samples + 0.1 * noise + timesteps[:, None, None, None] / 100.0


class _Head(nn.Module):
    def __init__(self, shape: tuple[int, ...] = (20, 8, 2)) -> None:
        super().__init__()
        values = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).view(
            shape
        )
        self.plan_anchor = nn.Parameter(values - 80.0, requires_grad=False)
        self.diffusion_scheduler = _RecordingScheduler()


def test_sample_noisy_plan_anchors_applies_exact_xy_transform() -> None:
    head = _Head()
    torch.manual_seed(20260809)

    result = sample_noisy_plan_anchors(head, 2, torch.device("cpu"))

    torch.manual_seed(20260809)
    expected_anchor = head.plan_anchor.detach().unsqueeze(0).repeat(2, 1, 1, 1)
    expected_normalized = expected_anchor.clone()
    expected_normalized[..., 0] /= 50.0
    expected_normalized[..., 1] /= 20.0
    expected_timesteps = torch.randint(0, 50, (2,), dtype=torch.long)
    expected_noise = torch.randn_like(expected_normalized)
    expected_noisy = (
        expected_normalized
        + 0.1 * expected_noise
        + expected_timesteps[:, None, None, None] / 100.0
    ).float()
    expected_noisy = expected_noisy.clamp(-1.0, 1.0)
    expected_decoder = expected_noisy.clone()
    expected_decoder[..., 0] *= 50.0
    expected_decoder[..., 1] *= 20.0

    assert result.plan_anchor.shape == (2, 20, 8, 2)
    assert result.plan_anchor.dtype == torch.float32
    assert result.plan_anchor.data_ptr() != head.plan_anchor.data_ptr()
    assert torch.equal(result.plan_anchor, expected_anchor)
    assert torch.equal(result.timesteps, expected_timesteps)
    assert torch.equal(result.noise, expected_noise)
    assert torch.equal(result.decoder_input, expected_decoder)
    assert len(head.diffusion_scheduler.calls) == 1
    scheduler_anchor, scheduler_noise, scheduler_timesteps = (
        head.diffusion_scheduler.calls[0]
    )
    assert torch.equal(scheduler_anchor, expected_normalized)
    assert torch.equal(scheduler_noise, expected_noise)
    assert torch.equal(scheduler_timesteps, expected_timesteps)


def test_sample_noisy_plan_anchors_permits_timestep_49(monkeypatch) -> None:
    head = _Head()
    torch.manual_seed(49)

    def _timestep_49(low, high, size, *, device, dtype):
        assert (low, high, size, device, dtype) == (
            0,
            50,
            (2,),
            torch.device("cpu"),
            torch.long,
        )
        return torch.tensor([49, 0], device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randint", _timestep_49)

    result = sample_noisy_plan_anchors(head, 2, torch.device("cpu"))

    assert result.timesteps.tolist() == [49, 0]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_sample_noisy_plan_anchors_rejects_nonpositive_batches(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        sample_noisy_plan_anchors(_Head(), batch_size, torch.device("cpu"))


def test_sample_noisy_plan_anchors_rejects_wrong_anchor_shape() -> None:
    with pytest.raises(ValueError, match="plan_anchor"):
        sample_noisy_plan_anchors(_Head((20, 7, 2)), 1, torch.device("cpu"))
