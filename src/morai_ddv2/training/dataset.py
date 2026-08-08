from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from morai_ddv2.cache.loader import (
    VerifiedManifest,
    load_tensor_sample,
)
from morai_ddv2.errors import IntegrityError
from morai_ddv2.training.sampler import StatefulPermutationSampler


VALIDATION_RUN_ID = "morai-ddv2-v1-oracle-command-validation-v1"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260806
    epochs: int = 30
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    amp: bool = True
    grad_clip_norm: float = 1.0
    trajectory_weight: float = 12.0
    preservation_weight: float = 1.0e-3
    validation_batch_size: int = 1
    validation_run_id: str = VALIDATION_RUN_ID
    checkpoint_interval_steps: int = 100

    def __post_init__(self) -> None:
        _require_integer(self.seed, "seed", minimum=0)
        _require_integer(self.epochs, "epochs", minimum=1)
        _require_integer(self.batch_size, "batch_size", minimum=1)
        _require_integer(self.num_workers, "num_workers", minimum=0)
        _require_float(self.learning_rate, "learning_rate", strictly_positive=True)
        _require_float(self.weight_decay, "weight_decay", strictly_positive=False)
        if type(self.amp) is not bool:
            raise TypeError("amp must be a bool")
        _require_float(self.grad_clip_norm, "grad_clip_norm", strictly_positive=True)
        _require_float(
            self.trajectory_weight, "trajectory_weight", strictly_positive=True
        )
        _require_float(
            self.preservation_weight, "preservation_weight", strictly_positive=True
        )
        _require_integer(self.validation_batch_size, "validation_batch_size", minimum=1)
        _require_integer(
            self.checkpoint_interval_steps,
            "checkpoint_interval_steps",
            minimum=1,
        )
        if self.trajectory_weight != 12.0:
            raise ValueError("trajectory_weight must be exactly 12.0")
        if self.preservation_weight != 1.0e-3:
            raise ValueError("preservation_weight must be exactly 1e-3")
        if self.validation_batch_size != 1:
            raise ValueError("validation_batch_size must be exactly 1")
        if self.validation_run_id != VALIDATION_RUN_ID:
            raise ValueError("validation_run_id is fixed by the validation contract")


def _require_integer(value: Any, name: str, *, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError("{} must be an integer".format(name))
    if value < minimum:
        raise ValueError("{} must be at least {}".format(name, minimum))


def _require_float(value: Any, name: str, *, strictly_positive: bool) -> None:
    if type(value) is not float:
        raise TypeError("{} must be a float".format(name))
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    if strictly_positive and value <= 0.0:
        raise ValueError("{} must be positive".format(name))
    if not strictly_positive and value < 0.0:
        raise ValueError("{} must be nonnegative".format(name))


def _owned_float32_tensor(value: np.ndarray) -> torch.Tensor:
    owned = np.array(value, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(owned)


class MoraiTrainingDataset(Dataset):
    """Deterministic split view over a previously verified MORAI cache."""

    def __init__(
        self,
        manifest: VerifiedManifest,
        split: Literal["train", "validation"],
    ) -> None:
        if type(manifest) is not VerifiedManifest:
            raise TypeError("manifest must be a VerifiedManifest")
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        if manifest.dataset_record["dirty_override"] is not None:
            raise IntegrityError(
                "dirty/development cache is forbidden for training datasets"
            )

        train_bags = set(manifest.dataset_record["train_relative_bag_paths"])
        validation_bags = set(manifest.dataset_record["validation_relative_bag_paths"])
        if train_bags & validation_bags:
            raise IntegrityError("training and validation bag splits overlap")
        declared_split = {path: "train" for path in train_bags}
        declared_split.update({path: "validation" for path in validation_bags})

        selected = []
        seen_sample_ids = set()
        for record in manifest.sample_records:
            relative_path = record["relative_bag_path"]
            expected_split = declared_split.get(relative_path)
            if expected_split is None or record["split"] != expected_split:
                raise IntegrityError("sample does not match its declared bag split")
            sample_id = record["sample_id"]
            if sample_id in seen_sample_ids:
                raise IntegrityError("training manifest repeats a sample ID")
            seen_sample_ids.add(sample_id)
            if record["split"] == split:
                selected.append(record)
        selected.sort(
            key=lambda record: (
                record["relative_bag_path"].encode("utf-8"),
                record["timestamp_ns"],
            )
        )
        self._manifest = manifest
        self._records = tuple(selected)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        record = self._records[index]
        sample = load_tensor_sample(self._manifest, record["sample_id"])
        return {
            "features": {
                "camera_feature": _owned_float32_tensor(sample.camera_feature),
                "lidar_feature": _owned_float32_tensor(sample.lidar_feature),
                "status_feature": _owned_float32_tensor(sample.status_feature),
            },
            "trajectory": _owned_float32_tensor(sample.trajectory),
            "metadata": dict(record),
        }


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (1 << 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(
    manifest: VerifiedManifest,
    config: TrainingConfig,
) -> tuple[
    DataLoader,
    DataLoader,
    StatefulPermutationSampler,
    torch.Generator,
]:
    if type(manifest) is not VerifiedManifest:
        raise TypeError("manifest must be a VerifiedManifest")
    if type(config) is not TrainingConfig:
        raise TypeError("config must be a TrainingConfig")
    train_dataset = MoraiTrainingDataset(manifest, split="train")
    validation_dataset = MoraiTrainingDataset(manifest, split="validation")
    sampler = StatefulPermutationSampler(train_dataset, seed=config.seed)
    dataloader_generator = torch.Generator(device="cpu")
    dataloader_generator.manual_seed(config.seed)
    common = {
        "num_workers": config.num_workers,
        "worker_init_fn": _seed_worker,
        "generator": dataloader_generator,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, sampler, dataloader_generator


__all__ = (
    "MoraiTrainingDataset",
    "TrainingConfig",
    "VALIDATION_RUN_ID",
    "build_dataloaders",
)
