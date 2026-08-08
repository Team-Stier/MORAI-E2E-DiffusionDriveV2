from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler

import morai_ddv2.training.dataset as dataset_module
from morai_ddv2.cache.domain import publish_domain_snapshot
from morai_ddv2.cache.hdf5 import publish_bag_artifacts
from morai_ddv2.cache.loader import load_tensor_sample, load_verified_manifest
from morai_ddv2.cache.manifest import (
    SAMPLE_RECORD_FIELDS,
    BagManifestInput,
    DirtyOverride,
    RejectionReport,
    assign_bag_splits,
    publish_manifest,
)
from morai_ddv2.config import ResolvedDomainArtifact
from morai_ddv2.errors import IntegrityError
from morai_ddv2.ros1 import BagDisposition
from morai_ddv2.training.dataset import (
    MoraiTrainingDataset,
    TrainingConfig,
    build_dataloaders,
)
from tests.unit.cache.test_hdf5 import (
    HIGH_GRID,
    HIGH_TIMESTAMP,
    SOURCE_COMMIT,
    _domain,
    _sample,
)
from tests.unit.cache.test_manifest import (
    ALL_ZERO_DROPS,
    _diagnostic,
    _publish_fixture,
)


PRESET_PATH = Path(__file__).resolve().parents[3] / "sensor_data/0725demo_ros.json"


def _publish_training_manifest(cache_root: Path):
    domain = _domain()
    snapshot = publish_domain_snapshot(
        cache_root=cache_root,
        artifact=ResolvedDomainArtifact(document=domain.document, sha256=domain.sha256),
    )
    relative_paths = ["drive/{:02d}.bag".format(index) for index in range(5)]
    splits = assign_bag_splits(relative_paths)
    artifacts = []
    bag_inventory = []
    for bag_index, relative_path in enumerate(relative_paths):
        source_sha = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
        timestamp = HIGH_TIMESTAMP + bag_index * 2_000_000_000
        grid = HIGH_GRID + bag_index * 2_000_000_000
        samples = (
            _sample(
                timestamp_ns=timestamp,
                grid_receipt_ns=grid,
                front_record_index=100 + 2 * bag_index,
                marker=1 + bag_index,
            ),
            _sample(
                timestamp_ns=timestamp + 500_000_000,
                grid_receipt_ns=grid + 500_000_000,
                front_record_index=101 + 2 * bag_index,
                marker=1 + bag_index,
            ),
        )
        artifact = publish_bag_artifacts(
            cache_root=cache_root,
            domain=domain,
            source_bag_sha256=source_sha,
            relative_bag_path=relative_path,
            samples=samples,
            source_code_commit=SOURCE_COMMIT,
        )
        artifacts.append(artifact)
        bag_inventory.append(
            BagManifestInput(
                relative_bag_path=relative_path,
                source_bag_sha256=source_sha,
                disposition=BagDisposition.ELIGIBLE,
                artifact=artifact,
                drop_counts=ALL_ZERO_DROPS,
                rejection_code=None,
                tf_ego_diagnostic=_diagnostic(),
            )
        )
    record = publish_manifest(
        cache_root=cache_root,
        domain_snapshot=snapshot,
        bags=tuple(artifacts),
        splits=splits,
        rejection_report=RejectionReport(bags=tuple(bag_inventory)),
        dirty_override=None,
    )
    return load_verified_manifest(
        path=cache_root / record.path,
        sensor_preset_path=PRESET_PATH,
        expected_domain_id=domain.document["domain_id"],
        expected_resolved_domain_sha256=domain.sha256,
        production=True,
    )


@pytest.fixture(scope="module")
def verified_manifest(tmp_path_factory):
    return _publish_training_manifest(tmp_path_factory.mktemp("training-cache"))


def test_dataset_uses_manifest_bag_splits_without_frame_leakage(
    verified_manifest,
) -> None:
    train = MoraiTrainingDataset(verified_manifest, split="train")
    validation = MoraiTrainingDataset(verified_manifest, split="validation")
    train_items = [train[index] for index in range(len(train))]
    validation_items = [validation[index] for index in range(len(validation))]

    train_bags = set(verified_manifest.dataset_record["train_relative_bag_paths"])
    validation_bags = set(
        verified_manifest.dataset_record["validation_relative_bag_paths"]
    )
    train_metadata = [item["metadata"] for item in train_items]
    validation_metadata = [item["metadata"] for item in validation_items]

    assert train_bags.isdisjoint(validation_bags)
    assert {item["relative_bag_path"] for item in train_metadata} == train_bags
    assert {
        item["relative_bag_path"] for item in validation_metadata
    } == validation_bags
    assert {item["sample_id"] for item in train_metadata}.isdisjoint(
        item["sample_id"] for item in validation_metadata
    )
    assert {item["split"] for item in train_metadata} == {"train"}
    assert {item["split"] for item in validation_metadata} == {"validation"}


def test_dataset_order_and_item_contract_are_exact(verified_manifest) -> None:
    dataset = MoraiTrainingDataset(verified_manifest, split="train")
    items = [dataset[index] for index in range(len(dataset))]
    metadata = [item["metadata"] for item in items]
    expected_records = sorted(
        (
            record
            for record in verified_manifest.sample_records
            if record["split"] == "train"
        ),
        key=lambda record: (
            record["relative_bag_path"].encode("utf-8"),
            record["timestamp_ns"],
        ),
    )

    assert [item["sample_id"] for item in metadata] == [
        item["sample_id"] for item in expected_records
    ]
    assert [
        (item["relative_bag_path"], item["timestamp_ns"]) for item in metadata
    ] == sorted((item["relative_bag_path"], item["timestamp_ns"]) for item in metadata)
    assert len({item["timestamp_ns"] for item in metadata}) == len(metadata)

    for item, expected in zip(items, expected_records):
        assert set(item) == {"features", "trajectory", "metadata"}
        assert set(item["features"]) == {
            "camera_feature",
            "lidar_feature",
            "status_feature",
        }
        assert item["features"]["camera_feature"].shape == (3, 256, 1024)
        assert item["features"]["lidar_feature"].shape == (1, 256, 256)
        assert item["features"]["status_feature"].shape == (8,)
        assert item["trajectory"].shape == (8, 3)
        assert all(
            tensor.dtype == torch.float32
            for tensor in (*item["features"].values(), item["trajectory"])
        )
        assert tuple(item["metadata"]) == SAMPLE_RECORD_FIELDS
        assert item["metadata"] == dict(expected)
        assert item["metadata"]["timestamp_ns"] == expected["timestamp_ns"]
        assert item["metadata"]["timestamp_ns"] != expected["grid_receipt_ns"]


def test_dataset_defers_each_tensor_read_to_verified_loader(
    verified_manifest, monkeypatch
) -> None:
    calls = []

    def tracked_loader(manifest, sample_id):
        calls.append((manifest, sample_id))
        return load_tensor_sample(manifest, sample_id)

    monkeypatch.setattr(dataset_module, "load_tensor_sample", tracked_loader)
    dataset = MoraiTrainingDataset(verified_manifest, split="train")

    assert calls == []
    item = dataset[0]

    assert calls == [(verified_manifest, item["metadata"]["sample_id"])]


def test_training_dataset_rejects_verified_development_cache(tmp_path: Path) -> None:
    record = _publish_fixture(
        tmp_path,
        dirty_override=DirtyOverride(
            allow_dirty_development=True,
            source_code_commit="b" * 40,
            dirty_diff_sha256="f" * 64,
        ),
    )
    manifest_path = tmp_path / record.path
    manifest = load_verified_manifest(
        path=manifest_path,
        sensor_preset_path=PRESET_PATH,
        expected_domain_id="morai-ddv2-v1-oracle-command",
        expected_resolved_domain_sha256=manifest_path.parent.name,
        production=False,
    )

    with pytest.raises(IntegrityError, match="dirty|development"):
        MoraiTrainingDataset(manifest, split="validation")
    with pytest.raises(IntegrityError, match="dirty|development"):
        build_dataloaders(manifest, TrainingConfig())


def test_dataloaders_use_separate_generators_sampler_and_validation_b1(
    verified_manifest,
) -> None:
    config = replace(TrainingConfig(), batch_size=2, num_workers=0)
    train, validation, sampler, generator = build_dataloaders(verified_manifest, config)

    assert train.batch_size == 2
    assert train.sampler is sampler
    assert validation.batch_size == 1
    assert isinstance(validation.sampler, SequentialSampler)
    assert train.generator is generator
    assert validation.generator is generator
    assert generator.device.type == "cpu"
    assert sampler._generator is not generator

    validation_batch = next(iter(validation))
    assert validation_batch["features"]["camera_feature"].shape == (
        1,
        3,
        256,
        1024,
    )
    assert validation_batch["trajectory"].shape == (1, 8, 3)


def test_worker_init_seeds_python_and_numpy_from_torch_initial_seed(
    verified_manifest, monkeypatch
) -> None:
    train, _, _, _ = build_dataloaders(
        verified_manifest, replace(TrainingConfig(), num_workers=0)
    )
    python_seeds = []
    numpy_seeds = []
    initial_seed = (1 << 40) + 12345
    monkeypatch.setattr(dataset_module.torch, "initial_seed", lambda: initial_seed)
    monkeypatch.setattr(random, "seed", python_seeds.append)
    monkeypatch.setattr(np.random, "seed", numpy_seeds.append)

    train.worker_init_fn(7)

    expected = initial_seed % (1 << 32)
    assert python_seeds == [expected]
    assert numpy_seeds == [expected]
