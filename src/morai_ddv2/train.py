from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import (
    configure_finetuning,
    load_official_model,
    save_checkpoint_atomic,
    supervised_trajectory_output,
)
from .training.loss import compose_finetune_loss, snapshot_source_parameters


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class MoraiHDF5Dataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"training HDF5 not found: {self.path}")
        expected = {
            "camera": ((3, 256, 1024), np.dtype("uint8")),
            "lidar": ((1, 256, 256), np.dtype("uint8")),
            "status": ((8,), np.dtype("float32")),
            "trajectory": ((8, 3), np.dtype("float32")),
        }
        with h5py.File(self.path, "r") as source:
            missing = tuple(name for name in expected if name not in source)
            if missing:
                raise ValueError(f"training HDF5 is missing datasets: {missing}")
            lengths = set()
            for name, (sample_shape, dtype) in expected.items():
                dataset = source[name]
                if tuple(dataset.shape[1:]) != sample_shape:
                    raise ValueError(f"invalid {name} shape: {dataset.shape}")
                if dataset.dtype != dtype:
                    raise ValueError(f"invalid {name} dtype: {dataset.dtype}")
                lengths.add(dataset.shape[0])
            if len(lengths) != 1 or next(iter(lengths)) == 0:
                raise ValueError("training HDF5 datasets must share one nonzero length")
            self.length = next(iter(lengths))
        self._source: Optional[h5py.File] = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        index = int(index)
        if not 0 <= index < self.length:
            raise IndexError(index)
        if self._source is None:
            self._source = h5py.File(self.path, "r")
        camera = np.array(self._source["camera"][index], dtype=np.float32)
        lidar = np.array(self._source["lidar"][index], dtype=np.float32)
        status = np.array(self._source["status"][index], dtype=np.float32)
        trajectory = np.array(self._source["trajectory"][index], dtype=np.float32)
        if lidar.max(initial=0.0) > 5.0:
            raise ValueError(f"lidar sample {index} exceeds encoded count 5")
        if not np.isfinite(status).all() or not np.isfinite(trajectory).all():
            raise ValueError(f"sample {index} contains a non-finite value")
        camera /= 255.0
        lidar /= 5.0
        return {
            "camera_feature": torch.from_numpy(camera),
            "lidar_feature": torch.from_numpy(lidar),
            "status_feature": torch.from_numpy(status),
            "trajectory": torch.from_numpy(trajectory),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_source"] = None
        return state

    def close(self) -> None:
        source = getattr(self, "_source", None)
        if source is not None:
            source.close()
            self._source = None

    def __del__(self) -> None:
        self.close()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune official DiffusionDriveV2 SEL sensor fusion on MORAI HDF5"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--test-dataset", type=Path)
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument(
        "--checkout", type=Path, default=REPOSITORY_ROOT / "third_party/DiffusionDriveV2"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints/diffusiondrivev2_sel.ckpt",
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints/resnet34.a1_in1k/pytorch_model.bin",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=REPOSITORY_ROOT / "third_party/DiffusionDriveV2/kmeans_navsim_traj_20.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints/morai_ddv2_finetuned.ckpt",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers nonnegative")
    optimizer_values = (args.learning_rate, args.weight_decay, args.grad_clip)
    if not all(math.isfinite(value) for value in optimizer_values):
        parser.error("optimizer arguments must be finite")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error("invalid optimizer argument")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    return args


def _resolve_device(value: str) -> torch.device:
    selected = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected if value == "auto" else value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _trajectory_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    error = torch.linalg.vector_norm(prediction[..., :2] - target[..., :2], dim=-1)
    return error.mean().item(), error[:, -1].mean().item()


def _evaluate(model, loader, device, amp, reference, manifest):
    modes = tuple((module, module.training) for module in model.modules())
    model.eval()
    totals = np.zeros(5, dtype=np.float64)
    sample_count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                camera = batch["camera_feature"].to(device, non_blocking=amp)
                lidar = batch["lidar_feature"].to(device, non_blocking=amp)
                status = batch["status_feature"].to(device, non_blocking=amp)
                trajectory = batch["trajectory"].to(device, non_blocking=amp)
                with torch.cuda.amp.autocast(enabled=amp):
                    trajectory_loss, prediction = supervised_trajectory_output(
                        model,
                        camera,
                        lidar,
                        status,
                        trajectory,
                        deterministic=True,
                    )
                    loss = compose_finetune_loss(
                        trajectory_loss, model, reference, manifest
                    )
                ade, fde = _trajectory_metrics(prediction, trajectory)
                count = camera.shape[0]
                sample_count += count
                totals += count * np.array(
                    [
                        loss.total.item(),
                        loss.trajectory.item(),
                        loss.preservation.item(),
                        ade,
                        fde,
                    ]
                )
    finally:
        for module, training in modes:
            module.training = training
    return {
        "samples": sample_count,
        "total_loss": totals[0] / sample_count,
        "trajectory_loss": totals[1] / sample_count,
        "preservation_loss": totals[2] / sample_count,
        "ade": totals[3] / sample_count,
        "fde": totals[4] / sample_count,
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device)

    dataset = MoraiHDF5Dataset(args.dataset)
    test_dataset = MoraiHDF5Dataset(args.test_dataset) if args.test_dataset else None
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        if test_dataset is not None
        else None
    )
    model, checkpoint = load_official_model(
        args.checkout, args.checkpoint, args.backbone, args.anchor, device
    )
    manifest = configure_finetuning(model)
    reference = snapshot_source_parameters(model, manifest)
    named_parameters = dict(model.named_parameters())
    parameters = [named_parameters[name] for name in manifest.trainable_parameters]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp = not args.no_amp and device.type == "cuda"
    # The official SEL decoder overflows during the first backward pass at
    # GradScaler's 65536 default; 128 retains FP16 range on the pinned model.
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.0)
    metrics_stream = None
    if args.metrics_jsonl:
        metrics_path = args.metrics_jsonl.expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_stream = metrics_path.open("a", encoding="utf-8")

    try:
        for epoch in range(1, args.epochs + 1):
            totals = np.zeros(5, dtype=np.float64)
            sample_count = 0
            for batch in loader:
                camera = batch["camera_feature"].to(device, non_blocking=amp)
                lidar = batch["lidar_feature"].to(device, non_blocking=amp)
                status = batch["status_feature"].to(device, non_blocking=amp)
                trajectory = batch["trajectory"].to(device, non_blocking=amp)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    trajectory_loss, prediction = supervised_trajectory_output(
                        model, camera, lidar, status, trajectory
                    )
                    loss = compose_finetune_loss(
                        trajectory_loss, model, reference, manifest
                    )
                scaler.scale(loss.total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                ade, fde = _trajectory_metrics(prediction.detach(), trajectory)
                count = camera.shape[0]
                sample_count += count
                totals += count * np.array(
                    [
                        loss.total.detach().item(),
                        loss.trajectory.detach().item(),
                        loss.preservation.detach().item(),
                        ade,
                        fde,
                    ]
                )
            record = {
                "epoch": epoch,
                "train": {
                    "samples": sample_count,
                    "total_loss": totals[0] / sample_count,
                    "trajectory_loss": totals[1] / sample_count,
                    "preservation_loss": totals[2] / sample_count,
                    "ade": totals[3] / sample_count,
                    "fde": totals[4] / sample_count,
                },
            }
            if test_loader is not None:
                record["test"] = _evaluate(
                    model, test_loader, device, amp, reference, manifest
                )
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            if metrics_stream is not None:
                metrics_stream.write(line + "\n")
                metrics_stream.flush()
    finally:
        dataset.close()
        if test_dataset is not None:
            test_dataset.close()
        if metrics_stream is not None:
            metrics_stream.close()

    output = save_checkpoint_atomic(args.output, checkpoint, model)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
