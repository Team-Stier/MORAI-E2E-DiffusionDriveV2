from __future__ import annotations

import os
from pathlib import Path


def atomic_write_bytes(output_path: Path, payload: bytes) -> None:
    """Durably publish bytes through a sibling ``.partial`` file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")

    with partial_path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

    os.replace(partial_path, output_path)

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(str(output_path.parent), directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
