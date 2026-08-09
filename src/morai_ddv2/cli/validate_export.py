from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from morai_ddv2.export.validator_preflight import verify_validator_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an exported MORAI DiffusionDriveV2 Validator bundle"
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--validator-root", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    verify_validator_bundle(arguments.profile, arguments.validator_root)
    return 0


__all__ = ("main",)
