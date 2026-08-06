from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import yaml

from morai_ddv2.config.preset import load_sensor_preset
from morai_ddv2.config.resolver import publish_resolved_domain, resolve_domain
from morai_ddv2.errors import ConfigurationError


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DOMAIN_YAML = (
    _PROJECT_ROOT / "configs" / "domains" / "morai-ddv2-v1-oracle-command.yaml"
)
_DEFAULT_SENSOR_PRESET = _PROJECT_ROOT / "sensor_data" / "0725demo_ros.json"
_DEFAULT_OUTPUT = (
    _PROJECT_ROOT
    / "artifacts"
    / "domain"
    / "morai-ddv2-v1-oracle-command"
    / "resolved-domain.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve the pinned MORAI DDV2 domain")
    parser.add_argument("--domain-yaml", type=Path, default=_DEFAULT_DOMAIN_YAML)
    parser.add_argument("--sensor-preset", type=Path, default=_DEFAULT_SENSOR_PRESET)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--source-commit")
    return parser


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ConfigurationError("unable to determine source commit")
    return completed.stdout.strip()


def _load_domain_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError("unable to load domain YAML: {}".format(path)) from error
    if not isinstance(value, Mapping):
        raise ConfigurationError("domain YAML must contain an object")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    raw_domain = _load_domain_yaml(arguments.domain_yaml)
    source = raw_domain.get("sensor_preset")
    if not isinstance(source, Mapping) or not isinstance(source.get("sha256"), str):
        raise ConfigurationError("domain YAML must pin sensor_preset.sha256")

    preset = load_sensor_preset(arguments.sensor_preset, source["sha256"])
    artifact = resolve_domain(
        preset=preset,
        raw_domain=raw_domain,
        source_commit=arguments.source_commit or _current_commit(),
    )
    publish_resolved_domain(artifact, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
