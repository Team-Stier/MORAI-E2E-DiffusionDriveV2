#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

VALIDATOR_ROOT="$PWD/third_party/E2E-OpenLoop-ROS-Validator"
VALIDATOR_PYTHONPATH="$VALIDATOR_ROOT/src/Validator_pkg/validator_inference/src"
PROFILE_FILE="$VALIDATOR_ROOT/src/Validator_pkg/config/profile.json"
MANIFEST_FILE="$VALIDATOR_ROOT/models/diffusiondrive_v2/asset-manifest.json"
SOURCE_ROOT="$VALIDATOR_ROOT/models/sources/diffusiondrive_v2"

require_clean_checkout() {
  git -C "$1" diff --quiet HEAD -- || {
    echo "tracked checkout is not clean: $1" >&2
    exit 1
  }
}

require_clean_checkout "$PWD/third_party/DiffusionDriveV2"
require_clean_checkout "$VALIDATOR_ROOT"
if [[ -d "$SOURCE_ROOT/.git" ]]; then
  require_clean_checkout "$SOURCE_ROOT"
fi

PYTHONPATH="$VALIDATOR_PYTHONPATH" \
  .venv-train/bin/python \
  third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_inference/scripts/fetch_model_assets.py \
  --model-id diffusiondrive_v2 \
  --profile-file third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json \
  --project-root third_party/E2E-OpenLoop-ROS-Validator

PYTHONPATH="$VALIDATOR_PYTHONPATH" \
  .venv-train/bin/python - "$PROFILE_FILE" "$VALIDATOR_ROOT" "$MANIFEST_FILE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from validator_inference.assets import verify_runtime_assets
from validator_inference.profile_store import load_selected_profile


profile_file = Path(sys.argv[1]).resolve()
validator_root = Path(sys.argv[2]).resolve()
manifest_file = Path(sys.argv[3]).resolve()
selected = load_selected_profile(profile_file, "diffusiondrive_v2")
observed = verify_runtime_assets(selected.data, project_root=validator_root)
expected_manifest = {
    "model_id": "diffusiondrive_v2",
    "source": observed["source"],
    "assets": observed["assets"],
}
published_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
if published_manifest != expected_manifest:
    raise SystemExit(
        "published DiffusionDriveV2 asset manifest does not match fresh Validator verification"
    )
print(manifest_file)
PY

require_clean_checkout "$PWD/third_party/DiffusionDriveV2"
require_clean_checkout "$VALIDATOR_ROOT"
[[ -d "$SOURCE_ROOT/.git" ]] || {
  echo "fetched DiffusionDriveV2 source checkout is missing" >&2
  exit 1
}
require_clean_checkout "$SOURCE_ROOT"
