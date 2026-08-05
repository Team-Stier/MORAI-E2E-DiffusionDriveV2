#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

/usr/bin/python3 -m venv --system-site-packages .venv-ros
.venv-ros/bin/python -m pip install --upgrade pip==24.3.1
.venv-ros/bin/python -m pip install \
  -r requirements/preprocess.txt \
  -r requirements/dev.txt
.venv-ros/bin/python -m pip install --no-deps -e .
