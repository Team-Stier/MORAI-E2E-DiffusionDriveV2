#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"
unset PYTHONPATH

/home/stier/.local/bin/uv python install 3.9.25
if [[ ! -x .venv-train/bin/python ]]; then
  /home/stier/.local/bin/uv venv --python 3.9.25 .venv-train
fi
.venv-train/bin/python -c 'from pathlib import Path; import sys; assert sys.version_info[:3] == (3, 9, 25); assert Path(sys.prefix).resolve() == (Path.cwd() / ".venv-train").resolve()'
export UV_TORCH_BACKEND=cu118
/home/stier/.local/bin/uv pip sync --python .venv-train/bin/python requirements/training.txt
/home/stier/.local/bin/uv pip install --python .venv-train/bin/python --no-deps -e .
