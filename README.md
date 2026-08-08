# MORAI-diffusion-drive-v2
this repository fine-tune diffusion drive v2 for MORAI sim

## DDV2 training environment

Training and export run in a dedicated Python 3.9.25 environment. It is
separate from the Python 3.8 ROS preprocessing environment and must not add
ROS system packages to `PYTHONPATH`.

```bash
env -u PYTHONPATH bash scripts/bootstrap_train_env.sh
env -u PYTHONPATH bash scripts/fetch_ddv2_assets.sh
```

The bootstrap synchronizes the pinned Validator DiffusionDriveV2 lock plus
the project's h5py, rfc8785, Ruff, and mypy versions into `.venv-train`.
The asset command uses the pinned Validator profile and fetcher to verify the
official SEL checkpoint, ResNet-34 backbone, trajectory anchors, source
checkout, and generated PDM-free derivative. Verified assets and the emitted
asset manifest remain under the ignored Validator `models/` directory.

Run the environment gates with:

```bash
env -u PYTHONPATH .venv-train/bin/python -m pytest \
  tests/unit/training/test_environment_contract.py -q -m 'not gpu'
nvidia-smi
env -u PYTHONPATH .venv-train/bin/python -m pytest \
  tests/unit/training/test_environment_contract.py -q -m gpu
```
