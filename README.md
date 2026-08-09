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
the project's h5py, rfc8785, Ruff, and mypy versions and mypy's mandatory
mypy-extensions closure pin into `.venv-train`.
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

## MORAI bag preprocessing

Preprocessing uses the isolated ROS Noetic/Python 3.8 environment. The sensor
preset is the project-local `sensor_data/0725demo_ros.json`; generated domain,
report, and cache artifacts are ignored by Git.

```bash
source /opt/ros/noetic/setup.bash
source /home/stier/catkin_ws/devel/setup.bash
bash scripts/bootstrap_ros_env.sh

.venv-ros/bin/python scripts/resolve_domain.py \
  --domain-yaml configs/domains/morai-ddv2-v1-oracle-command.yaml \
  --sensor-preset sensor_data/0725demo_ros.json \
  --output artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
  --source-commit "$(git rev-parse HEAD)"

.venv-ros/bin/python scripts/inspect_bags.py \
  --bag-root /home/stier/rosbag/just_drive \
  --resolved-domain artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
  --report artifacts/reports/just-drive-inventory.json

.venv-ros/bin/python scripts/preprocess_bags.py \
  --bag-root /home/stier/rosbag/just_drive \
  --resolved-domain artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
  --cache-root cache
```

Production preprocessing refuses a dirty source tree. The development override
is provenance-bound and is not accepted by the production cache loader.

Verify the unique manifest without modifying cache contents:

```bash
MORAI_MANIFEST="$(find cache/manifests/morai-ddv2-v1-oracle-command \
  -mindepth 2 -maxdepth 2 -type f -name manifest.jsonl -print)"
test "$(printf '%s\n' "$MORAI_MANIFEST" | sed '/^$/d' | wc -l)" -eq 1
.venv-ros/bin/python scripts/verify_cache.py \
  --manifest "$MORAI_MANIFEST" \
  --sensor-preset sensor_data/0725demo_ros.json \
  --report artifacts/reports/just-drive-cache-verification.json
```
