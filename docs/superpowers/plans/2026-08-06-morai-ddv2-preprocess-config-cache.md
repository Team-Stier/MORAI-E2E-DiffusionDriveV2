# MORAI DDV2 Preprocessing, Config, and Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the finalized GT-rich MORAI ROS1 bags into deterministic, hash-bound HDF5/cache artifacts whose tensors exactly match `morai-ddv2-v1-oracle-command`.

**Architecture:** A read-only sensor preset and versioned domain YAML are resolved once into canonical JSON. A ROS-only two-pass reader indexes messages and materializes only selected payloads; all timestamp, geometry, image, LiDAR, trajectory, and command math lives in a ROS-free NumPy core shared later by training and Validator. Per-bag HDF5 and command sidecars are atomically published first, and a verified manifest is the final dataset commit marker.

**Tech Stack:** Ubuntu 20.04, ROS Noetic, `/usr/bin/python3` 3.8, NumPy, OpenCV, PyYAML, h5py, rfc8785, pytest, Git submodules

## Global Constraints

- Normative contracts are `docs/domain-specs/morai-diffusiondrivev2-v1.md` and `docs/superpowers/specs/2026-08-06-morai-diffusiondrivev2-domain-adaptation-design.md`. If code and prose differ, stop and amend the plan/spec before changing behavior.
- Treat `sensor_data/0725demo_ros.json` as read-only user input. Never create, copy, reformat, overwrite, or stage it. Its required SHA-256 is `b7e69c70cc0b2948c3a5b352eee022be04f8884a1df135697f89bb4bb88263fb`.
- Run ROS-dependent preprocessing with `/usr/bin/python3` 3.8 and `--system-site-packages`; the `/usr/local/bin/python3` 3.14 interpreter cannot import the Noetic stack.
- Begin every project Python module with `from __future__ import annotations`; Python 3.8 code must not evaluate PEP 585/604 annotations at import time.
- Use integer nanoseconds for every selection and interpolation decision. Bag receipt time is provenance only.
- Unit tests under `tests/unit` must not import ROS. ROS imports are confined to `src/morai_ddv2/ros1`.
- The cache generator must refuse dirty source by default. A development override records the dirty diff hash and produces artifacts that the production training loader rejects.
- Never include `.bag.active`, never cross bag boundaries, and never substitute GPS/IMU for missing `/tf` or `/Ego_topic` in v1.
- All file publications use a sibling `.partial`, file `fsync`, `os.replace`, and parent-directory `fsync`.
- Do not stage `.superpowers/`, generated caches, bags, checkpoints, or `sensor_data/0725demo_ros.json` in any commit below.

---

## Dependency and Deliverable Order

This is plan 1 of 3 and must complete before:

1. `2026-08-06-morai-ddv2-training-export.md`
2. `2026-08-06-morai-ddv2-validator-evaluation.md`

Final file topology created by this plan:

```text
.gitignore
.gitmodules
pyproject.toml
requirements/preprocess.txt
requirements/dev.txt
scripts/bootstrap_ros_env.sh
scripts/resolve_domain.py
scripts/inspect_bags.py
scripts/preprocess_bags.py
scripts/verify_cache.py
configs/domains/morai-ddv2-v1-oracle-command.yaml
sensor_data/.gitignore
src/morai_ddv2/
  __init__.py
  errors.py
  atomic.py
  config/{__init__.py,models.py,canonical.py,preset.py,resolver.py}
  preprocess/{__init__.py,models.py,timestamps.py,geometry.py,diagnostics.py,camera.py,lidar.py,trajectory.py,planner.py,materialize.py}
  ros1/{__init__.py,messages.py,bag_reader.py}
  cache/{__init__.py,models.py,hdf5.py,sidecar.py,domain.py,manifest.py,loader.py}
  cli/{__init__.py,resolve_domain.py,inspect_bags.py,preprocess_bags.py,verify_cache.py}
tests/
  conftest.py
  factories.py
  unit/
  integration/
  fixtures/golden/
third_party/DiffusionDriveV2/
third_party/E2E-OpenLoop-ROS-Validator/
```

## Task 1: Pin Repository Inputs and Create the ROS Test Scaffold

**Files:**

- Create: `.gitignore`
- Create: `.gitmodules`
- Create: `sensor_data/.gitignore`
- Create: `pyproject.toml`
- Create: `requirements/preprocess.txt`
- Create: `requirements/dev.txt`
- Create: `scripts/bootstrap_ros_env.sh`
- Create: `src/morai_ddv2/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/test_repository_contract.py`
- Add gitlinks: `third_party/DiffusionDriveV2`, `third_party/E2E-OpenLoop-ROS-Validator`

**Interfaces:**

- `morai_ddv2.__version__ == "0.1.0"`
- `scripts/bootstrap_ros_env.sh` creates `.venv-ros` from `/usr/bin/python3 --system-site-packages`, installs preprocessing/test dependencies, and installs this project editable with `--no-deps`.
- Pinned gitlinks are exactly V2 `1cd12a1e155c34dcc471261835444c8d5587580b` and Validator `09a3b88e19ae3354903ed676a35e02416284cb58`.

- [ ] Create only the package/requirements/bootstrap skeleton and `tests/conftest.py`; its `repo_root` fixture returns `Path(__file__).resolve().parents[1]`. Run `bash scripts/bootstrap_ros_env.sh` so the red test can execute with Python 3.8 and pytest before repository behavior exists.
- [ ] Write the repository-contract test next. It must inspect `git -C third_party/DiffusionDriveV2 rev-parse HEAD` and `git -C third_party/E2E-OpenLoop-ROS-Validator rev-parse HEAD`, assert the two exact commits, assert `sys.version_info[:2] == (3, 8)` in the ROS environment, and assert that `sensor_data/0725demo_ros.json` is ignored by Git but exists and hashes correctly.

```python
def test_sensor_preset_is_read_only_user_input(repo_root: Path) -> None:
    preset = repo_root / "sensor_data/0725demo_ros.json"
    assert preset.is_file()
    assert hashlib.sha256(preset.read_bytes()).hexdigest() == (
        "b7e69c70cc0b2948c3a5b352eee022be04f8884a1df135697f89bb4bb88263fb"
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(preset)], cwd=repo_root, check=False
    )
    assert ignored.returncode == 0
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/test_repository_contract.py -q` before adding ignore rules or submodules. Expected: test collection succeeds under Python 3.8, then assertions FAIL because the ignore rule and pinned submodules do not exist.
- [ ] Add the two submodules and check out their immutable commits.

```bash
git submodule add https://github.com/hustvl/DiffusionDriveV2.git third_party/DiffusionDriveV2
git -C third_party/DiffusionDriveV2 checkout 1cd12a1e155c34dcc471261835444c8d5587580b
git submodule add https://github.com/Team-Stier/E2E-OpenLoop-ROS-Validator.git third_party/E2E-OpenLoop-ROS-Validator
git -C third_party/E2E-OpenLoop-ROS-Validator checkout 09a3b88e19ae3354903ed676a35e02416284cb58
git submodule update --init --recursive
```

- [ ] Add package metadata with `requires-python = ">=3.8,<3.10"`. Keep core dependencies empty so the same package can be installed `--no-deps` into both ROS and training environments; pin runtime packages in the two requirements files.
- [ ] Pin preprocessing packages to `h5py==3.10.0` and `rfc8785==0.1.4`; pin development packages to `pytest==8.3.5`, `pytest-cov==5.0.0`, `ruff==0.12.7`, and `mypy==1.14.1`. NumPy, OpenCV, PyYAML, `rosbag`, and MORAI messages come from the Noetic/system environment.
- [ ] Protect sensor input and generated artifacts with these exact ignore categories: `.venv-ros/`, `.venv-train/`, `cache/`, `cache-repro/`, `artifacts/`, `runs/`, `*.bag`, `*.bag.active`, `.pytest_cache/`, `__pycache__/`, `.superpowers/`, and `sensor_data/*` except `sensor_data/.gitignore`.
- [ ] Rerun `bash scripts/bootstrap_ros_env.sh`, then run `.venv-ros/bin/python -m pytest tests/unit/test_repository_contract.py -q`. Expected: PASS.
- [ ] Commit only scaffold and gitlinks.

```bash
git add -- .gitignore .gitmodules pyproject.toml requirements/preprocess.txt requirements/dev.txt scripts/bootstrap_ros_env.sh sensor_data/.gitignore src/morai_ddv2/__init__.py tests/conftest.py tests/unit/test_repository_contract.py third_party/DiffusionDriveV2 third_party/E2E-OpenLoop-ROS-Validator
git commit -m "build: pin MORAI DDV2 project dependencies"
```

## Task 2: Resolve the Sensor Preset and Domain YAML into Canonical JSON

**Files:**

- Create: `configs/domains/morai-ddv2-v1-oracle-command.yaml`
- Create: `src/morai_ddv2/errors.py`
- Create: `src/morai_ddv2/atomic.py`
- Create: `src/morai_ddv2/config/__init__.py`
- Create: `src/morai_ddv2/config/models.py`
- Create: `src/morai_ddv2/config/canonical.py`
- Create: `src/morai_ddv2/config/preset.py`
- Create: `src/morai_ddv2/config/resolver.py`
- Create: `src/morai_ddv2/cli/__init__.py`
- Create: `src/morai_ddv2/cli/resolve_domain.py`
- Create: `scripts/resolve_domain.py`
- Create: `tests/unit/config/test_canonical.py`
- Create: `tests/unit/config/test_preset.py`
- Create: `tests/unit/config/test_resolver.py`
- Create: `tests/fixtures/golden/resolved-domain.json`

**Interfaces:**

```python
def canonical_json_bytes(value: Mapping[str, Any]) -> bytes
def sha256_bytes(value: bytes) -> str
def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str
def morai_camera_quaternion(*, roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]
def load_sensor_preset(path: Path, expected_sha256: str) -> SensorPreset
def resolve_domain(*, preset: SensorPreset, raw_domain: Mapping[str, Any], source_commit: str) -> ResolvedDomainArtifact
def load_resolved_domain(path: Path) -> ResolvedDomain
def publish_resolved_domain(artifact: ResolvedDomainArtifact, output_path: Path) -> None
```

- [ ] Write tests for RFC 8785 byte stability, lowercase SHA-256, rejection of a modified preset, rejection of nonzero camera roll, and all three canonical camera quaternions. Quaternion comparison must use sign-invariant angular distance rather than component equality.

```python
@pytest.mark.parametrize(
    ("pitch", "yaw", "expected"),
    [
        (2.0, 0.0, (0.0, 0.0174524058, 0.0, 0.9998477101)),
        (10.0, 70.0, (-0.0499904864, -0.0713938102, -0.5713938475, -0.8160349727)),
        (10.0, 290.0, (-0.0499904864, 0.0713938102, -0.5713938475, 0.8160349727)),
    ],
)
def test_morai_camera_rotation_matches_bag_tf(pitch, yaw, expected):
    actual = morai_camera_quaternion(roll_deg=0.0, pitch_deg=pitch, yaw_deg=yaw)
    assert quaternion_angular_distance(actual, expected) < 1.0e-7
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/config -q`. Expected: collection/import FAIL.
- [ ] Encode policy and non-preset contracts in the YAML: required ROS message types, 2 Hz grid and tolerances, camera resize/crop, LiDAR mask/binning, trajectory horizon, ordered command rules, diagnostics, cache schema, and split seed. Resolver-derived topics, frames, physical dimensions/FOV/extrinsics come only from `0725demo_ros.json`; do not duplicate those preset-owned values in YAML.
- [ ] Implement `morai_camera_quaternion` as the Hamilton product `q_y(pitch) * q_z(wrap_degrees(yaw))`, normalize the result, and reject `abs(roll_deg) > 1e-12`.
- [ ] Parse camera IDs 1/2/3 and LiDAR ID 4 from the preset, converting numeric strings once into immutable dataclasses. Record but do not use `focalLengthpixel`; record the FOV/principal-point inconsistency in the resolved document.
- [ ] Build the resolved document without its own hash, serialize it with `rfc8785.dumps`, compute the digest, then expose the digest beside the immutable document. `resolved_domain_sha256` is never included in the bytes it hashes.
- [ ] Implement `publish_resolved_domain` with `.partial`/`fsync`/`os.replace`/parent `fsync`; implement `load_resolved_domain` to recompute and verify the self-excluding hash plus the current source preset hash.
- [ ] Add a thin CLI accepting `--domain-yaml`, `--sensor-preset`, `--output`, and `--source-commit`; its defaults point at the exact project paths but it never writes the preset.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/config -q`. Expected: PASS, including three canonical quaternion cases and bad-hash rejection; these implementation tests do not require the not-yet-generated golden file.
- [ ] Commit the resolver implementation, domain YAML, CLI, and implementation tests without staging the user JSON or the golden file.

```bash
git add -- configs/domains/morai-ddv2-v1-oracle-command.yaml src/morai_ddv2/errors.py src/morai_ddv2/atomic.py src/morai_ddv2/config/__init__.py src/morai_ddv2/config/models.py src/morai_ddv2/config/canonical.py src/morai_ddv2/config/preset.py src/morai_ddv2/config/resolver.py src/morai_ddv2/cli/__init__.py src/morai_ddv2/cli/resolve_domain.py scripts/resolve_domain.py tests/unit/config/test_canonical.py tests/unit/config/test_preset.py tests/unit/config/test_resolver.py
git commit -m "feat: resolve immutable MORAI domain configuration"
```

- [ ] From that clean implementation commit, generate the tracked golden snapshot twice with the actual 40-character HEAD and compare the bytes.

```bash
git diff --quiet
git diff --cached --quiet
test -z "$(git status --short)"
MORAI_CONFIG_COMMIT="$(git rev-parse HEAD)"
test "${#MORAI_CONFIG_COMMIT}" -eq 40
MORAI_GOLDEN_TMP="$(mktemp -d)"
trap 'rm -rf "$MORAI_GOLDEN_TMP"' EXIT
.venv-ros/bin/python scripts/resolve_domain.py \
  --domain-yaml configs/domains/morai-ddv2-v1-oracle-command.yaml \
  --sensor-preset sensor_data/0725demo_ros.json \
  --output tests/fixtures/golden/resolved-domain.json \
  --source-commit "$MORAI_CONFIG_COMMIT"
.venv-ros/bin/python scripts/resolve_domain.py \
  --domain-yaml configs/domains/morai-ddv2-v1-oracle-command.yaml \
  --sensor-preset sensor_data/0725demo_ros.json \
  --output "$MORAI_GOLDEN_TMP/resolved-domain.json" \
  --source-commit "$MORAI_CONFIG_COMMIT"
cmp tests/fixtures/golden/resolved-domain.json "$MORAI_GOLDEN_TMP/resolved-domain.json"
```

Expected: both clean-tree checks and `cmp` exit 0; the golden `source_code_commit` is exactly `MORAI_CONFIG_COMMIT`.

- [ ] Add a regression assertion to `tests/unit/config/test_resolver.py` that reads the golden `source_code_commit`, resolves the same preset/YAML with that commit, and compares the resulting canonical bytes to `tests/fixtures/golden/resolved-domain.json` byte-for-byte.
- [ ] Rerun `.venv-ros/bin/python -m pytest tests/unit/config -q`. Expected: PASS with the clean-HEAD golden byte comparison enabled.
- [ ] Commit the generated golden file separately.

```bash
git add -- tests/unit/config/test_resolver.py tests/fixtures/golden/resolved-domain.json
git commit -m "test: pin resolved MORAI domain golden"
```

## Task 3: Implement Deterministic Timestamp Selection and Duplicate Handling

**Files:**

- Create: `src/morai_ddv2/preprocess/__init__.py`
- Create: `src/morai_ddv2/preprocess/models.py`
- Create: `src/morai_ddv2/preprocess/timestamps.py`
- Create: `tests/factories.py`
- Create: `tests/unit/preprocess/test_timestamps.py`

**Interfaces:**

```python
T = TypeVar("T")

@dataclass(frozen=True)
class StampedRecord(Generic[T]):
    stamp_ns: int
    stable_record_index: int
    value: T

@dataclass(frozen=True)
class Bracket(Generic[T]):
    before: StampedRecord[T]
    after: StampedRecord[T]
    weight: float

def ceil_div(value: int, divisor: int) -> int
def make_grid_ns(first_front_ns: int, last_front_ns: int, period_ns: int) -> tuple[int, ...]
def stable_sort(records: Sequence[StampedRecord[T]]) -> tuple[StampedRecord[T], ...]
def nearest_record(records: Sequence[StampedRecord[T]], target_ns: int, tolerance_ns: int) -> StampedRecord[T] | None
def collapse_duplicate_stamps(records: Sequence[StampedRecord[T]], equivalent: Callable[[T, T], bool]) -> tuple[StampedRecord[T], ...]
def interpolation_bracket(records: Sequence[StampedRecord[T]], target_ns: int, max_gap_ns: int) -> Bracket[T] | None
```

- [ ] Write tests for out-of-order input, exact integer grid origin, equal-distance earlier-stamp tie, same-stamp stable-index tie, inclusive tolerance, exact `[t,t]` bracket with weight `0`, maximum 100 ms bracket, rejection at 100 ms plus 1 ns, and no extrapolation.
- [ ] Write separate duplicate tests: equivalent values collapse to the earliest original record; conflicting values raise `DuplicateRecordConflict` rather than becoming a sample drop.

```python
def test_nearest_tie_prefers_earlier_source_then_record_index():
    records = (
        StampedRecord(1_010, 4, "later"),
        StampedRecord(990, 8, "earlier-second"),
        StampedRecord(990, 3, "earlier-first"),
    )
    chosen = nearest_record(records, target_ns=1_000, tolerance_ns=10)
    assert chosen == StampedRecord(990, 3, "earlier-first")
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_timestamps.py -q`. Expected: import FAIL.
- [ ] Implement selection over sorted tuples with key `(abs(stamp_ns-target_ns), stamp_ns, stable_record_index)` and inclusive `<= tolerance_ns`.
- [ ] Implement exact-match brackets before binary search. For non-exact matches require both sides, compute `weight=(target-before)/(after-before)`, and reject only when `after-before > max_gap_ns`.
- [ ] Keep record indices untouched through sorting/collapse so HDF5 provenance always points to the original topic-local bag read order.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_timestamps.py -q`, then `.venv-ros/bin/python -m pytest tests/unit/preprocess -q`. Expected: both commands PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/preprocess/__init__.py src/morai_ddv2/preprocess/models.py src/morai_ddv2/preprocess/timestamps.py tests/factories.py tests/unit/preprocess/test_timestamps.py
git commit -m "feat: add deterministic timestamp selection"
```

## Task 4: Implement Quaternion, Pose, and TF/Ego Diagnostics

**Files:**

- Create: `src/morai_ddv2/preprocess/geometry.py`
- Create: `src/morai_ddv2/preprocess/diagnostics.py`
- Create: `tests/unit/preprocess/test_geometry.py`
- Create: `tests/unit/preprocess/test_diagnostics.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class Pose:
    translation_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

def normalize_quaternion(q: Sequence[float]) -> np.ndarray
def quaternion_angular_distance(a: Sequence[float], b: Sequence[float]) -> float
def slerp_shortest(a: Sequence[float], b: Sequence[float], weight: float) -> np.ndarray
def interpolate_pose(bracket: Bracket[Pose]) -> Pose
def interpolate_vector(bracket: Bracket[np.ndarray]) -> np.ndarray
def transform_points(points_xyz: np.ndarray, base_from_sensor: Pose) -> np.ndarray
def relative_se2(current: Pose, future: Pose) -> tuple[float, float, float]
def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float
def compute_tf_ego_diagnostic(ego_records, tf_records, max_bracket_ns: int) -> TfEgoDiagnostic
```

- [ ] Write geometry tests for zero-norm rejection, q/−q angular equivalence, `dot < 0` sign correction, the `dot > 0.9995` normalized-linear branch, a general SLERP branch, yaw `179.9° → -179.9°`, rigid point transform, and current-frame SE(2) relative coordinates.
- [ ] Write diagnostics tests with at least 10 comparable pairs; use nearest-rank index `ceil(0.99*N)-1`. For N=100, one isolated high maximum must not move p99 beyond tolerance, while two over-tolerance values must reject. Heading comparison converts Ego degrees to radians and uses circular difference.

```python
def slerp_shortest(a: Sequence[float], b: Sequence[float], weight: float) -> np.ndarray:
    q0 = normalize_quaternion(a)
    q1 = normalize_quaternion(b)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quaternion(q0 + weight * (q1 - q0))
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    return normalize_quaternion(
        math.sin((1.0 - weight) * theta) / math.sin(theta) * q0
        + math.sin(weight * theta) / math.sin(theta) * q1
    )
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_geometry.py tests/unit/preprocess/test_diagnostics.py -q`. Expected: FAIL before implementation.
- [ ] Implement all quaternion computations in float64, normalize inputs and output, and wrap heading to `[-pi, pi)`; cast only final cached trajectory/status arrays to float32.
- [ ] Exclude unbracketable bag-edge Ego stamps from diagnostics and record max as diagnostic only. Add explicit comparable-count cases 0, 9, and 10: counts below 10 raise bag-level `TF_EGO_POSE_MISMATCH`; count 10 proceeds to nearest-rank p99, which also rejects over 0.05 m or 0.5 degrees.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_geometry.py tests/unit/preprocess/test_diagnostics.py -q`, then `.venv-ros/bin/python -m pytest tests/unit/preprocess -q`. Expected: both commands PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/preprocess/geometry.py src/morai_ddv2/preprocess/diagnostics.py tests/unit/preprocess/test_geometry.py tests/unit/preprocess/test_diagnostics.py
git commit -m "feat: add canonical pose interpolation diagnostics"
```

## Task 5: Implement the Three-Camera Adapter

**Files:**

- Create: `src/morai_ddv2/preprocess/camera.py`
- Create: `tests/unit/preprocess/test_camera.py`
- Create: `tests/fixtures/golden/camera-layout.json`

**Interfaces:**

```python
def inspect_jpeg(payload: bytes) -> tuple[int, int]
def decode_jpeg_rgb(payload: bytes, expected_wh: tuple[int, int]) -> np.ndarray
def stitch_camera_rgb(left: np.ndarray, front: np.ndarray, right: np.ndarray, config: CameraAdapterConfig) -> np.ndarray
def camera_model_tensor(camera_rgb: np.ndarray) -> np.ndarray
```

- [ ] First write a coordinate-gradient test that does not use JPEG. Inputs are left/right `[480,640,3]` and front `[720,1280,3]`; assert `INTER_AREA`, resized widths `(341,455,341)`, stitched width 1137, crop `[56,1080)`, output HWC `[256,1024,3]`, and seam boundaries x=285 and x=740.
- [ ] Add three solid-color JPEG fixtures in memory to prove OpenCV BGR decoding is converted to RGB. Reject corrupt JPEG, wrong front resolution, or wrong side resolution.
- [ ] Assert the loader conversion is `np.transpose(HWC,(2,0,1)).astype(np.float32)/255.0`, giving `[3,256,1024]` in `[0,1]`; the leading 3 is RGB channels, not camera count.

```python
def stitch_camera_rgb(left, front, right, config):
    resized = (
        cv2.resize(left, (341, 256), interpolation=cv2.INTER_AREA),
        cv2.resize(front, (455, 256), interpolation=cv2.INTER_AREA),
        cv2.resize(right, (341, 256), interpolation=cv2.INTER_AREA),
    )
    stitched = np.concatenate(resized, axis=1)
    output = np.ascontiguousarray(stitched[:, 56:1080, :])
    if output.shape != (256, 1024, 3) or output.dtype != np.uint8:
        raise TensorContractError("camera_rgb must be uint8 [256,1024,3]")
    return output
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_camera.py -q`. Expected: import FAIL.
- [ ] Implement decode/resize/stitch/crop exactly; do not add padding, warping, blending, EXIF correction, ImageNet normalization, or a 1920×1080 intermediate.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_camera.py -q`, then `.venv-ros/bin/python -m pytest tests/unit/preprocess -q`. Expected: both commands PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/preprocess/camera.py tests/unit/preprocess/test_camera.py tests/fixtures/golden/camera-layout.json
git commit -m "feat: adapt MORAI camera strip for DDV2"
```

## Task 6: Implement LiDAR Transform, 3D Self-Mask, and BEV Histogram

**Files:**

- Create: `src/morai_ddv2/preprocess/lidar.py`
- Create: `tests/unit/preprocess/test_lidar.py`
- Create: `tests/fixtures/golden/lidar-points.json`

**Interfaces:**

```python
def validate_pointcloud_layout(layout: PointCloudLayout) -> None
def build_lidar_count(points_xyzi_lidar: np.ndarray, base_from_lidar: Pose, config: LidarConfig) -> np.ndarray
def lidar_model_tensor(lidar_count: np.ndarray) -> np.ndarray
```

- [ ] Construct a hand-auditable point fixture containing exact zero padding; points on every inclusive self-mask face; a road point with x/y inside but z below 0.8 that must survive; z equal to 0.2 and 100 that must fail the exclusive height filter; x/y at −32, an internal edge, +32, and outside; and six points in one bin to prove clipping at 5.
- [ ] Assert the canonical extrinsic translates LiDAR points by `[1.5,0,1.25]` with identity rotation before masking. Assert histogram axis 0 is x-forward and axis 1 is y-left with no transpose/rotate/flip.

```python
def build_lidar_count(points_xyzi_lidar, base_from_lidar, config):
    xyz = np.asarray(points_xyzi_lidar[:, :3], dtype=np.float64)
    xyz = xyz[~np.all(xyz == 0.0, axis=1)]
    xyz = transform_points(xyz, base_from_lidar)
    inside_self = (
        (0.0 <= xyz[:, 0]) & (xyz[:, 0] <= 2.0)
        & (-1.0 <= xyz[:, 1]) & (xyz[:, 1] <= 1.0)
        & (0.8 <= xyz[:, 2]) & (xyz[:, 2] <= 1.6)
    )
    xyz = xyz[~inside_self]
    xyz = xyz[(0.2 < xyz[:, 2]) & (xyz[:, 2] < 100.0)]
    edges = np.linspace(-32.0, 32.0, 257, dtype=np.float64)
    counts, _ = np.histogramdd(xyz[:, :2], bins=(edges, edges))
    return np.clip(counts, 0, 5).astype(np.uint8)
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_lidar.py -q`. Expected: import FAIL.
- [ ] Validate finite x/y/z and required PointCloud fields/types before building the histogram; intensity is accepted for layout validation but not used as a model feature.
- [ ] Return cache count `[256,256] uint8` in `[0,5]`; loader conversion adds the leading channel and divides by exactly 5 to `[1,256,256] float32`.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_lidar.py -q`, then `.venv-ros/bin/python -m pytest tests/unit/preprocess -q`. Expected: both commands PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/preprocess/lidar.py tests/unit/preprocess/test_lidar.py tests/fixtures/golden/lidar-points.json
git commit -m "feat: build canonical MORAI lidar count BEV"
```

## Task 7: Build Relative Trajectories and Oracle Commands

**Files:**

- Create: `src/morai_ddv2/preprocess/trajectory.py`
- Create: `tests/unit/preprocess/test_trajectory.py`
- Create: `tests/unit/preprocess/test_command.py`
- Create: `tests/fixtures/golden/command-rules.json`

**Interfaces:**

```python
class Command(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    UNKNOWN = 3

def build_relative_trajectory(current_pose: Pose, future_poses: Sequence[Pose]) -> np.ndarray
def classify_oracle_command(trajectory: np.ndarray, rules: CommandRules) -> Command
def build_status(command: Command, velocity_xy: np.ndarray, acceleration_xy: np.ndarray) -> np.ndarray
```

- [ ] Write trajectory tests for eight exact horizons, current-base inverse SE(2), relative heading wrap at ±π, float32 `[8,3]`, and rejection of a nonfinite or wrong-shape pose series.
- [ ] Write command boundary tests in classification order: path length `0.999999`→UNKNOWN and `1.0` continues; `(heading,y)=(0.20,0.50)`→LEFT; `(−0.20,−0.50)`→RIGHT; `(abs heading,abs y)=(0.10,0.50)`→STRAIGHT; mixed signs and threshold gaps→UNKNOWN. A nonfinite trajectory raises `InvalidCommand` and never returns UNKNOWN as fallback.

```python
def classify_oracle_command(trajectory: np.ndarray, rules: CommandRules) -> Command:
    target = np.asarray(trajectory, dtype=np.float64)
    if target.shape != (8, 3) or not np.isfinite(target).all():
        raise InvalidCommand("trajectory must be finite [8,3]")
    xy = np.vstack((np.zeros((1, 2)), target[:, :2]))
    path_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    final_y = float(target[7, 1])
    final_heading = float(target[7, 2])
    if path_length < 1.0:
        return Command.UNKNOWN
    if final_heading >= 0.20 and final_y >= 0.50:
        return Command.LEFT
    if final_heading <= -0.20 and final_y <= -0.50:
        return Command.RIGHT
    if abs(final_heading) <= 0.10 and abs(final_y) <= 0.50:
        return Command.STRAIGHT
    return Command.UNKNOWN
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_trajectory.py tests/unit/preprocess/test_command.py -q`. Expected: FAIL.
- [ ] Implement status as exact one-hot `[LEFT,STRAIGHT,RIGHT,UNKNOWN,vx,vy,ax,ay]`, with unrotated/unscaled body-frame SI values and no clipping or normalization.
- [ ] Serialize the exact ordered mapping/threshold/rule subtree to the golden JSON; later the sidecar hashes these canonical bytes as `command_rule_sha256`.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_trajectory.py tests/unit/preprocess/test_command.py -q`, then `.venv-ros/bin/python -m pytest tests/unit/preprocess -q`. Expected: both commands PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/preprocess/trajectory.py tests/unit/preprocess/test_trajectory.py tests/unit/preprocess/test_command.py tests/fixtures/golden/command-rules.json
git commit -m "feat: generate trajectory targets and oracle commands"
```

## Task 8: Index ROS1 Bags without Retaining Sensor Payloads

**Files:**

- Create: `src/morai_ddv2/ros1/__init__.py`
- Create: `src/morai_ddv2/ros1/messages.py`
- Create: `src/morai_ddv2/ros1/bag_reader.py`
- Create: `tests/unit/ros1/test_messages.py`
- Create: `tests/unit/ros1/test_bag_reader.py`
- Create: `tests/integration/test_real_bag_inventory.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RecordRef:
    topic: str
    stamp_ns: int
    stable_record_index: int

@dataclass(frozen=True)
class BagIndex:
    path: Path
    source_bag_sha256: str
    inventory: BagInventory
    front: tuple[RecordRef, ...]
    left: tuple[RecordRef, ...]
    right: tuple[RecordRef, ...]
    lidar: tuple[RecordRef, ...]
    ego: tuple[StampedRecord[EgoValue], ...]
    poses: tuple[StampedRecord[Pose], ...]
    extrinsics: tuple[StampedRecord[Extrinsic], ...]

def index_bag(path: Path, domain: ResolvedDomain) -> BagIndex
def iter_selected_payloads(path: Path, selected_indices: Mapping[str, set[int]]) -> Iterator[SensorPayload]
def discover_bags(bag_root: Path) -> tuple[Path, ...]
```

- [ ] Test ROS message conversion with minimal fake objects so imports remain lazy. `stamp_ns` is `secs*1_000_000_000+nsecs`; camera stable indices are per topic; `/tf` stable indices count flattened `TransformStamped` entries in message order, not TFMessage envelopes.
- [ ] Test inventory classification: finalized bags with all required topics are eligible; `.bag.active` is excluded; finalized bags missing both `/tf` and `/Ego_topic` are `SENSOR_ONLY_EXCLUDED`; exactly one missing GT topic or a missing core sensor is a bag rejection.
- [ ] Test pass 2 with a fake reader: only the selected topic-local record indices are decoded/materialized, while unrelated JPEG/PointCloud payloads are never copied into `BagIndex`.

```python
def stamp_to_ns(stamp) -> int:
    return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)

def pointcloud_xyz_i(message) -> np.ndarray:
    rows = sensor_msgs.point_cloud2.read_points(
        message, field_names=("x", "y", "z", "intensity"), skip_nans=False
    )
    return np.asarray(tuple(rows), dtype=np.float32)
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/ros1 -q`. Expected: FAIL.
- [ ] Implement pass 1 to validate connection types, Header frames, parseable JPEG resolutions, PointCloud layout, canonical `/tf` extrinsics, duplicate Ego fields, duplicate `map→base_link` transforms, and TF/Ego p99. A JPEG whose header/payload cannot be decoded is a per-record `DECODE_ERROR`, not a bag contract failure; a successfully inspected/decoded image with a noncanonical size is bag-level `RESOLUTION_MISMATCH`. Stream-decode every front JPEG during indexing, discard its pixels immediately, and include only fully decode-valid front records when determining `first_front_ns`; a corrupt early front frame can never shift grid origin. Keep left/right record ordering unchanged so selection occurs before pass-2 decode, and reject the entire bag only for a declared bag-level enum reason.
- [ ] Store `/Ego_topic` only as velocity/acceleration plus diagnostic position/degree heading. Store canonical pose only from `/tf map→base_link`. Normalize leading slashes in frame names.
- [ ] Add an opt-in real-bag inventory test keyed by `MORAI_REAL_BAG`; when unset it skips. When set it records the file size/mtime before and after and proves inspection did not mutate it.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/ros1 -q`. Expected: PASS. Then run the opt-in real-bag inventory test:

```bash
MORAI_REAL_BAG=/home/stier/rosbag/just_drive/s01-sensor-domain__morai__map-kcity__wx-clear__tod-day__scenario-000__20260731T170301KST.bag \
.venv-ros/bin/python -m pytest tests/integration/test_real_bag_inventory.py -m realbag -vv
```

Expected: PASS with required topics, physical resolutions, canonical extrinsics, and unchanged bag stat; the preceding unit command also exits 0.

- [ ] Commit.

```bash
git add -- src/morai_ddv2/ros1/__init__.py src/morai_ddv2/ros1/messages.py src/morai_ddv2/ros1/bag_reader.py tests/unit/ros1/test_messages.py tests/unit/ros1/test_bag_reader.py tests/integration/test_real_bag_inventory.py
git commit -m "feat: index MORAI ROS1 bags in two passes"
```

## Task 9: Plan and Materialize Valid Samples

**Files:**

- Create: `src/morai_ddv2/preprocess/planner.py`
- Create: `src/morai_ddv2/preprocess/materialize.py`
- Create: `src/morai_ddv2/cache/__init__.py`
- Create: `src/morai_ddv2/cache/models.py`
- Create: `tests/unit/preprocess/test_planner.py`
- Create: `tests/unit/preprocess/test_materialize.py`

**Interfaces:**

```python
class SampleDropReason(str, Enum):
    DECODE_ERROR = "DECODE_ERROR"
    CAMERA_SYNC_MISS = "CAMERA_SYNC_MISS"
    LIDAR_SYNC_MISS = "LIDAR_SYNC_MISS"
    EGO_STATE_INTERPOLATION_MISS = "EGO_STATE_INTERPOLATION_MISS"
    POSE_INTERPOLATION_MISS = "POSE_INTERPOLATION_MISS"
    FUTURE_HORIZON_MISS = "FUTURE_HORIZON_MISS"
    NONFINITE_VALUE = "NONFINITE_VALUE"
    INVALID_COMMAND = "INVALID_COMMAND"
    EMPTY_LIDAR = "EMPTY_LIDAR"

def build_sample_plan(bag: BagIndex, domain: ResolvedDomain) -> BagSamplePlan
def materialize_samples(*, bag_path: Path, plan: BagSamplePlan, domain: ResolvedDomain) -> Iterator[CacheSample]
```

- [ ] Test the 2 Hz grid from the first valid front JPEG, front selection within inclusive 50 ms, side selection within 30 ms, LiDAR within 45 ms, Ego/current/future brackets within 100 ms, and eight future requests at `t + (0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0) s`. Assert each missed contract maps to exactly one declared sample-drop reason.
- [ ] Assert `sensor_source_stamp_ns`/indices order is front,left,right,lidar; offsets are left/right/lidar minus front; all current/future bracket stamps, original indices, and float32 weights are retained.

```python
@dataclass(frozen=True)
class CacheSample:
    camera_rgb: np.ndarray
    lidar_count: np.ndarray
    status: np.ndarray
    trajectory: np.ndarray
    timestamp_ns: int
    sensor_source_stamp_ns: tuple[int, int, int, int]
    sensor_source_record_index: tuple[int, int, int, int]
    sensor_sync_offset_ns: tuple[int, int, int]
    ego_bracket_stamp_ns: tuple[int, int]
    ego_bracket_record_index: tuple[int, int]
    ego_interp_weight: float
    pose_bracket_stamp_ns: tuple[int, int]
    pose_bracket_record_index: tuple[int, int]
    pose_interp_weight: float
    future_pose_bracket_stamp_ns: np.ndarray
    future_pose_bracket_record_index: np.ndarray
    future_pose_interp_weight: np.ndarray
    command: Command
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_planner.py tests/unit/preprocess/test_materialize.py -q`. Expected: FAIL.
- [ ] Implement planning over lightweight refs and interpolated Ego/TF values. A single raw sensor record may serve several anchors, but each sample retains actual source stamp/index.
- [ ] Materialize selected JPEG/PointCloud payloads in pass 2, call only the pure camera/LiDAR/trajectory/status functions, validate dtype/shape/range/finite before yielding, and reject a bag that ends with no valid samples.
- [ ] Ensure a corrupt selected sensor payload drops only that anchor as `DECODE_ERROR`; a successfully decoded noncanonical camera size remains bag-level `RESOLUTION_MISMATCH`. Preset resolution, type, frame, extrinsic, duplicate conflict, and TF/Ego p99 failures are all discovered before final cache publication.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_planner.py tests/unit/preprocess/test_materialize.py -q`, then `.venv-ros/bin/python -m pytest tests/unit -q`. Expected: both commands PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/preprocess/planner.py src/morai_ddv2/preprocess/materialize.py src/morai_ddv2/cache/__init__.py src/morai_ddv2/cache/models.py tests/unit/preprocess/test_planner.py tests/unit/preprocess/test_materialize.py
git commit -m "feat: plan and materialize MORAI training samples"
```

## Task 10: Atomically Publish HDF5 and Oracle Command Sidecars

**Files:**

- Create: `src/morai_ddv2/cache/hdf5.py`
- Create: `src/morai_ddv2/cache/sidecar.py`
- Create: `tests/unit/cache/test_hdf5.py`
- Create: `tests/unit/cache/test_sidecar.py`
- Create: `tests/unit/cache/test_artifact_failure.py`

**Interfaces:**

```python
def publish_bag_artifacts(*, cache_root: Path, domain: ResolvedDomain, source_bag_sha256: str, relative_bag_path: str, samples: Iterable[CacheSample], source_code_commit: str) -> BagArtifactRecord
def read_command_sidecar(path: Path, expected_identity: CommandSidecarIdentity) -> Mapping[int, CommandRecord]
def verify_bag_artifact(record: BagArtifactRecord, cache_root: Path) -> None
```

- [ ] Write a two-sample round-trip test that asserts every required HDF5 dataset name, exact dtype/shape, exact attributes, ascending timestamps, sample IDs, and all provenance arrays. Assert `status[:4]` is exactly the one-hot form of the same command record written to JSONL.
- [ ] Assert sidecar records include domain, resolved-domain, preset, bag, and command-rule SHA-256 identities; LEFT/STRAIGHT/RIGHT/UNKNOWN indices are 0/1/2/3; provenance is exactly `future_trajectory_oracle`; duplicate timestamps fail.
- [ ] Inject write, file-fsync, replace, and directory-fsync failures. No final HDF5/sidecar path may appear unless its contents are complete and hash-verifiable; a `.partial` may remain and must be ignored later.

```python
HDF5_DATASETS = {
    "camera_rgb": (np.uint8, (256, 1024, 3)),
    "lidar_count": (np.uint8, (256, 256)),
    "status": (np.float32, (8,)),
    "trajectory": (np.float32, (8, 3)),
    "timestamp_ns": (np.int64, ()),
    "sensor_source_stamp_ns": (np.int64, (4,)),
    "sensor_source_record_index": (np.int64, (4,)),
    "sensor_sync_offset_ns": (np.int64, (3,)),
    "ego_bracket_stamp_ns": (np.int64, (2,)),
    "ego_bracket_record_index": (np.int64, (2,)),
    "ego_interp_weight": (np.float32, ()),
    "pose_bracket_stamp_ns": (np.int64, (2,)),
    "pose_bracket_record_index": (np.int64, (2,)),
    "pose_interp_weight": (np.float32, ()),
    "future_pose_bracket_stamp_ns": (np.int64, (8, 2)),
    "future_pose_bracket_record_index": (np.int64, (8, 2)),
    "future_pose_interp_weight": (np.float32, (8,)),
}
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/cache/test_hdf5.py tests/unit/cache/test_sidecar.py tests/unit/cache/test_artifact_failure.py -q`. Expected: FAIL.
- [ ] Write per-bag paths exactly under `bags/{domain_id}/{resolved_domain_sha256}/{source_bag_sha256}.h5` and `commands/{domain_id}/{resolved_domain_sha256}/{source_bag_sha256}.jsonl`; compute their SHA-256 only after final rename.
- [ ] Use one ordered `CacheSample` stream to populate HDF5 and sidecar so status/command cannot diverge. If streaming would consume the iterator twice, write both in the same loop and verify equal counts before either final publication.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/cache/test_hdf5.py tests/unit/cache/test_sidecar.py tests/unit/cache/test_artifact_failure.py -q`, then `.venv-ros/bin/python -m pytest tests/unit -q`. Expected: both commands PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/cache/hdf5.py src/morai_ddv2/cache/sidecar.py tests/unit/cache/test_hdf5.py tests/unit/cache/test_sidecar.py tests/unit/cache/test_artifact_failure.py
git commit -m "feat: atomically publish MORAI bag cache artifacts"
```

## Task 11: Publish the Dataset Manifest, Deterministic Split, and Verified Loader

**Files:**

- Create: `src/morai_ddv2/cache/manifest.py`
- Create: `src/morai_ddv2/cache/loader.py`
- Create: `src/morai_ddv2/cache/domain.py`
- Create: `tests/unit/cache/test_manifest.py`
- Create: `tests/unit/cache/test_loader.py`
- Create: `tests/unit/cache/test_split.py`

**Interfaces:**

```python
class DirtyOverrideJson(TypedDict):
    allow_dirty_development: Literal[True]
    source_code_commit: str
    dirty_diff_sha256: str

class DatasetManifestRecord(TypedDict):
    record_type: Literal["dataset"]
    schema_version: Literal[1]
    domain_id: str
    resolved_domain_path: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    source_code_commit: str
    split_seed_prefix: Literal["20260806:"]
    train_relative_bag_paths: list[str]
    validation_relative_bag_paths: list[str]
    dirty_override: DirtyOverrideJson | None

class BagManifestRecord(TypedDict):
    record_type: Literal["bag"]
    relative_bag_path: str
    source_bag_sha256: str
    disposition: Literal["ELIGIBLE", "SENSOR_ONLY_EXCLUDED", "REJECTED"]
    split: Literal["train", "validation"] | None
    hdf5_path: str | None
    hdf5_sha256: str | None
    sample_count: int
    sidecar_path: str | None
    sidecar_sha256: str | None
    sidecar_record_count: int
    drop_counts: dict[str, int]
    rejection_code: str | None

class SampleManifestRecord(TypedDict):
    record_type: Literal["sample"]
    sample_id: str
    relative_bag_path: str
    source_bag_sha256: str
    split: Literal["train", "validation"]
    hdf5_path: str
    hdf5_row_index: int
    timestamp_ns: int
    sensor_source_stamp_ns: list[int]
    sensor_source_record_index: list[int]
    sensor_sync_offset_ns: list[int]
    ego_bracket_stamp_ns: list[int]
    ego_bracket_record_index: list[int]
    ego_interp_weight: float
    pose_bracket_stamp_ns: list[int]
    pose_bracket_record_index: list[int]
    pose_interp_weight: float
    future_pose_bracket_stamp_ns: list[list[int]]
    future_pose_bracket_record_index: list[list[int]]
    future_pose_interp_weight: list[float]
    sidecar_path: str
    sidecar_row_index: int
    command_index: Literal[0, 1, 2, 3]
    command_name: Literal["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]
    command_provenance: Literal["future_trajectory_oracle"]

ManifestJsonRecord = Union[
    DatasetManifestRecord,
    BagManifestRecord,
    SampleManifestRecord,
]

@dataclass(frozen=True)
class ModelSample:
    sample_id: str
    source_bag_sha256: str
    relative_bag_path: str
    split: Literal["train", "validation"]
    timestamp_ns: int
    command_name: Literal["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]
    command_index: Literal[0, 1, 2, 3]
    camera_feature: np.ndarray
    lidar_feature: np.ndarray
    status_feature: np.ndarray
    trajectory: np.ndarray

def relative_bag_path(bag_root: Path, bag_path: Path) -> str
def assign_bag_splits(relative_paths: Sequence[str]) -> Mapping[str, Literal["train", "validation"]]
def publish_domain_snapshot(*, cache_root: Path, artifact: ResolvedDomainArtifact) -> DomainSnapshotRecord
def publish_manifest(*, cache_root: Path, domain_snapshot: DomainSnapshotRecord, bags: Sequence[BagArtifactRecord], splits: Mapping[str, str], rejection_report: RejectionReport, dirty_override: DirtyOverride | None) -> ManifestRecord
def load_verified_manifest(path: Path, sensor_preset_path: Path, expected_domain_id: str, expected_resolved_domain_sha256: str, production: bool = True) -> VerifiedManifest
def load_tensor_sample(manifest: VerifiedManifest, sample_id: str) -> ModelSample
```

- [ ] Write split tests using symlink-resolved regular files and UTF-8 POSIX relative paths. Sort by `(SHA256(b"20260806:"+path_bytes).digest(), path_bytes)`; with N=30 assert exactly 24 train and 6 validation bags and no bag appears in both.
- [ ] Write loader tests proving the final manifest is the only commit marker: referenced valid artifacts load; orphan final files and every `.partial` are ignored; missing/hash-mismatched HDF5 or sidecar fails; changed path/hash/split list fails reuse; dirty-override artifacts fail in production mode.
- [ ] Publish the exact canonical snapshot at `domains/{domain_id}/{resolved_domain_sha256}/resolved-domain.json`; its bytes/hash are immutable and the manifest stores its relative path and hash. Loader derives this path from the manifest, verifies its bytes/self-excluding hash and the current `sensor_preset_path` hash, and never re-resolves YAML/preset.
- [ ] Define JSONL record types exactly: one `dataset` record with domain/snapshot/code/split identities; one `bag` record per eligible/excluded/rejected path with source hash, split, artifact paths/hashes/counts and drop stats; and one timestamp-sorted `sample` record per cached row with `sample_id`, source bag/path, split, HDF5 path/row index, timestamp, source stamps/indices, sync offsets, sidecar path, command index/name/provenance. Tests reject unknown/missing fields and duplicate sample IDs.
- [ ] Assert model conversion only at load time: camera NHWC uint8→NCHW float32 `/255`; LiDAR uint8→leading channel float32 `/5`; status/trajectory remain float32; all shapes and finite values are checked.

```python
def split_key(relative_path: str) -> tuple[bytes, bytes]:
    path_bytes = relative_path.encode("utf-8")
    return hashlib.sha256(b"20260806:" + path_bytes).digest(), path_bytes

def assign_bag_splits(relative_paths):
    ordered = sorted(relative_paths, key=split_key)
    train_count = math.floor(0.8 * len(ordered))
    return {path: ("train" if i < train_count else "validation") for i, path in enumerate(ordered)}
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/cache/test_manifest.py tests/unit/cache/test_loader.py tests/unit/cache/test_split.py -q`. Expected: FAIL.
- [ ] Publish `manifests/{domain_id}/{resolved_domain_sha256}/manifest.jsonl` only after re-reading and verifying every per-bag artifact hash/count. Include eligible/excluded/rejected bag inventory, split, sample/drop counts, source paths/hashes, sidecar path/hash/count, source commit, resolved snapshot identity, and dirty override provenance.
- [ ] Recompute current preset hash when loading but never reinterpret raw YAML/preset. Require artifact-bound `resolved-domain.json` and matching source hashes.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/cache/test_manifest.py tests/unit/cache/test_loader.py tests/unit/cache/test_split.py -q`, then `.venv-ros/bin/python -m pytest tests/unit -q`. Expected: both commands PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/cache/manifest.py src/morai_ddv2/cache/loader.py src/morai_ddv2/cache/domain.py tests/unit/cache/test_manifest.py tests/unit/cache/test_loader.py tests/unit/cache/test_split.py
git commit -m "feat: verify MORAI cache manifests and splits"
```

## Task 12: Add Inventory, Preprocessing, and Cache-Verification CLIs

**Files:**

- Create: `src/morai_ddv2/cli/inspect_bags.py`
- Create: `src/morai_ddv2/cli/preprocess_bags.py`
- Create: `src/morai_ddv2/cli/verify_cache.py`
- Create: `scripts/inspect_bags.py`
- Create: `scripts/preprocess_bags.py`
- Create: `scripts/verify_cache.py`
- Create: `tests/unit/cli/test_inspect_bags.py`
- Create: `tests/unit/cli/test_preprocess_bags.py`
- Create: `tests/unit/cli/test_verify_cache.py`
- Create: `tests/integration/test_fake_bag_pipeline.py`
- Create: `tests/integration/test_real_bag_smoke.py`
- Modify: `README.md`

**Interfaces:**

```text
scripts/inspect_bags.py --bag-root PATH --resolved-domain PATH --report PATH
scripts/preprocess_bags.py --bag-root PATH --resolved-domain PATH --cache-root PATH [--allow-dirty-development]
scripts/verify_cache.py --manifest PATH --sensor-preset PATH --report PATH [--compare-manifest PATH]
```

- [ ] Write CLI tests for missing resolved snapshot, wrong preset hash, `.bag.active` exclusion, 30 expected eligible/10 sensor-only classification, rejection report completion, default dirty-tree refusal, and development override provenance.
- [ ] Build a fake bag-reader backend that exercises discovery→index→plan→materialize→HDF5/sidecar→manifest without ROS serialization. Assert two identical runs produce byte-identical canonical JSON/JSONL and tensor-equal HDF5 datasets.
- [ ] Write `tests/unit/cli/test_verify_cache.py` for a read-only verification run over every manifest sample. Assert the atomically published canonical JSON report contains the manifest SHA-256, domain/snapshot identities, total and per-bag sample/drop/command counts, camera/LiDAR global min/max, per-sensor signed sync-offset min/nearest-rank-p50/nearest-rank-p95/max, maximum Ego/current-pose/future-pose bracket gaps, and per-bag TF/Ego position/heading p99. Hash every cache file before and after and assert verification mutates none.
- [ ] Add comparison cases using two independently verified manifests. Require identical resolved snapshot bytes, bag inventory, split assignment, sample IDs/order, sample/drop/command records, command sidecar bytes, and HDF5 dataset names/dtypes/shapes/values/attributes. If HDF5 container bytes differ, allow only the corresponding verified HDF5 file-hash fields to differ in manifest records; every other manifest field must match. Any semantic difference exits nonzero and no success report is published.
- [ ] Make each thin script call its same-named `morai_ddv2.cli` module's `main`; all business logic remains importable. The batch preprocessor continues after bag-level failures to complete the rejection report, publishes no manifest if any expected eligible bag rejects, and exits nonzero. `verify_cache.py` passes `path=args.manifest`, `sensor_preset_path=args.sensor_preset`, `expected_domain_id="morai-ddv2-v1-oracle-command"`, `expected_resolved_domain_sha256=args.manifest.parent.name`, and `production=True` to `load_verified_manifest`, verifies every referenced row/sidecar identity, writes its report with the shared `.partial`/file-fsync/replace/parent-fsync helper, and never opens cache artifacts for writing.
- [ ] Document the exact production commands.

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

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/cli/test_inspect_bags.py tests/unit/cli/test_preprocess_bags.py tests/unit/cli/test_verify_cache.py tests/integration/test_fake_bag_pipeline.py -q` before implementation. Expected: collection/import FAIL for the missing CLI modules.
- [ ] Implement the three CLIs and thin scripts, then rerun `.venv-ros/bin/python -m pytest tests/unit/cli/test_inspect_bags.py tests/unit/cli/test_preprocess_bags.py tests/unit/cli/test_verify_cache.py tests/integration/test_fake_bag_pipeline.py -q`. Expected: PASS.
- [ ] Run the opt-in real bag smoke against the smallest GT-rich bag. Assert at least one valid sample, exact tensor contracts, all sync offsets inside tolerance, sidecar/status identity, and unchanged source size/mtime.

```bash
MORAI_REAL_BAG=/home/stier/rosbag/just_drive/s01-sensor-domain__morai__map-kcity__wx-clear__tod-day__scenario-000__20260731T170301KST.bag \
.venv-ros/bin/python -m pytest tests/integration/test_real_bag_smoke.py -m realbag -vv
```

- [ ] Run static and full tests.

```bash
.venv-ros/bin/python -m ruff check src tests scripts
.venv-ros/bin/python -m mypy src/morai_ddv2
.venv-ros/bin/python -m pytest -q
git diff --check
```

Expected: all commands exit 0; realbag tests may skip only when their explicit environment variable is absent.

- [ ] Commit.

```bash
git add -- README.md src/morai_ddv2/cli/inspect_bags.py src/morai_ddv2/cli/preprocess_bags.py src/morai_ddv2/cli/verify_cache.py scripts/inspect_bags.py scripts/preprocess_bags.py scripts/verify_cache.py tests/unit/cli/test_inspect_bags.py tests/unit/cli/test_preprocess_bags.py tests/unit/cli/test_verify_cache.py tests/integration/test_fake_bag_pipeline.py tests/integration/test_real_bag_smoke.py
git commit -m "feat: add deterministic MORAI preprocessing CLIs"
```

## Task 13: Execute the Full 30-Bag Preprocessing Acceptance Gate

**Files:**

- Generated, not committed: `artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json`
- Generated, not committed: `artifacts/reports/just-drive-inventory.json`
- Generated, not committed: `cache/bags/{domain_id}/{resolved_domain_sha256}/{source_bag_sha256}.h5`
- Generated, not committed: `cache/commands/{domain_id}/{resolved_domain_sha256}/{source_bag_sha256}.jsonl`
- Generated, not committed: `cache/manifests/{domain_id}/{resolved_domain_sha256}/manifest.jsonl`
- Generated, not committed: `artifacts/reports/just-drive-cache-verification.json`
- Generated, not committed: `artifacts/reports/just-drive-cache-repro-compare.json`
- Generated, not committed: `cache-repro/bags/{domain_id}/{resolved_domain_sha256}/{source_bag_sha256}.h5`
- Generated, not committed: `cache-repro/commands/{domain_id}/{resolved_domain_sha256}/{source_bag_sha256}.jsonl`
- Generated, not committed: `cache-repro/domains/{domain_id}/{resolved_domain_sha256}/resolved-domain.json`
- Generated, not committed: `cache-repro/manifests/{domain_id}/{resolved_domain_sha256}/manifest.jsonl`
- Modify only if acceptance exposes a specification defect: the normative spec and this plan, followed by a new reviewed implementation task

**Interfaces:**

- Production run has exactly 30 eligible finalized GT-rich bags, 10 `SENSOR_ONLY_EXCLUDED`, and every `.bag.active` excluded.
- Split contains 24 train and 6 validation bags.
- Every final manifest reference is hash-valid and every cached sample has a matching oracle-command sidecar record.

- [ ] Start from a clean tracked tree and verify the sensor preset hash, submodule commits, ROS Python, free disk, and GPU-independent preprocessing dependencies.

```bash
git status --short
sha256sum sensor_data/0725demo_ros.json
git -C third_party/DiffusionDriveV2 rev-parse HEAD
git -C third_party/E2E-OpenLoop-ROS-Validator rev-parse HEAD
.venv-ros/bin/python -c 'import rosbag, cv2, h5py, numpy, yaml, rfc8785; print("ROS_PREPROCESS_ENV_OK")'
df -h .
```

- [ ] In one Bash shell, resolve, inspect, and preprocess with no dirty override, then prove exactly one final production manifest exists and bind it to `MORAI_MANIFEST`.

```bash
source /opt/ros/noetic/setup.bash
source /home/stier/catkin_ws/devel/setup.bash
MORAI_DOMAIN_ID=morai-ddv2-v1-oracle-command
MORAI_BAG_ROOT=/home/stier/rosbag/just_drive
MORAI_RESOLVED=artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json
MORAI_CACHE_ROOT=cache
.venv-ros/bin/python scripts/resolve_domain.py \
  --domain-yaml configs/domains/morai-ddv2-v1-oracle-command.yaml \
  --sensor-preset sensor_data/0725demo_ros.json \
  --output "$MORAI_RESOLVED" \
  --source-commit "$(git rev-parse HEAD)"
.venv-ros/bin/python scripts/inspect_bags.py \
  --bag-root "$MORAI_BAG_ROOT" \
  --resolved-domain "$MORAI_RESOLVED" \
  --report artifacts/reports/just-drive-inventory.json
.venv-ros/bin/python scripts/preprocess_bags.py \
  --bag-root "$MORAI_BAG_ROOT" \
  --resolved-domain "$MORAI_RESOLVED" \
  --cache-root "$MORAI_CACHE_ROOT"
test -d "$MORAI_CACHE_ROOT/manifests/$MORAI_DOMAIN_ID"
mapfile -t MORAI_MANIFEST_CANDIDATES < <(
  find "$MORAI_CACHE_ROOT/manifests/$MORAI_DOMAIN_ID" -type f -name manifest.jsonl -print | LC_ALL=C sort
)
test "${#MORAI_MANIFEST_CANDIDATES[@]}" -eq 1
MORAI_MANIFEST="${MORAI_MANIFEST_CANDIDATES[0]}"
test -s "$MORAI_MANIFEST"
printf 'MORAI_MANIFEST=%s\n' "$MORAI_MANIFEST"
```

Expected: inventory reports exactly 30 eligible and 10 `SENSOR_ONLY_EXCLUDED`, preprocessing exits 0 with no expected-eligible rejection, and the final `test` commands exit 0.

- [ ] Rediscover the unique production manifest, run the read-only loader over every sample, and atomically publish the complete verification report.

```bash
MORAI_DOMAIN_ID=morai-ddv2-v1-oracle-command
test -d "cache/manifests/$MORAI_DOMAIN_ID"
mapfile -t MORAI_MANIFEST_CANDIDATES < <(
  find "cache/manifests/$MORAI_DOMAIN_ID" -type f -name manifest.jsonl -print | LC_ALL=C sort
)
test "${#MORAI_MANIFEST_CANDIDATES[@]}" -eq 1
MORAI_MANIFEST="${MORAI_MANIFEST_CANDIDATES[0]}"
.venv-ros/bin/python scripts/verify_cache.py \
  --manifest "$MORAI_MANIFEST" \
  --sensor-preset sensor_data/0725demo_ros.json \
  --report artifacts/reports/just-drive-cache-verification.json
```

Expected: exit 0; the report contains all Task 12 aggregates and the verified `manifest_sha256`; cache file hashes are unchanged before and after verification.

- [ ] Require a fresh ignored `cache-repro/`, preprocess the same bags and resolved snapshot into it, and prove exactly one final reproduction manifest exists as `MORAI_REPRO_MANIFEST`.

```bash
MORAI_DOMAIN_ID=morai-ddv2-v1-oracle-command
MORAI_BAG_ROOT=/home/stier/rosbag/just_drive
MORAI_RESOLVED=artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json
test ! -e cache-repro
.venv-ros/bin/python scripts/preprocess_bags.py \
  --bag-root "$MORAI_BAG_ROOT" \
  --resolved-domain "$MORAI_RESOLVED" \
  --cache-root cache-repro
test -d "cache-repro/manifests/$MORAI_DOMAIN_ID"
mapfile -t MORAI_REPRO_MANIFEST_CANDIDATES < <(
  find "cache-repro/manifests/$MORAI_DOMAIN_ID" -type f -name manifest.jsonl -print | LC_ALL=C sort
)
test "${#MORAI_REPRO_MANIFEST_CANDIDATES[@]}" -eq 1
MORAI_REPRO_MANIFEST="${MORAI_REPRO_MANIFEST_CANDIDATES[0]}"
test -s "$MORAI_REPRO_MANIFEST"
printf 'MORAI_REPRO_MANIFEST=%s\n' "$MORAI_REPRO_MANIFEST"
```

Expected: every command exits 0 and preprocessing publishes no final artifact outside `cache-repro/`.

- [ ] Rediscover both unique manifests and run the exact semantic comparison.

```bash
MORAI_DOMAIN_ID=morai-ddv2-v1-oracle-command
mapfile -t MORAI_MANIFEST_CANDIDATES < <(
  find "cache/manifests/$MORAI_DOMAIN_ID" -type f -name manifest.jsonl -print | LC_ALL=C sort
)
mapfile -t MORAI_REPRO_MANIFEST_CANDIDATES < <(
  find "cache-repro/manifests/$MORAI_DOMAIN_ID" -type f -name manifest.jsonl -print | LC_ALL=C sort
)
test "${#MORAI_MANIFEST_CANDIDATES[@]}" -eq 1
test "${#MORAI_REPRO_MANIFEST_CANDIDATES[@]}" -eq 1
MORAI_MANIFEST="${MORAI_MANIFEST_CANDIDATES[0]}"
MORAI_REPRO_MANIFEST="${MORAI_REPRO_MANIFEST_CANDIDATES[0]}"
.venv-ros/bin/python scripts/verify_cache.py \
  --manifest "$MORAI_MANIFEST" \
  --compare-manifest "$MORAI_REPRO_MANIFEST" \
  --sensor-preset sensor_data/0725demo_ros.json \
  --report artifacts/reports/just-drive-cache-repro-compare.json
```

Expected: exit 0 with identical resolved bytes, bag/split/sample/drop/command semantics, sidecar bytes, HDF5 datasets/attributes, and sample IDs. Raw HDF5 bytes and their manifest hash fields may differ only when all semantic HDF5 content compares equal and each run's recorded hash verifies.

- [ ] Confirm the tracked tree is unchanged, the sensor JSON remains ignored and unstaged, and print the verified production manifest SHA-256 for plan 2.

```bash
git diff --quiet
git diff --cached --quiet
git check-ignore -q sensor_data/0725demo_ros.json
test -z "$(git diff --cached --name-only -- sensor_data/0725demo_ros.json)"
MORAI_DOMAIN_ID=morai-ddv2-v1-oracle-command
mapfile -t MORAI_MANIFEST_CANDIDATES < <(
  find "cache/manifests/$MORAI_DOMAIN_ID" -type f -name manifest.jsonl -print | LC_ALL=C sort
)
test "${#MORAI_MANIFEST_CANDIDATES[@]}" -eq 1
MORAI_MANIFEST="${MORAI_MANIFEST_CANDIDATES[0]}"
MORAI_MANIFEST_SHA256="$(sha256sum "$MORAI_MANIFEST" | awk '{print $1}')"
test "${#MORAI_MANIFEST_SHA256}" -eq 64
printf 'MORAI_MANIFEST_SHA256=%s\n' "$MORAI_MANIFEST_SHA256"
```

Expected: every command exits 0. Record the printed digest and the verification-report path for plan 2; do not begin training if any acceptance condition fails.

## Specification Coverage Matrix

| Normative area | Implemented/tested in |
|---|---|
| Source hashes, MORAI Euler, resolved config | Tasks 1–2 |
| Integer timestamps, stable ties, duplicates, brackets | Tasks 3, 8–9 |
| Quaternion/SLERP, SE(2), TF/Ego p99 | Task 4 |
| Mixed camera source sizes and final RGB tensor | Task 5 |
| LiDAR extrinsic, 3D self-mask, histogram | Task 6 |
| `[8,3]` target, command generation, status | Task 7 |
| Topic/type/frame/resolution/extrinsic bag gates | Task 8 |
| Sample drop and bag reject semantics | Tasks 8–9 |
| HDF5 and oracle-command sidecar schema | Task 10 |
| Atomic manifest, deterministic 24/6 split, loader | Task 11 |
| 30-bag execution and reproducibility | Tasks 12–13 |
