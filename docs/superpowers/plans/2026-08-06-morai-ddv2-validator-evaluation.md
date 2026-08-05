# MORAI DDV2 Validator and ROS Bag Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the exported MORAI-adapted DiffusionDriveV2 checkpoint through Team-Stier's pinned ROS Validator with raw bag playback, the exact shared preprocessing contract, oracle-command sidecars, deterministic learned selection, and offline/ROS metric parity.

**Architecture:** The Validator gitlink remains at its approved clean commit. A content-hashed project patch and one authenticated project-package symlink are applied only to an ignored runtime clone to enforce export identity and one-inference-per-anchor policy. A project-owned ROS adapter buffers raw MORAI topics, delegates all math to the shared preprocessing package, publishes the three exact Validator inputs, and drives rosbag backpressure with stamped `geometry_msgs/PointStamped` heartbeats. Each bag runs on a fresh private ROS master, and a result collector converts Validator paths back to canonical base_link trajectories, joins them to manifest GT by bag hash/timestamp, and produces the same metrics as offline validation.

**Tech Stack:** ROS Noetic, project `.venv-ros` Python 3.8 with system ROS packages for project-owned nodes, `/usr/bin/python3` for the existing recorder, Python 3.9.25 model environment for inference, catkin, rosbag, Team-Stier E2E-OpenLoop-ROS-Validator, PyTorch, NumPy, pytest

## Global Constraints

- This is plan 3 of 3. It requires a verified production cache/sidecar/manifest from plan 1 and a preflight-clean exported checkpoint/metadata/profile from plan 2.
- Keep `third_party/E2E-OpenLoop-ROS-Validator` clean at `09a3b88e19ae3354903ed676a35e02416284cb58`. Never commit inside the submodule or move its gitlink to a patched commit.
- Apply the tracked, SHA-256-pinned patch and the one manifest-declared Catkin package symlink only to `.runtime/validator`, which is ignored. Every evaluation report records the base Validator commit, patch SHA-256, authenticated component tree SHA-256, and resulting runtime tree SHA-256.
- Do not launch `validator_pipeline/launch/openloop_validator.launch`, `validator_inference/launch/openloop_inference.launch`, `image_adaptor.py`, `lidar_bev_processor.py`, `opt_feeder_node.py`, or `0716demo.json` for this domain.
- The only model ID is exactly `diffusiondrive_v2`; the only run ID is exactly `morai-ddv2-v1-oracle-command-validation-v1`.
- A missing/mismatched metadata file, sensor preset, resolved domain, training manifest, bag hash, sidecar hash/identity/timestamp, or checkpoint state hash is fatal before model construction or sample publication. Never fall back to STRAIGHT or UNKNOWN.
- ROS camera input is published as HWC `32FC3` `[256,1024,3]` in RGB `[0,1]`; Validator converts it to CHW. LiDAR is `32FC1` `[256,256]` in `[0,1]`. Status is float32 length 8.
- The published header stamp for camera, LiDAR, and DrivingFeature is the selected front anchor `timestamp_ns`, not the latest received camera or wall time.
- Runtime output is the PDM-free learned selector's final `[8,3]` trajectory. Do not use GT-nearest candidate selection. Heading is the final V2 Bézier-derived heading already returned by the derivative.
- Project-owned feature/collector nodes run only through the absolute project `.venv-ros/bin/python`; the existing recorder uses its Catkin `/usr/bin/python3` relay; the inference node runs only through Validator `model_env_exec`. The model child receives only its verified `.ros-overlay` and runtime Catkin generated-message path, never the project `src`, `.venv-ros`, `/usr/lib/python3/dist-packages`, or another activated environment.
- Source only `/opt/ros/noetic/setup.bash` and `.runtime/validator/devel/setup.bash`. `/home/stier/catkin_ws` is not an allowed build or runtime overlay because the runtime clone builds the pinned nested `src/@morai_msgs` gitlink itself.
- Every bag owns a fresh loopback `roscore` on a unique `ROS_MASTER_URI`; no evaluation reuses an ambient ROS master or claims that namespaces isolate Validator's absolute topics.
- Begin every project Python module with `from __future__ import annotations`; ROS-side modules must import under Python 3.8.
- Evaluation modifies no source bag, cache, sidecar, checkpoint, metadata, or resolved-domain artifact.

---

## File Topology

```text
patches/validator/
  manifest.json
  0001-morai-oracle-runtime.patch
scripts/prepare_validator_runtime.py
src/morai_ddv2/validator/
  __init__.py
  runtime_patch.py
  coordinator.py
  ros_messages.py
  prediction.py
  report.py
  launcher.py
ros/morai_ddv2_validator/
  CMakeLists.txt
  package.xml
  launch/morai_oracle_validation.launch
  scripts/morai_oracle_feature_node.py
  scripts/morai_prediction_collector_node.py
tests/
  unit/validator/
  integration/test_validator_patch.py
  integration/test_ros_offline_tensor_parity.py
  integration/test_ros_bag_validation.py
```

## Task 1: Prepare an Ignored Runtime Clone from the Exact Validator Gitlink

**Files:**

- Create: `src/morai_ddv2/validator/__init__.py`
- Create: `src/morai_ddv2/validator/runtime_patch.py`
- Create: `scripts/prepare_validator_runtime.py`
- Create: `patches/validator/manifest.json`
- Create: `tests/unit/validator/test_runtime_patch.py`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**

```python
@dataclass(frozen=True)
class RuntimeLinkedComponent:
    link_path: PurePosixPath
    source_path: PurePosixPath
    source_tree_sha256: str

@dataclass(frozen=True)
class ValidatorRuntimeIdentity:
    base_commit: str
    patch_sha256: str | None
    nested_submodule_commits: tuple[tuple[str, str], ...]
    linked_components: tuple[RuntimeLinkedComponent, ...]
    runtime_tree_sha256: str
    runtime_root: Path

def prepare_validator_runtime(*, submodule_root: Path, runtime_root: Path, patch_manifest: Path) -> ValidatorRuntimeIdentity
def verify_validator_runtime(identity: ValidatorRuntimeIdentity) -> None
```

- [ ] Write tests using a tiny temporary Git repository: refuse a dirty/wrong-base source; clone to a new runtime path; check detached base commit; initialize every declared nested submodule at its exact gitlink commit; apply declared patches only after SHA verification and `git apply --check`, using `git apply --index` so patch-created paths are authenticated index entries rather than unrelated untracked files; reject an already-diverged runtime; leave the tracked source untouched. Give the fixture `.gitmodules` an unreachable HTTPS URL and prove preparation still succeeds from the initialized source checkout without a network call.
- [ ] Start `patches/validator/manifest.json` with exact base commit and an empty ordered patch list. This is the complete valid unpatched state for Task 1.

```json
{
  "validator_repository": "https://github.com/Team-Stier/E2E-OpenLoop-ROS-Validator",
  "base_commit": "09a3b88e19ae3354903ed676a35e02416284cb58",
  "patches": [],
  "runtime_components": []
}
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/validator/test_runtime_patch.py -q`. Expected: import FAIL.
- [ ] Implement runtime creation with noninteractive local Git operations, an explicit destination `.runtime/validator`, detached checkout, recursive submodule update, and no writes to the source gitlink. For `src/@morai_msgs`, set only the runtime clone's local Git config `submodule.src/@morai_msgs.url` to the canonical initialized source checkout, run `git -c protocol.file.allow=always submodule update --init --recursive --checkout` with `GIT_TERMINAL_PROMPT=0`, and verify the nested checkout is clean at the superproject gitlink commit. Never edit tracked `.gitmodules` or fall back to its HTTPS URL. If `.runtime/validator` exists, verify its recorded identity or fail; never silently reset user content.
- [ ] Define `runtime_tree_sha256` as SHA-256 over RFC 8785 canonical JSON containing every tracked runtime path in byte-sorted order with its Git mode and working-tree byte SHA-256, every nested submodule path and checked-out commit, and every authenticated runtime component's link path, literal relative link target, and source tree SHA-256. Exclude ignored build/model/environment outputs. Reject any tracked mutation not produced by the ordered manifest patches, any untracked path other than a manifest-declared symlink, a dirty/wrong-commit nested submodule, a non-symlink component, a changed link target, or a changed component tree hash.
- [ ] Add `.runtime/` and ROS build products to `.gitignore`. The preparer verifies sensor preset and source submodule hashes but never copies or stages them.
- [ ] Run the exact runtime tests and a real unpatched preparation:

```bash
.venv-ros/bin/python -m pytest tests/unit/validator/test_runtime_patch.py -q
.venv-ros/bin/python scripts/prepare_validator_runtime.py \
  --submodule-root third_party/E2E-OpenLoop-ROS-Validator \
  --runtime-root .runtime/validator \
  --patch-manifest patches/validator/manifest.json
git -C third_party/E2E-OpenLoop-ROS-Validator status --short
git -C .runtime/validator rev-parse HEAD
```

Expected: source status empty, runtime base commit exact.

- [ ] Commit.

```bash
git add -- .gitignore README.md patches/validator/manifest.json scripts/prepare_validator_runtime.py src/morai_ddv2/validator/__init__.py src/morai_ddv2/validator/runtime_patch.py tests/unit/validator/test_runtime_patch.py
git commit -m "build: prepare pinned Validator runtime clone"
```

## Task 2: Patch Validator Model Load to Verify the Export Bundle First

**Files:**

- Create: `patches/validator/0001-morai-oracle-runtime.patch`
- Modify: `patches/validator/manifest.json`
- Runtime patch creates: `src/Validator_pkg/validator_inference/src/validator_inference/export_identity.py`
- Runtime patch modifies: `src/Validator_pkg/validator_inference/src/validator_inference/models/diffusiondrive_v2.py`
- Runtime patch creates: `src/Validator_pkg/validator_inference/test/test_export_identity.py`
- Create: `tests/integration/test_validator_patch.py`

**Interfaces inside the applied Validator runtime:**

```python
@dataclass(frozen=True)
class ExportIdentity:
    domain_id: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    training_manifest_sha256: str
    export_checkpoint_sha256: str
    export_state_dict_sha256: str
    raw_schema_sha256: str

def verify_export_bundle(profile: Mapping[str, Any], project_root: Path) -> ExportIdentity | None
```

- [ ] In a temporary clone of exact Validator commit, write failing tests that prove baseline DDV2 profiles without `checkpoint_metadata` retain existing behavior, while a profile declaring `artifact_identity.domain_id == "morai-ddv2-v1-oracle-command"` requires metadata and all new runtime assets.
- [ ] Add failure-order spies: existing `verify_runtime_assets` first; required metadata read; profile checkpoint SHA = metadata export-checkpoint SHA = actual checkpoint file SHA; raw state dict exactly 972 keys; canonical tensor-state hash; exact 972-key name/shape schema hash; domain/resolved/sensor/training-manifest identity; only then derivative construction and 971-key strict load. Include same-count renamed-key and changed-shape failures.
- [ ] Implement both `state_dict_sha256` and `state_schema_sha256` exactly as plan 2 and test them against its shared golden vectors. The patch must not use Validator's sorted-JSON `profile_sha256` as a substitute for RFC 8785 artifact identity.
- [ ] Call `verify_export_bundle` from `DiffusionDriveV2InferenceModel.load()` after existing asset verification and before `_DiffusionDriveSelectorRuntime` construction. Return `None` only for baseline profiles with no MORAI artifact identity; any partially declared MORAI identity is fatal.

```python
def verify_export_bundle(profile, project_root):
    identity = profile.get("artifact_identity")
    if identity is None:
        return None
    if identity.get("domain_id") != "morai-ddv2-v1-oracle-command":
        raise ModelRuntimeError("unsupported DDV2 artifact domain identity")
    roles = resolve_required_roles(profile, project_root)
    metadata = read_metadata(roles["checkpoint_metadata"])
    assert_file_hash_triplet(profile, metadata, roles["checkpoint"])
    raw_state = load_raw_state_dict(roles["checkpoint"])
    assert len(raw_state) == 972
    assert state_dict_sha256(raw_state) == metadata["export_state_dict_sha256"]
    assert state_schema_sha256(raw_state) == metadata["raw_schema_sha256"]
    if identity.get("raw_schema_sha256") != metadata["raw_schema_sha256"]:
        raise ModelRuntimeError("raw DDV2 schema identity mismatch")
    assert_identity_fields(identity, metadata, roles)
    return ExportIdentity.from_mappings(identity, metadata)
```

- [ ] Generate the patch with `git diff --binary` from the temporary exact-base clone, inspect that it touches only the three declared paths, and calculate its lowercase SHA-256. Add that exact filename/hash to the manifest.
- [ ] If `.runtime/validator` exists, inspect its type, Git root, HEAD, and status, then move it intact under a fresh `.runtime/validator-backup.XXXXXX/validator` directory created by `mktemp -d`; never delete or reset it, even when it contains unexpected user state. Create a new `.runtime/validator`, apply the patch through the preparer, and run the exact patch tests in the declared environments.

```bash
mkdir -p .runtime
MORAI_RUNTIME_BACKUP="$(mktemp -d .runtime/validator-backup.XXXXXX)"
if test -e .runtime/validator; then
  test -d .runtime/validator
  test ! -L .runtime/validator
  ls -ld .runtime/validator
  if test -e .runtime/validator/.git; then
    git -C .runtime/validator rev-parse --show-toplevel
    git -C .runtime/validator rev-parse HEAD
    git -C .runtime/validator status --short
  fi
  mv .runtime/validator "$MORAI_RUNTIME_BACKUP/validator"
fi
.venv-ros/bin/python scripts/prepare_validator_runtime.py \
  --submodule-root third_party/E2E-OpenLoop-ROS-Validator \
  --runtime-root .runtime/validator \
  --patch-manifest patches/validator/manifest.json

PYTHONPATH=.runtime/validator/src/Validator_pkg/validator_inference/src \
.venv-train/bin/python -m pytest \
  .runtime/validator/src/Validator_pkg/validator_inference/test/test_export_identity.py -q

.venv-ros/bin/python -m pytest tests/integration/test_validator_patch.py -q
```

Expected: baseline compatibility PASS; every metadata/hash/identity mutation fails before network construction.

- [ ] Confirm the source gitlink is still clean and commit only patch/manifest/project tests.

```bash
git -C third_party/E2E-OpenLoop-ROS-Validator status --short
git add -- patches/validator/0001-morai-oracle-runtime.patch patches/validator/manifest.json tests/integration/test_validator_patch.py
git commit -m "feat: verify MORAI export identity in Validator runtime"
```

## Task 3: Patch the Engine for One Accepted Inference per MORAI Anchor

**Files:**

- Modify: `patches/validator/0001-morai-oracle-runtime.patch`
- Modify: `patches/validator/manifest.json`
- Runtime patch modifies: `src/Validator_pkg/validator_inference/src/validator_inference/engine.py`
- Runtime patch modifies: `src/Validator_pkg/validator_inference/test/test_engine.py`
- Modify: `tests/integration/test_validator_patch.py`

**Interfaces inside the applied Validator runtime:**

- Optional profile field: `inference_policy.reference_stamp_policy` with allowed values `repeat_latest` (default) and `new_stamp_once`.
- Baseline profiles with no field retain current 10 Hz repeat-latest behavior.
- MORAI profile requires `new_stamp_once`; an accepted stamp is never submitted again.

- [ ] Add failing engine tests: same latest stamp is accepted exactly once under `new_stamp_once`; repeated timer ticks emit no duplicate submission; a busy rejection does not mark a new stamp consumed and is retried; a later stamp submits; completion is still published before new submission handling; baseline behavior remains unchanged.

```python
def _reference_stamp_ns(stamp):
    to_nsec = getattr(stamp, "to_nsec", None)
    if callable(to_nsec):
        return int(to_nsec())
    return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)

# In __init__:
self._reference_stamp_policy = str(
    selected_profile.data.get("inference_policy", {}).get(
        "reference_stamp_policy", "repeat_latest"
    )
)
self._last_accepted_reference_ns = None

# After obtaining reference_stamp, before snapshot:
reference_ns = _reference_stamp_ns(reference_stamp)
if (
    self._reference_stamp_policy == "new_stamp_once"
    and reference_ns == self._last_accepted_reference_ns
):
    return

# Only after `submitted = self._worker.try_submit(snapshot)` and `submitted.accepted`:
self._last_accepted_reference_ns = reference_ns
```

- [ ] Validate the policy at construction and reject any other string. Do not add a new model ID or alter worker seed generation.
- [ ] Regenerate the single patch from the exact base so it now contains Task 2 plus Task 3 changes; update the manifest hash and assert its touched-path allowlist.
- [ ] Rebuild runtime and run patched export/engine tests plus original engine tests. Expected: PASS.

```bash
PYTHONPATH=.runtime/validator/src/Validator_pkg/validator_inference/src \
.venv-train/bin/python -m pytest \
  .runtime/validator/src/Validator_pkg/validator_inference/test/test_export_identity.py -q

source /opt/ros/noetic/setup.bash
catkin_make -C .runtime/validator
source .runtime/validator/devel/setup.bash
.venv-ros/bin/python -m pytest \
  .runtime/validator/src/Validator_pkg/validator_inference/test/test_engine.py -q
```

- [ ] Confirm the source gitlink is clean and commit updated patch/manifest/test.

```bash
git add -- patches/validator/0001-morai-oracle-runtime.patch patches/validator/manifest.json tests/integration/test_validator_patch.py
git commit -m "feat: submit each MORAI anchor once in Validator"
```

## Task 4: Create the Project-Owned Catkin Adapter Package

**Files:**

- Create: `ros/morai_ddv2_validator/CMakeLists.txt`
- Create: `ros/morai_ddv2_validator/package.xml`
- Create: `ros/morai_ddv2_validator/launch/morai_oracle_validation.launch`
- Create: `ros/morai_ddv2_validator/scripts/morai_oracle_feature_node.py`
- Create: `ros/morai_ddv2_validator/scripts/morai_prediction_collector_node.py`
- Create: `tests/unit/validator/test_catkin_package.py`
- Modify: `scripts/prepare_validator_runtime.py`
- Modify: `patches/validator/manifest.json`

**Interfaces:**

- Package name: `morai_ddv2_validator`.
- Fixed ROS node names are `/morai_oracle_feature`, `/morai_prediction_collector`, `/inference_engine`, and `/recorder`; launch tests reject anonymous names or remaps of the required absolute inference topics.
- Runtime preparation creates exactly one relative symlink `.runtime/validator/src/morai_ddv2_validator -> ../../../ros/morai_ddv2_validator`. `patches/validator/manifest.json` allowlists that link/source pair and pins the lowercase canonical source-tree SHA-256; no other untracked runtime path is allowed. Recompute and verify the complete runtime tree identity after linking and after every Catkin/model-environment preparation without changing the tracked Validator gitlink.
- The launcher resolves `ros_python=request.project_root / ".venv-ros/bin/python"` and `project_src=request.project_root / "src"` to absolute paths and passes those two arguments. Both project-owned nodes use `launch-prefix="$(arg ros_python)"` and a node-local `PYTHONPATH=$(arg project_src):$(env PYTHONPATH)`. The inference node receives neither value and uses only `launch-prefix="$(find validator_inference)/scripts/model_env_exec"` with canonical `OPENLOOP_PROJECT_ROOT`, `OPENLOOP_MODEL_ENV_ROOT`, and `OPENLOOP_MODEL_ID` values.
- Catkin dependencies: `rospy`, `rosbag`, `geometry_msgs`, `sensor_msgs`, `std_msgs`, `std_srvs`, `tf2_msgs`, `nav_msgs`, `morai_msgs`, `validator_msgs`, `validator_inference`.

- [ ] Write static package tests for valid XML, executable Python node scripts, all declared dependencies including `geometry_msgs`, absence of copied sensor config, and the launch-node allowlist. Assert the four exact non-anonymous node names, both project nodes' `.venv-ros` prefix and project-only `PYTHONPATH`, the inference node's exclusive `model_env_exec` prefix plus its three exact environment variables, and that launch text contains neither `opt_feeder_node.py` nor any old pipeline script/config.
- [ ] Extend runtime preparation tests to assert `lstat` reports one symlink at the exact allowlisted path, `readlink` equals the exact relative target above, source-tree and resulting runtime-tree hashes are stable, and a changed package byte, link target, additional untracked file, or replacement directory fails verification before Catkin runs. Calculate the completed package's canonical tree SHA-256, write that exact lowercase digest to the manifest component entry, and assert the manifest digest in `ValidatorRuntimeIdentity`.
- [ ] Start the scripts as importable thin entrypoints calling package functions; keep ROS behavior out of top-level import so unit tests can use fakes.
- [ ] Create launch arguments for absolute project root, `.venv-ros` interpreter, project `src`, profile, resolved domain, sensor preset, bag, sidecar, training manifest, output directory, recorder directory, model environment root, fixed model/run IDs, checkpoint-manifest identity, and patch/runtime identities. Require `/use_sim_time=true`; pass `~run_id` to the engine and `~checkpoint_manifest` plus the unique recorder directory to the existing recorder.
- [ ] The launch file starts exactly: `morai_oracle_feature_node.py` as `/morai_oracle_feature` and `morai_prediction_collector_node.py` as `/morai_prediction_collector` under the explicit `.venv-ros` interpreter; patched `inference_engine_node.py` as `/inference_engine` under `model_env_exec`; and existing `recorder_node.py` as `/recorder` under its system ROS Catkin relay. It starts no raw bag player; the project launcher owns playback after readiness.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/validator/test_catkin_package.py -q`. Expected: FAIL before package creation, then PASS.
- [ ] Prepare runtime, build, source, and verify ROS package discovery.

```bash
source /opt/ros/noetic/setup.bash
.venv-ros/bin/python scripts/prepare_validator_runtime.py \
  --submodule-root third_party/E2E-OpenLoop-ROS-Validator \
  --runtime-root .runtime/validator \
  --patch-manifest patches/validator/manifest.json
./.runtime/validator/openloop setup diffusiondrive_v2
./.runtime/validator/openloop verify diffusiondrive_v2
catkin_make -C .runtime/validator
source .runtime/validator/devel/setup.bash
rospack find morai_ddv2_validator
```

Expected: setup prepares the model environment, verify reports the active `diffusiondrive_v2` environment valid, `catkin_make` builds the runtime workspace including the project package, and `rospack` resolves the project-owned runtime link.

- [ ] Commit.

```bash
git add -- patches/validator/manifest.json ros/morai_ddv2_validator/CMakeLists.txt ros/morai_ddv2_validator/package.xml ros/morai_ddv2_validator/launch/morai_oracle_validation.launch ros/morai_ddv2_validator/scripts/morai_oracle_feature_node.py ros/morai_ddv2_validator/scripts/morai_prediction_collector_node.py scripts/prepare_validator_runtime.py tests/unit/validator/test_catkin_package.py
git commit -m "feat: add MORAI Validator catkin adapter"
```

## Task 5: Coordinate Raw Topic Buffers against Sidecar Anchors

**Files:**

- Create: `src/morai_ddv2/validator/coordinator.py`
- Create: `tests/unit/validator/test_coordinator.py`
- Create: `tests/unit/validator/test_sidecar_runtime.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RuntimeAnchor:
    timestamp_ns: int
    command: Command
    sidecar_record: CommandRecord

@dataclass(frozen=True)
class RuntimeTensorSample:
    timestamp_ns: int
    camera_chw: np.ndarray
    lidar_chw: np.ndarray
    status: np.ndarray
    source_stamps_ns: tuple[int, int, int, int]
    source_indices: tuple[int, int, int, int]

class AnchorCoordinator:
    def ingest_camera(self, view: str, record: StampedRecord[bytes]) -> None
    def ingest_lidar(self, record: StampedRecord[np.ndarray]) -> None
    def ingest_ego(self, record: StampedRecord[EgoValue]) -> None
    def ingest_tf(self, record: StampedRecord[Extrinsic]) -> None
    def try_build_next(self) -> RuntimeTensorSample | None
    def mark_inference_completed(self, timestamp_ns: int) -> None
```

- [ ] Write tests loading a verified command sidecar and manifest identity before accepting messages. Wrong bag hash, sidecar hash/count, domain/resolved/preset/command-rule identity, duplicate/missing anchor, or unrecognized command index is fatal.
- [ ] Feed out-of-order fake records and prove selection delegates to the shared timestamp/camera/LiDAR/Ego/status functions. The front source must exactly match sidecar anchor; left/right are nearest within 30 ms; LiDAR within 45 ms; Ego is bracket-only within 100 ms.
- [ ] Require canonical sensor extrinsics from `/tf` to match resolved domain before the first sample. Runtime does not recompute future GT trajectory; the verified future-derived sidecar supplies command, and HDF5 supplies GT later for reporting.
- [ ] Allow one published/in-flight anchor only. Do not advance or drop buffered records until matching `inference_completed`; reject completion for another stamp. After completion, prune only records provably older than the next anchor's selection/bracket window.

```python
def try_build_next(self):
    if self._in_flight_timestamp_ns is not None or self._next_index == len(self._anchors):
        return None
    anchor = self._anchors[self._next_index]
    selected = self._selector.select_runtime_inputs(anchor.timestamp_ns, self._buffers)
    if selected is None:
        return None
    command = lookup_oracle_command(
        self._sidecar,
        bag_sha256=self._bag_sha256,
        timestamp_ns=anchor.timestamp_ns,
        resolved_domain_sha256=self._resolved_domain.sha256,
    )
    sample = self._builder.build_runtime_tensor_sample(selected, command)
    self._in_flight_timestamp_ns = anchor.timestamp_ns
    return sample
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/validator/test_coordinator.py tests/unit/validator/test_sidecar_runtime.py -q`. Expected: FAIL.
- [ ] Implement finite/shape/range checks immediately before returning a sample; no UNKNOWN/STRAIGHT fallback and no interpolation/extrapolation different from plan 1.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/validator/test_coordinator.py tests/unit/validator/test_sidecar_runtime.py -q`. Expected: PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/validator/coordinator.py tests/unit/validator/test_coordinator.py tests/unit/validator/test_sidecar_runtime.py
git commit -m "feat: coordinate raw ROS inputs by oracle anchors"
```

## Task 6: Publish Exact Validator Messages with Completion Backpressure

**Files:**

- Create: `src/morai_ddv2/validator/ros_messages.py`
- Modify: `ros/morai_ddv2_validator/scripts/morai_oracle_feature_node.py`
- Create: `tests/unit/validator/test_ros_messages.py`
- Create: `tests/unit/validator/test_feature_node.py`

**Interfaces:**

```python
def camera_image_message(camera_chw: np.ndarray, stamp: Any) -> sensor_msgs.msg.Image
def lidar_image_message(lidar_chw: np.ndarray, stamp: Any) -> sensor_msgs.msg.Image
def driving_feature_message(status: np.ndarray, stamp: Any) -> validator_msgs.msg.DrivingFeature
def rate_control_message(stamp: Any) -> geometry_msgs.msg.PointStamped
```

- [ ] Test message bytes without ROS master. Camera input `[3,256,1024] float32` becomes HWC `32FC3`, height 256, width 1024, step `1024*3*4`, little-endian, RGB; round-trip through Validator `ros_image_to_chw` is exact. LiDAR `[1,256,256]` becomes `32FC1`, step `256*4`, exact round-trip.
- [ ] Test DrivingFeature fields: matching anchor header, `model_id="diffusiondrive_v2"`, eight exact status floats, empty radar/calibration fields, and `source="future_trajectory_oracle"`.
- [ ] Test the rate-control message is exactly `geometry_msgs/PointStamped`, whose first slot/type is `header`/`std_msgs/Header`; its stamp is the supplied nonzero simulated ROS time, its frame is `morai_ddv2_rate_control`, and its point is `(0,0,0)`. This first-field Header contract is mandatory because Noetic `rosbag play` extracts rate-control time from the first serialized Header.
- [ ] Simulate node callbacks. It publishes camera on `/image/diffusiondrive_v2/processed`, LiDAR on `/lidar/diffusiondrive_v2/processed`, then DrivingFeature on `/driving_feature` once for an anchor, all with the same stamp; no next anchor publishes until a matching message on `/inference_metric` with `InferenceMetric(outcome="inference_completed")` arrives.
- [ ] Expose `/morai_ddv2/ready` as `std_srvs/Trigger`. It returns success only after the manifest/sidecar/domain/preset are verified and each of the feature node's camera, LiDAR, and DrivingFeature publishers reports at least one subscriber connection. This is the model-ready boundary: the existing inference node performs synchronous `model.load()` before `InputBuffer.configure()` creates those three subscribers. Recheck all three connections immediately before every sample publication and refuse publication if any is absent.
- [ ] Expose `/morai_ddv2/raw_eof` as `std_srvs/Trigger`. The launcher calls it only after natural zero-status `rosbag play` exit. The feature node marks raw input closed, drains any anchors fully satisfiable from existing buffers one at a time, and exits nonzero with `eof_incomplete` when no anchor is in flight, the next anchor cannot be built, and expected anchors remain.
- [ ] Publish a non-latched `/morai_ddv2/rate_control` `geometry_msgs/PointStamped` heartbeat at 20 Hz only while the coordinator needs raw messages or has just completed an anchor. Do not publish while `rospy.Time.now()` is zero. Stamp every heartbeat with current simulated `rospy.Time.now()`, stop after feature publication while inference is in flight, and resume on the matching completion. This lets `rosbag play --rate-control-topic` pause raw playback during GPU inference without dropping later source windows.
- [ ] Fatal callbacks store the fatal exception, call `rospy.signal_shutdown`, and cause the thin entrypoint to raise `SystemExit(1)` after `rospy.spin()` returns; they are never warnings, zero-status exits, or fallback data. Test the nonzero exit path.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/validator/test_ros_messages.py tests/unit/validator/test_feature_node.py -q`. Expected: FAIL, then PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/validator/ros_messages.py ros/morai_ddv2_validator/scripts/morai_oracle_feature_node.py tests/unit/validator/test_ros_messages.py tests/unit/validator/test_feature_node.py
git commit -m "feat: publish MORAI tensors to Validator with backpressure"
```

## Task 7: Collect Canonical `[8,3]` Predictions from Validator Paths

**Files:**

- Create: `src/morai_ddv2/validator/prediction.py`
- Modify: `ros/morai_ddv2_validator/scripts/morai_prediction_collector_node.py`
- Create: `tests/unit/validator/test_prediction.py`
- Create: `tests/unit/validator/test_prediction_collector.py`

**Interfaces:**

```python
def path_to_base_trajectory(path: nav_msgs.msg.Path, base_from_lidar: Pose) -> np.ndarray
class PredictionCollector:
    def ingest_path(self, path: Any) -> None
    def ingest_metric(self, metric: Any) -> None
    def finalize(self, expected_timestamps_ns: Sequence[int]) -> Path
```

- [ ] Build a round-trip golden test using Validator `PathMessageBuilder` with `T_base_from_lidar=[1.5,0,1.25]`, identity rotation. Convert a canonical finite `[8,3]` NAVSIM trajectory to `nav_msgs/Path` and back; require position/heading within `1e-6`, exact stamp, eight poses, and no sign inversion of y/yaw.
- [ ] `/morai_prediction_collector` subscribes to `/waypoints` and `/inference_metric`, then joins paths to only `InferenceMetric.outcome="inference_completed"` by header nanoseconds. Assert model ID/run ID/profile hash, per-sample seed, one prediction per expected sidecar timestamp, and no unexpected stamp. Busy/not-ready rows remain in Validator CSV but never become predictions.
- [ ] Write timestamp-sorted prediction JSONL to `.partial`, fsync, atomic rename, and parent fsync. Each row includes bag/domain/export/manifest/patch identities, timestamp, seed, float32 trajectory, and path frame.

```python
def path_to_base_trajectory(path, base_from_lidar):
    if path.header.frame_id != "Lidar3D-4" or len(path.poses) != 8:
        raise PredictionContractError("Validator path must contain 8 Lidar3D-4 poses")
    rows = []
    for pose_stamped in path.poses:
        lidar_xyz = position_array(pose_stamped.pose.position)
        base_xyz = transform_point(lidar_xyz, base_from_lidar)
        base_yaw = compose_base_yaw(pose_stamped.pose.orientation, base_from_lidar)
        rows.append((base_xyz[0], base_xyz[1], wrap_to_pi(base_yaw)))
    return np.asarray(rows, dtype=np.float32)
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/validator/test_prediction.py tests/unit/validator/test_prediction_collector.py -q`. Expected: FAIL, then PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/validator/prediction.py ros/morai_ddv2_validator/scripts/morai_prediction_collector_node.py tests/unit/validator/test_prediction.py tests/unit/validator/test_prediction_collector.py
git commit -m "feat: collect canonical Validator predictions"
```

## Task 8: Build a Preflighted ROS Launch and Rate-Controlled Bag Player

**Files:**

- Create: `src/morai_ddv2/validator/launcher.py`
- Modify: `ros/morai_ddv2_validator/launch/morai_oracle_validation.launch`
- Create: `tests/unit/validator/test_launcher.py`
- Create: `tests/unit/validator/test_launch_xml.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RosEvaluationRequest:
    bag: Path
    manifest: Path
    profile: Path
    resolved_domain: Path
    sensor_preset: Path
    output_dir: Path
    runtime_root: Path

def preflight_ros_evaluation(request: RosEvaluationRequest) -> RosEvaluationIdentity
def build_roslaunch_argv(request: RosEvaluationRequest, identity: RosEvaluationIdentity) -> list[str]
def build_rosbag_play_argv(request: RosEvaluationRequest) -> list[str]
def run_ros_evaluation(request: RosEvaluationRequest) -> RosEvaluationResult
def allocate_private_ros_master() -> RosMasterLease
def wait_for_required_ros_graph(master_uri: str, timeout_s: float) -> None
def wait_for_recorder_barrier(csv_dir: Path, expected_timestamps_ns: Sequence[int], timeout_s: float) -> Path
```

- [ ] Test preflight before any ROS process creation: production manifest, bag path/hash/split, sidecar path/hash/count/identity, profile/export metadata/state hash/raw-schema hash, sensor/resolved hashes, exact patch/component/runtime tree identity, ROS package build, fixed IDs, GPU availability, and output nonexistence. Run `[str(request.runtime_root / "openloop"), "verify", "diffusiondrive_v2"]` and require zero exit before allocating a ROS master; `catkin_make` or directory existence is not a model-environment substitute.
- [ ] Test argv allowlist. ROS launch uses absolute project root, `.venv-ros` interpreter, project `src`, `profile_file`, runtime root/model environment root, `model_id=diffusiondrive_v2`, fixed run ID, unique empty recorder directory, bag/sidecar/manifest parameters, checkpoint metadata path, and patch/component/runtime hashes. Assert fixed node names and absolute processed/output topics, and assert forbidden substrings are absent: `openloop_validator.launch`, `openloop_inference.launch`, `image_adaptor.py`, `lidar_bev_processor.py`, `opt_feeder_node.py`, `0716demo.json`, `/home/stier/catkin_ws`.
- [ ] Build bag playback exactly with simulated clock, subscriber readiness, rate-control heartbeat, and required raw topics:

```python
def build_rosbag_play_argv(request):
    return [
        "rosbag", "play", "--clock", "--wait-for-subscribers",
        "--rate-control-topic=/morai_ddv2/rate_control",
        "--rate-control-max-delay=0.2",
        str(request.bag),
        "--topics",
        "/image_jpeg/front/compressed",
        "/image_jpeg/left/compressed",
        "/image_jpeg/right/compressed",
        "/lidar3D", "/Ego_topic", "/tf",
    ]
```

- [ ] Test `geometry_msgs/PointStamped` rate-control integration with a fake simulated clock: zero-time heartbeats are suppressed, stamped heartbeats keep playback within the 0.2 s bag-time window while raw data is needed, heartbeat silence pauses it during inference, and matching completion resumes it.
- [ ] For every single-bag run, allocate an unused loopback port by binding `127.0.0.1:0`, release it immediately before spawning `subprocess.Popen(["roscore", "-p", str(lease.port)], env=child_env, start_new_session=True)`, and retry allocation if `roscore` loses that bind race. Set `ROS_MASTER_URI=f"http://127.0.0.1:{lease.port}/"` and `ROS_HOSTNAME="127.0.0.1"` in `child_env` for `roscore`, roslaunch, rosbag, readiness/service clients, and monitoring code. Never attach to an ambient master.
- [ ] Test `wait_for_required_ros_graph` against fake master states. Before rosbag may start, require `/inference_engine` to subscribe to `/image/diffusiondrive_v2/processed`, `/lidar/diffusiondrive_v2/processed`, and `/driving_feature`; require `/inference_engine` to publish `/waypoints` to `/morai_prediction_collector`; and require its non-latched `/inference_metric` publisher to have `/recorder`, `/morai_prediction_collector`, and `/morai_oracle_feature` as subscribers. A missing or renamed edge must time out before any raw message is played.
- [ ] Launcher order is exact: run all preflights including `openloop verify`; create the unique empty recorder directory; start the private `roscore`; start roslaunch; wait for `/morai_ddv2/ready` to return success, which proves the model loaded and all three processed-topic engine subscriptions exist; wait for the complete input/output graph above, which prevents loss of the first non-latched path/metric; then start rosbag. The feature node independently refuses publication if any processed-input connection disappears.
- [ ] Define player EOF/completion semantics explicitly. A natural zero-status rosbag exit is EOF: call `/morai_ddv2/raw_eof`, allow already buffered or in-flight anchors to drain under the current 120 s anchor deadline, and succeed only if all expected timestamps complete. If no further anchor can be built, the feature node exits with `eof_incomplete`; a nonzero unexpected player exit fails immediately. If all expected anchors complete before natural EOF, send SIGINT to rosbag and treat only that launcher-requested exit as normal. Never keep the player resident past EOF.
- [ ] Keep roslaunch alive after the collector reaches the expected completion count. `record_dir = request.output_dir / "validator_csv"` is unique and empty; wait for exactly one `inference_*.csv`, reopen it until it contains exactly the expected `inference_completed` stamps with the fixed run/model/profile identities, and rely on the existing recorder's per-row flush. Only after this recorder CSV barrier and collector atomic finalization succeed may the launcher stop roslaunch.
- [ ] Start each child in its own process group. On success or failure, send SIGINT and wait 10 s, then SIGTERM and wait 5 s, then SIGKILL only to a still-live owned process group; wait/reap every child and verify the private master port is closed before returning or starting the next bag.
- [ ] Add a hard timeout per anchor based on a configured 120 s wall clock. Timeout is an evaluation failure, not permission to publish the next anchor.
- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/validator/test_launcher.py tests/unit/validator/test_launch_xml.py -q`. Expected: FAIL, then PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/validator/launcher.py ros/morai_ddv2_validator/launch/morai_oracle_validation.launch tests/unit/validator/test_launcher.py tests/unit/validator/test_launch_xml.py
git commit -m "feat: orchestrate rate-controlled ROS validation"
```

## Task 9: Join ROS Predictions to Cache Ground Truth and Report Metrics

**Files:**

- Create: `src/morai_ddv2/validator/report.py`
- Create: `tests/unit/validator/test_report.py`
- Create: `tests/unit/validator/test_offline_ros_comparison.py`

**Interfaces:**

```python
def build_ros_validation_report(*, predictions: Path, validator_csv: Path, manifest: VerifiedManifest, bag_sha256: str, identity: RosEvaluationIdentity) -> ValidationReport
def compare_offline_ros_predictions(offline: Sequence[PredictionRecord], ros: Sequence[PredictionRecord], *, atol: float = 1.0e-5) -> ParityReport
```

- [ ] Test exact join key `(source_bag_sha256,timestamp_ns)` against HDF5 target and command sidecar. Missing/duplicate/unexpected predictions fail; do not silently reduce denominator.
- [ ] Reuse plan 2 metric functions for sample-micro ADE, FDE, circular heading MAE, bag-macro ADE, and per-command results. Report `count=0, metric=null`, never NaN, and label domain/oracle command explicitly.
- [ ] Verify each completed CSV row has the expected run/model/profile/calibration hashes and the seed recomputed from stamp. Record not-ready/busy counts separately; under `new_stamp_once`, each expected timestamp has exactly one completion.
- [ ] Test offline/ROS parity: identical status/input tensors must be exact; selected prediction may differ only by the documented Path float/transform round-trip tolerance `atol=1e-5`; metrics must agree within `1e-6`.
- [ ] Publish canonical JSON report atomically and include all bag/cache/checkpoint/profile/patch identities, command counts, sync/drop statistics, timings, and parity result.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/validator/test_report.py tests/unit/validator/test_offline_ros_comparison.py -q`. Expected: FAIL, then PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/validator/report.py tests/unit/validator/test_report.py tests/unit/validator/test_offline_ros_comparison.py
git commit -m "feat: report MORAI ROS open-loop metrics"
```

## Task 10: Prove Offline and ROS Preprocessing Tensor Parity

**Files:**

- Create: `tests/integration/test_ros_offline_tensor_parity.py`
- Create: `tests/fixtures/golden/ros-runtime-sample.json`
- Modify only if parity exposes a defect: the owning shared preprocessing module plus its exact test among `tests/unit/preprocess/test_timestamps.py`, `tests/unit/preprocess/test_geometry.py`, `tests/unit/preprocess/test_camera.py`, `tests/unit/preprocess/test_lidar.py`, `tests/unit/preprocess/test_planner.py`, and `tests/unit/preprocess/test_materialize.py`; never add a ROS-only alternative implementation

**Interfaces:**

- For one selected real-bag anchor, offline cache and ROS adapter use the same raw source records and produce exact camera, LiDAR, and status tensors.
- Source stamp/index provenance is identical to the HDF5 row.

- [ ] Add a synthetic callback-order integration test with JPEG and PointCloud fixtures. Feed records through the node coordinator in bag order and compare its published decoded arrays to `materialize_samples`; require bit-exact camera/status and LiDAR counts/model float values.
- [ ] Add an opt-in real-bag parity test keyed by `MORAI_REAL_BAG`, `MORAI_CACHE_MANIFEST`, and `MORAI_RESOLVED_DOMAIN`. Choose the first manifest sample from that bag, replay raw messages through coordinator callbacks without ROS master, and compare exact tensors/stamps/indices to HDF5.
- [ ] Assert source bag stat and every artifact hash are unchanged before/after.
- [ ] If shared preprocessing changed, run `.venv-ros/bin/python -m pytest tests/unit/preprocess/test_timestamps.py tests/unit/preprocess/test_geometry.py tests/unit/preprocess/test_camera.py tests/unit/preprocess/test_lidar.py tests/unit/preprocess/test_planner.py tests/unit/preprocess/test_materialize.py -q`. Expected: PASS.
- [ ] Run:

```bash
.venv-ros/bin/python -m pytest tests/integration/test_ros_offline_tensor_parity.py -m 'not realbag' -q

MORAI_MANIFEST="$(find cache/manifests/morai-ddv2-v1-oracle-command -mindepth 2 -maxdepth 2 -type f -name manifest.jsonl -print)"
test "$(printf '%s\n' "$MORAI_MANIFEST" | sed '/^$/d' | wc -l)" -eq 1
MORAI_REAL_BAG=/home/stier/rosbag/just_drive/s01-sensor-domain__morai__map-kcity__wx-clear__tod-day__scenario-000__20260731T170301KST.bag \
MORAI_CACHE_MANIFEST="$MORAI_MANIFEST" \
MORAI_RESOLVED_DOMAIN=artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
.venv-ros/bin/python -m pytest tests/integration/test_ros_offline_tensor_parity.py -m realbag -vv
```

Expected: exact tensor/provenance parity.

- [ ] Commit.

```bash
git add -- tests/integration/test_ros_offline_tensor_parity.py tests/fixtures/golden/ros-runtime-sample.json
git commit -m "test: prove offline ROS tensor parity"
```

## Task 11: Add Single-Bag and Held-Out Evaluation CLIs

**Files:**

- Modify: `src/morai_ddv2/cli/evaluate.py`
- Modify: `scripts/evaluate.py`
- Create: `tests/unit/cli/test_ros_evaluate_cli.py`
- Create: `tests/unit/validator/test_heldout_runner.py`
- Modify: `README.md`

**Interfaces:**

```text
scripts/evaluate.py ros --bag PATH --manifest PATH --resolved-domain PATH --profile PATH --runtime-root PATH --output-dir PATH
scripts/evaluate.py ros-heldout --manifest PATH --resolved-domain PATH --profile PATH --runtime-root PATH --output-dir PATH
```

- [ ] Test CLI rejects train-split bag unless `--allow-train-debug` is explicitly used and labels that run non-validation. `ros-heldout` selects exactly the six manifest validation bags in deterministic relative-path order.
- [ ] Import-test the `ros`/`ros-heldout` command path under `.venv-ros` with Torch deliberately unavailable. CLI dispatch must lazily import the Torch model path only for `offline`; ROS reporting reuses the NumPy-only metric helpers from plan 2.
- [ ] Test each bag gets a distinct output subdirectory, source bag hash, fresh owned `roscore`, and unique `ROS_MASTER_URI`. Assert the prior master/process groups are gone and their port is closed before the next bag starts; failures stop the held-out run with a partial diagnostic but do not publish a misleading aggregate report.
- [ ] Implement `offline` from plan 2 and `ros` to use the same fixed profile/checkpoint/run ID. Implement `ros-heldout` as six sequential single-bag runs, then aggregate sample-micro and bag-macro results only after all complete.
- [ ] Document exact execution:

```bash
MORAI_MANIFEST="$(find cache/manifests/morai-ddv2-v1-oracle-command -mindepth 2 -maxdepth 2 -type f -name manifest.jsonl -print)"
test "$(printf '%s\n' "$MORAI_MANIFEST" | sed '/^$/d' | wc -l)" -eq 1
source /opt/ros/noetic/setup.bash
source .runtime/validator/devel/setup.bash
./.runtime/validator/openloop verify diffusiondrive_v2

.venv-ros/bin/python scripts/evaluate.py ros-heldout \
  --manifest "$MORAI_MANIFEST" \
  --resolved-domain artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
  --profile configs/validator/profile-morai-ddv2-oracle.json \
  --runtime-root .runtime/validator \
  --output-dir artifacts/evaluation/morai-ddv2-v1-oracle-command
```

- [ ] Run `.venv-ros/bin/python -m pytest tests/unit/cli/test_ros_evaluate_cli.py tests/unit/validator/test_heldout_runner.py -q`. Expected: FAIL, then PASS.
- [ ] Commit.

```bash
git add -- README.md src/morai_ddv2/cli/evaluate.py scripts/evaluate.py tests/unit/cli/test_ros_evaluate_cli.py tests/unit/validator/test_heldout_runner.py
git commit -m "feat: evaluate held-out MORAI bags in ROS Validator"
```

## Task 12: Run the Full Patched Validator Regression and ROS Acceptance Gate

**Files:**

- Create: `tests/integration/test_ros_bag_validation.py`
- Generated, not committed: `.runtime/validator/`
- Generated, not committed: `artifacts/evaluation/morai-ddv2-v1-oracle-command/`

**Interfaces:**

- Patched runtime retains all baseline Validator tests and adds MORAI identity/new-stamp behavior.
- Each of six validation bags yields exactly one finite `[8,3]` prediction for every sidecar/cache sample.
- ROS and offline runs share sample seeds and agree within declared tensor/path tolerances.

- [ ] Inspect and preserve any existing runtime in a fresh ignored backup, create the acceptance runtime from scratch, verify patch hash, build Catkin, and run original plus patched Validator tests.

```bash
source /opt/ros/noetic/setup.bash
mkdir -p .runtime
MORAI_RUNTIME_BACKUP="$(mktemp -d .runtime/validator-acceptance-backup.XXXXXX)"
if test -e .runtime/validator; then
  test -d .runtime/validator
  test ! -L .runtime/validator
  ls -ld .runtime/validator
  if test -e .runtime/validator/.git; then
    git -C .runtime/validator rev-parse --show-toplevel
    git -C .runtime/validator rev-parse HEAD
    git -C .runtime/validator status --short
  fi
  mv .runtime/validator "$MORAI_RUNTIME_BACKUP/validator"
fi
.venv-ros/bin/python scripts/prepare_validator_runtime.py \
  --submodule-root third_party/E2E-OpenLoop-ROS-Validator \
  --runtime-root .runtime/validator \
  --patch-manifest patches/validator/manifest.json
./.runtime/validator/openloop setup diffusiondrive_v2
./.runtime/validator/openloop verify diffusiondrive_v2
catkin_make -C .runtime/validator
source .runtime/validator/devel/setup.bash
.venv-ros/bin/python -m pytest \
  .runtime/validator/src/Validator_pkg/validator_inference/test \
  .runtime/validator/src/Validator_pkg/validator_msgs/test \
  --ignore=.runtime/validator/src/Validator_pkg/validator_inference/test/test_export_identity.py -q
PYTHONPATH=.runtime/validator/src/Validator_pkg/validator_inference/src \
.venv-train/bin/python -m pytest \
  .runtime/validator/src/Validator_pkg/validator_inference/test/test_export_identity.py -q
```

- [ ] Return to project root and run every project test in its correct interpreter. ROS-independent training/export tests use `.venv-train`; ROS/config/cache/adapter tests use `.venv-ros`.

```bash
.venv-ros/bin/python -m pytest tests/unit/config tests/unit/preprocess tests/unit/ros1 tests/unit/cache tests/unit/validator -q
.venv-train/bin/python -m pytest tests/unit/training tests/unit/export -q
.venv-ros/bin/python -m pytest tests/integration/test_ros_offline_tensor_parity.py -m 'not realbag' -q
git diff --check
```

- [ ] Require GPU, exported-bundle preflight, exact submodule/patch identities, and clean tracked tree. Run one real validation bag first; assert expected prediction count, no fallback, no duplicate completion, and offline/ROS parity.
- [ ] Run all six held-out bags sequentially. After completion, verify every prediction JSONL/Validator CSV/report hash and exact sample coverage.
- [ ] Compare aggregate offline and ROS reports. Require same seeds, micro/macro/per-command counts, prediction tolerance `1e-5`, metric tolerance `1e-6`, and no NaN.
- [ ] Add the opt-in ROS integration test that checks an existing completed run by hash rather than rerunning it on every unit-test invocation.
- [ ] Confirm source bags, preset, cache, sidecars, manifest, exported checkpoint, metadata, baseline Validator profile, and both gitlinks are unchanged. Commit only the integration test.

```bash
git add -- tests/integration/test_ros_bag_validation.py
git commit -m "test: accept held-out MORAI Validator evaluation"
```

## Specification Coverage Matrix

| Normative area | Implemented/tested in |
|---|---|
| Exact pinned Validator plus traceable runtime patch | Tasks 1–3 |
| Metadata/file/state/domain identity before model load | Task 2 |
| One inference per anchor, unchanged baseline behavior | Task 3 |
| Separate ROS/model environments and MORAI-only launch | Task 4 |
| Shared timestamp/camera/LiDAR/Ego/status/sidecar logic | Tasks 5–6 |
| Exact HWC/CHW tensor messages and no fallback command | Task 6 |
| Learned-selector `[8,3]` Path round-trip | Task 7 |
| Rate-controlled raw bag playback and fail-fast launch | Task 8 |
| ADE/FDE/heading/per-command and identity-rich report | Task 9 |
| Offline/ROS raw tensor parity | Task 10 |
| Six held-out bag orchestration | Task 11 |
| Full Validator regression and end-to-end acceptance | Task 12 |
