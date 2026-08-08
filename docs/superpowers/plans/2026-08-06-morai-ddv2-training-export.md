# MORAI DDV2 Supervised Training and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune only DiffusionDriveV2 SEL camera/LiDAR encoders and cross-modal fusion on the verified MORAI trajectory cache, then export a schema-preserving 972-key checkpoint and identity-bound Validator profile.

**Architecture:** The official SEL checkpoint remains the authoritative deployment container. A PDM-free 971-key derivative is strict-loaded for learning, and a parameter/buffer-free supervised adapter reuses its shared feature graph plus its single trajectory decoder layer. Training updates one exact allowlist and its BatchNorm state with trajectory-only loss; export merges only that mutable state back into the untouched raw 972-key container, publishes external metadata, and verifies the bundle before runtime integration.

**Tech Stack:** Python 3.9.25, PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118, diffusers 0.35.2, official DiffusionDriveV2 SEL source/checkpoint, Validator model environment, NumPy, h5py 3.10.0, rfc8785 0.1.4, pytest, Ruff, mypy

## Global Constraints

- This plan depends on a completed and verified schema-v2 `2026-08-06-morai-ddv2-preprocess-config-cache.md` with exactly 28 HDF5 datasets, 24 non-tensor provenance values repeated by each sample manifest row, and the independent 13-field command sidecar, including the corrected Option D resolved-domain hash, strict selected-front identity, raw TF anchor provenance, independent zero-based TF/Ego epochs, epoch-safe TF/Ego diagnostic report hashes, a production manifest from exactly 30 eligible bags, and a 24/6 bag split. Schema 1, the superseded 27-dataset schema-v2 shape, or ambiguous single-stamp/global-epoch provenance is rejected.
- Use the pinned source gitlinks only: DiffusionDriveV2 `1cd12a1e155c34dcc471261835444c8d5587580b`, Validator `09a3b88e19ae3354903ed676a35e02416284cb58`, and DiffusionDrive reference `9b52ed0ec06b073d82d6f392ab084c7b301c8681` as provenance only.
- Use Python 3.9.25 and the Validator `diffusiondrive_v2.lock`; do not import ROS into this environment and do not mix `/usr/lib/python3/dist-packages` into it.
- Begin every project Python module with `from __future__ import annotations`; the shared package must import under both Python 3.8 and 3.9.
- Current machine inspection on 2026-08-06 found `nvidia-smi` unable to communicate with the driver. CPU unit/schema work may proceed, but no GPU training or finite-forward acceptance may be declared successful until the GPU preflight passes.
- Never call original SEL public `forward`, its PDM/RL constructor path, `TransfuserAgent.compute_loss`, `transfuser_loss`, `TransfuserTargetBuilder`, object/BEV heads, PDM reward, or selector loss during supervised training.
- Do not instantiate the two-layer DiffusionDrive IL model. The pinned V2 SEL derivative has exactly one decoder layer.
- Do not wrap frozen downstream modules in `torch.no_grad()`: gradients must flow through their operations back into the trainable backbone/fusion parameters.
- Never call global `model.train()` after configuring the freeze boundary. Reapply the exact mode boundary at every epoch.
- Best-checkpoint selection remains strictly lower selected sample-micro ADE. `minADE@32` and `minFDE@32` are diagnostic-only metrics over the exact `coarse_top32_post_bezier` shortlist captured from the same public forward; they never select a checkpoint or a deployment trajectory.
- Every completed epoch has one atomic canonical JSON source of truth. JSONL/CSV/PNG curves are rebuilt derived artifacts, never append-only resume state.
- The raw 972-key checkpoint, including `_trajectory_head.vocab`, is immutable except for allowlisted parameter/BatchNorm entries. Metadata is external and never inserted into the checkpoint state dict.
- Generated runs and checkpoints live under ignored `runs/` and `artifacts/`. The machine-specific profile is generated only at `configs/validator/profile-morai-ddv2-oracle.json`, is ignored by the tracked `configs/validator/.gitignore`, and is never committed because it contains absolute paths and hashes that exist only after export. Commits contain code, static training config, the profile ignore rule, and tests only.

---

## File Topology

```text
requirements/training.txt
scripts/bootstrap_train_env.sh
scripts/fetch_ddv2_assets.sh
configs/training/morai-ddv2-encoder-fusion-v1.yaml
configs/validator/.gitignore
# generated and ignored: configs/validator/profile-morai-ddv2-oracle.json
src/morai_ddv2/
  upstream/{__init__.py,ddv2_sel.py}
  training/{__init__.py,contracts.py,dataset.py,sampler.py,supervised_adapter.py,freeze.py,loss.py,rng.py,checkpoint.py,metrics.py,validation.py,telemetry.py,trainer.py}
  export/{__init__.py,schema.py,state_hash.py,metadata.py,exporter.py,validator_profile.py,validator_preflight.py}
  cli/{train.py,evaluate.py,export_checkpoint.py,validate_export.py}
scripts/{train.py,evaluate.py,export_checkpoint.py,validate_export.py}
tests/
  fakes/sel_model.py
  unit/training/
  unit/export/
  integration/test_pinned_sel_adapter.py
  integration/test_pinned_export_validator.py
```

## Task 1: Reproduce the Pinned DDV2 Training Environment and Assets

**Files:**

- Create: `requirements/training.txt`
- Create: `scripts/bootstrap_train_env.sh`
- Create: `scripts/fetch_ddv2_assets.sh`
- Create: `tests/unit/training/test_environment_contract.py`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**

- `.venv-train/bin/python` is CPython 3.9.25.
- Torch is `2.0.1+cu118`, torchvision `0.15.2+cu118`, diffusers `0.35.2`, NumPy `1.23.4`, SciPy `1.13.1`, timm `1.0.28`, OpenCV `4.9.0.80`, PyYAML `6.0.3`, and pytest `8.4.2`, matching the pinned Validator lock. The project-owned additions are exactly h5py `3.10.0`, rfc8785 `0.1.4`, Ruff `0.12.7`, and mypy `1.14.1`.
- `scripts/fetch_ddv2_assets.sh` uses Validator's official profile/fetcher and verifies, rather than fabricates, the SEL checkpoint, ResNet-34 backbone, trajectory anchors, generated PDM-free derivative, and source checkout.

- [ ] Write `requirements/training.txt` as the one executable sync input with these exact bytes. It includes the immutable Validator lock and pins only the four project-owned packages missing from that lock:

```text
-r ../third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_inference/requirements/diffusiondrive_v2.lock
h5py==3.10.0
mypy==1.14.1
rfc8785==0.1.4
ruff==0.12.7
```

- [ ] Implement `scripts/bootstrap_train_env.sh` with `/home/stier/.local/bin/uv python install 3.9.25`, `/home/stier/.local/bin/uv venv --python 3.9.25 .venv-train`, and `/home/stier/.local/bin/uv pip sync --python .venv-train/bin/python requirements/training.txt`; then install this project editable with `/home/stier/.local/bin/uv pip install --python .venv-train/bin/python --no-deps -e .`. Never sync the Validator lock separately, because that would omit or remove the four project-owned packages.
- [ ] Run `bash scripts/bootstrap_train_env.sh`, then confirm `.venv-train/bin/python -c 'import sys; assert sys.version_info[:3] == (3, 9, 25)'` exits 0. This environment-only bootstrap is required before pytest can collect the red repository/asset contract.
- [ ] Write a subprocess-based environment test that asserts interpreter/package versions, exact submodule commits, presence of the generated derivative, and base asset hashes from the baseline Validator profile. Mark the CUDA communication assertion `gpu` so CPU implementation work can run while the host driver is repaired.

```python
def test_training_runtime_versions():
    assert sys.version_info[:3] == (3, 9, 25)
    assert torch.__version__ == "2.0.1+cu118"
    assert torchvision.__version__ == "0.15.2+cu118"
    assert diffusers.__version__ == "0.35.2"

@pytest.mark.gpu
def test_cuda_training_preflight():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_properties(0).total_memory > 0
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_environment_contract.py -q -m 'not gpu'` before implementing asset fetch. Expected: collection succeeds under Python 3.9.25, then FAIL because the verified DDV2 assets and generated derivative are absent.
- [ ] Implement asset fetch with this exact command inside `scripts/fetch_ddv2_assets.sh`, then run Validator asset verification against the emitted manifest:

```bash
PYTHONPATH="$PWD/third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_inference/src" \
  .venv-train/bin/python \
  third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_inference/scripts/fetch_model_assets.py \
  --model-id diffusiondrive_v2 \
  --profile-file third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json \
  --project-root third_party/E2E-OpenLoop-ROS-Validator
```
- [ ] Run:

```bash
bash scripts/bootstrap_train_env.sh
bash scripts/fetch_ddv2_assets.sh
.venv-train/bin/python -m pytest tests/unit/training/test_environment_contract.py -q -m 'not gpu'
nvidia-smi
.venv-train/bin/python -m pytest tests/unit/training/test_environment_contract.py -q -m gpu
```

Expected: non-GPU tests PASS. GPU commands must PASS before Tasks 10 or 14 execute actual training/forward; if they fail, record the host-driver blocker and continue only through CPU-safe tasks.

- [ ] Commit environment code only.

```bash
git add -- .gitignore README.md requirements/training.txt scripts/bootstrap_train_env.sh scripts/fetch_ddv2_assets.sh tests/unit/training/test_environment_contract.py
git commit -m "build: reproduce pinned DDV2 training environment"
```

## Task 2: Strict-Load the Official 972-Key SEL Container into the 971-Key Derivative

**Files:**

- Create: `src/morai_ddv2/upstream/__init__.py`
- Create: `src/morai_ddv2/upstream/ddv2_sel.py`
- Create: `tests/fakes/sel_model.py`
- Create: `tests/unit/training/test_sel_loader.py`
- Create: `tests/integration/test_pinned_sel_loader.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SelAssets:
    v2_checkout: Path
    validator_root: Path
    checkpoint: Path
    image_backbone: Path
    trajectory_anchor: Path

@dataclass(frozen=True)
class RawSelCheckpoint:
    container: Mapping[str, Any]
    raw_state_dict: Mapping[str, torch.Tensor]
    checkpoint_sha256: str

def load_raw_sel_checkpoint(path: Path) -> RawSelCheckpoint
def normalize_sel_key(raw_key: str) -> str
def normalize_sel_runtime_state(raw: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]
def construct_pdm_free_sel_network(assets: SelAssets, device: torch.device) -> nn.Module
def load_pdm_free_sel_network(assets: SelAssets, device: torch.device) -> tuple[nn.Module, RawSelCheckpoint]
```

- [ ] Write fake-state tests for optional `module.` removal followed by `agent._transfuser_model.` removal, duplicate normalized key rejection, exact exclusion of normalized `_trajectory_head.vocab`, raw count 972, runtime count 971, and strict failure for any other missing/unexpected/shape-mismatched key.

```python
def normalize_sel_runtime_state(raw):
    normalized = OrderedDict()
    removed = []
    for raw_key, tensor in raw.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("agent._transfuser_model."):
            key = key[len("agent._transfuser_model."):]
        if key == "_trajectory_head.vocab":
            removed.append(key)
            continue
        if key in normalized:
            raise SelSchemaError(f"duplicate normalized key: {key}")
        normalized[key] = tensor
    if removed != ["_trajectory_head.vocab"]:
        raise SelSchemaError(f"expected one vocab key, got {removed}")
    return normalized
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_sel_loader.py -q`. Expected: import FAIL.
- [ ] Implement `construct_pdm_free_sel_network` with `navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel_inference.V2TransfuserModel` from the Validator-generated PDM-free derivative; set `config.bkb_path` and `config.plan_anchor_path` from verified assets. Never import/construct the original PDM SEL class.
- [ ] Load checkpoint on CPU, preserve its whole top-level container and raw state mapping, enforce exact 972/971 counts, and call pinned Validator symbol `validator_inference.models.diffusiondrive_v2.load_diffusiondrive_selector_state_dict(network, dict(raw_state), checkpoint_path=path)`. That helper performs prefix normalization and the actual derivative `strict=True` load; then move the network to the requested device.
- [ ] Run unit tests and the actual checkpoint CPU integration test:

```bash
.venv-train/bin/python -m pytest tests/unit/training/test_sel_loader.py -q
MORAI_DDV2_VALIDATOR_ROOT="$PWD/third_party/E2E-OpenLoop-ROS-Validator" \
MORAI_DDV2_SEL_CHECKPOINT="$PWD/third_party/E2E-OpenLoop-ROS-Validator/models/diffusiondrive_v2/navsim_v2/diffusiondrivev2_sel.ckpt" \
.venv-train/bin/python -m pytest tests/integration/test_pinned_sel_loader.py -m upstream -q
```

Expected: raw 972, normalized 971, strict load PASS.

- [ ] Commit.

```bash
git add -- src/morai_ddv2/upstream/__init__.py src/morai_ddv2/upstream/ddv2_sel.py tests/fakes/sel_model.py tests/unit/training/test_sel_loader.py tests/integration/test_pinned_sel_loader.py
git commit -m "feat: strict-load official DDV2 SEL state"
```

## Task 3: Enforce the Trainable Parameter and BatchNorm Boundary

**Files:**

- Create: `src/morai_ddv2/training/__init__.py`
- Create: `src/morai_ddv2/training/contracts.py`
- Create: `src/morai_ddv2/training/freeze.py`
- Create: `tests/unit/training/test_freeze.py`

**Interfaces:**

```python
TRAINABLE_PREFIXES = (
    "_backbone.image_encoder.",
    "_backbone.lidar_encoder.",
    "_backbone.transformers.",
    "_backbone.lidar_channel_to_img.",
    "_backbone.img_channel_to_lidar.",
)

@dataclass(frozen=True)
class FreezeManifest:
    trainable_parameters: tuple[str, ...]
    frozen_parameters: tuple[str, ...]
    mutable_state_entries: tuple[str, ...]

def configure_training_boundary(model: nn.Module) -> FreezeManifest
def assert_training_boundary(model: nn.Module, manifest: FreezeManifest) -> None
def extract_mutable_state(model: nn.Module, manifest: FreezeManifest) -> OrderedDict[str, torch.Tensor]
def build_optimizer(model: nn.Module, manifest: FreezeManifest, *, lr: float, weight_decay: float) -> torch.optim.Optimizer
```

- [ ] Test an explicit fake tree containing all five allowed paths, BatchNorm buffers, and frozen c5/up-conv, BEV, embedding, status, trajectory, agent, semantic, and selector branches. Assert optimizer parameter IDs equal the allowlisted `requires_grad=True` IDs exactly once.
- [ ] Assert `model.training` is false globally, only the five exact submodules are in training mode, and every frozen module remains eval. BatchNorm running stats under allowed modules are mutable state but never optimizer parameters.

```python
def configure_training_boundary(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    for module_name, module in model.named_modules():
        if any(module_name == prefix[:-1] or module_name.startswith(prefix) for prefix in TRAINABLE_PREFIXES):
            module.train()
    manifest = build_freeze_manifest(model)
    assert_training_boundary(model, manifest)
    return manifest
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_freeze.py -q`. Expected: FAIL.
- [ ] Implement mutable state as allowlisted named parameters plus `running_mean`, `running_var`, and `num_batches_tracked` buffers belonging to BatchNorm modules under allowed prefixes. No other buffer is exportable.
- [ ] Build AdamW only over manifest trainable parameters. Reapply `configure_training_boundary` at epoch start; prohibit any later `model.train()` call in trainer tests.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_freeze.py -q`. Expected: PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/training/__init__.py src/morai_ddv2/training/contracts.py src/morai_ddv2/training/freeze.py tests/unit/training/test_freeze.py
git commit -m "feat: freeze DDV2 outside encoder fusion boundary"
```

## Task 4: Reproduce the Shared SEL Feature Graph without Agent or BEV Heads

**Files:**

- Create: `src/morai_ddv2/training/supervised_adapter.py`
- Create: `tests/unit/training/test_supervised_context.py`
- Create: `tests/integration/test_pinned_sel_context.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SharedTrajectoryContext:
    trajectory_query: torch.Tensor
    agents_query: torch.Tensor
    cross_bev_feature: torch.Tensor
    bev_spatial_shape: torch.Size
    status_encoding: torch.Tensor
    status_feature: torch.Tensor
    camera_feature: torch.Tensor

def build_shared_trajectory_context(model: nn.Module, features: Mapping[str, torch.Tensor]) -> SharedTrajectoryContext
```

- [ ] Instrument a fake SEL model so module calls are recorded. Assert the adapter calls `_backbone`, `_bev_downscale`, `_status_encoding`, `bev_proj`, and `_tf_decoder`. Give `_keyval_embedding` and `_query_embedding` sentinel `.weight` tensors and make their `forward()` methods raise; assert the sentinels reach the decoder, proving the adapter reads `.weight` directly and never calls either embedding module. Assert it never calls `_bev_semantic_head`, `_agent_head`, selector scorer, PDM, or the public model forward.
- [ ] Assert exact input shapes `[B,3,256,1024]`, `[B,1,256,256]`, `[B,8]`, shared batch size/dtype/device, and finite values. Assert context shapes agree with the pinned graph.

```python
def build_shared_trajectory_context(model, features):
    camera = features["camera_feature"]
    lidar = features["lidar_feature"]
    status = features["status_feature"]
    batch_size = status.shape[0]
    bev_up, bev_concat, _ = model._backbone(camera, lidar)
    spatial = bev_up.shape[2:]
    concat_spatial = bev_concat.shape[2:]
    bev_tokens = model._bev_downscale(bev_concat).flatten(-2, -1).permute(0, 2, 1)
    status_encoding = model._status_encoding(status)
    keyval = torch.cat((bev_tokens, status_encoding[:, None]), dim=1)
    keyval = keyval + model._keyval_embedding.weight[None]
    concat_cross = keyval[:, :-1].permute(0, 2, 1).contiguous().view(batch_size, -1, *concat_spatial)
    concat_cross = F.interpolate(concat_cross, size=spatial, mode="bilinear", align_corners=False)
    cross = torch.cat((concat_cross, bev_up), dim=1)
    cross = model.bev_proj(cross.flatten(-2, -1).permute(0, 2, 1))
    cross = cross.permute(0, 2, 1).contiguous().view(batch_size, -1, *spatial)
    query = model._query_embedding.weight[None].repeat(batch_size, 1, 1)
    query_out = model._tf_decoder(query, keyval)
    trajectory_query, agents_query = query_out.split(model._query_splits, dim=1)
    return SharedTrajectoryContext(trajectory_query, agents_query, cross, spatial, status_encoding, status, camera)
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_supervised_context.py -q`. Expected: FAIL.
- [ ] Implement the adapter as dataclasses/functions only; it must not subclass `nn.Module`, own parameters, register buffers, or clone upstream modules.
- [ ] Run the exact fake and pinned-context commands:

```bash
.venv-train/bin/python -m pytest tests/unit/training/test_supervised_context.py -q
MORAI_DDV2_VALIDATOR_ROOT="$PWD/third_party/E2E-OpenLoop-ROS-Validator" \
  .venv-train/bin/python -m pytest tests/integration/test_pinned_sel_context.py -m upstream -q
```

Expected: PASS, the embedding `forward()` traps remain untouched, and no PDM pool is constructed.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/training/supervised_adapter.py tests/unit/training/test_supervised_context.py tests/integration/test_pinned_sel_context.py
git commit -m "feat: expose PDM-free DDV2 supervised context"
```

## Task 5: Sample One Noisy Anchor Group and Run the Single Decoder Layer

**Files:**

- Modify: `src/morai_ddv2/training/supervised_adapter.py`
- Create: `tests/unit/training/test_noisy_anchors.py`
- Create: `tests/unit/training/test_supervised_adapter.py`
- Create: `tests/integration/test_pinned_sel_adapter.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class NoisyAnchorBatch:
    plan_anchor: torch.Tensor
    decoder_input: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor

@dataclass(frozen=True)
class SupervisedTrajectoryResult:
    trajectory_loss: torch.Tensor
    poses_reg: torch.Tensor
    poses_cls: torch.Tensor
    plan_anchor: torch.Tensor
    timesteps: torch.Tensor

def sample_noisy_plan_anchors(head: nn.Module, batch_size: int, device: torch.device) -> NoisyAnchorBatch
def forward_trajectory_supervised(model: nn.Module, features: Mapping[str, torch.Tensor], target_trajectory: torch.Tensor) -> SupervisedTrajectoryResult
```

- [ ] Seed Torch and test exact anchor operations: repeat `[20,8,2]` to `[B,20,8,2]`; normalize x/50 and y/20; `torch.randint(0,50,(B,))`; `torch.randn_like`; scheduler `add_noise`; `.float()`; clamp `[-1,1]`; denormalize x×50/y×20. Test that timestep 49 is permitted.
- [ ] Test decoder contracts: `len(diff_decoder.layers)==1`; regression source `[B,20,8,3]`; classification source `[B,1,20]`; squeeze only axis 1 to `[B,20]`; `targets={"trajectory": target}` with target `[B,8,3]`; call the existing `head.loss_computer` once. Before the call assert `head.loss_computer.cls_loss_weight == 10.0` and `head.loss_computer.reg_loss_weight == 8.0`; any other upstream config is a hard contract failure.
- [ ] Assert the trajectory head, embeddings, and frozen decoder participate in autograd even though their parameters do not receive gradients; at least one allowed backbone parameter must receive a finite nonzero gradient.

```python
def sample_noisy_plan_anchors(head, batch_size, device):
    anchor = head.plan_anchor.to(device=device, dtype=torch.float32)
    anchor = anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    normalized = anchor.clone()
    normalized[..., 0] /= 50.0
    normalized[..., 1] /= 20.0
    timesteps = torch.randint(0, 50, (batch_size,), device=device, dtype=torch.long)
    noise = torch.randn_like(normalized)
    noisy = head.diffusion_scheduler.add_noise(normalized, noise, timesteps).float()
    noisy = noisy.clamp(-1.0, 1.0)
    decoder_input = noisy.clone()
    decoder_input[..., 0] *= 50.0
    decoder_input[..., 1] *= 20.0
    return NoisyAnchorBatch(anchor, decoder_input, timesteps, noise)
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_noisy_anchors.py tests/unit/training/test_supervised_adapter.py -q`. Expected: FAIL.
- [ ] Reproduce the pinned head's sine position embedding, `plan_anchor_encoder`, timestep embedding/MLP, and `diff_decoder` call exactly. Do not create a second group/layer, do not use predicted argmax for loss, and do not manually recompute the 10/8 inner weights.
- [ ] Run unit tests, then pinned integration on GPU when available:

```bash
.venv-train/bin/python -m pytest tests/unit/training/test_noisy_anchors.py tests/unit/training/test_supervised_adapter.py -q
CUDA_VISIBLE_DEVICES=0 MORAI_DDV2_VALIDATOR_ROOT="$PWD/third_party/E2E-OpenLoop-ROS-Validator" \
.venv-train/bin/python -m pytest tests/integration/test_pinned_sel_adapter.py -m upstream -q
```

Expected: finite scalar trajectory loss and exact output shapes.

- [ ] Commit.

```bash
git add -- src/morai_ddv2/training/supervised_adapter.py tests/unit/training/test_noisy_anchors.py tests/unit/training/test_supervised_adapter.py tests/integration/test_pinned_sel_adapter.py
git commit -m "feat: run one-layer DDV2 trajectory supervision"
```

## Task 6: Add Source-Preservation and Exact Finetune Loss Accounting

**Files:**

- Create: `src/morai_ddv2/training/loss.py`
- Create: `tests/unit/training/test_loss.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class FinetuneLoss:
    total: torch.Tensor
    trajectory: torch.Tensor
    preservation: torch.Tensor

def snapshot_source_parameters(model: nn.Module, manifest: FreezeManifest) -> Mapping[str, torch.Tensor]
def source_preservation_loss(model: nn.Module, reference: Mapping[str, torch.Tensor], manifest: FreezeManifest) -> torch.Tensor
def compose_finetune_loss(trajectory_loss: torch.Tensor, model: nn.Module, reference: Mapping[str, torch.Tensor], manifest: FreezeManifest, *, trajectory_weight: float = 12.0, preservation_weight: float = 1.0e-3) -> FinetuneLoss
def assert_loss_contract(head: nn.Module, *, trajectory_weight: float, preservation_weight: float) -> None
```

- [ ] Use hand-calculated tensors to prove `L_sp` is total squared-error sum divided by total allowlisted scalar count, not the mean of per-tensor means. Verify reference tensors are detached FP32 CPU/GPU-compatible clones not registered in model state.
- [ ] Assert upstream `LossComputer` has exact scalar fields `cls_loss_weight == 10.0` and `reg_loss_weight == 8.0`, so it yields `L_traj = 10*L_cls + 8*L_reg`; `assert_loss_contract` also requires outer weights `trajectory_weight == 12.0` and `preservation_weight == 1e-3` before any forward/backward. Test each of the four weight mutations fails, and test that the valid composite is exactly `12*L_traj + 1e-3*L_sp`. Test raw heading scalar L1 across the ±π boundary; no circular correction belongs in training loss.

```python
def source_preservation_loss(model, reference, manifest):
    numerator = torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)
    denominator = 0
    named = dict(model.named_parameters())
    for name in manifest.trainable_parameters:
        current = named[name].float()
        baseline = reference[name].to(current.device, dtype=torch.float32)
        numerator = numerator + (current - baseline).square().sum()
        denominator += current.numel()
    if denominator == 0:
        raise TrainingBoundaryError("no trainable parameters")
    return numerator / denominator
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_loss.py -q`. Expected: FAIL.
- [ ] Compute preservation and final loss in FP32 even when adapter forward uses autocast. Expose individual components for logging without detaching the total used for backward.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_loss.py -q`. Expected: PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/training/loss.py tests/unit/training/test_loss.py
git commit -m "feat: compose trajectory domain adaptation loss"
```

## Task 7: Bind the Verified Cache to a Deterministic Training Dataset

**Files:**

- Create: `src/morai_ddv2/training/dataset.py`
- Create: `src/morai_ddv2/training/sampler.py`
- Create: `configs/training/morai-ddv2-encoder-fusion-v1.yaml`
- Create: `tests/unit/training/test_dataset.py`
- Create: `tests/unit/training/test_sampler.py`
- Create: `tests/unit/training/test_training_config.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    amp: bool
    grad_clip_norm: float
    trajectory_weight: float
    preservation_weight: float
    validation_batch_size: int
    validation_run_id: str
    checkpoint_interval_steps: int

@dataclass(frozen=True)
class SamplerState:
    dataset_size: int
    epoch: int
    permutation: torch.Tensor
    committed_cursor: int
    generator_state: torch.Tensor

class StatefulPermutationSampler(torch.utils.data.Sampler[int]):
    def __iter__(self) -> Iterator[int]
    def __len__(self) -> int
    def mark_consumed(self, count: int) -> None
    def finish_epoch(self) -> None
    def state_dict(self) -> SamplerState
    def load_state_dict(self, state: SamplerState) -> None

class MoraiTrainingDataset(torch.utils.data.Dataset):
    def __getitem__(self, index: int) -> Mapping[str, Any]

def build_dataloaders(manifest: VerifiedManifest, config: TrainingConfig) -> tuple[DataLoader, DataLoader, StatefulPermutationSampler, torch.Generator]
```

- [ ] Test that only manifest-declared train bags enter train and only validation bags enter validation, sample order is stable by `(relative_bag_path,timestamp_ns)`, no frame leakage exists, and production loader rejects dirty/development caches.
- [ ] Assert each returned item has exactly three top-level keys: `features`, `trajectory`, and `metadata`. `features` has exactly `camera_feature [3,256,1024]`, `lidar_feature [1,256,256]`, and `status_feature [8]`; `trajectory` is `[8,3]`; all four tensors are float32. `metadata` carries `sample_id`, bag/path/split, strictly increasing unique selected-front Header `timestamp_ns`, `grid_receipt_ns`, all four sensor Header/receipt/index arrays, raw trajectory-anchor TF Header/receipt/original-index/clock-epoch, independent `ego_bracket_clock_epoch_index`, all brackets, command name/index, and the complete manifest row identity. Training/validation seed uses only `timestamp_ns`; receipt never replaces it.
- [ ] Encode these exact recommended defaults: seed `20260806`, epochs `30`, batch size `8`, workers `4`, AdamW learning rate `1e-5`, weight decay `1e-4`, AMP true, gradient clip norm `1.0`, trajectory weight `12.0`, preservation weight `1e-3`, validation batch size `1`, fixed run ID `morai-ddv2-v1-oracle-command-validation-v1`, and an operational mid-epoch checkpoint interval of `100` optimizer steps. Reject any trajectory/preservation weights other than `12.0` and `1e-3`, and reject validation batch sizes other than `1`.
- [ ] Implement `StatefulPermutationSampler` with one private CPU `torch.Generator`. It materializes one `randperm` per epoch and stores the permutation, generator state, epoch, and **committed** cursor. `__iter__` may run ahead because DataLoader workers prefetch, so it must not advance the committed cursor; the trainer calls `mark_consumed(actual_batch_size)` only after a successful optimizer/scaler step. `finish_epoch()` is legal only when the committed cursor equals the dataset size, then increments the epoch and draws the next permutation. `load_state_dict` rejects a different dataset size, a non-permutation, an out-of-range cursor, or an epoch mismatch.

```python
class StatefulPermutationSampler(torch.utils.data.Sampler[int]):
    def __init__(self, data_source: Sized, seed: int) -> None:
        self._size = len(data_source)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self._epoch = 0
        self._permutation = torch.randperm(self._size, generator=self._generator)
        self._committed_cursor = 0

    def __iter__(self) -> Iterator[int]:
        start = self._committed_cursor
        for position in range(start, self._size):
            yield int(self._permutation[position])

    def __len__(self) -> int:
        return self._size - self._committed_cursor

    def mark_consumed(self, count: int) -> None:
        if count <= 0 or self._committed_cursor + count > self._size:
            raise ResumeContractError("invalid consumed sample count")
        self._committed_cursor += count

    def finish_epoch(self) -> None:
        if self._committed_cursor != self._size:
            raise ResumeContractError("cannot finish an incomplete sampler epoch")
        self._epoch += 1
        self._permutation = torch.randperm(self._size, generator=self._generator)
        self._committed_cursor = 0

    def state_dict(self) -> SamplerState:
        return SamplerState(
            dataset_size=self._size,
            epoch=self._epoch,
            permutation=self._permutation.clone(),
            committed_cursor=self._committed_cursor,
            generator_state=self._generator.get_state().clone(),
        )

    def load_state_dict(self, state: SamplerState) -> None:
        permutation = state.permutation.detach().cpu().to(dtype=torch.long)
        if state.dataset_size != self._size:
            raise ResumeContractError("sampler dataset size changed")
        if state.epoch < 0 or not 0 <= state.committed_cursor <= self._size:
            raise ResumeContractError("invalid sampler epoch or cursor")
        if permutation.shape != (self._size,):
            raise ResumeContractError("invalid sampler permutation shape")
        if not torch.equal(permutation.sort().values, torch.arange(self._size)):
            raise ResumeContractError("sampler state is not a permutation")
        self._epoch = state.epoch
        self._permutation = permutation.clone()
        self._committed_cursor = state.committed_cursor
        self._generator.set_state(state.generator_state.detach().cpu().clone())
```

- [ ] Test a prefetched mid-epoch stop: request more indices from `iter(sampler)` than the trainer marks consumed, serialize state, restore into a fresh sampler, and assert iteration restarts at the first uncommitted index. Also assert an uninterrupted two-epoch index stream is exactly equal to a save/restore stream.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_dataset.py tests/unit/training/test_sampler.py tests/unit/training/test_training_config.py -q`. Expected: FAIL.
- [ ] Use the stateful sampler instead of `shuffle=True`; pass a separate seeded CPU `torch.Generator` to DataLoader for worker base seeds, and seed Python/NumPy from `torch.initial_seed()` in worker init. Dataset `__getitem__` remains deterministic and performs no worker-local random augmentation. Validation never shuffles and always uses B=1.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_dataset.py tests/unit/training/test_sampler.py tests/unit/training/test_training_config.py -q`. Expected: PASS.
- [ ] Commit.

```bash
git add -- configs/training/morai-ddv2-encoder-fusion-v1.yaml src/morai_ddv2/training/dataset.py src/morai_ddv2/training/sampler.py tests/unit/training/test_dataset.py tests/unit/training/test_sampler.py tests/unit/training/test_training_config.py
git commit -m "feat: bind verified MORAI cache to training"
```

## Task 8: Implement AMP Training Steps and Exact Resume

**Files:**

- Create: `src/morai_ddv2/training/rng.py`
- Create: `src/morai_ddv2/training/checkpoint.py`
- Create: `src/morai_ddv2/training/trainer.py`
- Modify: `src/morai_ddv2/training/contracts.py`
- Create: `tests/unit/training/test_rng_checkpoint.py`
- Create: `tests/unit/training/test_train_step.py`
- Create: `tests/unit/training/test_resume.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RngState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]
    dataloader_cpu: torch.Tensor

@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    best_micro_ade: float
    sampler: SamplerState
    accumulator: EpochAccumulatorState
    completed_epoch_record_sha256: str | None

@dataclass(frozen=True)
class EpochAccumulatorState:
    epoch: int
    completed_batches: int
    completed_samples: int
    total_loss_numerator: float
    trajectory_loss_numerator: float
    preservation_loss_numerator: float
    grad_norm_sum: float
    grad_norm_count: int

@dataclass(frozen=True)
class StepMetrics:
    total_loss: float
    trajectory_loss: float
    preservation_loss: float
    grad_norm: float

    @classmethod
    def from_tensors(cls, loss: FinetuneLoss, grad_norm: torch.Tensor) -> StepMetrics

@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    global_step: int
    completed_batches: int
    completed_samples: int
    mean_total_loss: float
    mean_trajectory_loss: float
    mean_preservation_loss: float

def capture_rng_state(dataloader_generator: torch.Generator) -> RngState
def restore_rng_state(state: RngState, dataloader_generator: torch.Generator) -> None
def save_training_checkpoint(path: Path, *, mutable_state, optimizer, scaler, epoch: int, global_step: int, best_micro_ade: float, rng: RngState, sampler_state: SamplerState, accumulator: EpochAccumulatorState, completed_epoch_record_sha256: str | None, base_checkpoint_sha256: str, manifest: FreezeManifest, config: Mapping[str, Any]) -> None
def load_training_checkpoint(path: Path, *, model, optimizer, scaler, sampler: StatefulPermutationSampler, dataloader_generator: torch.Generator, manifest: FreezeManifest, expected_base_checkpoint_sha256: str) -> ResumeState
def train_step(model, batch, optimizer, scaler, reference, manifest, *, amp: bool, grad_clip_norm: float, trajectory_weight: float, preservation_weight: float) -> StepMetrics
def train_one_epoch(*, model: nn.Module, dataloader: DataLoader, sampler: StatefulPermutationSampler, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, reference: Mapping[str, torch.Tensor], manifest: FreezeManifest, amp: bool, grad_clip_norm: float, trajectory_weight: float, preservation_weight: float, epoch: int, global_step: int, accumulator: EpochAccumulatorState) -> EpochMetrics
```

- [ ] Test capture/restore for Python, NumPy, Torch CPU, every CUDA generator, and the DataLoader CPU generator. Test resume equivalence at a **mid-epoch committed batch boundary** with worker prefetch: uninterrupted steps and sample IDs must exactly equal step→`sampler.mark_consumed(batch_size)`→accumulator update→save→fresh base load→mutable/optimizer/scaler/sampler/all-RNG/accumulator restore→remaining steps. Assert both executions finish with identical model tensors, optimizer/scaler state, sample-weighted loss numerators/means, grad-norm sum/count, sample order, epoch, and global step.
- [ ] Test operation order: zero grad; autocast adapter; FP32 preservation/composite; scaled backward; unscale; clip only allowlisted parameters; optimizer/scaler step. Frozen parameters remain byte-identical and allowed BN buffers may change.

```python
def train_step(model, batch, optimizer, scaler, reference, manifest, *, amp, grad_clip_norm, trajectory_weight, preservation_weight):
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=amp):
        result = forward_trajectory_supervised(model, batch["features"], batch["trajectory"])
    loss = compose_finetune_loss(
        result.trajectory_loss.float(), model, reference, manifest,
        trajectory_weight=trajectory_weight,
        preservation_weight=preservation_weight,
    )
    scaler.scale(loss.total).backward()
    scaler.unscale_(optimizer)
    parameters = [dict(model.named_parameters())[name] for name in manifest.trainable_parameters]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    assert_training_boundary(model, manifest)
    return StepMetrics.from_tensors(loss, grad_norm)
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_rng_checkpoint.py tests/unit/training/test_train_step.py tests/unit/training/test_resume.py -q`. Expected: FAIL.
- [ ] Save training checkpoints separately from deployment checkpoints. They contain only mutable normalized state plus optimizer/scaler/current epoch/global step/best/config/freeze, `SamplerState`, `EpochAccumulatorState`, optional `completed_epoch_record_sha256`, and every `RngState` field. Load order is base 971 strict load→boundary/reference snapshot→mutable state→optimizer/scaler→sampler→accumulator→global and DataLoader RNG. Restore all state before creating the resumed DataLoader iterator. Reject accumulator epoch/counts that disagree with sampler epoch/committed cursor.
- [ ] After each successful optimizer/scaler step, call `sampler.mark_consumed(actual_batch_size)`, update the accumulator with `metric * actual_batch_size` numerators, increment global step, and atomically publish `last.pt` whenever `global_step % checkpoint_interval_steps == 0`. Do not clear the accumulator or call `sampler.finish_epoch()` until Task 10 has durably published that epoch's canonical telemetry JSON. `best.pt` is written only after Task 9 validation and only on strictly lower selected sample-micro ADE.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_rng_checkpoint.py tests/unit/training/test_train_step.py tests/unit/training/test_resume.py -q`. Expected: PASS, including exact mid-epoch sample-order and state equivalence.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/training/contracts.py src/morai_ddv2/training/rng.py src/morai_ddv2/training/checkpoint.py src/morai_ddv2/training/trainer.py tests/unit/training/test_rng_checkpoint.py tests/unit/training/test_train_step.py tests/unit/training/test_resume.py
git commit -m "feat: train and resume DDV2 encoder adaptation"
```

## Task 9: Implement Deterministic PDM-Free Selector Validation

**Files:**

- Create: `src/morai_ddv2/training/metrics.py`
- Create: `src/morai_ddv2/training/validation.py`
- Create: `tests/unit/training/test_metrics.py`
- Create: `tests/unit/training/test_validation.py`

**Interfaces:**

```python
VALIDATION_RUN_ID = "morai-ddv2-v1-oracle-command-validation-v1"

@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    bag_sha256: str
    command_name: str
    timestamp_ns: int
    seed: int
    prediction: np.ndarray
    target: np.ndarray
    candidate_source: Literal["coarse_top32_post_bezier"]
    candidate_count: Literal[32]
    min_ade_32: float
    min_fde_32: float
    argmin_ade_rank: int
    argmin_fde_rank: int

@dataclass(frozen=True)
class MetricSummary:
    count: int
    ade: float | None
    fde: float | None
    heading_mae: float | None

@dataclass(frozen=True)
class ValidationReport:
    domain_id: str
    command_source: str
    run_id: str
    micro: MetricSummary
    oracle_min_ade_32: float | None
    oracle_min_fde_32: float | None
    oracle_candidate_source: Literal["coarse_top32_post_bezier"]
    oracle_candidate_count: Literal[32]
    bag_macro_ade: float | None
    per_command: Mapping[str, MetricSummary]
    records: tuple[PredictionRecord, ...]
    drop_statistics: Mapping[str, int]
    sync_statistics: Mapping[str, int | float]
    identities: Mapping[str, str]

def selector_seed(timestamp_ns: int, run_id: str = VALIDATION_RUN_ID) -> int
def reset_torch_seed(seed: int) -> None
def bezier_headings(selected_xy: np.ndarray) -> np.ndarray
@contextmanager
def capture_coarse_top32_post_bezier(model: nn.Module) -> Iterator[CapturedShortlist]
def compute_best_of_32_metrics(candidates: np.ndarray, target: np.ndarray) -> BestOf32Metrics
def validation_loss_seed(timestamp_ns: int, run_id: str = VALIDATION_RUN_ID) -> int
def validate_supervised_loss_b1(model: nn.Module, samples: Iterable[Mapping[str, Any]], *, reference, manifest, trajectory_weight: float, preservation_weight: float, run_id: str = VALIDATION_RUN_ID) -> ValidationLossSummary
def validate_selector_b1(model: nn.Module, samples: Iterable[Mapping[str, Any]], *, run_id: str = VALIDATION_RUN_ID) -> ValidationReport
def compute_openloop_metrics(records: Sequence[PredictionRecord]) -> ValidationReport
```

- [ ] Test the fixed seed vector: timestamp `1722500000123456789` and run ID `morai-ddv2-v1-oracle-command-validation-v1` produce digest `a9e2a2b3aef37e1993e5de61d7dc84581a4acd8cf938d0296b43c8fb54aec4b0` and unsigned big-endian seed `12241525629764468249`. Reset CPU and all CUDA RNG immediately before each sample forward.
- [ ] Test degree-8 Bézier headings from control points `P0=(0,0),P1..P8`, derivative at `t=k/8` for k=1..8, and `atan2(dy,dx)`. This NumPy-only metric helper is importable in the ROS Python environment. It describes final runtime heading; decoder heading is not the reported prediction.
- [ ] Test micro ADE, FDE, circular heading MAE, bag-macro ADE, and per-command metrics. Test `minADE@32` and `minFDE@32` with hand-computed literal candidates, independent ADE/FDE argmins, duplicate candidates retained, XY-only errors, rank in `[0,31]`, and float64 sample aggregation. Empty class output is `{count: 0, metric: null}`, never NaN.
- [ ] Test order invariance by evaluating the same fake samples forward and reversed; each sample ID must have an exactly equal selected `[8,3]` prediction and seed.
- [ ] Test validation-only capture against the pinned call shape: wrap the original `bezier_xyyaw` and `_select_topk`, call each original exactly once, retain an unchanged detached `[B,800,8,3]` pool and `[B,32]` indices, return the exact original objects, gather `[B,32,8,3]`, and restore both callables in `finally` on success and exception. Capture-on/off public outputs must be bitwise identical and the selected trajectory must equal at least one shortlist row. Never expose candidates in the deployed return dict and never run a second selector forward for oracle metrics.
- [ ] Test validation loss at B=1 with seed `int.from_bytes(SHA256(b"diffusiondrive_v2_supervised_loss:<timestamp_ns>:<run_id>")[:8], "big")`: reset CPU/all CUDA RNG immediately before each supervised-adapter call, use no-grad, aggregate total/trajectory/preservation by sample count, and restore the exact training boundary afterward. Manifest and reverse traversal produce identical finite means.

```python
def selector_seed(timestamp_ns: int, run_id: str = VALIDATION_RUN_ID) -> int:
    payload = f"diffusiondrive_v2:{timestamp_ns}:{run_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)

def reset_torch_seed(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_metrics.py tests/unit/training/test_validation.py -q`. Expected: FAIL.
- [ ] Set the network eval while validating, use the same PDM-free learned selector public derivative used by Validator, B=1 only, and final returned `trajectory`. Capture the exact post-Bézier coarse top-32 from that same forward only for diagnostic `minADE@32`/`minFDE@32`. Never use GT-nearest `m_star`, GT trajectory, decoder logits, the full 800 proposal pool, or the four stage winners to select a candidate or compute the named @32 metrics.
- [ ] Keep metric aggregation, heading reconstruction, wrapping, and seed derivation NumPy/stdlib-only in `training/metrics.py`; import Torch locally only inside `reset_torch_seed` or in `validation.py`. This lets the ROS report path reuse exact metrics without importing Torch.
- [ ] Restore the exact training boundary after selector and loss validation. Write reports with domain/oracle labels, run ID, per-sample seed, selected metrics, candidate source/count, per-sample min/ranks, aggregate minADE@32/minFDE@32, counts, drop statistics, separate side-Header/sensor-receipt/raw-TF-anchor-receipt/TF-epoch/Ego-epoch statistics, bound epoch-safe TF/Ego diagnostic counters/report hashes, 28-dataset cache-schema identity and 24 repeated manifest provenance values, and no non-JSON floats. Sort serialized `records` by `sample_id` and omit traversal order and wall-clock generation time so manifest-order and reverse-order runs with identical per-sample outputs produce byte-identical canonical reports.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_metrics.py tests/unit/training/test_validation.py -q`. Expected: PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/training/metrics.py src/morai_ddv2/training/validation.py tests/unit/training/test_metrics.py tests/unit/training/test_validation.py
git commit -m "feat: validate learned DDV2 selector deterministically"
```

## Task 10: Add the Main Training and Offline Evaluation CLIs

**Files:**

- Modify: `src/morai_ddv2/training/trainer.py`
- Modify: `src/morai_ddv2/training/contracts.py`
- Create: `src/morai_ddv2/training/telemetry.py`
- Create: `src/morai_ddv2/cli/train.py`
- Create: `src/morai_ddv2/cli/evaluate.py`
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`
- Create: `tests/unit/training/test_fit.py`
- Create: `tests/unit/training/test_telemetry.py`
- Create: `tests/unit/cli/test_train_cli.py`
- Create: `tests/unit/cli/test_evaluate_cli.py`

**Interfaces:**

```text
scripts/train.py --manifest PATH --resolved-domain PATH --base-profile PATH --config PATH --run-dir PATH [--resume PATH] [--max-epochs POSITIVE_INT] [--max-train-batches POSITIVE_INT] [--max-validation-samples POSITIVE_INT]
scripts/evaluate.py offline --manifest PATH --resolved-domain PATH --base-profile PATH --training-checkpoint PATH --report PATH [--sample-order {manifest,reverse}]
```

```python
@dataclass(frozen=True)
class ProductionRunAttestation:
    training_config_sha256: str
    training_manifest_sha256: str
    resolved_domain_sha256: str
    cache_schema_version: Literal[2]
    clock_policy_version: Literal[1]
    base_checkpoint_sha256: str
    completed_epochs: int
    best_checkpoint_sha256: str
    best_micro_ade: float
    validation_run_id: str
    epoch_telemetry_count: int
    metrics_history_sha256: str
    epoch_metrics_csv_sha256: str
    training_curves_sha256: str

@dataclass(frozen=True)
class EpochTelemetry:
    schema_version: Literal[1]
    epoch: int
    global_step: int
    train: EpochMetrics
    validation_loss: ValidationLossSummary
    validation: ValidationReport
    validation_report_sha256: str
    identities: Mapping[str, str]
    best_epoch: int
    best_selected_micro_ade: float
    is_best: bool

def publish_epoch_telemetry(run_dir: Path, telemetry: EpochTelemetry) -> tuple[Path, str]
def rebuild_training_artifacts(run_dir: Path) -> TrainingArtifactHashes

def publish_production_run_attestation(run_dir: Path, attestation: ProductionRunAttestation) -> Path
```

- [ ] Test CLI preflight failures before model construction: schema other than 2, a 27-field or otherwise incomplete schema 2, wrong corrected Option D resolved-domain/domain/preset/manifest hash, non-increasing/duplicate front identity, missing raw TF or Ego epoch provenance, mismatched diagnostic report/counters, mixed-axis provenance, dirty cache, wrong source commit, missing assets, wrong Python/Torch, unavailable CUDA for an actual training invocation, zero/negative values for each `--max-*` smoke bound, and any offline `--sample-order` other than `manifest` or `reverse`. Test that omitted bounds preserve the YAML values, supplied positive bounds are recorded in the resolved run config and limit only that invocation, and reverse order changes traversal only while canonical report rows remain sample-ID sorted.
- [ ] Test a two-epoch fake-model fit: training boundary reapplied each epoch, selector and supervised-loss validation B=1, `last.pt` every epoch, `best.pt` only on strictly lower selected micro ADE, selected FDE/heading/macro plus minADE@32/minFDE@32 logged, and resume continues epoch/global step/RNG/accumulator exactly. Test crash reconciliation at (a) mid-epoch, (b) durable epoch JSON before sampler-finished `last.pt`, and (c) next-epoch `last.pt` with missing/mismatched epoch JSON, which must fail. Atomically publish `run-complete.json` only after all configured epochs finish with all three smoke bounds absent; it binds config/manifest/base/best hashes, epoch count, history/CSV/PNG hashes. Interrupted and bounded smoke/debug runs must not have this attestation.
- [ ] Write telemetry tests first. One authoritative file per completed epoch is `metrics/epochs/epoch-{epoch:04d}.json`, RFC 8785 canonical bytes, published by sibling `.partial`/file-fsync/replace/parent-fsync with no wall-clock or nonfinite value. Existing exact bytes are idempotent; conflicting bytes fail. From sorted epoch files, atomically rebuild `metrics/history.jsonl`, `metrics/epoch_metrics.csv`, and `plots/training-curves.png` rather than appending. CSV columns are epoch/global_step, train and validation total/trajectory/preservation loss, selected ADE/FDE/heading, minADE@32/minFDE@32, and best flag. PNG has exactly three labeled panels: train/validation total loss; selected ADE/FDE; minADE@32/minFDE@32. A generated plot is never resume state.
- [ ] Implement `fit` with constant recommended AdamW learning rate from config; do not add an unapproved scheduler. Before constructing the optimizer, call `assert_loss_contract` against the loaded upstream head and resolved outer weights, then pass both outer weights explicitly to every train and validation-loss call. At epoch end atomically publish the full validation report and `best.pt` if improved, publish epoch telemetry, call `sampler.finish_epoch()`, clear the accumulator for the next epoch, publish `last.pt` with the epoch-record SHA-256, rebuild JSONL/CSV/PNG, then start the next epoch. A cursor-at-end resume with an exact existing epoch record completes this sequence idempotently; a next-epoch checkpoint without the bound record fails. Keep step JSONL and resolved training config with all source/cache/checkpoint identities and smoke bounds.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training/test_fit.py tests/unit/training/test_telemetry.py tests/unit/cli/test_train_cli.py tests/unit/cli/test_evaluate_cli.py -q`. Expected: PASS after the initial import failures are resolved.
- [ ] When GPU preflight passes, run a one-batch smoke using real cached data and the official checkpoint with `epochs=1` override. Assert finite loss/gradients/metrics, nonzero allowed gradient, frozen tensor hashes unchanged, and last/best resume artifacts load.

```bash
MORAI_MANIFEST="$(find cache/manifests/morai-ddv2-v1-oracle-command -mindepth 2 -maxdepth 2 -type f -name manifest.jsonl -print)"
test "$(printf '%s\n' "$MORAI_MANIFEST" | sed '/^$/d' | wc -l)" -eq 1
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/train.py \
  --manifest "$MORAI_MANIFEST" \
  --resolved-domain artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
  --base-profile third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json \
  --config configs/training/morai-ddv2-encoder-fusion-v1.yaml \
  --run-dir runs/morai-ddv2-smoke \
  --max-epochs 1 \
  --max-train-batches 1 \
  --max-validation-samples 2
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/training tests/unit/cli/test_train_cli.py tests/unit/cli/test_evaluate_cli.py -q`. Expected: PASS, then commit.

```bash
.venv-train/bin/python -m pytest tests/unit/training tests/unit/cli/test_train_cli.py tests/unit/cli/test_evaluate_cli.py -q
git add -- src/morai_ddv2/training/contracts.py src/morai_ddv2/training/telemetry.py src/morai_ddv2/training/trainer.py src/morai_ddv2/cli/train.py src/morai_ddv2/cli/evaluate.py scripts/train.py scripts/evaluate.py tests/unit/training/test_fit.py tests/unit/training/test_telemetry.py tests/unit/cli/test_train_cli.py tests/unit/cli/test_evaluate_cli.py
git commit -m "feat: add MORAI DDV2 training loop"
```

## Task 11: Implement Canonical Tensor Hashing and 972-Key Schema Merge

**Files:**

- Create: `src/morai_ddv2/export/__init__.py`
- Create: `src/morai_ddv2/export/state_hash.py`
- Create: `src/morai_ddv2/export/schema.py`
- Create: `tests/unit/export/test_state_hash.py`
- Create: `tests/unit/export/test_schema.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SelSchemaReport:
    raw_key_count: int
    runtime_key_count: int
    raw_shapes: Mapping[str, tuple[int, ...]]
    normalized_to_raw: Mapping[str, str]

@dataclass(frozen=True)
class TensorSchemaEntry:
    key: str
    shape: tuple[int, ...]

def tensor_data_sha256(tensor: torch.Tensor) -> str
def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str
def state_schema_sha256(state: Mapping[str, torch.Tensor]) -> str
def validate_raw_schema(raw_state: Mapping[str, torch.Tensor], expected_raw_state: Mapping[str, torch.Tensor]) -> SelSchemaReport
def merge_mutable_state(raw_state: Mapping[str, torch.Tensor], mutable_normalized_state: Mapping[str, torch.Tensor], manifest: FreezeManifest) -> OrderedDict[str, torch.Tensor]
def assert_only_allowed_changed(base_raw, exported_raw, manifest) -> None
```

- [ ] Create fixed hashing vectors for float32, float16, int64, noncontiguous tensors, and dictionary insertion-order permutations. Each entry is `{key,dtype,shape,data_sha256}`, sorted by UTF-8 key bytes and RFC 8785 hashed. Tensor data is detached contiguous CPU little-endian C-order bytes.
- [ ] Test exact 972 raw keys, exact 971 normalized runtime keys, one preserved vocab key, exact equality to the authoritative base raw key set and every per-key shape, and byte hashes unchanged outside allowlisted parameter plus BN buffer entries. `validate_raw_schema(candidate, base)` compares sets before counts and shapes before normalization; a same-count renamed key, swapped key, changed shape, duplicate normalized key, or unauthorized mutable key must fail. `state_schema_sha256` hashes RFC 8785 canonical `{key,shape}` entries sorted by UTF-8 key bytes so metadata can bind the complete 972-key schema without embedding 972 entries.

```python
def state_dict_sha256(state):
    entries = []
    for key in sorted(state, key=lambda item: item.encode("utf-8")):
        tensor = state[key].detach().cpu().contiguous()
        array = tensor.numpy()
        little = array.astype(array.dtype.newbyteorder("<"), copy=False)
        entries.append({
            "key": key,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data_sha256": hashlib.sha256(little.tobytes(order="C")).hexdigest(),
        })
    return hashlib.sha256(rfc8785.dumps(entries)).hexdigest()
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/export/test_state_hash.py tests/unit/export/test_schema.py -q`. Expected: FAIL.
- [ ] Preserve original raw key strings/order when merging. Map normalized mutable names back to their unique raw keys; clone to CPU with original dtype and verify shape before replacement.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/export/test_state_hash.py tests/unit/export/test_schema.py -q`. Expected: PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/export/__init__.py src/morai_ddv2/export/state_hash.py src/morai_ddv2/export/schema.py tests/unit/export/test_state_hash.py tests/unit/export/test_schema.py
git commit -m "feat: hash and merge DDV2 checkpoint state"
```

## Task 12: Atomically Publish the Deployment Checkpoint and Metadata

**Files:**

- Create: `src/morai_ddv2/export/metadata.py`
- Create: `src/morai_ddv2/export/exporter.py`
- Create: `tests/unit/export/test_metadata.py`
- Create: `tests/unit/export/test_exporter.py`
- Create: `tests/unit/export/test_export_failure.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ExportIdentities:
    domain_id: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    training_manifest_sha256: str
    cache_schema_version: Literal[2]
    clock_policy_version: Literal[1]
    training_config_sha256: str
    source_commits: Mapping[str, str]
    best_micro_ade: float
    validation_run_id: str

@dataclass(frozen=True)
class ExportMetadata:
    base_checkpoint_sha256: str
    base_state_dict_sha256: str
    export_checkpoint_sha256: str
    export_state_dict_sha256: str
    raw_schema_sha256: str
    resolved_domain_sha256: str
    sensor_preset_sha256: str
    training_manifest_sha256: str
    cache_schema_version: Literal[2]
    clock_policy_version: Literal[1]
    training_config_sha256: str
    domain_id: str
    source_commits: Mapping[str, str]
    trainable_prefixes: tuple[str, ...]
    mutable_bn_policy: str
    raw_key_count: int
    runtime_key_count: int
    best_micro_ade: float
    validation_run_id: str
    exported_at_utc: str

@dataclass(frozen=True)
class ExportBundle:
    checkpoint: Path
    metadata: Path
    checkpoint_sha256: str
    state_dict_sha256: str

def build_export_metadata(*, base: RawSelCheckpoint, checkpoint_sha256: str, state_dict_sha256: str, identities: ExportIdentities, manifest: FreezeManifest) -> ExportMetadata
def write_metadata_atomic(path: Path, metadata: ExportMetadata) -> None
def export_sel_bundle(base: RawSelCheckpoint, mutable_state: Mapping[str, torch.Tensor], output: Path, *, identities: ExportIdentities, manifest: FreezeManifest) -> ExportBundle
```

- [ ] Test that the top-level raw checkpoint container is shallow-preserved and only its state mapping is replaced. Assert 972 keys, 971 runtime keys, original vocab bytes, unchanged frozen hashes, and absence of metadata keys in the checkpoint.
- [ ] Assert metadata contains exactly the `ExportMetadata` fields above, including the authoritative `raw_schema_sha256`: base checkpoint/state hashes, export checkpoint/state hashes, resolved-domain/preset/training-manifest hashes, explicit cache schema 2 and clock-policy version 1, domain ID, source commits, five trainable prefixes, mutable BN policy, raw/runtime counts 972/971, training config hash, best micro ADE, run ID, and export timestamp. It must not contain a self-referential `metadata_sha256`.
- [ ] Inject checkpoint write/fsync/rename and metadata write/fsync/rename failures. Metadata final path must never exist before the finalized checkpoint's actual file hash is known; a metadata failure leaves a verifiable checkpoint but an incomplete bundle that preflight rejects.

```python
def export_sel_bundle(base, mutable_state, output, *, identities, manifest):
    merged = merge_mutable_state(base.raw_state_dict, mutable_state, manifest)
    validate_raw_schema(merged, base.raw_state_dict)
    assert_only_allowed_changed(base.raw_state_dict, merged, manifest)
    container = dict(base.container)
    container["state_dict"] = merged
    write_torch_checkpoint_atomic(output, container)
    checkpoint_hash = file_sha256(output)
    state_hash = state_dict_sha256(merged)
    metadata = build_export_metadata(
        base=base, checkpoint_sha256=checkpoint_hash,
        state_dict_sha256=state_hash, identities=identities, manifest=manifest,
    )
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    write_metadata_atomic(metadata_path, metadata)
    return ExportBundle(output, metadata_path, checkpoint_hash, state_hash)
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/export/test_metadata.py tests/unit/export/test_exporter.py tests/unit/export/test_export_failure.py -q`. Expected: FAIL.
- [ ] Implement publication with sibling `.partial`, fsync, replace, and parent fsync. Reopen/recompute both checkpoint file and state hashes before metadata publication.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/export/test_metadata.py tests/unit/export/test_exporter.py tests/unit/export/test_export_failure.py -q`. Expected: PASS.
- [ ] Commit.

```bash
git add -- src/morai_ddv2/export/metadata.py src/morai_ddv2/export/exporter.py tests/unit/export/test_metadata.py tests/unit/export/test_exporter.py tests/unit/export/test_export_failure.py
git commit -m "feat: publish schema-preserving DDV2 checkpoint"
```

## Task 13: Generate and Preflight the Exact-ID Validator Profile

**Files:**

- Create: `src/morai_ddv2/export/validator_profile.py`
- Create: `src/morai_ddv2/export/validator_preflight.py`
- Create: `src/morai_ddv2/cli/export_checkpoint.py`
- Create: `src/morai_ddv2/cli/validate_export.py`
- Create: `scripts/export_checkpoint.py`
- Create: `scripts/validate_export.py`
- Create: `configs/validator/.gitignore`
- Generated and ignored: `configs/validator/profile-morai-ddv2-oracle.json`
- Create: `tests/unit/export/test_validator_profile.py`
- Create: `tests/unit/export/test_validator_preflight.py`
- Create: `tests/unit/cli/test_export_cli.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class VerifiedValidatorBundle:
    selected_profile: Any
    metadata: ExportMetadata
    raw_checkpoint: RawSelCheckpoint
    runtime_state: OrderedDict[str, torch.Tensor]
    network: nn.Module

def build_morai_validator_profile(*, baseline_profile: Path, output_profile: Path, validator_root: Path, bundle: ExportBundle, resolved_domain: Path, sensor_preset: Path, training_manifest: Path) -> Path
def resolve_profile_asset(asset: Mapping[str, Any], validator_root: Path) -> Path
def sel_assets_from_verified_roles(profile: Mapping[str, Any], roles: Mapping[str, Path], validator_root: Path) -> SelAssets
def verify_validator_bundle(profile_file: Path, validator_root: Path) -> VerifiedValidatorBundle
```

- [ ] Create `configs/validator/.gitignore` with exactly these bytes, including the final newline. Test that the official baseline `src/Validator_pkg/config/profile.json` remains byte-identical, the generated file is exactly the ignored `configs/validator/profile-morai-ddv2-oracle.json`, and the selected entry ID remains `diffusiondrive_v2`, never a new ID. Unit tests use temporary output paths; only Task 14 creates the machine-specific project path.

```gitignore
/profile-morai-ddv2-oracle.json
```
- [ ] Test runtime assets with real absolute paths and actual hashes for checkpoint, checkpoint metadata, resolved domain, sensor preset, training manifest, image backbone, and trajectory anchor. Profile identity repeats domain ID, resolved hash, exported state hash, exact raw-schema hash, training manifest hash, cache schema 2, and clock-policy version 1.
- [ ] Assert the generated profile's camera contract names the 640×480 side and 1280×720 front sources plus the exact 341/455/341 resize, `[56,1080)` crop, and RGB float32 output; LiDAR contract names `[1.5,0,1.25]`, the 3D self-mask, and count/5. Set `static_transforms.T_base_from_lidar` to identity rotation with translation `[1.5,0.0,1.25]`, and set `inference_policy.reference_stamp_policy` to `new_stamp_once`.
- [ ] Test preflight order with spies: Validator `load_selected_profile`→existing `verify_runtime_assets`→mandatory metadata read→checkpoint/profile/metadata file-hash equality→972 tensor and exact key/shape schema hashes→domain/sensor/manifest identity. Every metadata/hash/identity mismatch must fail before network construction. Only after those checks, construct the pinned PDM-free derivative, normalize 972→971, and invoke Validator's `load_diffusiondrive_selector_state_dict`, which must reach the derivative's actual `load_state_dict(normalized, strict=True)`; same-count renamed keys and wrong shapes fail there.

```python
def verify_validator_bundle(profile_file, validator_root):
    selected = load_selected_profile(profile_file, "diffusiondrive_v2")
    verify_runtime_assets(selected.data, validator_root)
    roles = {asset["role"]: resolve_profile_asset(asset, validator_root) for asset in selected.data["runtime_assets"]}
    metadata = load_required_metadata(roles["checkpoint_metadata"])
    assert_checkpoint_hash_identity(selected.data, metadata, roles["checkpoint"])
    raw = load_raw_sel_checkpoint(roles["checkpoint"])
    assert state_dict_sha256(raw.raw_state_dict) == metadata.export_state_dict_sha256
    assert state_schema_sha256(raw.raw_state_dict) == metadata.raw_schema_sha256
    assert_export_identity(selected.data, metadata, roles)
    runtime = normalize_sel_runtime_state(raw.raw_state_dict)
    if len(runtime) != 971:
        raise SelSchemaError(f"expected 971 runtime keys, got {len(runtime)}")
    assets = sel_assets_from_verified_roles(selected.data, roles, validator_root)
    network = construct_pdm_free_sel_network(assets, torch.device("cpu"))
    load_diffusiondrive_selector_state_dict(
        network,
        dict(raw.raw_state_dict),
        checkpoint_path=roles["checkpoint"],
    )
    return VerifiedValidatorBundle(selected, metadata, raw, runtime, network)
```

- [ ] Run `.venv-train/bin/python -m pytest tests/unit/export/test_validator_profile.py tests/unit/export/test_validator_preflight.py tests/unit/cli/test_export_cli.py -q`. Expected: FAIL.
- [ ] Make export CLI consume a verified `best.pt`, its sibling `run-complete.json`, the exact base checkpoint, resolved snapshot, production manifest, and clean source tree. Verify the attested best file/config/manifest/base hashes and completed epoch count before loading the base model; an incomplete, interrupted, or bounded smoke/debug run has no valid attestation and is rejected. Publish checkpoint/metadata/profile in that order, then run preflight automatically. Never export `last.pt` implicitly.
- [ ] Make validation CLI return nonzero for missing metadata or any hash/schema/identity discrepancy. Plan 3 will make this same preflight mandatory before ROS launch.
- [ ] Run `.venv-train/bin/python -m pytest tests/unit/export/test_validator_profile.py tests/unit/export/test_validator_preflight.py tests/unit/cli/test_export_cli.py -q`. Expected: PASS, including proof that Validator's real strict loader is reached only after identity checks; then commit.

```bash
git add -- configs/validator/.gitignore src/morai_ddv2/export/validator_profile.py src/morai_ddv2/export/validator_preflight.py src/morai_ddv2/cli/export_checkpoint.py src/morai_ddv2/cli/validate_export.py scripts/export_checkpoint.py scripts/validate_export.py tests/unit/export/test_validator_profile.py tests/unit/export/test_validator_preflight.py tests/unit/cli/test_export_cli.py
git commit -m "feat: bind DDV2 export to Validator profile"
```

## Task 14: Run Official Checkpoint Export and Pinned Validator Compatibility Gates

**Files:**

- Create: `tests/integration/test_pinned_export_validator.py`
- Modify only if a test exposes an actual contract defect: code/tests/spec through a new reviewed task
- Generated, not committed: `runs/morai-ddv2-v1-oracle-command/`
- Generated, not committed: `artifacts/checkpoints/morai-ddv2-v1-oracle-command/`
- Generated and ignored, never committed: `configs/validator/profile-morai-ddv2-oracle.json`; tracked owner: `configs/validator/.gitignore`

**Interfaces:**

- Official raw checkpoint loads as 972 keys; deployment derivative strict-loads 971 keys.
- Only allowlisted parameters and encoder BatchNorm buffers differ from base.
- Fixed validation samples yield finite `[1,8,3]` learned-selector output under deterministic per-sample seeds.

- [ ] Run the complete CPU-safe suite and upstream strict-load tests.

```bash
.venv-train/bin/python -m ruff check src tests scripts
.venv-train/bin/python -m mypy src/morai_ddv2
.venv-ros/bin/python -m pytest \
  tests/unit/test_repository_contract.py \
  tests/unit/config \
  tests/unit/preprocess \
  tests/unit/ros1 \
  tests/unit/cache \
  tests/unit/cli/test_inspect_bags.py \
  tests/unit/cli/test_preprocess_bags.py \
  tests/unit/cli/test_verify_cache.py -q
.venv-train/bin/python -m pytest \
  tests/unit/training \
  tests/unit/export \
  tests/unit/cli/test_train_cli.py \
  tests/unit/cli/test_evaluate_cli.py \
  tests/unit/cli/test_export_cli.py -q
MORAI_DDV2_VALIDATOR_ROOT="$PWD/third_party/E2E-OpenLoop-ROS-Validator" \
.venv-train/bin/python -m pytest tests/integration/test_pinned_sel_loader.py -m upstream -q
git diff --check
```

Expected: the ROS/preprocess suite runs only under Python 3.8, the training/export suite runs only under Python 3.9.25, and every command passes. Never run the Python-3.8 repository contract through `.venv-train`.

- [ ] Require `nvidia-smi` and Torch CUDA preflight to pass. If they do not, stop before real training and report this as an environment blocker; do not mark the plan complete based on fake-model tests.
- [ ] Resolve the one production manifest and run full 30-epoch training. Retain resolved config, logs, `last.pt`, best-micro-ADE `best.pt`, validation reports, and RNG/resume metadata under the run directory.

```bash
MORAI_MANIFEST="$(find cache/manifests/morai-ddv2-v1-oracle-command -mindepth 2 -maxdepth 2 -type f -name manifest.jsonl -print)"
test "$(printf '%s\n' "$MORAI_MANIFEST" | sed '/^$/d' | wc -l)" -eq 1
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/train.py \
  --manifest "$MORAI_MANIFEST" \
  --resolved-domain artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
  --base-profile third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json \
  --config configs/training/morai-ddv2-encoder-fusion-v1.yaml \
  --run-dir runs/morai-ddv2-v1-oracle-command
test -s runs/morai-ddv2-v1-oracle-command/run-complete.json
test "$(find runs/morai-ddv2-v1-oracle-command/metrics/epochs -type f -name 'epoch-*.json' | wc -l)" -eq 30
test "$(wc -l < runs/morai-ddv2-v1-oracle-command/metrics/epoch_metrics.csv)" -eq 31
test -s runs/morai-ddv2-v1-oracle-command/metrics/history.jsonl
test -s runs/morai-ddv2-v1-oracle-command/plots/training-curves.png
```
- [ ] Re-run offline validation for `best.pt` twice in manifest order and once in reverse order, then require byte-identical canonical reports. Each invocation resets the per-sample seed immediately before forward, and `cmp` therefore proves exact sample-ID prediction/seed and metric invariance.

```bash
MORAI_MANIFEST="$(find cache/manifests/morai-ddv2-v1-oracle-command -mindepth 2 -maxdepth 2 -type f -name manifest.jsonl -print)"
test "$(printf '%s\n' "$MORAI_MANIFEST" | sed '/^$/d' | wc -l)" -eq 1
for MORAI_REPORT_NAME in validation-manifest-1 validation-manifest-2; do
  CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/evaluate.py offline \
    --manifest "$MORAI_MANIFEST" \
    --resolved-domain artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
    --base-profile third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json \
    --training-checkpoint runs/morai-ddv2-v1-oracle-command/best.pt \
    --report "runs/morai-ddv2-v1-oracle-command/${MORAI_REPORT_NAME}.json" \
    --sample-order manifest
done
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/evaluate.py offline \
  --manifest "$MORAI_MANIFEST" \
  --resolved-domain artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
  --base-profile third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json \
  --training-checkpoint runs/morai-ddv2-v1-oracle-command/best.pt \
  --report runs/morai-ddv2-v1-oracle-command/validation-reverse.json \
  --sample-order reverse
cmp runs/morai-ddv2-v1-oracle-command/validation-manifest-1.json \
    runs/morai-ddv2-v1-oracle-command/validation-manifest-2.json
cmp runs/morai-ddv2-v1-oracle-command/validation-manifest-1.json \
    runs/morai-ddv2-v1-oracle-command/validation-reverse.json
```
- [ ] Export the best training checkpoint and generate the profile:

```bash
MORAI_MANIFEST="$(find cache/manifests/morai-ddv2-v1-oracle-command -mindepth 2 -maxdepth 2 -type f -name manifest.jsonl -print)"
test "$(printf '%s\n' "$MORAI_MANIFEST" | sed '/^$/d' | wc -l)" -eq 1
.venv-train/bin/python scripts/export_checkpoint.py \
  --training-checkpoint runs/morai-ddv2-v1-oracle-command/best.pt \
  --base-checkpoint third_party/E2E-OpenLoop-ROS-Validator/models/diffusiondrive_v2/navsim_v2/diffusiondrivev2_sel.ckpt \
  --manifest "$MORAI_MANIFEST" \
  --resolved-domain artifacts/domain/morai-ddv2-v1-oracle-command/resolved-domain.json \
  --sensor-preset sensor_data/0725demo_ros.json \
  --baseline-profile third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json \
  --output artifacts/checkpoints/morai-ddv2-v1-oracle-command/ddv2-morai.ckpt \
  --profile-output configs/validator/profile-morai-ddv2-oracle.json

.venv-train/bin/python scripts/validate_export.py \
  --profile configs/validator/profile-morai-ddv2-oracle.json \
  --validator-root third_party/E2E-OpenLoop-ROS-Validator
git check-ignore -q configs/validator/profile-morai-ddv2-oracle.json
test -z "$(git ls-files --error-unmatch configs/validator/profile-morai-ddv2-oracle.json 2>/dev/null || true)"
```

- [ ] Run actual integration on GPU using fixed validation samples. Assert raw/runtime counts, strict load, unchanged frozen tensor hashes, finite output, `[1,8,3]`, and deterministic order invariance.

```bash
CUDA_VISIBLE_DEVICES=0 MORAI_DDV2_VALIDATOR_ROOT="$PWD/third_party/E2E-OpenLoop-ROS-Validator" \
.venv-train/bin/python -m pytest tests/integration/test_pinned_export_validator.py -m upstream -q
```

- [ ] Run the pinned Validator loader regression without modifying its baseline profile.

```bash
PYTHONPATH=third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_inference/src \
.venv-train/bin/python -m pytest -q \
  third_party/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_inference/test/test_runtime_integrity.py \
  -k 'ddv2_loader or diffusion_derivative'
```

- [ ] Record the export checkpoint, state-dict, metadata file, resolved-domain, sensor-preset, and training-manifest hashes for plan 3. Commit only the integration test, never run artifacts.

```bash
git add -- tests/integration/test_pinned_export_validator.py
git commit -m "test: verify exported DDV2 Validator compatibility"
```

## Specification Coverage Matrix

| Normative area | Implemented/tested in |
|---|---|
| Pinned Python/source/assets and GPU preflight | Task 1 |
| Raw 972 → derivative 971 strict initialization | Task 2 |
| Five-prefix parameter/BN boundary | Task 3 |
| Shared feature graph without agent/BEV/PDM paths | Task 4 |
| One anchor group, XY noise, single decoder layer | Task 5 |
| `10/8`, outer `12`, source preservation `1e-3` | Task 6 |
| Verified cache inputs and recommended hyperparameters | Task 7 |
| AMP, clipping, RNG-complete resume, loss accumulator, best/last | Tasks 8, 10 |
| PDM-free selector, Bézier heading, selected ADE/FDE/heading, minADE@32/minFDE@32 | Task 9 |
| Atomic epoch JSON and rebuilt JSONL/CSV/three-panel PNG curves | Task 10 |
| Canonical state hash and allowed-only merge | Task 11 |
| Atomic 972-key export and external metadata | Task 12 |
| Exact-ID profile and mandatory identity preflight | Task 13 |
| Real training/export/strict-load/finite output | Task 14 |
