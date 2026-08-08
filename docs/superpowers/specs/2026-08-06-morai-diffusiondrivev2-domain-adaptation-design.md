# MORAI DiffusionDriveV2 입력 도메인 적응 설계

- 상태: 사용자 승인 완료
- 작성일: 2026-08-06
- 대상 저장소: MORAI-diffusion-drive-v2
- 규범적 입력 계약: docs/domain-specs/morai-diffusiondrivev2-v1.md
- clock 감사 근거: docs/domain-specs/morai-ddv2-clock-audit-20260808.md

## 1. 목적

MORAI에서 변경된 카메라·LiDAR 위치, 영상 렌더링 및 센서 분포에 DiffusionDriveV2의 입력 표현을 적응시킨다. 기존 주행 정책과 V2 selector는 유지하고, 센서 encoder와 cross-modal fusion만 지도학습으로 미세조정한다.

이번 작업은 다음 결과를 제공한다.

1. ROS1 bag을 버전이 있는 학습 cache로 변환하는 전처리기
2. 입력 encoder/fusion만 갱신하는 독립 PyTorch 학습 루프
3. 기존 DiffusionDriveV2 SEL checkpoint schema를 유지하는 exporter
4. E2E-OpenLoop-ROS-Validator와 동일한 전처리 계약 및 평가 연결
5. 학습과 배포가 공유하는 MORAI 도메인 스펙 및 설정

## 2. 범위 밖

- 객체 detection head 재학습
- HD map·BEV semantic head 재학습
- PDM reward를 사용하는 V2 RL 재학습
- V2 selector scorer 재학습
- 센서로 객체 GT나 HD map pseudo-label 생성
- bag에 없는 rear-axle 위치 추측 또는 lever-arm 보정

## 3. 고정 기준

| 구성요소 | 기준 |
|---|---|
| DiffusionDrive 참고 구현 | hustvl/DiffusionDrive main@9b52ed0ec06b073d82d6f392ab084c7b301c8681 |
| DiffusionDriveV2 모델 | hustvl/DiffusionDriveV2@1cd12a1e155c34dcc471261835444c8d5587580b |
| ROS Validator | Team-Stier/E2E-OpenLoop-ROS-Validator@09a3b88e19ae3354903ed676a35e02416284cb58 |
| 프로젝트 센서 preset | sensor_data/0725demo_ros.json |
| 센서 preset SHA-256 | b7e69c70cc0b2948c3a5b352eee022be04f8884a1df135697f89bb4bb88263fb |
| 원본 물리 preset | /home/stier/catkin_ws/src/0725demo.json |
| 원본 물리 preset SHA-256 | cbe7dec43bf48299eed6008d03e722ecdfc9a5d21a75130b7c21d99cf9344958 |

프로젝트에는 DiffusionDriveV2와 Validator를 third_party 아래 고정 commit의 git submodule로 추가한다. DiffusionDrive 원본 저장소는 V2 checkout에 포함된 IL 코드와 공식 URL/commit을 참고 기준으로만 사용한다.

## 4. 데이터 현황과 학습 자격

/home/stier/rosbag/just_drive에는 40개 .bag과 1개 .bag.active가 있다. 모든 파일에 전·좌·우 JPEG, LiDAR, IMU, GPS 및 수동 제어가 있다. 앞쪽 31개에는 /Ego_topic과 /tf가 있고, 마지막 10개에는 두 topic이 없다.

v1 지도학습 cache는 다음 조건을 모두 만족한 bag만 사용한다.

- 확장자가 .bag이며 기록이 종료되어 있다.
- 세 카메라, /lidar3D, /Ego_topic, /tf가 존재한다.
- 센서 frame ID와 해상도가 0725 preset과 일치한다.
- 파일 SHA-256이 manifest에 기록된다.

현재 조건을 만족하는 파일은 30개다. .bag.active는 완료·고정된 뒤에만 포함할 수 있다. 세 camera와 LiDAR는 있지만 `/tf`와 `/Ego_topic`이 둘 다 없는 마지막 10개는 `SENSOR_ONLY_EXCLUDED`로 분류한다. 두 GT topic 중 하나만 없거나 core sensor topic이 빠진 finalized bag은 reject한다. GPS/IMU 기반 pose 추정은 별도 버전의 데이터 계약으로 취급한다.

분할은 frame이 아니라 bag을 단위로 한다. 문자열 20260806:과 eligible relative bag path의 SHA-256 digest로 정렬하고 앞의 floor(0.8×N)를 train, 나머지를 validation으로 사용한다. 현재 N=30이면 24/6이다. 생성된 분할 목록은 manifest에 저장하고 이후 실행에서 재사용한다.

## 5. 전체 구조

~~~mermaid
flowchart LR
    B[ROS1 bags] --> I[Inventory and quality gate]
    S[0725 sensor preset] --> C[Resolved domain config]
    D[MORAI domain YAML] --> C
    C --> P[Shared pure preprocessing library]
    I --> P
    P --> H[Per-bag HDF5 cache and manifest]
    H --> T[Domain adaptation trainer]
    W[Official V2 SEL checkpoint] --> T
    T --> E[Schema-preserving SEL checkpoint]
    E --> V[E2E OpenLoop ROS Validator]
    C --> V
~~~

학습 전처리와 Validator 전처리는 별도 수학 구현을 가지지 않는다. 둘 다 이 저장소의 순수 NumPy 전처리 패키지를 호출한다. ROS node는 메시지 decode, timestamp buffering, publish만 담당한다.

## 6. 설정 소유권

설정은 세 계층으로 분리한다.

1. sensor_data/0725demo_ros.json
   - MORAI의 물리 센서 위치, 회전, 해상도, FOV, frame ID, ROS topic
   - 사람이 직접 복제한 extrinsic을 다른 파일에 두지 않는다.
   - 모든 프로젝트 코드가 read-only user input으로 취급하고 생성·복구·덮어쓰기·재포맷·자동 git-add하지 않는다.
2. configs/domains/morai-ddv2-v1-oracle-command.yaml
   - 동기화, camera adapter, LiDAR self-mask/BEV, target, command 및 품질 정책
   - sensor preset 상대경로와 예상 SHA-256을 가진다.
3. resolved-domain.json
   - 전처리 시작 시 두 설정을 합쳐 cache에 자동 기록하는 불변 snapshot
   - 입력 파일 hash, 코드 commit, 실제 해석된 수치 및 domain ID를 포함한다.

domain resolver만 sensor preset과 domain YAML을 읽고 canonical `resolved-domain.json`을 만든다. preprocessor는 생성 직후 snapshot을 사용하고, trainer/exporter/Validator는 artifact에 결속된 바로 그 snapshot만 로드한다. 이 세 consumer가 raw YAML과 preset을 독립적으로 다시 합치면 안 된다. 대신 snapshot에 기록된 source hash와 현재 파일 hash를 검증한다. 외부 `/home/stier/catkin_ws/src/0725demo.json` 경로는 provenance일 뿐 runtime dependency가 아니다.

코드에는 센서 translation, rotation, 해상도 또는 topic을 하드코딩하지 않는다. preset hash가 다르면 기존 cache를 재사용하지 않고 명시적으로 새 domain version을 요구한다.

`morai-ddv2-v1-oracle-command`는 corrected clock audit 시점에 첫 production artifact가 publish되지 않은 pre-production domain이다. 이 정정은 첫 publication 전에 같은 ID로 고정한다. 첫 production cache/checkpoint/Validator artifact가 publish된 뒤 clock policy 또는 cache schema를 바꾸려면 기존 ID를 수정하지 않고 새 domain ID를 발급한다.

## 7. 전처리 구성요소

### 7.1 Inventory와 동기화

30-bag clock audit 결과 Header 하나로 모든 topic을 동기화할 수 없으므로 Option D dual-clock hybrid를 사용한다. bag receipt는 recorder observation/order clock이지 acquisition time이 아니다. raw ROS record는 `header_stamp_ns`, `receipt_stamp_ns`, `stable_record_index`를 항상 함께 가지며 ambiguous `stamp_ns`를 사용하지 않는다. production에서 affine receipt→Header bridge는 금지한다.

grid 원점은 decode/frame/resolution 검사를 통과한 최소 front receipt `first_valid_front_receipt_ns`에 대해 `ceil_div(first_valid_front_receipt_ns,500000000)*500000000`으로 정한다. 모든 계산은 integer nanoseconds이고 tolerance 경계는 포함한다.

- front: receipt grid 기준 50 ms, tie `(abs receipt offset, receipt, stable index)`; receipt-grid order에서 selected Header strict 증가·유일성과 stable record 미재사용을 assert하고 위반 시 bag-level `FRONT_TIMESTAMP_ORDER_VIOLATION`
- left/right: Header 30 ms와 receipt 100 ms를 먼저 dual-filter, tie `(abs Header offset, Header, stable index)`
- LiDAR: selected-front receipt 기준 45 ms, receipt tie; LiDAR Header는 provenance only
- trajectory-anchor TF: duplicate-collapse 전 raw `map→base_link` view에서 selected-front receipt 기준 50 ms receipt tie; 선택 raw record의 original index와 exact current pose 보존
- Ego: raw Ego receipt-order strict-Header-decrease zero-based epoch를 별도로 만들고 TF anchor receipt-nearest raw Ego가 고른 epoch 안에서만 native Header `T0`를 bracket, receipt tolerance 없이 gap 최대 100 ms
- future TF: `T0+0.5,...,T0+4.0 s`를 선택 raw anchor부터 같은 zero-based TF epoch의 causal suffix에서만 Header bracket, gap 최대 100 ms
- nearest record가 이후 gate를 통과하지 않아도 두 번째 후보로 fallback하지 않음
- 하나라도 초과하거나 future가 epoch/reset 또는 bag 경계를 넘으면 anchor를 drop

TF clock epoch는 raw `map -> base_link` record를 `(receipt_stamp_ns,stable_record_index)`로 순회할 때 Header가 nondecreasing인 maximal 구간이며 index는 0부터 시작한다. strict Header backstep에서만 새 epoch를 시작한다. raw anchor view와 immutable epoch/interpolation view를 분리한다. future는 raw anchor를 포함하는 same-epoch causal suffix를 먼저 만든 후 equivalent equal-Header만 earliest-at-or-after-anchor로 collapse하며 within-epoch conflict는 bag reject다. 같은 Header의 cross-epoch reuse는 epoch index가 disambiguate하므로 허용한다.

full-stream TF/Ego gate는 각 raw Ego record의 receipt-nearest raw TF가 고른 TF epoch-local collapsed Header view만 사용한다. no-bracket/ambiguous/cross-epoch-prevented record는 분리 exclude/report하고 raw Ego, selected TF epoch/index와 bracket provenance를 남긴다. receipt-aligned/global-Header/affine diagnostic은 금지한다.

drop 사유는 bag별·topic별 counter와 sample manifest에 기록한다.

### 7.2 Camera

카메라 한 장이 3×256×1024인 것이 아니다. 원본 HWC는 left/right 480×640×3, front 720×1280×3이고, 세 뷰를 합친 최종 모델 tensor가 RGB CHW 3×256×1024다.

v1 adapter는 각 원본의 종횡비를 보존한다.

1. JPEG를 decode하고 BGR이면 RGB로 변환
2. INTER_AREA로 left/right를 341×256, front를 455×256으로 축소
3. LEFT, FRONT, RIGHT 순서로 1137×256 strip 생성
4. strip의 바깥쪽만 x=[56,1080)으로 crop
5. 최종 HWC 256×1024×3 uint8
6. loader에서 CHW float32로 변환하고 255로 나눔

최종 경계는 left [0,285), front [285,740), right [740,1024)다. ImageNet mean/std normalization, view blending, cylindrical warp, padding은 적용하지 않는다. 0725 preset의 FOV와 extrinsic은 입력 검증 및 provenance에 사용하지만, upstream이 perspective image를 직접 stitch하므로 v1에서 새로운 projection을 도입하지 않는다.

0725 preset의 camera Euler는 MORAI convention이며 ROS 고정축 RPY가 아니다. roll=0인 이 preset은 Hamilton 곱 기준 `q_y(pitch) ⊗ q_z(yaw)`로 변환하고, bag에서 확인된 `base_link -> Camera-{1,2,3}` quaternion을 canonical contract로 검증한다. frame 이름은 leading slash를 제거한 뒤 비교하고 quaternion `q/-q`를 동일하게 취급한다. nonzero roll은 추측하지 않고 새 domain version을 요구한다.

### 7.3 LiDAR

PointCloud2의 x, y, z만 사용하고 intensity는 사용하지 않는다.

1. 정확한 (0,0,0) padding point 제거
2. 0725 preset과 bag `/tf`가 함께 확인한 T_base_link_from_lidar, translation [1.5,0,1.25] m와 identity quaternion으로 base_link 변환
3. ego roof/body self-mask 제거
4. 0.2 < z < 100 조건 적용
5. x/y 각각 [-32,32] m를 0.25 m bin으로 histogram
6. count를 5에서 clip하고 5로 나눔

self-mask는 base_link 좌표의 x=[0,2] m, y=[-1,1] m, z=[0.8,1.6] m다. z 제한 없는 XY prism은 근거리 도로를 지울 수 있으므로 금지한다.

출력은 float32 1×256×256이며 첫 공간축이 x-bin, 두 번째가 y-bin이다. rotate/flip은 하지 않는다.

### 7.4 Ego status

status_feature은 다음 순서의 float32 8-vector다.

~~~text
[LEFT, STRAIGHT, RIGHT, UNKNOWN, vx, vy, ax, ay]
~~~

/Ego_topic의 velocity와 acceleration은 이미 m/s, m/s²의 body frame x-forward/y-left 값이다. 회전, km/h 변환 또는 추가 normalization을 하지 않는다.

### 7.5 Trajectory와 command

현재와 미래 pose의 canonical source는 `/tf map -> base_link`다. current pose는 receipt-nearest raw trajectory-anchor TF record를 exact 사용하고 original Header/receipt/index bracket을 동일 record 두 개, weight 0으로 기록한다. future target이 exact selected-suffix Header source면 같은 규칙을 쓰고, 아니면 그 suffix interpolation view의 바로 전후 Header만 사용한다. translation은 bracket 안에서 선형 보간한다. quaternion은 정규화하고 dot<0이면 두 번째 quaternion 부호를 뒤집은 shortest-arc SLERP를 적용하며, dot>0.9995에서는 normalized lerp를 쓴다. 결과 quaternion도 정규화한다. bracket 간격은 최대 100 ms이고 extrapolation, TF epoch crossing과 anchor 이전 receipt-order record 사용은 금지한다. `/Ego_topic.position`과 degree 단위 `heading`은 epoch-safe full-stream TF 일치 진단에만 사용하며 XY/heading nearest-rank p99가 각각 0.05 m/0.5° 이하여야 한다. `/Ego_topic` velocity/acceleration은 TF anchor receipt-nearest raw Ego가 선택한 독립 zero-based Ego epoch의 collapsed Header view에서만 `T0`를 bracket한다.

현재 base_link pose를 원점으로 미래 t+0.5, ..., t+4.0초 pose를 상대변환한다. target은 8×3의 (x_m, y_m, relative_heading_rad)이며 heading은 [-π,π)로 wrap한다.

Diffusion latent와 plan anchor는 XY 8×2지만 decoder와 stock IL target/prediction은 8×3이다. nearest-anchor mode 선택은 XY만 사용하고 stock trajectory regression은 x, y, heading 전부를 사용한다. V2 SEL 최종 inference는 decoder heading을 직접 복사하지 않고, control point `(0,0),P1..P8`의 8차 Bézier curve를 `t_k=k/8, k=1..8`에서 미분한 접선 `atan2(dy,dx)`로 heading을 재생성해 최종 [8,3]을 반환한다.

학습 command는 같은 미래경로에서 결정적으로 만든다.

path length는 origin (0,0)에서 시작해 8개 XY target 사이의 Euclidean segment 길이를 모두 합한 값이고, heading과 end y는 4.0초 target을 사용한다.

- path length < 1.0 m: UNKNOWN
- heading ≥ 0.20 rad이고 end y ≥ 0.50 m: LEFT
- heading ≤ -0.20 rad이고 end y ≤ -0.50 m: RIGHT
- |heading| ≤ 0.10 rad이고 |end y| ≤ 0.50 m: STRAIGHT
- 나머지 또는 조건의 부호가 충돌: UNKNOWN

NaN command는 허용하지 않는다. 이 v1의 domain ID는 `morai-ddv2-v1-oracle-command`이며 미래 GT를 입력 command에 사용하므로 명시적으로 non-causal이다. 학습, held-out validation, ROS bag-playback Validator는 모두 bag SHA-256과 timestamp로 조회하는 동일한 command sidecar를 사용하고 metric을 `oracle-command`로 표시한다. sidecar 누락·불일치는 평가 오류이며 UNKNOWN fallback을 사용하지 않는다. UNKNOWN은 위 미래경로 분류 규칙이 정상적으로 생성한 정지·애매 class에만 쓴다.

route/planner command 또는 all-UNKNOWN을 사용하는 실시간 causal 경로는 `morai-ddv2-v1-causal`이라는 별도 domain/cache/checkpoint가 필요하다. 두 domain의 artifact는 혼합하지 않는다. 현재 Validator의 고정 STRAIGHT fallback은 두 경우 모두 제거한다.

## 8. Cache 계약

cache/HDF5, command sidecar와 manifest schema는 2다. resolved root schema 1은 유지할 수 있지만 clock-policy 변경이 resolved-domain hash에 포함된다. cache는 bag 하나당 `bags/<domain_id>/<resolved_domain_sha256>/<source_bag_sha256>.h5`, command sidecar와 전체 JSONL manifest로 구성한다. sample ID는 `<source_bag_sha256>:<timestamp_ns>`이고 `timestamp_ns`는 selected-front Header다. selected front Header는 receipt-grid 순서로 strict 증가·유일하고 stable record도 재사용되지 않아야 한다.

필수 배열은 정확히 28개다.

| 이름 | cache dtype/shape | loader 출력 |
|---|---|---|
| camera_rgb | uint8 [N,256,1024,3] | float32 [N,3,256,1024], [0,1] |
| lidar_count | uint8 [N,256,256], 0..5 | float32 [N,1,256,256], count/5 |
| status | float32 [N,8] | 동일 |
| trajectory | float32 [N,8,3] | 동일 |
| timestamp_ns | int64 [N] | selected-front Header |
| grid_receipt_ns | int64 [N] | receipt-grid target |
| sensor_source_header_stamp_ns | int64 [N,4] | front/left/right/lidar Header |
| sensor_source_receipt_ns | int64 [N,4] | front/left/right/lidar receipt |
| sensor_source_record_index | int64 [N,4] | 원본 topic stable record index |
| side_camera_header_offset_ns | int64 [N,2] | left/right Header minus front Header |
| sensor_receipt_offset_ns | int64 [N,3] | left/right/lidar receipt minus front receipt |
| trajectory_anchor_tf_header_ns | int64 [N] | exact current TF native Header T0 |
| trajectory_anchor_tf_receipt_ns | int64 [N] | exact current TF receipt |
| trajectory_anchor_tf_record_index | int64 [N] | flattened TF stable index |
| trajectory_anchor_tf_clock_epoch_index | int64 [N] | receipt-order TF epoch |
| ego_bracket_clock_epoch_index | int64 [N] | independent receipt-order Ego epoch |
| ego_bracket_header_stamp_ns | int64 [N,2] | Ego Header bracket |
| ego_bracket_receipt_ns | int64 [N,2] | 같은 Ego source의 receipt provenance |
| ego_bracket_record_index | int64 [N,2] | Ego source record index |
| ego_interp_weight | float32 [N] | Ego interpolation weight |
| current_pose_bracket_header_stamp_ns | int64 [N,2] | exact `[T0,T0]` |
| current_pose_bracket_receipt_ns | int64 [N,2] | exact `[R0,R0]` |
| current_pose_bracket_record_index | int64 [N,2] | exact `[i0,i0]` |
| current_pose_interp_weight | float32 [N] | exact zero |
| future_pose_bracket_header_stamp_ns | int64 [N,8,2] | future TF Header brackets |
| future_pose_bracket_receipt_ns | int64 [N,8,2] | 같은 TF source의 receipt provenance |
| future_pose_bracket_record_index | int64 [N,8,2] | future TF source record index |
| future_pose_interp_weight | float32 [N,8] | future TF weights |

Header와 receipt를 한 offset field에 섞지 않는다. LiDAR Header는 source provenance에만 남고 sync offset으로 해석하지 않는다. 28 HDF5 datasets 중 camera/LiDAR/status/trajectory 네 tensor를 제외한 24 provenance values를 manifest sample record가 같은 이름과 값으로 전부 반복한다. tensor payload를 JSONL에 복제하지 않는다.

HDF5 attribute는 `cache_schema_version=2`와 `clock_policy_version=1`을 identity hash들과 함께 저장한다. command sidecar는 `commands/<domain_id>/<resolved_domain_sha256>/<source_bag_sha256>.jsonl`에 생성한다. 각 timestamp-sorted 13-field record는 `schema_version=2`, domain/resolved-domain/sensor-preset/source-bag/command-rule SHA-256, `timestamp_ns`, `grid_receipt_ns`, `front_receipt_ns`, `front_record_index`, command index/name과 `future_trajectory_oracle` provenance를 저장하고 cache status one-hot과 exact하게 일치해야 한다. lookup key는 계속 bag SHA-256+front Header `timestamp_ns`다.

HDF5와 sidecar를 각각 `.partial`에 쓰고 fsync·atomic rename한 뒤 hash와 count를 검증한다. 마지막에 `manifests/<domain_id>/<resolved_domain_sha256>/manifest.jsonl`을 `.partial`에서 atomic rename하고 parent directory를 fsync해 dataset commit marker로 삼는다. loader는 final manifest가 참조한 hash-valid artifact만 사용하며 orphan과 `.partial`은 무시한다.

## 9. 파인튜닝

### 9.1 초기화와 trainable 경계

공식 DiffusionDriveV2 SEL checkpoint 전체를 authoritative export container로 로드한다. raw 972-key state_dict는 별도로 보존한다. 학습 network 초기화 때만 `agent._transfuser_model.` prefix를 제거하고 `_trajectory_head.vocab` 정확히 한 key를 제외한 971개 key를 `strict=True`로 load한다. export는 보존한 raw key set과 vocab 값을 유지한다.

고정 SEL 모델의 public forward는 IL 학습에 사용하지 않는다. 이 forward는 PDM/RL 경로를 전제로 하고, 참고 DiffusionDrive IL의 2-layer decoder와 달리 SEL checkpoint에는 decoder layer가 하나만 있기 때문이다.

프로젝트는 parameter와 buffer를 소유하지 않는 `forward_trajectory_supervised` adapter를 둔다. 이 adapter는 SEL 모델의 shared feature graph와 기존 `_trajectory_head.diff_decoder.layers.0`을 직접 호출한다. PDM, RL, agent/BEV loss와 selector 경로는 호출하지 않는다. 2-layer DiffusionDrive IL 모델은 지도학습 noise 절차와 loss 의미를 확인하는 참고 구현일 뿐 instantiate하거나 export하지 않는다.

지도학습 candidate 생성은 정확히 한 anchor group만 사용한다. XY `plan_anchor` [20,8,2]를 batch에 복제한 [B,20,8,2]에 V2 normalization `x/50, y/20`을 적용한다. batch별 `t ~ UniformInt{0,...,49}`와 같은 shape의 `epsilon ~ N(0,I)`를 생성하고 `z_t = diffusion_scheduler.add_noise(normalized_anchor, epsilon, t).float()`를 계산한다. `z_t`를 [-1,1]로 clamp한 뒤 `x*50, y*20`으로 denormalize하여 decoder input으로 사용한다. 학습 checkpoint는 deterministic resume을 위해 Python, NumPy, Torch CPU와 모든 CUDA RNG state를 함께 저장·복원한다.

decoder의 one-group source 출력은 `poses_reg_list[0]` [B,20,8,3]과 `poses_cls_list[0]` [B,1,20]이다. adapter는 layer 수가 정확히 1인지 assert하고 `poses_cls_list[0].squeeze(1)`이 [B,20]인지 검증한다. `targets={"trajectory": target_trajectory}`와 `target_trajectory` [B,8,3]을 만들어 기존 `LossComputer(poses_reg, poses_cls, targets, plan_anchor)`를 호출한다.

normalized checkpoint name 기준 trainable parameter prefix는 다음 다섯 개뿐이다.

~~~text
_backbone.image_encoder.
_backbone.lidar_encoder.
_backbone.transformers.
_backbone.lidar_channel_to_img.
_backbone.img_channel_to_lidar.
~~~

그 외 parameter는 requires_grad=False와 eval mode로 고정한다. 특히 다음은 동결한다.

- _backbone.c5_conv 및 up_conv 계열
- _bev_downscale와 bev_proj
- _keyval_embedding, _query_embedding 및 _tf_decoder
- _status_encoding
- _trajectory_head 전체
- _agent_head와 _bev_semantic_head
- V2 coarse/fine selector scorer

optimizer에는 위 allowlist의 parameter만 들어간다. 학습 중 trainable encoder의 BatchNorm running_mean, running_var 및 num_batches_tracked는 갱신 가능한 export state로 취급한다. 학습 시작 시 trainable/frozen parameter 목록, buffer 정책 및 개수를 출력하고 checkpoint metadata에 저장한다.

### 9.2 Loss

SEL checkpoint의 단일 trajectory decoder layer에 대해:

~~~text
d[b,m] = mean_t(||GT_XY[b,t] - anchor_XY[m,t]||_2)
m_star[b] = argmin_m d[b,m]
L_cls = sigmoid_focal_mean(mode_logits, one_hot(m_star,20), alpha=0.25, gamma=2)
L_reg = mean_abs(pred_pose[b,m_star,:,x/y/heading] - GT_pose[b,:,x/y/heading])
L_traj = 10 × L_cls + 8 × L_reg

L_sp = mean((theta_trainable - theta_pretrained)^2)
L_finetune = 12 × L_traj + 1e-3 × L_sp
~~~

`L_cls`는 20개 binary sigmoid focal element 전체의 mean이고 `L_reg`는 batch×8×3 element 전체의 mean이다. heading도 stock raw scalar L1이며 ±π circular wrap을 loss에서 다시 적용하지 않는다. 학습 regression mode는 predicted argmax가 아니라 GT XY로 정한 `m_star`다. `L_sp`는 allowlist parameter의 모든 scalar를 합친 element mean이다.

독립 학습 루프는 TransfuserAgent.compute_loss, transfuser_loss 및 TransfuserTargetBuilder를 호출하지 않는다. cache의 trajectory [B,8,3]만 supervised adapter와 LossComputer에 전달한다. agent/BEV frozen head는 forward에서 우회한다. 객체, bounding-box, BEV semantic, PDM reward 및 selector loss는 계산하지 않으며 DiffusionDriveV2 RL 경로의 XY-only loss도 사용하지 않는다.

학습 루프는 AMP, gradient clipping, deterministic seed, resume, best/last checkpoint를 지원한다. validation 기준은 learned selector가 최종 선택한 trajectory의 sample-micro ADE를 우선하고 FDE와 heading MAE를 함께 기록한다. 추가 진단으로 같은 forward의 post-Bézier coarse top-32 후보에 대한 `minADE@32`와 `minFDE@32`를 기록하지만, 이 oracle 진단값은 checkpoint 선택에 사용하지 않는다.

### 9.3 Checkpoint export

원본 raw SEL state_dict 972개 key와 agent._transfuser_model. prefix를 authoritative container로 삼는다. trainable prefix에 속하는 모든 state_dict entry를 병합하며, 여기에는 parameter뿐 아니라 해당 encoder의 BatchNorm buffer도 포함한다. 그 외 trajectory head, selector 및 frozen module tensor는 byte-for-byte 원본 값이어야 한다.

export 검증:

- 원본 raw checkpoint와 동일한 972-key set과 tensor shape
- 허용된 key 외 tensor hash 불변
- Validator loader가 prefix를 제거하고 PDM-only _trajectory_head.vocab 한 key를 제외한 971개 key를 derivative에 strict=True로 load
- 고정 golden sample에서 finite 8×3 출력

metadata를 state_dict에 추가하면 972-key schema가 깨지므로 `<checkpoint>.metadata.json` 외부 sidecar에만 기록한다. checkpoint file을 `.partial`에서 fsync·atomic rename하고 실제 file hash를 계산한 다음 metadata를 마지막에 atomic publish한다. metadata는 최소한 `base_checkpoint_sha256`, `base_state_dict_sha256`, `export_checkpoint_sha256`, `export_state_dict_sha256`, `resolved_domain_sha256`, `sensor_preset_sha256`, `training_manifest_sha256`, source commit, trainable prefix와 972/971 key count를 가진다. 자기참조 `metadata_sha256`은 metadata 내부에 넣지 않는다.

state_dict hash는 raw 972개 key 각각에 대해 contiguous CPU tensor의 little-endian C-order raw byte SHA-256을 만든 뒤 `{key,dtype,shape,data_sha256}` entry를 key의 UTF-8 byte 순서로 정렬한 RFC 8785 canonical JSON SHA-256이다. Validator는 checkpoint file hash와 이 tensor hash를 모두 검증한다.

exporter는 Validator의 official `src/Validator_pkg/config/profile.json`을 덮어쓰지 않고 프로젝트의 `configs/validator/profile-morai-ddv2-oracle.json`을 생성한다. pinned Validator의 exact-ID allowlist를 유지하기 위해 이 별도 파일 안의 model entry ID는 계속 `diffusiondrive_v2`다. entry는 export checkpoint, `<checkpoint>.metadata.json`, resolved-domain.json과 sensor preset을 각각 required runtime asset으로 선언하고 실제 file SHA-256을 pin한다. profile identity에는 domain ID, resolved-domain, exported state_dict 및 training manifest SHA-256도 넣는다.

Validator는 `~profile_file=<absolute-path-to-profile-morai-ddv2-oracle.json>`와 `~model_id=diffusiondrive_v2`로 실행한다. 새 model ID를 추가하거나 기존 model/worker/environment allowlist를 우회하면 안 된다.

Validator는 기존 `verify_runtime_assets`로 profile asset hash를 먼저 확인한 뒤 metadata를 필수로 읽는다. profile checkpoint SHA, metadata의 `export_checkpoint_sha256` 및 실제 checkpoint file SHA가 모두 같아야 한다. 이어 recompute한 `export_state_dict_sha256`과 profile/metadata의 domain·sensor·training-manifest identity를 확인한 다음에만 972→971 strict load한다. metadata 누락이나 어느 hash 불일치도 model load 전에 실패시킨다.

## 10. 평가

### 10.1 Offline validation

held-out bag cache 평가는 network를 eval mode로 두고 Validator와 동일한 PDM-free learned selector inference를 실행한다. GT-nearest `m_star` 또는 GT trajectory는 candidate 선택에 사용하지 않고, selector가 반환한 최종 `trajectory` [B,8,3]만 metric 입력으로 사용한다.

학습 곡선용 oracle 진단은 배포 선택 결과를 바꾸지 않는다. validation-only context manager가 한 번의 정상 public forward 동안 `bezier_xyyaw`가 반환한 변경 전 post-Bézier proposal pool `[B,800,8,3]`과 `_select_topk`의 `topk_idx [B,32]`를 임시로 capture하고 `finally`에서 원래 callable을 복원한다. capture 유무와 관계없이 public `trajectory [B,8,3]`는 bitwise identical이어야 하며, public selected trajectory가 capture한 top-32 중 하나와 exact하게 일치해야 한다. full 800개 proposal, 4개 stage winner, GT-nearest mode는 `minADE@32`/`minFDE@32` 후보로 사용하지 않는다.

sample `i`, shortlist rank `k`, horizon `t`에 대해 `ADE[i,k]=mean_t ||C[i,k,t,xy]-G[i,t,xy]||_2`, `FDE[i,k]=||C[i,k,8,xy]-G[i,8,xy]||_2`다. `minADE@32=mean_i min_k ADE[i,k]`, `minFDE@32=mean_i min_k FDE[i,k]`이며 두 metric은 독립 argmin을 사용하고 후보를 deduplicate하지 않는다. XY만 평가하고 float64로 누적한 뒤 finite JSON number로 기록한다. candidate source는 정확히 `coarse_top32_post_bezier`, `k=32`로 명시한다.

확률적 diffusion sampling도 Validator와 동일하게 고정한다. `validation_run_id`는 `morai-ddv2-v1-oracle-command-validation-v1`이고 각 sample seed는 SHA-256 문자열 `diffusiondrive_v2:<timestamp_ns>:<validation_run_id>`의 앞 8 byte를 unsigned big-endian integer로 해석한다. offline과 ROS 평가는 exact parity를 위해 inference batch size 1을 사용하고 매 sample forward 직전에 Torch CPU와 모든 CUDA RNG를 해당 seed로 재설정한다. 따라서 sample 순서를 바꿔도 sample ID별 출력이 같아야 한다. 보고서에 run ID와 seed를 기록한다.

다음을 계산한다.

- ADE: sample별 8개 XY Euclidean error mean의 전체 sample mean
- FDE: sample별 4.0초 XY Euclidean error의 전체 sample mean
- minADE@32/minFDE@32: 같은 forward의 `coarse_top32_post_bezier` 후보에 대한 독립 best-of-32 XY 진단
- heading MAE: `abs(wrap_to_pi(pred_heading-target_heading))`의 sample×time mean
- LEFT/STRAIGHT/RIGHT/UNKNOWN별 ADE/FDE
- 모든 결과에 `morai-ddv2-v1-oracle-command`와 `oracle-command` label
- drop률과 sensor sync 통계

best checkpoint 기준은 위 sample-micro ADE다. bag별 ADE 평균의 macro mean도 함께 기록한다. command class에 sample이 없으면 count=0과 metric=null을 기록하며 NaN을 쓰지 않는다.

매 epoch validation에는 selector metric과 별도로 supervised adapter의 validation loss도 계산한다. batch size 1에서 각 sample 직전 UTF-8 `diffusiondrive_v2_supervised_loss:<timestamp_ns>:<validation_run_id>`의 SHA-256 앞 8 byte를 unsigned big-endian seed로 사용해 Torch CPU/CUDA RNG를 재설정하고, `no_grad` 아래 같은 timestep/noise sampling과 `L_finetune=12*L_traj+1e-3*L_sp`를 실행한다. train/validation의 total/trajectory/preservation loss는 sample-weighted mean이며 validation loss forward 뒤 training boundary를 exact하게 복원한다.

완료된 epoch의 authoritative telemetry는 `runs/.../metrics/epochs/epoch-0000.json` 형식의 RFC 8785 canonical JSON이다. sibling `.partial`, file fsync, `os.replace`, parent-directory fsync로 원자 publish하며 schema version, epoch/global step/sample 수, config/manifest/base-checkpoint hash, train 및 validation loss 세 component, selected ADE/FDE/heading/bag-macro, `candidate_source=coarse_top32_post_bezier`, `k=32`, `min_ade`, `min_fde`, validation report SHA-256와 best-checkpoint 상태를 기록한다. wall-clock field와 NaN/Infinity는 금지한다.

`last.pt`는 mid-epoch resume 시 곡선 누락을 막기 위해 현재 epoch의 completed batch/sample 수, 세 loss의 sample-weighted numerator와 grad-norm step sum/count를 가진 `EpochAccumulatorState`를 저장한다. epoch JSON을 먼저 durable publish한 뒤 sampler를 다음 epoch로 advance하고 해당 epoch-record hash를 포함한 `last.pt`를 durable publish한다. resume 시 기존 epoch JSON bytes가 같으면 idempotent하게 받아들이고 다르면 실패한다. next-epoch checkpoint가 가리키는 epoch JSON이 없으면 실패하며, epoch-end cursor와 기존 JSON이 있고 checkpoint가 아직 advance되지 않은 crash 상태는 검증 후 finish/publish를 재개한다.

`metrics/history.jsonl`, `metrics/epoch_metrics.csv`, `plots/training-curves.png`는 authoritative epoch JSON들을 epoch 번호순으로 다시 읽어 매번 atomic rebuild하는 파생물이다. CSV는 epoch, global step, train/validation total·trajectory·preservation loss, selected ADE/FDE/heading, minADE@32/minFDE@32와 best flag를 가진다. PNG는 (1) train/validation total loss, (2) selected ADE/FDE, (3) minADE@32/minFDE@32의 세 panel을 가진다. history에 blind append하지 않고 plot을 resume state로 사용하지 않는다.

### 10.2 ROS Validator

Validator를 submodule로 고정하고 다음만 연결한다.

- shared dual-clock policy, camera, LiDAR, Ego interpolation/status assembly 및 command-sidecar lookup package 사용
- schema-v2 manifest의 per-topic stable index, independent TF/Ego epoch indices와 Ego/current/future bracket을 callback counter로 exact replay하고 callback now/Header nearest 재선택 금지; callback은 observed Header만 검증하고 exact bag receipt provenance는 manifest에서 공급
- nonmonotonic LiDAR buffer는 Header/receipt cutoff가 아니라 다음 manifest-required stable index로 prune
- checkpoint/cache와 동일 hash의 canonical resolved-domain.json 로드 및 source preset/domain YAML hash 검증
- official baseline을 보존한 별도 MORAI profile file의 `model_id=diffusiondrive_v2` entry와 required metadata asset 사용
- bag SHA-256과 timestamp로 oracle command sidecar 조회
- sidecar 누락·불일치 시 즉시 평가 실패
- offline validation과 같은 run ID/seed 규칙 및 PDM-free learned selector 사용
- checkpoint/domain hash 검증
- 기존 PDM-free learned selector inference 유지

Validator 내부의 0716 preset LiDAR translation (2.0,0,1.5), 1920×1080 camera 강제 resize 및 고정 STRAIGHT는 MORAI v1 profile에서 사용하지 않는다.

## 11. 오류 처리

다음 global 오류는 어떤 cache도 쓰기 전에 실행 전체를 즉시 실패시킨다.

- sensor preset hash 불일치
- checkpoint schema 불일치
- trainable allowlist 밖 parameter 변경
- cache와 runtime domain ID 불일치

다음은 offending bag 전체를 reject하며 그 bag의 cache를 만들지 않는다.

- 필수 topic/message type 누락 또는 불일치
- 예상 frame ID·camera resolution 불일치
- canonical `/tf` extrinsic 또는 TF/Ego pose 일치 tolerance 초과
- selected front Header non-increasing/duplicate 또는 stable record 재사용 (`FRONT_TIMESTAMP_ORDER_VIOLATION`)

batch preprocessor는 나머지 bag도 검사해 전체 rejection report를 완성하지만 예상 eligible bag이 하나라도 reject되면 마지막에 nonzero로 종료한다. Validator 평가에서는 oracle command sidecar 누락·bag hash·timestamp 불일치를 즉시 실행 오류로 처리한다.

개별 sample만 drop하고 계속할 수 있는 조건:

- sync tolerance 초과
- 필수 future horizon 부족
- 비유한 pose/status
- 빈 또는 손상된 JPEG/PointCloud2

모든 drop은 이유와 원본 stamp를 남긴다.

## 12. 테스트

1. Config test
   - 0725 preset parse, hash, topic/frame/resolution 및 MORAI Euler-to-canonical-quaternion extrinsic
2. Camera unit/golden test
   - RGB 순서, resize, crop, seam 경계, shape/range
3. LiDAR unit/golden test
   - zero padding, transform, 3D self-mask, bin orientation, clip
4. Timestamp test
   - receipt-grid/front/LiDAR/TF tie, side Header+receipt dual-bound second-candidate, bag boundary
   - selected front Header strict identity bag gate
   - raw TF anchor index 보존, TF native Header reset에서 zero-based epoch 분리, raw-anchor causal same-epoch suffix duplicate collapse
   - receipt-nearest raw Ego가 고른 independent Ego epoch의 current bracket과 epoch-safe full-stream TF/Ego gate provenance/counters
5. Target test
   - `/tf` bracket/shortest-arc SLERP와 q/-q, SE(2) 상대변환, heading wrap, command 경계값
6. Cache round-trip
   - HDF5 → loader tensor exactness, explicit Header/receipt/index/TF·Ego epoch/bracket provenance와 domain hash
7. Freeze test
   - optimizer allowlist와 한 step 뒤 frozen tensor hash
8. Checkpoint compatibility
   - full SEL schema merge와 Validator strict load
9. End-to-end parity
   - 동일 raw fixture/manifest/sidecar에서 offline preprocessor와 ROS Validator가 exact stable indices/brackets를 replay하고 tensor가 수치적으로 동일
   - callback time을 바꿔도 source selection은 변하지 않고 Header-backstep LiDAR prune은 manifest index를 따름
   - validation sample 순서 변경 전후 seed와 selected trajectory가 sample ID별 동일

## 13. 계획된 저장소 구조

~~~text
configs/
  domains/morai-ddv2-v1-oracle-command.yaml
  validator/profile-morai-ddv2-oracle.json
docs/
  domain-specs/morai-diffusiondrivev2-v1.md
  superpowers/specs/2026-08-06-morai-diffusiondrivev2-domain-adaptation-design.md
sensor_data/
  0725demo_ros.json
src/morai_ddv2/
  config/
  preprocess/
  data/
  training/
  export/
scripts/
  inspect_bags.py
  preprocess_bags.py
  train.py
  evaluate.py
third_party/
  DiffusionDriveV2/
  E2E-OpenLoop-ROS-Validator/
tests/
~~~

## 14. 완료 조건

- 완성 GT-rich bag을 deterministic cache로 변환할 수 있다.
- 학습과 Validator가 같은 resolved domain config와 tensor builder를 사용한다.
- 객체 GT와 HD map 없이 trajectory-only loss로 encoder/fusion만 변경된다.
- export checkpoint가 기존 V2 SEL schema로 strict load된다.
- held-out bag과 ROS Validator에서 8×3 trajectory 및 metric을 생성한다.
- 모든 artifact가 sensor, domain, source bag 및 checkpoint hash로 추적 가능하다.
