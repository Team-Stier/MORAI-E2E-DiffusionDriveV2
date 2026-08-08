# MORAI DiffusionDriveV2 Domain Specification v1

- Domain ID: morai-ddv2-v1-oracle-command
- 상태: 사용자 승인 완료
- 작성일: 2026-08-06
- 관련 설계: docs/superpowers/specs/2026-08-06-morai-diffusiondrivev2-domain-adaptation-design.md
- clock 감사 근거: docs/domain-specs/morai-ddv2-clock-audit-20260808.md

이 문서는 ROS bag에서 생성되는 학습 sample과 DiffusionDriveV2가 실제로 받는 tensor의 단일 기준이다. 전처리기, 학습 loader, checkpoint metadata 및 ROS Validator는 모두 이 계약을 만족해야 한다.

규범 용어 MUST, MUST NOT, SHOULD는 각각 필수, 금지, 권장을 뜻한다.

## 1. Source identity

### 1.1 Sensor preset

프로젝트의 권위 있는 센서 파일은 다음과 같다.

~~~yaml
path: sensor_data/0725demo_ros.json
sha256: b7e69c70cc0b2948c3a5b352eee022be04f8884a1df135697f89bb4bb88263fb
physical_source_path: /home/stier/catkin_ws/src/0725demo.json
physical_source_sha256: cbe7dec43bf48299eed6008d03e722ecdfc9a5d21a75130b7c21d99cf9344958
~~~

sensor_data/0725demo_ros.json은 물리 preset과 같은 센서 geometry를 가지며 ROS topic이 채워진 프로젝트용 snapshot이다. 이 파일은 read-only user input이다. resolver, preprocessor, scaffolder, setup을 포함한 모든 프로젝트 코드는 이 파일을 생성·복구·덮어쓰기·재포맷하거나 자동 git-add하면 안 된다. 파일이 없거나 SHA-256이 다르면 복사본을 만들지 말고 시작을 거부해야 한다. `physical_source_path`는 provenance일 뿐 trainer나 Validator의 runtime dependency가 아니다.

domain resolver만 sensor preset과 domain YAML을 결합해 canonical resolved-domain.json을 생성한다. preprocessor, trainer, exporter와 Validator는 artifact에 기록된 동일 snapshot을 사용하며 raw 두 파일을 각자 다시 해석하면 안 된다. consumer는 snapshot에 기록된 source hash가 현재 프로젝트 파일과 일치하는지만 검증한다.

### 1.2 Model source

~~~yaml
diffusiondrive_reference:
  repository: https://github.com/hustvl/DiffusionDrive
  commit: 9b52ed0ec06b073d82d6f392ab084c7b301c8681
diffusiondrive_v2:
  repository: https://github.com/hustvl/DiffusionDriveV2
  commit: 1cd12a1e155c34dcc471261835444c8d5587580b
validator:
  repository: https://github.com/Team-Stier/E2E-OpenLoop-ROS-Validator
  commit: 09a3b88e19ae3354903ed676a35e02416284cb58
~~~

## 2. Coordinate and time conventions

### 2.1 Canonical vehicle frame

모든 geometry는 base_link를 기준으로 한다.

~~~yaml
x: forward
y: left
z: up
distance_unit: meter
angle_unit: radian
heading_positive: counter_clockwise
heading_range: [-pi, pi)
~~~

NAVSIM의 공식 trajectory origin은 rear axle이지만 MORAI bag에는 rear axle frame이나 차량 치수가 없다. v1은 base_link를 명시적인 domain origin으로 사용하고 추정 lever-arm 보정을 금지한다.

### 2.2 Timestamp

- `Header.stamp`와 rosbag read API가 반환하는 bag receipt stamp를 서로 다른 clock axis로 취급해야 한다.
- bag receipt stamp는 recorder가 message를 관측하고 기록한 순서의 clock이다. 센서 acquisition time이라고 주장하거나 물리 상태의 시간축으로 해석하면 안 된다.
- grid와 front/LiDAR/trajectory-anchor TF association에는 receipt axis를 사용한다. side camera exposure association과 Ego/future-pose interpolation에는 native Header axis를 사용한다.
- affine 또는 piecewise receipt→Header 변환을 production sample selection에 사용하면 안 된다.
- offline rosbag record API와 cache provenance field는 반드시 `header_stamp_ns`와 `receipt_stamp_ns`를 명시해야 한다. ROS record에 의미가 불분명한 단일 `stamp_ns` field를 사용하면 안 된다. ROS callback API가 exact bag receipt를 노출하지 않는 runtime에서는 receipt를 callback time이나 `/clock`으로 꾸며내지 않고 verified manifest 값을 사용한다.
- 같은 bag 안에서만 입력 선택과 future interpolation을 수행해야 한다.

~~~yaml
sample_rate_hz: 2.0
grid_period_ns: 500000000
grid_axis: receipt_stamp_ns
grid_origin_receipt_ns: ceil_div(first_valid_front_receipt_ns, 500000000) * 500000000
front_receipt_tolerance_ns: 50000000
side_header_tolerance_ns: 30000000
side_receipt_sanity_tolerance_ns: 100000000
lidar_receipt_tolerance_ns: 45000000
trajectory_anchor_tf_receipt_tolerance_ns: 50000000
ego_state_max_bracket_ms: 100
pose_max_bracket_ms: 100
drop_on_required_sensor_miss: true
~~~

`first_valid_front_receipt_ns`는 다른 sensor sync 결과와 무관하게 JPEG decode, Camera-1 frame 및 1280×720 resolution 검사를 통과한 front message 중 가장 작은 receipt nanosecond 값이다. 모든 grid, tolerance 및 offset 계산은 float seconds가 아니라 signed integer nanoseconds로 수행한다. 모든 tolerance 경계는 `abs(offset_ns) <= tolerance_ns`로 포함한다.

2 Hz receipt grid마다 front는 `(abs(front_receipt-grid_receipt), front_receipt, stable_record_index)` 최소값을 50 ms 안에서 선택한다. 선택된 front의 `Header.stamp`는 `timestamp_ns`, receipt-grid target은 `grid_receipt_ns`다.

left/right는 각각 후보 집합을 먼저 다음 두 inclusive 조건의 교집합으로 제한한다.

1. `abs(side_header-front_header) <= 30 ms`
2. `abs(side_receipt-front_receipt) <= 100 ms`

그 뒤 `(abs(side_header-front_header), side_header, stable_record_index)` 최소값을 선택한다. receipt 조건을 통과하지 않은 Header-nearest frame으로 fallback하면 안 된다. LiDAR는 `(abs(lidar_receipt-front_receipt), lidar_receipt, stable_record_index)` 최소값을 45 ms 안에서 선택하며 LiDAR Header를 sync 결정에 사용하면 안 된다.

`map -> base_link` trajectory anchor TF는 `(abs(tf_receipt-front_receipt), tf_receipt, stable_record_index)` 최소값을 50 ms 안에서 정확히 하나 선택한다. 두 번째 후보로 재선택하지 않는다. 선택한 record의 native Header를 `T0`로 삼고 그 TF pose를 current pose로 exact 사용한다. `/Ego_topic` velocity/acceleration은 native Header axis에서 `T0`를 bracket-only 보간한다.

TF clock epoch는 `map -> base_link` record를 `(receipt_stamp_ns, stable_record_index)` 순서로 읽었을 때 native Header가 nondecreasing인 maximal 구간이다. 이전 record보다 Header가 strict하게 작아지면 새 epoch를 시작한다. 같은 Header의 equivalent duplicate는 그 epoch 안에서 가장 이른 record만 남기고 conflict는 bag을 reject한다. future target은 `T0+0.5,...,T0+4.0 s`이며 선택 anchor부터 시작하는 동일 epoch의 receipt-order causal suffix 안에서만 Header bracket을 만든다. bracket gap은 100 ms 이하여야 하고 extrapolation, epoch/reset crossing 및 anchor 이전 record 재사용은 금지한다.

`stable_record_index`는 camera/LiDAR/Ego topic message의 zero-based bag read order다. `/tf`에서는 모든 TFMessage 안의 TransformStamped를 message 순서대로 flatten한 zero-based order다. 동일 native Header의 `/Ego_topic` used fields는 값이 같은 duplicate를 가장 이른 record 하나로 collapse한다. Ego used fields의 finite float 값은 수치적으로 같아야 하며, TF duplicate는 translation 1e-6 m와 sign-invariant rotation angle 1e-6 rad 안에서 같아야 한다. 이를 넘는 conflicting duplicate는 bag을 reject한다.

## 3. Raw ROS contract

### 3.1 Required topics

| 역할 | Topic | ROS type | Header frame |
|---|---|---|---|
| Front camera | /image_jpeg/front/compressed | sensor_msgs/CompressedImage | Camera-1 |
| Left camera | /image_jpeg/left/compressed | sensor_msgs/CompressedImage | Camera-2 |
| Right camera | /image_jpeg/right/compressed | sensor_msgs/CompressedImage | Camera-3 |
| LiDAR | /lidar3D | sensor_msgs/PointCloud2 | Lidar3D-4 |
| Ego state | /Ego_topic | morai_msgs/EgoVehicleStatus | message-defined |
| Pose/extrinsic | /tf | tf2_msgs/TFMessage | map/base_link/sensor frames |

### 3.2 Camera physical contract

| View | Resolution | Position xyz m | Preset MORAI Euler deg | Horizontal FOV | Distortion |
|---|---:|---:|---:|---:|---:|
| Front / Camera-1 | 1280×720 | [1.90,0.00,1.20] | [0,2,0] | 90° | [0,0,0] |
| Left / Camera-2 | 640×480 | [1.15,0.65,1.20] | [0,10,70] | 130° | [0,0,0] |
| Right / Camera-3 | 640×480 | [1.15,-0.65,1.20] | [0,10,290] | 130° | [0,0,0] |

`0725demo`의 Euler 필드는 MORAI의 회전 convention이다. 특히 side camera 값을 ROS의 고정축 roll-pitch-yaw로 직접 해석하면 bag의 회전과 일치하지 않는다. bag에서 관측한 다음 `base_link -> sensor` quaternion을 v1의 canonical extrinsic으로 사용한다.

| Child frame | Translation xyz m | Quaternion xyzw |
|---|---:|---:|
| Camera-1 | [1.8999999762,0.0,1.2000000477] | [0.0,0.0174524058,0.0,0.9998477101] |
| Camera-2 | [1.1499999762,0.6499999762,1.2000000477] | [-0.0499904864,-0.0713938102,-0.5713938475,-0.8160349727] |
| Camera-3 | [1.1499999762,-0.6499999762,1.2000000477] | [-0.0499904864,0.0713938102,-0.5713938475,0.8160349727] |
| Lidar3D-4 | [1.5,0.0,1.25] | [0.0,0.0,0.0,1.0] |

0725 camera의 preset roll은 모두 0이다. 이 domain에서 회전은 Hamilton quaternion 곱 기준 `q_base_from_camera = q_y(pitch) ⊗ q_z(wrap_degrees(yaw))`로 해석한다. nonzero roll이 들어오면 추측하지 말고 domain version을 올려 converter를 새로 규정해야 한다.

frame 비교 전 이름의 leading slash를 제거한다. 즉 `/base_link`와 `base_link`는 같은 frame이다. quaternion은 정규화한 뒤 `q`와 `-q`를 같은 회전으로 취급하고, 회전 오차는 `2*acos(clamp(abs(dot(q_ref,q_obs)),0,1))`로 계산한다. 위 MORAI converter는 canonical quaternion을 재현하는 golden test를 통과해야 하며, 일반 ROS RPY 변환으로 대체하면 안 된다.

cameraFOV와 principalPoint를 calibration metadata의 기준으로 사용한다. preset의 focalLengthpixel은 cameraFOV와 일치하지 않으므로 v1 projection 계산에는 사용하면 안 된다. v1 camera adapter는 projection warp를 하지 않지만 이 불일치를 resolved-domain.json에 기록해야 한다.

### 3.3 LiDAR physical contract

~~~yaml
frame_id: Lidar3D-4
translation_base_from_lidar_m: [1.5, 0.0, 1.25]
rotation_base_from_lidar_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
point_fields:
  x: float32
  y: float32
  z: float32
  intensity: float32
source_period_s: 0.06666667014360428
~~~

전처리기는 위 canonical 값과 bag `/tf`를 비교해야 한다. `/tf`가 있는 v1 학습 bag에서 translation 또는 quaternion rotation이 tolerance를 넘으면 해당 bag 전체를 거부한다.

~~~yaml
extrinsic_translation_tolerance_m: 1.0e-4
extrinsic_rotation_tolerance_rad: 1.0e-4
~~~

### 3.4 Canonical ego state and pose

- 현재 pose와 8개 future pose의 유일한 canonical source는 `/tf`의 `map -> base_link` transform이다.
- `/Ego_topic`에서는 `velocity.{x,y}`와 `acceleration.{x,y}`만 모델 feature로 사용한다.
- `/Ego_topic.position`과 `/Ego_topic.heading`은 TF 일치 진단에만 사용한다. `heading`의 단위는 degree이므로 진단 전에 radian으로 변환한다.
- `/Ego_topic` velocity/acceleration은 SI 단위의 `base_link` body-frame 값이며 회전 또는 scale 변환 없이 사용한다.
- bag에는 rear-axle frame과 확정 차량 geometry가 없으므로 base_link-to-rear-axle 보정을 추정해서 적용하면 안 된다.

current pose는 receipt-nearest로 선택한 trajectory-anchor TF 자체를 사용하고 Header/receipt/index bracket을 각각 `[T0,T0]`, `[R0,R0]`, `[i0,i0]`, weight를 0으로 기록한다. Ego와 future pose의 native Header 요청 시각과 동일한 source가 있으면 해당 source를 직접 사용하고 Header/receipt/index bracket의 양쪽을 같은 record로 기록한다. 그 외에는 허용된 source series에서 바로 이전 `t0`와 바로 다음 `t1`만 bracket으로 사용하고 `w=(t-t0)/(t1-t0)`로 계산한다. interpolation weight cache dtype은 float32다.

TF translation은 양쪽 stamp 사이에서 선형 보간하고 rotation quaternion은 shortest-arc SLERP한다. `q0,q1`을 먼저 정규화하고 dot(q0,q1)<0이면 q1의 부호를 뒤집는다. 그 뒤 dot>0.9995면 normalized linear interpolation을, 아니면 SLERP를 적용하고 결과를 다시 정규화한다. `/Ego_topic` velocity/acceleration도 양쪽 stamp 사이에서 선형 보간한다. 모든 보간은 앞뒤 두 source가 있는 bracket 안에서만 허용하며 extrapolation은 금지한다. 현재 pose, future pose 및 Ego state 각각의 bracket 간격은 100 ms 이하여야 한다.

sample selection과 별개로, full-stream `/Ego_topic` native Header마다 같은 native Header axis에서 canonical TF pose를 보간해 진단용 Ego XY 및 yaw와 비교한다. receipt-aligned pose 비교나 affine bridge 결과를 이 bag gate에 사용하면 안 된다. TF bracket을 만들 수 없는 bag 양끝 stamp는 진단 집합에서 제외하되 비교 가능한 pair가 10개 미만이면 bag을 거부한다. position error는 XY Euclidean distance, heading error는 degree-to-radian 변환 후 circular absolute difference다. 각 error를 오름차순 정렬하고 `ceil(0.99*N)-1` zero-based index를 택하는 nearest-rank p99가 다음 tolerance를 넘으면 해당 bag을 거부한다. isolated maximum은 report에 기록하되 단독 reject 조건으로 쓰지 않는다.

~~~yaml
tf_ego_position_p99_tolerance_m: 0.05
tf_ego_heading_p99_tolerance_deg: 0.5
~~~

## 4. Sample selection

1. bag 시작 시 필수 topic, message type, frame 및 sensor preset을 검증한다.
2. valid front receipt 범위에 2 Hz receipt grid를 만들고 grid마다 receipt-nearest front를 50 ms 안에서 선택한다.
3. 선택한 front Header를 sample `timestamp_ns`, grid target을 `grid_receipt_ns`로 사용한다.
4. left/right는 Header 30 ms와 receipt 100 ms dual-filter 뒤 Header tie로 선택한다.
5. LiDAR는 front receipt 기준 45 ms 안에서 receipt-nearest로 선택한다.
6. `map -> base_link` TF는 front receipt 기준 50 ms 안에서 receipt-nearest로 하나 선택하고 그 native Header `T0`의 pose를 current pose로 exact 사용한다.
7. `/Ego_topic` velocity/acceleration은 native Header `T0`에 bracket-only 선형 보간한다.
8. 선택 TF와 같은 receipt-order monotone TF clock epoch의 causal suffix에서 native Header `T0+0.5,...,T0+4.0 s`를 translation 선형 보간과 quaternion SLERP로 보간한다.
9. 어느 필수 값이라도 없거나 비유한 값이면 sample을 생성하지 않는다.

한 raw 메시지는 여러 anchor에 재사용할 수 있지만 sample metadata에는 실제 Header, receipt와 stable record index를 모두 기록해야 한다. Ego 또는 future-pose bracket 간격이 100 ms를 넘으면 sample을 drop해야 한다. source 앞이나 뒤로 extrapolation하거나 TF clock epoch/reset을 넘으면 안 된다. nearest 후보가 이후 validation을 통과하지 못해도 두 번째 후보를 고르는 fallback은 금지한다.

## 5. Camera feature

### 5.1 Decoding

- JPEG decode 결과는 HWC uint8이어야 한다.
- OpenCV decode를 사용하면 BGR에서 RGB로 반드시 변환해야 한다.
- EXIF orientation이나 자동 색보정은 적용하면 안 된다.
- 입력 해상도가 preset과 다르면 sample이 아니라 bag 전체를 거부해야 한다.

### 5.2 Resize, stitch, crop

~~~yaml
interpolation: cv2.INTER_AREA
view_order: [left, front, right]
left_resize_wh: [341, 256]
front_resize_wh: [455, 256]
right_resize_wh: [341, 256]
stitched_wh: [1137, 256]
outer_crop_x_start: 56
outer_crop_x_end_exclusive: 1080
output_wh: [1024, 256]
output_boundaries_x: [0, 285, 740, 1024]
~~~

처리 순서:

1. left 640×480 RGB를 341×256으로 축소
2. front 1280×720 RGB를 455×256으로 축소
3. right 640×480 RGB를 341×256으로 축소
4. LEFT, FRONT, RIGHT 순서로 가로 결합
5. 결합 영상 x=[56,1080)만 보존

이 방식은 각 뷰의 종횡비를 보존하고 front 전체와 side의 front 인접 영역을 유지한다. view blending, padding, 개별 1920×1080 강제 resize 및 NAVSIM의 고정 [416:-416] crop은 금지한다.

### 5.3 Cache and model tensors

~~~yaml
cache:
  name: camera_rgb
  layout: NHWC
  dtype: uint8
  shape_per_sample: [256, 1024, 3]
  range: [0, 255]
model:
  name: camera_feature
  layout: NCHW
  dtype: float32
  shape_per_sample: [3, 256, 1024]
  range: [0.0, 1.0]
  conversion: transpose_HWC_to_CHW_then_divide_255
  imagenet_normalization: false
~~~

여기서 3은 카메라 개수가 아니라 RGB channel 수다.

## 6. LiDAR feature

### 6.1 Point filtering

처리 순서는 바꿀 수 없다.

1. x=y=z=0인 padding point 제거
2. preset의 rigid transform으로 Lidar3D-4에서 base_link로 변환
3. self-mask에 포함된 point 제거
4. height 범위 적용
5. XY histogram 생성

~~~yaml
self_mask_base_link_m:
  x: [0.0, 2.0]
  y: [-1.0, 1.0]
  z: [0.8, 1.6]
height_filter_m:
  min_exclusive: 0.2
  max_exclusive: 100.0
~~~

self-mask는 `0<=x<=2`, `-1<=y<=1`, `0.8<=z<=1.6`을 모두 만족하는 point만 제거한다. z 조건을 생략한 XY prism은 금지한다.

### 6.2 Histogram

~~~yaml
x_range_m: [-32.0, 32.0]
y_range_m: [-32.0, 32.0]
pixels_per_meter: 4
bin_size_m: 0.25
bins_xy: [256, 256]
histogram_axis_order: [x_bin, y_bin]
count_clip: 5
normalization: divide_by_5
rotate_or_flip: false
ground_plane_channel: false
~~~

bin edge는 x와 y 각각 `linspace(-32,32,257)`이고 `numpy.histogramdd(points_xy, bins=(x_edges,y_edges))`의 경계 규칙을 따른다. 즉 내부 bin은 left-inclusive/right-exclusive이고 마지막 bin만 rightmost 32.0 m를 포함하며 범위 밖 point는 버린다.

### 6.3 Cache and model tensors

~~~yaml
cache:
  name: lidar_count
  dtype: uint8
  shape_per_sample: [256, 256]
  range: [0, 5]
model:
  name: lidar_feature
  dtype: float32
  shape_per_sample: [1, 256, 256]
  range: [0.0, 1.0]
~~~

## 7. Status feature

~~~yaml
name: status_feature
dtype: float32
shape_per_sample: [8]
fields:
  0: command_left
  1: command_straight
  2: command_right
  3: command_unknown
  4: velocity_x_mps
  5: velocity_y_mps
  6: acceleration_x_mps2
  7: acceleration_y_mps2
frame: base_link
normalization: none
~~~

command의 앞 네 값은 정확히 하나만 1인 one-hot이어야 한다. /Ego_topic의 velocity와 acceleration은 body frame SI 단위로 직접 사용한다. 회전, km/h 변환, clipping 또는 standardization을 하면 안 된다.

## 8. Trajectory target and model output

### 8.1 Canonical target

~~~yaml
name: trajectory
dtype: float32
shape_per_sample: [8, 3]
sampling_times_s: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
fields: [x_m, y_m, relative_heading_rad]
origin: current_base_link
heading_range: [-pi, pi)
~~~

미래 global pose는 현재 base_link SE(2)의 역변환으로 상대 좌표화한다.

### 8.2 XY diffusion versus pose output

~~~yaml
plan_anchor:
  shape: [20, 8, 2]
  fields: [x_m, y_m]
diffusion_state:
  shape: [8, 2]
  fields: [x_m, y_m]
decoder_prediction:
  shape: [20, 8, 3]
  fields: [x_m, y_m, heading_rad]
selected_model_output:
  shape: [8, 3]
  fields: [x_m, y_m, heading_rad]
~~~

Diffusion/noise scheduler는 XY만 갱신한다. decoder는 heading을 추가로 예측하며 stock IL regression은 x, y, heading 전체를 학습한다.

V2 SEL 최종 inference는 decoder heading을 그대로 반환하지 않는다. control point는 `P0=(0,0)`과 최종 XY candidate `P1..P8`이고, 이 8차 Bézier curve를 `t_k=k/8, k=1..8`에서 미분해 `atan2(dB_y/dt,dB_x/dt)`로 heading을 다시 만든다. 따라서 runtime output은 분명히 [8,3]의 x/y/heading이지만 세 번째 channel은 selected XY에서 유도된 값이다. validation heading MAE는 이 최종 유도 heading과 target relative heading 사이에서 계산한다.

## 9. Automatic driving command

`morai-ddv2-v1-oracle-command`의 command는 학습 sample의 정답 미래 trajectory로 생성한다. 따라서 이 domain은 명시적으로 non-causal이며, 이 입력을 사용하는 모든 학습·검증 metric은 `oracle-command`로 표시해야 한다.

~~~yaml
minimum_path_length_m: 1.0
left:
  minimum_final_heading_rad: 0.20
  minimum_final_lateral_m: 0.50
right:
  maximum_final_heading_rad: -0.20
  maximum_final_lateral_m: -0.50
straight:
  maximum_abs_final_heading_rad: 0.10
  maximum_abs_final_lateral_m: 0.50
ambiguous_or_static_class: UNKNOWN
nan_allowed: false
~~~

path length는 `p0=(0,0)`, `p1..p8=trajectory[:,0:2]`에 대해 `sum(i=1..8, ||p_i-p_{i-1}||_2)`로 계산한다. final heading과 lateral은 각각 4.0초 target의 `trajectory[7,2]`와 `trajectory[7,1]`이다.

분류 순서:

1. 미래 trajectory가 누락되거나 비유한 값이면 command를 만들지 않고 sample을 drop
2. path length < 1 m이면 UNKNOWN
3. heading과 lateral 조건이 모두 LEFT를 만족하면 LEFT
4. 두 조건이 모두 RIGHT를 만족하면 RIGHT
5. 두 straight 조건을 모두 만족하면 STRAIGHT
6. 나머지는 UNKNOWN

전처리기는 bag SHA-256과 anchor `timestamp_ns`를 key로 하는 command sidecar를 cache와 함께 생성해야 한다. held-out validation과 ROS bag-playback Validator는 모두 이 sidecar의 future-derived command를 사용해야 한다. sidecar가 없거나 bag hash·timestamp가 맞지 않으면 평가를 실패시키며 UNKNOWN으로 조용히 대체하면 안 된다. UNKNOWN은 위 규칙에서 정지·애매한 미래경로에 대해 정상 생성된 class일 때만 허용한다.

실시간 causal 평가에는 route/planner command 또는 모든 sample의 UNKNOWN처럼 미래 GT를 사용하지 않는 별도 입력 계약이 필요하다. 이는 `morai-ddv2-v1-causal` domain/cache/checkpoint로 분리하고 oracle-command artifact와 학습·평가에서 혼합하면 안 된다.

## 10. Cache record

cache/manifest/command-sidecar schema version은 2다. resolved-domain root schema는 1을 유지할 수 있지만 resolved-domain hash는 clock policy와 cache schema 변경으로 반드시 바뀐다.

bag 하나는 cache root 기준 `bags/<domain_id>/<resolved_domain_sha256>/<source_bag_sha256>.h5` 하나로 직렬화한다. sample ID는 lowercase bag SHA-256, colon, unsigned decimal selected-front Header를 연결한 `<source_bag_sha256>:<timestamp_ns>`이며 dataset 전체에서 유일해야 한다. `timestamp_ns`는 selected front Header이고 `grid_receipt_ns`나 callback time으로 대체하면 안 된다.

| Dataset | dtype | Shape |
|---|---|---|
| camera_rgb | uint8 | [N,256,1024,3] |
| lidar_count | uint8 | [N,256,256] |
| status | float32 | [N,8] |
| trajectory | float32 | [N,8,3] |
| timestamp_ns | int64 | [N] |
| grid_receipt_ns | int64 | [N] |
| sensor_source_header_stamp_ns | int64 | [N,4] |
| sensor_source_receipt_ns | int64 | [N,4] |
| sensor_source_record_index | int64 | [N,4] |
| side_camera_header_offset_ns | int64 | [N,2] |
| sensor_receipt_offset_ns | int64 | [N,3] |
| trajectory_anchor_tf_header_ns | int64 | [N] |
| trajectory_anchor_tf_receipt_ns | int64 | [N] |
| trajectory_anchor_tf_record_index | int64 | [N] |
| trajectory_anchor_tf_clock_epoch_index | int64 | [N] |
| ego_bracket_header_stamp_ns | int64 | [N,2] |
| ego_bracket_receipt_ns | int64 | [N,2] |
| ego_bracket_record_index | int64 | [N,2] |
| ego_interp_weight | float32 | [N] |
| current_pose_bracket_header_stamp_ns | int64 | [N,2] |
| current_pose_bracket_receipt_ns | int64 | [N,2] |
| current_pose_bracket_record_index | int64 | [N,2] |
| current_pose_interp_weight | float32 | [N] |
| future_pose_bracket_header_stamp_ns | int64 | [N,8,2] |
| future_pose_bracket_receipt_ns | int64 | [N,8,2] |
| future_pose_bracket_record_index | int64 | [N,8,2] |
| future_pose_interp_weight | float32 | [N,8] |

sensor source Header/receipt/index의 마지막 축 순서는 front, left, right, lidar다. `side_camera_header_offset_ns`는 left/right Header에서 selected-front Header를 뺀 `[left,right]` signed nanosecond다. LiDAR Header는 provenance로만 저장하며 sync offset 의미를 부여하지 않는다. `sensor_receipt_offset_ns`는 left/right/lidar receipt에서 selected-front receipt를 뺀 `[left,right,lidar]` signed nanosecond다. 서로 다른 clock axis를 한 offset dataset에 섞으면 안 된다.

`trajectory_anchor_tf_*`는 receipt-nearest로 선택한 exact current TF를 가리킨다. 따라서 `current_pose_bracket_*`는 Header `[T0,T0]`, receipt `[R0,R0]`, index `[i0,i0]`, weight `0`이어야 한다. bracket record index는 duplicate collapse 뒤 선택된 원본 stable record를 가리킨다. 각 interpolation weight는 native Header axis의 `t=(1-w)*t0+w*t1`에서 계산한 `w`이며 범위는 [0,1]이다. receipt bracket은 같은 두 source record의 receipt provenance이지 weight 계산 axis가 아니다.

command sidecar 경로는 cache root 기준 `commands/<domain_id>/<resolved_domain_sha256>/<source_bag_sha256>.jsonl`이다. `.partial` 파일에 쓴 뒤 fsync하고 원자적으로 rename한다. record는 `timestamp_ns` 오름차순이고 timestamp가 중복되면 안 된다. 각 record는 `schema_version=2`, `domain_id`, `resolved_domain_sha256`, `sensor_preset_sha256`, `source_bag_sha256`, `command_rule_sha256`, `timestamp_ns`, `grid_receipt_ns`, `front_receipt_ns`, `front_record_index`, `command_index`, `command_name`, `provenance=future_trajectory_oracle`을 가진다. lookup key는 계속 `(source_bag_sha256,timestamp_ns)`이고 receipt를 key로 바꾸면 안 된다. index mapping은 LEFT=0, STRAIGHT=1, RIGHT=2, UNKNOWN=3이며 같은 값은 `status[:,0:4]`와 exact하게 일치해야 한다.

schema-v2 manifest의 각 `sample` record는 위 HDF5 provenance field 전부를 같은 이름, 축 순서 및 값으로 반복하고 `hdf5_row_index`, `sidecar_row_index`와 artifact identity를 추가한다. manifest가 HDF5와 다른 clock axis, source index 또는 interpolation bracket을 기록하면 전체 artifact를 reject한다. `dataset` record의 `schema_version`도 2이며 loader는 schema 1과 2를 암묵적으로 혼용하거나 field 이름을 추측해 migration하면 안 된다.

`command_rule_sha256`은 resolved-domain.json에 기록된 command mapping, threshold와 분류 순서를 canonical JSON으로 직렬화한 SHA-256이다. 전체 manifest는 완성 sidecar의 상대경로, record count와 파일 SHA-256을 저장한다. Validator는 namespace, 모든 identity field, record count 및 sidecar SHA-256을 추론 전에 검증해야 한다.

HDF5 attribute:

| Attribute | 규칙 |
|---|---|
| cache_schema_version | 정수 2 |
| clock_policy_version | 정수 1 |
| domain_id | 정확히 morai-ddv2-v1-oracle-command |
| sensor_preset_sha256 | 정확히 b7e69c70cc0b2948c3a5b352eee022be04f8884a1df135697f89bb4bb88263fb |
| resolved_domain_sha256 | canonical resolved-domain.json의 실제 64자리 lowercase SHA-256 |
| source_bag_sha256 | 해당 bag 파일의 실제 64자리 lowercase SHA-256 |
| source_code_commit | 전처리 실행 checkout의 실제 40자리 git commit |

`canonical JSON`은 RFC 8785 JSON Canonicalization Scheme의 UTF-8 byte sequence를 뜻한다. `resolved_domain_sha256`은 snapshot의 자기 자신 hash field를 제외한 canonical JSON SHA-256이고, `command_rule_sha256`은 command index mapping, threshold 및 ordered classification rule subtree의 canonical JSON SHA-256이다. 모든 digest 문자열은 64자리 lowercase hexadecimal이다.

hash나 commit을 계산할 수 없는 dirty/uncommitted 실행은 기본적으로 cache 생성을 거부한다. 개발용 override를 사용한 경우 dirty diff hash와 override 사실을 manifest에 기록하고 production training에서는 해당 cache를 거부한다.

dataset manifest 경로는 `manifests/<domain_id>/<resolved_domain_sha256>/manifest.jsonl`이다. preprocessor는 각 HDF5와 command sidecar를 `.partial`에 완전히 쓰고 fsync한 뒤 같은 filesystem에서 원자적으로 rename한다. 모든 artifact의 SHA-256과 record count를 검증한 다음 manifest를 `.partial`에 쓰고 fsync하며, manifest rename과 parent-directory fsync를 마지막 commit marker로 수행한다. loader는 final manifest가 참조하고 hash가 일치하는 artifact만 읽고, crash 뒤 남은 orphan artifact나 `.partial`은 무시한다.

## 11. Sample validity and drop reasons

각 sample은 다음 조건을 모두 만족해야 한다.

- 세 JPEG가 decode되고 preset 해상도와 일치
- front/LiDAR/TF receipt offset과 side Header/receipt dual offset이 각자의 tolerance 안
- LiDAR x/y/z가 finite
- Ego velocity/acceleration과 현재/future pose가 finite
- 8개 future pose가 모두 같은 bag 및 trajectory-anchor TF clock epoch의 causal suffix 안에 존재
- status command가 유효한 one-hot
- 생성 tensor의 shape, dtype, range가 계약과 일치

sample_drop_reason enum:

~~~text
DECODE_ERROR
CAMERA_SYNC_MISS
LIDAR_SYNC_MISS
EGO_STATE_INTERPOLATION_MISS
POSE_INTERPOLATION_MISS
FUTURE_HORIZON_MISS
NONFINITE_VALUE
INVALID_COMMAND
EMPTY_LIDAR
~~~

bag_reject_reason enum:

~~~text
MISSING_REQUIRED_TOPIC
MESSAGE_TYPE_MISMATCH
FRAME_ID_MISMATCH
RESOLUTION_MISMATCH
EXTRINSIC_MISMATCH
TF_EGO_POSE_MISMATCH
DUPLICATE_STATE_CONFLICT
NO_VALID_SAMPLE
~~~

bag-level 오류에서는 해당 bag cache를 생성하지 않는다. batch preprocessor는 나머지 bag도 검사해 전체 rejection report를 완성하지만, 예상 eligible bag이 하나라도 reject되면 마지막에 nonzero로 종료한다. sample-level 오류만 해당 anchor를 drop하고 정상 cache 생성을 계속한다.

## 12. Dataset eligibility and split

v1 supervised dataset은 종료된 .bag이면서 /Ego_topic과 /tf를 가진 파일만 포함한다. 현재 예상 자격 파일 수는 30개다.

- .bag.active: 제외
- 세 camera와 LiDAR는 있으나 `/tf`와 `/Ego_topic`이 둘 다 없는 마지막 10개: `SENSOR_ONLY_EXCLUDED`
- `/tf`와 `/Ego_topic` 중 하나만 없거나 core sensor topic이 없는 finalized bag: reject
- object/collision topic: 무시
- HD map과 route geometry: 요구하지 않음

split은 bag group 단위다. relative bag path는 CLI `--bag-root` 아래의 symlink-resolved regular file을 기준으로 한 `/` separator POSIX path이며 UTF-8로 encode한다. literal ASCII prefix `20260806:`과 relative path byte를 이어 SHA-256을 계산하고 `(digest_bytes, relative_path_bytes)` 오름차순으로 정렬한다. 앞의 floor(0.8×N)개를 train, 나머지를 validation으로 사용한다. 현재 N=30이면 train 24개, validation 6개다. frame 단위 random split은 금지한다. 생성된 path/hash/split 목록은 manifest에 고정하고 이후 cache 재사용 시 exact match를 요구한다.

## 13. Training contract

model input:

~~~yaml
camera_feature: float32 [B,3,256,1024]
lidar_feature: float32 [B,1,256,256]
status_feature: float32 [B,8]
target_trajectory: float32 [B,8,3]
~~~

supervised adapter는 SEL의 한 anchor group만 사용하며 다음 계약을 만족해야 한다.

~~~yaml
plan_anchor_xy: float32 [B,20,8,2]
anchor_normalization: [x_divide_50, y_divide_20]
diffusion_timestep: uniform_integer_inclusive [0,49]
diffusion_noise: standard_normal_same_shape_as_anchor
noisy_normalized_clamp: [-1.0,1.0]
decoder_regression_source: poses_reg_list[0] float32 [B,20,8,3]
decoder_logits_source: poses_cls_list[0] float32 [B,1,20]
decoder_logits_for_loss: squeeze_axis_1 float32 [B,20]
loss_target_dict: {trajectory: target_trajectory}
~~~

noise 순서는 V2 `x/50,y/20` normalization, scheduler `add_noise`, float32 변환, [-1,1] clamp, `x*50,y*20` denormalization 순이어야 한다. decoder layer 수는 정확히 1이어야 하고 모든 중간 shape를 assert한다. deterministic resume checkpoint는 Python, NumPy, Torch CPU 및 모든 CUDA RNG state를 저장·복원해야 한다.

v1 loss:

~~~text
d[b,m] = mean_t(norm_2(target[b,t,0:2] - plan_anchor[m,t,0:2]))
m_star[b] = argmin_m(d[b,m])

L_cls = sigmoid_focal_mean(mode_logits, one_hot(m_star, 20), alpha=0.25, gamma=2.0)
L_reg = mean_abs(predicted_pose[b,m_star,:,:] - target_pose[b,:,:])
L_traj = 10 * L_cls + 8 * L_reg

L_finetune = 12 * L_traj
           + 1e-3 * mean((trainable_parameter - pretrained_parameter)^2)
~~~

`L_cls`는 softmax cross entropy가 아니라 20개 mode logit 각각에 적용한 sigmoid focal loss의 전체 element mean이다. `L_reg`는 batch, 8개 timestep 및 x/y/heading 3개 field 전체의 mean이다. heading 항도 stock `F.l1_loss`의 raw scalar difference이며 ±π에서 circular wrap을 다시 적용하지 않는다. source-preservation 항은 allowlist trainable parameter의 모든 scalar를 합친 element mean이다.

이 loss는 고정 V2 SEL checkpoint에 존재하는 단일 `_trajectory_head.diff_decoder.layers.0`의 prediction에 적용한다. DiffusionDrive IL 참고 모델의 2-layer decoder를 대신 instantiate하면 안 된다. 학습 때 regression mode는 predicted score가 아니라 GT XY에 가장 가까운 anchor `m_star`로 선택한다.

독립 학습 루프는 TransfuserAgent.compute_loss, transfuser_loss 및 TransfuserTargetBuilder를 호출하면 안 된다. 객체 GT, HD map, agent loss, BEV semantic loss, PDM reward 및 V2 selector loss는 요구하거나 계산하면 안 된다.

export checkpoint의 raw state_dict는 원본과 같은 972-key schema를 유지하고 metadata key를 추가하면 안 된다. `<checkpoint>.metadata.json` sidecar에 `base_checkpoint_sha256`, `base_state_dict_sha256`, `export_checkpoint_sha256`, `export_state_dict_sha256`, `resolved_domain_sha256`, `sensor_preset_sha256`, `training_manifest_sha256`, source commit, trainable prefix 및 raw/runtime key count 972/971을 기록한다. checkpoint를 먼저 atomic publish하고 실제 file hash를 계산한 뒤 metadata를 마지막에 atomic publish한다.

state_dict hash는 각 tensor를 contiguous CPU little-endian C-order raw byte로 만든 `data_sha256`과 `{key,dtype,shape,data_sha256}` entry를 구성하고, key의 UTF-8 byte 순서로 정렬한 RFC 8785 canonical JSON의 SHA-256이다. metadata 자체 hash를 metadata 안에 넣어 자기참조를 만들면 안 된다.

exporter는 pinned Validator의 official baseline profile을 수정하지 않고 별도 `configs/validator/profile-morai-ddv2-oracle.json`을 생성한다. pinned exact-ID allowlist와 seed/runtime dispatch를 유지하기 위해 이 파일의 model entry ID는 `diffusiondrive_v2`여야 한다. export checkpoint, metadata sidecar, resolved-domain.json과 sensor preset은 각각 실제 file SHA-256이 pin된 required runtime asset이어야 한다. profile은 domain ID, `resolved_domain_sha256`, `export_state_dict_sha256` 및 `training_manifest_sha256`도 가진다. Validator 실행 인자는 `~profile_file=<absolute MORAI profile path>`와 `~model_id=diffusiondrive_v2`다.

## 14. Runtime parity

offline preprocessor와 ROS Validator는 동일한 pure preprocessing functions와 resolved-domain.json을 사용해야 한다.

ROS bag playback은 schema-v2 manifest가 고정한 per-topic `stable_record_index`, selected source record와 Ego/current/future bracket을 callback counter로 exact replay해야 한다. callback 시점의 `rospy.Time.now()`, 도착 순서에서 다시 계산한 nearest, 또는 Header만으로 source를 재선택하면 안 된다. callback counter와 observed Header가 manifest-selected stable index/Header와 일치하는지 검증한 뒤 manifest 선택을 사용하며, exact receipt provenance는 verified manifest가 공급한다. 특히 Header backstep이 있는 LiDAR buffer는 Header/receipt cutoff가 아니라 다음 manifest-required stable index를 기준으로만 prune해야 한다.

Validator model load는 profile의 runtime asset hash 검증 후 metadata sidecar를 필수로 읽어야 한다. profile checkpoint SHA, metadata `export_checkpoint_sha256`와 실제 checkpoint SHA가 모두 같고, recompute한 state_dict hash가 `export_state_dict_sha256`와 같아야 한다. profile/metadata/cache의 domain, sensor 및 training-manifest identity도 일치한 뒤에만 prefix 제거와 vocab 한 key 제외 후 971-key `strict=True` load를 수행한다. baseline profile overwrite, metadata 누락 또는 hash/identity 불일치는 load 전에 실패해야 한다.

Validator는 다음을 검증해야 한다.

- sensor/domain/checkpoint hash
- camera, LiDAR, status tensor shape/dtype/range/finite
- bag SHA-256과 timestamp로 조회한 oracle command sidecar provenance
- model output shape [B,8,3]

held-out offline validation과 ROS Validator는 같은 PDM-free learned selector의 최종 `trajectory`를 metric에 사용해야 한다. GT-nearest training mode `m_star`를 validation prediction으로 사용하면 안 된다. deterministic evaluation의 `run_id`는 `morai-ddv2-v1-oracle-command-validation-v1`로 고정한다. sample seed는 UTF-8 문자열 `diffusiondrive_v2:<timestamp_ns>:<run_id>`의 SHA-256 digest 앞 8 byte를 unsigned big-endian integer로 해석한 값이다. `timestamp_ns`는 front anchor Header.stamp다. evaluation inference batch size는 1이고 매 sample forward 직전에 Torch CPU와 모든 CUDA RNG를 해당 seed로 재설정한다. sample 처리 순서가 바뀌어도 sample ID별 output이 exact하게 같아야 한다.

ADE는 sample별 8개 XY Euclidean error mean을 전체 sample에서 다시 평균한 값이고, FDE는 4.0초 XY Euclidean error의 전체 sample mean이다. heading MAE는 `abs(wrap_to_pi(pred_heading-target_heading))`의 sample×8 mean이다. best checkpoint는 selected trajectory의 sample-micro ADE로 선택하고 bag별 ADE의 macro mean도 기록한다. command class 표본 수가 0이면 count=0과 metric=null을 기록하며 NaN을 기록하면 안 된다.

epoch validation은 배포 selection을 바꾸지 않는 oracle 진단 `minADE@32`와 `minFDE@32`를 함께 기록한다. 한 번의 정상 public selector forward 중 변경 전 post-Bézier proposal `[B,800,8,3]`과 coarse `topk_idx [B,32]`를 validation-only context manager로 capture해 exact shortlist `[B,32,8,3]`을 얻고, context 종료 시 원래 callable을 `finally`에서 복원한다. capture on/off의 public output은 bitwise identical이어야 하고 selected trajectory는 shortlist 중 하나여야 한다. candidate source는 `coarse_top32_post_bezier`, `k=32`다. sample별 각 후보의 8-step XY ADE와 4.0초 XY FDE를 계산한 뒤 각각 독립 argmin하고, 후보를 deduplicate하지 않으며 float64로 sample mean을 누적한다. 800개 전체 proposal, 4개 stage winner, GT-nearest mode는 이 metric 후보가 아니다.

각 epoch은 train loss와 deterministic validation loss의 total/trajectory/preservation sample-weighted mean을 기록한다. validation loss sample seed는 UTF-8 `diffusiondrive_v2_supervised_loss:<timestamp_ns>:morai-ddv2-v1-oracle-command-validation-v1`의 SHA-256 앞 8 byte unsigned big-endian 값이며, batch size 1/no_grad에서 supervised adapter와 정확한 `12*L_traj+1e-3*L_sp`를 실행한 뒤 training boundary를 복원한다.

authoritative epoch telemetry는 `metrics/epochs/epoch-0000.json`의 RFC 8785 canonical JSON이며 `.partial`/file fsync/replace/parent fsync로 publish한다. mid-epoch `last.pt`는 completed batch/sample 수, 세 loss의 sample-weighted numerator와 grad-norm sum/count를 저장한다. epoch JSON을 먼저 durable publish하고 다음-epoch sampler state와 epoch-record hash를 포함한 `last.pt`를 publish한 뒤에만 다음 epoch를 시작한다. resume은 같은 epoch bytes를 idempotent하게 허용하고 충돌은 실패시킨다. `metrics/history.jsonl`, `metrics/epoch_metrics.csv`, `plots/training-curves.png`는 정렬된 epoch JSON에서 매번 atomic rebuild하는 파생물이며 PNG는 loss, selected ADE/FDE, minADE@32/minFDE@32의 세 panel을 가진다.

bag playback 중 command sidecar가 없거나 record가 누락되면 Validator는 즉시 실패해야 한다. 고정 STRAIGHT/UNKNOWN fallback, 0716 LiDAR translation [2.0,0.0,1.5], side camera의 1920×1080 anisotropic resize는 `morai-ddv2-v1-oracle-command`에서 금지한다.

## 15. Golden acceptance tests

domain 구현은 다음 검사를 통과해야 한다.

1. MORAI Euler converter와 bag `/tf`가 canonical sensor quaternion을 재현
2. camera seam이 x=285와 x=740에 존재
3. camera model tensor가 RGB float32 [3,256,1024], [0,1]
4. self-mask 밖의 synthetic LiDAR point가 삭제되지 않음
5. LiDAR bin 방향이 x-forward/y-left와 일치
6. quaternion `q1=-q0`와 heading ±π 경계에서 shortest-arc 보간과 wrap이 연속적
7. command threshold 경계가 규정된 class 또는 UNKNOWN을 반환
8. offline과 ROS 경로의 golden tensor가 exact 또는 float tolerance 안에서 동일
9. command sidecar와 cache status one-hot이 bag hash·timestamp별 exact match
10. sidecar 누락 시 Validator가 UNKNOWN fallback 없이 실패
11. export checkpoint가 V2 SEL runtime에서 strict load
12. 모델 출력이 finite [8,3]
13. Header-nearest side 1순위가 receipt 100 ms를 위반하고 2순위가 dual-bound를 만족하면 2순위를 선택
14. TF native Header backstep에서 새 epoch를 만들고 future bracket이 reset을 넘지 않음
15. offline과 Validator가 manifest stable index/bracket을 replay하며 callback time으로 재선택하지 않음

## 16. Versioning rule

다음 중 하나라도 바뀌면 domain ID를 올리고 cache를 다시 생성해야 한다.

- sensor preset 또는 hash
- topic/frame mapping
- clock axis/policy, sync rate/tolerance 또는 tie-break
- cache/manifest/sidecar schema
- camera resize/crop/interpolation/color order
- LiDAR transform/mask/range/binning
- status field order 또는 units
- trajectory horizon/origin/fields
- command rule

단순 학습 hyperparameter 변경은 domain ID를 바꾸지 않지만 checkpoint metadata에는 기록해야 한다.
