# MORAI DDV2 30-Bag Clock Audit

- 감사일: 2026-08-08
- 대상: `/home/stier/rosbag/just_drive`의 GT-rich finalized bag 30개
- 결론: Option D dual-clock hybrid를 `morai-ddv2-v1-oracle-command`의 production 전처리 계약으로 사용
- 규범 계약: `docs/domain-specs/morai-diffusiondrivev2-v1.md`

이 문서는 clock policy를 선택한 측정 근거다. bag receipt stamp는 recorder가 message를 관측하고 기록한 순서의 clock일 뿐이며 센서 acquisition time으로 간주하지 않는다. 아래 affine fit은 clock 상태를 감사하기 위한 진단이지 production 변환식이 아니다.

## 1. 측정 방법

각 sensor와 `/Ego_topic`, `/tf map -> base_link` record에서 native `Header.stamp`, rosbag receipt stamp와 stable read index를 수집했다. receipt seconds를 독립변수, Header seconds를 종속변수로 bag별 least-squares affine fit을 만들고 absolute residual의 nearest-rank p99를 계산했다.

표의 열은 다음 뜻이다.

- `Cam p99 s`: front/left/right 세 fit residual p99 중 최댓값
- `LiDAR p99 s/backsteps`: LiDAR fit residual p99와 receipt-order Header strict backstep 수
- `Ego slope/p99 s`, `TF slope/p99 s`: receipt→Header affine slope와 residual p99
- `Ego↔TF NN-r p99 ms`: receipt axis에서 Ego record마다 nearest TF receipt까지의 absolute delta p99

## 2. 30-bag audit table

| Bag timestamp | Cam p99 s | LiDAR p99 s / backsteps | Ego slope / p99 s | TF slope / p99 s | Ego↔TF NN-r p99 ms |
|---|---:|---:|---:|---:|---:|
| 20260731T141646KST | 0.247 | 3.167 / 583 | 1.000133 / 0.191 | 1.000157 / 0.234 | 58.95 |
| 20260731T143443KST | 0.053 | 2.372 / 401 | 1.000041 / 0.042 | 1.000040 / 0.033 | 25.51 |
| 20260731T144057KST | 0.104 | 3.172 / 536 | 1.000058 / 0.057 | 1.000056 / 0.059 | 26.13 |
| 20260731T152423KST | 0.523 | 4.420 / 796 | 1.000251 / 0.428 | 1.000285 / 0.453 | 73.32 |
| 20260731T152922KST | 0.105 | 1.468 / 67 | 1.000034 / 0.041 | 1.000035 / 0.036 | 26.15 |
| 20260731T153358KST | 0.471 | 4.006 / 967 | 1.000276 / 0.428 | 1.000308 / 0.457 | 77.63 |
| 20260731T154843KST | 0.331 | 10.828 / 782 | 1.000101 / 0.238 | 1.000108 / 0.276 | 37.64 |
| 20260731T155610KST | 1.288 | 5.525 / 1533 | 0.999580 / 1.227 | 0.999573 / 1.250 | 77.27 |
| 20260731T160507KST | 0.175 | 9.437 / 1018 | 1.000300 / 0.160 | 1.000337 / 0.200 | 35.14 |
| 20260731T161756KST | 1.718 | 14.787 / 1759 | 0.758577 / 15.268 | 0.759862 / 15.296 | 25.04 |
| 20260731T162710KST | 6.517 | 7.017 / 1674 | 0.589867 / 14.718 | 0.591696 / 14.795 | 26.55 |
| 20260731T163308KST | 1.989 | 10.336 / 1975 | 0.671613 / 17.513 | 0.673759 / 17.697 | 25.38 |
| 20260731T163928KST | 5.843 | 10.579 / 1799 | 0.605865 / 16.800 | 0.607078 / 16.964 | 27.27 |
| 20260731T164630KST | 0.180 | 14.659 / 1506 | 0.748231 / 11.056 | 0.751070 / 10.886 | 23.96 |
| 20260731T165105KST | 4.646 | 10.686 / 1963 | 0.589009 / 15.296 | 0.592119 / 15.579 | 26.93 |
| 20260731T165700KST | 5.032 | 10.214 / 1853 | 0.607884 / 16.991 | 0.608918 / 17.134 | 26.82 |
| 20260731T170301KST | 0.154 | 14.547 / 143 | 0.844449 / 0.984 | 0.848028 / 0.981 | 19.64 |
| 20260731T171754KST | 1.600 | 18.968 / 969 | 0.908384 / 14.980 | 0.918077 / 15.066 | 23.49 |
| 20260731T181003KST | 0.046 | 0.200 / 0 | 1.000027 / 0.037 | 1.000021 / 0.033 | 26.91 |
| 20260731T181638KST | 0.044 | 0.078 / 0 | 1.000016 / 0.036 | 1.000009 / 0.034 | 26.45 |
| 20260731T182159KST | 0.047 | 0.096 / 0 | 1.000014 / 0.035 | 1.000010 / 0.032 | 26.12 |
| 20260731T182725KST | 0.102 | 1.893 / 26 | 1.000028 / 0.043 | 1.000023 / 0.035 | 27.64 |
| 20260731T183730KST | 0.092 | 1.838 / 15 | 1.000016 / 0.042 | 1.000012 / 0.034 | 27.35 |
| 20260731T184345KST | 0.073 | 0.572 / 0 | 1.000010 / 0.041 | 1.000008 / 0.035 | 27.45 |
| 20260803T150216KST | 0.055 | 0.080 / 0 | 1.000017 / 0.037 | 1.000015 / 0.028 | 27.33 |
| 20260803T151735KST | 0.077 | 0.435 / 0 | 1.000009 / 0.043 | 1.000009 / 0.033 | 27.18 |
| 20260803T152942KST | 0.054 | 0.088 / 0 | 1.000009 / 0.039 | 1.000006 / 0.029 | 26.68 |
| 20260803T154018KST | 0.044 | 0.051 / 0 | 1.000009 / 0.036 | 1.000000 / 0.026 | 25.08 |
| 20260803T154333KST | 0.041 | 0.053 / 0 | 1.000020 / 0.035 | 1.000017 / 0.025 | 27.02 |
| 20260803T154803KST | 0.122 | 2.386 / 288 | 1.000013 / 0.044 | 1.000013 / 0.033 | 28.23 |

LiDAR native Header는 30개 중 21개 bag에서 strict backstep이 있었고 총 20,653회였다. 18개 bag은 100회 이상이었다. TF affine residual p99가 100 ms를 넘는 bag도 15/30이었다. 따라서 모든 sensor를 Header 하나로 sync하거나 bag 전체에 하나의 affine bridge를 적용하는 계약은 이 dataset을 설명하지 못한다.

## 3. A/B/C/D 비교

검토한 선택지는 다음과 같다.

- A: sensor receipt와 같은 receipt axis에서 TF/Ego 상태를 직접 보간
- B: bag별 global affine receipt→Header fit으로 TF/Ego target Header 생성
- C: receipt grid/sensor bundle에서 local receipt-nearest TF 하나를 고르고 그 native Header를 상태축으로 사용; side camera도 receipt-nearest
- D: C의 local TF native-Header anchor를 유지하되 side camera는 Header 30 ms와 receipt 100 ms dual-bound 후 Header-nearest; LiDAR는 receipt-nearest

A와 B의 아래 수치는 초기 Header-grid 실험이며 D의 receipt-grid count와 직접 동일한 denominator라고 가정하면 안 된다.

| Option | Bag | Planning counts | Full-horizon count | Selected-anchor position p99 m | heading p99 deg |
|---|---|---|---:|---:|---:|
| A receipt-state | 141646 | — | 482 | 0.4036 | 0.4753 |
| A receipt-state | 154803 | — | 539 | 0.4717 | 0.5342 |
| A receipt-state | 170301 | — | 35 | 79.371 | 81.458 |
| B global affine | 141646 | anchors/sensor 617/593; current 609 | 587 | 0.04045 | 0.03147 |
| B global affine | 154803 | anchors/sensor 577/549; current 567 | 539 | 0.01952 | 0.02340 |
| B global affine | 170301 | anchors/sensor 47/39; current 40 | 36 | 0.1011 | 0.0144 |

A는 recorder observation time을 물리 상태축으로 사용해 170301에서 79 m/81° 오차를 만들었다. B는 일부 bag에서 맞지만 170301 selected-anchor position p99가 0.05 m gate를 넘고 TF fit residual p99가 0.981 s였다. 161756의 TF residual p99는 15.296 s여서 하나의 global affine mapping을 production 계약으로 사용할 수 없다.

다음 표는 같은 receipt-grid 계열에서 C와 D를 비교한다. `C all/full`은 C의 all-sensor/full-horizon count다. `D camera/all/full`은 side dual-bound 통과, all-sensor+TF-anchor 통과, full-horizon count다. anchor p99는 선택 sample 진단이며 bag accept/reject gate는 마지막 두 열의 full-stream native-Header TF/Ego p99다.

| Bag | Receipt grid | C all/full | D camera/all/full | D anchor p99 position m / heading deg | Native full-stream p99 position m / heading deg |
|---|---:|---:|---:|---:|---:|
| 141646 | 617 | 494 / 482 | 539 / 520 / 508 | 0.02982 / 0.03294 | 0.02770 / 0.03041 |
| 161756 | 618 | 471 / 455 | 381 / 345 / 342 | 0.01303 / 0.02419 | 0.01550 / 0.02230 |
| 170301 | 46 | 33 / 30 | 38 / 36 / 29 | 0.18675 / 0.04907 | 0.03275 / 0.03915 |
| 181003 | 615 | 527 / 519 | 615 / 606 / 598 | 0.02357 / 0.02385 | 0.02151 / 0.02512 |

170301의 D anchor position p99는 29개 중 isolated outlier 하나 때문에 높다. 규범 bag gate는 sample-selected subset이 아니라 full-stream native-Header diagnostic이며 이 bag은 0.03275 m/0.03915°로 통과한다. isolated maximum은 report에 남기되 단독 reject 조건으로 사용하지 않는 기존 정책과 일치한다.

TF receipt tolerance는 50 ms로 선택했다. 25 ms는 141646과 181003에서 각각 35개와 29개 valid anchor를 잃었고, 100 ms는 141646에서 5개만 추가하면서 더 먼 association을 허용했다.

## 4. Production decision and limitations

Option D는 다음을 동시에 보존한다.

1. recorder 관측 순서에서 sensor bundle과 local TF record를 association한다.
2. camera exposure는 native Header proximity를 요구하되 100 ms receipt sanity bound로 지연 queue의 잘못된 Header-nearest 후보를 막는다.
3. current/future physics는 선택 TF의 native Header `T0`와 동일 receipt-order monotone TF clock epoch에 남는다.
4. LiDAR의 잦은 Header backstep을 sync 결정에 사용하지 않는다.
5. full-stream TF/Ego native-Header diagnostic을 독립 bag gate로 유지한다.

이 감사만으로 receipt stamp가 acquisition time임을 증명하지 않는다. 또한 표의 planning count는 최종 cache sample count가 아니다. production preprocessor는 JPEG decode/resolution, PointCloud layout, extrinsic, duplicate conflict, finite tensor 및 artifact identity gate를 추가로 적용해야 한다. 새 bag 또는 sensor clock behavior가 이 정책을 벗어나면 자동 affine 보정으로 숨기지 말고 새 domain version과 재감사를 요구한다.
