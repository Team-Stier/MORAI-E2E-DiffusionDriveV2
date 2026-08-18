# MORAI DiffusionDriveV2 fine-tuning

기존 파이프라인을 폐기하고 ROS1 bag 파인튜닝에 필요한 두 단계만 새로 구현한 저장소이다.

- 전처리: raw ROS bag → HDF5
- 학습: HDF5 → 공식 DiffusionDriveV2 SEL checkpoint
- 보존: 기존 loss 함수와 `sensor_data/0725demo_ros.json`

## 데이터 규칙

기본 topic은 `/image_jpeg/{left,front,right}/compressed`, `/lidar3D`, `/Ego_topic`이다. 카메라와 LiDAR는 [E2E-OpenLoop-ROS-Validator `aac7acf`](https://github.com/Team-Stier/E2E-OpenLoop-ROS-Validator/tree/aac7acf01b5259c61466c7af0bc45cf075560321)의 `prepare_diffusiondrive_v2(..., morai_domain=True)` 소스를 그대로 사용한다.

각 타깃은 현재 ego 좌표계의 `(x, y, heading)` 8개다. 10 Hz Ego frame에서 정확히 `+5,+10,+15,+20,+25,+30,+35,+40` frame, 즉 `+0.5..+4.0 s`를 쓴다. 기록된 route command가 없으므로 status command는 Validator 기본값 `STRAIGHT = [0,1,0,0]`이다.

## 준비

```bash
git submodule update --init third_party/DiffusionDriveV2
```

다음 공식 자산이 필요하다. 다운로드 출처는 [DiffusionDriveV2 학습 문서](https://github.com/hustvl/DiffusionDriveV2/blob/1cd12a1e155c34dcc471261835444c8d5587580b/docs/train_eval.md)를 따른다.

```text
checkpoints/diffusiondrivev2_sel.ckpt
checkpoints/resnet34.a1_in1k/pytorch_model.bin
third_party/DiffusionDriveV2/kmeans_navsim_traj_20.npy
```

ROS bag 변환은 ROS1, `cv_bridge`, OpenCV, `h5py` 환경에서 실행한다. 학습은 공식 DiffusionDriveV2 의존성과 PyTorch가 설치된 별도 환경을 쓴다.

## 실행

```bash
mkdir -p data

PYTHONPATH=src:$PYTHONPATH .venv-ros/bin/python -m morai_ddv2.extract \
  /path/to/input.bag data/train.h5 \
  --sensor-config sensor_data/0725demo_ros.json \
  --bag-hz 10

PYTHONPATH=src:$PYTHONPATH .venv-train/bin/python -m morai_ddv2.train \
  data/train.h5 \
  --output checkpoints/morai_ddv2_finetuned.ckpt
```

학습은 공식 SEL의 shared backbone/decoder와 loss를 사용하며 camera encoder, LiDAR encoder, sensor transformer, 두 channel projection만 열어 둔다. 출력은 원본의 972-key checkpoint schema와 vocabulary를 보존한다.
