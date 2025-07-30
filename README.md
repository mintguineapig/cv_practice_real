# README

## Overview
1. 이 저장소는 CIFAR-10, CIFAR-100, TinyImageNet 데이터셋에서 여러 Vision 모델을 훈련하고 비교한 실험 모음입니다.  
2. 각 실험은 **weak** 또는 **default** augmentation 설정을 적용했고, **Adam** 옵티마이저(일부 실험은 **SGD**)를 사용했습니다.  
3. 실험별 `exp-name` 규칙은 다음과 같습니다:

{데이터셋코드}_{모델명}[_입력크기]_augmentation_Optimizer

- **데이터셋코드**: `10`=CIFAR-10, `100`=CIFAR-100, `200`=TinyImageNet  
- **입력크기**: 기본 요구 크기(224×224) 사용 시 생략, 32×32로 강제 조정 시 `32x32` 표기  
- **augmentation**: `weak` 또는 `default` (ResNet-18만 `strong`도 실행)  
- **Optimizer**: `Adam` 또는 `SGD`

## Purpose

1. 본 연구는 CIFAR-10, CIFAR-100, Tiny ImageNet과 같은 저해상도 이미지 분류 작업에서 최신 컨볼루션 및 트랜스포머 기반 아키텍처들의 성능, 메모리 효율성, 학습 안정성을 체계적으로 평가하고 비교하는 것을 목표로 합니다.
2. 데이터 증강 전략, 최적화 알고리즘, 모델 아키텍처 제한된 계산 자원 하에서 효율적인 학습과 높은 정확도를 달성하는 데 미치는 영향을 중점적으로 다룹니다.

## Aim
- 서로 다른 모델들이 `32x32` 및 `64x64` 입력 이미지를 얼마나 잘 처리하는지 평가 (성능평가) -> 마저 수행
- 각 아키텍처별 메모리 및 연산 비용의 상충관계 이해 (OOM 여부 판단)
- 최적화 전략이 모델의 수렴 속도, 성능에 미치는 영향 평가 (`Adam` vs `SGD`)
- 다양한 증강 전략이 모델 일반화 능력에 미치는 영향 평가 (`weak` vs `default` vs `strong`)
- 자원 제한 환경에 적합한 모델 및 설정을 선택하기 위한 실용적인 가이드 제공






---

## **Installation**
```bash
# 시스템 요구
Python 3.8  
PyTorch ≥1.12  
CUDA 12.8  

# 필수 패키지
pip install \
  wandb==0.20.0 \
  pydantic==1.10.2
```

## **Docker**
```bash
docker run --gpus all -it -h cv_practice_gpu \
  -p 1290:1290 \
  --ipc=host \
  --name cv_practice_gpu \
  -v /m2:/projects \
  nvcr.io/nvidia/pytorch:22.12-py3 bash
```



## **Usage**
```bash
--dataname {CIFAR10|CIFAR100|TinyImagenet}
--num-classes {10|100|200}
--model-name <timm model name or resnet18>
--opt-name {Adam|SGD}
--aug-name {weak|default|strong}
--batch-size <int>
--lr <float>             # optional, omit to use default
--use_scheduler
--epochs <int>
--img-size <int>         # specify only if forcing 32×32 input
--exp-name <experiment name>
```

## **Examples**
### ResNet-18 on CIFAR-10 (default augment)
```bash
python main.py \
  --dataname CIFAR10 \
  --num-classes 10 \
  --model-name resnet18 \
  --opt-name Adam \
  --aug-name default \
  --batch-size 64 \
  --lr 0.1 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_resnet18_default_Adam
```

### EfficientNet-B0 on CIFAR-10 (weak augment)
```bash
./run_timm.sh efficientnet_b0 \
  --dataname CIFAR10 \
  --num-classes 10 \
  --opt-name Adam \
  --aug-name weak \
  --batch-size 64 \
  --lr 0.1 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_efficientnet_b0_weak_Adam
```

### ViT-Small (32×32) on CIFAR-10
```bash
./run_timm.sh vit_small_patch32_32 \
  --dataname CIFAR10 \
  --num-classes 10 \
  --model-name vit_small_patch32_32 \
  --opt-name Adam \
  --aug-name default \
  --batch-size 128 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --img-size 32 \
  --exp-name 10_ViTsmall32x32_default_Adam
```

### ConvNeXt-Base on CIFAR-10 (weak augment)
```bash
python main.py \
  --model-name convnext_base \
  --dataname CIFAR10 \
  --num-classes 10 \
  --opt-name Adam \
  --aug-name weak \
  --batch-size 128 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_ConvNeXt_weak_Adam
```

### naflexvit_base on CIFAR-10 (weak augment)
```bash
python main.py \
  --model-name naflexvit_base_patch16_gap.e300_s576_in1k \
  --dataname CIFAR10 \
  --num-classes 10 \
  --opt-name Adam \
  --aug-name weak \
  --batch-size 64 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_naflexvit_weak_Adam
```

### naflexvit_base on CIFAR-10 (default augment)
```bash
python main.py \
  --model-name naflexvit_base_patch16_gap.e300_s576_in1k \
  --dataname CIFAR10 \
  --num-classes 10 \
  --opt-name Adam \
  --aug-name default \
  --batch-size 64 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_naflexvit_default_Adam
```

### ResNet-18 on CIFAR-100 (weak augment)
```bash
python main.py \
  --dataname CIFAR100 \
  --num-classes 100 \
  --model-name resnet18 \
  --opt-name Adam \
  --aug-name weak \
  --batch-size 128 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 100_resnet_weak_Adam
```

### ResNet-18 on TinyImageNet (default augment)
```bash
python main.py \
  --dataname TinyImagenet \
  --num-classes 200 \
  --model-name resnet18 \
  --opt-name Adam \
  --aug-name default \
  --batch-size 128 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 200_resnet18
```

---


| Exp Name                          | Dataset      | Model           | Input Size | Augment | Batch | Optimizer | Scheduler | Epochs |
| --------------------------------- | ------------ | --------------- | ---------- | ------- | ----- | --------- | --------- | ------ |
| `10_resnet18_default_Adam`        | CIFAR-10     | ResNet-18       | 32×32      | default | 64    | Adam      | Yes       | 50     |
| `10_resnet18_weak_Adam`           | CIFAR-10     | ResNet-18       | 32×32      | weak    | 64    | Adam      | Yes       | 50     |
| `10_resnet18_strong_Adam`         | CIFAR-10     | ResNet-18       | 32×32      | strong  | 64    | Adam      | Yes       | 50     |
| `10_efficientnet_b0_default_Adam` | CIFAR-10     | EfficientNet-B0 | 32×32\*    | default | 64    | Adam      | Yes       | 50     |
| `10_efficientnet_b0_default_SGD`  | CIFAR-10     | EfficientNet-B0 | 32×32\*    | default | 64    | SGD       | Yes       | 50     |
| `10_efficientnet_b0_weak_Adam`    | CIFAR-10     | EfficientNet-B0 | 32×32\*    | weak    | 64    | Adam      | Yes       | 50     |
| `10_ViTsmall32x32_default_Adam`   | CIFAR-10     | ViT-Small       | 32×32      | default | 128   | Adam      | Yes       | 50     |
| `10_ViTsmall32x32_weak_Adam`      | CIFAR-10     | ViT-Small       | 32×32      | weak    | 128   | Adam      | Yes       | 50     |
| `10_ConvNeXt_default_Adam`        | CIFAR-10     | ConvNeXt-Base   | 32×32\*    | default | 128   | Adam      | Yes       | 50     |
| `10_ConvNeXt_weak_Adam`           | CIFAR-10     | ConvNeXt-Base   | 32×32\*    | weak    | 128   | Adam      | Yes       | 50     |
| `10_naflexvit_weak_Adam`          | CIFAR-10     | naflexvit\_base | 224×224    | weak    | 64    | Adam      | Yes       | 50     |
| `10_naflexvit_default_Adam`       | CIFAR-10     | naflexvit\_base | 224×224    | default | 64    | Adam      | Yes       | 50     |
| `100_resnet18_weak_Adam`            | CIFAR-100    | ResNet-18       | 32×32      | weak    | 128   | Adam      | Yes       | 50     |
| `200_resnet18_default_Adam`                    | TinyImageNet | ResNet-18       | 224×224    | default | 128   | Adam      | Yes       | 50     |

* CIFAR inputs (32×32) were forced via resizing.


