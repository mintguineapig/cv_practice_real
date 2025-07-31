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

## Result
1. CNN based model vs Transformer based model
   
  1.1. Performance    
  <img width="439" height="914" alt="image" src="https://github.com/user-attachments/assets/3f5f681e-dcce-452b-8d93-e1bc37c3395d" />
  

  파란색/초록색(푸른계열)이 각각 ViTsmall, naflexvit (Transformer 기반) 로 성능이 매우 나빴음.   
  주황색/빨간색/분홍색(붉은 계열) 이 각각 efficientnet, ResNet, ConvNeXt (CNN 기반) 으로 성능이 비교적 우수했음.   
  전통적인 resnet, efficientnet 이 convnext 보다 성능이 좋았음.    
  convnext 의 파라미터가 많아 overfitting 되었을 것으로 추정.    
  
  

  
  
2. Augmentation method - weak vs default   
  2.1. Performance
   
    2.1.1. Resnet18 (11M)
   
    <img width="602" height="315" alt="image" src="https://github.com/user-attachments/assets/a8252f9b-d9ff-4609-8ae5-2eb4b8438a56" />
   
    default 에서 오버피팅, 최종 성능이 weak 에서 우수  
    weak 성능 good: 데이터가 그나마 좀 더 복잡해졌기 떄문이라고 추정
    default 성능 bad: 데이터의 다양성이 부족했기 때문이라고 추정
      
    2.1.2. efficientnet (5.3M)
   
    <img width="592" height="311" alt="image" src="https://github.com/user-attachments/assets/5e8e5f88-1747-407c-b0ed-0ffeabb48020" />
    
    파라미터가 resnet 보다 적음
    dafault 와 weak 에서 최종성능은 유사했으니, default 에서 loss 가 튀는 현상 발생  
    loss가 튀는 이유?
   
    2.1.3. vit_small_patch16_32
   
    <img width="607" height="312" alt="image" src="https://github.com/user-attachments/assets/fe2662c5-bb53-4458-9bbf-711611d5cc90" />
    
    최악의 결과 (당연)   
    default와 weak에서 모두 loss가 점차 감소하는 경향은 보였지만,  
    default 는 끝까지 수렴하지 못하고, 최종성능도 더 낮음  
    epoch 을 증가시키면 둘 다 수렴할 가능성 존재
   해당 모델이 patch size = 16 이기 때문에 2*2 토큰으로 학습하기 때문에 충분한 정보를 학습X (너무 적은 정보로 학습)  
    patch size를 줄여서 실험 재수행 필요
   
    2.1.4. naflexvit
   
    <img width="599" height="311" alt="image" src="https://github.com/user-attachments/assets/64ee1bcd-0cac-482b-82e7-6c0a699e040a" />
   

    weak은 loss는 감소하는데, 정확도는 0.5 를 간신히 넘었으며,   
    default 는 loss 가 다시 점점 커지고, 정확도도 0.55로 수렴함.
    성능이 좋지 않음은 분명하나, 
    naflexvit 가 데이터 사이즈에 유연하긴 하나, 그것은 세로/가로 비율과 다양한 크기의 데이터셋에서 유연하게 작용하는 것이라 본 데이터(32*32 통일) 에서는 큰 장점을 발휘하지 못함.
    남은 차이점은 (1) 하이퍼파라미터, (2) 위치 임베딩 인데, 이 점이 미세한 성능 향상을 야기했을 것이라 예상     

       
    2.1.5. ConvNeXt (88M)
   
    <img width="594" height="316" alt="image" src="https://github.com/user-attachments/assets/5c90f06b-e075-4186-9e81-3360fda36e16" />
   
    수렴은 하였으나 오버피팅. 특히 증강을 하지 않은 default 에서 더 심함. 
    모델의 용량이 큰 것에 비해 데이터셋이 너무 저해상도로 작기 때문일 것이라고 추정.
    
    
    

    

    


    

## Problem -> Planned Experiments

1. augmentation 방법을 strong 으로 실행시킨 모델의 train_loss 가 감소하는 중에 학습이 중단됨    
-> epoch을 epoches 를 200으로 실험 실행

2. resnet18, efficientnet b0 와 같이 기본적인 모델에서만 실행시켜 파라미터 크기에 따른 성능을 충분히 비교하지 못함    
-> resnet50, efficientnetv2 와 같이 더 크고 개선된 모델 사용

3. 앙상블 했을 때 성능 개선이 궁금했으나 실행하지 못함    
-> 코드를 다시 작성하여 앙상블 시도

4. 데이터셋을 CIFAR10으로 제한하여 저해상도 이미지 처리에 대한 insight 부족    
-> CIFAR100, TinyImageNet dataset 을 실행하며 공통점, 차이점을 비교하며 insight 도출
   <img width="894" height="685" alt="image" src="https://github.com/user-attachments/assets/c5534e50-0690-45cf-8c8a-5218f24155ad" />
   오류 수정 예정..


6. 최신 모델은 고해상도 이미지에 특화되어 있어 성능이 충분히 나오지 않음    
-> 고해상도 이미지 데이터셋 Open Images V6, Places365, COCO(Common Objects in Context) 이용해 고해상도 이미지와 저해상도 이미지 처리에 어떠한 모델 특성이 유리한지 확인 (복잡한 모델이 고해상도에서 좋은 성능이 나오는지 확인!)

7. 오버피팅을 해결해보지 못함    
-> 오버피팅이 발생하는 경우가 있었고, 이러한 문제를 해결하기 위한 (1) Early stopping, (2) Regularization / Dropout 등 정규화 기법을 시도하여 각 모델별 최적 파라미터에 대한 실험 수행

