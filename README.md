# [스마트축사 데이터 활용 대회](http://aifactory.space/competition/detail/1952)
- By: 연세대학교 Data Science Lab 5기 조신형, 5기 박채은, 6기 박상윤, 6기 이승재

## 주제 및 데이터
- **주제** : 한우의 발정 시기를 파악하여 발정 시기를 놓치지 않고 수정에 성공하도록 스마트축사 이미지 데이터 내에 발정행동중인 한우를 판별하는 **instance segmentation model**을 개발
- **데이터**: 축사 내 한우의 이미지 파일 *(저작권의 이유로 본 repository에 업로드하지 않음)*

## Model 
- Coco dataset에 pretrained된 Mask R-CNN Model을 한우 이미지를 통해 fine-tuning
- Fine-tuning 시 다양한 hyper-parameter를 조정해 보았으나, 최고 성능을 낸 hyper-parameter는 아래와 같음
    - ```rpn_nms_threshold = 0.6```
    - ```lr = 0.001```
    - ```layers = all```
- Test data에 inference 시 ```detection_min_confidence= 0.4```

## Customized Code
- ```mrcnn/cow.py```: Pretrained Mask R-CNN을 fine-tuning하기 위한 Code
- ```mrcnn/contour.py```: Model의 inference 결과에서 테두리의 좌표값을 구하는 code
- ```inference.ipydb```: Inference Code

## Sample Inference
**anestrus(비발정)**, **estrus(발정)**





