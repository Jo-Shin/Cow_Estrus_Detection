# [스마트축사 데이터 활용 대회](http://aifactory.space/competition/detail/1952)
- By: 연세대학교 Data Science Lab 5기 조신형, 5기 박채은, 6기 박상윤, 6기 이승재

## 주제 및 데이터
- **주제** : 한우의 발정 시기를 파악하여 발정 시기를 놓치지 않고 수정에 성공하도록 스마트축사 이미지 데이터 내에 발정행동중인 한우를 판별하는 **instance segmentation model**을 개발
  - *instance segmentation의 정의는 [다음](https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb)을 참고*
- **데이터**: 축사 내 한우의 이미지 파일 (*저작권의 이유로 본 repository에 업로드하지 않음*)

## Model 
- coco dataset에 pretrained된 mask R-cnn model을 한우 이미지를 통해 fine-tuning
- fine-tuning 시 다양한 hyper-parameter를 조정해 보았으나, 최고 성능을 낸 hyper-parameter는 아래와 같음
    - rpn_nms_threshold = 0.6
    - lr = 0.001
    - layers = all
- test data에 inference 시 detection_min_confidence는 0.4

## Sample inference
**anestrus(비발정)**, **estrus(발정)**





