# 💻UNet_breastcancer
- 데이터: Kaggle - [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- 이미지 분류 학습: Youtube - [DigitalSreeni](https://www.youtube.com/watch?v=azM57JuQpQI&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE)
- 환경: Python 3.8.8

## 프로젝트 목적
- UNet을 활용한 유방암 이미지 분류

## 프로젝트 방법
- 전처리
  -  이미지 Resize(128*128)
  -  종양이 2개 이상일 때 마스크 이미지를 하나로 만들기 (동일한 부분이 겹칠 시 해당 영역을 하나로 표현)
- UNet
  - 이미지 분류, Segmentation
