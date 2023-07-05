## 진짜는 가짜를 알아보는 법!
#### 가짜같은 진짜들 - 16기 임정준, 17기 김지윤, 문성빈, 백서경, 임종우



### 주제: 합성 데이터를 활용한 자동차 탐지 AI 모델 개발

### 데이터 소개
1. train
합성 이미지로 구성된 학습 데이터
이미지는 png 확장자, 동일한 파일명으로 매핑되는 txt파일에 annotation 정보가 제공됨.
annotation 정보 : 1) class_id  2) LabelMe 형식의 Bounding Box 좌표  
3) (객체의 4개 꼭지점(x, y) 좌표 정보)

2. test
실제 이미지로 구성된 평가 데이터
이미지는 png 확장자

3. classes.txt
class_id, class_name 존재
총 34개의 자동차 세부 종류 Class


### 시도한 모델
1) Faster-RCNN : baseline 코드
2) YOLO - v5, v8 : real-time object detection의 시초, yolo
3) YOLO - NAS : 가장 최근에 나온 yolo family의 foundation model

### 모델링 과정
1) 최종 모델 선정 : YOLO - v8
2) Image Augmentation (MotionBlur, AdvancedBlur, GaussianNoise)
3) Weighted Box Fusion

### 대회 결과
Public | 35th 
score : 0.94556 






