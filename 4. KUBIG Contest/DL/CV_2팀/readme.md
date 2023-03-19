![image](https://user-images.githubusercontent.com/109740391/226185363-e0abaec6-2fb4-4ef6-8929-1d8d92b6d417.png)

# Computer Vision 이상치 탐지 프로젝트

### 팀원 : 16기 이수찬, 17기 김지윤, 17기 이서연

## 대회 소개 

대회 링크: [https://dacon.io/competitions/official/235894/overview/description](https://dacon.io/competitions/official/235894/overview/description)

MVtec AD Dataset에 들어있는 사물의 종류를 분류하고 정상 샘플과 비정상(이상치) 샘플을 분류   
-> 불균형 데이터 셋을 학습하여 사물의 종류와 상태를 분류할 수 있는 컴퓨터 비전 알고리즘 개발

## 1. 데이터

  MVtec AD Dataset

1. train [Folder] : 4277개 이미지, 88개 label
2. test[Folder] : 2154개 이미지 
3. train_df(csv) : train folder에 대한 정보(인덱스, 파일명, 클래스 등)
4. test_df(csv) : test folder에 대한 정보(인덱스, 파일명)

## 2. 전처리 
- Augmentation
- Normalization

## 3. 모델 학습
- 1차 시도 : transforms.RandomAffine((-45,-45)) + efficientnet_b3 + 
epoch수(30)
- 2차 시도 : transforms.RandomAffine((-180,180)) +efficientnet_b3
+epoch수(40)
- 3차 시도 : 2차 시도 + epoch수(50)로 진행
- 4차 시도 : 3차 시도 + epoch수(40)+post-processing 진행

## 4. 후처리
- 1) class 불균형으로 인한 good label의 과한 예측 방지
- 2) i. 의 결과를 이용해도 헷갈려하는 class(zipper,toothbrush)에 대해 추가 학습
