# Deep Learning NLP Project
#### TEAM : 16기: 김진서, 노연수, 정은미/ 17기: 안영지

### 데이콘 소설 작가 분류 AI 경진대회
#### link : https://dacon.io/competitions/open/235670/overview/description

문체 분석 알고리즘 개발을 통한 작가 예측

## Data
    - index
    - author: 5명의 작가
    - text : 54879개의 문장뭉치
    
## 전처리
    1. 부호 제거
    2. 불용어 처리X
    3. 표제어 추출
    4. GloVe 임베딩 사용

## 모델
    1. CNN
    2. RNN기반: BI-LSTM, GRU
    3. CNN+RNN

## Result
     최종모델) BI-LSTM 모델 - logloss: 0.322 


