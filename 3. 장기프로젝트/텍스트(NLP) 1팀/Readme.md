# 영어 답안 생성 chatbot

## 야, 너두 쓸 수 있어.
### 16기 박종혁 정은미 하예은 17기 김희준

### 주제 소개
- IELTS 영어 스피킹 문항에 대한 답안을 생성해주는 chatbot 구현

### 데이터
- TED 강연 4,214개 대본 크롤링
- null값 처리, 정규표현식 등 전처리 진행

### 모델링
- GPT2 직접 구현 후 fine tuning.
- 다소 아쉬운 성능 탓에 openAI에서 GPT3 API 사용, fine tuning
- curie 모델로 fine tuning 진행

### 결과
- 파라미터인 max_token, temperature 값에 따라 다양한 답변이 출력되는 것을 확인.
