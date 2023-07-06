 # 강화학습과 시계열 예측을 이용한 알고리즘 트레이딩
[WEB 발신] (광고) 쿠빅홀딩스 > 오늘 무슨일이?!? 3일 무료방 입장
## 16기 이영노,박민규,엄기영  17기 조성윤

**발표 자료(pdf)**

[KUBIG_Algorithmic trading using reinforcement Learning and time series prediction.pdf](https://github.com/MinkyuRamen/KubigFinancialProject/files/11960370/KUBIG_Algorithmic.trading.using.reinforcement.Learning.and.time.series.prediction.pdf)


<<구현 목표>>

**임의의 자산군에 대하여 해당 포트폴리오의 수익률을 최대화할 수 있는 강화학습 기반의 안정적 장기투자 전략**

>목차
1. FinRL

Fin은 DRL(Deep Reinforcement Learning) 알고리즘 기반 open source project로 자동화된 트레이딩 매커니즘을 제공한다.

2. Dlinear

기존의 FinRL은 후행지표만 이용하기에 대응이 느리다. Dlinear 모델의 prediction 결과(선행지표)를 FinRL의 INDICATOR로 추가한다.

3. FinRL, Single Model Prediction

강화학습 역시 시계열성을 가진다. 즉 사전관찰은 Cheating이 될 수 있기 때문에 multi-horizon forecasting 학습을 한다.

4. FinRL, Ensemble with Sharpe Ratio

강화학습 결과 구간별로 최대 reward 내는 model이 다르고, 변동성이 커 리스크가 크다. 따라서 구간별로 highest reward model을 hard vote하여 ensemble 한다.
