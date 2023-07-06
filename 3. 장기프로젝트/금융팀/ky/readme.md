
# FinRL prediction adding

- Prediction Data 이전의 기본 소스코드 참조시 AlgorithmTrading folder  참조.

- Prediction Data 추가시 코드 확인하려면 FinRL_w_prediction folder 참조.

----

### Prediction 적용 방법
1. 만약 직접 prediction 부터 다시 진행을 원하는 경우 prediction_pipeline.ipynb 에서 수정할 것. 
    
    window_size(과거 시점 얼마나 참조할건지, default = 252)
    
    forecast_size(몇 시점을 예측할 건지, default = 30) 
2. 데이터 그냥 사용하려면 prediction_5 ~ 30 csv를 본인 디렉토리로 복사 + 붙여넣기 후, FINRL_application.ipynb에 있는 주석 달린 코드 블럭 네 개를 본인 코드에 집어 넣으면 됩니다.

    주석 : #prediction 업로드시 확인 필요부분


