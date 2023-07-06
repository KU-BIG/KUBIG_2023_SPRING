# parametric

unknown environment --> parametric assumption : 주가가 OU process를 따른다. (ornstein uhlenbeck process. aka mean-reversion process)

  - parametric assumption 통한 statistical efficiency 확보 (충분한 episode 확보 되었을때)
  - precise dynamics

turbulence_threshold 99% --> 90% 더 보수적으로 수정

(model train) hyperparameters tuned. 

  - optuna
  - A2C convergence issue로 인해 std=1 로 맞추는 ent_coef : 0.005, learning_rate : 0.0007 사용
  - DDPG convergence issue로 인해 learning_rate : 0.0005 사용

DLinear 지표 추가 (5,10,30 lag) 
