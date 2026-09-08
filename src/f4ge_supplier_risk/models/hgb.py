"""Gradient boosting 후보 — HistGradientBoosting, poisson 손실 · exposure 가중.

라벨은 (표본 n, 불합격 k) 쌍이고 k 는 대부분 0 이다. 비율 k/n 을 그냥 회귀하면 n=50 의 0 과 n=500 의 0 이 같은 무게가 된다.
poisson 손실에 target = k/n, sample_weight = n 을 주면 손실이 exposure n 짜리 Poisson 카운트의 deviance 와 같아진다 —
2단 모델의 Gamma-Poisson 과 같은 관측 모형을 트리로 푸는 셈이다.

공장 정체는 피처로 넣지 않는다(공장 id 원핫은 12개 수준에 300건이라 과적합). 공장 수준은 L0 이력(hist_*)이 대신 담는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

_FLOOR = 1e-7


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: tuple[str, ...], seed: int = 0) -> np.ndarray:
    lab = train[train["y_inspected"].notna() & (train["y_inspected"] > 0)]
    X = lab[list(cols)].to_numpy(float)
    n = lab["y_inspected"].to_numpy(float)
    k = lab["y_reject"].to_numpy(float)
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=15,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(X, k / n, sample_weight=n)
    return np.clip(model.predict(test[list(cols)].to_numpy(float)), _FLOOR, 1 - 1e-6)
