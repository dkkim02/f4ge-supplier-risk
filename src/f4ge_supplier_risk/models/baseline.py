"""베이스라인 — 이항 로지스틱 회귀.

y 가 `(표본 n, 불합격 k)` 쌍이라 **이항 관측**이다. 표본 200개에서의 0과
표본 50개에서의 0은 정보량이 다르고, 그 차이를 무시하면 소량 오더가 모델을 흔든다.

sklearn 에 이항 GLM 이 없으므로 오더 하나를 두 행(불량 1 · 정상 0)으로 펼치고
표본 수를 가중치로 준다 — 이항 로지스틱 회귀와 정확히 같은 우도다.

**소표본이라 여기서 큰 모델을 쓰지 않는다.** 900 오더에 로트 불합격 20여 건이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, cols: tuple[str, ...]
) -> np.ndarray:
    """테스트 오더별 **유닛당 예상 불량 확률**(= 예상 escape rate)."""
    xtr = train[list(cols)].to_numpy(dtype=float)
    n = train["y_inspected"].to_numpy(dtype=float)
    k = train["y_reject"].to_numpy(dtype=float)

    x2 = np.vstack([xtr, xtr])
    y2 = np.concatenate([np.ones(len(xtr)), np.zeros(len(xtr))])
    w2 = np.concatenate([k, np.maximum(n - k, 0.0)])

    keep = w2 > 0
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )
    model.fit(x2[keep], y2[keep], logisticregression__sample_weight=w2[keep])
    return model.predict_proba(test[list(cols)].to_numpy(dtype=float))[:, 1]
