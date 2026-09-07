"""평가 지표.

**이진 지표를 주 지표로 쓰지 않는다.** 900 오더에 로트 불합격 20여 건이고
시간순 테스트 구간에는 열 건이 안 되게 남는다. 신뢰구간이 결론보다 넓다.

주 지표는 둘이다.
  · `rank_corr_true`  — 예측 위험도가 **진짜 escape rate** 순서를 얼마나 맞히나.
    합성 데이터에서만 잴 수 있고, 그래서 ground_truth 를 따로 만들어 둔 것이다.
  · `binom_logloss`   — 관측된 (k, n) 에 대한 유닛당 이항 로그손실. 실데이터에서도 잰다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_EPS = 1e-9


def evaluate(test: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    n = test["y_inspected"].to_numpy(dtype=float)
    k = test["y_reject"].to_numpy(dtype=float)
    p = np.clip(pred, _EPS, 1 - _EPS)

    ll = -(k * np.log(p) + (n - k) * np.log(1 - p)).sum() / n.sum()
    rank = spearmanr(pred, test["true_escape_rate"].to_numpy()).statistic

    return {
        "rank_corr_true": float(rank),
        "binom_logloss": float(ll),
        "pred_ppm_median": float(np.median(pred) * 1e6),
    }


def split_by_time(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시간순 분할. 무작위 분할은 금지 — 기존 프로젝트에서 무작위 분할이
    ROC-AUC 를 0.558 → 0.71~0.80 으로 부풀린 사례가 있다."""
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
