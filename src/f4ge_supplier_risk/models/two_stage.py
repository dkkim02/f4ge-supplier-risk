"""2단 모델 — 풍부한 신호가 순위를, 희소한 라벨이 눈금을 잡는다.

1단 모델은 escape 를 직접 맞힌다. 그런데 오더 하나에서 우리가 보는 것은
**표본 88개 중 불량 0.11개**다(8 seed 중앙). 900 오더를 다 모아도 학습 신호가 거의 없다.

대수적으로 쪼개면 이렇게 된다.

    escape = internal x (1 - detection)          <- 우리가 원하는 값
    보고   = internal x bias                     <- 전수에 가깝다. 사건 수 수백 배
    ------------------------------------------------
    escape = 보고 x (1 - detection) / bias
                    \\_______ kappa (공장별 상수) _______/

**`detection` 과 `bias` 를 따로 알 필요가 없다.** 둘의 비 kappa 만 알면 escape 가 나온다.
그리고 kappa 는 공장당 스칼라 하나라, 희소한 라벨로도 추정이 선다.

    1단계  보고 카운트 (k_rep, n_rep) 를 L0 사전분포 쪽으로 shrink  ->  internal_hat
    2단계  우리 입고검사 카운트로 공장별 kappa 를 Gamma-Poisson conjugate prior 로 추정

**두 단계 모두 시간을 지킨다** — kappa 는 그 오더를 발주하던 시점까지 **도착한** 라벨만으로
갱신한다. 유럽 오더는 라벨이 두 달 뒤에 오므로, 이걸 어기면 현실에서 불가능한 추정이 된다.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 보고 카운트를 사전분포 쪽으로 당기는 세기(의사관측 수).
# 생산 100개짜리 보고는 크게 당기고, 2,000개짜리는 거의 그대로 둔다.
_SHRINK_PSEUDO = 150.0
# kappa 사전분포. 이력 없는 공장은 전체 평균 kappa 쪽으로 당겨진다(partial pooling).
_KAPPA_PRIOR_STRENGTH = 3.0
_FLOOR = 1e-7


def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _stage_a(train: pd.DataFrame, frame: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    """보고 카운트 + L0 사전분포 -> 내부 불량률 추정.

    보고가 결측이면 `n_rep` 가 0이라 사전분포 값이 그대로 나온다. 결측을 0으로 채우는
    것과 다르다 — **모른다는 것과 없다는 것을 구분한다.**
    """
    use = [c for c in cols if not c.startswith("l1_rep")]

    ktr = train["l1_rep_defects"].to_numpy(float)
    ntr = train["l1_rep_produced"].to_numpy(float)
    seen = ntr > 0
    ytr = np.log((ktr[seen] + 0.5) / (ntr[seen] + 50.0))

    prior_model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0)
    )
    prior_model.fit(train.loc[seen, use], ytr)
    prior = np.exp(prior_model.predict(frame[use]))

    k = frame["l1_rep_defects"].to_numpy(float)
    n = frame["l1_rep_produced"].to_numpy(float)
    return (k + _SHRINK_PSEUDO * prior) / (n + _SHRINK_PSEUDO)


def _kappa_by_factory(
    train: pd.DataFrame, internal_hat_train: np.ndarray
) -> tuple[dict[str, tuple[float, float]], float]:
    """공장별 kappa 의 Gamma 사후분포 파라미터.

    관측 모형은 `k ~ Poisson(n x internal_hat x kappa)` 이고 Gamma 가 conjugate prior 라
    사후분포가 닫힌 형태로 나온다 — `Gamma(a + sum k, b + sum n x internal_hat)`.
    표본을 200개 뽑아 0건이 나온 것과 50개 뽑아 0건이 나온 것이 자동으로 다르게 반영된다.
    """
    n = train["y_inspected"].to_numpy(float)
    k = train["y_reject"].to_numpy(float)
    exposure = n * internal_hat_train

    pooled = k.sum() / max(exposure.sum(), _FLOOR)
    a0 = _KAPPA_PRIOR_STRENGTH
    b0 = a0 / max(pooled, _FLOOR)

    post: dict[str, tuple[float, float]] = {}
    for fid, idx in train.groupby("factory_id").indices.items():
        post[str(fid)] = (a0 + k[idx].sum(), b0 + exposure[idx].sum())
    return post, pooled


def explain(
    train: pd.DataFrame, test: pd.DataFrame, cols: tuple[str, ...]
) -> dict[str, object]:
    """모델 내부값 — 화면에서 "왜 이 숫자인가" 를 보여주려면 필요하다.

    `escape_hat = internal_hat x kappa` 라는 두 단계가 그대로 노출된다.
    kappa 는 `(1 - detection) / bias` 이므로 **"이 공장의 보고를 얼마로 보정하는가"** 다.
    """
    internal_train = _stage_a(train, train, cols)
    post, pooled = _kappa_by_factory(train, internal_train)
    return {
        "internal_hat": _stage_a(train, test, cols),
        "kappa": {fid: a / b for fid, (a, b) in post.items()},
        "kappa_pooled": pooled,
        "kappa_obs": {fid: int(a - _KAPPA_PRIOR_STRENGTH) for fid, (a, _) in post.items()},
    }


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, cols: tuple[str, ...]
) -> np.ndarray:
    """테스트 오더별 유닛당 예상 불량 확률.

    테스트 구간에서는 kappa 를 **온라인으로 갱신**한다. 실제 운영이 그렇게 돌아간다 —
    오더를 배분하는 시점에 그때까지 도착한 라벨만 손에 있다.
    """
    internal_train = _stage_a(train, train, cols)
    post, pooled = _kappa_by_factory(train, internal_train)
    a0 = _KAPPA_PRIOR_STRENGTH
    b0 = a0 / max(pooled, _FLOOR)

    internal_test = _stage_a(train, test, cols)

    # 테스트 오더의 라벨이 언제 도착하는지 — 그 시점 이후에만 kappa 갱신에 쓸 수 있다
    # 운영에서는 진행 중인 오더에 라벨이 없다(label_available_at None). 그 행은 κ 갱신에 못 쓴다 —
    # 도착하지 않은 라벨을 쓰면 시간 규율(§5.4)을 어기는 것과 같다.
    arrivals = sorted(
        (_dt(r.label_available_at), str(r.factory_id), float(r.y_inspected), float(r.y_reject), i)
        for i, r in enumerate(test.itertuples())
        if isinstance(r.label_available_at, str) and r.y_inspected == r.y_inspected
    )
    order = np.argsort([_dt(s) for s in test["ordered_at"]])

    out = np.empty(len(test))
    ptr = 0
    for pos in order:
        now = _dt(test["ordered_at"].iloc[pos])
        while ptr < len(arrivals) and arrivals[ptr][0] <= now:
            _, fid, n_i, k_i, idx = arrivals[ptr]
            a, b = post.get(fid, (a0, b0))
            post[fid] = (a + k_i, b + n_i * internal_test[idx])
            ptr += 1
        a, b = post.get(str(test["factory_id"].iloc[pos]), (a0, b0))
        out[pos] = internal_test[pos] * (a / b)

    return np.clip(out, _FLOOR, 1 - 1e-6)
