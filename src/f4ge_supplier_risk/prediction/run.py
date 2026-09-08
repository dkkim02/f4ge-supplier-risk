"""채점 파이프라인 — 생성 데이터 → `supplier-risk-score.v1` 줄들.

시간순으로 자르고, 학습 구간에서 모델과 threshold을 잡고, 테스트 구간을 채점한다.
테스트 구간이 곧 **현재 수주 잔고**에 해당한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f4ge_supplier_risk.evaluation.metrics import evaluate, split_by_time
from f4ge_supplier_risk.features.build import LAYERS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import baseline, discrepancy, two_stage
from f4ge_supplier_risk.prediction.score import build_scores, to_contract

FULL = LAYERS["L0+Cell+MES"]
L0 = LAYERS["L0"]
_EPS = 1e-7


def _predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """보고가 있는 오더는 2단, 없는 오더는 **2단의 공장 수준 × 1단 L0 의 공장 내 편차.**

    보고 없는 오더에 2단만 쓰면 사전분포 × κ 라 공장 안에서 오더를 구분하지 못하고(유형 c 순위 0.425),
    1단 L0 만 쓰면 공장 수준을 놓친다 — L0 는 학습 구간에 고정되지만 2단은 테스트 구간에서
    κ 를 온라인 갱신해 공장 수준을 따라간다. 둘을 곱해 각자 잘하는 것만 남긴다.
    scripts/mixed_calibration.py (8 seed): 전체 0.600 → 0.661, 7/8 seed 개선, 최악 −0.100.
    유형 b 0.425 → 0.523 · c 0.425 → 0.545 · a 는 그대로 0.851.
    """
    p2 = two_stage.fit_predict(train, test, FULL)
    p0 = baseline.fit_predict(train, test, L0)
    ev = test["l1_rep_produced"].to_numpy(float) > 0
    # 공장별 L0 기하평균으로 나누면 공장 수준이 빠지고 오더 간 편차만 남는다 (라벨을 쓰지 않는다)
    logp0 = np.log(p0 + _EPS)
    fac_mean = pd.Series(logp0).groupby(test["factory_id"].to_numpy()).transform("mean").to_numpy()
    return np.where(ev, p2, p2 * np.exp(logp0 - fac_mean))


def score_all(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """`(채점 결과, 공장 신뢰도, 평가 지표)`."""
    data = build_dataset(cfg)
    table = build(data)
    train, test = split_by_time(table)

    pred_tr = _predict(train, train)
    pred = _predict(train, test)
    disc_tr = discrepancy.fit_predict(train, train)
    disc = discrepancy.fit_predict(train, test)

    scored = build_scores(
        train, pred_tr, disc_tr, test, pred, disc, discrepancy.reasons(train, test)
    )
    trust = discrepancy.factory_trust(scored)
    metrics = evaluate(test, pred)
    return scored, trust, metrics


def write_scores(scored: pd.DataFrame, path: Path | str) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for _, row in scored.iterrows():
            fh.write(json.dumps(to_contract(row), ensure_ascii=False) + "\n")
    return len(scored)
