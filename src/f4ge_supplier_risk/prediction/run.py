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
from f4ge_supplier_risk.models import discrepancy, two_stage
from f4ge_supplier_risk.prediction.score import build_scores, to_contract

FULL = LAYERS["L0+Cell+MES+ERP"]


def _predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """2단 모델을 전 오더에 적용한다.

    09-08 저녁: 네 소스(MES·CellOS·ERP·포지 기록)가 12곳 전부에서 오므로 "보고 없는 오더" 분기(혼합 C)가 필요 없다.
    1공장 1오더 데이터에서도 2단 전체 적용(0.526)이 혼합(0.433)보다 높았다(layers_by_type, 8 seed).
    """
    return two_stage.fit_predict(train, test, FULL)


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
