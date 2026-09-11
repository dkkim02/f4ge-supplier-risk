"""운영 채점 — 저장된 계약 행에서 피처를 만들고 채점해 결과를 저장한다.

시간 규율: 학습은 **라벨이 도착한 오더**(label_available_at ≤ 지금)만, 채점은 저장된 오더 전부다.
결과가 없는 오더(진행 중)는 y 가 NaN 이라 학습에서 빠지고, κ 온라인 갱신에도 들어가지 않는다.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from f4ge_supplier_risk.features.build import LAYERS, build
from f4ge_supplier_risk.ingestion import derive
from f4ge_supplier_risk.models import discrepancy, factory_params, two_stage
from f4ge_supplier_risk.prediction.run import _predict
from f4ge_supplier_risk.prediction.score import build_scores, to_contract
from f4ge_supplier_risk.web import db
from f4ge_supplier_risk.web.dashboard_data import build_dashboard

MIN_LABELED = 40  # 이보다 적으면 κ·threshold 가 서지 않는다


def assemble(eng: Engine) -> dict[str, list[dict[str, Any]]]:
    import json

    facs = []
    for f in db.load_factories(eng):
        prof = json.loads(f["profile"]) if f.get("profile") else {}
        facs.append({"factory_id": f["factory_id"], "product_id": f"prd_{f['product_code']}" if f.get("product_code") else "",
                     "has_mes": True, "has_cell": True, "region": f.get("region"), **prof})
    return derive.dataset_from_contracts(
        products=db.load_products(eng), orders=db.load_rows(eng, "supplier-order.v1"),
        reports=db.load_rows(eng, "factory-report.v1"), fai=db.load_rows(eng, "fai-report.v1"),
        cell=db.load_rows(eng, "cell-daily.v1"), outcomes=db.load_rows(eng, "order-quality-outcome.v1"),
        erp=db.load_rows(eng, "erp-daily.v1"), factories=facs,
    )


def score_from_db(eng: Engine, now: datetime | None = None, note: str = "") -> dict[str, Any]:
    now = now or datetime.now(UTC)
    now_s = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    data = assemble(eng)
    if not data["orders"]:
        raise ValueError("오더가 없다 — 먼저 supplier-order.v1 을 넣어야 한다")
    df = build(data)

    labeled = df["label_available_at"].notna() & (df["label_available_at"] <= now_s)
    train = df[labeled].copy()
    if len(train) < MIN_LABELED:
        raise ValueError(f"라벨이 도착한 오더가 {len(train)}건 — 최소 {MIN_LABELED}건이 필요하다")
    test = df[df["ordered_at"] <= now_s].copy()

    pred_tr, pred = _predict(train, train), _predict(train, test)
    disc_tr, disc = discrepancy.fit_predict(train, train), discrepancy.fit_predict(train, test)
    explain = two_stage.explain(train, test, LAYERS["L0+Cell+MES+ERP"])
    scored = build_scores(train, pred_tr, disc_tr, test, pred, disc, discrepancy.reasons(train, test), explain=explain)
    scored["scored_at"] = now_s
    params = factory_params.estimate(train)
    explain["params_pooled"] = params.attrs["pooled"]
    explain["scrap_share"] = params.attrs["scrap_share"]

    dash = build_dashboard(data, scored, explain, today=now)
    score_rows = [to_contract(r) for _, r in scored.iterrows()]
    run_id = now.strftime("%Y%m%dT%H%M%S") + "_" + secrets.token_hex(3)
    db.save_run(eng, run_id, score_rows, dash, n_train=len(train), note=note)
    db.save_profile(eng, run_id, now_s, explain)   # 공장별 d 축적 (§3 ③)
    acts = scored["recommended_action"].value_counts().to_dict()
    return {"run_id": run_id, "scored_at": now_s, "n_train": len(train), "n_scored": len(test), "actions": acts}
