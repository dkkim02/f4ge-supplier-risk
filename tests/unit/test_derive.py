"""계약 행 → 파생 → build() 가 생성기 직행 build() 와 같은 피처를 내는가 (왕복)."""

import numpy as np
import pandas as pd

from f4ge_supplier_risk import contracts
from f4ge_supplier_risk.features.build import build
from f4ge_supplier_risk.ingestion import derive


def _contract_rows(dataset):
    fac = {o["order_id"]: o["factory_id"] for o in dataset["orders"]}
    products = [
        {"product_code": p["product_id"].removeprefix("prd_"), "nominal_cycle_sec": p["nominal_cycle_sec"],
         "material_per_unit": p["material_per_unit"], "difficulty": p["difficulty"]}
        for p in dataset["products"]
    ]
    return dict(
        products=products,
        orders=[contracts.order_row(o) for o in dataset["orders"]],
        reports=[contracts.report_row(r, fac[r["order_id"]]) for r in dataset["factory_reports"]],
        fai=[contracts.fai_row(f, fac[f["order_id"]]) for f in dataset["fai_reports"]],
        cell=[contracts.cell_row(c, fac[c["order_id"]]) for c in dataset["cell_daily"]],
        erp=[contracts.erp_row(e, fac[e["order_id"]]) for e in dataset["erp_daily"]],
        outcomes=[contracts.outcome_row(q, fac[q["order_id"]]) for q in dataset["quality_outcomes"]],
    )


def test_roundtrip_features_match_generator(dataset):
    direct = build(dataset)
    via = build(derive.dataset_from_contracts(**_contract_rows(dataset)))
    assert list(via["order_id"]) == list(direct["order_id"])
    feature_cols = [c for c in direct.columns if c.startswith(("l0_", "l0m_", "l1_", "l2_", "l3_", "y_"))]
    for c in feature_cols:
        a, b = direct[c].to_numpy(float), via[c].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        # lead_slack 은 생성기가 4자리로 반올림한 값이고 운영 경로는 초 단위 ISO 날짜에서 다시 계산한다 → 1e-4 차이
        tol = 1e-3 if c == "l0_lead_slack" else 1e-6
        assert np.allclose(a[~both_nan], b[~both_nan], atol=tol, equal_nan=True), c
    # 정답값은 운영 경로에 없다 — 전부 NaN 이어야 한다
    assert via["true_escape_rate"].isna().all()


def test_open_orders_have_no_label(dataset):
    """결과가 아직 없는 오더는 y 가 NaN 이고 라벨 시각도 없다 — 0 이 아니다."""
    rows = _contract_rows(dataset)
    rows["outcomes"] = rows["outcomes"][: len(rows["outcomes"]) // 2]
    df = build(derive.dataset_from_contracts(**rows))
    open_ = df["label_available_at"].isna()
    assert open_.any()
    assert df.loc[open_, "y_inspected"].isna().all()
    assert df.loc[~open_, "y_inspected"].notna().all()
