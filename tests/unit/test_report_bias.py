"""보고는 노이즈가 아니라 **편향**이어야 한다."""

import numpy as np

from f4ge_supplier_risk.generator import reports
from f4ge_supplier_risk.generator.production import counts_at, progress_at


def test_reported_defects_are_understated(cfg_biased, world_biased):
    """공장이 보고한 불량은 실제보다 적다. 평균이 어긋나야 불일치 탐지가 값을 한다.

    **현실 편향 세계에서만 성립한다** — 기본 설정은 2026-09-10 부터 편향을 통제한다.
    """
    cfg, world = cfg_biased, world_biased
    diffs = []
    for order, factory, product, lat, truth in world:
        rows, _ = reports.build_reports(cfg, order, factory, product, lat, truth)
        filed = [r for r in rows if not r["is_missing"]]
        if not filed:
            continue
        last = filed[-1]
        true_c = counts_at(truth, order["order_qty"], 1.0)
        diffs.append(
            (true_c["scrap"] + true_c["rework"]) - (last["scrap_qty"] + last["rework_qty"])
        )
    assert np.median(diffs) >= 0
    assert np.mean(np.array(diffs) > 0) > 0.5


def test_erp_material_follows_truth(cfg, world):
    """ERP 자재 소진(불출 − 재고)은 물리량이라 실제 (생산 + 폐기) × 단위 소요를 따라간다 — 수량 부풀리기의 대조군."""
    from f4ge_supplier_risk.generator import erp

    checked = 0
    for order, factory, product, lat, truth in world:
        rows = erp.build_erp_daily(cfg, order, factory, product, lat, truth)
        filed = [r for r in rows if not r["is_missing"]]
        if not filed:
            continue
        last = filed[-1]
        # 마지막 마감이 결측이면 filed[-1] 은 그 전날이다 — 그 날의 진척으로 비교한다
        c = counts_at(truth, order["order_qty"], progress_at(truth, last["day_index"] + 1))
        expect = (c["produced"] + c["scrap"]) * product["material_per_unit"]
        consumed = last["material_issued_qty"] - last["material_on_hand_qty"]
        assert abs(consumed - expect) <= 0.02 * expect + 1.0, order["order_id"]
        checked += 1
    assert checked > 0


def test_missing_reports_carry_no_numbers(cfg, world):
    """결측 회차는 값을 만들어내지 않는다. 결측 자체가 신호다."""
    for order, factory, product, lat, truth in world:
        rows, _ = reports.build_reports(cfg, order, factory, product, lat, truth)
        for r in rows:
            if r["is_missing"]:
                assert "produced_qty" not in r
                assert r["reported_at"] is None
