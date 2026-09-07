"""보고는 노이즈가 아니라 **편향**이어야 한다."""

import numpy as np

from f4ge_supplier_risk.generator import reports
from f4ge_supplier_risk.generator.production import counts_at


def test_reported_defects_are_understated(cfg, world):
    """공장이 보고한 불량은 실제보다 적다. 평균이 어긋나야 불일치 탐지가 값을 한다."""
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


def test_material_consumption_is_not_biased(cfg, world):
    """자재 소진은 물리량이라 실제값을 따라간다 — 수량 부풀리기의 대조군."""
    order, factory, product, lat, truth = world[0]
    rows, _ = reports.build_reports(cfg, order, factory, product, lat, truth)
    for r in rows:
        if r["is_missing"] or "material_consumed_qty" not in r:
            continue
        assert r["material_consumed_qty"] > 0


def test_missing_reports_carry_no_numbers(cfg, world):
    """결측 회차는 값을 만들어내지 않는다. 결측 자체가 신호다."""
    for order, factory, product, lat, truth in world:
        rows, _ = reports.build_reports(cfg, order, factory, product, lat, truth)
        for r in rows:
            if r["is_missing"]:
                assert "produced_qty" not in r
                assert r["reported_at"] is None
