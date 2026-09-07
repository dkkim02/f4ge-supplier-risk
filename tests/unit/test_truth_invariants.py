"""진실 경과가 지켜야 할 불변식."""

import numpy as np

from f4ge_supplier_risk.generator.production import counts_at, progress_at


def test_counts_are_monotone(world):
    """누적값은 절대 줄어들지 않는다.

    진척률 시점마다 새로 뽑으면 이게 깨지고, 비례 추정으로 메우면
    25% 시점과 100% 시점이 같은 값을 읽어 답이 새어 나간다.
    """
    for order, _f, _p, _lat, truth in world:
        prev = {"produced": -1, "scrap": -1, "rework": -1, "defects": -1, "escaped": -1}
        for step in np.linspace(0.0, 1.0, 21):
            c = counts_at(truth, order["order_qty"], float(step))
            for k, v in c.items():
                assert v >= prev[k], f"{order['order_id']} {k} 감소"
                prev[k] = v


def test_final_counts_close_the_books(world):
    """진척률 1.0 에서 실제 생산 기록과 정확히 맞아야 한다."""
    for order, _f, _p, _lat, truth in world:
        c = counts_at(truth, order["order_qty"], 1.0)
        assert c["produced"] == order["order_qty"]
        assert c["defects"] == truth["n_defect"]
        assert c["scrap"] + c["rework"] + c["escaped"] == truth["n_defect"]


def test_setup_produces_nothing(world):
    for _o, _f, _p, _lat, truth in world:
        assert progress_at(truth, truth["actual_days"] * 0.05) == 0.0
