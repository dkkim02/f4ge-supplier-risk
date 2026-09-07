"""불일치 탐지 — 정답값을 쓰지 않는가, 그리고 무엇을 잡는가."""

import numpy as np

from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import LAYERS
from f4ge_supplier_risk.models import discrepancy, two_stage
from f4ge_supplier_risk.prediction.score import build_scores, to_contract


def test_uses_no_self_reported_severity():
    """자기신고 항목이 증거 쪽에 섞이면 지표가 무의미해진다."""
    assert not set(discrepancy.SELF_COLS) & set(discrepancy.HARD_COLS)
    assert "l1_reported_defect_rate" not in discrepancy.HARD_COLS + discrepancy.CONTEXT_COLS


def test_ranks_factories_by_honesty(table, cfg):
    """편향이 심한 공장이 불일치 상위로 올라와야 한다."""
    from scipy.stats import spearmanr

    from f4ge_supplier_risk.generator import masters

    tr, te = split_by_time(table)
    pred_tr = two_stage.fit_predict(tr, tr, LAYERS["L0+L0′+L1"])
    d_tr = discrepancy.fit_predict(tr, tr)
    d = discrepancy.fit_predict(tr, te)
    scored = build_scores(
        tr, pred_tr, d_tr, te, two_stage.fit_predict(tr, te, LAYERS["L0+L0′+L1"]), d
    )

    products = masters.build_products(cfg)
    bias = {f["factory_id"]: f["report_bias"] for f in masters.build_factories(cfg, products)}
    trust = discrepancy.factory_trust(scored)
    r = spearmanr(trust["discrepancy_mean"], [-bias[i] for i in trust.index]).statistic
    assert r > 0.2, f"정직도 순위를 못 잡는다 (r={r:.3f})"


def test_order_level_signal_is_weak_by_design(table):
    """오더 단위 불일치로 "이 오더가 나쁘다" 를 말하면 안 된다.

    불일치는 정의상 보고가 낮은 쪽이고 보고는 실제 불량률과 상관되므로,
    불일치 상위 오더는 오히려 실제로는 나쁘지 않은 쪽으로 기운다.
    이 성질이 뒤집히면 지표의 해석 전체를 다시 써야 한다.
    """
    from scipy.stats import spearmanr

    tr, te = split_by_time(table)
    d = discrepancy.fit_predict(tr, te)
    r = spearmanr(d, te["true_escape_rate"].to_numpy()).statistic
    assert r < 0.1, f"불일치가 실제 위험과 양의 상관이면 해석이 달라진다 (r={r:.3f})"


def test_missing_report_is_neutral_not_zero(table):
    """보고를 안 낸 것은 "괜찮다고 말한" 것이 아니다."""
    tr, te = split_by_time(table)
    d = discrepancy.fit_predict(tr, te)
    missing = te["l1_rep_produced"].to_numpy() <= 0
    if missing.any():
        assert np.allclose(d[missing], 0.0)


def test_contract_row_keeps_null_for_missing_report(table):
    tr, te = split_by_time(table)
    pred_tr = two_stage.fit_predict(tr, tr, LAYERS["L0+L0′+L1"])
    scored = build_scores(
        tr,
        pred_tr,
        discrepancy.fit_predict(tr, tr),
        te,
        two_stage.fit_predict(tr, te, LAYERS["L0+L0′+L1"]),
        discrepancy.fit_predict(tr, te),
    )
    rows = [to_contract(r) for _, r in scored.iterrows()]
    assert {r["recommended_action"] for r in rows} <= {
        "none",
        "call",
        "tighten_inspection",
        "site_visit",
    }
    for r in rows:
        assert r["predicted_escape_ppm"] >= 0
        assert r["risk_level"] in ("low", "medium", "high", "critical")
