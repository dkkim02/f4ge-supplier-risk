"""κ → 검출률 d 재파라미터화 (2026-09-11, [[모델_방향_결정]] §3 ①).

`d = 1/(1+κ)` 는 표기 변환이다 — 예측값은 그대로이고 공장별 상수의 표기만 바뀐다.
셋(two_stage · deep_kappa · hier_bayes) 다 같은 값을 `detection` 으로 낸다는 것을 확인한다.
"""

import numpy as np
import pytest

from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import LAYERS
from f4ge_supplier_risk.models import deep_kappa, hier_bayes, two_stage

FULL = LAYERS["L0+Cell+MES+ERP"]


def test_detection_rate_is_the_algebra():
    """κ = (1−d)/d 의 역함수. 단조 감소, (0, 1) 안."""
    d = np.array([0.7, 0.95, 0.999])
    kappa = (1 - d) / d
    assert np.allclose(two_stage.detection_rate(kappa), d)
    assert two_stage.detection_rate(0.0) == 1.0
    assert two_stage.detection_rate(1.0) == 0.5


def test_two_stage_explain_detection_matches_kappa(table):
    tr, te = split_by_time(table)
    ex = two_stage.explain(tr, te, FULL)
    assert set(ex["detection"]) == set(ex["kappa"])
    for fid, k in ex["kappa"].items():
        assert ex["detection"][fid] == pytest.approx(1.0 / (1.0 + k))
        assert 0.0 < ex["detection"][fid] < 1.0
    assert ex["detection_pooled"] == pytest.approx(1.0 / (1.0 + ex["kappa_pooled"]))


@pytest.mark.skipif(not deep_kappa.available(), reason="torch 없음")
def test_deep_kappa_explain_reproduces_prediction(table):
    """공장별 κ 를 따로 낸 것이 오더 예측과 같은 g(·) 에서 왔는지 — `예측 = ih × κ_f` 로 되돌려 확인."""
    tr, te = split_by_time(table)
    pred = deep_kappa.fit_predict(tr, te, (), seed=0, mode="hier", epochs=200)
    ex = deep_kappa.explain(tr, te, seed=0, mode="hier", epochs=200)
    ih = two_stage._stage_a(tr, te, deep_kappa._cols_for_stage_a())
    kap = np.array([ex["kappa"][f] for f in te["factory_id"]])
    assert np.allclose(pred, np.clip(ih * kap, 1e-7, 1 - 1e-6), rtol=1e-4)
    for f, d in ex["detection"].items():
        assert d == pytest.approx(1.0 / (1.0 + ex["kappa"][f]))


@pytest.mark.skipif(not hier_bayes.available(), reason="numpyro 없음")
def test_hier_bayes_posterior_detection_interval(table):
    """d 구간은 κ 구간을 뒤집어 옮긴 것 — q05(d) = d(q95(κ)), 평균은 [q05, q95] 안."""
    tr, te = split_by_time(table)
    post = hier_bayes.posterior(tr, te, mode="pool", seed=0, warmup=200, samples=200)
    for f in post["kappa_mean"]:
        lo, hi, m = post["detection_q05"][f], post["detection_q95"][f], post["detection_mean"][f]
        assert 0.0 < lo <= m <= hi < 1.0
        assert lo == pytest.approx(1.0 / (1.0 + post["kappa_q95"][f]), rel=1e-6)
        assert hi == pytest.approx(1.0 / (1.0 + post["kappa_q05"][f]), rel=1e-6)


def test_contract_detection_comes_from_kappa(table):
    """계약 `factory_detection_rate` 의 계산처가 분해식이 아니라 2단 모델 κ 다."""
    from f4ge_supplier_risk.models import discrepancy
    from f4ge_supplier_risk.prediction.score import build_scores

    tr, te = split_by_time(table)
    pred_tr, pred = two_stage.fit_predict(tr, tr, FULL), two_stage.fit_predict(tr, te, FULL)
    scored = build_scores(tr, pred_tr, discrepancy.fit_predict(tr, tr), te, pred, discrepancy.fit_predict(tr, te))
    ex = two_stage.explain(tr, te, FULL)
    want = te["factory_id"].map(ex["detection"]).fillna(ex["detection_pooled"]).to_numpy(float)
    assert np.allclose(scored["factory_detection_rate"].to_numpy(float), np.round(want, 4))


def test_two_stage_detection_interval_brackets_mean(table):
    """conjugate Gamma 사후분포에서 낸 d 90% 구간 — q05 ≤ d ≤ q95.

    (폭이 사건 수만으로 정해지지는 않는다 — d 의 절대 폭은 κ 수준에도 비례한다. 폭은 단언하지 않는다.)
    """
    tr, te = split_by_time(table)
    ex = two_stage.explain(tr, te, FULL)
    for f, d in ex["detection"].items():
        lo, hi = ex["detection_q05"][f], ex["detection_q95"][f]
        assert 0.0 < lo <= d <= hi < 1.0
