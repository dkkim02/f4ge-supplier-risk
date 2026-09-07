"""2단 모델 — 구조와 시간 규율."""

import numpy as np

from f4ge_supplier_risk.features.build import LAYERS
from f4ge_supplier_risk.models import baseline, two_stage


def test_beats_one_stage_on_rank(table):
    """보고를 **타깃 쪽**에서 쓰면 피처로만 쓸 때보다 낫다.

    이게 2단 구조의 존재 이유다 — 공장 자체검사는 전수에 가까워서
    우리 표본검사 라벨보다 정보량이 두 자릿수 배 크다.
    """
    from f4ge_supplier_risk.evaluation.metrics import evaluate, split_by_time

    tr, te = split_by_time(table)
    one = evaluate(te, baseline.fit_predict(tr, te, LAYERS["L0+L0′+L1"]))["rank_corr_true"]
    two = evaluate(te, two_stage.fit_predict(tr, te, LAYERS["L0+L0′+L1"]))["rank_corr_true"]
    assert two > one


def test_prediction_is_a_probability(table):
    from f4ge_supplier_risk.evaluation.metrics import split_by_time

    tr, te = split_by_time(table)
    p = two_stage.fit_predict(tr, te, LAYERS["L0+L0′+L1"])
    assert np.all((p > 0) & (p < 1))
    assert len(p) == len(te)


def test_kappa_uses_only_arrived_labels(table, monkeypatch):
    """라벨이 아직 도착하지 않았으면 kappa 갱신에 쓰이면 안 된다.

    라벨 도착 시각을 전부 먼 미래로 밀면 테스트 구간의 온라인 갱신이 하나도
    일어나지 않아야 한다 — 결과가 달라지면 시간 규율이 새고 있다는 뜻이다.
    """
    from f4ge_supplier_risk.evaluation.metrics import split_by_time

    tr, te = split_by_time(table)
    late = te.copy()
    late["label_available_at"] = "2099-01-01T00:00:00Z"

    normal = two_stage.fit_predict(tr, te, LAYERS["L0+L0′+L1"])
    frozen = two_stage.fit_predict(tr, late, LAYERS["L0+L0′+L1"])
    assert not np.allclose(normal, frozen), "온라인 갱신이 아예 일어나지 않았다"
