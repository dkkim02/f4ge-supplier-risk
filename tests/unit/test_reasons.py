"""사유 분해 — 짝별 관측치가 **그 메커니즘**을 실제로 잡는가.

생성기가 남긴 `mech_*` 정답값이 이 검증의 유일한 근거다. 실데이터에서는
"이 공장이 수량을 부풀렸는지" 를 영원히 알 수 없다.

⚠ 이 테스트가 잡아낸 실제 버그: `silent_delay` 를 `issue_any` 로 재고 있었는데,
곤란한 공장은 여러 회차 중 한 번은 이상을 신고하므로 신호가 0으로 죽어
**상관 부호가 뒤집혀 있었다**(−0.55). 신고 비율로 바꿔 +0.22 가 됐다.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.models import discrepancy

# (짝 관측치, 메커니즘, 부호가 음수여야 하는가)
_PAIRS = (
    ("l1_material_gap", "mech_inflation_max", False),
    ("l1_ot_vs_reject", "mech_shrink_min", True),
    ("l1_pace_gap", "mech_promise_stale_days", False),
    ("l1_silent_delay", "mech_issue_suppressed", False),
    ("l1_photo_stale", "mech_photo_reused", False),
)


@pytest.mark.parametrize(("col", "mech", "negative"), _PAIRS)
def test_pair_tracks_its_mechanism(table, truth_map, col, mech, negative):
    v = table[col].to_numpy(float)
    t = np.array([truth_map[i][mech] for i in table["order_id"]], dtype=float)
    m = ~np.isnan(v)
    r = spearmanr(v[m], t[m]).statistic
    assert (r < -0.15) if negative else (r > 0.15), f"{col} ↔ {mech} r={r:+.3f}"


def test_no_evidence_covers_the_unverifiable(table):
    """짝을 하나도 확인할 수 없는 오더에는 `no_evidence` 가 붙어야 한다.

    이걸 안 넣으면 사유 없는 현장 방문이 3분의 1 남는다 — 그리고 "대조할 게
    없다" 는 것은 "괜찮다" 가 아니다.
    """
    tr, te = split_by_time(table)
    rs = discrepancy.reasons(tr, te)
    blind = te["l1_material_gap"].isna() & te["l1_photo_stale"].isna()
    if blind.any():
        got = rs.loc[blind.to_numpy(), "reason_primary"]
        assert (got == discrepancy.NO_EVIDENCE[0]).any()


def test_threshold_comes_from_train_only(table):
    """임계를 테스트 분포에서 잡으면 현실에서 쓸 수 없는 값이 된다."""
    tr, te = split_by_time(table)
    a = discrepancy.reasons(tr, te)["reason_primary"].tolist()
    b = discrepancy.reasons(tr, te.iloc[:40])["reason_primary"].tolist()
    assert a[:40] == b
