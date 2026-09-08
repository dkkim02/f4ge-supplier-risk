"""불일치 탐지 — 보고가 말하는 상태 vs 증거가 말하는 상태.

단순 품질 점수는 공장도 자기가 안다. 우리만 할 수 있는 것은 이것이다.

**핵심은 관측을 누가 통제하느냐로 피처를 가르는 것이다.**

    자기신고   공장이 직접 채점한 값 — 불량률·검사 결과·특이사항 유무
    증거       공장이 형태를 바꾸기 어려운 값 — 우리가 기록한 시각, 물리 대조,
               실측 치수, 운영 실적

관측 모형에서 왜 이게 되는지가 보인다.

    보고   r = internal x bias x noise      <- bias 가 여기에만 걸린다
    증거   h ~ internal                     <- bias 와 무관하다

증거로 보고를 예측하는 함수를 **공장 전체에 걸쳐** 적합하면 그 함수는 평균적인
`internal -> 보고` 관계를 배운다. 그러면 **잔차가 그 공장의 bias 편차**가 된다.

    불일치 = log(증거가 예상한 보고) - log(실제 보고)

양수면 "증거는 나쁘다는데 보고는 괜찮다고 한다" — 전화·방문 트리거다.

**정답값을 쓰지 않는다.** `true_gap` 은 평가에만 쓰고 모델에는 들어가지 않는다.
실데이터에서 영원히 알 수 없는 값이기 때문이다.

## ⚠ 실측 — 오더 단위로는 약하고, 공장 단위로는 강하다

7 seed 로 재보면 이렇게 갈린다.

| 주장 | 평균 | 범위 | 부호 일관 |
| --- | --- | --- | --- |
| **공장 정직도 Spearman 상관** | **+0.673** | 0.52 ~ 0.84 | **7/7** |
| 불일치 ↔ 실제 축소량(오더 단위) | +0.164 | −0.03 ~ 0.31 | 6/7 |
| 불일치 ↔ 예측 오차 | **−0.013** | −0.16 ~ 0.16 | **3/7** |

**한 오더의 불일치는 노이즈다. 같은 공장에서 반복되면 습관이다.**
그래서 오더 단위 값은 순위에만 쓰고, **판단은 공장 단위에서 내린다.**

> 처음에는 "불일치가 높은 오더는 예측 오차가 2.2배" 라고 썼다. **단일 시드 값이었다.**
> 7 seed 로 재니 Spearman 상관이 −0.013(3/7)로 무너진다. 극단 꼬리에서만 약하게 보이는 것을
> 단조 관계로 일반화했던 것이다. 철회한다.

또 하나 — 불일치는 **진짜 escape 와 음의 상관(−0.49)** 이다. 구조상 당연하다:
불일치가 높다는 것은 정의상 보고가 낮다는 뜻이고, 보고는 실제 불량률과 상관되므로
불일치 상위 오더는 오히려 실제로는 나쁘지 않은 쪽으로 기운다.
**"불일치가 높다 = 이 오더가 불량 난다" 로 읽으면 틀린다.** 그건 위험 예측의 일이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 공장이 직접 채점한 값. 편향이 걸리는 쪽이라 증거로 쓰면 안 된다.
SELF_COLS = (
    "l1_reported_defect_rate",
    "l1_scrap_ratio",
    "l1_rework_ratio",
    "l1_reject_ratio",
    "l1_issue_any",
)

# 공장이 형태를 바꾸기 어려운 값.
#   l0m_*             우리가 자동으로 남긴 시각. 늦게 보내면 늦은 게 남는다
#   l1_material_gap   보고 수량 vs 자재 소진량. 수량을 부풀리면 어긋난다
#   l1_fai_*          실측 치수. 게이지를 대야 나오는 숫자다
#   l1_overtime_mean  운영 실적. 품질 지표를 위해 꾸미는 값은 아니다
HARD_COLS = (
    "l0m_delay_mean",
    "l0m_delay_max",
    "l0m_missing",
    "l0m_blank",
    "l0m_promise_changes",
    "l0m_promise_change_late",
    "l1_material_gap",
    "l1_overtime_mean",
    "l1_fai_margin_min",
    "l1_fai_out_of_tol",
    "l1_fai_submitted",
)

# 난이도 보정. 큰 오더가 원래 불량이 많은 것을 "숨긴다"고 읽으면 안 된다.
CONTEXT_COLS = ("l0_log_qty", "l0_lead_slack", "l0_load_index")

_EPS = 5e-4


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """오더별 불일치도. 양수 = 증거보다 좋게 보고했다."""
    cols = list(HARD_COLS + CONTEXT_COLS)

    seen_tr = train["l1_rep_produced"].to_numpy(float) > 0
    y = np.log(train.loc[seen_tr, "l1_reported_defect_rate"].to_numpy(float) + _EPS)

    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=4.0))
    model.fit(train.loc[seen_tr, cols], y)

    expected = model.predict(test[cols])
    actual = np.log(test["l1_reported_defect_rate"].to_numpy(float) + _EPS)

    gap = expected - actual
    # 보고를 아예 안 낸 오더는 "괜찮다고 말한" 것이 아니다. 결측은 결측대로
    # L0′ 가 이미 신호로 쓰고 있으므로, 여기서는 중립(0)으로 둔다.
    gap[test["l1_rep_produced"].to_numpy(float) <= 0] = 0.0
    return gap


# ── 사유 분해 ────────────────────────────────────────────────────────────────
#
# 불일치 스칼라 하나로는 **"가서 뭘 봐야 하는가"** 를 말할 수 없다.
# 교차검증 짝마다 통계를 따로 두고, 어느 짝이 어긋났는지를 낸다.
# 다섯 짝 전부 **값이 클수록 의심스럽다** — 방향을 통일해 두면 threshold를 하나로 쓴다.
#
# `확인할 것` 이 이 분해의 목적이다. 현장 방문이 탐색이 아니라 검증이 되어야 한다.
REASONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "qty_inflation",
        "l1_material_gap",
        "보고 수량이 자재 소진량과 맞지 않는다",
        "재공품 실사 · 자재 입출고 대장",
    ),
    (
        "defect_hiding",
        "l1_ot_vs_reject",
        "불량은 적다는데 잔업이 많다",
        "재작업 대장 · 특별공정 기록",
    ),
    (
        "optimistic_due",
        "l1_pace_gap",
        "지금 속도로는 보고한 납기를 맞출 수 없다",
        "설비 가동 계획 · 잔여 공정 일정",
    ),
    (
        "silent_trouble",
        "l1_silent_delay",
        "특이사항은 없다는데 보고가 계속 늦는다",
        "담당자 면담 · 현재 공정 단계 확인",
    ),
    (
        "stale_evidence",
        "l1_photo_stale",
        "보낸 사진이 그 주에 찍힌 것이 아니다",
        "현장 사진 즉석 촬영 요청",
    ),
)

# 여섯 번째 사유는 짝이 없다. **대조할 것이 하나도 없다는 것 자체가 사유다** —
# T2 를 안 낸 공장은 우리가 검증할 수단이 없다는 뜻이고, 그건 "괜찮다" 가 아니다.
# 이걸 넣지 않으면 사유 없는 현장 방문이 3분의 1 남는다.
NO_EVIDENCE = (
    "no_evidence",
    "대조할 증거가 하나도 없다 — 자재·검사·사진 기록을 받지 못했다",
    "T2 항목 제출 요구 · 자재 입출고와 검사 기록 확보",
)

# 학습 구간 분포의 이 분위를 넘으면 그 짝이 어긋난 것으로 본다.
_REASON_Q = 0.85


def reasons(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """짝별 발동 여부와 대표 사유.

    threshold는 **학습 구간 분포에서만** 잡는다. 결측인 짝은 발동하지 않는다 —
    T2 를 안 낸 오더는 "괜찮다" 가 아니라 "확인할 수 없다" 이고,
    그건 이미 결측 자체로 L0′ 가 신호로 쓰고 있다.
    """
    out = {"reason_primary": [], "reason_codes": [], "reason_score": []}
    cuts, ranks = {}, {}
    for code, col, _, _ in REASONS:
        v = train[col].to_numpy(float)
        v = v[~np.isnan(v)]
        cuts[code] = float(np.quantile(v, _REASON_Q)) if len(v) else np.inf
        # 순위로 환산해 짝끼리 비교 가능하게 만든다 (단위가 전부 다르다)
        ranks[code] = np.sort(v) if len(v) else np.array([0.0])

    for _, row in test.iterrows():
        fired, best, best_pct = [], None, 0.0
        checkable = 0
        for code, col, _, _ in REASONS:
            x = row[col]
            if x is None or (isinstance(x, float) and np.isnan(x)):
                continue
            checkable += 1
            pct = float(np.searchsorted(ranks[code], x) / max(len(ranks[code]), 1))
            if x >= cuts[code]:
                fired.append(code)
                if pct > best_pct:
                    best, best_pct = code, pct
        # 물리적 증거 쪽 짝을 하나도 확인할 수 없으면 그것이 사유다.
        # (③④ 는 T2 없이도 계산되므로 "증거 없음" 판정에서 제외한다)
        if best is None and checkable <= 2:
            best = NO_EVIDENCE[0]
            fired = fired + [NO_EVIDENCE[0]]
        out["reason_primary"].append(best)
        out["reason_codes"].append(fired)
        out["reason_score"].append(round(best_pct, 3))
    return pd.DataFrame(out, index=test.index)


def factory_trust(
    scored: pd.DataFrame, min_orders: int = 3
) -> pd.DataFrame:
    """공장별 보고 신뢰도 — 불일치가 반복되는가, 한 번뿐인가.

    한 오더의 불일치는 사고일 수 있다. 같은 공장에서 계속 나오면 습관이다.
    보고가 없는 오더는 신뢰도를 말해 주지 않으므로 뺀다 — MES 가 없는 공장은 표에 안 나온다.
    "모른다" 를 "믿을 만하다" 나 "못 믿는다" 로 바꿔 읽으면 안 된다.
    """
    if "reported_missing" in scored:
        scored = scored[~scored["reported_missing"].astype(bool)]
    g = scored.groupby("factory_id")
    out = pd.DataFrame(
        {
            "orders": g.size(),
            "discrepancy_mean": g["discrepancy"].mean(),
            "discrepancy_p90": g["discrepancy"].quantile(0.9),
            "predicted_ppm_median": g["predicted_escape"].median() * 1e6,
        }
    )
    out = out[out["orders"] >= min_orders]
    return out.sort_values("discrepancy_mean", ascending=False)
