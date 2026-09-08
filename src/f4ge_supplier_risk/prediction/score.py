"""출력 — 점수 하나가 아니라 세 값과 하나의 권고.

"위험도 0.72" 로는 아무도 행동하지 않는다. 관리자가 알아야 하는 것은 셋이다.

    예측 품질    이 오더가 어떻게 끝날 것 같은가        (2단 모델)
    보고 품질    공장은 뭐라고 하고 있는가              (그대로)
    불일치도     둘이 얼마나 어긋나는가                 (불일치 모델)

**두 축은 서로 다른 질문에 답하고, 서로 다른 단위에서 판단된다.**

    예측 위험     오더 단위   이 오더가 불량으로 끝날 것 같다      -> 검사를 조인다
    불일치        **공장 단위**  이 공장의 보고를 믿을 수 없다     -> 가서 본다

둘째 줄의 단위가 중요하다. 7 seed 실측에서 **오더 단위 불일치는 노이즈**이고
(예측 오차와의 Spearman 상관 −0.013, 부호 일관 3/7), **공장 단위 누적은 강하다**
(정직도 Spearman 상관 +0.673, 7/7). 한 오더의 어긋남은 사고일 수 있지만
같은 공장에서 반복되면 습관이다.

그래서 현장 방문은 **공장 신뢰도를 게이트로 걸고, 오더 불일치를 순위로 쓴다.**
오더 하나의 불일치만으로 방문을 보내는 것은 근거가 없다.

threshold 는 **학습 구간 분포에서만** 잡는다. 테스트 분포를 보고 자르면 현실에서
쓸 수 없는 threshold이 나온다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f4ge_supplier_risk.models.discrepancy import NO_EVIDENCE, REASONS

_REASON_TEXT = {code: (txt, chk) for code, _, txt, chk in REASONS}
_REASON_TEXT[NO_EVIDENCE[0]] = (NO_EVIDENCE[1], NO_EVIDENCE[2])

# 위험 등급 — 계약의 4-tier 를 승계한다.
# `high` 컷을 검사 강화 threshold(_RISK_TIGHTEN_Q)와 **같은 선에 맞춘다.**
# 어긋나 있으면 화면에 "위험 높음인데 권고 없음" 행이 생겨 자기모순으로 읽힌다.
_RISK_Q = (0.60, 0.90, 0.97)
_RISK_NAMES = ("low", "medium", "high", "critical")

# 개입 threshold.
#   방문은 **신뢰도가 낮은 공장**의 오더 중 불일치가 큰 것에만 건다.
#   오더 단위 불일치만 보고 보내면 노이즈를 쫓게 된다(실측 부호 일관 3/7).
_DISC_CALL_Q = 0.75
_DISC_VISIT_Q = 0.80
_RISK_TIGHTEN_Q = 0.90
# 공장 신뢰도 하위 몇 %를 "믿을 수 없다" 로 볼 것인가
_TRUST_GATE_Q = 0.75


def _levels(values: np.ndarray, cuts: np.ndarray) -> list[str]:
    return [_RISK_NAMES[i] for i in np.searchsorted(cuts, values, side="right")]


def build_scores(
    train: pd.DataFrame,
    train_pred: np.ndarray,
    train_disc: np.ndarray,
    test: pd.DataFrame,
    pred: np.ndarray,
    disc: np.ndarray,
    reasons: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`supplier-risk-score.v1` 모양의 행들.

    공장 신뢰도는 **학습 구간에서만** 계산한다 — 테스트 오더를 채점하는 시점에
    그 오더의 결과는 아직 없다.
    """
    risk_cuts = np.quantile(train_pred, _RISK_Q)
    disc_call = float(np.quantile(train_disc, _DISC_CALL_Q))
    disc_visit = float(np.quantile(train_disc, _DISC_VISIT_Q))
    risk_tighten = float(np.quantile(train_pred, _RISK_TIGHTEN_Q))

    by_factory = pd.Series(train_disc, index=train["factory_id"].to_numpy()).groupby(level=0).mean()
    trust_cut = float(by_factory.quantile(_TRUST_GATE_Q))
    low_trust = set(by_factory[by_factory >= trust_cut].index)

    # None 을 numpy 배열에 담으면 NaN 으로 바뀌고, NaN 은 truthy 라서
    # `x or None` 을 통과해 **JSON 에 NaN 이 그대로 나간다.** object dtype 로 둔다.
    prim = list(reasons["reason_primary"] if reasons is not None else [None] * len(test))
    prim = [None if p is None or (isinstance(p, float) and np.isnan(p)) else str(p) for p in prim]
    codes = (
        reasons["reason_codes"].tolist() if reasons is not None else [[] for _ in range(len(test))]
    )

    action, reason, check, trust_flag = [], [], [], []
    for p, d, fid, pc in zip(pred, disc, test["factory_id"].to_numpy(), prim, strict=True):
        untrusted = fid in low_trust
        trust_flag.append(untrusted)
        # 사유가 있으면 그것을 말한다. **"어긋났다" 보다 "무엇이 어긋났다" 가 행동을 만든다.**
        txt, chk = _REASON_TEXT.get(pc, (None, None))
        if untrusted and d >= disc_visit:
            action.append("site_visit")
            reason.append(txt or "보고 신뢰도가 낮은 공장이고, 이 오더도 증거와 어긋난다")
            check.append(chk or "현장 공정 · 검사 기록 전반")
        elif p >= risk_tighten:
            action.append("tighten_inspection")
            reason.append("예측 위험이 상위 10%")
            check.append("입고검사 강화 · 제3자 DUPRO 발주")
        elif d >= disc_call:
            action.append("call")
            reason.append(txt or "증거가 가리키는 것보다 보고가 좋다")
            check.append(chk or "보고 내용 재확인")
        else:
            action.append("none")
            reason.append("")
            check.append("")

    return pd.DataFrame(
        {
            "order_id": test["order_id"].to_numpy(),
            "factory_id": test["factory_id"].to_numpy(),
            "scored_at": test["ordered_at"].to_numpy(),
            "predicted_escape": pred,
            "predicted_ppm": np.round(pred * 1e6, 1),
            "risk_level": _levels(pred, risk_cuts),
            "reported_defect_rate": test["l1_reported_defect_rate"].to_numpy(),
            "reported_missing": test["l1_rep_produced"].to_numpy() <= 0,
            "discrepancy": np.round(disc, 4),
            "discrepancy_flag": disc >= disc_call,
            "factory_trust_low": trust_flag,
            "recommended_action": action,
            "action_reason": reason,
            "action_check": check,
            "reason_primary": pd.Series(prim, index=test.index, dtype=object),
            "reason_codes": codes,
        }
    )


def to_contract(row: pd.Series) -> dict[str, Any]:
    """`supplier-risk-score.v1` JSON 한 줄."""
    return {
        "schema_version": "supplier_risk.score.v1",
        "order_id": row["order_id"],
        "factory_id": row["factory_id"],
        "scored_at": row["scored_at"],
        "predicted_escape_ppm": float(row["predicted_ppm"]),
        "risk_level": row["risk_level"],
        # 결측과 무결함은 다르다. 보고를 안 낸 오더는 0 이 아니라 null 이다.
        "reported_defect_rate": (
            None if row["reported_missing"] else float(row["reported_defect_rate"])
        ),
        "discrepancy": float(row["discrepancy"]),
        "discrepancy_flag": bool(row["discrepancy_flag"]),
        "factory_trust_low": bool(row["factory_trust_low"]),
        "recommended_action": row["recommended_action"],
        "action_reason": row["action_reason"] or None,
        "action_check": row["action_check"] or None,
        "reason_primary": row["reason_primary"] if isinstance(row["reason_primary"], str) else None,
        "reason_codes": list(row["reason_codes"]),
    }
