"""출력 — 점수 하나가 아니라 세 값과 하나의 권고.

"위험도 0.72" 로는 아무도 행동하지 않는다. 관리자가 알아야 하는 것은 셋이다.

    예측 품질    이 오더가 어떻게 끝날 것 같은가        (2단 모델)
    보고 품질    공장은 뭐라고 하고 있는가              (그대로)
    불일치도     둘이 얼마나 어긋나는가                 (불일치 모델)

**두 축은 서로 다른 질문에 답하고, 서로 다른 단위에서 판단된다.**

    예측 위험     오더 단위   이 오더가 불량으로 끝날 것 같다
    불일치        **공장 단위**  이 공장의 보고를 믿을 수 없다

두 축 중 어느 쪽이든 임계를 넘으면 권고는 하나 — `review_needed`(검토 필요) — 다.
무엇을 할지는 화면이 아니라 담당자가 정한다. 대신 **왜 걸렸는지**(`action_reason`)와
**무엇을 대조할지**(`action_check`)를 같이 낸다. (09-08: 세 갈래 권고를 하나로 합쳤다.)

둘째 줄의 단위가 중요하다. 7 seed 실측에서 **오더 단위 불일치는 노이즈**이고
(예측 오차와의 Spearman 상관 −0.013, 부호 일관 3/7), **공장 단위 누적은 강하다**
(정직도 Spearman 상관 +0.673, 7/7). 한 오더의 어긋남은 사고일 수 있지만
같은 공장에서 반복되면 습관이다.

그래서 불일치 쪽 검토는 **공장 신뢰도를 게이트로 걸고, 오더 불일치를 순위로 쓴다.**
오더 하나의 불일치만으로 검토를 거는 것은 근거가 없다.

threshold 는 **학습 구간 분포에서만** 잡는다. 테스트 분포를 보고 자르면 현실에서
쓸 수 없는 threshold이 나온다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f4ge_supplier_risk.models import factory_params as fp
from f4ge_supplier_risk.models.discrepancy import NO_EVIDENCE, REASONS

_REASON_TEXT = {code: (txt, chk) for code, _, txt, chk in REASONS}
_REASON_TEXT[NO_EVIDENCE[0]] = (NO_EVIDENCE[1], NO_EVIDENCE[2])

# 위험 등급 — 계약의 4-tier 를 승계한다.
# `high` 컷을 예측 위험 쪽 검토 threshold(_RISK_TIGHTEN_Q)와 **같은 선에 맞춘다.**
# 어긋나 있으면 화면에 "위험 높음인데 권고 없음" 행이 생겨 자기모순으로 읽힌다.
_RISK_Q = (0.60, 0.90, 0.97)
_RISK_NAMES = ("low", "medium", "high", "critical")

# 검토 threshold. 둘 중 하나라도 넘으면 `review_needed`.
#   ① 예측 위험 상위 10% (_RISK_TIGHTEN_Q — 위험 등급 high 컷과 같은 선)  ← 2026-09-10 부터 이것 하나만 산다
# 오더 단독 불일치 축(_DISC_CALL_Q)은 **끈다(None)** — 09-08 저녁 sweep(8 seed, scripts/review_threshold_sweep.py):
#   네 소스가 12곳 전부에서 오자 이 축이 전 오더에 걸려 100건 중 35건이 검토 필요가 됐고 정밀도는 기저 수준이었다.
#   끄고 ②를 0.90 으로 올리면 14.5건 · 실제 위험 정밀도 27% → 57% · 불일치 축의 편향 공장 정밀도 52% → 57%(기저 50%).
#   한 오더의 불일치는 노이즈, 공장 단위 누적이 신호라는 실측(불일치탐지.md §2)과 같은 결론이다.
#
# 2026-09-10 — **불일치+저신뢰 축(_DISC_VISIT_Q)도 끈다.** 남는 축은 예측 위험 하나다.
#   8 seed 실측(편향 통제 세계): 이 축이 넣는 6.2건의 실제 위험 적중이 **2.5%** 다. 기저 20.2% 의 8분의 1.
#   위험 축과 겹치지도 않는다 — 현행 16.4건 = 위험축 10.2 + 이 축 6.2 로 정확히 갈린다. 순수 추가 오염이다.
#   끄면 16.4건 → 10.2건, 정밀도 53.7% → 86.0%, 실제로 잡는 위험 건수는 8.8 로 그대로다.
#   통제 전에도 이 축은 편향 공장 적중이 기저 이하였다(47.6% vs 49.9%, corr -0.85 에서 43.6%).
#   되살리려면 0.90 을 넣는다. 관련 측정은 scripts/legacy/ 로 옮겼다.
_DISC_CALL_Q: float | None = None
_DISC_VISIT_Q: float | None = None
_RISK_TIGHTEN_Q = 0.90
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
    params: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`supplier-risk-score.v1` 모양의 행들.

    공장 신뢰도는 **학습 구간에서만** 계산한다 — 테스트 오더를 채점하는 시점에
    그 오더의 결과는 아직 없다.
    `params` = 공장 파라미터 분해(제조 품질 p · 검수 품질 d · 보고 정직도 b, models/factory_params.py).
    없으면 학습 구간에서 계산한다.
    """
    if params is None:
        params = fp.estimate(train)
    internal = fp.order_internal_rate(test, params)
    pooled = params.attrs.get("pooled", {})
    fac = test["factory_id"]
    d_f = fac.map(params["d"]).fillna(pooled.get("d", np.nan)).to_numpy(float)
    p_f = fac.map(params["p"]).fillna(pooled.get("p", np.nan)).to_numpy(float)
    obs_f = fac.map(params["n_labels"]).fillna(0).to_numpy(int)
    risk_cuts = np.quantile(train_pred, _RISK_Q)
    risk_tighten = float(np.quantile(train_pred, _RISK_TIGHTEN_Q))

    # 불일치는 **증거가 있는 오더에서만 정의된다.** MES 가 없는 공장(유형 b·c)의 오더는
    # 불일치가 0 으로 들어오는데, 그 0 을 분위수와 신뢰도 게이트에 섞으면 —
    # 그런 오더가 75% 라 컷 자체가 0 이 되고, 증거 없는 공장 전부가 "못 믿을 공장" 이 된다
    # (09-08 실측: 270건 중 219건 검토 필요). 모르는 것과 못 믿는 것은 다르다.
    ev_tr = train["l1_rep_produced"].to_numpy(float) > 0
    ev_te = test["l1_rep_produced"].to_numpy(float) > 0
    if ev_tr.any():
        # _DISC_CALL_Q 가 None 이면 오더 단독 불일치 축을 끈다(한 오더의 불일치는 노이즈 — 불일치탐지.md §2)
        disc_visit = (
            float(np.quantile(train_disc[ev_tr], _DISC_VISIT_Q))
            if _DISC_VISIT_Q is not None
            else float("inf")
        )
        disc_call = (
            float(np.quantile(train_disc[ev_tr], _DISC_CALL_Q)) if _DISC_CALL_Q is not None else disc_visit
        )
        by_factory = (
            pd.Series(train_disc[ev_tr], index=train["factory_id"].to_numpy()[ev_tr])
            .groupby(level=0)
            .mean()
        )
        trust_cut = float(by_factory.quantile(_TRUST_GATE_Q))
        low_trust = set(by_factory[by_factory >= trust_cut].index)
    else:
        disc_call = disc_visit = float("inf")
        low_trust = set()

    # None 을 numpy 배열에 담으면 NaN 으로 바뀌고, NaN 은 truthy 라서
    # `x or None` 을 통과해 **JSON 에 NaN 이 그대로 나간다.** object dtype 로 둔다.
    prim = list(reasons["reason_primary"] if reasons is not None else [None] * len(test))
    prim = [None if p is None or (isinstance(p, float) and np.isnan(p)) else str(p) for p in prim]
    codes = (
        reasons["reason_codes"].tolist() if reasons is not None else [[] for _ in range(len(test))]
    )

    # 권고 코드는 하나지만 **어느 축이 걸렸는지는 남겨야 한다.** 그러지 않으면
    # "예측 위험이 실제 불량을 잡는 비율"과 "불일치가 편향 공장을 잡는 비율"을 따로 잴 수 없다 —
    # `action_reason` 은 사유 6종 텍스트와 섞여 있어 축을 식별하지 못한다(09-08).
    action, reason, check, trust_flag, trigger = [], [], [], [], []
    for p, d, fid, pc, ev in zip(
        pred, disc, test["factory_id"].to_numpy(), prim, ev_te, strict=True
    ):
        untrusted = fid in low_trust
        trust_flag.append(untrusted)
        # 사유가 있으면 그것을 말한다. **"어긋났다" 보다 "무엇이 어긋났다" 가 행동을 만든다.**
        txt, chk = _REASON_TEXT.get(pc, (None, None))
        # 권고 코드는 하나(review_needed)다. 사유·확인할 것은 어느 축이 걸렸는지에 따라 다르다.
        if ev and untrusted and d >= disc_visit:
            action.append("review_needed")
            trigger.append("discrepancy_low_trust")
            reason.append(txt or "보고 신뢰도가 낮은 공장이고, 이 오더도 증거와 어긋난다")
            check.append(chk or "공정 · 검사 기록 전반")
        elif p >= risk_tighten:
            action.append("review_needed")
            trigger.append("risk")
            reason.append("예측 위험이 상위 10%")
            check.append("입고검사 기록 · 공정 이력")
        elif ev and _DISC_CALL_Q is not None and d >= disc_call:
            action.append("review_needed")
            trigger.append("discrepancy")
            reason.append(txt or "증거가 가리키는 것보다 보고가 좋다")
            check.append(chk or "보고 내용 재확인")
        else:
            action.append("none")
            trigger.append(None)
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
            # ── 목표 둘 (09-08 저녁): 제조 품질 · 검수 품질 ──
            # 보고 정직도 b 는 2026-09-10 산출에서 뺐다 — 편향 통제 세계에서 항상 1.0 이고,
            # 통제 전에도 「중앙 공장 = 1.0」 규약 위의 상대값이라 절대 수준이 아니었다.
            "predicted_internal_rate": np.round(internal, 5),
            "factory_internal_rate": np.round(p_f, 5),
            "factory_detection_rate": np.round(d_f, 4),
            "factory_param_obs": obs_f,
            "reported_defect_rate": test["l1_reported_defect_rate"].to_numpy(),
            "reported_missing": test["l1_rep_produced"].to_numpy() <= 0,
            "discrepancy": np.round(disc, 4),
            "discrepancy_flag": (disc >= disc_call) & ev_te,
            "factory_trust_low": trust_flag,
            "recommended_action": action,
            "action_trigger": pd.Series(trigger, index=test.index, dtype=object),
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
        "predicted_internal_defect_rate": float(row["predicted_internal_rate"]),
        "factory_internal_defect_rate": float(row["factory_internal_rate"]),
        "factory_detection_rate": float(row["factory_detection_rate"]),
        # 결측과 무결함은 다르다. 보고를 안 낸 오더는 0 이 아니라 null 이다.
        "reported_defect_rate": (
            None if row["reported_missing"] else float(row["reported_defect_rate"])
        ),
        "discrepancy": float(row["discrepancy"]),
        "discrepancy_flag": bool(row["discrepancy_flag"]),
        "factory_trust_low": bool(row["factory_trust_low"]),
        "recommended_action": row["recommended_action"],
        "action_trigger": row["action_trigger"] if isinstance(row["action_trigger"], str) else None,
        "action_reason": row["action_reason"] or None,
        "action_check": row["action_check"] or None,
        "reason_primary": row["reason_primary"] if isinstance(row["reason_primary"], str) else None,
        "reason_codes": list(row["reason_codes"]),
    }
