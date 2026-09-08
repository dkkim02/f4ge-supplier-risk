"""피처 테이블 — L0 / L0+L1.

**이 파일의 제약은 시간 순서다.**

공장 이력 피처는 "그 오더를 발주하던 시점에 우리가 실제로 알고 있던 것"만 써야 한다.
입고검사 라벨은 출하 뒤 한참 지나 도착하고(유럽이면 두 달), 그래서 오더 n번을
배분할 때 오더 n−1번의 결과는 아직 없는 경우가 흔하다. `label_available_at` 을
무시하면 현실에서 불가능한 정보로 학습하게 된다.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

# 이력이 없는 공장의 사전분포. 관측이 쌓이기 전까지 전체 평균 쪽으로 당긴다.
_PRIOR_DEFECTS = 0.5
_PRIOR_INSPECTED = 400.0


def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _history_features(orders: list[dict], outcomes: list[dict]) -> dict[str, dict[str, float]]:
    """발주 시점까지 **도착한** 라벨만으로 만든 공장 이력."""
    out_by_id = {o["order_id"]: o for o in outcomes}
    events = sorted(
        ((_dt(o["label_available_at"]), o["order_id"]) for o in outcomes),
        key=lambda x: x[0],
    )
    order_by_id = {o["order_id"]: o for o in orders}

    acc: dict[str, dict[str, float]] = {}
    feats: dict[str, dict[str, float]] = {}
    ptr = 0
    for order in sorted(orders, key=lambda o: _dt(o["ordered_at"])):
        now = _dt(order["ordered_at"])
        while ptr < len(events) and events[ptr][0] <= now:
            oid = events[ptr][1]
            fid = order_by_id[oid]["factory_id"]
            oc = out_by_id[oid]
            a = acc.setdefault(fid, {"n": 0.0, "k": 0.0, "orders": 0.0, "rejects": 0.0})
            a["n"] += oc["incoming_inspected_qty"]
            a["k"] += oc["incoming_reject_qty"]
            a["orders"] += 1
            a["rejects"] += oc["lot_result"] == "reject"
            ptr += 1

        a = acc.get(order["factory_id"], {"n": 0.0, "k": 0.0, "orders": 0.0, "rejects": 0.0})
        feats[order["order_id"]] = {
            "hist_orders": a["orders"],
            "hist_inspected": a["n"],
            # 평활 불량률 — 관측이 없으면 사전분포 값이 그대로 나온다
            "hist_escape_rate": (a["k"] + _PRIOR_DEFECTS) / (a["n"] + _PRIOR_INSPECTED),
            "hist_reject_rate": (a["rejects"] + 0.5) / (a["orders"] + 10.0),
        }
    return feats


def build(data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    orders = data["orders"]
    outcomes = {o["order_id"]: o for o in data["quality_outcomes"]}
    meta = {m["order_id"]: m for m in data["order_meta"]}
    fai = {f["order_id"]: f for f in data["fai_reports"]}
    truth = {t["order_id"]: t for t in data["ground_truth"]}

    reports_by_order: dict[str, list[dict]] = {}
    for r in data["factory_reports"]:
        reports_by_order.setdefault(r["order_id"], []).append(r)
    cell_by_order: dict[str, list[dict]] = {}
    for c in data["cell_daily"]:
        cell_by_order.setdefault(c["order_id"], []).append(c)

    hist = _history_features(orders, data["quality_outcomes"])
    products = {p["product_id"]: p for p in data["products"]}

    rows = []
    for o in orders:
        oc = outcomes[o["order_id"]]
        m = meta[o["order_id"]]
        reps = sorted(reports_by_order.get(o["order_id"], []), key=lambda r: r["seq"])
        filed = [r for r in reps if not r["is_missing"]]
        last = filed[-1] if filed else {}
        t2 = [r for r in filed if "inspected_qty" in r]

        # ── L1 — 공장이 준 것. 전부 편향이 걸려 있다 ──
        # MES/FactoryOS 가 없는 공장은 이 값이 **아예 오지 않는다.**
        # 0 으로 채우면 "불량 없음" 과 구분되지 않는다.
        rep_defects = last.get("scrap_qty", 0) + last.get("rework_qty", 0)
        rep_produced = last.get("produced_qty", 0) or 1
        rep_rate = (rep_defects / rep_produced) if filed else np.nan

        # 교차검증 쌍: 보고 수량 vs 자재 소진. 수량을 부풀리면 여기가 어긋난다
        # 마지막 회차만 보면 안 된다 — 부풀리기는 특정 회차에서 일어나고
        # 다음 회차에 실제 생산이 따라붙으면 흔적이 지워진다. 최댓값을 쓴다.
        mat_gap = np.nan
        if t2:
            per_unit = products[o["product_id"]]["material_per_unit"]
            gaps = [
                (r["produced_qty"] - r["material_consumed_qty"] / per_unit)
                / max(r["material_consumed_qty"] / per_unit, 1.0)
                for r in t2
                if r["material_consumed_qty"] > 0
            ]
            if gaps:
                mat_gap = max(gaps)

        # ── 교차검증 짝별 관측치 ──
        # 자기신고 항목과 물리적 증거를 짝지어 놓은 값들이다. 불일치 스칼라 하나로는
        # "왜 어긋났는지" 를 말할 수 없어서, 짝마다 통계를 따로 만든다.

        # ③ 납기 ↔ 진척 추세: 지금 속도로 언제 끝나는가 vs 공장이 말하는 납기
        pace_gap = np.nan
        if last and last.get("produced_qty"):
            elapsed = (_dt(last["reported_at"]) - _dt(o["ordered_at"])).total_seconds() / 86400
            rate = last["produced_qty"] / max(elapsed, 1e-6)
            if rate > 0:
                projected = o["order_qty"] / rate
                promised = (
                    _dt(last.get("promised_date_reported", o["promised_date"]))
                    - _dt(o["ordered_at"])
                ).total_seconds() / 86400
                pace_gap = (projected - promised) / max(promised, 1.0)

        # ④ "이상 없음" ↔ 보고 지연: 문제가 없다는데 연락은 느려진다
        #    `any` 로 재면 안 된다 — 곤란한 공장은 여러 회차 중 한 번은 신고하게 되고
        #    그러면 신호가 0으로 죽어 **상관 부호가 뒤집힌다**(실측 −0.55).
        #    회차 대비 **신고 비율**로 재야 "대체로 조용한데 늦다" 가 잡힌다.
        delay_mean = m["report_delay_days_mean"] or 0.0
        issue_rate = sum(1 for r in filed if r.get("issue_flag")) / len(filed) if filed else 0.0
        issue_any = float(issue_rate > 0)
        silent_delay = delay_mean * (1.0 - issue_rate)

        # ⑤ 사진 ↔ 보고 시각: 지난주 사진을 다시 보냈는가
        photo_stale = np.nan
        stale = [
            (_dt(r["reported_at"]) - _dt(r["photo_taken_at"])).total_seconds() / 3600
            for r in filed
            if "photo_taken_at" in r
        ]
        if stale:
            photo_stale = max(stale)

        # ② 자체 불량률 ↔ 잔업: 불량은 없다는데 잔업은 늘어난다
        ot_vs_reject = np.nan
        if t2:
            ot = float(np.mean([r["overtime_hours"] for r in t2]))
            ot_vs_reject = ot / (rep_rate * 100.0 + 0.3)

        # ── L2 — CellOS 설비 신호. 사람 손이 닿지 않아 편향이 없다 ──
        cells = sorted(cell_by_order.get(o["order_id"], []), key=lambda c: c["day_index"])
        if cells:
            up = np.array([c["uptime_ratio"] for c in cells], dtype=float)
            ct = np.array([c["cycle_time_median"] for c in cells], dtype=float)
            cell_uptime = float(up.mean())
            cell_uptime_min = float(up.min())
            cell_ct_cv = float(ct.std() / max(ct.mean(), 1e-9))
            cell_anom = float(sum(c["anomaly_event_count"] for c in cells))
            cell_anom_rate = cell_anom / len(cells)
            cell_tool = float(sum(c["tool_change_count"] for c in cells))
            # ★ 교차검증: 보고 수량 vs 설비 카운터. 물리적으로 어긋날 수 없는 대조다
            counter = float(cells[-1]["produced_by_counter"])
            counter_gap = (rep_produced - counter) / max(counter, 1.0) if last else np.nan
        else:
            cell_uptime = cell_uptime_min = cell_ct_cv = np.nan
            cell_anom = cell_anom_rate = cell_tool = counter_gap = np.nan

        f = fai.get(o["order_id"])
        rows.append(
            {
                "order_id": o["order_id"],
                "factory_id": o["factory_id"],
                "ordered_at": o["ordered_at"],
                # ── L0 ──
                "l0_lead_slack": o["lead_slack"],
                "l0_price_zscore": o["price_zscore"],
                "l0_log_qty": math.log(o["order_qty"]),
                "l0_quote_response_h": o["quote_response_h"],
                "l0_load_index": o["load_index"],
                "l0_market_eu": float(o["market"] == "eu"),
                **{f"l0_{k}": v for k, v in hist[o["order_id"]].items()},
                # ── L0′ — 우리가 자동으로 남기는 것. 조작 불가 ──
                "l0m_delay_mean": m["report_delay_days_mean"]
                if m["report_delay_days_mean"] is not None
                else 0.0,
                "l0m_delay_max": m["report_delay_days_max"]
                if m["report_delay_days_max"] is not None
                else 0.0,
                "l0m_missing": m["report_missing_count"],
                "l0m_blank": m["field_blank_count"],
                "l0m_promise_changes": m["promised_date_change_count"],
                "l0m_promise_change_late": m["promised_date_change_last_progress"],
                # ── L1 — 공장 자진 보고 ──
                # 원 카운트를 그대로 남긴다. 2단 모델이 비율이 아니라 (불량수, 생산수)
                # 쌍을 쓴다 — 생산 100개에서의 3%와 2,000개에서의 3%는 정보량이 다르다.
                "l1_rep_produced": float(rep_produced if filed else 0),
                "l1_rep_defects": float(rep_defects),
                "l1_reported_defect_rate": rep_rate,
                "l1_scrap_ratio": last.get("scrap_qty", 0) / rep_produced,
                "l1_rework_ratio": last.get("rework_qty", 0) / rep_produced,
                "l1_issue_any": issue_any,
                "l1_overtime_mean": float(np.mean([r["overtime_hours"] for r in t2]))
                if t2
                else np.nan,
                "l1_reject_ratio": (
                    float(np.mean([r["reject_qty"] / max(r["inspected_qty"], 1) for r in t2]))
                    if t2
                    else np.nan
                ),
                "l1_material_gap": mat_gap,
                # ── L2 ──
                "l2_uptime_mean": cell_uptime,
                "l2_uptime_min": cell_uptime_min,
                "l2_cycle_cv": cell_ct_cv,
                "l2_anomaly_rate": cell_anom_rate,
                "l2_tool_changes": cell_tool,
                "l2_counter_gap": counter_gap,
                "l1_pace_gap": pace_gap,
                "l1_silent_delay": silent_delay,
                "l1_photo_stale": photo_stale,
                "l1_ot_vs_reject": ot_vs_reject,
                "l1_fai_margin_min": f["margin_min"] if f else np.nan,
                "l1_fai_out_of_tol": f["out_of_tol_count"] if f else np.nan,
                "l1_fai_submitted": float(f is not None),
                # ── y ──
                "y_inspected": oc["incoming_inspected_qty"],
                "y_reject": oc["incoming_reject_qty"],
                "y_lot_reject": float(oc["lot_result"] == "reject"),
                "label_available_at": oc["label_available_at"],
                # ── 진실 (평가 전용, 학습 금지) ──
                "true_escape_rate": truth[o["order_id"]]["escape_rate_true"],
                "true_internal_rate": truth[o["order_id"]]["internal_defect_rate_true"],
                "true_gap": truth[o["order_id"]]["reported_vs_true_gap"],
            }
        )

    return pd.DataFrame(rows).sort_values("ordered_at").reset_index(drop=True)


L0_COLS = tuple(
    c
    for c in (
        "l0_lead_slack",
        "l0_price_zscore",
        "l0_log_qty",
        "l0_quote_response_h",
        "l0_load_index",
        "l0_market_eu",
        "l0_hist_orders",
        "l0_hist_inspected",
        "l0_hist_escape_rate",
        "l0_hist_reject_rate",
    )
)
L0M_COLS = (
    "l0m_delay_mean",
    "l0m_delay_max",
    "l0m_missing",
    "l0m_blank",
    "l0m_promise_changes",
    "l0m_promise_change_late",
)
# 교차검증 짝의 관측치. 자기신고 항목과 물리적 증거를 짝지어 놓은 값들이고,
# `models/discrepancy.py` 가 **어느 짝이 어긋났는지** 를 여기서 읽는다.
PAIR_COLS = (
    "l1_material_gap",  # ① 보고 수량 ↔ 자재 소진량
    "l1_ot_vs_reject",  # ② 자체 불량률 ↔ 잔업
    "l1_pace_gap",  # ③ 보고 납기 ↔ 진척 추세
    "l1_silent_delay",  # ④ "이상 없음" ↔ 보고 지연
    "l1_photo_stale",  # ⑤ 사진 시각 ↔ 보고 시각
)

L1_COLS = (
    "l1_rep_produced",
    "l1_rep_defects",
    "l1_reported_defect_rate",
    "l1_scrap_ratio",
    "l1_rework_ratio",
    "l1_issue_any",
    "l1_overtime_mean",
    "l1_reject_ratio",
    *PAIR_COLS,
    "l1_fai_margin_min",
    "l1_fai_out_of_tol",
    "l1_fai_submitted",
)

# L2 — CellOS 설비 신호. 편향이 없다.
L2_COLS = (
    "l2_uptime_mean",
    "l2_uptime_min",
    "l2_cycle_cv",
    "l2_anomaly_rate",
    "l2_tool_changes",
    "l2_counter_gap",
)

# FAI 는 유형과 무관하게 온다(고객 요구사항). MES 계층과 분리해 둔다 —
# 설치 가치를 재려면 "FAI 만 있는 상태" 를 따로 볼 수 있어야 한다.
FAI_COLS = ("l1_fai_margin_min", "l1_fai_out_of_tol", "l1_fai_submitted")

# MES/FactoryOS 가 주는 생산·불량 정보.
MES_COLS = tuple(c for c in L1_COLS if c not in FAI_COLS)

# ★ 계층이 곧 **설치 단계**다. 세 숫자가 설치 가치를 말한다.
LAYERS = {
    "L0": L0_COLS + L0M_COLS + FAI_COLS,
    "L0+Cell": L0_COLS + L0M_COLS + FAI_COLS + L2_COLS,
    "L0+Cell+MES": L0_COLS + L0M_COLS + FAI_COLS + L2_COLS + MES_COLS,
    # 하위 호환 — 기존 실험 스크립트가 쓴다
    "L0+L0′": L0_COLS + L0M_COLS,
    "L0+L0′+L1": L0_COLS + L0M_COLS + L1_COLS + L2_COLS,
}
