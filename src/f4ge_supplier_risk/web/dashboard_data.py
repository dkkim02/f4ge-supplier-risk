"""화면이 읽는 JSON 한 덩어리 — 정적 발행(scripts/export_dashboard.py)과 서버(web/service.py)가 같은 것을 쓴다.

입력은 생성기 모양 레코드(운영에서는 ingestion.derive 가 만든 것)와 채점 결과다.
완료/진행 중은 **결과(입고검사)를 우리가 알게 됐는가**(label_available_at ≤ 기준일)로 가른다 — 운영에서 알 수 있는 유일한 기준이다.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from f4ge_supplier_risk.models import discrepancy

PRD = {"prd_bracket": "브래킷", "prd_shaft": "샤프트", "prd_housing": "하우징", "prd_flange": "플랜지"}
MKT = {"eu": "유럽", "us_west": "미 서안", "us_east": "미 동안"}
# 권고 정밀도 — scripts/coverage_sweep.py a=3 행 (8 seed). "편향 심한 공장" = 보고 공장 내 실효 편향 중앙값 아래.
EFFECT = {
    "seeds": 8, "reporting": 3,
    "rows": [["site_visit", 6.8, 0.0, 0.375], ["tighten_inspection", 22.9, 0.772, 0.406],
             ["call", 8.9, 0.062, 0.237], ["none", 231.5, 0.155, None]],
    "base": {"risky": 0.20, "biased_reporting": 0.315},
    "note_all12": {"visit_prec": 0.949, "visit_prec_floor": 0.618, "rank_all": 0.868, "rank_3": 0.664},
}


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def _prd(pid: str) -> str:
    return PRD.get(pid, pid.removeprefix("prd_"))


def build_dashboard(
    data: dict[str, list[dict[str, Any]]], scored: pd.DataFrame, explain: dict[str, Any],
    *, today: datetime, shift: Any = None, role: str = "admin",
) -> dict[str, Any]:
    """`shift` 는 합성 데이터의 날짜를 오늘로 옮길 때만 쓴다(timedelta). 운영에서는 None."""
    D = (lambda d: (d + shift).date().isoformat()) if shift else (lambda d: d.date().isoformat())
    cut = today - shift if shift else today

    facs = {f["factory_id"]: f for f in data["factories"]}
    orders = {o["order_id"]: o for o in data["orders"]}
    meta = {m["order_id"]: m for m in data["order_meta"]}
    outc = {q["order_id"]: q for q in data["quality_outcomes"]}
    fai = {f["order_id"]: f for f in data["fai_reports"]}
    t0 = min(_dt(o["ordered_at"]) for o in data["orders"]) if data["orders"] else cut
    trust_low = set(scored.loc[scored["factory_trust_low"].astype(bool), "factory_id"])

    rows = []
    for _, s in scored.iterrows():
        oid = s["order_id"]; o = orders[oid]; m = meta[oid]; q = outc.get(oid); f = fai.get(oid)
        start, promised = _dt(o["ordered_at"]), _dt(o["promised_date"])
        # 끝난 오더 = 결과를 **우리가 알게 된** 오더. 결과 행이 있어도 label_available_at 이 기준일 뒤면
        # 아직 모르는 것이다(시간 규율). 출하는 됐지만 입고검사 전인 오더가 여기 들어오고, 검사 강화 권고가 그 자리다.
        late = q is not None and _dt(q["label_available_at"]) <= cut
        span = max((promised - start).total_seconds(), 1.0)
        prog = 1.35 if late else float(min((cut - start).total_seconds() / span, 1.35))
        has_mes = bool(facs.get(o["factory_id"], {}).get("has_mes", False))
        rows.append({
            "id": oid, "fac": o["factory_id"], "prd": _prd(o["product_id"]), "mkt": MKT.get(o["market"], o["market"]),
            "qty": o["order_qty"], "ppm": round(float(s["predicted_ppm"])), "risk": s["risk_level"],
            "rep": None if bool(s["reported_missing"]) else round(float(s["reported_defect_rate"]), 4),
            "hasMes": has_mes, "disc": round(float(s["discrepancy"]), 4), "discFlag": bool(s["discrepancy_flag"]),
            "trustLow": bool(s["factory_trust_low"]), "act": s["recommended_action"],
            "why": s["action_reason"] or None, "check": s["action_check"] or None,
            "reason": s["reason_primary"] if isinstance(s["reason_primary"], str) else None,
            "reasons": list(s["reason_codes"]),
            "orderedAt": D(start), "promised": D(promised), "day": (start - t0).days,
            "prog": round(prog, 2), "late": late, "slack": round(float(o["lead_slack"]), 4),
            "insp": q["incoming_inspected_qty"] if q else None, "rej": q["incoming_reject_qty"] if q else None,
            "lotRej": (q["lot_result"] == "reject") if q else None,
            "delay": m["report_delay_days_mean"], "miss": m["report_missing_count"], "blank": m["field_blank_count"],
            "fai": round(f["margin_min"], 4) if f else None, "faiSub": bool(f),
        })

    hist: dict[str, list[int]] = {}
    for q in data["quality_outcomes"]:
        h = hist.setdefault(orders[q["order_id"]]["factory_id"], [0, 0]) if q["order_id"] in orders else None
        if h is not None:
            h[0] += 1; h[1] += int(q["lot_result"] == "reject")

    factories = []
    for fid, f in facs.items():
        mine = [r for r in rows if r["fac"] == fid]
        if not mine:
            continue
        ev = [r for r in mine if r["rep"] is not None]
        ppms = [r["ppm"] for r in mine]
        acts = Counter(r["act"] for r in mine)
        mix = Counter(c for r in mine for c in r["reasons"])
        kobs = explain["kappa_obs"].get(fid, 0)
        dl = [r["delay"] for r in mine if r["delay"] is not None]
        has_mes = bool(f.get("has_mes", False))
        factories.append({
            "id": fid, "prd": _prd(f.get("product_id", "")), "type": "a" if has_mes else "b",
            "label": "MES 연동" if has_mes else "생산·불량 미연동", "hasMes": has_mes, "hasCell": True,
            "orders": len(mine), "ppmMed": round(float(np.median(ppms))), "ppmP90": round(float(np.quantile(ppms, 0.9))),
            "disc": round(float(np.mean([r["disc"] for r in ev])), 4) if ev else None, "trustLow": fid in trust_low,
            "kappa": round(float(explain["kappa"].get(fid, explain["kappa_pooled"])), 4), "kappaObs": int(max(kobs, 0)),
            "actions": {a: acts.get(a, 0) for a in ("site_visit", "tighten_inspection", "call", "none")},
            "reasonMix": dict(mix),
            "delayMed": round(float(np.median(dl)), 1) if dl else None,
            "missSum": int(sum(r["miss"] or 0 for r in mine)) if has_mes else None,
            "faiRate": round(float(np.mean([r["faiSub"] for r in mine])), 4),
            "histRejRate": round(hist.get(fid, [1, 0])[1] / max(hist.get(fid, [1, 0])[0], 1), 4),
            "inspSum": int(sum(r["insp"] or 0 for r in mine)), "rejSum": int(sum(r["rej"] or 0 for r in mine)),
        })

    reason_text = {code: {"t": txt, "c": chk} for code, _, txt, chk in discrepancy.REASONS}
    reason_text[discrepancy.NO_EVIDENCE[0]] = {"t": discrepancy.NO_EVIDENCE[1], "c": discrepancy.NO_EVIDENCE[2]}
    n_a = sum(1 for f in facs.values() if f.get("has_mes"))
    return {
        "today": today.date().isoformat(), "role": role, "rows": rows, "factories": factories,
        "kappaPooled": round(float(explain["kappa_pooled"]), 4), "metrics": {"orders": len(rows)},
        "coverage": {"a": n_a, "b": len(facs) - n_a, "c": 0, "total": len(facs)},
        "effect": EFFECT, "reasonText": reason_text,
    }


def restrict(dash: dict[str, Any], factory_id: str) -> dict[str, Any]:
    """현장 계정 — 서버에서 자기 공장으로 잘라 준다. 다른 공장은 행 자체가 없다."""
    out = dict(dash)
    out["role"] = "site"
    out["rows"] = [r for r in dash["rows"] if r["fac"] == factory_id]
    out["factories"] = [f for f in dash["factories"] if f["id"] == factory_id]
    out["metrics"] = {**dash["metrics"], "orders": len(out["rows"])}
    return out
