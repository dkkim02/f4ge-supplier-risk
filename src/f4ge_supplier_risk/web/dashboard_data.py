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
# 권고 정밀도 — 09-08 저녁, **1공장 1오더 데이터**(330건, 시간순 테스트 100건) · 권고 코드 review_needed 하나 · 8 seed.
#   scripts/discrepancy_precision.py (a=3) · scripts/coverage_sweep.py 3 12 (note_all12).
# 행 = [코드, 건수, 실제 상위 20% 위험 비율, 편향 심한 공장 비율]. "편향 심한 공장" = 실효 편향이 12곳 중앙값 아래.
# ⚠ 코드 통합으로 두 축의 오더가 한 집합에 섞여 있다(축별 값은 risk_level·discrepancy_flag 로 사후 분할).
#   900건 시절 46.2% → 330건에서 37.0%. 보고 오더의 편향 공장 정밀도는 기저와 같다(40.8 vs 42.0%).
# 09-08 저녁 3차 — 네 소스 12곳 전부(유형 폐기) · 1공장 1오더 330건 · review_needed 단일 코드 · 8 seed
#   scripts/discrepancy_precision.py · scripts/model_bench.py(순위상관 0.787). 12곳 전부 보고하므로 "보고 있는 오더" 행 = 전체.
#   임계 재조정(scripts/review_threshold_sweep.py): 오더 단독 불일치 축 끔 · 신뢰 낮은 공장 × 불일치 0.90 → 100건 중 14.5건, 정밀도 56.5%(기저 20.2%, 2.8배).
EFFECT = {
    "seeds": 8, "reporting": 12, "n_test": 100,
    "rows": [["review_needed", 14.5, 0.565, 0.476], ["review_needed_reporting", 14.5, None, 0.574],
             ["none", 85.6, 0.139, None]],
    "base": {"risky": 0.202, "biased_reporting": 0.499},
    "note_all12": {"visit_prec": 0.487, "visit_base": 0.499, "rank_all": 0.787, "rank_3": 0.787},
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
        # 아직 모르는 것이다(시간 규율). 출하는 됐지만 입고검사 전인 오더가 여기 들어오고, 예측 위험 쪽 검토 필요가 그 자리다.
        late = q is not None and _dt(q["label_available_at"]) <= cut
        # 출하는 됐지만 입고검사 결과가 아직인 오더 — 한 공장에 한 오더 전제에서 "생산 중"과 "결과 대기"를 가른다
        shipped = q is not None and _dt(q["shipped_at"]) <= cut
        span = max((promised - start).total_seconds(), 1.0)
        prog = 1.35 if late else float(min((cut - start).total_seconds() / span, 1.35))
        has_mes = bool(facs.get(o["factory_id"], {}).get("has_mes", False))
        rows.append({
            "id": oid, "fac": o["factory_id"], "prd": _prd(o["product_id"]), "mkt": MKT.get(o["market"], o["market"]),
            "qty": o["order_qty"], "ppm": round(float(s["predicted_ppm"])), "risk": s["risk_level"],
            # 제조 품질 — 이 오더의 내부 불량률 추정 (09-08 저녁, 사용자 목표 ①)
            "internal": round(float(s["predicted_internal_rate"]), 5),
            "rep": None if bool(s["reported_missing"]) else round(float(s["reported_defect_rate"]), 4),
            "hasMes": has_mes, "disc": round(float(s["discrepancy"]), 4), "discFlag": bool(s["discrepancy_flag"]),
            "trustLow": bool(s["factory_trust_low"]), "act": s["recommended_action"],
            "why": s["action_reason"] or None, "check": s["action_check"] or None,
            "reason": s["reason_primary"] if isinstance(s["reason_primary"], str) else None,
            "reasons": list(s["reason_codes"]),
            "orderedAt": D(start), "promised": D(promised), "day": (start - t0).days,
            "prog": round(prog, 2), "late": late, "shipped": shipped, "slack": round(float(o["lead_slack"]), 4),
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
        has_mes = True  # 네 소스 전부 온다(09-08 저녁). 유형 구분은 폐기 — 구 화면 호환용 상수
        s0 = scored[scored["factory_id"] == fid].iloc[0]
        fm, fe = f.get("fields_mes") or {}, f.get("fields_erp") or {}
        factories.append({
            "id": fid, "prd": _prd(f.get("product_id", "")), "region": f.get("region") or None,
            "type": "a", "label": "4소스", "hasMes": True, "hasCell": True,
            # ── 목표 둘 + 정직도 (공장 파라미터 분해) ──
            "internal": round(float(s0["factory_internal_rate"]), 5),
            "detection": round(float(s0["factory_detection_rate"]), 4),
            "honesty": round(float(s0["factory_report_honesty"]), 3),
            "paramObs": int(s0["factory_param_obs"]),
            # ── 보고 프로필: 형식 · 채우는 필드 ──
            "profile": {
                "period": f.get("mes_period"), "unit": f.get("qty_unit"), "lotSize": f.get("lot_size"),
                "codeScheme": f.get("defect_code_scheme"),
                "fieldsMes": [k for k, v in fm.items() if v], "fieldsMesAll": list(fm),
                "fieldsErp": [k for k, v in fe.items() if v], "fieldsErpAll": list(fe),
            },
            "orders": len(mine), "ppmMed": round(float(np.median(ppms))), "ppmP90": round(float(np.quantile(ppms, 0.9))),
            "disc": round(float(np.mean([r["disc"] for r in ev])), 4) if ev else None, "trustLow": fid in trust_low,
            "kappa": round(float(explain["kappa"].get(fid, explain["kappa_pooled"])), 4), "kappaObs": int(max(kobs, 0)),
            "actions": {a: acts.get(a, 0) for a in ("review_needed", "none")},
            "reasonMix": dict(mix),
            "delayMed": round(float(np.median(dl)), 1) if dl else None,
            "missSum": int(sum(r["miss"] or 0 for r in mine)) if has_mes else None,
            "faiRate": round(float(np.mean([r["faiSub"] for r in mine])), 4),
            "histRejRate": round(hist.get(fid, [1, 0])[1] / max(hist.get(fid, [1, 0])[0], 1), 4),
            "inspSum": int(sum(r["insp"] or 0 for r in mine)), "rejSum": int(sum(r["rej"] or 0 for r in mine)),
        })

    log = _daily_log(data, {r["id"] for r in rows}, facs, cut, D)

    reason_text = {code: {"t": txt, "c": chk} for code, _, txt, chk in discrepancy.REASONS}
    reason_text[discrepancy.NO_EVIDENCE[0]] = {"t": discrepancy.NO_EVIDENCE[1], "c": discrepancy.NO_EVIDENCE[2]}
    n_a = len(facs)
    return {
        "today": today.date().isoformat(), "role": role, "rows": rows, "factories": factories,
        "kappaPooled": round(float(explain["kappa_pooled"]), 4), "metrics": {"orders": len(rows)},
        "coverage": {"a": n_a, "b": 0, "c": 0, "total": len(facs), "sources": ["MES", "CellOS", "ERP", "포지 기록"]},
        "params": {"pooled": {k: round(float(v), 5) for k, v in (explain.get("params_pooled") or {}).items()}},
        "effect": EFFECT, "reasonText": reason_text, "log": log,
    }


def _daily_log(data, order_ids: set[str], facs, cut: datetime, D) -> dict[str, list[dict[str, Any]]]:
    """공장별 이벤트 로그 — 화면 하단 「일자별 로그」. 채점 구간 오더의, 기준일 이전 사건만.

    사건 종류(k): order 발주 · report MES 집계 도착 · missing MES 집계 누락 · erp_missing ERP 마감 누락 · fai 초도품 검사 · ship 출하 · label 입고검사 결과 도착.
    """
    fac_of = {o["order_id"]: o["factory_id"] for o in data["orders"]}
    out: dict[str, list[dict[str, Any]]] = {fid: [] for fid in facs}
    def add(oid, when, k, t):
        if oid in order_ids and when <= cut:
            out[fac_of[oid]].append({"d": D(when), "o": oid.removeprefix("ord_"), "k": k, "t": t})
    for o in data["orders"]:
        add(o["order_id"], _dt(o["ordered_at"]), "order",
            f"발주 · {_prd(o['product_id'])} {o['order_qty']:,}개 · 납기 {_dt(o['promised_date']).date().isoformat()[5:]} · {MKT.get(o['market'], o['market'])}")
    for r in data["factory_reports"]:
        if fac_of.get(r["order_id"]) is None:
            continue
        if r.get("is_missing"):
            add(r["order_id"], _dt(r["due_at"]), "missing", f"MES 집계 누락 (#{r['seq']})")
        else:
            late = (_dt(r["reported_at"]) - _dt(r["due_at"])).total_seconds() / 86400
            add(r["order_id"], _dt(r["reported_at"]), "report",
                f"MES 집계 #{r['seq']} · 생산 {r['produced_qty']:,} · 폐기 {r['scrap_qty']} · 재작업 {r['rework_qty']}"
                + (f" · 자체검사 불합격 {r['reject_qty']}" if "reject_qty" in r else "") + f" · {r['stage']}"
                + (f" · {late:.1f}일 지연" if late >= 1 else "") + (" · 특이사항" if r.get("issue_flag") else ""))
    for e in data.get("erp_daily", []):
        if e.get("is_missing") and fac_of.get(e["order_id"]):
            add(e["order_id"], _dt(e["due_at"]), "erp_missing", f"ERP 일 마감 누락 (D+{e['day_index'] + 1})")
    for f in data["fai_reports"]:
        add(f["order_id"], _dt(f["fai_at"]), "fai",
            f"초도품 검사 제출 · 치수 {f['dim_count']} · 최소 여유 {f['margin_min']:.2f} · 공차 밖 {f['out_of_tol_count']}")
    for q in data["quality_outcomes"]:
        add(q["order_id"], _dt(q["shipped_at"]), "ship", "출하")
        add(q["order_id"], _dt(q["label_available_at"]), "label",
            f"입고검사 결과 · 표본 {q['incoming_inspected_qty']} 중 불합격 {q['incoming_reject_qty']} · 로트 {'불합격' if q['lot_result'] == 'reject' else '합격'}"
            + (" · 클레임" if q.get("claim_occurred") else ""))
    for evs in out.values():
        evs.sort(key=lambda e: e["d"], reverse=True)
    return out


def restrict(dash: dict[str, Any], factory_id: str) -> dict[str, Any]:
    """현장 계정 — 서버에서 자기 공장으로 잘라 준다. 다른 공장은 행 자체가 없다."""
    out = dict(dash)
    out["role"] = "site"
    out["rows"] = [r for r in dash["rows"] if r["fac"] == factory_id]
    out["factories"] = [f for f in dash["factories"] if f["id"] == factory_id]
    out["log"] = {factory_id: dash.get("log", {}).get(factory_id, [])}
    out["metrics"] = {**dash["metrics"], "orders": len(out["rows"])}
    return out
