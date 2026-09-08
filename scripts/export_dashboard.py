"""관제 대시보드 데이터 — `sr score` 파이프라인을 돌려 화면이 읽는 JSON 을 만들고 템플릿에 채운다.

    python scripts/export_dashboard.py            → datasets/generated/dashboard_data.json + /tmp 발행용 html 경로 출력

화면 템플릿은 src/f4ge_supplier_risk/web/static/관제.html (디자인 시스템은 f4ge-quality-prediction 승계).
날짜는 합성 데이터의 기준일을 오늘로 옮겨 표시한다 — 절대 날짜는 의미가 없고 순서와 간격만 의미가 있다.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import LAYERS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import discrepancy, two_stage
from f4ge_supplier_risk.prediction.run import _predict
from f4ge_supplier_risk.prediction.score import build_scores

PRD = {"prd_bracket": "브래킷", "prd_shaft": "샤프트", "prd_housing": "하우징", "prd_flange": "플랜지"}
MKT = {"eu": "유럽", "us_west": "미 서안", "us_east": "미 동안"}
# CellOS·FactoryOS 는 협력 조건이라 12곳 전부 있다. 차이는 자체 MES 가 FactoryOS 에 연동돼 생산·불량 데이터가 오는가.
TYPE = {"a": {"label": "MES 연동", "hasMes": True, "hasCell": True},
        "b": {"label": "생산·불량 미연동", "hasMes": False, "hasCell": True},
        "c": {"label": "생산·불량 미연동", "hasMes": False, "hasCell": True}}
# 권고 정밀도 — scripts/coverage_sweep.py a=3 행 (8 seed). "편향 심한 공장" = 보고 공장 내 실효 편향 중앙값 아래.
EFFECT = {
    "seeds": 8, "reporting": 3,
    "rows": [
        ["site_visit", 6.8, 0.0, 0.375], ["tighten_inspection", 22.9, 0.772, 0.406],
        ["call", 8.9, 0.062, 0.237], ["none", 231.5, 0.155, None],
    ],
    "base": {"risky": 0.20, "biased_reporting": 0.315},
    # 12곳 전부 보고 + MES 하한 OFF 이면 방문 정밀도 94.9% — 불일치탐지.md §4-1
    "note_all12": {"visit_prec": 0.949, "visit_prec_floor": 0.618, "rank_all": 0.868, "rank_3": 0.664},
}


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main() -> None:
    cfg = config.load(str(ROOT / "configs/generator.yaml"))
    data = build_dataset(cfg)
    facs = {f["factory_id"]: f for f in data["factories"]}
    orders = {o["order_id"]: o for o in data["orders"]}
    meta = {m["order_id"]: m for m in data["order_meta"]}
    outc = {q["order_id"]: q for q in data["quality_outcomes"]}
    fai = {f["order_id"]: f for f in data["fai_reports"]}
    truth = {t["order_id"]: t for t in data["ground_truth"]}

    df = build(data)
    tr, te = split_by_time(df)
    pred_tr, pred = _predict(tr, tr), _predict(tr, te)
    scored = build_scores(
        tr, pred_tr, discrepancy.fit_predict(tr, tr),
        te, pred, discrepancy.fit_predict(tr, te), discrepancy.reasons(tr, te),
    )
    ex = two_stage.explain(tr, te, LAYERS["L0+Cell+MES"])
    trust_low = set(scored.loc[scored["factory_trust_low"].astype(bool), "factory_id"])

    # 기준일: 테스트 구간 발주일의 80% 지점. 그 뒤에 끝나는 오더가 "진행 중".
    t0 = _dt(data["orders"][0]["ordered_at"])
    ordered = [_dt(o["ordered_at"]) for o in orders.values() if o["order_id"] in set(te["order_id"])]
    cut = sorted(ordered)[int(len(ordered) * 0.8)]
    today_real = datetime(2026, 9, 8, tzinfo=timezone.utc)
    shift = today_real - cut
    D = lambda d: (d + shift).date().isoformat()

    rows = []
    for _, s in scored.iterrows():
        oid = s["order_id"]; o = orders[oid]; m = meta[oid]; q = outc[oid]; t = truth[oid]
        start = _dt(o["ordered_at"]); end = start + timedelta(days=t["actual_days"])
        late = end <= cut
        prog = min((cut - start).total_seconds() / max((end - start).total_seconds(), 1), 1.35) if not late else 1.35
        f = fai.get(oid)
        rows.append({
            "id": oid, "fac": o["factory_id"], "prd": PRD[o["product_id"]], "mkt": MKT[o["market"]],
            "qty": o["order_qty"], "ppm": round(float(s["predicted_ppm"])), "risk": s["risk_level"],
            "rep": None if bool(s["reported_missing"]) else round(float(s["reported_defect_rate"]), 4),
            "hasMes": bool(facs[o["factory_id"]]["has_mes"]),
            "disc": round(float(s["discrepancy"]), 4), "discFlag": bool(s["discrepancy_flag"]),
            "trustLow": bool(s["factory_trust_low"]), "act": s["recommended_action"],
            "why": s["action_reason"] or None, "check": s["action_check"] or None,
            "reason": s["reason_primary"] if isinstance(s["reason_primary"], str) else None,
            "reasons": list(s["reason_codes"]),
            "orderedAt": D(start), "promised": D(_dt(o["promised_date"])), "day": (start - t0).days,
            "prog": round(prog, 2), "late": late, "slack": round(o["lead_slack"], 4),
            "insp": q["incoming_inspected_qty"], "rej": q["incoming_reject_qty"],
            "lotRej": q["lot_result"] == "reject", "delay": m["report_delay_days_mean"],
            "miss": m["report_missing_count"], "blank": m["field_blank_count"],
            "fai": round(f["margin_min"], 4) if f else None, "faiSub": bool(f),
        })

    hist = {}
    for _, r in tr.iterrows():
        h = hist.setdefault(r["factory_id"], [0, 0])
        h[0] += 1; h[1] += int(r["y_lot_reject"])

    factories = []
    for fid, f in facs.items():
        mine = [r for r in rows if r["fac"] == fid]
        ev = [r for r in mine if r["rep"] is not None]
        ppms = [r["ppm"] for r in mine]
        acts = Counter(r["act"] for r in mine)
        mix = Counter(c for r in mine for c in r["reasons"])
        kobs = ex["kappa_obs"].get(fid, 0)
        factories.append({
            "id": fid, "prd": PRD[f["product_id"]], "type": f["factory_type"], **TYPE[f["factory_type"]],
            "orders": len(mine), "ppmMed": round(float(np.median(ppms))) if ppms else None,
            "ppmP90": round(float(np.quantile(ppms, 0.9))) if ppms else None,
            "disc": round(float(np.mean([r["disc"] for r in ev])), 4) if ev else None,
            "trustLow": fid in trust_low,
            "kappa": round(float(ex["kappa"].get(fid, ex["kappa_pooled"])), 4), "kappaObs": int(max(kobs, 0)),
            "actions": {a: acts.get(a, 0) for a in ("site_visit", "tighten_inspection", "call", "none")},
            "reasonMix": dict(mix),
            # 보고 경로가 없는 공장(b·c)은 지연·누락 자체가 정의되지 않는다 — 0 이 아니라 null
            "delayMed": (round(float(np.median(dl)), 1) if (dl := [r["delay"] for r in mine if r["delay"] is not None]) else None),
            "missSum": (int(sum(r["miss"] for r in mine if r["miss"] is not None)) if f["has_mes"] else None),
            "faiRate": round(float(np.mean([r["faiSub"] for r in mine])), 4) if mine else 0.0,
            "histRejRate": round(hist.get(fid, [1, 0])[1] / max(hist.get(fid, [1, 0])[0], 1), 4),
            "inspSum": int(sum(r["insp"] for r in mine)), "rejSum": int(sum(r["rej"] for r in mine)),
        })

    reason_text = {code: {"t": txt, "c": chk} for code, _, txt, chk in discrepancy.REASONS}
    reason_text[discrepancy.NO_EVIDENCE[0]] = {"t": discrepancy.NO_EVIDENCE[1], "c": discrepancy.NO_EVIDENCE[2]}
    counts = Counter(f["type"] for f in factories)
    out = {
        "today": today_real.date().isoformat(), "rows": rows, "factories": factories,
        "kappaPooled": round(float(ex["kappa_pooled"]), 4),
        "metrics": {"orders": len(rows), "spearman": round(float(spearmanr(pred, te["true_escape_rate"].to_numpy()).statistic), 3)},
        "coverage": {"a": counts["a"], "b": counts["b"], "c": counts["c"], "total": len(factories)},
        "effect": EFFECT, "reasonText": reason_text,
    }
    (ROOT / "datasets/generated/dashboard_data.json").write_text(json.dumps(out, ensure_ascii=False))
    tpl = (ROOT / "src/f4ge_supplier_risk/web/static/관제.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "datasets/generated/관제.html"
    dst.write_text(html)
    inflight = sum(not r["late"] for r in rows)
    print(f"rows {len(rows)} · 진행 중 {inflight} · 공장 {counts} · Spearman {out['metrics']['spearman']}")
    print(f"→ {dst}")


if __name__ == "__main__":
    main()
