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

from f4ge_supplier_risk.web.dashboard_data import build_dashboard  # 서버와 같은 빌더


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main() -> None:
    cfg = config.load(str(ROOT / "configs/generator.yaml"))
    data = build_dataset(cfg)
    orders = {o["order_id"]: o for o in data["orders"]}

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

    out = build_dashboard(data, scored, ex, today=today_real, shift=shift)
    out["metrics"]["spearman"] = round(float(spearmanr(pred, te["true_escape_rate"].to_numpy()).statistic), 3)
    counts = {"a": out["coverage"]["a"], "b": out["coverage"]["b"]}
    (ROOT / "datasets/generated/dashboard_data.json").write_text(json.dumps(out, ensure_ascii=False))
    tpl = (ROOT / "src/f4ge_supplier_risk/web/static/관제.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "datasets/generated/관제.html"
    dst.write_text(html)
    inflight = sum(not r["late"] for r in out["rows"])
    print(f"rows {len(out['rows'])} · 진행 중 {inflight} · 공장 {counts} · Spearman {out['metrics']['spearman']}")
    print(f"→ {dst}")


if __name__ == "__main__":
    main()
