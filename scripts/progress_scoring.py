"""생산 10% 시점에 채점하면 어떻게 되나 — 진척 sweep (2026-09-10).

**왜 재나.** 지금까지의 모든 벤치는 오더가 **끝난 뒤**(보고 전량 도착) 채점한 값이다.
실제 쓰임은 「10% 생산됐을 때 100% 시점 불량을 예보한다」인데 그 성능을 한 번도 안 쟀다.

구조는 이미 그렇게 되어 있다 — 2단 1단계가 `(k_rep + 150·prior)/(n_rep + 150)` 이라
보고가 적을 때 자동으로 사전분포 쪽으로 당겨진다. 진척 10% 면 n_rep ≈ 83 이라
사전분포 비중이 64%, 100% 면 15% 다. 그 설계가 실제로 작동하는지 본다.

학습은 완료된 오더(보고 전량)로, 채점은 진척 p 까지만 잘라서 한다 — 실제 배포와 같은 조건.

    python scripts/progress_scoring.py     → docs/_진척_채점.md
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import evaluate, split_by_time
from f4ge_supplier_risk.features.build import FULL_COLS, LAYERS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import baseline
from f4ge_supplier_risk.models import factory_params as fp
from f4ge_supplier_risk.models import two_stage as TS

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
PROGRESS = (0.10, 0.25, 0.50, 0.75, 1.00)


def truncate(data, p):
    """진척 p 까지만 남긴다. MES 는 누적 생산 기준, Cell·ERP 는 그 시점 날짜 기준."""
    if p >= 1.0:
        return data
    qty = {o["order_id"]: o["order_qty"] for o in data["orders"]}
    out = dict(data)

    # Cell 카운터로 진척 p 에 닿는 날을 찾는다
    cut_day = {}
    for r in data["cell_daily"]:
        oid = r["order_id"]
        if r.get("produced_by_counter", 0) <= p * qty.get(oid, 1):
            cut_day[oid] = max(cut_day.get(oid, -1), r["day_index"])

    out["factory_reports"] = [
        r for r in data["factory_reports"]
        if r.get("is_missing") or r.get("produced_qty", 0) <= p * qty.get(r["order_id"], 1)
    ]
    out["cell_daily"] = [r for r in data["cell_daily"] if r["day_index"] <= cut_day.get(r["order_id"], -1)]
    out["erp_daily"] = [r for r in data["erp_daily"] if r["day_index"] <= cut_day.get(r["order_id"], -1)]
    return out


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    rows = []
    pre = []
    for s in SEEDS:
        c = copy.deepcopy(cfg); c["seed"] = s
        data = build_dataset(c)
        df = build(data)
        tr, te = split_by_time(df)
        pre.append((data, tr, len(tr), len(te)))
        print(f"  seed {s} 준비", flush=True)

    for p in PROGRESS:
        acc = {k: [] for k in ("two_stage", "l0", "p_order", "n_rep", "w_prior", "no_report")}
        for (data, tr, n_tr, n_te), s in zip(pre, SEEDS):
            dt = truncate(data, p)
            dfp = build(dt)
            te = dfp.iloc[n_tr:n_tr + n_te]
            nrep = te["l1_rep_produced"].replace(0, np.nan)
            acc["n_rep"].append(float(nrep.median()))
            acc["w_prior"].append(float((150.0 / (nrep.fillna(0) + 150.0)).median()))
            acc["no_report"].append(float((te["l1_rep_produced"] <= 0).mean()))
            acc["two_stage"].append(evaluate(te, TS.fit_predict(tr, te, FULL_COLS))["rank_corr_true"])
            acc["l0"].append(evaluate(te, baseline.fit_predict(tr, te, LAYERS["L0"]))["rank_corr_true"])
            P = fp.estimate(tr)
            from scipy.stats import spearmanr
            acc["p_order"].append(spearmanr(fp.order_internal_rate(te, P), te["true_internal_rate"]).statistic)
        rows.append(dict(진척=p, **{k: np.mean(v) for k, v in acc.items()}))
        r = rows[-1]
        print(f"  진척 {p:.0%}  보고 누적생산 중앙 {r['n_rep']:.0f} · 사전분포 비중 {r['w_prior']:.0%} · "
              f"미보고 {r['no_report']:.0%} → 유출 {r['two_stage']:+.3f} · 제조품질 {r['p_order']:+.3f} · L0만 {r['l0']:+.3f}", flush=True)

    T = pd.DataFrame(rows)
    lines = [f"# 진척 시점별 채점 — {len(SEEDS)} seed", "",
             "학습은 완료 오더(보고 전량), 채점은 진척 p 까지만 잘라서. 실제 배포와 같은 조건이다.", "",
             "| 진척 | 보고 누적생산(중앙) | 사전분포 비중 | 미보고 오더 | 유출 순위상관 | 제조품질 순위상관 | L0 만 |",
             "|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        lines.append(f"| {r.진척:.0%} | {r.n_rep:.0f} | {r.w_prior:.0%} | {r.no_report:.0%} | "
                     f"**{r.two_stage:+.3f}** | {r.p_order:+.3f} | {r.l0:+.3f} |")
    out = Path(__file__).resolve().parents[1] / "docs" / "_진척_채점.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + T.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
