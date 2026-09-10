"""피처를 줄이면 유출 예측이 어떻게 되나 — 8 seed (2026-09-10).

`two_stage` 는 카운트(l1_rep_defects · l1_rep_produced)를 cols 와 무관하게 직접 읽는다.
그래서 여기서 줄이는 것은 **1단계 Ridge 의 회귀 피처**다. 카운트는 어느 구성에서도 살아 있다.

    python scripts/feature_ablation.py     → docs/_피처_ablation.md
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
from f4ge_supplier_risk.features.build import (
    FAI_COLS, L0_COLS, L0M_COLS, L2_COLS, L3_COLS, LAYERS, MES_COLS, build,
)
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import two_stage

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
FULL = LAYERS["L0+Cell+MES+ERP"]
COUNTS = ("l1_rep_produced", "l1_rep_defects", "l1_reported_defect_rate")   # Ridge 에서 자동 제외
MES_NC = tuple(c for c in MES_COLS if c not in COUNTS)                      # 보고 파생 7개
PAIR = ("l3_material_gap", "l3_wip_gap", "l3_ot_vs_reject")                 # MES↔ERP 대조
ERP_PURE = tuple(c for c in L3_COLS if c not in PAIR)

def R(cols):  # Ridge 회귀 피처 수
    return len([c for c in cols if not c.startswith("l1_rep")])

CONFIGS = [
    ("전체 (현행)",              FULL),
    ("− L0 포지 기록",           tuple(c for c in FULL if c not in L0_COLS)),
    ("− L0′ 보고 메타",          tuple(c for c in FULL if c not in L0M_COLS)),
    ("− FAI",                    tuple(c for c in FULL if c not in FAI_COLS)),
    ("− L2 CellOS",              tuple(c for c in FULL if c not in L2_COLS)),
    ("− L3 ERP",                 tuple(c for c in FULL if c not in L3_COLS)),
    ("− MES 파생 7",             tuple(c for c in FULL if c not in MES_NC)),
    ("− 교차검증 짝 3",          tuple(c for c in FULL if c not in PAIR)),
    ("카운트 + L2 만",           COUNTS + L2_COLS),
    ("카운트 + L3 만",           COUNTS + L3_COLS),
    ("카운트 + L0 만",           COUNTS + L0_COLS),
    ("카운트 + L2 + ERP순수",    COUNTS + L2_COLS + ERP_PURE),
    ("카운트 + L2 + L3",         COUNTS + L2_COLS + L3_COLS),
]

def main() -> None:
    cfg = config.load("configs/generator.yaml")
    pre = []
    for s in SEEDS:
        c = copy.deepcopy(cfg); c["seed"] = s
        df = build(build_dataset(c))
        pre.append(split_by_time(df))
        print(f"  seed {s} 준비", flush=True)

    rows = []
    for name, cols in CONFIGS:
        v = [evaluate(te, two_stage.fit_predict(tr, te, tuple(cols)))["rank_corr_true"] for tr, te in pre]
        v = np.array(v)
        rows.append(dict(구성=name, 피처=R(cols), 평균=v.mean(), sd=v.std(), 최소=v.min(), 양수=int((v > 0).sum())))
        print(f"  {name:22s} 피처 {R(cols):2d}  {v.mean():+.3f} ±{v.std():.3f}  최소 {v.min():+.3f}", flush=True)

    T = pd.DataFrame(rows)
    base = T.loc[0, "평균"]
    T["전체 대비"] = T["평균"] - base

    lines = ["# 피처 ablation — 유출 순위상관, 8 seed", "",
             "`two_stage` 는 카운트를 cols 와 무관하게 직접 읽는다 — 줄이는 것은 **1단계 Ridge 회귀 피처**다.", "",
             "| 구성 | Ridge 피처 | 순위상관 | ±sd | 최소 | 전체 대비 |", "|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        mark = "**" if r.구성 == "전체 (현행)" else ""
        lines.append(f"| {mark}{r.구성}{mark} | {r.피처} | {r.평균:+.3f} | {r.sd:.3f} | {r.최소:+.3f} | {r['전체 대비']:+.3f} |")
    out = Path(__file__).resolve().parents[1] / "docs" / "_피처_ablation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
