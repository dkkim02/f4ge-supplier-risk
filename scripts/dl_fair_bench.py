"""딥러닝 재도전 — 조건을 맞추고, 구조를 모델 안에 넣는다. 8 seed (2026-09-10).

기존 벤치는 MLP·HGB 에 FULL_COLS 42개만 주고 **공장 identity 를 안 줬다**. 2단은 공장별 κ 12개를 갖는다.
조건을 맞춰 재측정하고, 거기서 나온 진단대로 새 모델(models/deep_kappa.py)을 붙인다.

    python scripts/dl_fair_bench.py            → docs/_딥러닝_재도전.md
    python scripts/dl_fair_bench.py coldstart  → 공장 홀드아웃까지
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
from f4ge_supplier_risk.features.build import FULL_COLS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import deep_kappa, hgb, mlp
from f4ge_supplier_risk.models import two_stage as TS

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)


def prep(seed):
    c = copy.deepcopy(config.load("configs/generator.yaml"))
    c["seed"] = seed
    df = build(build_dataset(c))
    tr, te = split_by_time(df)
    tr, te = tr.copy(), te.copy()
    tr["ih"] = TS._stage_a(tr, tr, FULL_COLS)
    te["ih"] = TS._stage_a(tr, te, FULL_COLS)
    fids = sorted(set(tr["factory_id"]) | set(te["factory_id"]))
    for d in (tr, te):
        for f in fids:
            d[f"fac_{f}"] = (d["factory_id"] == f).astype(float)
    return tr, te, tuple(f"fac_{f}" for f in fids)


def ts_no_online(tr, te, cols):
    """온라인 κ 갱신 없이 — 학습 구간 κ 를 고정해 채점. deep_kappa 와 같은 조건이다."""
    ih_tr = TS._stage_a(tr, tr, cols)
    post, pooled = TS._kappa_by_factory(tr, ih_tr)
    a0 = TS._KAPPA_PRIOR_STRENGTH
    b0 = a0 / max(pooled, TS._FLOOR)
    ih_te = TS._stage_a(tr, te, cols)
    kap = np.array([(lambda ab: ab[0] / ab[1])(post.get(str(f), (a0, b0))) for f in te["factory_id"]])
    return np.clip(ih_te * kap, TS._FLOOR, 1 - 1e-6)


CANDS = [
    ("two_stage (기준)",            lambda tr, te, FAC, s: TS.fit_predict(tr, te, FULL_COLS)),
    ("two_stage — κ 고정",          lambda tr, te, FAC, s: ts_no_online(tr, te, FULL_COLS)),
    ("hgb  피처 42",                lambda tr, te, FAC, s: hgb.fit_predict(tr, te, FULL_COLS)),
    ("mlp  피처 42",                lambda tr, te, FAC, s: mlp.fit_predict(tr, te, FULL_COLS, seed=s)),
    ("mlp  피처 42 + 공장",          lambda tr, te, FAC, s: mlp.fit_predict(tr, te, FULL_COLS + FAC, seed=s)),
    ("mlp  카운트비율 하나만",         lambda tr, te, FAC, s: mlp.fit_predict(tr, te, ("ih",), seed=s)),
    ("hgb  카운트비율 + 공장",         lambda tr, te, FAC, s: hgb.fit_predict(tr, te, ("ih",) + FAC)),
    ("★ deep_kappa  embed",        lambda tr, te, FAC, s: deep_kappa.fit_predict(tr, te, (), seed=s, mode="embed")),
    ("★ deep_kappa  feat",         lambda tr, te, FAC, s: deep_kappa.fit_predict(tr, te, (), seed=s, mode="feat")),
]


def main() -> None:
    cold = len(sys.argv) > 1 and sys.argv[1] == "coldstart"
    pre = [prep(s) for s in SEEDS]
    print("  준비 완료", flush=True)

    rows = []
    for name, fn in CANDS:
        v = np.array([evaluate(te, fn(tr, te, FAC, s))["rank_corr_true"] for (tr, te, FAC), s in zip(pre, SEEDS)])
        rows.append(dict(구성=name, 평균=v.mean(), sd=v.std(), 최소=v.min(), 양수=int((v > 0).sum())))
        print(f"  {name:24s} {v.mean():+.3f} ±{v.std():.3f}  최소 {v.min():+.3f}", flush=True)
    T = pd.DataFrame(rows)

    lines = ["# 딥러닝 재도전 — 8 seed", "",
             "기존 벤치는 MLP·HGB 에 공장 identity 를 주지 않았다. 조건을 맞추고 재측정했다.", "",
             "| 구성 | 순위상관 | ±sd | 최소 | 양수 seed |", "|---|---|---|---|---|"]
    for _, r in T.iterrows():
        lines.append(f"| {r.구성} | {r.평균:+.3f} | {r.sd:.3f} | {r.최소:+.3f} | {r.양수}/8 |")

    if cold:
        print("\n── 콜드스타트: 공장 3곳을 학습에서 통째로 뺀다 ──", flush=True)
        cr = []
        for (tr, te, FAC), s in zip(pre, SEEDS):
            fids = sorted(set(tr["factory_id"]))
            held = set(fids[-3:])
            tr2 = tr[~tr["factory_id"].isin(held)]
            te2 = te[te["factory_id"].isin(held)]
            if len(te2) < 10 or len(tr2) < 60:
                continue
            cr.append(dict(
                two_stage=evaluate(te2, TS.fit_predict(tr2, te2, FULL_COLS))["rank_corr_true"],
                deep_feat=evaluate(te2, deep_kappa.fit_predict(tr2, te2, (), seed=s, mode="feat"))["rank_corr_true"],
                deep_embed=evaluate(te2, deep_kappa.fit_predict(tr2, te2, (), seed=s, mode="embed"))["rank_corr_true"],
                n=len(te2)))
        CT = pd.DataFrame(cr)
        lines += ["", "## 콜드스타트 — 학습에서 못 본 공장 3곳만 채점", "",
                  f"테스트 오더 중앙 {CT['n'].median():.0f}건 · {len(CT)} seed", "",
                  "| 모델 | 순위상관 | ±sd | 최소 |", "|---|---|---|---|"]
        print(f"{'모델':16s} {'평균':>8} {'sd':>7} {'최소':>8}")
        for k in ("two_stage", "deep_feat", "deep_embed"):
            v = CT[k]
            print(f"{k:16s} {v.mean():>+8.3f} {v.std():>7.3f} {v.min():>+8.3f}")
            lines.append(f"| {k} | {v.mean():+.3f} | {v.std():.3f} | {v.min():+.3f} |")

    out = Path(__file__).resolve().parents[1] / "docs" / "_딥러닝_재도전.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
