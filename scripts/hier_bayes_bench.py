"""계층 베이즈 1차 — shrinkage 세기를 추정하면 손으로 박은 것보다 나은가. 8 seed (2026-09-10).

`two_stage` 와 `deep_kappa` 는 관측 모형이 같고 `a0 = 3.0` 을 손으로 박았다.
`hier_bayes` 는 같은 자리를 `tau` 로 두고 추정한다. 재는 것 넷.

  ① 순위상관    — 기존 둘과 동률 이상인가
  ② tau 추정치  — 손으로 박은 세기와 얼마나 다른가. 그리고 데이터가 tau 를 움직이는가
  ③ funnel     — non-centered 가 실제로 필요한가 (centered 와 divergence 수 비교)
  ④ prior sens — 공장 12곳에서 tau 사전분포가 결과를 좌우하는가

    python scripts/hier_bayes_bench.py        → docs/_계층베이즈_1차.md
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_fair_bench import ts_no_online

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import evaluate, split_by_time
from f4ge_supplier_risk.features.build import FULL_COLS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import deep_kappa, hier_bayes
from f4ge_supplier_risk.models import two_stage as TS

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)


def prep(seed):
    c = copy.deepcopy(config.load("configs/generator.yaml"))
    c["seed"] = seed
    df = build(build_dataset(c))
    return split_by_time(df)


CANDS = [
    ("two_stage (기준)",        lambda tr, te, s: TS.fit_predict(tr, te, FULL_COLS)),
    ("two_stage — κ 고정",      lambda tr, te, s: ts_no_online(tr, te, FULL_COLS)),
    ("deep_kappa hier",        lambda tr, te, s: deep_kappa.fit_predict(tr, te, (), seed=s, mode="hier")),
    ("★ hier_bayes pool",      lambda tr, te, s: hier_bayes.fit_predict(tr, te, seed=0, mode="pool")),
    ("★ hier_bayes feat",      lambda tr, te, s: hier_bayes.fit_predict(tr, te, seed=0, mode="feat")),
    ("   hier_bayes centered", lambda tr, te, s: hier_bayes.fit_predict(tr, te, seed=0, mode="pool", centered=True)),
]


def main() -> None:
    pre = [prep(s) for s in SEEDS]
    print(f"  준비 완료 — 학습 {len(pre[0][0])} · 채점 {len(pre[0][1])}", flush=True)

    rows = []
    for name, fn in CANDS:
        t = time.time()
        v = np.array([evaluate(te, fn(tr, te, s))["rank_corr_true"] for (tr, te), s in zip(pre, SEEDS)])
        rows.append(dict(구성=name, 평균=v.mean(), sd=v.std(), 최소=v.min(),
                         양수=int((v > 0).sum()), 초=(time.time() - t) / len(SEEDS)))
        print(f"  {name:24s} {v.mean():+.3f} ±{v.std():.3f}  최소 {v.min():+.3f}  {rows[-1]['초']:.1f}s/seed", flush=True)
    T = pd.DataFrame(rows)

    # ── ② tau · ③ funnel ───────────────────────────────────────────────
    print("\n── tau 추정 · divergence ──", flush=True)
    pr = []
    for (tr, te), s in zip(pre, SEEDS):
        a = hier_bayes.posterior(tr, te, mode="pool", seed=0)
        b = hier_bayes.posterior(tr, te, mode="pool", seed=0, centered=True)
        pr.append(dict(seed=s, tau=a["tau"], lo=a["tau_q"][0], hi=a["tau_q"][1],
                       div_nc=a["divergences"], div_c=b["divergences"],
                       사건=a["n_events"]))
        print(f"  seed {s:<9} tau {a['tau']:.3f} [{a['tau_q'][0]:.3f}, {a['tau_q'][1]:.3f}]  "
              f"divergence  non-centered {a['divergences']:>3}  centered {b['divergences']:>3}", flush=True)
    P = pd.DataFrame(pr)

    # ── ④ prior sensitivity ────────────────────────────────────────────
    print("\n── tau 사전분포 sensitivity ──", flush=True)
    base = hier_bayes._TAU_SCALE
    sens = []
    for scale in (0.25, 0.5, 1.0, 2.0, 5.0):
        hier_bayes._TAU_SCALE = scale
        v, ta = [], []
        for (tr, te), s in zip(pre, SEEDS):
            v.append(evaluate(te, hier_bayes.fit_predict(tr, te, seed=0, mode="pool"))["rank_corr_true"])
            ta.append(hier_bayes.posterior(tr, te, mode="pool", seed=0)["tau"])
        sens.append(dict(사전분포=f"HalfNormal({scale})", 사전평균=scale * np.sqrt(2 / np.pi),
                         tau평균=float(np.mean(ta)), 순위상관=float(np.mean(v)), sd=float(np.std(v))))
        print(f"  HalfNormal({scale})  사전평균 {sens[-1]['사전평균']:.3f} → 사후 tau {sens[-1]['tau평균']:.3f}"
              f"   순위상관 {sens[-1]['순위상관']:+.3f}", flush=True)
    hier_bayes._TAU_SCALE = base
    S = pd.DataFrame(sens)

    # ── 문서 ───────────────────────────────────────────────────────────
    L = ["# 계층 베이즈 1차 — 8 seed", "",
         "`a0 = 3.0` 을 손으로 박는 자리를 `tau` 추정으로 바꿨다. 관측 모형은 셋 다 같다 —",
         "`k ~ Binomial(n, ih × κ_f)`. 학습 구간 κ 를 고정해 채점한다(온라인 갱신 없음).", "",
         "## ① 순위상관", "",
         "| 구성 | 순위상관 | ±sd | 최소 | 양수 seed | 초/seed |", "|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        L.append(f"| {r.구성} | {r.평균:+.3f} | {r.sd:.3f} | {r.최소:+.3f} | {r.양수}/8 | {r.초:.1f} |")

    L += ["", "## ② tau — shrinkage 세기를 추정한 값", "",
          f"사건 {int(P['사건'].median())}건 · 공장 12곳. 90% 구간은 사후분포 quantile.", "",
          "| seed | tau | 90% 구간 | divergence (non-centered) | divergence (centered) |",
          "|---|---|---|---|---|"]
    for _, r in P.iterrows():
        L.append(f"| {int(r.seed)} | {r.tau:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | {int(r.div_nc)} | {int(r.div_c)} |")
    L += ["", f"평균 tau **{P['tau'].mean():.3f}** · 구간 폭 중앙 **{(P['hi'] - P['lo']).median():.3f}**",
          f"· divergence 합계 non-centered **{int(P['div_nc'].sum())}** / centered **{int(P['div_c'].sum())}**"]

    L += ["", "## ④ tau 사전분포 sensitivity", "",
          "공장 12곳에서 variance component 가 사전분포에 얼마나 끌려가는가.", "",
          "| tau 사전분포 | 사전평균 | 사후 tau 평균 | 순위상관 | ±sd |", "|---|---|---|---|---|"]
    for _, r in S.iterrows():
        L.append(f"| {r.사전분포} | {r.사전평균:.3f} | {r.tau평균:.3f} | {r.순위상관:+.3f} | {r.sd:.3f} |")

    out = Path(__file__).resolve().parents[1] / "docs" / "_계층베이즈_1차.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
