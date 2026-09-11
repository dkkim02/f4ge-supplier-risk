"""κ → 검출률 d 재파라미터화 검증 (2026-09-11, [[모델_방향_결정]] §3 ①).

`d = 1/(1+κ)` 가 표기 변환이라는 주장을 8 seed 로 잰다. 재는 것 셋.

  ① 모델 셋의 d 가 **진실 검출률**과 얼마나 맞나 — Spearman · 절대오차 중앙값
  ② 현행 분해식 d(`factory_params.estimate`, 계약 `factory_detection_rate` 의 계산처)와 얼마나 다른가
  ③ `hier_bayes.posterior()` 의 d 90% 구간이 진실을 얼마나 덮나 — 검출률 칸에 구간을 넣을 재료

산출: docs/_검출률_재파라미터화.md
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import FULL_COLS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import deep_kappa, hier_bayes, two_stage
from f4ge_supplier_risk.models import factory_params as fp

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
warnings.filterwarnings("ignore")


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    rows, cover = [], []
    for seed in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        data = build_dataset(c)
        df = build(data)
        tr, te = split_by_time(df)
        facs = sorted(f["factory_id"] for f in data["factories"])
        d_true = pd.Series({f["factory_id"]: f["detection_rate"] for f in data["factories"]}).reindex(facs)

        P = fp.estimate(tr)
        est = {
            "분해식 (현행)": P["d"].reindex(facs),
            "two_stage": pd.Series(two_stage.explain(tr, te, FULL_COLS)["detection"]).reindex(facs),
        }
        if deep_kappa.available():
            est["deep_kappa hier"] = pd.Series(deep_kappa.explain(tr, te, seed=seed, mode="hier")["detection"]).reindex(facs)
        if hier_bayes.available():
            post = hier_bayes.posterior(tr, te, mode="pool", seed=0)
            est["hier_bayes pool"] = pd.Series(post["detection_mean"]).reindex(facs)
            lo = pd.Series(post["detection_q05"]).reindex(facs)
            hi = pd.Series(post["detection_q95"]).reindex(facs)
            cover.append({
                "seed": seed,
                "coverage": float(((lo <= d_true) & (d_true <= hi)).mean()),
                "width": float((hi - lo).median()),
                "divergences": post["divergences"],
            })

        rec = {"seed": seed}
        for name, d in est.items():
            rec[f"rho|{name}"] = spearmanr(d, d_true).statistic
            rec[f"mae|{name}"] = float((d - d_true).abs().median())
            rec[f"bias|{name}"] = float((d - d_true).median())
            if name != "분해식 (현행)":
                rec[f"vs_fp|{name}"] = float((d - est["분해식 (현행)"]).abs().median())
        rows.append(rec)
        print(f"  seed {seed:>9}  " + "  ".join(f"{n} ρ{rec[f'rho|{n}']:+.2f}/±{rec[f'mae|{n}']:.4f}" for n in est), flush=True)

    t = pd.DataFrame(rows).set_index("seed")
    names = [c.split("|", 1)[1] for c in t.columns if c.startswith("rho|")]
    n = len(SEEDS)
    L = [f"# κ → 검출률 d 재파라미터화 — {n} seed · 공장 12곳", "",
         "`d = 1/(1+κ)`. 표기 변환이라는 주장을 진실 검출률(`factory.detection_rate`)과 대조해 잰다.",
         ("학습 구간에서 적합한 공장별 상수 하나씩. `분해식 (현행)` = `factory_params.estimate` 의 `d` "
          "(계약 `factory_detection_rate` 의 지금 계산처)."), "",
         "## ① 진실 대비", "",
         "| 계산처 | Spearman ρ | ±sd | 양수 seed | 절대오차 중앙 | 편향 중앙 (추정 − 진실) |", "|---|---|---|---|---|---|"]
    for k in names:
        L.append(f"| {k} | {t[f'rho|{k}'].mean():+.3f} | {t[f'rho|{k}'].std():.3f} | {int((t[f'rho|{k}'] > 0).sum())}/{n} "
                 f"| {t[f'mae|{k}'].mean():.4f} | {t[f'bias|{k}'].mean():+.4f} |")
    L += ["", "## ② 분해식 d 와의 차이 — 절대오차 중앙값 (8 seed 평균)", "", "| 계산처 | vs 분해식 |", "|---|---|"]
    for k in names:
        if k != "분해식 (현행)":
            L.append(f"| {k} | {t[f'vs_fp|{k}'].mean():.4f} |")
    if cover:
        C = pd.DataFrame(cover).set_index("seed")
        L += ["", "## ③ hier_bayes d 90% 구간", "",
              f"진실 포함률 **{C['coverage'].mean():.3f}** (명목 0.90) · 구간 폭 중앙 **{C['width'].mean():.4f}** · divergence 합 {int(C['divergences'].sum())}", "",
              "| seed | 포함률 | 폭 중앙 | divergence |", "|---|---|---|---|"]
        for s, r in C.iterrows():
            L.append(f"| {s} | {r.coverage:.3f} | {r.width:.4f} | {int(r.divergences)} |")
    L += ["", "⚠ 8 seed 라 ρ 의 표준오차는 0.03~0.05 다. 그 안쪽 차이는 판별되지 않는다.",
          "", "재현: `python scripts/detection_reparam.py`"]
    out = Path(__file__).resolve().parents[1] / "docs" / "_검출률_재파라미터화.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"표 저장: {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
