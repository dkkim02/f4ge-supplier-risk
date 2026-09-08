"""검토 필요(review_needed) 권고의 정밀도 — 8 seed.

채점 파이프라인(`prediction.run._predict`)과 같은 예측을 쓴다.
09-08 저녁부터 권고 코드는 `review_needed` 하나다. 네 소스가 12곳 전부에서 오므로 모집단은 전체 오더 하나다 —
review_needed 가 실제 상위 20% 위험을 얼마나 잡는가 · 편향 심한 공장을 얼마나 잡는가. "편향 심한 공장" = 실효 편향 max(report_bias, mes_input_bias) 이
12곳 중앙값 아래인 공장. 공장 정직도 Spearman 은 보고 있는 공장 3곳 위의 값이라 방향만 본다.
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import discrepancy
from f4ge_supplier_risk.prediction.run import _predict
from f4ge_supplier_risk.prediction.score import build_scores

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
ACTIONS = ("review_needed", "none")


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    cnt = defaultdict(list)
    top20 = defaultdict(list)
    biased = defaultdict(list)
    base_top20, base_biased, base_biased_a, honesty = [], [], [], []
    cnt_a, biased_a = [], []
    n_test = 0  # 보고 있는 오더(유형 a) 안의 review_needed

    for seed in SEEDS:
        cf = copy.deepcopy(cfg)
        cf["seed"] = seed
        data = build_dataset(cf)
        facs = {f["factory_id"]: f for f in data["factories"]}
        eff = {k: max(f["report_bias"], f["mes_input_bias"]) for k, f in facs.items()}
        cut = float(np.median(list(eff.values())))
        is_biased = {k: v < cut for k, v in eff.items()}

        df = build(data)
        tr, te = split_by_time(df)
        scored = build_scores(
            tr,
            _predict(tr, tr),
            discrepancy.fit_predict(tr, tr),
            te,
            _predict(tr, te),
            discrepancy.fit_predict(tr, te),
            discrepancy.reasons(tr, te),
        )
        truth = te["true_escape_rate"].to_numpy()
        risky = truth >= np.quantile(truth, 0.8)
        fac = te["factory_id"].to_numpy()
        b = np.array([is_biased[f] for f in fac])
        act = scored["recommended_action"].to_numpy()
        n_test = len(te)

        base_top20.append(risky.mean())
        base_biased.append(b.mean())
        base_biased_a.append(np.nan)
        for a in ACTIONS:
            m = act == a
            cnt[a].append(int(m.sum()))
            top20[a].append(risky[m].mean() if m.any() else np.nan)
            biased[a].append(b[m].mean() if m.any() else np.nan)

        trust = discrepancy.factory_trust(scored)
        if len(trust) >= 3:
            honesty.append(
                spearmanr(trust["discrepancy_mean"], [-eff[i] for i in trust.index]).statistic
            )
        print(f"  seed {seed:>9}  " + "  ".join(f"{a} {cnt[a][-1]:3d}" for a in ACTIONS), flush=True)

    def f(v):
        v = np.array(v, float)
        return f"{np.nanmean(v):.1%}"

    print(f"\n{len(SEEDS)} seed 평균, 테스트 {n_test}건")
    print(f"{'권고':20s} {'건수':>6s} {'실제 상위20% 위험':>18s} {'편향 심한 공장':>16s}")
    lines = ["| 권고 | 건수 | 실제 상위 20% 위험 | 편향 심한 공장 |", "|---|---|---|---|"]
    for a in ACTIONS:
        print(f"{a:20s} {np.mean(cnt[a]):6.1f} {f(top20[a]):>18s} {f(biased[a]):>16s}")
        lines.append(f"| {a} | {np.mean(cnt[a]):.1f} | {f(top20[a])} | {f(biased[a])} |")
    print(f"{'— 기저율 (전체)':20s} {n_test:6d} {f(base_top20):>18s} {f(base_biased):>16s}")
    lines.append(f"| — 기저율 전체 | {n_test} | {f(base_top20)} | {f(base_biased)} |")
    h = np.array(honesty)
    print(f"\n공장 정직도 Spearman (보고 있는 공장 {len(h) and 3}곳): 평균 {h.mean():+.3f} · 양수 {int((h>0).sum())}/{len(h)} · 범위 {h.min():+.2f}~{h.max():+.2f}")
    lines.append(f"\n공장 정직도 Spearman (3곳): 평균 {h.mean():+.3f} · 양수 {int((h>0).sum())}/{len(h)}")
    out = Path(__file__).resolve().parents[1] / "docs" / "_불일치_정밀도_표.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
