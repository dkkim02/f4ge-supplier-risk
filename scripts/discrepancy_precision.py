"""불일치 탐지 권고의 정밀도 — 09-07 값(검사강화 74.7% · 현장방문 95.6%)을 09-08 생성기에서 다시 잰다.

채점 파이프라인(`prediction.run._predict`)과 같은 예측을 쓴다.
09-08 재설계로 보고가 오는 공장이 3곳(유형 a)이 됐고, 같은 날 오후 규칙으로 보고 없는 오더에는
site_visit·call 이 걸리지 않는다. 그래서 두 권고의 모집단이 다르다 —
  검사 강화(tighten_inspection)  전체 오더에서 예측 위험 상위 10%
  현장 방문(site_visit)           보고 있는 오더(유형 a) 중 못 믿을 공장 × 큰 불일치
기저율을 전체와 유형 a 두 가지로 같이 낸다. "편향 심한 공장" = 실효 편향 max(report_bias, mes_input_bias) 이
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
ACTIONS = ("site_visit", "tighten_inspection", "call", "none")


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    cnt = defaultdict(list)
    top20 = defaultdict(list)
    biased = defaultdict(list)
    base_top20, base_biased, base_biased_a, honesty = [], [], [], []

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
        is_a = np.array([facs[f]["factory_type"] == "a" for f in fac])
        act = scored["recommended_action"].to_numpy()

        base_top20.append(risky.mean())
        base_biased.append(b.mean())
        base_biased_a.append(b[is_a].mean() if is_a.any() else np.nan)
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

    print(f"\n{len(SEEDS)} seed 평균, 테스트 270건")
    print(f"{'권고':20s} {'건수':>6s} {'실제 상위20% 위험':>18s} {'편향 심한 공장':>16s}")
    lines = ["| 권고 | 건수 | 실제 상위 20% 위험 | 편향 심한 공장 |", "|---|---|---|---|"]
    for a in ACTIONS:
        print(f"{a:20s} {np.mean(cnt[a]):6.1f} {f(top20[a]):>18s} {f(biased[a]):>16s}")
        lines.append(f"| {a} | {np.mean(cnt[a]):.1f} | {f(top20[a])} | {f(biased[a])} |")
    print(f"{'— 기저율 (전체)':20s} {270:6d} {f(base_top20):>18s} {f(base_biased):>16s}")
    print(f"{'— 기저율 (유형 a)':20s} {'':6s} {'':>18s} {f(base_biased_a):>16s}")
    lines.append(f"| — 기저율 전체 | 270 | {f(base_top20)} | {f(base_biased)} |")
    lines.append(f"| — 기저율 유형 a | | | {f(base_biased_a)} |")
    h = np.array(honesty)
    print(f"\n공장 정직도 Spearman (보고 있는 공장 {len(h) and 3}곳): 평균 {h.mean():+.3f} · 양수 {int((h>0).sum())}/{len(h)} · 범위 {h.min():+.2f}~{h.max():+.2f}")
    lines.append(f"\n공장 정직도 Spearman (3곳): 평균 {h.mean():+.3f} · 양수 {int((h>0).sum())}/{len(h)}")
    out = Path(__file__).resolve().parents[1] / "docs" / "_불일치_정밀도_표.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
