"""MES 설치 공장 수 sweep — 불일치 탐지(핵심 상품)는 몇 곳부터 성립하나.

09-08 재측정에서 보고 공장 3곳으로는 현장 방문 정밀도가 기저 아래로 내려갔다(docs/불일치탐지.md §4-1).
"증거 → 보고" 관계를 공장 전체에 걸쳐 적합하고 잔차를 공장 편향으로 읽는 방법이라,
공장 수준의 표본 수가 곧 통계력이다. 유형 a 를 3 → 6 → 9 → 12곳으로 늘리며 잰다.

    python scripts/coverage_sweep.py 3 6              # 수준을 인자로. 없으면 넷 다
    python scripts/coverage_sweep.py 3 12 --no-floor   # MES 하한을 끄고(0.02) 잰다 — 표본 수와 하한의 효과를 가른다
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
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
# a(MES 연동)가 늘면 b(설비 신호만)가 준다. Cell 은 12곳 전부.
# 09-08 저녁: CellOS·FactoryOS 는 12곳 전부 — 나머지는 전부 b(설비 신호만). c 는 없다.
LEVELS = {3: (3, 9, 0), 6: (6, 6, 0), 9: (9, 3, 0), 12: (12, 0, 0)}


def run_level(cfg: dict, n_a: int, no_floor: bool = False) -> dict[str, list[float]]:
    a, b, c = LEVELS[n_a]
    out: dict[str, list[float]] = {k: [] for k in (
        "visit_n", "visit_prec", "visit_base", "tighten_prec", "honesty", "rank_all", "rank_a", "call_prec",
    )}
    for seed in SEEDS:
        cf = copy.deepcopy(cfg)
        cf["seed"] = seed
        cf["factory_types"] = {"a": a, "b": b, "c": c}
        if no_floor:
            cf["assumptions"]["mes_input_bias_range"] = [0.02, 0.02]  # 09-08 오전 이전의 하한
        data = build_dataset(cf)
        facs = {f["factory_id"]: f for f in data["factories"]}
        rep_facs = [k for k, f in facs.items() if f["has_mes"]]
        eff = {k: max(facs[k]["report_bias"], facs[k]["mes_input_bias"]) for k in rep_facs}
        cut = float(np.median(list(eff.values())))
        biased = {k: v < cut for k, v in eff.items()}  # 보고 공장 안에서 중앙값 아래 = 편향 심한 쪽

        df = build(data)
        tr, te = split_by_time(df)
        pred_tr, pred = _predict(tr, tr), _predict(tr, te)
        scored = build_scores(
            tr, pred_tr, discrepancy.fit_predict(tr, tr),
            te, pred, discrepancy.fit_predict(tr, te), discrepancy.reasons(tr, te),
        )
        truth = te["true_escape_rate"].to_numpy()
        risky = truth >= np.quantile(truth, 0.8)
        fac = te["factory_id"].to_numpy()
        is_a = np.array([f in eff for f in fac])
        is_biased = np.array([biased.get(f, False) for f in fac])
        act = scored["recommended_action"].to_numpy()

        v = act == "site_visit"
        out["visit_n"].append(int(v.sum()))
        out["visit_prec"].append(is_biased[v].mean() if v.any() else np.nan)
        out["visit_base"].append(is_biased[is_a].mean())
        t = act == "tighten_inspection"
        out["tighten_prec"].append(risky[t].mean() if t.any() else np.nan)
        cl = act == "call"
        out["call_prec"].append(is_biased[cl].mean() if cl.any() else np.nan)
        trust = discrepancy.factory_trust(scored)
        trust = trust[trust.index.isin(rep_facs)]
        out["honesty"].append(
            spearmanr(trust["discrepancy_mean"], [-eff[i] for i in trust.index]).statistic
            if len(trust) >= 3 else np.nan
        )
        out["rank_all"].append(spearmanr(pred, truth).statistic)
        out["rank_a"].append(spearmanr(pred[is_a], truth[is_a]).statistic if is_a.sum() > 5 else np.nan)
        print(f"  a={n_a:2d} seed {seed:>9}  방문 {out['visit_n'][-1]:3d}건 정밀 {out['visit_prec'][-1]:.0%}  "
              f"정직도 r {out['honesty'][-1]:+.2f}", flush=True)
    return out


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    no_floor = "--no-floor" in sys.argv
    levels = [int(x) for x in sys.argv[1:] if x.isdigit()] or list(LEVELS)
    rows = []
    for n_a in levels:
        r = run_level(cfg, n_a, no_floor)
        m = {k: float(np.nanmean(v)) for k, v in r.items()}
        h = np.array(r["honesty"], float)
        pos = f"{int(np.nansum(h > 0))}/{int((~np.isnan(h)).sum())}"
        rows.append(
            f"| {n_a} ({LEVELS[n_a][0]}/{LEVELS[n_a][1]}/{LEVELS[n_a][2]}) | {m['visit_n']:.1f} | "
            f"**{m['visit_prec']:.1%}** | {m['visit_base']:.1%} | {m['honesty']:+.3f} ({pos}) | "
            f"{m['tighten_prec']:.1%} | {m['call_prec']:.1%} | {m['rank_all']:+.3f} | {m['rank_a']:+.3f} |"
        )
        print(rows[-1], flush=True)
    hdr = ("| MES 공장 수 (a/b/c) | 현장 방문 건수 | 방문 정밀도 (편향 심한 공장) | 기저 (보고 공장 내) | "
           "공장 정직도 Spearman (양수 seed) | 검사 강화 정밀도 | 전화 정밀도 | 전체 순위상관 | 유형 a 순위상관 |")
    table = "\n".join([hdr, "|---|---|---|---|---|---|---|---|---|", *rows])
    print("\n" + table)
    tag = "_".join(map(str, levels)) + ("_nofloor" if no_floor else "")
    out = Path(__file__).resolve().parents[1] / "docs" / f"_coverage_sweep_{tag}.md"
    out.write_text(table, encoding="utf-8")
    print(f"표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
