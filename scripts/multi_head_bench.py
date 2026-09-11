"""multi-head 벤치 — 폐기 head 를 붙이면 무엇이 달라지나. 8 seed × 진척 {10%, 100%} (2026-09-11, [[모델_방향_결정]] §3 ②).

학습 프레임은 진척 p 로 잘라(채점 조건과 같게), 타깃은 완료 시점(`train_full`). 채점은 진척 p 까지만 본 테스트 오더.

재는 것 넷
  ① 유출 순위     two_stage · hier_bayes pool · hier_bayes multi   — 나빠지지 않아야 한다(완료 시점 동률 기대)
  ② 제조 품질 순위 분해식(factory_params) vs multi `internal = c_o (1+κ_f)`  — 진실 p 대비 (합성에서만 가능)
  ③ 폐기율 순위   ih(p) vs multi `scrap_rate = c_o s_f`               — **테스트 오더의 실측 S/N 대비**. 실데이터에서 되는 채점
  ④ 검출률 d      pool vs multi — 진실 대비

    python scripts/multi_head_bench.py     → docs/_multi_head.md
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from progress_scoring import truncate

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import evaluate, split_by_time
from f4ge_supplier_risk.features.build import FULL_COLS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import factory_params as fp
from f4ge_supplier_risk.models import hier_bayes, two_stage

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
PROGRESS = (0.10, 1.00)
warnings.filterwarnings("ignore")


def main() -> None:
    cfg = config.load(str(ROOT / "configs/generator.yaml"))
    rows = []
    for seed in SEEDS:
        c = copy.deepcopy(cfg); c["seed"] = seed
        data = build_dataset(c)
        df = build(data)
        tr, te = split_by_time(df)
        n_tr, n_te = len(tr), len(te)
        d_true = {f["factory_id"]: f["detection_rate"] for f in data["factories"]}
        sn = (te["obs_phys_scrap"] / te["obs_produced_counter"]).to_numpy(float)
        ok = np.isfinite(sn)
        P = fp.estimate(tr)
        for p in PROGRESS:
            dfp = build(truncate(data, p))
            trp, tep = dfp.iloc[:n_tr], dfp.iloc[n_tr:n_tr + n_te]
            ih = two_stage._stage_a(trp, tep, FULL_COLS)
            pool = hier_bayes.fit_predict(trp, tep, seed=0)
            pool_d = hier_bayes.posterior(trp, tep, seed=0)["detection_mean"]
            m = hier_bayes.fit_predict_multi(trp, tep, train_full=tr, seed=0)
            facs = sorted(d_true)
            rec = {
                "seed": seed, "progress": p,
                "esc_two_stage": evaluate(tep, two_stage.fit_predict(trp, tep, FULL_COLS))["rank_corr_true"],
                "esc_pool": evaluate(tep, pool)["rank_corr_true"],
                "esc_multi": evaluate(tep, m["escape"])["rank_corr_true"],
                "p_fp": spearmanr(fp.order_internal_rate(tep, P), te["true_internal_rate"]).statistic,
                "p_multi": spearmanr(m["internal"], te["true_internal_rate"]).statistic,
                "scrap_ih": spearmanr(ih[ok], sn[ok]).statistic,
                "scrap_multi": spearmanr(m["scrap_rate"][ok], sn[ok]).statistic,
                "d_pool": spearmanr([pool_d[f] for f in facs], [d_true[f] for f in facs]).statistic,
                "d_multi": spearmanr([m["detection"][f] for f in facs], [d_true[f] for f in facs]).statistic,
                "tau_c": m["tau_c"], "div": m["divergences"],
            }
            rows.append(rec)
            print(f"  seed {seed:>9} 진척 {p:.0%}  유출 ts {rec['esc_two_stage']:+.3f} pool {rec['esc_pool']:+.3f} multi {rec['esc_multi']:+.3f}"
                  f" | 제조 fp {rec['p_fp']:+.3f} multi {rec['p_multi']:+.3f} | 폐기 ih {rec['scrap_ih']:+.3f} multi {rec['scrap_multi']:+.3f}"
                  f" | d pool {rec['d_pool']:+.2f} multi {rec['d_multi']:+.2f} | tau_c {rec['tau_c']:.2f} div {rec['div']}", flush=True)

    T = pd.DataFrame(rows)
    L = [f"# multi-head — 폐기 head 를 붙이면 (8 seed · 진척 {' · '.join(f'{p:.0%}' for p in PROGRESS)})", "",
         "학습 프레임은 진척 p 로 잘라 채점 조건과 같게, 타깃(R·S)은 완료 시점. 순위상관은 Spearman, 8 seed 평균 ±sd.",
         "`multi` = `hier_bayes.fit_predict_multi` (유출 binomial + 보고 Poisson + 폐기 Poisson, gamma_f 공유). `pool` = 유출 head 만.", ""]
    for p, g in T.groupby("progress"):
        def cell(k):
            return f"{g[k].mean():+.3f} ±{g[k].std():.3f}"
        wins = lambda a, b: int((g[a] > g[b]).sum())
        L += [f"## 진척 {p:.0%}", "",
              "| 지표 | 기준 | multi | multi 가 이긴 seed |", "|---|---|---|---|",
              f"| ① 유출 순위 (진실) | two_stage {cell('esc_two_stage')} · pool {cell('esc_pool')} | {cell('esc_multi')} | vs pool {wins('esc_multi', 'esc_pool')}/{len(g)} |",
              f"| ② 제조 품질 p 순위 (진실) | 분해식 {cell('p_fp')} | {cell('p_multi')} | {wins('p_multi', 'p_fp')}/{len(g)} |",
              f"| ③ 폐기율 순위 (**실측 S/N**) | ih {cell('scrap_ih')} | {cell('scrap_multi')} | {wins('scrap_multi', 'scrap_ih')}/{len(g)} |",
              f"| ④ 검출률 d 순위 (진실, 12곳) | pool {cell('d_pool')} | {cell('d_multi')} | {wins('d_multi', 'd_pool')}/{len(g)} |",
              "", f"tau_c 평균 {g['tau_c'].mean():.3f} · divergence 합 {int(g['div'].sum())}", ""]
    L += ["⚠ 8 seed 라 표준오차 0.03~0.05. 그 안쪽은 판별되지 않는다.", "", "재현: `python scripts/multi_head_bench.py`"]
    out = ROOT / "docs" / "_multi_head.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"표 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
