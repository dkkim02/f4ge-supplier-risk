"""bias_capability_corr 이 **불일치 탐지 정밀도**를 바꾸나 — 8 seed (2026-09-10).

`scripts/sensitivity.py` 는 예측 순위상관과 gap 복원만 잰다. CTO 질문 22 의 주장
("나쁜 공장이 더 숨기면 불일치 탐지가 나쁜 오더까지 함께 잡는다")은 여기서만 확인된다.
계산은 `scripts/discrepancy_precision.py` 와 같고 corr 축만 걸어 돈다.

    python scripts/corr_axis_discrepancy.py     → docs/_상관축_불일치정밀도.md
"""
import copy, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import discrepancy
from f4ge_supplier_risk.prediction.run import _predict
from f4ge_supplier_risk.prediction.score import build_scores
from sensitivity import SEEDS, apply_value

cfg = config.load("configs/generator.yaml")
lines = ["| corr | review_needed 건수 | 실제 상위 20% 위험 | 편향 심한 공장 | 기저(편향) | 공장 정직도 ρ | 양수 seed |",
         "|---|---|---|---|---|---|---|"]
print(f"{'corr':>6} {'건수':>7} {'실제상위20%':>12} {'편향공장':>10} {'기저(편향)':>11} {'정직도ρ':>9} {'양수':>6}")
for v in (0.0, -0.3, -0.6, -0.85):
    cnt, top20, biased, base_b, hon = [], [], [], [], []
    for s in SEEDS:
        cf = apply_value(cfg, "bias_capability_corr", v); cf["seed"] = s
        data = build_dataset(cf)
        facs = {f["factory_id"]: f for f in data["factories"]}
        eff = {k: max(f["report_bias"], f["mes_input_bias"]) for k, f in facs.items()}
        cut = float(np.median(list(eff.values())))
        is_biased = {k: x < cut for k, x in eff.items()}
        df = build(data); tr, te = split_by_time(df)
        scored = build_scores(tr, _predict(tr, tr), discrepancy.fit_predict(tr, tr),
                              te, _predict(tr, te), discrepancy.fit_predict(tr, te),
                              discrepancy.reasons(tr, te))
        truth = te["true_escape_rate"].to_numpy()
        risky = truth >= np.quantile(truth, 0.8)
        b = np.array([is_biased[f] for f in te["factory_id"].to_numpy()])
        m = scored["recommended_action"].to_numpy() == "review_needed"
        cnt.append(int(m.sum())); base_b.append(b.mean())
        top20.append(risky[m].mean() if m.any() else np.nan)
        biased.append(b[m].mean() if m.any() else np.nan)
        trust = discrepancy.factory_trust(scored)
        if len(trust) >= 3:
            hon.append(spearmanr(trust["discrepancy_mean"], [-eff[i] for i in trust.index]).statistic)
    h = np.array(hon)
    print(f"{v:>6.2f} {np.mean(cnt):>7.1f} {np.nanmean(top20):>11.1%} {np.nanmean(biased):>9.1%} "
          f"{np.mean(base_b):>10.1%} {h.mean():>+9.3f} {int((h>0).sum()):>4}/{len(h)}", flush=True)
    mark = " ←현재" if v == 0.0 else ""
    lines.append(f"| **{v}**{mark} | {np.mean(cnt):.1f} | {np.nanmean(top20):.1%} | {np.nanmean(biased):.1%} "
                 f"| {np.mean(base_b):.1%} | {h.mean():+.3f} | {int((h>0).sum())}/{len(h)} |")

out = Path(__file__).resolve().parents[1] / "docs" / "_상관축_불일치정밀도.md"
out.write_text("# bias_capability_corr × 불일치 탐지 — 8 seed\n\n"
               "음수 = 나쁜 공장일수록 더 축소 보고. CTO 질문 22 판정용.\n\n" + "\n".join(lines), encoding="utf-8")
print(f"\n표 저장: {out.relative_to(out.parents[2])}")
