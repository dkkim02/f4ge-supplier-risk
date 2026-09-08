"""권고를 `review_needed` 하나로 합친 뒤에도 두 상품을 따로 재는 법.

09-08 에 권고 코드 3종(call · tighten_inspection · site_visit)을 **`review_needed` 하나로 합쳤다.**
무엇을 할지는 담당자가 정하고 계약은 근거만 준다는 취지다. 그런데 지표는 코드에 묶여 있었다 —
「검사 강화가 실제 나쁜 오더를 잡는 비율」과 「현장 방문이 편향 공장을 잡는 비율」은 서로 다른 모집단이고,
합치면 두 집단의 평균이 되어 어느 쪽도 재지 못한다.

그래서 `action_trigger`(risk / discrepancy_low_trust / discrepancy)를 계약에 남기고,
정밀도를 축별로 낸다. 코드는 하나, 측정은 둘이다.

    python scripts/trigger_precision.py          # a=3(현재) · a=12
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
TRIGGERS = ("risk", "discrepancy_low_trust", "discrepancy")
LABEL = {"risk": "예측 위험 (구 검사 강화)",
         "discrepancy_low_trust": "불일치+저신뢰 (구 현장 방문)",
         "discrepancy": "불일치 (구 전화)"}


def run(cfg: dict, n_a: int) -> dict:
    acc = {t: {"n": [], "risky": [], "biased": []} for t in TRIGGERS}
    base = {"risky": [], "biased": []}
    for seed in SEEDS:
        cf = copy.deepcopy(cfg)
        cf["seed"] = seed
        cf["factory_types"] = {"a": n_a, "b": 12 - n_a, "c": 0}
        data = build_dataset(cf)
        facs = {f["factory_id"]: f for f in data["factories"]}
        eff = {k: max(f["report_bias"], f["mes_input_bias"]) for k, f in facs.items() if f["has_mes"]}
        cut = float(np.median(list(eff.values())))
        biased_fac = {k: v < cut for k, v in eff.items()}

        df = build(data)
        tr, te = split_by_time(df)
        pred_tr, pred = _predict(tr, tr), _predict(tr, te)
        scored = build_scores(tr, pred_tr, discrepancy.fit_predict(tr, tr),
                              te, pred, discrepancy.fit_predict(tr, te),
                              discrepancy.reasons(tr, te))
        truth = te["true_escape_rate"].to_numpy()
        risky = truth >= np.quantile(truth, 0.8)
        fac = te["factory_id"].to_numpy()
        is_a = np.array([f in eff for f in fac])
        is_biased = np.array([biased_fac.get(f, False) for f in fac])
        trig = scored["action_trigger"].to_numpy()

        base["risky"].append(risky.mean())
        base["biased"].append(is_biased[is_a].mean() if is_a.any() else np.nan)
        for t in TRIGGERS:
            m = trig == t
            acc[t]["n"].append(int(m.sum()))
            acc[t]["risky"].append(risky[m].mean() if m.any() else np.nan)
            acc[t]["biased"].append(is_biased[m].mean() if m.any() else np.nan)
    return acc, base


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    blocks = []
    for n_a in (3, 12):
        acc, base = run(cfg, n_a)
        rows = [f"**MES {n_a}곳** — 기저: 실제 위험 상위 20% = {np.nanmean(base['risky']):.1%} · "
                f"편향 심한 공장 = {np.nanmean(base['biased']):.1%}\n",
                "| 걸린 축 | 건수 | 실제 나쁜 오더 적중 | 편향 심한 공장 적중 |", "|---|---|---|---|"]
        for t in TRIGGERS:
            a = acc[t]
            rows.append("| %s | %.1f | **%.1f%%** | **%.1f%%** |" % (
                LABEL[t], np.mean(a["n"]), 100 * np.nanmean(a["risky"]), 100 * np.nanmean(a["biased"])))
        blocks.append("\n".join(rows))
        print(blocks[-1] + "\n", flush=True)
    out = Path(__file__).resolve().parents[1] / "docs" / "_축별_정밀도_표.md"
    out.write_text("# 권고 축별 정밀도 — 8 seed\n\n권고 코드는 `review_needed` 하나이고, "
                   "`action_trigger` 로 축을 갈라 잰다.\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    print("→", out)


if __name__ == "__main__":
    main()
