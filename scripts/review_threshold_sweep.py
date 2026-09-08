"""검토 필요(review_needed) 임계 sweep — 8 seed (2026-09-08 저녁).

네 소스가 12곳 전부에서 오면서 불일치 임계(학습 분위 0.75)가 전 오더에 걸려 100건 중 35건이 검토 필요가 됐다.
세 임계(예측 위험 분위 · 오더 불일치 분위 · 공장 신뢰 게이트)를 조합해 다음을 잰다:
  flagged      100건 중 검토 필요 건수
  prec_risky   검토 필요 중 실제 상위 20% 위험 비율 (기저 20%)
  recall_risky 실제 상위 20% 위험 중 검토 필요로 잡힌 비율
  prec_biased  불일치 축으로 걸린 것 중 편향 심한 공장 비율 (기저 ≈ 50%)
  useful       검토 필요 중 (실제 위험 또는 편향 공장) 비율 — 헛걸음 아닌 비율

    python scripts/review_threshold_sweep.py     → docs/_검토임계_sweep.md
"""

from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import discrepancy
from f4ge_supplier_risk.prediction import score
from f4ge_supplier_risk.prediction.run import _predict

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
GRID = {
    "risk": (0.90,),                       # 예측 위험 분위 — 위험 등급 high 컷과 같은 선을 유지한다
    "call": (0.75, 0.85, 0.90, None),      # 오더 불일치 단독 임계. None = 이 축 제거
    "visit": (0.80, 0.90, 0.95),           # 신뢰 낮은 공장 × 오더 불일치 임계
    "trust": (0.75, 0.85),                 # 공장 신뢰 게이트 — 불일치 평균 상위 몇 % 공장을 "못 믿는다" 로 보나
}


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    per_seed = []
    for seed in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        data = build_dataset(c)
        df = build(data)
        tr, te = split_by_time(df)
        facs = {f["factory_id"]: f for f in data["factories"]}
        eff = {k: max(f["report_bias"], f["mes_input_bias"]) for k, f in facs.items()}
        cut = float(np.median(list(eff.values())))
        biased = np.array([eff[f] < cut for f in te["factory_id"]])
        truth = te["true_escape_rate"].to_numpy()
        risky = truth >= np.quantile(truth, 0.8)
        pre = dict(tr=tr, te=te, pred_tr=_predict(tr, tr), pred=_predict(tr, te),
                   disc_tr=discrepancy.fit_predict(tr, tr), disc=discrepancy.fit_predict(tr, te),
                   reasons=discrepancy.reasons(tr, te), biased=biased, risky=risky)
        per_seed.append(pre)
        print(f"  seed {seed} 준비", flush=True)

    rows = []
    for risk_q, call_q, visit_q, trust_q in itertools.product(*GRID.values()):
        score._RISK_TIGHTEN_Q = risk_q
        score._DISC_CALL_Q = call_q  # None = 오더 단독 불일치 축 제거
        score._DISC_VISIT_Q = visit_q
        score._TRUST_GATE_Q = trust_q
        m = {"flagged": [], "prec_risky": [], "recall_risky": [], "prec_biased": [], "useful": []}
        for p in per_seed:
            s = score.build_scores(p["tr"], p["pred_tr"], p["disc_tr"], p["te"], p["pred"], p["disc"], p["reasons"])
            act = s["recommended_action"].to_numpy() == "review_needed"
            by_disc = act & (s["action_reason"].to_numpy() != "예측 위험이 상위 10%")
            m["flagged"].append(act.sum() / len(act) * 100)
            m["prec_risky"].append(p["risky"][act].mean() if act.any() else np.nan)
            m["recall_risky"].append(act[p["risky"]].mean() if p["risky"].any() else np.nan)
            m["prec_biased"].append(p["biased"][by_disc].mean() if by_disc.any() else np.nan)
            m["useful"].append((p["risky"] | p["biased"])[act].mean() if act.any() else np.nan)
        rows.append({"risk": risk_q, "call": call_q, "visit": visit_q, "trust": trust_q,
                     **{k: float(np.nanmean(v)) for k, v in m.items()}})
    t = pd.DataFrame(rows)
    t["call"] = t["call"].fillna(0).map(lambda v: "—" if v == 0 else f"{v:.2f}")
    t = t.sort_values(["flagged"]).reset_index(drop=True)

    lines = [f"# 검토 필요 임계 sweep — {len(SEEDS)} seed · 테스트 100건 · 네 소스 · 1공장 1오더", "",
             "기저: 실제 상위 20% 위험 20% · 편향 심한 공장 ≈ 50%. 현재값 = risk 0.90 · call 0.75 · visit 0.80 · trust 0.75.", "",
             "| risk | call | visit | trust | 검토 필요 /100 | 실제 위험 정밀도 | 실제 위험 재현율 | 불일치축의 편향공장 정밀도 | 헛걸음 아닌 비율 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r.risk:.2f} | {r.call} | {r.visit:.2f} | {r.trust:.2f} | {r.flagged:.1f} | {r.prec_risky:.1%} | {r.recall_risky:.1%} | "
                     f"{'—' if np.isnan(r.prec_biased) else f'{r.prec_biased:.1%}'} | {r.useful:.1%} |")
    out = Path(__file__).resolve().parents[1] / "docs" / "_검토임계_sweep.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
