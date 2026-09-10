"""모델 벤치 — 유출 불량 예측기 후보를 같은 데이터·같은 분할에서 8 seed 로 나란히 잰다 (2026-09-08 저녁).

사용자: "딥러닝 써도 되니 성능을 최대한 끌어올려라. 24시간 배치라 비용 제약 없다."
후보:
  two_stage   2단 생성모형 (보고 × κ). 지금 채점 파이프라인
  hgb         HistGradientBoosting, poisson 손실 · exposure = 표본 수 (k/n 을 n 가중으로)
  mlp         torch MLP, binomial NLL (log-rate 출력, 표본 n·불합격 k 그대로). torch 없으면 건너뜀
  ens         세 후보 log 평균
지표: 진짜 escape 순위상관(Spearman) · binomial log loss · review_needed 정밀도는 별도 스크립트.
함께: 공장 파라미터 분해(p·d·b)가 진실을 얼마나 따라가는가.

    python scripts/model_bench.py            → docs/_모델_벤치.md
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import evaluate, split_by_time
from f4ge_supplier_risk.features.build import FULL_COLS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import factory_params as fp
from f4ge_supplier_risk.models import hgb, two_stage

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
warnings.filterwarnings("ignore")

try:
    from f4ge_supplier_risk.models import mlp

    HAS_MLP = mlp.available()
except ImportError:  # pragma: no cover
    HAS_MLP = False


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    rows = []
    for seed in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        data = build_dataset(c)
        df = build(data)
        tr, te = split_by_time(df)
        facs = {f["factory_id"]: f for f in data["factories"]}

        preds = {"two_stage": two_stage.fit_predict(tr, te, FULL_COLS), "hgb": hgb.fit_predict(tr, te, FULL_COLS)}
        if HAS_MLP:
            preds["mlp"] = mlp.fit_predict(tr, te, FULL_COLS, seed=seed)
        preds["ens"] = np.exp(np.mean([np.log(np.clip(v, 1e-7, 1)) for v in preds.values()], axis=0))

        rec = {"seed": seed}
        for k, v in preds.items():
            m = evaluate(te, v)
            rec[f"rank_{k}"] = m["rank_corr_true"]
            rec[f"ll_{k}"] = m["binom_logloss"]

        P = fp.estimate(tr)
        d_true = [facs[i]["detection_rate"] for i in P.index]
        p_true_f = df.groupby("factory_id")["true_internal_rate"].mean().reindex(P.index)
        p_ord = fp.order_internal_rate(te, P)
        rec["d_rho"] = spearmanr(P["d"], d_true).statistic
        rec["pf_rho"] = spearmanr(P["p"], p_true_f).statistic
        rec["p_order_rho"] = spearmanr(p_ord, te["true_internal_rate"]).statistic
        rec["esc_decomp_rho"] = spearmanr(fp.order_escape_rate(te, P, p_ord), te["true_escape_rate"]).statistic
        rows.append(rec)
        print(f"  seed {seed:>9}  " + "  ".join(f"{k} {rec[f'rank_{k}']:+.3f}" for k in preds) +
              f"  | p_order {rec['p_order_rho']:+.2f} d {rec['d_rho']:+.2f}", flush=True)

    t = pd.DataFrame(rows).set_index("seed")
    mean, std = t.mean(), t.std()
    names = [c.removeprefix("rank_") for c in t.columns if c.startswith("rank_")]
    best = max(names, key=lambda k: mean[f"rank_{k}"])
    wins = {k: int((t[f"rank_{k}"] >= t[[f"rank_{j}" for j in names]].max(axis=1) - 1e-12).sum()) for k in names}
    lines = [f"# 모델 벤치 — {len(SEEDS)} seed · 테스트 {len(te)}건 · 네 소스 전부 · 1공장 1오더 · 330건", "",
             "| 후보 | 진짜 escape 순위상관 | ±sd | binomial log loss | 1위 seed 수 |", "|---|---|---|---|---|"]
    for k in names:
        lines.append(f"| {'**' + k + '**' if k == best else k} | {mean[f'rank_{k}']:+.3f} | {std[f'rank_{k}']:.3f} | {mean[f'll_{k}']:.5f} | {wins[k]}/{len(SEEDS)} |")
    lines += ["", "## 공장 파라미터 분해 (제조 품질 p · 검수 품질 d) — 진실과의 Spearman", "",
              "| 값 | 평균 | ±sd | 양수 seed |", "|---|---|---|---|"]
    for k, label in (("p_order_rho", "오더별 내부 불량률 p (테스트 오더)"), ("pf_rho", "공장 내부 불량률 p_f (12곳)"),
                     ("d_rho", "검출률 d (12곳)"),  # 보고 정직도 b 행은 09-10 삭제 — 편향 통제로 상수가 되어 Spearman 이 nan
                     ("esc_decomp_rho", "분해로 낸 escape = p·(1−d) 순위상관")):
        lines.append(f"| {label} | {mean[k]:+.3f} | {std[k]:.3f} | {int((t[k] > 0).sum())}/{len(SEEDS)} |")
    lines += ["", f"MLP {'포함' if HAS_MLP else '제외 (torch 없음)'} · 후보 정의는 scripts/model_bench.py 머리말."]
    out = Path(__file__).resolve().parents[1] / "docs" / "_모델_벤치.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
