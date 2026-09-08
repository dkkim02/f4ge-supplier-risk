"""폐기 비중 가정(s̄) 민감도 — 공장 파라미터 분해(제조 품질 p · 검수 품질 d · 보고 정직도 b)가 s̄ 에 얼마나 흔들리나 (2026-09-08 저녁).

s̄ = 잡은 불량 중 폐기 비중. 생성기 진실은 오더마다 0.03~0.20 균등(평균 0.115)이고 모델은 상수 하나를 쓴다.
대수적으로 b = R·s̄/S 는 공장 전부에 같은 배수라 **순위는 s̄ 에 불변**이고, p·d 는 유출 항(e) 때문에 순위가 조금 움직인다.
그래서 두 가지를 따로 잰다 — ① 순위(진실과의 Spearman) ② 눈금(추정/진실 비의 중앙값).

    python scripts/scrap_share_sensitivity.py     → docs/_폐기비중_민감도.md
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import factory_params as fp

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
SHARES = (0.05, 0.08, 0.115, 0.15, 0.20, 0.30)


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    pre = []
    for seed in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        data = build_dataset(c)
        df = build(data)
        tr, te = split_by_time(df)
        facs = {f["factory_id"]: f for f in data["factories"]}
        pre.append(dict(tr=tr, te=te, facs=facs, p_true_f=df.groupby("factory_id")["true_internal_rate"].mean()))
        print(f"  seed {seed} 준비", flush=True)

    rows = []
    for s_bar in SHARES:
        m = {k: [] for k in ("p_order_rho", "pf_rho", "d_rho", "b_rho", "p_ratio", "d_gap", "b_ratio")}
        for p in pre:
            P = fp.estimate(p["tr"], scrap_share=s_bar)
            facs = p["facs"]
            d_true = np.array([facs[i]["detection_rate"] for i in P.index])
            b_true = np.array([max(facs[i]["report_bias"], facs[i]["mes_input_bias"]) for i in P.index])
            p_true = p["p_true_f"].reindex(P.index).to_numpy()
            p_ord = fp.order_internal_rate(p["te"], P)
            m["p_order_rho"].append(spearmanr(p_ord, p["te"]["true_internal_rate"]).statistic)
            m["pf_rho"].append(spearmanr(P["p"], p_true).statistic)
            m["d_rho"].append(spearmanr(P["d"], d_true).statistic)
            m["b_rho"].append(spearmanr(P["b"], b_true).statistic)
            m["p_ratio"].append(float(np.median(P["p"].to_numpy() / p_true)))
            m["d_gap"].append(float(np.median(P["d"].to_numpy() - d_true)))
            m["b_ratio"].append(float(np.median(P["b"].to_numpy() / b_true)))
        rows.append({"s_bar": s_bar, **{k: float(np.nanmean(v)) for k, v in m.items()}})
        print(f"  s̄={s_bar:.3f}  p_order ρ {rows[-1]['p_order_rho']:+.3f}  d ρ {rows[-1]['d_rho']:+.3f}  b ρ {rows[-1]['b_rho']:+.3f}  "
              f"| p 눈금 ×{rows[-1]['p_ratio']:.2f}  d 편차 {rows[-1]['d_gap']:+.3f}  b 눈금 ×{rows[-1]['b_ratio']:.2f}", flush=True)

    t = pd.DataFrame(rows)
    lines = [f"# 폐기 비중 가정 s̄ 민감도 — {len(SEEDS)} seed · 생성기 진실 0.03~0.20 (평균 0.115)", "",
             "| s̄ | 오더 p 순위 ρ | 공장 p 순위 ρ | 검출률 d 순위 ρ | 정직도 b 순위 ρ | p 눈금 (추정/진실 중앙) | d 편차 (추정−진실 중앙) | b 눈금 (추정/진실 중앙) |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        mark = "**" if abs(r.s_bar - fp.SCRAP_SHARE) < 1e-9 else ""
        lines.append(f"| {mark}{r.s_bar:.3f}{mark} | {r.p_order_rho:+.3f} | {r.pf_rho:+.3f} | {r.d_rho:+.3f} | {r.b_rho:+.3f} | ×{r.p_ratio:.2f} | {r.d_gap:+.3f} | ×{r.b_ratio:.2f} |")
    lines += ["", "순위(ρ)는 s̄ 에 거의 불변이다 — b 는 대수적으로 완전 불변, p·d 는 유출 항 때문에 소수점 둘째 자리에서 움직인다.",
              "눈금은 s̄ 에 비례해 움직인다: s̄ 를 반으로 줄이면 잡은 불량 추정이 2배가 되어 p 가 커지고 b 가 작아진다.",
              "→ 화면의 **순위·비교**는 가정에 안전하고, **절대값**(내부 불량률 %, 정직도 %)은 s̄ 를 실측(공장 폐기 대장 · 재작업 대장)으로 고정한 뒤에만 인용한다."]
    out = Path(__file__).resolve().parents[1] / "docs" / "_폐기비중_민감도.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
