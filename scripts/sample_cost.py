"""표본 설계의 화폐 환산 — ×2 · ×4 를 결정 사안으로 (2026-09-11, [[모델_방향_결정]] §3 ④ 후속).

`_표본설계.md` 는 「검사량이 전부다」까지 말했고 비용은 검사 유닛 수로만 냈다.
여기서 유닛 수를 **시간 · 인력 · 돈 · 리드타임**으로 바꾼다. 얻는 것(순위상관 Δ)은 그 표의 측정값을 그대로 인용한다.

    검사 시간   = 유닛 × 유닛당 소요(분)
    인력        = 월 검사 시간 ÷ 월 근로시간
    비용        = 검사 시간 × 시간당 단가        (내부 검사원 · 외주 검사 두 눈금)
    리드타임    = 오더 하나의 표본을 검사원 1명이 끝내는 시간 → 입고 지연(일)

⚠ **단가 셋은 실측이 아니다** — `ASSUMED` 블록. 대표·운영에서 실제 값을 받아 바꾸면 표가 다시 나온다.
   구조(유닛 수 · 오더 수 · 표본 크기)는 생성기 진실에서 계산한 값이고 8 seed 평균이다.

    python scripts/sample_cost.py        → docs/_표본설계_화폐환산.md
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sample_design import SEEDS, resample

from f4ge_supplier_risk import config
from f4ge_supplier_risk.generator.pipeline import build_dataset

warnings.filterwarnings("ignore")

# ── ⚠ 가정 — 실측 아님. 확인 뒤 바꿀 것 ─────────────────────────────────────────────
ASSUMED = {
    "min_per_unit": 3.0,          # 유닛당 검사 소요(분). CNC 정밀가공 부품 치수·외관 입고검사 1건. **가정**
    "hours_per_fte_month": 160.0, # 검사원 1명 월 근로시간
    "krw_per_hour_internal": 22_000,  # 내부 검사원 시간당 완전 인건비(월 350만원 ÷ 160h). **가정**
    "krw_per_hour_external": 50_000,  # 외주 검사(SGS·BV 류) man-day 300 USD ÷ 8h × 1,350원. **가정**
    "hours_per_day": 8.0,         # 리드타임 환산용 근무시간
}

# `_표본설계.md` 측정값 인용 (8 seed, c211bb8). 여기서 다시 적합하지 않는다.
MEASURED = {
    # 이름: (균일 배수, d 순위상관, d 최소 seed, 유출 순위상관)
    "현행 (ISO 2859-1 수준 II)": (1.0, 0.573, 0.007, 0.832),
    "균일 ×2 (수준 III 상당)":    (2.0, 0.684, 0.566, 0.871),
    "균일 ×4":                    (4.0, 0.782, 0.685, 0.899),
    "균일 ×8":                    (8.0, 0.834, 0.594, 0.922),
    "균일 전수":                  (None, 0.883, 0.762, 0.934),
}


def volumes(cfg: dict) -> pd.DataFrame:
    """설계별 검사 유닛 · 오더당 표본(중앙) — 8 seed 평균. 모델 적합 없음."""
    rows = []
    for seed in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        data = build_dataset(c)
        months = cfg["scale"]["months"]
        shipped = sum(int(o["order_qty"]) for o in data["orders"])
        n_orders = len(data["orders"])
        for name, (mult, *_rest) in MEASURED.items():
            d = resample(data, mult, 0.0, False, np.random.default_rng(seed))
            n = np.array([oc["incoming_inspected_qty"] for oc in d["quality_outcomes"]], float)
            rows.append({"design": name, "seed": seed, "units": n.sum(), "shipped": shipped,
                         "orders": n_orders, "months": months, "n_med": np.median(n), "n_p90": np.quantile(n, 0.9)})
    return pd.DataFrame(rows).groupby("design", sort=False).mean(numeric_only=True)


def main() -> None:
    cfg = config.load(str(ROOT / "configs/generator.yaml"))
    V = volumes(cfg)
    A = ASSUMED
    base = V.iloc[0]

    T = pd.DataFrame(index=V.index)
    T["검사율"] = V["units"] / V["shipped"]
    T["유닛/월"] = V["units"] / V["months"]
    T["추가 유닛/월"] = T["유닛/월"] - base["units"] / base["months"]
    T["시간/월"] = T["유닛/월"] * A["min_per_unit"] / 60
    T["추가 시간/월"] = T["추가 유닛/월"] * A["min_per_unit"] / 60
    T["FTE"] = T["시간/월"] / A["hours_per_fte_month"]
    T["추가 FTE"] = T["추가 시간/월"] / A["hours_per_fte_month"]
    T["추가 비용/월 내부(만원)"] = T["추가 시간/월"] * A["krw_per_hour_internal"] / 1e4
    T["추가 비용/월 외주(만원)"] = T["추가 시간/월"] * A["krw_per_hour_external"] / 1e4
    T["오더 표본 중앙"] = V["n_med"]
    T["오더당 검사 h"] = V["n_med"] * A["min_per_unit"] / 60
    T["입고 지연 일 (검사원 1명)"] = T["오더당 검사 h"] / A["hours_per_day"]
    T["d ρ"] = [MEASURED[k][1] for k in T.index]
    T["d 최소 seed"] = [MEASURED[k][2] for k in T.index]
    T["유출 ρ"] = [MEASURED[k][3] for k in T.index]
    T["Δ유출 ρ"] = T["유출 ρ"] - T["유출 ρ"].iloc[0]
    T["Δd ρ"] = T["d ρ"] - T["d ρ"].iloc[0]
    cost = T["추가 비용/월 내부(만원)"].replace(0, np.nan)
    T["Δ유출 / 100만원·월"] = T["Δ유출 ρ"] / (cost / 100)

    orders_per_month = base["orders"] / base["months"]
    L = [f"# 표본 설계 — 화폐 환산 (8 seed · {int(base['months'])}개월 · 공장 12곳)", "",
         "`_표본설계.md` 의 「검사량이 전부다」를 결정 사안으로 만든다. 순위상관은 그 표의 측정값을 인용했고(재적합 없음),",
         "유닛 수 · 오더 수 · 표본 크기는 생성기 진실에서 다시 뽑았다(Hypergeometric, 모델 적합 없음).", "",
         "## ⚠ 가정 — 이 셋을 실제 값으로 바꾸면 표가 다시 나온다", "",
         "| 항목 | 값 | 근거 |", "|---|---|---|",
         f"| 유닛당 검사 소요 | **{A['min_per_unit']:.0f}분** | 가정. CNC 정밀가공 부품 치수·외관 1건. 운영 확인 필요 |",
         f"| 내부 검사원 시간당 완전 인건비 | **{A['krw_per_hour_internal']:,}원** | 가정. 월 350만원 ÷ {A['hours_per_fte_month']:.0f}h |",
         f"| 외주 검사 시간당 단가 | **{A['krw_per_hour_external']:,}원** | 가정. SGS·BV 류 man-day 300 USD ÷ 8h × 1,350원 |",
         f"| 월 근로시간 · 일 근무시간 | {A['hours_per_fte_month']:.0f}h · {A['hours_per_day']:.0f}h | 환산 상수 |",
         "", f"규모: 월 출하 **{base['shipped'] / base['months']:,.0f} 유닛** · 월 오더 **{orders_per_month:.1f}건** · 오더 수량 중앙 800.", "",
         "## 설계별 비용 · 인력 · 리드타임 · 얻는 것", "",
         "| 설계 | 검사율 | 유닛/월 | 추가 시간/월 | FTE (추가) | 추가 비용/월 내부 | 추가 비용/월 외주 | 오더당 검사 | 입고 지연 | d ρ (최소) | 유출 ρ | Δ유출 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in T.iterrows():
        L.append(f"| {name} | {r['검사율']:.0%} | {r['유닛/월']:,.0f} | {r['추가 시간/월']:,.0f}h | "
                 f"{r['FTE']:.1f} (+{r['추가 FTE']:.1f}) | {r['추가 비용/월 내부(만원)']:,.0f}만원 | {r['추가 비용/월 외주(만원)']:,.0f}만원 | "
                 f"{r['오더 표본 중앙']:,.0f}개 · {r['오더당 검사 h']:.1f}h | {r['입고 지연 일 (검사원 1명)']:.1f}일 | "
                 f"{r['d ρ']:+.3f} ({r['d 최소 seed']:+.3f}) | {r['유출 ρ']:+.3f} | {r['Δ유출 ρ']:+.3f} |")
    x2, x4 = T.iloc[1], T.iloc[2]
    L += ["", "## 결정 사안 — ×2 인가 ×4 인가", "",
          f"- **×2 (검사율 {x2['검사율']:.0%})**: 검사원 **+{x2['추가 FTE']:.1f}명**, 월 **{x2['추가 비용/월 내부(만원)']:,.0f}만원**(내부) / "
          f"{x2['추가 비용/월 외주(만원)']:,.0f}만원(외주). 오더당 검사 {x2['오더당 검사 h']:.1f}h → 입고 지연 {x2['입고 지연 일 (검사원 1명)']:.1f}일. "
          f"얻는 것: 유출 순위 **+{x2['Δ유출 ρ']:.3f}** · d 순위 +{x2['Δd ρ']:.3f}, **d 최소 seed 0.007 → 0.566** — 「검출률」을 처음 주장할 수 있게 되는 선",
          f"- **×4 (검사율 {x4['검사율']:.0%})**: 검사원 **+{x4['추가 FTE']:.1f}명**, 월 **{x4['추가 비용/월 내부(만원)']:,.0f}만원** / "
          f"{x4['추가 비용/월 외주(만원)']:,.0f}만원. 오더당 검사 {x4['오더당 검사 h']:.1f}h → 입고 지연 {x4['입고 지연 일 (검사원 1명)']:.1f}일. "
          f"얻는 것: 유출 +{x4['Δ유출 ρ']:.3f} · d +{x4['Δd ρ']:.3f} (ρ 0.78 — 상품 수준 d 의 시작)",
          f"- 비용 대비: ×2 는 100만원·월당 Δ유출 **{x2['Δ유출 / 100만원·월']:+.4f}**, ×4 는 {x4['Δ유출 / 100만원·월']:+.4f}. 수확체감이 돈에서도 그대로다",
          "- 비교 기준: 모델 셋의 격차 ±0.003 은 비용 0 이지만 얻는 것도 0 이다. ×2 의 +0.039 는 그 13배",
          "", "## 이 표가 말하지 않는 것", "",
          "- **얻는 쪽의 화폐 환산은 없다.** 유출 순위 +0.039 가 클레임·재작업을 얼마나 줄이는지는 클레임 단가와 검토 정밀도 변화가 있어야 한다 — 다음 단계",
          "- 리드타임은 검사원 1명이 한 오더를 순차 처리하는 가정이다. 검사원을 늘리면 지연은 줄고 FTE 는 그대로다",
          "- 검사율 31%(×4) 이상은 ISO 2859-1 표본검사의 범위를 벗어난다 — 실무에서는 전수 검사 라인 또는 공장 측 검사 인수(공장 `d` 를 우리 검사가 대체)로 읽어야 한다",
          "- 심각도·lot_result 는 재추출하지 않았다(`_표본설계.md` 와 같은 한계)",
          "", "재현: `python scripts/sample_cost.py` · 가정은 스크립트 `ASSUMED` 블록"]
    out = ROOT / "docs" / "_표본설계_화폐환산.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n표 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
