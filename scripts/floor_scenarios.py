"""MES 입력 정직도 하한을 흔들어 두 상품의 비중이 어떻게 갈리는지 잰다.

2026-09-08 확인: **FactoryOS 에 사람이 입력하는 경로는 없다.** 데이터는 공장 MES 와 ERP,
그리고 CellOS 에서 자동으로 온다(출처 4곳 = 우리 + MES + ERP + CellOS).
그러면 편향이 남는 자리는 MES/ERP 안에서 사람이 판정하는 단계뿐이고(폐기냐 재작업이냐),
재고가 맞아야 하므로 **하한이 지금 가정(0.55)보다 높은 쪽일 가능성이 크다.**

이 가정 하나가 「예측이 주력이냐 불일치 탐지가 주력이냐」를 정한다(설계_해설 §8.3, CTO 질문 29).
그래서 하한 × MES 공장 수 격자로 재서 어느 쪽을 앞세울지 판단할 근거를 만든다.

    python scripts/floor_scenarios.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coverage_sweep import run_level  # noqa: E402

from f4ge_supplier_risk import config  # noqa: E402

FLOORS = ((0.55, 1.0), (0.80, 1.0), (0.95, 1.0))
LEVELS = (3, 12)


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    rows = []
    for n_a in LEVELS:
        for lo, hi in FLOORS:
            r = run_level(cfg, n_a, floor=(lo, hi))
            m = {k: float(np.nanmean(v)) for k, v in r.items()}
            h = np.array(r["honesty"], float)
            pos = f"{int(np.nansum(h > 0))}/{int((~np.isnan(h)).sum())}"
            tag = "현재 가정" if lo == 0.55 else ""
            rows.append(
                f"| {n_a}곳 | {lo:.2f}~{hi:.2f} {tag} | {m['rank_all']:+.3f} | {m['rank_a']:+.3f} | "
                f"{m['risk_n']:.1f} | **{m['risk_prec']:.1%}** | {m['disc_n']:.1f} | "
                f"**{m['disc_prec']:.1%}** | {m['disc_base']:.1%} | {m['honesty']:+.3f} ({pos}) |"
            )
            print(rows[-1], flush=True)
    hdr = ("| MES 공장 | 입력 정직도 하한 | 전체 순위상관 | 유형 a 순위상관 | 위험축 건수 | "
           "위험축 정밀도 | 불일치축 건수 | 불일치축 정밀도 | 기저 | 공장 정직도 r (양수 seed) |")
    table = "\n".join([hdr, "|---|---|---|---|---|---|---|---|---|---|", *rows])
    print("\n" + table)
    out = Path(__file__).resolve().parents[1] / "docs" / "_하한_시나리오_표.md"
    out.write_text(
        "# MES 입력 정직도 하한 × MES 공장 수 — 8 seed\n\n"
        "FactoryOS 에 사람 입력 경로가 없다는 2026-09-08 확인에 따라 하한을 올려 가며 잰 표.\n"
        "하한이 높을수록 보고가 정직하고, 예측 기여는 오르고 불일치 탐지 기여는 내려간다.\n\n"
        + table + "\n", encoding="utf-8")
    print("\n→", out)


if __name__ == "__main__":
    main()
