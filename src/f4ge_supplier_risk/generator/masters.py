"""공장·제품군 마스터 — 잠재값이 여기서 고정된다.

공장 잠재값 6개 중 다섯이 ``configs/generator.yaml`` 의 ``assumptions`` 블록에서 온다.
실측 근거가 없다는 뜻이고, 민감도 분석 대상이라는 뜻이다(docs/생성기_캘리브레이션.md §2.3~2.4).

공장은 제품군에 **전속**된다 — 한 공장이 한 제품만 만든다(CTO 답변 3번).
대신 한 제품군을 여러 공장이 나눠 만들어서, 같은 제품 안에서 공장을 비교할 수 있다.
이 구조가 아니면 "공장이 나쁜 건지 제품이 어려운 건지"가 분리되지 않는다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from f4ge_supplier_risk.rng import stream

# 제품군별 고정 물성. 절대값 자체는 중요하지 않고 서로 다르기만 하면 된다.
_PRODUCT_SPECS = (
    ("prd_bracket", 42.0, 1.15),
    ("prd_shaft", 61.0, 1.08),
    ("prd_housing", 95.0, 1.22),
    ("prd_flange", 38.0, 1.12),
)


def build_products(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seed = cfg["seed"]
    a = cfg["assumptions"]
    n = cfg["scale"]["products"]
    out = []
    for i, (pid, cycle, mat) in enumerate(_PRODUCT_SPECS[:n]):
        rng = stream(seed, "product", pid)
        out.append(
            {
                "product_id": pid,
                "difficulty": float(rng.normal(0.0, a["product_difficulty_sd"])),
                "nominal_cycle_sec": cycle,
                "material_per_unit": mat,
            }
        )
    return out


def build_factories(cfg: dict[str, Any], products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed = cfg["seed"]
    a = cfg["assumptions"]
    n_fac = cfg["scale"]["factories"]
    installed = set(cfg["telemetry"]["installed_factories"])
    per_product = n_fac // len(products)

    out = []
    for idx in range(n_fac):
        fid = f"fac_kr_{idx + 1:02d}"
        rng = stream(seed, "factory", fid)
        product = products[idx // per_product]

        # detection_rate — 이 공장 검사가 내부 불량 중 몇 %를 잡나.
        # escape 를 만드는 열쇠이자, "우리가 불합격시킨 비율"로 관측되는 값이다.
        # 범위는 escape 목표에서 역산했다. 내부 불량 3% 인 공장이 escape 0.1% 로
        # 출하하려면 검출률이 96.7% 여야 한다 — 핵심 치수 전수검사면 현실적인 값이다.
        lo, hi = 0.88, 0.998
        detection = float(lo + (hi - lo) * rng.beta(6.0, 2.0))

        disc_lo, disc_hi = a["report_discipline_range"]
        t2_lo, t2_hi = a["t2_compliance_range"]

        out.append(
            {
                "factory_id": fid,
                "product_id": product["product_id"],
                # 잠재값 — 모델은 이 파일을 보지 못한다
                "capability": float(rng.normal(0.0, a["factory_capability_sd"])),
                "detection_rate": detection,
                "report_bias": float(
                    np.clip(rng.normal(a["report_bias_mean"], a["report_bias_sd"]), 0.15, 1.0)
                ),
                "bias_sensitivity": float(rng.uniform(0.0, 2.0 * a["bias_sensitivity"])),
                "report_discipline": float(rng.uniform(disc_lo, disc_hi)),
                "t2_compliance": float(rng.uniform(t2_lo, t2_hi)),
                "telemetry_installed": fid in installed,
            }
        )
    return out
