"""공장·제품군 마스터 — 잠재값이 여기서 고정된다.

공장 잠재값 6개 중 다섯이 ``configs/generator.yaml`` 의 ``assumptions`` 블록에서 온다.
실측 근거가 없다는 뜻이고, 민감도 분석 대상이라는 뜻이다(docs/생성기_캘리브레이션.md §2.3~2.4).

공장은 제품군에 **전속**된다 — 한 공장이 한 제품만 만든다(CTO 답변 3번).
대신 한 제품군을 여러 공장이 나눠 만들어서, 같은 제품 안에서 공장을 비교할 수 있다.
이 구조가 아니면 "공장이 나쁜 건지 제품이 어려운 건지"가 분리되지 않는다.
"""

from __future__ import annotations

import math
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
    per_product = n_fac // len(products)

    # 공장 유형. 계측 수준이 공장마다 다르다.
    #   a  생산·불량 + 설비 신호 (MES 보유 또는 FactoryOS 입력 + CellOS)
    #   b  설비 신호만 (CellOS 만 설치, 생산·불량 입력 경로 없음)
    #   c  아무것도 없음 — L0 만
    ft = cfg["factory_types"]
    pool = ["a"] * ft["a"] + ["b"] * ft["b"] + ["c"] * ft["c"]
    # 제품군을 가로질러 나눠준다. 한 제품군의 세 곳이 전부 같은 유형이면
    # "같은 제품 안에서 유형을 비교" 하는 것이 불가능해진다.
    n_group = n_fac // per_product  # 제품군 수
    types = [""] * n_fac
    for j, t in enumerate(pool):
        types[(j % n_group) * per_product + (j // n_group)] = t

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

        capability = float(rng.normal(0.0, a["factory_capability_sd"]))
        capability_z = capability / max(a["factory_capability_sd"], 1e-9)

        out.append(
            {
                "factory_id": fid,
                "product_id": product["product_id"],
                # 잠재값 — 모델은 이 파일을 보지 못한다.
                # 값이 클수록 나쁜 공장이다(logit 에 양수로 들어간다).
                "capability": capability,
                "detection_rate": detection,
                # 나쁜 공장이 더 숨기는가? 지금 기본값은 **독립**(0.0)이다.
                # 현실에서 상관이 있다면 불일치 탐지가 나쁜 오더까지 함께 잡는다 —
                # 우리가 관측으로 확인할 수 없는 성질이라 CTO 확인 항목으로 올려 뒀다.
                "report_bias": float(
                    np.clip(
                        a["report_bias_mean"]
                        + a["report_bias_sd"]
                        * (
                            a["bias_capability_corr"] * capability_z
                            + math.sqrt(max(1.0 - a["bias_capability_corr"] ** 2, 0.0))
                            * rng.normal()
                        ),
                        0.15,
                        1.0,
                    )
                ),
                "bias_sensitivity": float(rng.uniform(0.0, 2.0 * a["bias_sensitivity"])),
                "report_discipline": float(rng.uniform(disc_lo, disc_hi)),
                "factory_type": types[idx],
                "has_mes": types[idx] == "a",  # 자체 MES 가 FactoryOS 에 연동돼 생산·불량 정보가 오는가
                "has_cell": types[idx] in ("a", "b"),  # 설비 신호. CellOS 는 협력 조건이라 실제로는 전부 True (c 는 0곳)
                # MES 입력 단계의 편향. 폐기·재작업 판정은 사람이 하므로 남는다.
                "mes_input_bias": float(rng.uniform(*a["mes_input_bias_range"])),
            }
        )
    return out
