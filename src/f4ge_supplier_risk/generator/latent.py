"""잠재변수 → 두 불량률.

**불량률이 두 종류라는 것이 이 프로젝트에서 제일 틀리기 쉬운 지점이다.**

    internal_defect_rate  공정에서 발생. 0.6~6.7% (미국 제조 3~4시그마 실태)
    escape_rate           검사를 통과해 고객에게 도달. 0.05~0.5%  ← 우리 y

기존 `f4ge-quality-prediction` 의 defect_rate 0.05~0.25 는 **전자**다.
그걸 그대로 가져오면 100배 틀린다. 근거는 docs/생성기_캘리브레이션.md §0.

시점 상태 ``state_t`` 는 공장별 AR(1). 이 하나가 "공장이 나빠지고 있다"를
탐지할 수 있는지를 결정한다 — ρ가 0이면 과거가 미래를 말해주지 못하고,
1에 가까우면 이력만으로 다 맞힌다. 값은 가정이므로 민감도 분석 대상.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from f4ge_supplier_risk.rng import stream

# sigmoid(BASE) 가 대략 목표 중앙값 0.030 이 되도록 잡은 절편.
_BASE = -3.5
# 잠재값 → logit 기여 가중치. 상대 크기만 의미가 있다.
_W_CAPABILITY = 0.55
_W_DIFFICULTY = 0.45
_W_STATE = 0.40
_W_PRESSURE = 0.60
_W_MATERIAL = 0.35
_W_RANDOM = 0.30


def factory_state_series(cfg: dict[str, Any], factory_id: str, n_weeks: int) -> np.ndarray:
    """공장별 주간 상태 AR(1). 부하율·인력 이탈·설비 노후의 합성."""
    rho = cfg["assumptions"]["within_factory_ar1"]
    rng = stream(cfg["seed"], "state", factory_id)
    innov_sd = math.sqrt(1.0 - rho**2)  # 정상분산 1로 정규화
    out = np.empty(n_weeks)
    x = rng.normal(0.0, 1.0)
    for t in range(n_weeks):
        x = rho * x + rng.normal(0.0, innov_sd)
        out[t] = x
    return out


def material_lot_effect(cfg: dict[str, Any], material_lot_id: str) -> float:
    """자재 로트 효과. **공급사를 공유하는 공장끼리 같이 나빠지는 유일한 경로다.**"""
    return float(stream(cfg["seed"], "material", material_lot_id).normal(0.0, 1.0))


def order_latents(
    cfg: dict[str, Any],
    order: dict[str, Any],
    factory: dict[str, Any],
    product: dict[str, Any],
    state_t: float,
) -> dict[str, float]:
    rng = stream(cfg["seed"], "order_latent", order["order_id"])

    # 납기 여유가 평소보다 적으면 서두름 → 불량. **중심화가 중요하다** —
    # 원값을 그대로 쓰면 상수항이 logit 전체를 밀어서 목표 불량률을 놓친다.
    # 한 공장에 한 오더(09-08 저녁 전제)라 동시 부하 항은 없다. 압박은 납기 여유 하나에서 온다.
    pressure = -(order["lead_slack"] - cfg["order"]["lead_slack_mean"])
    material = material_lot_effect(cfg, order["material_lot_id"])
    noise = float(rng.normal(0.0, 1.0))

    logit = (
        _BASE
        + _W_CAPABILITY * factory["capability"]
        + _W_DIFFICULTY * product["difficulty"]
        + _W_STATE * state_t
        + _W_PRESSURE * pressure
        + _W_MATERIAL * material
        + _W_RANDOM * noise
    )
    internal = 1.0 / (1.0 + math.exp(-logit))

    # 검출률 — 기본은 공장 상수다(κ = (1−d)/d 가 공장당 하나). `detection_order_sd` 를 올리면
    # 오더마다 logit 공간에서 흔들려 그 전제가 깨진다. **0 에서는 난수를 뽑지 않는다** —
    # 뽑으면 이후 스트림이 밀려 σ=0 재현성이 깨진다.
    d_sd = float(cfg["assumptions"].get("detection_order_sd", 0.0) or 0.0)
    detection = float(factory["detection_rate"])
    if d_sd > 0.0:
        d = min(max(detection, 1e-6), 1 - 1e-6)
        z = float(rng.normal(0.0, 1.0))
        detection = 1.0 / (1.0 + math.exp(-(math.log(d / (1 - d)) + d_sd * z)))
        detection = min(max(detection, 0.30), 0.9995)

    escape = internal * (1.0 - detection)

    return {
        "internal_defect_rate_true": internal,
        "escape_rate_true": escape,
        "detection_rate_order": detection,
        "state_t": float(state_t),
        "schedule_pressure": float(pressure),
        "material_effect": material,
        "order_random": noise,
    }
