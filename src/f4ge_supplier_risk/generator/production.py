"""오더 1건의 **진실** 생산 경과.

보고는 진실의 함수이므로 이것이 먼저 만들어져야 한다(명세 §8).

불량 위치를 미리 뽑아 두는 이유는 기존 프로젝트에서 배운 것이다 — 진척률 시점마다
새로 뽑으면 누적값이 단조 증가하지 않고, 비례 추정으로 메우면 25% 시점과 100% 시점이
같은 값을 읽어 **답이 새어 나간다**.

각 불량 유닛에 대해 미리 정해 두는 것 넷:
  · 생산 라인상의 위치(0~1)  → 언제 발생했나
  · 공장 검사에 걸리는가      → detection_rate
  · 걸렸다면 폐기인가 재작업인가 → 대부분 재작업으로 흡수된다(Hidden Factory)
  · 심각도(critical/major/minor) → **로트 합·불을 실제로 정하는 값**
"""

from __future__ import annotations

from typing import Any

import numpy as np

from f4ge_supplier_risk.rng import stream

_SETUP_FRACTION = 0.12  # 초반은 셋업이라 물건이 안 나온다
_SCRAP_SHARE = (0.03, 0.20)  # 잡아낸 불량 중 폐기 비율. 나머지는 재작업


def build_truth(
    cfg: dict[str, Any],
    order: dict[str, Any],
    factory: dict[str, Any],
    latents: dict[str, float],
) -> dict[str, Any]:
    rng = stream(cfg["seed"], "production", order["order_id"])
    qty = order["order_qty"]

    # 실제 소요일 — 무리한 일정과 나쁜 상태가 지연을 만든다
    slip = 0.10 * max(0.0, latents["schedule_pressure"]) + 0.06 * latents["state_t"]
    actual_days = max(order["_std_days"] * (1.0 + float(np.clip(slip, -0.15, 0.9))), 7.0)

    n_defect = int(rng.binomial(qty, latents["internal_defect_rate_true"]))
    pos = np.sort(rng.random(n_defect))  # 라인상 위치
    mix = cfg["assumptions"]["defect_severity_mix"]
    severity = rng.choice(
        ["critical", "major", "minor"],
        size=n_defect,
        p=[mix["critical"], mix["major"], mix["minor"]],
    )
    # 오더별 검출률(latent). detection_order_sd = 0 이면 공장 상수와 같은 값이다.
    caught = rng.random(n_defect) < latents.get("detection_rate_order", factory["detection_rate"])
    scrap_share = float(rng.uniform(*_SCRAP_SHARE))
    is_scrap = caught & (rng.random(n_defect) < scrap_share)
    is_rework = caught & ~is_scrap

    return {
        "actual_days": actual_days,
        "n_defect": n_defect,
        "defect_pos": pos,
        "defect_scrap": is_scrap,
        "defect_rework": is_rework,
        "defect_escaped": ~caught,
        "defect_severity": severity,
        "escaped_total": int((~caught).sum()),
        "escaped_by_severity": {
            sev: int(((~caught) & (severity == sev)).sum())
            for sev in ("critical", "major", "minor")
        },
    }


def progress_at(truth: dict[str, Any], elapsed_days: float) -> float:
    """경과일 → 실제 진척률. 셋업 구간을 지나야 물건이 나온다."""
    t = min(max(elapsed_days / truth["actual_days"], 0.0), 1.0)
    if t <= _SETUP_FRACTION:
        return 0.0
    return (t - _SETUP_FRACTION) / (1.0 - _SETUP_FRACTION)


def counts_at(truth: dict[str, Any], qty: int, progress: float) -> dict[str, int]:
    """그 시점까지의 **실제** 누적값. 단조 증가가 보장된다."""
    produced = round(float(qty * progress))
    upto = truth["defect_pos"] <= progress
    return {
        "produced": produced,
        "scrap": int((upto & truth["defect_scrap"]).sum()),
        "rework": int((upto & truth["defect_rework"]).sum()),
        "defects": int(upto.sum()),
        "escaped": int((upto & truth["defect_escaped"]).sum()),
    }
