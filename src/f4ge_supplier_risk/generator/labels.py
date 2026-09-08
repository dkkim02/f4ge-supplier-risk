"""라벨 — 입고검사와 클레임. 모델의 y 를 만든다.

두 가지가 이 파일의 전부다.

1. **표본검사다.** 전수가 아니다. `reject_qty` 는 로트 전체의 불량 수가 아니라
   ISO 2859-1 표본 n개에서 발견된 수다. 표본 200 × escape 0.1% = 기대 0.2개라
   **대부분의 오더에서 0이 나온다.** 그래서 y 를 단순 비율로 쓰면 신호가 없고,
   Beta-Binomial 로 (k, n) 쌍을 그대로 모델링해야 한다.

2. **라벨은 늦게 온다.** 그리고 시장마다 다르다 — 미 서안 18일, 유럽 60일.
   `label_available_at` 을 지키지 않으면 유럽 오더에서 실제로는 불가능한 학습을 하게 된다.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np

from f4ge_supplier_risk.generator.aql import sample_plan
from f4ge_supplier_risk.rng import stream

_CUSTOMS_DAYS = (3.0, 7.0)
_INSPECTION_DAYS = (1.0, 5.0)


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_outcome(
    cfg: dict[str, Any],
    order: dict[str, Any],
    truth: dict[str, Any],
    latents: dict[str, float],
) -> dict[str, Any]:
    rng = stream(cfg["seed"], "outcome", order["order_id"])
    qty = order["order_qty"]
    escaped = truth["escaped_total"]

    shipped = order["_ordered_dt"] + timedelta(days=truth["actual_days"])
    transit = cfg["market_transit_days"][order["market"]]
    arrived = shipped + timedelta(days=transit)

    plan = sample_plan(qty, order["aql_major"], order["aql_minor"])
    code, n, accept = plan["code"], min(int(plan["n"]), qty), plan["accept"]

    # 유한 로트에서 뽑으므로 초기하분포. 이항으로 근사하면 소량 오더에서 어긋난다.
    found = int(rng.hypergeometric(max(escaped, 0), max(qty - escaped, 0), n))

    # 심각도는 불량마다 독립으로 붙어 있으므로, 표본에서 발견된 것들의 등급 구성은
    # 다항분포다. **불합격의 대부분은 critical 쪽에서 나온다** — major 2.5% 기준은
    # 표본 80개에서 5개까지 허용해서 수백 PPM 공급사는 사실상 걸리지 않는다.
    sev_order = ("critical", "major", "minor")
    if escaped > 0 and found > 0:
        shares = np.array([truth["escaped_by_severity"][s] for s in sev_order], dtype=float)
        shares = shares / shares.sum() if shares.sum() else np.array([0.0, 0.0, 1.0])
        found_by = dict(zip(sev_order, rng.multinomial(found, shares)))
    else:
        found_by = dict.fromkeys(sev_order, 0)

    reasons = [s for s in sev_order if int(found_by[s]) > accept[s]]
    lot_result = "reject" if reasons else "accept"

    inspected_at = arrived + timedelta(days=float(rng.uniform(*_CUSTOMS_DAYS)))
    label_at = inspected_at + timedelta(days=float(rng.uniform(*_INSPECTION_DAYS)))

    # 클레임 — 로트가 **합격해버린** 고 escape 오더에서 나온다.
    # 희소 사건이므로 주 타깃이 아니라 비용 가중치 쪽에 쓴다.
    lot_escape = escaped / qty if qty else 0.0
    claim = lot_result == "accept" and rng.random() < min(0.9, 12.0 * lot_escape)
    claim_fields: dict[str, Any] = {
        "claim_occurred": bool(claim),
        "claim_at": None,
        "claim_qty": 0,
        "claim_cost": 0.0,
    }
    if claim:
        claim_at = arrived + timedelta(days=float(rng.uniform(20, 120)))
        claim_qty = int(max(1, round(qty * lot_escape * float(rng.uniform(0.2, 1.0)))))
        claim_fields |= {
            "claim_at": _iso(claim_at),
            "claim_qty": claim_qty,
            "claim_cost": round(claim_qty * order["unit_price"] * float(rng.uniform(1.5, 4.0)), 2),
        }
        label_at = max(label_at, claim_at)

    return {
        "order_id": order["order_id"],
        "shipped_at": _iso(shipped),
        "arrived_at": _iso(arrived),
        "aql_code": code,
        "incoming_inspected_qty": n,
        "incoming_reject_qty": found,
        "incoming_reject_critical": int(found_by["critical"]),
        "incoming_reject_major": int(found_by["major"]),
        "incoming_reject_minor": int(found_by["minor"]),
        "accept_number_major": accept["major"],
        "lot_result": lot_result,
        "reject_reason": reasons[0] if reasons else None,
        "label_available_at": _iso(label_at),
        **claim_fields,
    }
