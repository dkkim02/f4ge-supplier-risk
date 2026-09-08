"""L3 — 공장 ERP 일 스냅샷. **회계 숫자라 편향이 거의 없다. 대신 채우는 필드와 성실도가 공장마다 다르다.**

FactoryOS 에 들어오는 소스 4개(MES · CellOS · ERP · 포지 기록) 중 하나다(2026-09-08 저녁, 사용자 확정).
ERP 에서 오는 것: 자재 입출고·재고 · 재공(WIP) 잔량·공정별 재고 · 인원·잔업 · 구매 발주·외주(도금·열처리).

설계의 핵심 — **MES 보고의 대조군이 여기 있다.**
  MES 가 생산 수량을 부풀리면 ERP 자재 불출·재공과 어긋난다(qty_inflation · wip_mismatch).
  MES 가 불량을 줄이면 ERP 잔업이 설명되지 않는다(defect_hiding).
ERP 값은 진실의 함수이고 공장이 숫자를 꾸미지 않는다 — 재고와 회계가 맞아야 하기 때문이다.
다만 **성실도**(마감 누락·지연)와 **필드 가용**(잔업·재공·외주를 ERP 에 안 찍는 공장)은 공장마다 다르다.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np

from f4ge_supplier_risk.generator.production import counts_at, progress_at
from f4ge_supplier_risk.rng import stream

_MATERIAL_BATCHES = 4  # 자재는 오더 총량의 1/4 씩 불출된다 — 불출은 소진보다 앞선다
_FG_LAG = 0.15  # 가공 완료 → 완성품 입고까지의 진척 지연(외주 후처리·검사)
_OUTSOURCE_AT = 0.75  # 가공(machining) 을 마친 물량이 외주(도금·열처리)로 나간다


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def trouble_of(latents: dict[str, float]) -> float:
    """이 오더가 얼마나 곤란한가 (0~1). reports.py 와 같은 식 — 잔업·지연이 이 값을 따라간다."""
    return float(
        np.clip(
            0.5 * latents["state_t"]
            + 0.6 * latents["schedule_pressure"]
            + 3.0 * (latents["internal_defect_rate_true"] - 0.03),
            0.0,
            1.0,
        )
    )


def build_erp_daily(
    cfg: dict[str, Any],
    order: dict[str, Any],
    factory: dict[str, Any],
    product: dict[str, Any],
    latents: dict[str, float],
    truth: dict[str, Any],
) -> list[dict[str, Any]]:
    rng = stream(cfg["seed"], "erp", order["order_id"])
    qty = order["order_qty"]
    mpu = product["material_per_unit"]
    total_material = qty * mpu * 1.03  # 여유분 3%
    batch = total_material / _MATERIAL_BATCHES
    days = max(1, int(math.ceil(truth["actual_days"])))
    trouble = trouble_of(latents)
    fields = factory["fields_erp"]
    headcount_base = int(rng.integers(2, 7))

    out: list[dict[str, Any]] = []
    for d in range(days):
        due = order["_ordered_dt"] + timedelta(days=d + 1)
        # 마감 누락 — ERP 는 회계라 MES 보다 드물지만, 성실도 낮은 공장은 일 마감을 빠뜨린다
        if rng.random() < (1.0 - factory["erp_discipline"]) * 0.05:
            out.append({"order_id": order["order_id"], "day_index": d, "due_at": _iso(due),
                        "reported_at": None, "is_missing": True})
            continue
        delay = float(rng.gamma(1.2, (1.0 - factory["erp_discipline"]) * 1.5))

        prog = progress_at(truth, d + 1)
        c = counts_at(truth, qty, prog)
        consumed = (c["produced"] + c["scrap"]) * mpu
        # 불출은 배치 단위로 소진보다 앞선다 (첫 배치는 착수 시 나간다)
        issued = min(total_material, batch * max(1, math.ceil((consumed + 1e-9) / batch)))
        fg_prog = max(0.0, (prog - _FG_LAG) / (1.0 - _FG_LAG)) if prog > 0 else 0.0
        fg = counts_at(truth, qty, fg_prog)["produced"]
        fg_good = max(fg - c["scrap"], 0)
        wip = max(c["produced"] - c["scrap"] - fg_good, 0)
        po = int(round(qty * min(prog / _OUTSOURCE_AT, 1.0))) if prog > 0 else 0

        row: dict[str, Any] = {
            "order_id": order["order_id"],
            "day_index": d,
            "due_at": _iso(due),
            "reported_at": _iso(due + timedelta(days=delay)),
            "is_missing": False,
            # 자재 — 물리량. 항상 온다 (자재 없이 회계가 닫히지 않는다)
            "material_issued_qty": round(issued, 1),
            "material_on_hand_qty": round(max(issued - consumed, 0.0), 1),
            "material_lot_id": order["material_lot_id"],
        }
        if fields.get("wip_qty"):
            # 공정별 재고 — 재공을 가공/후처리/검사 단계로 나눈다. 합이 wip 다.
            split = rng.dirichlet([3.0, 2.0, 1.0])
            row |= {
                "wip_qty": int(wip),
                "wip_machining_qty": int(round(wip * split[0])),
                "wip_finishing_qty": int(round(wip * split[1])),
                "wip_inspection_qty": int(wip) - int(round(wip * split[0])) - int(round(wip * split[1])),
                "finished_goods_qty": int(fg_good),
            }
        if fields.get("headcount"):
            row["headcount"] = headcount_base + int(rng.integers(-1, 2))
        if fields.get("overtime_hours"):
            # 잔업은 실제 곤란을 따라간다 → 불량 은폐의 대조군 (MES 가 아니라 ERP 근태에서 온다)
            row["overtime_hours"] = round(float(rng.gamma(2.0, 3.0)) * (1.0 + 3.0 * trouble), 1)
        if fields.get("outsourcing"):
            recv = min(po, int(fg))
            row |= {"outsourcing_po_qty": po, "outsourcing_received_qty": int(recv)}
        out.append(row)
    return out
