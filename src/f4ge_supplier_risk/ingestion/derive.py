"""계약 행 → 생성기 모양 레코드 (+ 우리가 계산하는 파생값).

운영에서 들어오는 것은 계약(contracts/*.schema.json) 행이고, 피처 빌더(features/build.py)가 읽는 것은
생성기 레코드다. 둘 사이의 번역을 여기서만 한다 — 생성기 쪽 이름은 바꾸지 않는다(contracts.py 의 역방향).

파생값 셋은 **우리가 계산하는 L0** 이라 공장 협조가 필요 없다:
  lead_slack   = (약속 납기 − 표준 소요) / 표준 소요.  표준 소요는 제품 마스터(공칭 사이클)에서 — 생성기와 같은 식
  price_zscore = 제품군 안에서의 단가 z-score
L0′(order_meta)는 MES 집계 행에서 generator.meta.build_meta 로 그대로 만든다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from f4ge_supplier_risk.generator.meta import build_meta
from f4ge_supplier_risk.generator.orders import (
    _attach_price_zscore,
    _standard_days,
)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def _iso(d: datetime) -> str:
    return d.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def products_from_master(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """제품 마스터(product_code · nominal_cycle_sec · material_per_unit) → 생성기 products 레코드."""
    return [
        {
            "product_id": f"prd_{r['product_code']}",
            "difficulty": float(r.get("difficulty", 0.0)),
            "nominal_cycle_sec": float(r["nominal_cycle_sec"]),
            "material_per_unit": float(r["material_per_unit"]),
        }
        for r in rows
    ]


def orders_from_contract(rows: list[dict[str, Any]], products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prod = {p["product_id"]: p for p in products}
    out: list[dict[str, Any]] = []
    for r in rows:
        pid = f"prd_{r['product_code']}"
        if pid not in prod:
            raise KeyError(f"제품 마스터에 없는 product_code: {r['product_code']}")
        ordered, promised = _dt(r["ordered_at"]), _dt(r["promised_date"])
        std_days = _standard_days(int(r["order_quantity"]), prod[pid])
        lead_days = (promised - ordered).total_seconds() / 86400
        out.append(
            {
                "order_id": r["order_id"],
                "factory_id": r["factory_id"],
                "product_id": pid,
                "market": r["market"],
                "order_qty": int(r["order_quantity"]),
                "ordered_at": _iso(ordered),
                "promised_date": _iso(promised),
                "unit_price": float(r.get("unit_price", 0.0)),
                "quoted_at": r.get("quoted_at"),
                "quote_response_h": float(r.get("quote_response_h", 0.0)),
                "lead_slack": round(lead_days / std_days - 1.0, 4),
                "aql_major": r.get("aql_major", 0.025),
                "aql_minor": r.get("aql_minor", 0.040),
                "material_lot_id": r.get("material_lot_id"),
                "material_supplier": r.get("material_supplier"),
                "_ordered_dt": ordered,
                "_promised_dt": promised,
                "_lead_days": lead_days,
                "_std_days": std_days,
            }
        )
    out.sort(key=lambda o: o["_ordered_dt"])
    if out:
        _attach_price_zscore(out)
    return out


def reports_from_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        base = {
            "order_id": r["order_id"], "seq": int(r["seq"]), "due_at": r["due_at"],
            "reported_at": r.get("reported_at"), "is_missing": bool(r["is_missing"]),
        }
        if r["is_missing"]:
            out.append(base)
            continue
        rec = {
            **base,
            "produced_qty": int(r["produced_quantity"]), "scrap_qty": int(r["scrap_quantity"]),
            "rework_qty": int(r["rework_quantity"]), "stage": r["stage"],
            "promised_date_reported": r["promised_date_reported"], "issue_flag": bool(r["issue_flag"]),
        }
        rec["period"] = r.get("period", "daily")
        # 선택 필드 — 있는 것만 옮긴다. 없는 필드는 없는 채로 두어야 field_blank_count 가 맞다.
        for src, dst in (
            ("inspected_quantity", "inspected_qty"), ("inspection_type", "inspection_type"),
            ("reject_quantity", "reject_qty"), ("defect_type", "defect_type"),
            ("material_lot_id", "material_lot_id"), ("machine_id", "machine_id"),
            ("shift", "shift"), ("photo_taken_at", "photo_taken_at"),
        ):
            if src in r and r[src] is not None:
                rec[dst] = r[src]
        out.append(rec)
    return out


def fai_from_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "order_id": r["order_id"], "fai_at": r["fai_at"], "dim_count": int(r["dim_count"]),
            "margin_min": float(r["margin_min"]), "margin_mean": float(r.get("margin_mean", r["margin_min"])),
            "margin_p25": float(r.get("margin_p25", r["margin_min"])), "out_of_tol_count": int(r["out_of_tol_count"]),
        }
        for r in rows
    ]


def cell_from_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "order_id": r["order_id"], "day_index": int(r["day_index"]), "uptime_ratio": float(r["uptime_ratio"]),
            "cycle_time_median": float(r.get("cycle_time_median", 1.0)),
            "anomaly_event_count": int(r.get("anomaly_event_count", 0)),
            "tool_change_count": int(r.get("tool_change_count", 0)),
            "produced_by_counter": int(r["produced_by_counter"]),
        }
        for r in rows
    ]


def erp_from_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        base = {"order_id": r["order_id"], "day_index": int(r["day_index"]), "due_at": r["due_at"],
                "reported_at": r.get("reported_at"), "is_missing": bool(r["is_missing"])}
        if r["is_missing"]:
            out.append(base)
            continue
        rec = {**base, "material_issued_qty": float(r["material_issued_quantity"]),
               "material_on_hand_qty": float(r["material_on_hand_quantity"])}
        for src, dst in (
            ("material_lot_id", "material_lot_id"), ("wip_quantity", "wip_qty"),
            ("wip_machining_quantity", "wip_machining_qty"), ("wip_finishing_quantity", "wip_finishing_qty"),
            ("wip_inspection_quantity", "wip_inspection_qty"), ("finished_goods_quantity", "finished_goods_qty"),
            ("headcount", "headcount"), ("overtime_hours", "overtime_hours"),
            ("outsourcing_po_quantity", "outsourcing_po_qty"), ("outsourcing_received_quantity", "outsourcing_received_qty"),
        ):
            if src in r and r[src] is not None:
                rec[dst] = r[src]
        out.append(rec)
    return out


def outcomes_from_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "order_id": r["order_id"], "shipped_at": r.get("shipped_at"), "arrived_at": r.get("arrived_at"),
            "aql_code": r.get("aql_code"),
            "incoming_inspected_qty": int(r["inspected_quantity"]), "incoming_reject_qty": int(r["reject_quantity"]),
            "incoming_reject_critical": r.get("reject_critical", 0), "incoming_reject_major": r.get("reject_major", 0),
            "incoming_reject_minor": r.get("reject_minor", 0), "accept_number_major": r.get("accept_number_major"),
            "lot_result": r["lot_result"], "reject_reason": r.get("reject_reason"),
            "label_available_at": r["label_available_at"],
            "claim_occurred": bool(r.get("claim_occurred", False)), "claim_at": r.get("claim_at"),
            "claim_qty": r.get("claim_quantity"), "claim_cost": r.get("claim_cost"),
        }
        for r in rows
    ]


def meta_from_reports(orders: list[dict[str, Any]], reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_order: dict[str, list[dict[str, Any]]] = {}
    for r in reports:
        by_order.setdefault(r["order_id"], []).append(r)
    return [build_meta(o, sorted(by_order.get(o["order_id"], []), key=lambda r: r["seq"])) for o in orders]


def dataset_from_contracts(
    *, products: list[dict[str, Any]], orders: list[dict[str, Any]], reports: list[dict[str, Any]],
    fai: list[dict[str, Any]], cell: list[dict[str, Any]], outcomes: list[dict[str, Any]],
    erp: list[dict[str, Any]] | None = None, factories: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """features.build.build() 가 읽는 모양. ground_truth 는 없다 — 운영에는 정답이 없다."""
    prods = products_from_master(products)
    ords = orders_from_contract(orders, prods)
    reps = reports_from_contract(reports)
    return {
        "products": prods,
        "factories": factories or [],
        "orders": ords,
        "order_meta": meta_from_reports(ords, reps),
        "factory_reports": reps,
        "fai_reports": fai_from_contract(fai),
        "cell_daily": cell_from_contract(cell),
        "erp_daily": erp_from_contract(erp or []),
        "quality_outcomes": outcomes_from_contract(outcomes),
    }
