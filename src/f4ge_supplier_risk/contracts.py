"""생성 레코드 → 계약 행. 계약(contracts/*.schema.json)이 생성기와 어긋나지 않는지를 테스트가 이 함수들로 확인한다.

이름 규칙은 FactoryOS 계약을 승계한다 — ISO-8601 UTC · 소문자 prefix id · `*_quantity` · `product_id` 대신 `product_code`.
생성기 내부 이름(`order_qty`, `scrap_qty`, `incoming_reject_qty`)은 바꾸지 않고 여기서만 번역한다.
"""

from __future__ import annotations

from typing import Any

TENANT = "ten_f4ge"  # 생성기는 단일 테넌트다. 실서비스에서는 수신 측이 채운다.


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def order_row(o: dict[str, Any]) -> dict[str, Any]:
    return _drop_none({
        "schema_version": "supplier_risk.order.v1", "tenant_id": TENANT,
        "order_id": o["order_id"], "factory_id": o["factory_id"],
        "product_code": o["product_id"].removeprefix("prd_"), "market": o["market"],
        "order_quantity": int(o["order_qty"]), "ordered_at": o["ordered_at"], "promised_date": o["promised_date"],
        "quoted_at": o.get("quoted_at"), "quote_response_h": o.get("quote_response_h"), "unit_price": o.get("unit_price"),
        "aql_major": o.get("aql_major"), "aql_minor": o.get("aql_minor"),
        "material_lot_id": o.get("material_lot_id"), "material_supplier": o.get("material_supplier"),
    })


def report_row(r: dict[str, Any], factory_id: str, source: str = "mes_api") -> dict[str, Any]:
    base = {
        "schema_version": "supplier_risk.factory_report.v1", "tenant_id": TENANT,
        "order_id": r["order_id"], "factory_id": factory_id, "seq": int(r["seq"]),
        "due_at": r["due_at"], "is_missing": bool(r["is_missing"]), "source": source,
        "reported_at": r.get("reported_at"),
    }
    if r["is_missing"]:
        return base
    return _drop_none({
        **base,
        "produced_quantity": int(r["produced_qty"]), "scrap_quantity": int(r["scrap_qty"]),
        "rework_quantity": int(r["rework_qty"]), "stage": r["stage"],
        "promised_date_reported": r["promised_date_reported"], "issue_flag": bool(r["issue_flag"]),
        "inspected_quantity": r.get("inspected_qty"), "inspection_type": r.get("inspection_type"),
        "reject_quantity": r.get("reject_qty"), "defect_type": r.get("defect_type"),
        "material_lot_id": r.get("material_lot_id"), "material_supplier": r.get("material_supplier"),
        "material_consumed_quantity": r.get("material_consumed_qty"),
        "machine_id": r.get("machine_id"), "shift": r.get("shift"), "operator_count": r.get("operator_count"),
        "overtime_hours": r.get("overtime_hours"), "photo_taken_at": r.get("photo_taken_at"),
    })


def fai_row(f: dict[str, Any], factory_id: str) -> dict[str, Any]:
    return {
        "schema_version": "supplier_risk.fai_report.v1", "tenant_id": TENANT,
        "order_id": f["order_id"], "factory_id": factory_id, "fai_at": f["fai_at"],
        "dim_count": int(f["dim_count"]), "margin_min": f["margin_min"], "margin_mean": f["margin_mean"],
        "margin_p25": f["margin_p25"], "out_of_tol_count": int(f["out_of_tol_count"]),
    }


def cell_row(c: dict[str, Any], factory_id: str) -> dict[str, Any]:
    return {
        "schema_version": "supplier_risk.cell_daily.v1", "tenant_id": TENANT,
        "order_id": c["order_id"], "factory_id": factory_id, "day_index": int(c["day_index"]),
        "uptime_ratio": c["uptime_ratio"], "cycle_time_median": c["cycle_time_median"],
        "anomaly_event_count": int(c["anomaly_event_count"]), "tool_change_count": int(c["tool_change_count"]),
        "produced_by_counter": int(c["produced_by_counter"]),
    }


def outcome_row(q: dict[str, Any], factory_id: str) -> dict[str, Any]:
    return _drop_none({
        "schema_version": "supplier_risk.order_quality_outcome.v1", "tenant_id": TENANT,
        "order_id": q["order_id"], "factory_id": factory_id,
        "shipped_at": q.get("shipped_at"), "arrived_at": q.get("arrived_at"), "aql_code": q.get("aql_code"),
        "inspected_quantity": int(q["incoming_inspected_qty"]), "reject_quantity": int(q["incoming_reject_qty"]),
        "reject_critical": q.get("incoming_reject_critical"), "reject_major": q.get("incoming_reject_major"),
        "reject_minor": q.get("incoming_reject_minor"), "accept_number_major": q.get("accept_number_major"),
        "lot_result": q["lot_result"], "reject_reason": q.get("reject_reason"),
        "label_available_at": q["label_available_at"],
        "claim_occurred": q.get("claim_occurred"), "claim_at": q.get("claim_at"),
        "claim_quantity": q.get("claim_qty"), "claim_cost": q.get("claim_cost"),
    })
