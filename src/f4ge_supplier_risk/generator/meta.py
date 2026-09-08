"""L0′ — 우리가 자동으로 남기는 것.

공장 협조가 필요 없고, 공장이 조작할 수 없고, **오늘부터 기록 가능하다.**
늦게 보내면 늦은 게 그대로 남는다는 점이 이 계층의 성질이다.

⚠ 이 중 셋은 지나가면 복원 불가능하다 — 보고 도착 시각, 질의→응답 시간, 납기 변경 시점.
결과값은 나중에 소급 정리가 되지만 시각은 사라진다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# MES 선택 필드 — 공장이 채우지 않으면 빈칸. 09-08 저녁: 자재·잔업은 ERP 로 옮겨 여기서 빠졌다.
T2_FIELDS = ("inspected_qty", "defect_type", "machine_id")


def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def build_meta(order: dict[str, Any], reports: list[dict[str, Any]]) -> dict[str, Any]:
    filed = [r for r in reports if not r["is_missing"]]
    delays = [(_dt(r["reported_at"]) - _dt(r["due_at"])).total_seconds() / 86400 for r in filed]

    promised_seen: list[str] = []
    for r in filed:
        p = r.get("promised_date_reported")
        if p and (not promised_seen or p != promised_seen[-1]):
            promised_seen.append(p)

    last_change_progress = 0.0
    if len(promised_seen) > 1 and filed:
        for i, r in enumerate(filed):
            if r.get("promised_date_reported") == promised_seen[-1]:
                last_change_progress = (i + 1) / len(filed)
                break

    return {
        "order_id": order["order_id"],
        "report_expected": len(reports),
        "report_missing_count": sum(1 for r in reports if r["is_missing"]),
        # 집계 주기가 공장마다 달라(일/시프트/주) 회차 수가 다르다 — 비율로도 낸다
        "report_missing_rate": round(sum(1 for r in reports if r["is_missing"]) / len(reports), 4) if reports else 0.0,
        "report_delay_days_mean": round(sum(delays) / len(delays), 3) if delays else None,
        "report_delay_days_max": round(max(delays), 3) if delays else None,
        "field_blank_count": sum(1 for r in filed for f in T2_FIELDS if f not in r),
        "field_blank_rate": round(sum(1 for r in filed for f in T2_FIELDS if f not in r) / (len(filed) * len(T2_FIELDS)), 4) if filed else 0.0,
        "promised_date_change_count": max(0, len(promised_seen) - 1),
        "promised_date_change_last_progress": round(last_change_progress, 3),
        "quote_response_h": order["quote_response_h"],
        "price_zscore": order["price_zscore"],
        "lead_slack": order["lead_slack"],
    }
