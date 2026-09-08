"""오더 스케줄 36개월 — **한 공장은 한 번에 한 오더만 돈다** (사용자 전제, 2026-09-08 저녁).

한 오더 = 한 공장 × 한 제품 × 수량 × 약속 납기(CTO 답변 6번: 분할 발주 없음).
공장이 제품에 전속이므로 제품군을 고르면 후보 공장이 3곳으로 좁혀진다 —
**이 3곳 중에서 고르는 것이 배분 결정이고, 모델이 답해야 할 질문도 그것이다.**

발주는 공장별로 **직렬**이다: 다음 오더는 이전 오더의 약속 납기 뒤 `order_gap_days` 안에 발주된다.
그래서 같은 공장에서 기간이 겹치는 오더가 없고, 부하(동시 진행 오더 수)라는 개념이 없다 —
09-08 이전의 `load_index` 는 이 전제에서 상수 0 이라 제거했다. 오더 수는 기간 ÷ (리드 + 간격)에서 나온다(약 300건).

`lead_slack` 이 이 파일의 핵심 산출물이다. (약속 납기 − 표준 소요) / 표준 소요.
음수면 애초에 무리한 일정이고, 그것이 서두름을 거쳐 불량으로 간다.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from f4ge_supplier_risk.rng import stream

EPOCH = datetime(2024, 1, 1, tzinfo=UTC)
_MATERIAL_LOT_SPAN_DAYS = 30  # 자재 로트 하나가 걸치는 기간
_SUPPLIERS = ("sup_a", "sup_b", "sup_c")


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# 이 공장은 우리 오더만 돌리지 않는다. 하루 중 이 오더에 실제로 쓰이는 시간.
_EFFECTIVE_SEC_PER_DAY = 6 * 3600
_FRONT_DAYS = 18.0  # 자재 입고(특수 소재는 2주 이상) + 셋업
_BACK_DAYS = 10.0  # 외주 후처리(도금·열처리) + 검사 + 포장


def _standard_days(qty: int, product: dict[str, Any]) -> float:
    """표준 소요일.

    금속 절삭 부품은 **가공 시간이 달력 시간을 지배하지 않는다.** 수량 700개 ×
    42초면 순 가공은 하루가 안 되는데 실제 납기는 몇 주다. 대기·자재·후처리가
    대부분이라 앞뒤 상수가 가공 항보다 크다.
    """
    return _FRONT_DAYS + qty * product["nominal_cycle_sec"] / _EFFECTIVE_SEC_PER_DAY + _BACK_DAYS


def build_orders(
    cfg: dict[str, Any], factories: list[dict[str, Any]], products: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    sc = cfg["scale"]
    oc = cfg["order"]
    seed = cfg["seed"]
    prod_by_id = {p["product_id"]: p for p in products}

    markets = list(oc["market_mix"])
    market_p = [oc["market_mix"][m] for m in markets]

    horizon_days = sc["months"] * 30.4
    gap_lo, gap_hi = sc.get("order_gap_days", [0, 5])
    sigma = oc["qty_lognormal_sigma"]
    mu = math.log(oc["qty_lognormal_median"])

    orders: list[dict[str, Any]] = []
    for factory in factories:
        product_id = factory["product_id"]
        product = prod_by_id[product_id]
        # 공장마다 시작 시차를 둔다 — 12곳이 같은 날 시작하면 납기가 한 줄로 몰린다
        day = float(stream(seed, "order_start", factory["factory_id"]).uniform(0, 30))
        k = 0
        while day < horizon_days:
            rng = stream(seed, "order", f"{factory['factory_id']}:{k}")
            ordered_at = EPOCH + timedelta(days=day)
            qty = int(np.clip(round(math.exp(rng.normal(mu, sigma))), oc["qty_min"], oc["qty_max"]))

            # 납기는 여유율에서 유도한다. 여유율이 우리가 통제하고 싶은 값이고
            # 납기는 그 결과이기 때문이다 — 반대로 하면 여유율 분포를 못 잡는다.
            std_days = _standard_days(qty, product)
            lead_slack = float(
                np.clip(rng.normal(oc["lead_slack_mean"], oc["lead_slack_sd"]), -0.35, 1.2)
            )
            lead_days = std_days * (1.0 + lead_slack)
            promised = ordered_at + timedelta(days=lead_days)

            market = markets[int(rng.choice(len(markets), p=market_p))]

            # 견적 — 단가는 제품·수량의 함수 + 공장 편차. 비정상적으로 싸면 위험 신호
            base_price = 1200.0 * product["nominal_cycle_sec"] / 60.0
            price = base_price * float(rng.lognormal(0.0, 0.12)) * (1.0 - 0.03 * math.log(qty / 800))

            lot_day = int(day // _MATERIAL_LOT_SPAN_DAYS)
            supplier = _SUPPLIERS[int(rng.integers(len(_SUPPLIERS)))]

            orders.append(
                {
                    "order_id": "",  # 전 공장을 시간순으로 정렬한 뒤 매긴다
                    "factory_id": factory["factory_id"],
                    "product_id": product_id,
                    "market": market,
                    "order_qty": qty,
                    "ordered_at": _iso(ordered_at),
                    "promised_date": _iso(promised),
                    "unit_price": round(price, 2),
                    "quoted_at": _iso(ordered_at - timedelta(days=float(rng.uniform(2, 10)))),
                    "quote_response_h": round(float(rng.gamma(2.0, 12.0)), 1),
                    "lead_slack": round(lead_slack, 4),
                    "aql_major": cfg["aql"]["major"],
                    "aql_minor": cfg["aql"]["minor"],
                    # 아래 둘은 계약 필드가 아니라 생성 편의값 — 별도로 떼어 쓴다
                    "material_lot_id": f"mtl_{supplier}_{lot_day:03d}",
                    "material_supplier": supplier,
                    "_ordered_dt": ordered_at,
                    "_promised_dt": promised,
                    "_lead_days": lead_days,
                    "_std_days": std_days,
                }
            )
            # 다음 오더는 이 오더의 약속 납기 뒤에 발주된다 — 한 공장에 한 오더
            day = day + lead_days + float(rng.uniform(gap_lo, gap_hi))
            k += 1

    orders.sort(key=lambda o: o["_ordered_dt"])
    for i, o in enumerate(orders):
        o["order_id"] = f"ord_{i + 1:05d}"
    _attach_price_zscore(orders)
    return orders


def _attach_price_zscore(orders: list[dict[str, Any]]) -> None:
    """제품군 안에서의 단가 z-score. 우리가 계산하는 값이라 L0 에 속한다."""
    by_product: dict[str, list[dict[str, Any]]] = {}
    for o in orders:
        by_product.setdefault(o["product_id"], []).append(o)
    for group in by_product.values():
        prices = np.array([o["unit_price"] for o in group])
        mean, sd = prices.mean(), prices.std() or 1.0
        for o in group:
            o["price_zscore"] = round(float((o["unit_price"] - mean) / sd), 4)
