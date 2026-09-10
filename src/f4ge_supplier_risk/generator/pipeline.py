"""생성 파이프라인 — 명세 §8 순서 그대로.

3번(잠재값 → 두 불량률)이 끝나야 4번(보고)이 가능하다. **보고는 진실의 함수**이기 때문이다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from f4ge_supplier_risk.generator import erp, labels, latent, masters, meta, orders, production, reports

OUT_FILES = (
    "factories.jsonl",
    "products.jsonl",
    "orders.jsonl",
    "order_meta.jsonl",
    "factory_reports.jsonl",
    "fai_reports.jsonl",
    "cell_daily.jsonl",
    "erp_daily.jsonl",
    "quality_outcomes.jsonl",
    "ground_truth.jsonl",
)

# 모델이 보면 안 되는 생성 편의 필드
_PRIVATE_PREFIX = "_"


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    {k: v for k, v in r.items() if not k.startswith(_PRIVATE_PREFIX)},
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_dataset(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """파일로 쓰지 않고 메모리에 만든다. 민감도 스윕이 이 경로를 쓴다."""
    products = masters.build_products(cfg)
    factories = masters.build_factories(cfg, products)
    order_rows = orders.build_orders(cfg, factories, products)

    fac_by_id = {f["factory_id"]: f for f in factories}
    prod_by_id = {p["product_id"]: p for p in products}

    n_weeks = cfg["scale"]["months"] * 5 + 20
    states = {
        f["factory_id"]: latent.factory_state_series(cfg, f["factory_id"], n_weeks)
        for f in factories
    }

    all_reports: list[dict[str, Any]] = []
    all_fai: list[dict[str, Any]] = []
    all_outcomes: list[dict[str, Any]] = []
    all_meta: list[dict[str, Any]] = []
    all_truth: list[dict[str, Any]] = []
    all_cell: list[dict[str, Any]] = []
    all_erp: list[dict[str, Any]] = []

    for o in order_rows:
        factory = fac_by_id[o["factory_id"]]
        product = prod_by_id[o["product_id"]]
        week = int((o["_ordered_dt"] - orders.EPOCH).days // 7)

        lat = latent.order_latents(
            cfg, o, factory, product, float(states[factory["factory_id"]][week])
        )
        truth = production.build_truth(cfg, o, factory, lat)

        rep, mech = reports.build_reports(cfg, o, factory, product, lat, truth)
        all_cell.extend(reports.build_cell_daily(cfg, o, factory, lat, truth))
        all_erp.extend(erp.build_erp_daily(cfg, o, factory, product, lat, truth))
        fai = reports.build_fai(cfg, o, factory, lat, truth)
        outcome = labels.build_outcome(cfg, o, truth, lat)

        all_reports.extend(rep)
        if fai:
            all_fai.append(fai)
        all_outcomes.append(outcome)
        all_meta.append(meta.build_meta(o, rep))

        # 보고된 불량률 vs 실제 — 불일치의 **정답값**.
        # 실데이터에서는 영원히 알 수 없는 값이라, 여기서 재두지 않으면
        # 우리 핵심 기능의 성능을 한 번도 측정할 수 없다.
        filed = [r for r in rep if not r["is_missing"]]
        rep_defects = sum(r.get("scrap_qty", 0) + r.get("rework_qty", 0) for r in filed[-1:])
        rep_produced = filed[-1]["produced_qty"] if filed else 0
        reported_rate = rep_defects / rep_produced if rep_produced else None

        all_truth.append(
            {
                "order_id": o["order_id"],
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in lat.items()},
                "actual_days": round(truth["actual_days"], 2),
                "true_defect_count": truth["n_defect"],
                "escaped_total": truth["escaped_total"],
                "detection_rate_true": factory["detection_rate"],
                "detection_rate_order": lat.get("detection_rate_order", factory["detection_rate"]),
                "reported_defect_rate": round(reported_rate, 6)
                if reported_rate is not None
                else None,
                **{f"mech_{k}": v for k, v in mech.items()},
                "reported_vs_true_gap": (
                    round(lat["internal_defect_rate_true"] - reported_rate, 6)
                    if reported_rate is not None
                    else None
                ),
            }
        )

    return {
        "factories": factories,
        "products": products,
        "orders": order_rows,
        "order_meta": all_meta,
        "factory_reports": all_reports,
        "fai_reports": all_fai,
        "cell_daily": all_cell,  # L2 — CellOS, 12곳 전부
        "erp_daily": all_erp,  # L3 — ERP, 12곳 전부. 채우는 필드는 공장마다 다르다
        "quality_outcomes": all_outcomes,
        "ground_truth": all_truth,
    }


def generate(cfg: dict[str, Any], out_dir: Path | str) -> dict[str, int]:
    data = build_dataset(cfg)
    out = Path(out_dir)
    for name, rows in data.items():
        _write(out / f"{name}.jsonl", rows)
    return {k: len(v) for k, v in data.items() if k not in ("cell_daily", "erp_daily")}
