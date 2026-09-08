"""검증 게이트.

기존 프로젝트의 `qp calibrate` 방식을 승계한다 — **값을 고치려면 근거부터 바꿔야 한다.**
"모델이 잘 나오게" 타깃을 조정하는 순간 이 파일은 의미를 잃는다.

게이트는 두 종류다.
  · **수치 타깃** — `configs/calibration-targets.yaml` 의 값과 대조
  · **구조 검사** — 편향·불일치가 설계대로 작동하는지. 숫자 하나로는 못 잡는다
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _measure(cfg: dict[str, Any], data: Path) -> dict[str, float]:
    orders = _load(data / "orders.jsonl")
    truth = _load(data / "ground_truth.jsonl")
    outcomes = _load(data / "quality_outcomes.jsonl")
    reports = _load(data / "factory_reports.jsonl")

    qty = np.array([o["order_qty"] for o in orders], dtype=float)
    internal = np.array([t["internal_defect_rate_true"] for t in truth])
    escape = np.array([t["escape_rate_true"] for t in truth])

    filed = [r for r in reports if not r["is_missing"]]
    last_by_order: dict[str, dict[str, Any]] = {}
    for r in filed:
        cur = last_by_order.get(r["order_id"])
        if cur is None or r["seq"] > cur["seq"]:
            last_by_order[r["order_id"]] = r
    scrap = np.array([r.get("scrap_qty", 0) for r in last_by_order.values()], dtype=float)
    rework = np.array([r.get("rework_qty", 0) for r in last_by_order.values()], dtype=float)
    caught = scrap + rework

    # 로트 3,201~10,000 구간의 표본 크기가 ISO 표(200)와 맞는지
    big = [
        o
        for o in outcomes
        if 3201
        <= dict(zip([x["order_id"] for x in orders], [x["order_qty"] for x in orders]))[
            o["order_id"]
        ]
        <= 10000
    ]

    return {
        "order_qty_median": float(np.median(qty)),
        "order_qty_lognormal_sigma": float(np.std(np.log(qty))),
        "internal_defect_rate_median": float(np.median(internal)),
        "escape_rate_median": float(np.median(escape)),
        "escape_rate_max": float(escape.max()),
        "lot_reject_rate": float(np.mean([o["lot_result"] == "reject" for o in outcomes])),
        "scrap_share_of_internal": float(scrap.sum() / caught.sum()) if caught.sum() else 0.0,
        "aql_major": float(orders[0]["aql_major"]),
        "aql_minor": float(orders[0]["aql_minor"]),
        "sample_size_lot_3201_10000": float(np.median([o["incoming_inspected_qty"] for o in big]))
        if big
        else float("nan"),
        "transit_days_us_west": float(cfg["market_transit_days"]["us_west"]),
        "transit_days_us_east": float(cfg["market_transit_days"]["us_east"]),
        "transit_days_europe": float(cfg["market_transit_days"]["eu"]),
        "mes_interval_days": float(cfg["scale"]["mes_interval_days"]),
        "cell_coverage": float(
            sum(1 for f in _load(data / "factories.jsonl") if f["has_cell"])
            / max(len(_load(data / "factories.jsonl")), 1)
        ),
        "mes_coverage": float(
            sum(1 for f in _load(data / "factories.jsonl") if f["has_mes"])
            / max(len(_load(data / "factories.jsonl")), 1)
        ),
    }


def _check(name: str, spec: dict[str, Any], value: float) -> tuple[bool, str]:
    if np.isnan(value):
        return False, "측정값 없음"
    if "target" in spec:
        tol = float(spec.get("tol", 0.0))
        target = float(spec["target"])
        if target == 0:
            ok = abs(value) <= tol
        else:
            ok = abs(value - target) / abs(target) <= tol + 1e-12
        return ok, f"{value:.5g} vs {target:.5g} (±{tol:.0%})"
    lo = float(spec.get("min", -np.inf))
    hi = float(spec.get("max", np.inf))
    return lo <= value <= hi, f"{value:.5g} in [{lo:.4g}, {hi:.4g}]"


def _structural(data: Path) -> list[tuple[str, bool, str]]:
    """구조 검사 — 편향과 불일치가 설계대로 작동하는가."""
    truth = _load(data / "ground_truth.jsonl")
    factories = _load(data / "factories.jsonl")
    fai = _load(data / "fai_reports.jsonl")
    outcomes = _load(data / "quality_outcomes.jsonl")
    orders = _load(data / "orders.jsonl")
    bias_by_fac = {f["factory_id"]: f["report_bias"] for f in factories}
    fac_of = {o["order_id"]: o["factory_id"] for o in orders}

    out: list[tuple[str, bool, str]] = []

    # 1. 보고와 실제가 실제로 갈리는가
    gaps = np.array(
        [t["reported_vs_true_gap"] for t in truth if t["reported_vs_true_gap"] is not None]
    )
    out.append(
        (
            "보고 vs 실제 격차가 양수 쪽으로 치우친다",
            float(np.median(gaps)) > 0,
            f"중앙값 {np.median(gaps):+.4f} · 양수 비율 {np.mean(gaps > 0):.1%}",
        )
    )

    # 2. 편향이 큰 공장에서 격차가 더 큰가 (편향 파라미터가 실제로 작동하는가)
    bias = np.array(
        [bias_by_fac[fac_of[t["order_id"]]] for t in truth if t["reported_vs_true_gap"] is not None]
    )
    r = float(np.corrcoef(bias, gaps)[0, 1])
    out.append(("report_bias 가 낮을수록 격차가 크다", r < -0.05, f"r = {r:+.3f}"))

    # 3. FAI 공차 여유가 내부 불량률과 음의 상관인가 (10% 시점 신호가 살아 있는가)
    tmap = {t["order_id"]: t["internal_defect_rate_true"] for t in truth}
    m = np.array([f["margin_min"] for f in fai])
    d = np.array([tmap[f["order_id"]] for f in fai])
    r2 = float(np.corrcoef(m, d)[0, 1])
    out.append(("FAI margin_min 이 내부 불량률과 음의 상관", r2 < -0.2, f"r = {r2:+.3f}"))

    # 4. ⚠ 누수 — y 와 지나치게 상관된 관측값이 없는가
    #    기존 프로젝트에서 이 원칙이 KAMP 실데이터의 합성 라벨을 학습 전에 잡아냈다.
    y = np.array([o["incoming_reject_qty"] / max(o["incoming_inspected_qty"], 1) for o in outcomes])
    omap = {o["order_id"]: o for o in orders}
    cand = {
        "lead_slack": np.array([omap[o["order_id"]]["lead_slack"] for o in outcomes]),
        "price_zscore": np.array([omap[o["order_id"]]["price_zscore"] for o in outcomes]),
        "order_qty": np.array([omap[o["order_id"]]["order_qty"] for o in outcomes], dtype=float),
    }
    worst_name, worst = "", 0.0
    for k, v in cand.items():
        rr = abs(float(np.corrcoef(v, y)[0, 1]))
        if rr > worst:
            worst_name, worst = k, rr
    out.append((f"최대 |L0 피처–y 상관| < 0.40 ({worst_name})", worst < 0.40, f"{worst:.3f}"))

    return out


def run_gates(cfg: dict[str, Any], targets: dict[str, Any], data: Path) -> bool:
    measured = _measure(cfg, data)
    specs = targets["targets"]

    print("─ 수치 타깃 " + "─" * 52)
    n_ok = 0
    for name, spec in specs.items():
        if name not in measured:
            print(f"  ?  {name:32s} 미측정")
            continue
        ok, detail = _check(name, spec, measured[name])
        n_ok += ok
        print(f"  {'✓' if ok else '✗'}  {name:32s} {detail}")

    print("─ 구조 검사 " + "─" * 52)
    struct = _structural(data)
    for label, ok, detail in struct:
        n_ok += ok
        print(f"  {'✓' if ok else '✗'}  {label:44s} {detail}")

    total = len(specs) + len(struct)
    print("─" * 64)
    print(f"  {n_ok}/{total} 통과")
    return n_ok == total
