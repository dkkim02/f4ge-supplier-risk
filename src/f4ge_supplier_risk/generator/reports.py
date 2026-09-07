"""L1 — 공장 주간 보고. **편향·지연·결측이 전부 여기 들어간다.**

보고는 진실의 함수다. `production.build_truth` 가 먼저 돌아야 한다.

설계의 핵심은 **편향을 어디에 거느냐**이다.

  자기신고 항목(자체검사 불량, 진척 수량, 특이사항)  → 편향을 건다
  물리적 소모량과 시각(자재 소진, 잔업, 보고 도착)     → 실제값 그대로 둔다

이렇게 해야 "한 지표를 좋게 만들면 다른 축이 어긋난다"가 데이터 안에서 성립한다.
Goodhart 문헌의 표현대로 — 모든 지표를 결과 지표와 짝지으면 게이밍이 다른 곳에서
비용으로 드러난다. 우리 교차검증 쌍(수집명세 §3)이 그 구현이다.

**노이즈가 아니라 편향이라는 점**이 중요하다. 평균이 어긋나야 불일치 탐지가 값을 한다.

## 메커니즘 정답값

축소 보고는 한 가지 방식으로만 일어나지 않는다. 수량을 부풀리거나, 불량을 줄이거나,
납기를 고수하거나, 이상 없다고 하거나, 지난주 사진을 다시 보낸다.
**어느 방식이었는지를 `_mechanisms` 로 남긴다** — 사유 분해가 맞는지 확인할
유일한 근거이고, 실데이터에서는 영원히 알 수 없는 값이다.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np

from f4ge_supplier_risk.generator.production import counts_at, progress_at
from f4ge_supplier_risk.rng import stream

_STAGES = ("setup", "machining", "finishing", "inspection", "packing")
_DEFECT_TYPES = ("dimension", "surface", "function", "material", "other")


def _stage_for(progress: float) -> str:
    if progress <= 0.0:
        return "setup"
    if progress < 0.75:
        return "machining"
    if progress < 0.95:
        return "finishing"
    if progress < 1.0:
        return "inspection"
    return "packing"


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_reports(
    cfg: dict[str, Any],
    order: dict[str, Any],
    factory: dict[str, Any],
    product: dict[str, Any],
    latents: dict[str, float],
    truth: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """``(보고 회차들, 메커니즘 정답값)``.

    메커니즘을 보고 리스트에 섞어 넣지 않는다 — 보고를 순회하는 코드가
    전부 예외 처리를 해야 하고, 실제로 테스트 세 개가 그렇게 깨졌다.
    """
    rng = stream(cfg["seed"], "report", order["order_id"])
    qty = order["order_qty"]
    interval = cfg["scale"]["report_interval_days"]
    n_reports = max(2, int(np.ceil(truth["actual_days"] / interval)))

    # 이 오더가 얼마나 곤란한가 (0~1). 지연·결측·편향이 전부 이 값을 따라간다.
    trouble = float(
        np.clip(
            0.5 * latents["state_t"]
            + 0.6 * latents["schedule_pressure"]
            + 3.0 * (latents["internal_defect_rate_true"] - 0.03),
            0.0,
            1.0,
        )
    )

    out: list[dict[str, Any]] = []
    last_photo_at = None
    reported_promised = order["_promised_dt"]
    # 메커니즘별 정답값. 사유 분해를 검증할 유일한 근거다 — 실데이터에서는
    # "이 공장이 수량을 부풀렸는지" 를 영원히 알 수 없다.
    mech = {
        "inflation_max": 0.0,  # 수량 부풀리기
        "shrink_min": 1.0,  # 불량 축소 (작을수록 심하다)
        "promise_stale_days": 0.0,  # 낙관 보고 — 실제 완료가 보고 납기를 넘긴 최대 일수
        "issue_suppressed": 0,  # 문제 은폐 — 곤란한데 "이상 없음" 이라고 한 회차 수
        "photo_reused": 0,  # 증거 재활용
    }

    for k in range(1, n_reports + 1):
        elapsed = min(k * interval, truth["actual_days"])
        due = order["_ordered_dt"] + timedelta(days=k * interval)
        progress = progress_at(truth, elapsed)
        true_c = counts_at(truth, qty, progress)

        # ── 결측: 곤란하고 규율이 낮은 공장이 회차를 통째로 빠뜨린다 ──
        p_missing = (1.0 - factory["report_discipline"]) * (0.10 + 0.35 * trouble)
        if rng.random() < p_missing:
            out.append(
                {
                    "order_id": order["order_id"],
                    "seq": k,
                    "due_at": _iso(due),
                    "reported_at": None,
                    "is_missing": True,
                }
            )
            continue

        # ── 지연: 곤란한 공장은 연락이 느려진다. 조작 불가능한 신호다 ──
        delay = float(rng.gamma(1.5, (1.0 - factory["report_discipline"]) * (1.0 + 4.0 * trouble)))
        reported_at = due + timedelta(days=delay)

        # ── 진척 부풀리기: 늦은 공장이 앞서간 것처럼 보고한다 ──
        behind = max(0.0, (k * interval) / truth["actual_days"] - progress)
        inflation = factory["bias_sensitivity"] * behind * 0.5
        rep_produced = int(min(qty, round(true_c["produced"] * (1.0 + inflation))))
        mech["inflation_max"] = max(mech["inflation_max"], inflation)

        # ── 불량 축소 보고. 나쁠수록 더 줄인다 ──
        # 편향(평균이 어긋남)과 노이즈(들쭉날쭉함)는 다른 것이고 **둘 다 있어야 한다.**
        # 노이즈가 없으면 보고가 진짜 불량률을 지나치게 잘 대표해서, 보고를 쓰는 모델의
        # 성능이 실제보다 좋게 나온다. 대충 센 것 · 사람마다 다른 기준 ·
        # 형식적으로 채워 넣은 숫자가 전부 여기 들어간다.
        noise = float(rng.lognormal(0.0, cfg["assumptions"]["report_noise_cv"]))
        shrink = float(
            np.clip(
                factory["report_bias"] * (1.0 - factory["bias_sensitivity"] * trouble) * noise,
                0.02,
                2.0,
            )
        )
        mech["shrink_min"] = min(mech["shrink_min"], shrink)
        rep_scrap = round(float(true_c["scrap"] * min(1.0, shrink + 0.35)))  # 폐기는 감추기 어렵다
        rep_rework = round(float(true_c["rework"] * shrink))  # 재작업이 가장 감추기 쉽다

        # ── 납기 재보고: 명백해질 때까지 원래 날짜를 고수한다 ──
        projected_end = order["_ordered_dt"] + timedelta(days=truth["actual_days"])
        if (
            projected_end > reported_promised + timedelta(days=3)
            and rng.random() < 0.35 + 0.4 * trouble
        ):
            reported_promised = projected_end + timedelta(days=float(rng.uniform(0, 4)))
        mech["promise_stale_days"] = max(
            mech["promise_stale_days"], (projected_end - reported_promised).total_seconds() / 86400
        )

        row: dict[str, Any] = {
            "order_id": order["order_id"],
            "seq": k,
            "due_at": _iso(due),
            "reported_at": _iso(reported_at),
            "is_missing": False,
            "produced_qty": rep_produced,
            "scrap_qty": rep_scrap,
            "rework_qty": rep_rework,
            "stage": _stage_for(progress),
            "promised_date_reported": _iso(reported_promised),
            # 특이사항: 실제로 문제가 있어도 편향 때문에 "없음"이 나온다
            "issue_flag": bool(trouble > 0.35 and rng.random() > shrink),
        }
        if trouble > 0.35 and not row["issue_flag"]:
            mech["issue_suppressed"] += 1

        # ── T2: t2_compliance 확률로만 채워진다 ──
        if rng.random() < factory["t2_compliance"]:
            insp = round(float(true_c["produced"] * float(rng.uniform(0.3, 1.0))))
            true_found = true_c["scrap"] + true_c["rework"]
            row |= {
                "inspected_qty": insp,
                "inspection_type": "full" if insp >= true_c["produced"] else "sampling",
                "reject_qty": round(float(true_found * shrink)),
                "defect_type": _DEFECT_TYPES[int(rng.integers(len(_DEFECT_TYPES)))],
                # 자재 소진은 물리량이라 **실제값**을 따라간다 → 수량 부풀리기의 대조군
                "material_consumed_qty": round(
                    (true_c["produced"] + true_c["scrap"]) * product["material_per_unit"], 1
                ),
                "material_lot_id": order["material_lot_id"],
                "material_supplier": order["material_supplier"],
                "machine_id": f"mch_{factory['factory_id'][-2:]}_{int(rng.integers(1, 4))}",
                "shift": ("day", "swing", "night")[int(rng.integers(3))],
                "operator_count": int(rng.integers(2, 7)),
                # 잔업도 실제 곤란을 따라간다 → 불량 은폐의 대조군
                "overtime_hours": round(float(rng.gamma(2.0, 3.0)) * (1.0 + 3.0 * trouble), 1),
            }
            # 사진: 곤란하면 지난주 것을 재사용한다
            if last_photo_at is not None and rng.random() < 0.15 + 0.45 * trouble:
                row["photo_taken_at"] = _iso(last_photo_at)
                mech["photo_reused"] += 1
            else:
                taken = reported_at - timedelta(hours=float(rng.uniform(1, 20)))
                row["photo_taken_at"] = _iso(taken)
                last_photo_at = taken

        out.append(row)

    return out, mech


def build_fai(
    cfg: dict[str, Any],
    order: dict[str, Any],
    factory: dict[str, Any],
    latents: dict[str, float],
    truth: dict[str, Any],
) -> dict[str, Any] | None:
    """초도품 검사 — 오더 초반의 전 치수 실측.

    **진행률 10% 시점에 얻을 수 있는 유일한 상세 신호**다. 합/불이 아니라
    공차 여유(margin)라는 연속값이라 정보량이 훨씬 크다. 첫 물건부터 여유가
    없으면 공구가 마모될수록 넘어간다.
    """
    rng = stream(cfg["seed"], "fai", order["order_id"])
    if rng.random() > factory["t2_compliance"] * 0.9:
        return None

    n_dim = int(rng.integers(8, 25))
    # 공차 여유의 중심이 내부 불량률과 직결된다. 불량률이 높을수록 여유가 없다.
    center = 0.55 - 4.0 * latents["internal_defect_rate_true"]
    margins = rng.normal(center, 0.18, size=n_dim)

    fai_at = order["_ordered_dt"] + timedelta(
        days=truth["actual_days"] * 0.12 + float(rng.uniform(0, 2))
    )
    return {
        "order_id": order["order_id"],
        "fai_at": _iso(fai_at),
        "dim_count": n_dim,
        "margin_min": round(float(margins.min()), 4),
        "margin_mean": round(float(margins.mean()), 4),
        "margin_p25": round(float(np.percentile(margins, 25)), 4),
        "out_of_tol_count": int((margins < 0).sum()),
    }
