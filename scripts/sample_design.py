"""표본 설계 — 입고검사를 늘리면 `d` 순위가 얼마나 오르나. 비용 대비 효과. 8 seed (2026-09-10).

**왜 모델이 아니라 표본인가.** 09-10 실측에서 two_stage · deep_kappa · hier_bayes 셋이
±0.003 안에서 같았다. 모델 쪽에 남은 것이 없다. 그리고 `d` 순위상관은 **+0.531** 로 낮은데
원인이 모델이 아니라 관측이다 — 검출률이 12곳에서 0.957~0.999 로 좁게 몰려 있고,
그 좁은 차이를 표본 중앙 **88개**로 재고 있다.

N-mixture 문헌(Barker 2018)이 같은 말을 한다 — 카운트만으로는 abundance 와 detection 이
분리되지 않고, 처방은 **auxiliary data**(double sampling · 진실을 아는 calibration subset)다.
이 스크립트가 그 처방의 비용을 계산한다.

**재는 것 — 두 설계를 같은 검사 유닛 예산에서 비교한다.**

    균일         모든 오더의 표본을 m 배로. ISO 2859-1 검사수준을 올리는 것에 해당
    calibration  오더의 q% 만 **전수 검사**, 나머지는 그대로. double sampling 설계
    층화 calib   같은 q% 를 **공장마다 같은 수**로 뽑는다. `d` 가 공장별 파라미터이므로
                 무작위 추출은 공장에 몰릴 수 있다. 문헌의 double sampling 은 원래 층화다

**생성기를 건드리지 않는다.** 진실(`escaped_total`, `order_qty`)에서 표본만 다시 뽑는다 —
`found ~ Hypergeometric(escaped, qty − escaped, n')`. 설계 계산이 원래 그런 것이다:
「n' 개를 검사했다면 무엇을 봤겠는가」. 생성기 난수 소비는 한 줄도 바뀌지 않는다.

⚠ 심각도 분해(critical/major/minor)와 `lot_result` 는 다시 뽑지 않는다.
   여기서 재는 지표(`d` 순위 · 유출 순위)가 총량 `(k, n)` 만 쓰기 때문이다.
   불일치 탐지·로트 판정을 이 표로 논하면 안 된다.

    python scripts/sample_design.py        → docs/_표본설계.md
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import evaluate, split_by_time
from f4ge_supplier_risk.features.build import FULL_COLS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import factory_params as fp
from f4ge_supplier_risk.models import two_stage

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
warnings.filterwarnings("ignore")

# (이름, 균일 배수, calibration 비율, 공장 층화 여부). 배수 None = 전수.
DESIGNS: tuple[tuple[str, float | None, float, bool], ...] = (
    ("현행 (ISO 2859-1 수준 II)", 1.0, 0.0, False),
    ("균일 ×2  (수준 III 상당)", 2.0, 0.0, False),
    ("균일 ×4", 4.0, 0.0, False),
    ("균일 ×8", 8.0, 0.0, False),
    ("균일 전수", None, 0.0, False),
    ("calibration 10% 무작위", 1.0, 0.10, False),
    ("calibration 25% 무작위", 1.0, 0.25, False),
    ("calibration 50% 무작위", 1.0, 0.50, False),
    ("calibration 10% 공장층화", 1.0, 0.10, True),
    ("calibration 25% 공장층화", 1.0, 0.25, True),
    ("calibration 50% 공장층화", 1.0, 0.50, True),
)


def resample(data: dict, mult: float | None, calib: float, strat: bool,
             rng: np.random.Generator) -> dict:
    """표본 크기를 바꿔 입고검사를 다시 뽑는다. 원본 `data` 는 건드리지 않는다."""
    qty = {o["order_id"]: int(o["order_qty"]) for o in data["orders"]}
    esc = {g["order_id"]: int(g["escaped_total"]) for g in data["ground_truth"]}
    fac = {o["order_id"]: o["factory_id"] for o in data["orders"]}

    picked: set[str] = set()
    if calib > 0 and strat:
        # 공장마다 같은 **비율**로 뽑는다 — 오더 수가 공장마다 달라 같은 수로 뽑으면
        # 오더가 적은 공장에서 비율이 100% 를 넘는다.
        by: dict[str, list[str]] = {}
        for oid in sorted(qty):
            by.setdefault(fac[oid], []).append(oid)
        for ids_f in by.values():
            m = int(round(len(ids_f) * calib))
            if m:
                picked |= set(rng.choice(ids_f, size=m, replace=False))
    elif calib > 0:
        ids = sorted(qty)
        picked = set(rng.choice(ids, size=int(round(len(ids) * calib)), replace=False))

    out = copy.deepcopy(data)
    for oc in out["quality_outcomes"]:
        oid = oc["order_id"]
        q = qty[oid]
        if oid in picked or mult is None:
            n = q
        else:
            n = min(q, int(round(oc["incoming_inspected_qty"] * mult)))
        e = max(0, min(esc[oid], q))
        oc["incoming_inspected_qty"] = n
        oc["incoming_reject_qty"] = int(rng.hypergeometric(e, max(q - e, 0), n)) if n > 0 else 0
    return out


def score(data: dict) -> dict[str, float]:
    df = build(data)
    tr, te = split_by_time(df)
    facs = {f["factory_id"]: f for f in data["factories"]}

    P = fp.estimate(tr)
    d_true = [facs[i]["detection_rate"] for i in P.index]
    p_true_f = df.groupby("factory_id")["true_internal_rate"].mean().reindex(P.index)
    p_ord = fp.order_internal_rate(te, P)

    return {
        "d_rho": float(spearmanr(P["d"], d_true).statistic),
        "p_rho": float(spearmanr(P["p"], p_true_f).statistic),
        "p_order_rho": float(spearmanr(p_ord, te["true_internal_rate"]).statistic),
        "esc_rho": float(evaluate(te, two_stage.fit_predict(tr, te, FULL_COLS))["rank_corr_true"]),
        "units": float(df["y_inspected"].sum()),
        "events": float(df["y_reject"].sum()),
    }


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    raw = {}
    for s in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = s
        raw[s] = build_dataset(c)
    print(f"  준비 완료 — 오더 {len(raw[SEEDS[0]]['orders'])}건 · 공장 12곳", flush=True)

    rows = []
    for name, mult, calib, strat in DESIGNS:
        acc = []
        for s in SEEDS:
            rng = np.random.default_rng(90210 + s)   # 설계 재추출 전용. 생성기 스트림과 무관하다
            acc.append(score(resample(raw[s], mult, calib, strat, rng)))
        A = pd.DataFrame(acc)
        rows.append({
            "설계": name,
            "d_rho": A["d_rho"].mean(), "d_sd": A["d_rho"].std(), "d_min": A["d_rho"].min(),
            "d_양수": int((A["d_rho"] > 0).sum()),
            "p_rho": A["p_rho"].mean(), "esc_rho": A["esc_rho"].mean(),
            "유닛": A["units"].mean(), "사건": A["events"].mean(),
        })
        r = rows[-1]
        print(f"  {name:26s} d {r['d_rho']:+.3f} ±{r['d_sd']:.3f} ({r['d_양수']}/8)"
              f"  유출 {r['esc_rho']:+.3f}  검사유닛 {r['유닛']:>8,.0f}  사건 {r['사건']:>5.1f}", flush=True)
    T = pd.DataFrame(rows)

    base = T.iloc[0]
    total_units = float(T.loc[T["설계"] == "균일 전수", "유닛"].iloc[0])   # 전수 = 총 출하 유닛
    T["검사율"] = T["유닛"] / total_units
    T["Δd"] = T["d_rho"] - base["d_rho"]
    T["추가유닛"] = T["유닛"] - base["유닛"]
    T["유닛만당Δd"] = np.where(T["추가유닛"] > 0, T["Δd"] / (T["추가유닛"] / 10000.0), np.nan)

    L = ["# 표본 설계 — 입고검사 증설의 비용 대비 효과", "",
         "8 seed. 생성기는 건드리지 않고 진실에서 표본만 다시 뽑았다",
         "(`found ~ Hypergeometric(escaped, qty − escaped, n')`).", "",
         "`d` = 공장 검출률 추정치와 진실의 Spearman. `유출` = two_stage 오더 순위상관.",
         "`검사유닛` = 전 기간 입고검사 유닛 합 = 비용. `검사율` = 총 출하 유닛 대비.", "",
         "| 설계 | d 순위상관 | ±sd | 최소 | 양수 | 유출 | 검사유닛 | 검사율 | 라벨 사건 | Δd | 1만 유닛당 Δd |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        eff = "—" if not np.isfinite(r["유닛만당Δd"]) else f"{r['유닛만당Δd']:+.4f}"
        L.append(f"| {r['설계']} | {r['d_rho']:+.3f} | {r['d_sd']:.3f} | {r['d_min']:+.3f} | {r['d_양수']}/8 "
                 f"| {r['esc_rho']:+.3f} | {r['유닛']:,.0f} | {r['검사율']:.0%} | {r['사건']:.1f} "
                 f"| {r['Δd']:+.3f} | {eff} |")

    out = Path(__file__).resolve().parents[1] / "docs" / "_표본설계.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
