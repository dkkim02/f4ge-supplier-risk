"""구조를 깨면 딥러닝이 이기나 — 검출률 오더 변동 sweep (2026-09-10).

**왜 이걸 재나.** 생성기가 `escape = internal × (1 − d)` 로 만들고 `d` 가 공장 상수라,
`escape = 보고 × κ` 는 이 세계에서 **정확한 항등식**이다. two_stage 는 정답 구조를 알고 들어간다.
실제 공장에서 그 구조는 근사일 뿐이다 — 검출은 오더·제품·작업조마다 흔들린다.

`assumptions.detection_order_sd` 를 올리면 검출률이 오더마다 logit 공간에서 흔들려
**κ 가 공장당 하나라는 전제가 깨진다.** 구조 위반 강도를 축으로 두 모델의 교차점을 찾는다.

    python scripts/structure_violation.py     → docs/_구조위반_sweep.md
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import evaluate, split_by_time
from f4ge_supplier_risk.features.build import FULL_COLS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import deep_kappa as DK
from f4ge_supplier_risk.models import hgb
from f4ge_supplier_risk.models import two_stage as TS

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
SIGMAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    rows = []
    for sg in SIGMAS:
        acc = {k: [] for k in ("two_stage", "hier", "hgb", "kap_cv", "d_sd")}
        for s in SEEDS:
            c = copy.deepcopy(cfg); c["seed"] = s
            c["assumptions"]["detection_order_sd"] = sg
            data = build_dataset(c)
            df = build(data)
            tr, te = split_by_time(df)

            # 위반이 실제로 얼마나 걸렸나 — 공장 안에서 진짜 κ 가 얼마나 흩어지나
            fac_of = {o["order_id"]: o["factory_id"] for o in data["orders"]}
            g = pd.DataFrame([{"f": fac_of[r["order_id"]], "d": r["detection_rate_order"]}
                              for r in data["ground_truth"] if r["order_id"] in fac_of])
            g["kap"] = (1 - g["d"]) / g["d"].clip(lower=1e-6)
            within = g.groupby("f")["kap"].agg(lambda v: v.std() / max(v.mean(), 1e-9))
            acc["kap_cv"].append(float(within.mean()))
            acc["d_sd"].append(float(g.groupby("f")["d"].std().mean()))

            acc["two_stage"].append(evaluate(te, TS.fit_predict(tr, te, FULL_COLS))["rank_corr_true"])
            acc["hier"].append(evaluate(te, DK.fit_predict(tr, te, (), seed=s, mode="hier", hidden=32, epochs=2000))["rank_corr_true"])
            acc["hgb"].append(evaluate(te, hgb.fit_predict(tr, te, FULL_COLS))["rank_corr_true"])
        r = dict(sigma=sg, **{k: np.mean(v) for k, v in acc.items()})
        r["격차"] = r["hier"] - r["two_stage"]
        rows.append(r)
        print(f"  σ={sg:<5} 공장내 κ 변동계수 {r['kap_cv']:.2f} · two_stage {r['two_stage']:+.3f} · "
              f"deep hier {r['hier']:+.3f} · hgb {r['hgb']:+.3f} · 격차 {r['격차']:+.3f}", flush=True)

    T = pd.DataFrame(rows)
    lines = [f"# 구조 위반 sweep — 검출률 오더 변동 ({len(SEEDS)} seed)", "",
             "`escape = 보고 × κ` 는 검출률이 공장 상수일 때만 정확한 항등식이다.",
             "`detection_order_sd` 를 올려 그 전제를 깨고 두 모델을 비교한다.", "",
             "| σ | 공장 내 κ 변동계수 | 공장 내 검출률 sd | two_stage | deep hier | hgb | 격차 |",
             "|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        lines.append(f"| {r.sigma} | {r.kap_cv:.2f} | {r.d_sd:.3f} | **{r.two_stage:+.3f}** | "
                     f"{r.hier:+.3f} | {r.hgb:+.3f} | {r['격차']:+.3f} |")
    out = Path(__file__).resolve().parents[1] / "docs" / "_구조위반_sweep.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + T.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
