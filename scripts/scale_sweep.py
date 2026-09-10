"""규모가 커지면 딥러닝이 conjugate 를 따라잡나 — 공장 수 sweep (2026-09-10).

12곳에서는 two_stage(Gamma-Poisson conjugate)가 이긴다. 공장이 늘면 두 가지가 동시에 변한다:
  · 라벨 사건 수가 늘어 conjugate 의 prior 이점이 줄어든다
  · **κ 를 공장 특성의 함수로 배우는 것**(deep_kappa feat/hier)의 학습 표본이 늘어난다
    — conjugate 는 이걸 못 한다. 신규 공장은 늘 pooled 로 떨어진다.

그래서 콜드스타트(학습에서 못 본 공장)를 함께 잰다. 교차점이 있으면 로드맵 근거가 된다.

    python scripts/scale_sweep.py           → docs/_규모_sweep.md
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
from f4ge_supplier_risk.models import two_stage as TS

SEEDS = (20260907, 11, 4242, 7)     # 규모가 커서 4 seed. 12곳은 8 seed 값이 dl_fair_bench 에 있다
SCALES = (12, 24, 48, 96)


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    rows, cold = [], []
    for nf in SCALES:
        acc = {k: [] for k in ("two_stage", "embed", "hier", "feat", "orders", "events")}
        cacc = {k: [] for k in ("two_stage", "hier", "feat", "n", "held")}
        for s in SEEDS:
            c = copy.deepcopy(cfg); c["seed"] = s; c["scale"]["factories"] = nf
            df = build(build_dataset(c))
            tr, te = split_by_time(df)
            acc["orders"].append(len(df)); acc["events"].append(int(tr["y_reject"].sum()))
            acc["two_stage"].append(evaluate(te, TS.fit_predict(tr, te, FULL_COLS))["rank_corr_true"])
            for m in ("embed", "hier", "feat"):
                acc[m].append(evaluate(te, DK.fit_predict(tr, te, (), seed=s, mode=m))["rank_corr_true"])

            fids = sorted(set(tr["factory_id"]))
            held = set(fids[-max(3, len(fids) // 4):])
            tr2 = tr[~tr["factory_id"].isin(held)]
            te2 = te[te["factory_id"].isin(held)]
            if len(te2) >= 10 and len(tr2) >= 60:
                cacc["n"].append(len(te2)); cacc["held"].append(len(held))
                cacc["two_stage"].append(evaluate(te2, TS.fit_predict(tr2, te2, FULL_COLS))["rank_corr_true"])
                for m in ("hier", "feat"):
                    cacc[m].append(evaluate(te2, DK.fit_predict(tr2, te2, (), seed=s, mode=m))["rank_corr_true"])
            print(f"  공장 {nf:3d} seed {s} 완료", flush=True)

        rows.append(dict(공장=nf, 오더=int(np.mean(acc["orders"])), 사건=int(np.mean(acc["events"])),
                         **{m: np.mean(acc[m]) for m in ("two_stage", "embed", "hier", "feat")}))
        if cacc["n"]:
            cold.append(dict(공장=nf, 홀드아웃=int(np.mean(cacc["held"])), 테스트오더=int(np.mean(cacc["n"])),
                             **{m: np.mean(cacc[m]) for m in ("two_stage", "hier", "feat")},
                             hier_sd=np.std(cacc["hier"])))
        print(f"── 공장 {nf}: " + " · ".join(f"{m} {np.mean(acc[m]):+.3f}" for m in ("two_stage","embed","hier","feat")), flush=True)

    T, C = pd.DataFrame(rows), pd.DataFrame(cold)
    lines = [f"# 규모 sweep — 공장 수를 늘리면 딥러닝이 따라잡나 ({len(SEEDS)} seed)", "",
             "## 전체 (모든 공장 채점)", "",
             "| 공장 | 오더 | 라벨 사건 | two_stage | deep embed | deep hier | deep feat | 격차 |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        gap = max(r.embed, r.hier, r.feat) - r.two_stage
        lines.append(f"| {r.공장} | {r.오더} | {r.사건} | **{r.two_stage:+.3f}** | {r.embed:+.3f} | {r.hier:+.3f} | {r.feat:+.3f} | {gap:+.3f} |")
    lines += ["", "## 콜드스타트 (학습에서 못 본 공장만 채점)", "",
              "| 공장 | 홀드아웃 | 테스트 오더 | two_stage | deep hier | deep feat | 격차 |", "|---|---|---|---|---|---|---|"]
    for _, r in C.iterrows():
        gap = max(r.hier, r.feat) - r.two_stage
        lines.append(f"| {r.공장} | {r.홀드아웃} | {r.테스트오더} | **{r.two_stage:+.3f}** | {r.hier:+.3f} | {r.feat:+.3f} | {gap:+.3f} |")
    out = Path(__file__).resolve().parents[1] / "docs" / "_규모_sweep.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + T.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\n콜드스타트\n" + C.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
