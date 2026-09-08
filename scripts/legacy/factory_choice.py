"""배분 결정을 잰다 — 견적 시점에 **어느 공장에 줄 것인가**.

기존 지표(`rank_corr_true`)는 **오더 순위**를 잰다. 배분은 다른 질문이다:
같은 제품군의 공장 3곳 중 어디가 나은가. 결정 단위가 오더가 아니라 **공장**이다.

생성기 구조상 공장 12곳 = 제품군 4종 × 3곳이고, 배분은 그 3곳 안의 선택이다
(`generator/orders.py`). 그래서 제품군마다 따로 재고, seed 8개로 돌린다.

무엇을 비교하나
  L0 이력만   견적 시점에 우리 손에 있는 것만. 과거 입고검사 실적(평활)
  1단 L0      L0 + L0′ + FAI 로 오더를 예측하고 공장별 기하평균 → 공장 수준
  2단(κ)      1단 + 공장별 κ. 보고가 오는 공장(a)에서만 실제 신호가 붙는다

진실은 그 공장 테스트 오더의 `true_escape_rate` 기하평균이다.
지표는 셋 — top-1 적중(예측 1위가 진짜 1위인가), 3곳 순위 켄달, 후회
(고른 공장 escape ÷ 진짜 최선 공장 escape. 1.0 이면 최선을 골랐다).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import LAYERS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import baseline, two_stage

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
_EPS = 1e-12


def _gmean_by(vals: np.ndarray, keys: np.ndarray) -> dict[str, float]:
    out = {}
    for k in np.unique(keys):
        m = keys == k
        out[str(k)] = float(np.exp(np.mean(np.log(np.clip(vals[m], _EPS, None)))))
    return out


def _kendall3(pred: list[float], true: list[float]) -> float:
    """3곳 순위의 켄달 tau — 쌍 3개 중 몇 개의 순서가 맞나 (1.0 / 0.333 / -0.333 / -1.0)."""
    n, agree = 0, 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            n += 1
            agree += np.sign(pred[i] - pred[j]) == np.sign(true[i] - true[j])
    return (2.0 * agree / n) - 1.0 if n else 0.0


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    methods = ("L0 이력만", "1단 L0", "2단(κ)")
    hits = {m: [] for m in methods}
    taus = {m: [] for m in methods}
    regret = {m: [] for m in methods}
    hits_a = {m: [] for m in methods}   # MES 연동 공장이 후보에 있는 제품군만

    for seed in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        data = build_dataset(c)
        fam = {f["factory_id"]: f["product_id"] for f in data["factories"]}
        typ = {f["factory_id"]: f["factory_type"] for f in data["factories"]}
        df = build(data)
        tr, te = split_by_time(df)

        fid = te["factory_id"].to_numpy()
        truth_lvl = _gmean_by(te["true_escape_rate"].to_numpy(float), fid)

        # 견적 시점에 손에 있는 L0 이력 하나 — 그 공장 테스트 오더들의 평활 실적 평균
        hist_lvl = _gmean_by(te["l0_hist_escape_rate"].to_numpy(float), fid)
        p1 = baseline.fit_predict(tr, te, LAYERS["L0"])
        p2 = two_stage.fit_predict(tr, te, LAYERS["L0+Cell+MES"])
        lvl = {"L0 이력만": hist_lvl, "1단 L0": _gmean_by(p1, fid), "2단(κ)": _gmean_by(p2, fid)}

        for family in sorted(set(fam.values())):
            cands = sorted([f for f in truth_lvl if fam[f] == family])
            if len(cands) < 2:
                continue
            t = [truth_lvl[f] for f in cands]
            best_true = cands[int(np.argmin(t))]
            has_a = any(typ[f] == "a" for f in cands)
            for m in methods:
                p = [lvl[m][f] for f in cands]
                pick = cands[int(np.argmin(p))]
                hits[m].append(pick == best_true)
                taus[m].append(_kendall3(p, t))
                regret[m].append(truth_lvl[pick] / max(truth_lvl[best_true], _EPS))
                if has_a:
                    hits_a[m].append(pick == best_true)

    n = len(hits[methods[0]])
    print(f"제품군 4종 × seed 8 = 결정 {n}건 · 후보 3곳 중 1곳 선택 (무작위 기대 적중 33.3%)\n")
    print("| 방법 | top-1 적중 | 3곳 순위 켄달 | 후회(중앙) | 후회(최악) |")
    print("|---|---|---|---|---|")
    for m in methods:
        print("| %s | **%.1f%%** (%d/%d) | %+.2f | ×%.2f | ×%.2f |" % (
            m, 100 * np.mean(hits[m]), sum(hits[m]), n,
            np.mean(taus[m]), np.median(regret[m]), max(regret[m])))
    print("\nMES 연동 공장이 후보에 있는 제품군만 (%d건):" % len(hits_a[methods[0]]))
    for m in methods:
        print("  %-10s top-1 %.1f%%" % (m, 100 * np.mean(hits_a[m])))


if __name__ == "__main__":
    main()
