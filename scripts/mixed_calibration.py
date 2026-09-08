"""혼합 모델(a: 2단 · b,c: 1단 L0)의 눈금 맞추기 — 무엇이 유형 간 순서를 지키나.

혼합을 그대로 쓰면 유형 내부 순위는 좋아지는데 전체 순위가 seed 마다 −0.34~+0.21 흔들린다
(프로젝트_정리.md §5-1). 두 모델의 예측 눈금이 다르기 때문이다 — 그런데 눈금만의 문제가 아니다.
2단은 테스트 구간에서 κ 를 **온라인으로 갱신**해 공장 수준을 따라가고, 1단 L0 는 학습 구간에 고정된다.

후보 다섯:
  2단 전체         기준선. 보고 없는 오더도 2단(사전분포 × κ)
  혼합 raw         a: 2단 · b,c: L0. 눈금 안 맞춤
  혼합 B  비율     2단 을 학습 구간 a 오더에서 L0 와 중앙값이 같도록 상수배
  혼합 C  분해     b,c: 2단(공장 수준, 온라인) × L0 의 공장 내 편차 — 두 모델이 잘하는 것만 가져온다
  혼합 D  적층     학습 구간에서 (log 2단, log L0) → 라벨 로지스틱. ⚠ 2단 학습 예측은 in-sample κ 라 낙관적
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import LAYERS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import baseline, two_stage

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
FULL, L0 = LAYERS["L0+Cell+MES"], LAYERS["L0"]
_EPS = 1e-7


def _stack(p2_tr, p0_tr, tr, p2, p0) -> np.ndarray:
    x = np.log(np.column_stack([p2_tr, p0_tr]) + _EPS)
    n = tr["y_inspected"].to_numpy(float)
    k = tr["y_reject"].to_numpy(float)
    x2 = np.vstack([x, x])
    y2 = np.r_[np.ones(len(x)), np.zeros(len(x))]
    w2 = np.r_[k, np.maximum(n - k, 0)]
    keep = w2 > 0
    m = LogisticRegression(C=1.0, max_iter=2000).fit(x2[keep], y2[keep], sample_weight=w2[keep])
    return m.predict_proba(np.log(np.column_stack([p2, p0]) + _EPS))[:, 1]


def candidates(tr, te, is_a_tr, is_a_te) -> dict[str, np.ndarray]:
    p2 = two_stage.fit_predict(tr, te, FULL)
    p0 = baseline.fit_predict(tr, te, L0)
    p2_tr = two_stage.fit_predict(tr, tr, FULL)
    p0_tr = baseline.fit_predict(tr, tr, L0)

    raw = np.where(is_a_te, p2, p0)

    ratio = np.median(p0_tr[is_a_tr]) / max(np.median(p2_tr[is_a_tr]), _EPS)
    b = np.where(is_a_te, p2 * ratio, p0)

    # C — 공장별 L0 기하평균으로 나눠 공장 내 편차만 남기고, 공장 수준은 2단이 준다
    fac = te["factory_id"].to_numpy()
    logp0 = np.log(p0 + _EPS)
    fac_mean = pd.Series(logp0).groupby(fac).transform("mean").to_numpy()
    c = np.where(is_a_te, p2, p2 * np.exp(logp0 - fac_mean))

    d = _stack(p2_tr, p0_tr, tr, p2, p0)
    return {"2단 전체": p2, "혼합 raw": raw, "혼합 B 비율": b, "혼합 C 분해": c, "혼합 D 적층": d}


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    names = ["2단 전체", "혼합 raw", "혼합 B 비율", "혼합 C 분해", "혼합 D 적층"]
    types = ("a", "b", "c", "전체")
    res = {(nm, t): [] for nm in names for t in types}

    for seed in SEEDS:
        cf = copy.deepcopy(cfg)
        cf["seed"] = seed
        data = build_dataset(cf)
        typ = {f["factory_id"]: f["factory_type"] for f in data["factories"]}
        df = build(data)
        tr, te = split_by_time(df)
        tt = te["factory_id"].map(typ).to_numpy()
        is_a_tr = (tr["factory_id"].map(typ) == "a").to_numpy()
        truth = te["true_escape_rate"].to_numpy()
        for nm, p in candidates(tr, te, is_a_tr, tt == "a").items():
            for t in types:
                m = np.ones(len(te), bool) if t == "전체" else tt == t
                res[(nm, t)].append(float(spearmanr(p[m], truth[m]).statistic))
        row = "  ".join(f"{nm} {res[(nm, '전체')][-1]:+.3f}" for nm in names)
        print(f"seed {seed:>9}  {row}", flush=True)

    print(f"\n{len(SEEDS)} seed 평균 ±sd")
    lines = ["| 후보 | a | b | c | 전체 | 2단 대비 전체 (양수 seed) |", "|---|---|---|---|---|---|"]
    base = np.array(res[("2단 전체", "전체")])
    for nm in names:
        cells = [f"{np.mean(res[(nm, t)]):+.3f} ±{np.std(res[(nm, t)]):.3f}" for t in types]
        d = np.array(res[(nm, "전체")]) - base
        cmp_ = f"{d.mean():+.3f} ({int((d > 0).sum())}/{len(d)}, 최악 {d.min():+.3f})"
        print(f"  {nm:12s} " + "  ".join(f"{c:>16s}" for c in cells) + f"   {cmp_}")
        lines.append(f"| {nm} | " + " | ".join(cells) + f" | {cmp_} |")
    out = Path(__file__).resolve().parents[1] / "docs" / "_혼합_눈금_표.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
