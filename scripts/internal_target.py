"""타깃을 escape 가 아니라 internal 로 두면 무엇이 달라지나.

「지금 공장에서 제조가 잘 되고 있나」는 **internal**(공정에서 발생하는 불량률)을 묻는 질문이고,
「이 오더가 우리에게 불량을 보낼까」는 **escape** 를 묻는 질문이다. 같은 공장 안에서는 두 값이
같은 순서지만(검출률이 상수라 escape = internal × 상수), **공장을 줄 세울 때는 순위가 갈린다**
(공장 단위 Spearman 0.67 — 자체검사가 촘촘한 공장이 escape 로는 앞선다).

기존 측정은 전부 escape 를 타깃으로 했고 그때 「Cell 단독 증분 ±0.02」가 나왔다. 이 스크립트는
타깃을 internal 로 바꿔 같은 것을 다시 잰다 — 설비 신호는 원인 쪽이라 escape 보다 internal 에 가깝다.

  ① 상한   진짜 internal 을 타깃으로 준 회귀. 운영 불가 — 「신호에 정보가 있나」만 본다
  ② 배포형 a 공장 보고로 학습 → b 공장(MES 미연동) 채점, 진짜 internal 순위로 평가

    python scripts/internal_target.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import LAYERS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)


def ridge(tr, te, cols, y):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0))
    m.fit(tr[list(cols)], y)
    return m.predict(te[list(cols)])


def within(pred, tgt, fid):
    ws = [spearmanr(pred[fid == f], tgt[fid == f]).statistic for f in np.unique(fid) if (fid == f).sum() >= 5]
    return float(np.nanmedian(ws)) if ws else np.nan


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    up = {(t, lay): [] for t in ("internal", "escape") for lay in ("L0", "L0+Cell")}
    tr_ = {lay: {"전체": [], "공장안": []} for lay in ("L0 만", "L0+Cell")}

    for seed in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        data = build_dataset(c)
        df = build(data)
        tr, te = split_by_time(df)
        fid = te["factory_id"].to_numpy()

        # ① 상한 — 진짜값을 타깃으로
        for t, col in (("internal", "true_internal_rate"), ("escape", "true_escape_rate")):
            ytr = np.log(tr[col].to_numpy(float))
            tgt = te[col].to_numpy(float)
            for lay in ("L0", "L0+Cell"):
                up[(t, lay)].append(within(ridge(tr, te, LAYERS[lay], ytr), tgt, fid))

        # ② 배포형 — a 공장 보고로 학습, b 공장 평가
        m_tr = tr["l1_rep_produced"].to_numpy(float) > 0
        m_te = te["l1_rep_produced"].to_numpy(float) <= 0
        if not m_tr.any() or not m_te.any():
            continue
        y = np.log(tr.loc[m_tr, "l1_reported_defect_rate"].to_numpy(float) + 5e-4)
        ti = te.loc[m_te, "true_internal_rate"].to_numpy(float)
        fb = te.loc[m_te, "factory_id"].to_numpy()
        for name, lay in (("L0 만", "L0"), ("L0+Cell", "L0+Cell")):
            p = ridge(tr[m_tr], te[m_te], LAYERS[lay], y)
            tr_[name]["전체"].append(spearmanr(p, ti).statistic)
            tr_[name]["공장안"].append(within(p, ti, fb))

    lines = ["## ① 상한 — 진짜값을 타깃으로 준 회귀 (같은 공장 안 순위상관, 8 seed)", "",
             "| 타깃 | L0 | L0+Cell | Cell 증분 |", "|---|---|---|---|"]
    for t, name in (("internal", "진짜 internal"), ("escape", "진짜 escape")):
        a, b = np.array(up[(t, "L0")]), np.array(up[(t, "L0+Cell")])
        d = b - a
        lines.append("| %s | %+.3f | %+.3f | **%+.3f** (%d/8 양수) |"
                     % (name, np.nanmean(a), np.nanmean(b), np.nanmean(d), int((d > 0).sum())))
    lines += ["", "## ② 배포형 — a 공장 보고로 학습 → b 공장 채점 (진짜 internal 순위 대비)", "",
              "| 쓰는 신호 | b 공장 전체 | 같은 공장 안 |", "|---|---|---|"]
    for name in ("L0 만", "L0+Cell"):
        a, b = np.array(tr_[name]["전체"]), np.array(tr_[name]["공장안"])
        lines.append("| %s | %+.3f ±%.3f | %+.3f ±%.3f |"
                     % (name, np.nanmean(a), np.nanstd(a), np.nanmean(b), np.nanstd(b)))
    d = np.array(tr_["L0+Cell"]["전체"]) - np.array(tr_["L0 만"]["전체"])
    lines.append("")
    lines.append("Cell 증분(배포형, b 공장 전체): **%+.3f** · 양수 %d/%d · 최악 %+.3f"
                 % (np.nanmean(d), int((d > 0).sum()), len(d), np.nanmin(d)))
    body = "\n".join(lines)
    print(body)
    out = Path(__file__).resolve().parents[1] / "docs" / "_internal_타깃_표.md"
    out.write_text("# internal 을 타깃으로 두면 — 8 seed\n\n" + body + "\n", encoding="utf-8")
    print("\n→", out)


if __name__ == "__main__":
    main()
