"""시나리오 — MES 가 12곳 전부 연동된 상태에서 무엇이 달라지나.

⚠ 기본값이 아니다. CTO 확인 사실은 **a 3곳 · b 9곳**이고(`configs/generator.yaml`),
이 스크립트는 설치가 끝난 뒤의 상태를 가정해 상한을 보는 용도다. 게이트 `mes_coverage: 0.25` 는
이 시나리오에서 통과하지 않는다 — 통과시키려면 근거(설치 실적)부터 바뀌어야 한다.

재는 것 셋 (기존 sweep 에 없던 것)
  ① 계층 증분 — 희석이 사라지면 "MES 를 받으면 좋아진다"가 몇 조건에서 성립하나
  ② 배분 정확도 — 견적 시점 공장 선택 top-1
  ③ internal 타깃 — 지금 제조 상태를 읽는 정확도와 Cell 의 기여

    python scripts/scenario_full_mes.py
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
from f4ge_supplier_risk.models import baseline, two_stage

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
_E = 1e-12
RUNS = (("1단 L0", baseline, "L0"), ("1단 L0+Cell", baseline, "L0+Cell"),
        ("1단 L0+Cell+MES", baseline, "L0+Cell+MES"), ("2단 L0+Cell+MES", two_stage, "L0+Cell+MES"))


def gmean_by(v, k):
    return {str(x): float(np.exp(np.mean(np.log(np.clip(v[k == x], _E, None))))) for x in np.unique(k)}


def ridge(tr, te, cols, y):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0))
    m.fit(tr[list(cols)], y)
    return m.predict(te[list(cols)])


def main(n_a: int = 12) -> None:
    cfg = config.load("configs/generator.yaml")
    esc = {r[0]: {"전체": [], "공장안": []} for r in RUNS}
    pick = {"L0 이력만": [], "2단(κ)": []}
    inter = {"L0 만": [], "L0+Cell": [], "L0+Cell+MES": []}

    for seed in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        c["factory_types"] = {"a": n_a, "b": 12 - n_a, "c": 0}
        data = build_dataset(c)
        fam = {f["factory_id"]: f["product_id"] for f in data["factories"]}
        df = build(data)
        tr, te = split_by_time(df)
        fid = te["factory_id"].to_numpy()
        truth = te["true_escape_rate"].to_numpy(float)

        # ① escape 예측
        preds = {}
        for label, model, layer in RUNS:
            p = model.fit_predict(tr, te, LAYERS[layer])
            preds[label] = p
            esc[label]["전체"].append(spearmanr(p, truth).statistic)
            ws = [spearmanr(p[fid == f], truth[fid == f]).statistic for f in np.unique(fid)]
            esc[label]["공장안"].append(float(np.nanmedian(ws)))

        # ② 배분 — 제품군마다 3곳 중 고르기
        true_lvl = gmean_by(truth, fid)
        lvl = {"L0 이력만": gmean_by(te["l0_hist_escape_rate"].to_numpy(float), fid),
               "2단(κ)": gmean_by(preds["2단 L0+Cell+MES"], fid)}
        for family in sorted(set(fam.values())):
            cands = sorted([f for f in true_lvl if fam[f] == family])
            if len(cands) < 2:
                continue
            best = cands[int(np.argmin([true_lvl[f] for f in cands]))]
            for m in pick:
                pick[m].append(cands[int(np.argmin([lvl[m][f] for f in cands]))] == best)

        # ③ internal 타깃 — 보고로 학습, 진짜 internal 순위로 평가
        m_tr = tr["l1_rep_produced"].to_numpy(float) > 0
        y = np.log(tr.loc[m_tr, "l1_reported_defect_rate"].to_numpy(float) + 5e-4)
        ti = te["true_internal_rate"].to_numpy(float)
        for name, layer in (("L0 만", "L0"), ("L0+Cell", "L0+Cell"), ("L0+Cell+MES", "L0+Cell+MES")):
            p = ridge(tr[m_tr], te, LAYERS[layer], y)
            ws = [spearmanr(p[fid == f], ti[fid == f]).statistic for f in np.unique(fid)]
            inter[name].append(float(np.nanmedian(ws)))

    print(f"시나리오 a={n_a} · b={12-n_a} · 8 seed\n")
    print("① escape 예측 (진짜 escape 순위상관)")
    print("| 모델 | 전체 | 같은 공장 안 |")
    print("|---|---|---|")
    for label, _, _ in RUNS:
        a, b = np.array(esc[label]["전체"]), np.array(esc[label]["공장안"])
        print("| %s | %+.3f ±%.3f | %+.3f ±%.3f |" % (label, a.mean(), a.std(), b.mean(), b.std()))
    d = np.array(esc["2단 L0+Cell+MES"]["전체"]) - np.array(esc["1단 L0"]["전체"])
    print("\nMES 증분 (2단 − 1단 L0): %+.3f · 양수 %d/8 · 최악 %+.3f" % (d.mean(), (d > 0).sum(), d.min()))

    print("\n② 배분 top-1 (제품군 4 × seed 8 = 32건, 무작위 33.3%)")
    for m, v in pick.items():
        print("   %-10s %.1f%% (%d/%d)" % (m, 100 * np.mean(v), sum(v), len(v)))

    print("\n③ 지금 제조 상태 (진짜 internal, 같은 공장 안 순위상관)")
    for name, v in inter.items():
        print("   %-12s %+.3f ±%.3f" % (name, np.mean(v), np.std(v)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
