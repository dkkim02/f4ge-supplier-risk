"""계층 증분을 **공장 유형별로** 잰다 — 설치 가치는 희석을 걷어내야 보인다.

전체 평균은 MES 오더가 25% 뿐이라 L0 로 내려간 75% 에 묻힌다(프로젝트_정리.md §5-1).
모델은 전체 학습 구간으로 한 번 적합하고, 평가는 테스트 구간을 유형별로 갈라서 한다.

  a  MES + Cell   → L0 / +Cell / +MES 셋 다 실제 신호가 있다
  b  Cell 만      → +MES 는 결측(중앙값 대치). +Cell 까지만 의미 있다
  c  없음         → 전부 L0 와 같다. 대조군

seed 8개. 2단 모델의 seed 편차가 ±0.2 라 3 seed 로는 인접 수준을 구분하지 못했다.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import LAYERS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import baseline, two_stage

SEEDS = (20260907, 11, 4242, 7, 101, 2026, 31337, 909)
RUNS = (
    ("1단 L0", baseline, "L0"),
    ("1단 L0+Cell", baseline, "L0+Cell"),
    ("1단 L0+Cell+MES", baseline, "L0+Cell+MES"),
    ("2단 L0+Cell+MES", two_stage, "L0+Cell+MES"),
)
TYPES = ("a", "b", "c", "전체")
MIXED = "혼합 a:2단 · b,c:1단 L0"


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    res: dict[tuple[str, str], list[float]] = {(r[0], t): [] for r in RUNS for t in TYPES}
    res.update({(MIXED, t): [] for t in TYPES})
    n_by_type: dict[str, list[int]] = {t: [] for t in TYPES}

    for seed in SEEDS:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        data = build_dataset(c)
        type_of = {f["factory_id"]: f["factory_type"] for f in data["factories"]}
        df = build(data)
        tr, te = split_by_time(df)
        te_type = te["factory_id"].map(type_of).to_numpy()
        truth = te["true_escape_rate"].to_numpy()

        preds = {}
        for label, model, layer in RUNS:
            pred = model.fit_predict(tr, te, LAYERS[layer])
            preds[label] = pred
            for t in TYPES:
                m = np.ones(len(te), dtype=bool) if t == "전체" else te_type == t
                res[(label, t)].append(float(spearmanr(pred[m], truth[m]).statistic))
        # 혼합 — MES 가 있는 오더만 2단, 나머지는 1단 L0. 두 모델 모두 유닛당 escape 확률이라 눈금이 같다.
        mixed = np.where(te_type == "a", preds["2단 L0+Cell+MES"], preds["1단 L0"])
        for t in TYPES:
            m = np.ones(len(te), dtype=bool) if t == "전체" else te_type == t
            res[(MIXED, t)].append(float(spearmanr(mixed[m], truth[m]).statistic))
        for t in TYPES:
            n_by_type[t].append(int(len(te) if t == "전체" else (te_type == t).sum()))
        print(f"  seed {seed} 완료", flush=True)

    print(f"\n{len(SEEDS)} seed · 테스트 오더 수: " + " · ".join(f"{t} {int(np.mean(n_by_type[t]))}" for t in TYPES))
    hdr = f"{'':20s}" + "".join(f"{t:>18s}" for t in TYPES)
    print(hdr)
    lines = ["| 모델·계층 | " + " | ".join(TYPES) + " |", "|---|" + "---|" * len(TYPES)]
    for label in [r[0] for r in RUNS] + [MIXED]:
        cells = []
        for t in TYPES:
            v = np.array(res[(label, t)])
            cells.append(f"{v.mean():+.3f} ±{v.std():.3f}")
        print(f"{label:24s}" + "".join(f"{c:>18s}" for c in cells))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # 증분 — 같은 seed 안에서의 차이를 seed 별로 내고 평균/부호 일관성을 본다
    print("\n증분 (seed 별 차이의 평균 · 양수 seed 수)")
    lines.append("\n| 증분 | " + " | ".join(TYPES) + " |")
    lines.append("|---|" + "---|" * len(TYPES))
    for name, hi, lo in (
        ("+Cell (1단)", "1단 L0+Cell", "1단 L0"),
        ("+MES (1단)", "1단 L0+Cell+MES", "1단 L0+Cell"),
        ("+MES (2단)", "2단 L0+Cell+MES", "1단 L0+Cell"),
        ("혼합 − 2단 전체적용", MIXED, "2단 L0+Cell+MES"),
    ):
        cells = []
        for t in TYPES:
            d = np.array(res[(hi, t)]) - np.array(res[(lo, t)])
            cells.append(f"{d.mean():+.3f} ({int((d > 0).sum())}/{len(d)})")
        print(f"{name:20s}" + "".join(f"{c:>18s}" for c in cells))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    out = Path(__file__).resolve().parents[1] / "docs" / "_유형별_표.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
