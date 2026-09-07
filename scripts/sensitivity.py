"""민감도 분석 — 결론이 **근거 없는 가정**에 얼마나 의존하는가.

생성기 파라미터 다섯 개는 실측 근거가 없다(docs/생성기_캘리브레이션.md §2.3~2.4).
그 값들을 흔들었을 때 결론이 따라 흔들리면, 그건 데이터에서 나온 결론이 아니라
우리가 넣어 둔 가정을 되읽은 것이다.

재는 것 넷:
  · `L0`        거래 이력 + 우리 쪽 메타만으로 어디까지 되나 (1단 모델)
  · `L0+L0′+L1` 공장 보고까지 받고 **2단 모델**로 쓰면  ← **계층 증분이 이 프로젝트의 주장**
  · `gap 복원`  보고와 실제의 격차를 관측값에서 복원할 수 있나  ← **핵심 상품**
  · `불합격률`   생성 데이터가 여전히 실무 범위(1~10%)에 있나
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
from f4ge_supplier_risk.evaluation.metrics import evaluate, split_by_time
from f4ge_supplier_risk.features.build import LAYERS, build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import baseline, two_stage

SEEDS = (20260907, 11, 4242)

SWEEPS: dict[str, tuple[str, tuple]] = {
    "factory_capability_sd": ("공장 간 품질 분산", (0.5, 1.0, 1.5, 2.0)),
    "report_bias_mean": ("보고 편향 크기", (0.3, 0.5, 0.7, 0.9)),
    "bias_sensitivity": ("나쁠수록 더 축소", (0.0, 0.25, 0.5, 1.0)),
    "within_factory_ar1": ("시간 자기상관", (0.0, 0.3, 0.6, 0.85)),
    "severity_critical": ("critical 비중", (0.05, 0.15, 0.25, 0.40)),
    "report_noise_cv": ("보고의 들쭉날쭉함", (0.0, 0.45, 0.9, 1.5)),
}


def apply_value(cfg: dict, key: str, value: float) -> dict:
    c = copy.deepcopy(cfg)
    if key == "severity_critical":
        rest = 1.0 - value
        c["assumptions"]["defect_severity_mix"] = {
            "critical": value,
            "major": rest * 2 / 3,
            "minor": rest / 3,
        }
    else:
        c["assumptions"][key] = value
    return c


def gap_recovery(train, test, cols) -> float:
    """관측값만으로 **보고-실제 격차**를 얼마나 복원하나."""
    mask_tr = train["true_gap"].notna()
    mask_te = test["true_gap"].notna()
    if mask_tr.sum() < 50 or mask_te.sum() < 30:
        return float("nan")
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=5.0))
    model.fit(train.loc[mask_tr, list(cols)], train.loc[mask_tr, "true_gap"])
    pred = model.predict(test.loc[mask_te, list(cols)])
    return float(spearmanr(pred, test.loc[mask_te, "true_gap"]).statistic)


def run_one(cfg: dict, seed: int) -> dict[str, float]:
    c = copy.deepcopy(cfg)
    c["seed"] = seed
    data = build_dataset(c)
    df = build(data)
    tr, te = split_by_time(df)

    # L0 계층은 보고를 안 쓰므로 1단 모델이 유일한 선택지다.
    # 전체 계층은 2단(보고 카운트를 타깃 쪽에서 쓰는 구조)으로 잰다.
    base_cols = LAYERS["L0+L0′"]
    full_cols = LAYERS["L0+L0′+L1"]
    l0 = evaluate(te, baseline.fit_predict(tr, te, base_cols))["rank_corr_true"]
    full = evaluate(te, two_stage.fit_predict(tr, te, full_cols))["rank_corr_true"]
    return {
        "l0": l0,
        "full": full,
        "delta": full - l0,
        "gap": gap_recovery(tr, te, full_cols),
        "reject_rate": float(
            np.mean([o["lot_result"] == "reject" for o in data["quality_outcomes"]])
        ),
    }


def main() -> None:
    cfg = config.load("configs/generator.yaml")
    lines: list[str] = []
    all_delta: list[float] = []
    all_gap: list[float] = []

    for key, (label, values) in SWEEPS.items():
        base = (
            cfg["assumptions"].get(key)
            if key != "severity_critical"
            else cfg["assumptions"]["defect_severity_mix"]["critical"]
        )
        print(f"\n── {label} ({key}, 현재값 {base}) " + "─" * 24, flush=True)
        header = f"| {key} | L0 | L0+L0′+L1 | 증분 | gap 복원 | 불합격률 |"
        print(f"  {'값':>6} {'L0':>8} {'전체':>8} {'증분':>8} {'gap':>8} {'불합격':>8}", flush=True)
        lines += [f"\n### {label} — `{key}`\n", header, "|---|---|---|---|---|---|"]

        for v in values:
            c = apply_value(cfg, key, v)
            runs = [run_one(c, s) for s in SEEDS]
            agg = {
                k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs])) for k in runs[0]
            }
            # 평균이 양수인 것과 **매 실행이 양수인 것**은 다른 주장이다.
            # 시드 편차가 ±0.1 대라 최솟값을 함께 봐야 한다.
            d_min = min(r["delta"] for r in runs)
            g_min = min(r["gap"] for r in runs)
            all_delta.append(d_min)
            all_gap.append(g_min)
            mark = " ←현재" if abs(v - base) < 1e-9 else ""
            print(
                f"  {v:>6.2f} {agg['l0'][0]:>+8.3f} {agg['full'][0]:>+8.3f} "
                f"{agg['delta'][0]:>+8.3f} {agg['gap'][0]:>+8.3f} {agg['reject_rate'][0]:>7.1%}"
                f"  (증분 최소 {d_min:+.3f}){mark}",
                flush=True,
            )
            lines.append(
                f"| **{v}**{mark} | {agg['l0'][0]:+.3f} ±{agg['l0'][1]:.3f} "
                f"| {agg['full'][0]:+.3f} ±{agg['full'][1]:.3f} | {agg['delta'][0]:+.3f} "
                f"| {agg['gap'][0]:+.3f} | {agg['reject_rate'][0]:.1%} |"
            )

    n_cfg = len(all_delta)
    print(
        f"\n전체 {n_cfg}개 조건 × {len(SEEDS)} seed"
        f"\n  계층 증분이 **모든 개별 실행**에서 양수인 조건: "
        f"{sum(d > 0 for d in all_delta)}/{n_cfg}   (최악의 실행 {min(all_delta):+.3f})"
        f"\n  gap 복원이 **모든 개별 실행**에서 양수인 조건: "
        f"{sum(g > 0 for g in all_gap)}/{n_cfg}   (최악의 실행 {min(all_gap):+.3f})"
    )
    lines.append(
        f"\n**전 조건 요약** — 계층 증분이 모든 개별 실행에서 양수인 조건 "
        f"{sum(d > 0 for d in all_delta)}/{n_cfg} (최악 {min(all_delta):+.3f}) · "
        f"gap 복원 {sum(g > 0 for g in all_gap)}/{n_cfg} (최악 {min(all_gap):+.3f})"
    )

    out = Path(__file__).resolve().parents[1] / "docs" / "_민감도_표.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n표 저장: {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()
