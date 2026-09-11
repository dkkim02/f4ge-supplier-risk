"""SBC — simulation-based calibration. hier_bayes 의 추정 절차를 진짜 p 없이 채점한다 (2026-09-11, 할일 §2).

Talts et al. 2018. 사전분포에서 파라미터 θ̃ 를 뽑고, 그 θ̃ 로 관측 k̃ 를 생성한 뒤, k̃ 에 MCMC 를 돌려
θ̃ 가 사후표본 L 개 중 몇 번째인지(rank)를 본다. 절차가 맞으면 rank 는 **uniform** 이다.
데이터 설계(표본 n_o · offset ih_o · 공장 색인)는 실제 학습 프레임 그대로 두고 라벨 k 만 다시 만든다 —
「우리 표본 크기·사건 수에서 이 사후분포를 믿어도 되나」가 질문이기 때문이다.

이 검사가 가르는 것: d 순위상관 0.531 이 **추정 절차 탓**(rank 가 uniform 이 아니다)인지
**표본 탓**(절차는 맞는데 사후분포가 넓다)인지. 표본 설계 결정의 전제다.

⚠ 사전분포는 모델 것 그대로다 — mu ~ N(0,2) · tau ~ HalfNormal(2). 이 사전분포는 κ 가 1 을 넘는
(검출률 50% 미만) 세계도 자주 만들고, 그러면 p 가 1 에 붙어 관측이 퇴화한다. 그 비율을 같이 적는다.
SBC 는 사전분포를 바꿔 돌리면 다른 검사가 된다. 그래서 바꾸지 않고 퇴화 비율을 보고한다.

    python scripts/sbc.py [N_SIM]     → docs/_SBC.md
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare, kstest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f4ge_supplier_risk import config
from f4ge_supplier_risk.evaluation.metrics import split_by_time
from f4ge_supplier_risk.features.build import build
from f4ge_supplier_risk.generator.pipeline import build_dataset
from f4ge_supplier_risk.models import hier_bayes as hb

warnings.filterwarnings("ignore")
N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 else 120
WARMUP, SAMPLES, THIN = 500, 2000, 20          # L = 100 사후표본 → rank 0..100. 1차(1000/10)는 mu ESS 121/1000 로 자기상관 artifact → 간격 20
BINS = 20


def main() -> None:
    cfg = config.load(str(ROOT / "configs/generator.yaml"))
    df = build(build_dataset(cfg))
    tr, te = split_by_time(df)
    lab, _, fac_index, pos, _, _ = hb._prepare(tr, te, "pool")
    n = lab["y_inspected"].to_numpy(float)
    off = np.log(np.clip(lab["_ih"].to_numpy(float), hb._FLOOR, None))
    fac = np.array([pos[f] for f in lab["factory_id"]])
    F = len(fac_index)
    rng = np.random.default_rng(20260911)

    ranks = {"mu": [], "tau": [], "log_kappa": []}
    meta = []
    t0 = time.time()
    for i in range(N_SIM):
        mu = rng.normal(0.0, 2.0)
        tau = abs(rng.normal(0.0, hb._TAU_SCALE))
        lk = mu + tau * rng.normal(size=F)
        p = np.clip(np.exp(off + lk[fac]), hb._FLOOR, 1 - 1e-6)
        k = rng.binomial(n.astype(int), p).astype(float)
        sim = lab.copy(); sim["y_reject"] = k

        mcmc = hb._run_mcmc(sim, pos, None, False, seed=i, warmup=WARMUP, samples=SAMPLES)
        s = mcmc.get_samples()
        d_mu = np.asarray(s["mu"])[::THIN]; d_tau = np.asarray(s["tau"])[::THIN]; d_lk = np.asarray(s["log_kappa"])[::THIN]
        ranks["mu"].append(int((d_mu < mu).sum()))
        ranks["tau"].append(int((d_tau < tau).sum()))
        ranks["log_kappa"].extend(int((d_lk[:, f] < lk[f]).sum()) for f in range(F))
        meta.append({"sim": i, "mu": mu, "tau": tau, "events": int(k.sum()), "degenerate": bool((p > 0.5).mean() > 0.5),
                     "div": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())})
        if i % 10 == 9:
            print(f"  {i + 1}/{N_SIM}  {time.time() - t0:.0f}s  사건 {int(k.sum())}  div {meta[-1]['div']}", flush=True)

    L = SAMPLES // THIN
    M = pd.DataFrame(meta)

    def uniform_test(r):
        r = np.asarray(r)
        hist, _ = np.histogram(r, bins=BINS, range=(0, L + 1))
        chi_p = chisquare(hist).pvalue
        ks_p = kstest((r + 0.5) / (L + 1), "uniform").pvalue
        return hist, chi_p, ks_p

    L_ = [f"# SBC — hier_bayes pool 의 추정 절차 채점 ({N_SIM} 시뮬레이션 · 사후표본 L={L} · rank {BINS} 구간)", "",
          "사전분포에서 θ̃ 를 뽑아 관측을 만들고, 그 관측에 MCMC 를 돌려 θ̃ 의 사후 rank 를 본다. 절차가 맞으면 uniform.",
          f"데이터 설계는 seed {cfg['seed']} 학습 프레임 그대로(라벨 오더 {len(lab)}건 · 공장 {F}곳 · 표본 합 {int(n.sum()):,}). 라벨 k 만 다시 만든다.", "",
          "| 파라미터 | rank 수 | χ² p | KS p | 판정 |", "|---|---|---|---|---|"]
    verdicts = {}
    for name, r in ranks.items():
        hist, chi_p, ks_p = uniform_test(r)
        ok = chi_p > 0.05 and ks_p > 0.05
        verdicts[name] = ok
        L_.append(f"| {name} | {len(r)} | {chi_p:.3f} | {ks_p:.3f} | {'uniform — 절차 이상 없음' if ok else '**uniform 아님**'} |")
    L_ += ["", "## rank 히스토그램 (구간별 개수 · 기대값 = 총수 ÷ 20)", ""]
    for name, r in ranks.items():
        hist, _, _ = uniform_test(r)
        L_.append(f"- `{name}` (기대 {len(r) / BINS:.1f}): " + " ".join(f"{h}" for h in hist))
    deg = M["degenerate"].mean()
    L_ += ["", "## 시뮬레이션 세계의 모양", "",
           f"- 사건 수 중앙 **{int(M['events'].median())}** (실데이터 23~44) · 사분위 {int(M['events'].quantile(.25))}~{int(M['events'].quantile(.75))}",
           f"- 퇴화(오더 절반 이상에서 p > 0.5, 검출률 50% 미만 세계) **{deg:.0%}** — 사전분포 tau ~ HalfNormal(2) 가 만든 것. SBC 는 사전분포를 그대로 써야 하므로 포함했다",
           f"- divergence 합 {int(M['div'].sum())} (non-centered)", ""]
    if not M["degenerate"].all():
        keep = ~M["degenerate"].to_numpy()
        r_lk = np.asarray(ranks["log_kappa"]).reshape(N_SIM, F)[keep].ravel()
        _, chi_p, ks_p = uniform_test(r_lk)
        L_ += [f"- 퇴화 세계를 빼면(**{keep.sum()}** 시뮬레이션) log_kappa rank: χ² p {chi_p:.3f} · KS p {ks_p:.3f} — 참고용. 조건부 SBC 는 정식 검사가 아니다", ""]
    L_ += ["## 읽는 법", "",
           "- **uniform 이면** — d 순위상관 0.531 은 절차가 아니라 표본의 문제다. 사후분포가 넓은 것이 맞고, 좁히려면 관측을 늘려야 한다(`_표본설계.md`)",
           "- **uniform 이 아니면** — 어디로 기울었나. ∪ 자형은 사후분포가 너무 좁다(과신), ∩ 자형은 너무 넓다, 한쪽 기울기는 편향",
           "", "재현: `python scripts/sbc.py 120`"]
    out = ROOT / "docs" / "_SBC.md"
    out.write_text("\n".join(L_) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L_))
    print(f"표 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
