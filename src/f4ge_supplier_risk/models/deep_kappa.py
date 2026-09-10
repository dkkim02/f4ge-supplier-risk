"""딥러닝 재도전 — 구조를 모델 안에 넣는다 (2026-09-10).

**왜 다시 하나.** 기존 벤치에서 MLP 는 0.409 로 졌는데, 원인이 라벨 수만이 아니었다.
8 seed 재측정(scripts/dl_fair_bench.py):

    mlp, 피처 42개                 +0.409
    mlp, 카운트 비율 **하나만**       +0.710   ← 피처를 빼니 올랐다
    mlp, 42개 + 공장 one-hot        +0.489   ← 도로 떨어진다

즉 42개 피처는 신호가 아니라 노이즈였고(피처 ablation: 기여 +0.008), 정규화가 약한
학습기가 거기 과적합했다. 그리고 `escape = 보고 × κ` 라는 구조를 23개 사건으로
발견하라고 시킨 것이 무리였다.

**그래서 이렇게 짠다.**

    log(escape) = log(internal_hat)  +  g(·)
                  └ 고정 offset ┘      └ 신경망이 내는 log κ ┘

카운트 비율을 offset 으로 박아 구조를 공짜로 주고, 신경망은 **공장별 보정 하나만** 낸다.
손실은 관측 (n, k) 의 binomial NLL — 비율 회귀가 아니다.

**두 가지 mode.**

  embed : g = 공장 one-hot 의 선형층. 2단의 κ 와 같은 것을 gradient 로 배운다.
          conjugate 와 정면 비교용이고, 이기기 어렵다.
  feat  : g = **공장 수준 집계 피처**의 함수. 공장 identity 를 안 쓴다.
          → 학습에서 못 본 공장도 pooled 가 아닌 값을 받는다. **conjugate 가 못 하는 일이다.**
          콜드스타트가 이 모델의 존재 이유다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

from f4ge_supplier_risk.models import two_stage

_FLOOR = 1e-7
# 공장 수준 집계 — κ = (1−d)/d 를 설명할 만한 것만 최소로. 많이 주면 다시 과적합한다.
_FAC_FEATS = ("rep_rate", "phys_share", "log_units", "uptime", "cycle_cv", "anomaly")


def available() -> bool:
    return _TORCH


def factory_table(frame: pd.DataFrame) -> pd.DataFrame:
    """공장 수준 집계 6개. 라벨을 쓰지 않는다 — 신규 공장에서도 계산된다."""
    g = frame.groupby("factory_id")
    N = g["obs_produced_counter"].sum().clip(lower=1.0)
    out = pd.DataFrame({
        "rep_rate": g["l1_rep_defects"].sum() / N,
        "phys_share": g["obs_phys_scrap"].sum() / g["l1_rep_defects"].sum().clip(lower=1.0),
        "log_units": np.log1p(N),
        "uptime": g["l2_uptime_mean"].mean(),
        "cycle_cv": g["l2_cycle_cv"].mean(),
        "anomaly": g["l2_anomaly_rate"].mean(),
    })
    return out[list(_FAC_FEATS)]


def _design(frame, cols, mode, fac_index, fac_tab, norm):
    ih = frame["_ih"].to_numpy(float)
    if mode == "embed":
        X = np.zeros((len(frame), len(fac_index)), dtype=float)
        pos = {f: i for i, f in enumerate(fac_index)}
        for r, f in enumerate(frame["factory_id"]):
            if f in pos:
                X[r, pos[f]] = 1.0
    else:
        t = fac_tab.reindex(frame["factory_id"]).to_numpy(float)
        t = np.where(np.isnan(t), norm[0], t)
        X = (t - norm[0]) / norm[1]
    if cols:
        extra = frame[list(cols)].to_numpy(float)
        med = norm[2]
        extra = np.where(np.isnan(extra), med, extra)
        X = np.hstack([X, (extra - med) / norm[3]])
    return X, ih


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: tuple[str, ...] = (),
    seed: int = 0,
    mode: str = "feat",
    hidden: int = 8,
    epochs: int = 600,
    wd: float = 1e-2,
) -> np.ndarray:
    """테스트 오더별 유닛당 예상 불량 확률.

    `cols` 는 **비워 두는 것이 기본**이다 — 오더 수준 피처를 넣으면 나빠진다(위 docstring).
    """
    if not _TORCH:
        raise RuntimeError("torch 없음")
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = train.copy()
    te = test.copy()
    tr["_ih"] = two_stage._stage_a(train, train, _cols_for_stage_a())
    te["_ih"] = two_stage._stage_a(train, test, _cols_for_stage_a())

    lab = tr[tr["y_inspected"].notna() & (tr["y_inspected"] > 0)]
    fac_index = sorted(set(tr["factory_id"]))
    # ⚠ 공장 집계는 **학습 + 채점 프레임 둘 다**로 만든다. 라벨을 안 쓰는 값이라 채점 시점에
    #   신규 공장 것도 손에 있다(MES·Cell·ERP 는 이미 온다). 학습 프레임만 쓰면 처음 보는
    #   공장이 전부 평균 벡터로 대치돼 feat 모드가 embed 와 같아진다 — 콜드스타트가 죽는다.
    fac_tab = factory_table(pd.concat([tr, te], ignore_index=True))
    mu = fac_tab.to_numpy(float).mean(axis=0)
    sd = fac_tab.to_numpy(float).std(axis=0) + 1e-9
    emed = lab[list(cols)].median().to_numpy(float) if cols else np.zeros(0)
    esd = (lab[list(cols)].std().to_numpy(float) + 1e-9) if cols else np.ones(0)
    norm = (mu, sd, emed, esd)

    Xtr, ih_tr = _design(lab, cols, mode, fac_index, fac_tab, norm)
    Xte, ih_te = _design(te, cols, mode, fac_index, fac_tab, norm)

    n = torch.tensor(lab["y_inspected"].to_numpy(float), dtype=torch.float32)
    k = torch.tensor(lab["y_reject"].to_numpy(float), dtype=torch.float32)
    off = torch.tensor(np.log(np.clip(ih_tr, _FLOOR, None)), dtype=torch.float32)
    X = torch.tensor(Xtr, dtype=torch.float32)

    # κ_pooled 로 초기화 — 학습이 0 에서 시작하지 않게 한다
    pooled = float(k.sum() / max(float((n * torch.exp(off)).sum()), _FLOOR))
    b0 = float(np.log(max(pooled, _FLOOR)))

    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, 1))
    with torch.no_grad():
        net[-1].weight.mul_(0.01)
        net[-1].bias.fill_(b0)
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=wd)

    for _ in range(epochs):
        opt.zero_grad()
        logit = off + net(X).squeeze(-1)
        p = torch.clamp(torch.exp(logit), _FLOOR, 1 - 1e-6)
        loss = -(k * torch.log(p) + (n - k) * torch.log(1 - p)).sum() / n.sum()
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        out = np.exp(np.log(np.clip(ih_te, _FLOOR, None)) + net(torch.tensor(Xte, dtype=torch.float32)).squeeze(-1).numpy())
    return np.clip(out, _FLOOR, 1 - 1e-6)


def _cols_for_stage_a():
    from f4ge_supplier_risk.features.build import FULL_COLS
    return FULL_COLS
