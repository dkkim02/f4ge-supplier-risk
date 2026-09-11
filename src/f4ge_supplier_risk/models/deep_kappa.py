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

  embed : g = 공장 one-hot. 2단의 κ 와 같은 것을 gradient 로 배운다. 정면 비교용.
  feat  : g = **공장 수준 집계 피처**의 함수. 공장 identity 를 안 쓴다.
          학습에서 못 본 공장도 pooled 가 아닌 값을 받는다.
  hier  : g = f(공장 특성) + **공장별 잔차**.  ← 기본값
          알려진 공장은 잔차로 보정하고, 신규 공장은 f(특성)으로 떨어진다.
          conjugate 의 partial pooling 이 「전역 상수 쪽으로」인 것을 「특성 예측 쪽으로」 바꾼 것이다.
          f 가 쓸모없으면 자동으로 pooled 와 같아진다 — 잃을 게 없는 구조다.

**정규화는 weight decay 가 아니라 Gamma prior 다** (2026-09-10 2차).
1차 구현은 균일 weight decay 를 썼는데, 그것이 conjugate 에 진 이유였다 —
conjugate 사후분포 `Gamma(a₀+Σk, b₀+Σ노출)` 는 **노출이 적은 공장만** pooled 로 당기는데
weight decay 는 관측이 많은 공장도 같은 세기로 당긴다.
그래서 잔차에 Gamma 로그사전분포를 손실에 그대로 넣는다:

    penalty = a₀ · Σ_f ( exp(r_f) − 1 − r_f )      a₀ = 사건 a₀건어치 사전 정보

이 항은 r=0 에서 0 이고 볼록하며, **손실을 사건 수로 합산**하므로 관측이 많은 공장에서는
자동으로 상대적 무게가 줄어든다 — conjugate 와 같은 성질이다.
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
    mode: str = "hier",
    hidden: int = 8,
    epochs: int = 800,
    wd: float = 1e-3,
    a0: float | None = None,
) -> np.ndarray:
    """테스트 오더별 유닛당 예상 불량 확률.

    `cols` 는 **비워 두는 것이 기본**이다 — 오더 수준 피처를 넣으면 나빠진다(위 docstring).
    """
    return _fit(train, test, cols, seed, mode, hidden, epochs, wd, a0)[0]


def explain(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = 0,
    mode: str = "hier",
    hidden: int = 8,
    epochs: int = 800,
    wd: float = 1e-3,
    a0: float | None = None,
) -> dict[str, object]:
    """공장별 kappa 와 검출률 d = 1/(1+κ) (§3 ①). `two_stage.explain` 의 `kappa` · `detection` 과 같은 모양.

    오더 수준 피처(`cols`)가 없을 때 g(·) 는 공장의 함수라 공장당 값 하나가 정해진다.
    학습에서 못 본 공장(채점 프레임에만 있는 공장)은 예측과 같은 규칙으로 — `feat`·`hier` 는
    f(특성), `embed` 는 pooled — 떨어진다.
    """
    _, kappa = _fit(train, test, (), seed, mode, hidden, epochs, wd, a0)
    return {
        "kappa": kappa,
        "detection": {f: float(two_stage.detection_rate(k)) for f, k in kappa.items()},
    }


def _fit(train, test, cols, seed, mode, hidden, epochs, wd, a0):
    """학습 + 채점. `(오더별 예측, 공장별 kappa)`. 공장별 kappa 는 `cols` 가 비어 있을 때만 정의된다."""
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

    if a0 is None:
        a0 = two_stage._KAPPA_PRIOR_STRENGTH   # conjugate 와 같은 사전 강도(사건 3건어치)

    fmode = "embed" if mode == "embed" else "feat"
    Xtr, ih_tr = _design(lab, cols, fmode, fac_index, fac_tab, norm)
    Xte, ih_te = _design(te, cols, fmode, fac_index, fac_tab, norm)

    n = torch.tensor(lab["y_inspected"].to_numpy(float), dtype=torch.float32)
    k = torch.tensor(lab["y_reject"].to_numpy(float), dtype=torch.float32)
    off = torch.tensor(np.log(np.clip(ih_tr, _FLOOR, None)), dtype=torch.float32)
    X = torch.tensor(Xtr, dtype=torch.float32)

    pooled = float(k.sum() / max(float((n * torch.exp(off)).sum()), _FLOOR))
    logk0 = float(np.log(max(pooled, _FLOOR)))

    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, 1))
    with torch.no_grad():
        net[-1].weight.mul_(0.01)
        net[-1].bias.fill_(logk0)
    # ⚠ 출력 bias 를 weight decay 에서 뺀다. bias 가 log κ 의 수준(≈ −3.5)을 담는데
    #    wd 는 그걸 0(κ = 1)으로 당긴다 — 324개월 실측에서 0.883 → 0.797 로 깎였다.
    #    conjugate 는 pooled 쪽으로 당기지 1 쪽으로 당기지 않는다. 같은 성질을 맞춘다.
    decay = [q for n_, q in net.named_parameters() if not n_.endswith("bias")]
    nodecay = [q for n_, q in net.named_parameters() if n_.endswith("bias")]

    # hier — 공장별 잔차. Gamma 로그사전분포로 0 쪽으로 당긴다(conjugate 와 같은 성질).
    pos = {f: i for i, f in enumerate(fac_index)}
    if mode == "hier":
        res = torch.zeros(len(fac_index), requires_grad=True)
        idx_tr = torch.tensor([pos[f] for f in lab["factory_id"]], dtype=torch.long)
        seen_te = np.array([f in pos for f in te["factory_id"]])
        idx_te = torch.tensor([pos.get(f, 0) for f in te["factory_id"]], dtype=torch.long)
        nodecay = nodecay + [res]      # 잔차는 Gamma prior 가 잡는다. wd 를 겹쳐 걸지 않는다
    opt = torch.optim.Adam(
        [{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}], lr=0.02
    )
    scale = float(n.sum())

    for _ in range(epochs):
        opt.zero_grad()
        logit = off + net(X).squeeze(-1)
        pen = torch.tensor(0.0)
        if mode == "hier":
            logit = logit + res[idx_tr]
            # Gamma 로그사전분포: a0 건어치 사전 정보가 잔차를 0 으로 당긴다
            pen = a0 * (torch.exp(res) - 1.0 - res).sum()
        p = torch.clamp(torch.exp(logit), _FLOOR, 1 - 1e-6)
        nll = -(k * torch.log(p) + (n - k) * torch.log(1 - p)).sum()
        ((nll + pen) / scale).backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        lg = np.log(np.clip(ih_te, _FLOOR, None)) + net(torch.tensor(Xte, dtype=torch.float32)).squeeze(-1).numpy()
        if mode == "hier":
            r = res.detach().numpy()[idx_te.numpy()]
            lg = lg + np.where(seen_te, r, 0.0)   # 처음 보는 공장은 f(특성) 그대로
        out = np.exp(lg)

        # 공장별 kappa — 공장 하나를 행 하나로 둔 설계행렬에 같은 g(·) 를 적용한다.
        # 학습 + 채점 프레임에 나오는 공장 전부. 잔차는 학습에서 본 공장에만 있다.
        kappa: dict[str, float] = {}
        if not cols:
            all_fac = sorted(set(tr["factory_id"]) | set(te["factory_id"]))
            fac_frame = pd.DataFrame({"factory_id": all_fac, "_ih": 1.0})
            Xf, _ = _design(fac_frame, (), fmode, fac_index, fac_tab, norm)
            g = net(torch.tensor(Xf, dtype=torch.float32)).squeeze(-1).numpy()
            if mode == "hier":
                rf = res.detach().numpy()
                g = g + np.array([rf[pos[f]] if f in pos else 0.0 for f in all_fac])
            kappa = {f: float(np.exp(v)) for f, v in zip(all_fac, g, strict=True)}
    return np.clip(out, _FLOOR, 1 - 1e-6), kappa


def _cols_for_stage_a():
    from f4ge_supplier_risk.features.build import FULL_COLS
    return FULL_COLS
