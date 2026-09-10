"""계층 베이즈 — shrinkage 세기를 손으로 박지 않고 추정한다 (2026-09-10).

`two_stage` 와 `deep_kappa` 는 관측 모형이 같다.

    k ~ Binomial(n, ih x kappa_f)          ih = 보고 카운트에서 온 offset

다른 것은 **kappa 를 공장 사이에서 얼마나 당기는가**를 어디서 정하느냐뿐이다.

    two_stage    Gamma(a0, b0) conjugate prior.  a0 = 3.0 을 **손으로 박았다**
    deep_kappa   같은 a0 를 Gamma 로그사전분포 penalty 로 손실에 더한다. 역시 손으로 박았다
    hier_bayes   `log kappa_f = mu + tau x z_f` 로 두고 **tau 를 데이터에서 추정한다**

`tau` 가 공장 간 산포다. 0 이면 complete pooling(공장 구분이 무의미), 크면 no pooling.
이 값이 곧 shrinkage 세기라, 추정하면 a0 를 고르는 작업이 사라진다.
[[모델_방향_결정]] §5 1순위의 「정규화를 손으로 안 맞춘다」가 이 뜻이고,
09-10 에 weight decay 설정 하나로 0.086 을 잃은 것이 손으로 맞추는 비용이었다.

**non-centered parameterization 을 쓴다.** `z_f ~ Normal(0,1)` 을 샘플하고 `tau` 를 곱한다.
`log kappa_f ~ Normal(mu, tau)` 를 직접 샘플하면 tau 와 z 의 사후상관이 funnel geometry 를
만들어 HMC 가 tau 가 작은 구간에 못 들어간다. 관측이 적을수록 심하다 — 사건 23건이면 정확히 그 구간이다.
`centered=True` 로 두 방식을 재볼 수 있게 남겼다(divergence 수로 확인한다).

**mode**

    pool  log kappa_f = mu + tau z_f                     공장 identity 만
    feat  log kappa_f = mu + x_f . beta + tau z_f        + 공장 수준 집계 6개
          deep_kappa 의 hier 모드와 같은 구조인데 f 가 신경망이 아니라 선형이다.
          피처 ablation 이 +0.008 이었으므로 여기서 신경망을 쓸 근거가 없다 —
          NGMM(arXiv 2604.10976) 의 선형 특수경우다.

**신규 공장.** 학습에서 못 본 공장은 `z_f` 가 없다. `feat` 은 `mu + x_f . beta`,
`pool` 은 `mu` 로 떨어진다 — 상위 계층으로 자동 복귀한다.

**얻는 것이 하나 더 있다** — 사후분포. `posterior()` 가 공장별 kappa 의 구간을 낸다.
conjugate 도 Gamma 사후분포를 갖지만 `two_stage` 는 평균만 쓰고 버린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    _NUMPYRO = True
except ImportError:  # pragma: no cover
    _NUMPYRO = False

from f4ge_supplier_risk.models import deep_kappa, two_stage

_FLOOR = 1e-7
# tau 사전분포 HalfNormal(_TAU_SCALE).
# ⚠ **이 값을 조이면 진다.** 8 seed 실측 — 0.25 → +0.782 · 0.5 → +0.808 · 2.0 → +0.824 · 5.0 → +0.826.
#    two_stage 가 +0.823 이므로 2.0 이상에서 동률이고, 0.5 에서는 0.016 을 잃는다.
#    사후 tau 가 사전분포를 따라 0.22 → 0.58 로 계속 끌려간다 — 사건 23건 · 공장 12곳에서
#    variance component 는 데이터가 결정하지 못한다(문헌: 클러스터 수가 적을 때의 알려진 성질).
#    그래서 **느슨한 쪽으로 둔다.** 조이는 것은 검증되지 않은 사전 지식을 넣는 것이고, 그 대가가 실측된다.
_TAU_SCALE = 2.0
_BETA_SCALE = 0.3


def available() -> bool:
    return _NUMPYRO


def _model(n, k, off, fac_idx, n_fac, X, centered):
    """관측 (k, n) 의 binomial. 비율 회귀가 아니다 — 표본 88개 중 0.11개를 비율로 만들면
    88개짜리와 900개짜리가 같은 무게를 갖는다."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 2.0))
    tau = numpyro.sample("tau", dist.HalfNormal(_TAU_SCALE))

    loc = mu
    if X is not None:
        beta = numpyro.sample("beta", dist.Normal(0.0, _BETA_SCALE).expand([X.shape[1]]).to_event(1))
        loc = mu + X @ beta

    if centered:
        log_kappa = numpyro.sample("log_kappa", dist.Normal(loc, tau).expand([n_fac]).to_event(1))
    else:
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_fac]).to_event(1))
        log_kappa = numpyro.deterministic("log_kappa", loc + tau * z)

    logit = off + log_kappa[fac_idx]
    p = jnp.clip(jnp.exp(logit), _FLOOR, 1 - 1e-6)
    numpyro.sample("obs", dist.Binomial(total_count=n, probs=p), obs=k)


def _prepare(train, test, mode):
    """offset(ih) · 라벨 있는 학습행 · 공장 색인 · 공장 집계 설계행렬."""
    cols = deep_kappa._cols_for_stage_a()
    ih_tr = two_stage._stage_a(train, train, cols)
    ih_te = two_stage._stage_a(train, test, cols)

    tr = train.copy()
    tr["_ih"] = ih_tr
    lab = tr[tr["y_inspected"].notna() & (tr["y_inspected"] > 0)]

    fac_index = sorted(set(tr["factory_id"]))
    pos = {f: i for i, f in enumerate(fac_index)}

    X = None
    if mode == "feat":
        # ⚠ 공장 집계는 학습 + 채점 프레임 둘 다로 만든다. 라벨을 안 쓰는 값이라 채점 시점에
        #   신규 공장 것도 손에 있다(deep_kappa 와 같은 규율). 학습 프레임만 쓰면 콜드스타트가 죽는다.
        tab = deep_kappa.factory_table(pd.concat([train, test], ignore_index=True))
        mu_, sd_ = tab.to_numpy(float).mean(0), tab.to_numpy(float).std(0) + 1e-9
        X = ((tab.reindex(fac_index).to_numpy(float) - mu_) / sd_)
        X = np.nan_to_num(X)
        X_te_extra = ((tab.to_numpy(float) - mu_) / sd_)
        X_lookup = dict(zip(tab.index, np.nan_to_num(X_te_extra)))
    else:
        X_lookup = {}

    return lab, ih_te, fac_index, pos, X, X_lookup


def _run_mcmc(lab, pos, X, centered, seed, warmup, samples):
    n = jnp.asarray(lab["y_inspected"].to_numpy(float))
    k = jnp.asarray(lab["y_reject"].to_numpy(float))
    off = jnp.asarray(np.log(np.clip(lab["_ih"].to_numpy(float), _FLOOR, None)))
    fac_idx = jnp.asarray([pos[f] for f in lab["factory_id"]], dtype=jnp.int32)
    Xj = None if X is None else jnp.asarray(X)

    kernel = NUTS(_model, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(seed), n, k, off, fac_idx, len(pos), Xj, centered,
        extra_fields=("diverging",),
    )
    return mcmc


def posterior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    mode: str = "pool",
    centered: bool = False,
    seed: int = 0,
    warmup: int = 1000,
    samples: int = 1000,
) -> dict[str, object]:
    """사후분포 요약 — 공장별 kappa 의 구간, tau, divergence 수.

    `tau` 가 이 모델의 요점이다. conjugate 의 `a0 = 3.0` 에 해당하는 값을 **추정한 것**이라,
    손으로 박은 값이 데이터와 맞았는지 여기서 처음 확인된다.
    """
    if not _NUMPYRO:
        raise RuntimeError("numpyro 없음")
    lab, _, fac_index, pos, X, _ = _prepare(train, test, mode)
    mcmc = _run_mcmc(lab, pos, X, centered, seed, warmup, samples)
    s = mcmc.get_samples()
    lk = np.asarray(s["log_kappa"])
    kap = np.exp(lk)
    return {
        "tau": float(np.asarray(s["tau"]).mean()),
        "tau_q": tuple(float(q) for q in np.quantile(np.asarray(s["tau"]), [0.05, 0.95])),
        "mu": float(np.asarray(s["mu"]).mean()),
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
        "kappa_mean": {f: float(kap[:, i].mean()) for i, f in enumerate(fac_index)},
        "kappa_q05": {f: float(np.quantile(kap[:, i], 0.05)) for i, f in enumerate(fac_index)},
        "kappa_q95": {f: float(np.quantile(kap[:, i], 0.95)) for i, f in enumerate(fac_index)},
        "n_events": int(lab["y_reject"].sum()),
        "n_labeled_orders": len(lab),
    }


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: tuple[str, ...] = (),
    seed: int = 0,
    mode: str = "pool",
    centered: bool = False,
    warmup: int = 1000,
    samples: int = 1000,
) -> np.ndarray:
    """테스트 오더별 유닛당 예상 불량 확률 — 사후평균.

    `cols` 는 받지 않는다(서명만 다른 모델과 맞춘다). 오더 수준 피처를 넣으면 나빠진다 —
    피처 ablation 기여 +0.008, mlp 42개 0.409.

    ⚠ 온라인 kappa 갱신을 하지 않는다. 학습 구간 사후분포를 고정해 채점한다 —
    `deep_kappa` 및 `dl_fair_bench.ts_no_online` 과 같은 조건이다.
    """
    if not _NUMPYRO:
        raise RuntimeError("numpyro 없음")
    lab, ih_te, fac_index, pos, X, X_lookup = _prepare(train, test, mode)
    mcmc = _run_mcmc(lab, pos, X, centered, seed, warmup, samples)
    s = mcmc.get_samples()

    kappa_seen = np.exp(np.asarray(s["log_kappa"])).mean(axis=0)     # (공장,)
    mu = float(np.asarray(s["mu"]).mean())
    beta = np.asarray(s["beta"]).mean(axis=0) if "beta" in s else None

    def kappa_of(f):
        if f in pos:
            return kappa_seen[pos[f]]
        # 신규 공장 — 상위 계층으로 복귀. feat 이면 특성 예측까지 쓴다.
        if beta is not None and f in X_lookup:
            return float(np.exp(mu + X_lookup[f] @ beta))
        return float(np.exp(mu))

    kap = np.array([kappa_of(f) for f in test["factory_id"]])
    return np.clip(ih_te * kap, _FLOOR, 1 - 1e-6)
