"""공장 파라미터 분해 — **제조 품질 p · 검수 품질 d · 보고 정직도 b** (2026-09-08 저녁, 사용자 목표).

목표 둘: 공장 안에서 품질 좋게 제조되고 있는가(내부 불량률 p), 자체 검수가 그 불량을 얼마나 걸러 올리는가(검출률 d).
2단 모델의 κ = (1 − d) / b 는 둘을 합친 비(比)라 어느 쪽이 문제인지 말하지 못한다. 네 소스가 전부 오면 가를 수 있다.

관측 셋 (공장 f, 오더 합산 N = Cell 카운터 = 실제 생산):

    MES 보고 잡은 불량   R = p · d · b · N        ← 편향 b 가 여기에만 걸린다
    ERP·Cell 물리 폐기   S = p · d · s · N        ← 자재 소진 − 카운터. 회계·설비 값이라 편향이 없다
    우리 입고검사        K ~ Binom(n, p · (1 − d))  ← 라벨

s = 잡은 불량 중 폐기 비중.

**b 와 s 는 관측만으로 분리되지 않는다.** R/S = b/s 하나만 관측되고 미지수는 둘이다. 따라서 「s 를 실측한다」와
「b 를 가정한다」는 같은 문제의 두 얼굴이고, 하나를 정하면 다른 하나가 따라 나온다.

2026-09-10 결정(CTO 질문 29 — 실측 불가): **보고가 정직하다고 가정한다.** 눈금 규약으로만 쓴다 —
`scrap_share=None` 이면 중앙 공장의 b 가 1 이 되도록 s 를 데이터에서 뽑는다.

    s* = 1 / median_f(R_f / S_f)

전역 상수 0.115 를 쓰지 않는 이유는 그것이 생성기 `_SCRAP_SHARE` 의 평균이어서다. 실배포에는 그 값이 없다.
대가는 눈금뿐이다 — 8 seed 실측에서 p 눈금 0.961 → 0.873 (진실 대비 중앙비). **순위는 s 에 완전히 불변**이고
(b ∝ R/S 라 s 는 공장 전부에 같은 배수), 권고에도 영향이 없다(임계가 전부 분위수다). 실측:
b 순위 ρ 0.563(8/8) · p 0.87 · d 0.49 · 오더 p 0.90 이 s 0.115 ↔ 0.127 에서 동일하다.

⚠ 그래서 여기 나오는 `b` 는 **상대 정직도**다. 중앙 공장이 1.0 이라는 규약 위의 값이고 절대 수준이 아니다.
데이터는 b 가 공장마다 다르다는 것까지는 말한다 — S/R 의 공장 간 변동계수 0.253, 진짜 b 와 ρ 0.563(8/8).
s 가 정말 공장 무관 상수라면 S/R 도 상수여야 하는데 아니다.

    caught = S / s̄                     잡은 불량(진실 추정)
    b = R / caught                      보고 정직도. 1 = 정직, 0.5 = 절반만 보고
    e = (K + α) / (n + β)               유출 불량률 (Beta 사후 평균)
    p = caught / N + e                  내부 불량률 = 잡은 것 + 놓친 것
    d = caught / (caught + e · N)       검출률

시간 규율은 κ 와 같다 — 라벨이 도착한 오더만 합산한다. 라벨 없는 오더는 R·S 는 있지만 e 를 갱신하지 못한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCRAP_SHARE = 0.115  # 민감도 비교용 옛 상수. 기본 경로는 s* 를 데이터에서 뽑는다(모듈 docstring)
_ALPHA, _BETA = 0.5, 400.0  # 유출률 Beta 사전분포 (build.py 의 이력 사전분포와 같은 값)
_MIN_UNITS = 300.0  # 이보다 적게 관측된 공장은 전체 평균으로 당긴다
_B_RANGE = (0.10, 1.50)
_D_RANGE = (0.50, 0.999)


def derive_scrap_share(R: pd.Series, S: pd.Series) -> float:
    """「중앙 공장의 보고는 정직하다」(b_median = 1) 와 동치인 s. 모듈 docstring 참조."""
    ratio = R / S.clip(lower=1e-9)
    med = float(np.median(ratio[np.isfinite(ratio) & (ratio > 0)])) if len(ratio) else 0.0
    return 1.0 / med if med > 0 else SCRAP_SHARE


def estimate(train: pd.DataFrame, scrap_share: float | None = None) -> pd.DataFrame:
    """공장별 (p, d, b) 와 관측량. index = factory_id. `pooled` 행에 전체 합산값.

    `scrap_share=None` 이면 데이터에서 뽑는다(보고 정직 가정). 숫자를 주면 그 값을 쓴다 — 민감도 비교용.
    """
    lab = train[train["y_inspected"].notna()]
    g = lab.groupby("factory_id")
    N = g["obs_produced_counter"].sum()
    S = g["obs_phys_scrap"].sum()
    R = g["l1_rep_defects"].sum()
    K = g["y_reject"].sum()
    n = g["y_inspected"].sum()
    if scrap_share is None:
        scrap_share = derive_scrap_share(R, S)

    pooled_caught = S.sum() / scrap_share
    pooled = {
        "b": float(np.clip(R.sum() / max(pooled_caught, 1e-9), *_B_RANGE)),
        "e": float((K.sum() + _ALPHA) / (n.sum() + _BETA)),
    }
    pooled["p"] = pooled_caught / max(N.sum(), 1.0) + pooled["e"]
    pooled["d"] = float(np.clip(pooled_caught / max(pooled_caught + pooled["e"] * N.sum(), 1e-9), *_D_RANGE))

    caught = S / scrap_share
    e = (K + _ALPHA) / (n + _BETA)
    # partial pooling — 관측이 적은 공장은 전체값 쪽으로
    w = N / (N + _MIN_UNITS)
    b = np.clip(w * (R / caught.clip(lower=1e-9)) + (1 - w) * pooled["b"], *_B_RANGE)
    p = w * (caught / N.clip(lower=1.0) + e) + (1 - w) * pooled["p"]
    d = np.clip(w * (caught / (caught + e * N).clip(lower=1e-9)) + (1 - w) * pooled["d"], *_D_RANGE)

    out = pd.DataFrame({"p": p, "d": d, "b": b, "e": e, "n_labels": g.size(), "units": N, "phys_scrap": S, "reported_defects": R})
    out.attrs["pooled"] = pooled
    out.attrs["scrap_share"] = scrap_share
    return out


def order_internal_rate(frame: pd.DataFrame, params: pd.DataFrame, pseudo: float = 150.0) -> np.ndarray:
    """오더별 제조 품질(내부 불량률) 추정 — 보고를 정직도로 되돌린 값을 공장 수준 쪽으로 shrink.

        p_order = (R_order / b_f + pseudo · p_f) / (N_order + pseudo)  +  e_f

    R 은 MES 가 보고한 잡은 불량, N 은 Cell 카운터(없으면 보고 생산 수). 보고가 없으면 공장 수준 p_f.
    """
    pooled = params.attrs.get("pooled", {"p": float(params["p"].mean()), "b": 1.0, "e": 0.0})
    b = frame["factory_id"].map(params["b"]).fillna(pooled["b"]).to_numpy(float)
    pf = frame["factory_id"].map(params["p"]).fillna(pooled["p"]).to_numpy(float)
    ef = frame["factory_id"].map(params["e"]).fillna(pooled["e"]).to_numpy(float)
    R = frame["l1_rep_defects"].to_numpy(float)
    N = frame["obs_produced_counter"].to_numpy(float)
    N = np.where(np.isnan(N), frame["l1_rep_produced"].to_numpy(float), N)
    seen = N > 0
    caught_rate = np.where(seen, (R / b + pseudo * (pf - ef)) / (np.maximum(N, 0) + pseudo), pf - ef)
    return np.clip(caught_rate + ef, 1e-6, 0.5)


def order_escape_rate(frame: pd.DataFrame, params: pd.DataFrame, internal: np.ndarray) -> np.ndarray:
    """분해 일관성 — escape = p · (1 − d). 2단 모델과 별개로 낸 값이라 둘을 비교할 수 있다."""
    pooled = params.attrs.get("pooled", {"d": 0.95})
    d = frame["factory_id"].map(params["d"]).fillna(pooled["d"]).to_numpy(float)
    return np.clip(internal * (1.0 - d), 1e-7, 0.5)
