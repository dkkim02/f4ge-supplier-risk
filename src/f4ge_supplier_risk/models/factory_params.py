"""공장 파라미터 분해 — **제조 품질 p · 검수 품질 d · 보고 정직도 b** (2026-09-08 저녁, 사용자 목표).

목표 둘: 공장 안에서 품질 좋게 제조되고 있는가(내부 불량률 p), 자체 검수가 그 불량을 얼마나 걸러 올리는가(검출률 d).
2단 모델의 κ = (1 − d) / b 는 둘을 합친 비(比)라 어느 쪽이 문제인지 말하지 못한다. 네 소스가 전부 오면 가를 수 있다.

관측 셋 (공장 f, 오더 합산 N = Cell 카운터 = 실제 생산):

    MES 보고 잡은 불량   R = p · d · b · N        ← 편향 b 가 여기에만 걸린다
    ERP·Cell 물리 폐기   S = p · d · s · N        ← 자재 소진 − 카운터. 회계·설비 값이라 편향이 없다
    우리 입고검사        K ~ Binom(n, p · (1 − d))  ← 라벨

s = 잡은 불량 중 폐기 비중. **가정값**이다(생성기 `_SCRAP_SHARE` 0.03~0.20 의 평균 ≈ 0.115 — 실측 근거는 Hidden Factory
scrap 0.5% / 재작업 17.5% ≈ 0.03 이라 하한 쪽이다). 이 값 하나가 세 파라미터의 눈금을 정하므로 민감도 항목이다.

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

SCRAP_SHARE = 0.115  # ⚠ 가정 — 잡은 불량 중 폐기 비중. 민감도 분석 대상
_ALPHA, _BETA = 0.5, 400.0  # 유출률 Beta 사전분포 (build.py 의 이력 사전분포와 같은 값)
_MIN_UNITS = 300.0  # 이보다 적게 관측된 공장은 전체 평균으로 당긴다
_B_RANGE = (0.10, 1.50)
_D_RANGE = (0.50, 0.999)


def estimate(train: pd.DataFrame, scrap_share: float = SCRAP_SHARE) -> pd.DataFrame:
    """공장별 (p, d, b) 와 관측량. index = factory_id. `pooled` 행에 전체 합산값."""
    lab = train[train["y_inspected"].notna()]
    g = lab.groupby("factory_id")
    N = g["obs_produced_counter"].sum()
    S = g["obs_phys_scrap"].sum()
    R = g["l1_rep_defects"].sum()
    K = g["y_reject"].sum()
    n = g["y_inspected"].sum()

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
