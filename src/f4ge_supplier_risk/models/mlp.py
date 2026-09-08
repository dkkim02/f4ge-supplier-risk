"""딥러닝 후보 — torch MLP, binomial NLL.

출력은 log(유출률) 하나. 손실은 관측 (n, k) 의 binomial 음의 로그우도 그대로 —
비율을 회귀하는 것이 아니라 표본 200개 중 0건과 50개 중 0건을 다르게 취급한다(2단·HGB 와 같은 관측 모형).

라벨이 300건 남짓이라 모델을 작게 두고(2층 × 32) 학습 구간 뒤 20% 를 검증으로 잘라 early stopping 한다.
결측은 중앙값 대치 + 결측 지시자. 24시간 배치라 학습 비용은 문제가 아니다 — 문제는 라벨 수다.
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

_FLOOR = 1e-7


def available() -> bool:
    return _TORCH


def _prep(train: pd.DataFrame, frame: pd.DataFrame, cols: list[str]):
    Xtr = train[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def f(X):
        miss = np.isnan(X).astype(float)
        X = np.where(np.isnan(X), med, X)
        return np.hstack([X, miss])

    A = f(Xtr)
    mu, sd = A.mean(axis=0), A.std(axis=0) + 1e-9
    return (lambda X: (f(X) - mu) / sd), A.shape[1]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: tuple[str, ...], seed: int = 0,
                hidden: int = 32, epochs: int = 400, lr: float = 3e-3, weight_decay: float = 1e-3) -> np.ndarray:
    if not _TORCH:
        raise RuntimeError("torch 가 없다")
    torch.manual_seed(seed)
    cols = list(cols)
    lab = train[train["y_inspected"].notna() & (train["y_inspected"] > 0)].sort_values("ordered_at")
    xf, dim = _prep(lab, test, cols)
    X = torch.tensor(xf(lab[cols].to_numpy(float)), dtype=torch.float32)
    n = torch.tensor(lab["y_inspected"].to_numpy(float), dtype=torch.float32)
    k = torch.tensor(lab["y_reject"].to_numpy(float), dtype=torch.float32)
    cut = int(len(lab) * 0.8)
    idx_tr, idx_va = torch.arange(cut), torch.arange(cut, len(lab))

    base = float(np.log(max((k.sum() / n.sum()).item(), 1e-6)))
    net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
    with torch.no_grad():
        net[-1].bias.fill_(base)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    def nll(idx):
        logit = net(X[idx]).squeeze(-1)
        p = torch.sigmoid(logit).clamp(1e-7, 1 - 1e-6)
        return -(k[idx] * torch.log(p) + (n[idx] - k[idx]) * torch.log(1 - p)).sum() / n[idx].sum()

    best, best_state, patience = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = nll(idx_tr)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = nll(idx_va).item() if len(idx_va) else loss.item()
        if v < best - 1e-6:
            best, best_state, patience = v, {a: b.clone() for a, b in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 40:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = torch.sigmoid(net(torch.tensor(xf(test[cols].to_numpy(float)), dtype=torch.float32)).squeeze(-1)).numpy()
    return np.clip(out, _FLOOR, 1 - 1e-6)
