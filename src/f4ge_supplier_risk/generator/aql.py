"""ISO 2859-1 표본 검사 — 검사수준 II, 1회 샘플링, 보통검사.

우리 오더 수량은 200~5,000개라 코드 G~L 다섯 구간만 있으면 된다.
전체 표를 옮기지 않는 이유는 쓰지 않는 행을 검증할 방법이 없기 때문이다.

`합격판정개수(Ac)`는 AQL 수준별로 다르다. 표본 200개 · major 2.5%면 Ac=10 —
2.5%의 산술값(5개)과 다르다는 점에 주의. 표준에 내장된 통계적 위험 가정 때문이다.
"""

from __future__ import annotations

# (로트 상한, 코드, 표본 크기)
_LOT_TABLE: tuple[tuple[int, str, int], ...] = (
    (280, "G", 32),
    (500, "H", 50),
    (1200, "J", 80),
    (3200, "K", 125),
    (10000, "L", 200),
)

# 표본 크기 -> 합격판정개수. AQL 2.5% / 4.0%
# critical(AQL 0) 은 표에 없다 — 하나만 나와도 불합격이라 Ac = 0 이다.
_ACCEPT: dict[int, dict[float, int]] = {
    32: {0.025: 2, 0.040: 3},
    50: {0.025: 3, 0.040: 5},
    80: {0.025: 5, 0.040: 7},
    125: {0.025: 7, 0.040: 10},
    200: {0.025: 10, 0.040: 14},
}


def sample_plan(lot_size: int, aql_major: float, aql_minor: float) -> dict[str, object]:
    """``(코드, 표본 크기, 등급별 합격판정개수)``.

    **로트 판정은 세 등급을 각각 본다.** 하나라도 넘으면 불합격이다.
    실무에서 입고검사 불합격의 대부분은 critical 쪽에서 나온다 — major 2.5%
    기준은 표본 80개에서 5개까지 허용해서, 600 PPM 수준의 공급사는
    사실상 절대 걸리지 않는다.
    """
    for upper, code, n in _LOT_TABLE:
        if lot_size <= upper:
            break
    else:  # 10,000 초과는 생성 범위 밖 — 마지막 구간으로 붙인다
        upper, code, n = _LOT_TABLE[-1]
    return {
        "code": code,
        "n": n,
        "accept": {
            "critical": 0,
            "major": _ACCEPT[n][round(aql_major, 3)],
            "minor": _ACCEPT[n][round(aql_minor, 3)],
        },
    }
