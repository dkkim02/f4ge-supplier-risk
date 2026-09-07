"""불량률이 두 종류라는 것 — 이 프로젝트에서 제일 틀리기 쉬운 지점."""

import numpy as np


def test_escape_is_internal_times_miss_rate(world):
    for _o, factory, _p, lat, _t in world:
        expected = lat["internal_defect_rate_true"] * (1.0 - factory["detection_rate"])
        assert lat["escape_rate_true"] == expected


def test_escape_is_far_below_internal(world):
    """escape 는 내부 불량률보다 한 자릿수 이상 작다.

    기존 f4ge-quality-prediction 의 defect_rate 0.05~0.25 는 **내부 불량률**이다.
    그걸 y 로 가져오면 100배 틀린다.
    """
    ratio = np.median(
        [
            lat["escape_rate_true"] / lat["internal_defect_rate_true"]
            for _o, _f, _p, lat, _t in world
        ]
    )
    assert ratio < 0.10


def test_internal_rate_stays_in_industry_range(world):
    """3~4 시그마 실태 = 0.6~6.7%. 꼬리를 감안해 넉넉히 잡는다."""
    rates = np.array([lat["internal_defect_rate_true"] for _o, _f, _p, lat, _t in world])
    assert 0.001 < np.median(rates) < 0.15
