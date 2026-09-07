"""ISO 2859-1 검사수준 II 표본 계획."""

import pytest

from f4ge_supplier_risk.generator.aql import sample_plan


@pytest.mark.parametrize(
    "lot,code,n,ac_major,ac_minor",
    [
        (200, "G", 32, 2, 3),
        (500, "H", 50, 3, 5),
        (1200, "J", 80, 5, 7),
        (3200, "K", 125, 7, 10),
        (5000, "L", 200, 10, 14),
    ],
)
def test_sample_plan_matches_iso_table(lot, code, n, ac_major, ac_minor):
    plan = sample_plan(lot, 0.025, 0.040)
    assert (plan["code"], plan["n"]) == (code, n)
    assert plan["accept"]["major"] == ac_major
    assert plan["accept"]["minor"] == ac_minor


def test_critical_is_zero_tolerance():
    """critical 은 표에 없다 — 하나만 나와도 불합격이다."""
    assert sample_plan(5000, 0.025, 0.040)["accept"]["critical"] == 0
