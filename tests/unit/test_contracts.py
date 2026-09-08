"""계약 스키마가 생성기와 어긋나지 않는가 — 생성 레코드 전부를 각 계약으로 검증한다."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from f4ge_supplier_risk import contracts

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((C / f"{name}.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _check(name: str, rows: list[dict]) -> None:
    v = _validator(name)
    bad = [(r.get("order_id"), e.message) for r in rows for e in v.iter_errors(r)]
    assert not bad, f"{name}: {len(bad)}건 위반 — 처음 3건 {bad[:3]}"


@pytest.fixture(scope="module")
def fac_of(dataset):
    return {o["order_id"]: o["factory_id"] for o in dataset["orders"]}


def test_order_contract(dataset):
    _check("supplier-order.v1", [contracts.order_row(o) for o in dataset["orders"]])


def test_factory_report_contract(dataset, fac_of):
    rows = [contracts.report_row(r, fac_of[r["order_id"]]) for r in dataset["factory_reports"]]
    assert rows
    _check("factory-report.v1", rows)
    # 결측 행은 수량을 갖지 않는다 — 계약의 if/then 이 실제로 막는지 역방향으로 확인
    v = _validator("factory-report.v1")
    missing = next(r for r in rows if r["is_missing"])
    assert list(v.iter_errors({**missing, "produced_quantity": 1}))


def test_erp_daily_contract(dataset, fac_of):
    rows = [contracts.erp_row(e, fac_of[e["order_id"]]) for e in dataset["erp_daily"]]
    assert rows
    _check("erp-daily.v1", rows)
    v = _validator("erp-daily.v1")
    missing = next(r for r in rows if r["is_missing"])
    assert list(v.iter_errors({**missing, "material_issued_quantity": 1.0}))


def test_fai_contract(dataset, fac_of):
    _check("fai-report.v1", [contracts.fai_row(f, fac_of[f["order_id"]]) for f in dataset["fai_reports"]])


def test_cell_daily_contract(dataset, fac_of):
    _check("cell-daily.v1", [contracts.cell_row(c, fac_of[c["order_id"]]) for c in dataset["cell_daily"]])


def test_outcome_contract(dataset, fac_of):
    _check("order-quality-outcome.v1",
           [contracts.outcome_row(q, fac_of[q["order_id"]]) for q in dataset["quality_outcomes"]])


def test_score_contract_matches_generated_output():
    """출력 계약 — sr score 산출물이 있으면 그것으로 검증한다."""
    p = ROOT / "datasets/generated/scores.jsonl"
    if not p.exists():
        pytest.skip("scores.jsonl 없음 — sr score 먼저")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    _check("supplier-risk-score.v1", rows)
