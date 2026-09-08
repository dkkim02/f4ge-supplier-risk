"""`sr serve` — FastAPI 한 프로세스.

  POST /api/ingest/{contract}       계약 행 배치. 스키마 검증 후 저장 (ingest·admin)
  POST /api/masters/products|factories   마스터 (admin)
  POST /api/score/run                채점 실행 (admin)
  GET  /api/dashboard                최신 채점의 화면 JSON — site 는 자기 공장만 (admin·site)
  GET  /api/scores                   최신 채점 행 (admin·site)
  GET  /                             관제 화면. 데이터는 /api/dashboard 에서 fetch
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from jsonschema import Draft202012Validator, FormatChecker
from sqlalchemy.engine import Engine

from f4ge_supplier_risk.prediction import live
from f4ge_supplier_risk.web import auth, db
from f4ge_supplier_risk.web.dashboard_data import restrict

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
_validators: dict[str, Draft202012Validator] = {}


def _validator(contract: str) -> Draft202012Validator:
    if contract not in _validators:
        path = ROOT / "contracts" / f"{contract}.schema.json"
        if not path.exists():
            raise HTTPException(404, f"모르는 계약: {contract}")
        _validators[contract] = Draft202012Validator(json.loads(path.read_text()), format_checker=FormatChecker())
    return _validators[contract]


def create_app(eng: Engine) -> FastAPI:
    app = FastAPI(title="f4ge-supplier-risk", version="0.1.0")
    need = lambda *roles: Depends(auth.make_dependency(eng, *roles))  # noqa: E731

    @app.get("/health")
    def health() -> dict[str, Any]:
        run = db.latest_run(eng)
        return {"ok": True, "latest_run": {k: run[k] for k in ("run_id", "scored_at", "n_scored")} if run else None}

    @app.post("/api/ingest/{contract}")
    def ingest(contract: str, rows: list[dict[str, Any]], p: auth.Principal = need("ingest", "admin")):
        if contract not in db.CONTRACT_TABLES:
            raise HTTPException(404, f"입력 계약이 아니다: {contract}")
        v = _validator(contract)
        errors = [{"order_id": r.get("order_id"), "error": e.message} for r in rows for e in v.iter_errors(r)]
        if errors:
            # 배치 전체를 거절한다. 반쯤 들어간 배치는 재전송을 어렵게 만든다.
            raise HTTPException(422, {"invalid": len(errors), "first": errors[:5]})
        n = db.upsert_rows(eng, contract, rows)
        return {"contract": contract, "upserted": n}

    @app.post("/api/masters/products")
    def put_products(rows: list[dict[str, Any]], p: auth.Principal = need("admin")):
        db.upsert_master(eng, db.products, rows, "product_code")
        return {"upserted": len(rows)}

    @app.post("/api/masters/factories")
    def put_factories(rows: list[dict[str, Any]], p: auth.Principal = need("admin")):
        db.upsert_master(eng, db.factories, rows, "factory_id")
        return {"upserted": len(rows)}

    @app.post("/api/score/run")
    def score_run(p: auth.Principal = need("admin")):
        try:
            return live.score_from_db(eng, note=f"by {p.label}")
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    def _latest_dash(p: auth.Principal) -> dict[str, Any]:
        run = db.latest_run(eng)
        if not run:
            raise HTTPException(404, "채점 결과가 아직 없다 — POST /api/score/run")
        dash = json.loads(run["dashboard"])
        dash["runId"] = run["run_id"]
        return restrict(dash, p.factory_id) if p.role == "site" else dash

    @app.get("/api/dashboard")
    def dashboard(p: auth.Principal = need("admin", "site")):
        return JSONResponse(_latest_dash(p))

    @app.get("/api/scores")
    def score_rows(p: auth.Principal = need("admin", "site")):
        run = db.latest_run(eng)
        if not run:
            raise HTTPException(404, "채점 결과가 아직 없다")
        rows = [json.loads(s) for s in _score_payloads(eng, run["run_id"])]
        if p.role == "site":
            rows = [r for r in rows if r["factory_id"] == p.factory_id]
        return {"run_id": run["run_id"], "rows": rows}

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        html = (STATIC / "관제.html").read_text().replace("__DATA__", "__LIVE__")
        return HTMLResponse(html)

    return app


def _score_payloads(eng: Engine, run_id: str):
    import sqlalchemy as sa

    with eng.connect() as cx:
        return [p for (p,) in cx.execute(sa.select(db.scores.c.payload).where(db.scores.c.run_id == run_id))]


def seed_from_generated(eng: Engine, data: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """생성 데이터를 **같은 계약 경로**로 넣는다 — 운영 경로의 통합 테스트이자 데모 데이터."""
    from f4ge_supplier_risk import contracts

    fac_of = {o["order_id"]: o["factory_id"] for o in data["orders"]}
    db.upsert_master(eng, db.products, [
        {"product_code": p["product_id"].removeprefix("prd_"), "nominal_cycle_sec": p["nominal_cycle_sec"],
         "material_per_unit": p["material_per_unit"], "difficulty": p["difficulty"]} for p in data["products"]],
        "product_code")
    db.upsert_master(eng, db.factories, [
        {"factory_id": f["factory_id"], "product_code": f["product_id"].removeprefix("prd_"),
         "has_mes": bool(f["has_mes"]), "label": f["factory_id"]} for f in data["factories"]], "factory_id")
    counts = {
        "supplier-order.v1": db.upsert_rows(eng, "supplier-order.v1", [contracts.order_row(o) for o in data["orders"]]),
        "factory-report.v1": db.upsert_rows(eng, "factory-report.v1",
                                            [contracts.report_row(r, fac_of[r["order_id"]]) for r in data["factory_reports"]]),
        "fai-report.v1": db.upsert_rows(eng, "fai-report.v1", [contracts.fai_row(f, fac_of[f["order_id"]]) for f in data["fai_reports"]]),
        "cell-daily.v1": db.upsert_rows(eng, "cell-daily.v1", [contracts.cell_row(c, fac_of[c["order_id"]]) for c in data["cell_daily"]]),
        "order-quality-outcome.v1": db.upsert_rows(eng, "order-quality-outcome.v1",
                                                   [contracts.outcome_row(q, fac_of[q["order_id"]]) for q in data["quality_outcomes"]]),
    }
    return counts
