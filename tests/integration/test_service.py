"""운영 서버 통합 — 계약 경로로 넣고, 채점하고, 역할별로 잘라 주는가."""

import json

import pytest
from fastapi.testclient import TestClient

from f4ge_supplier_risk.web import auth, db
from f4ge_supplier_risk.web.service import create_app, seed_from_generated


@pytest.fixture(scope="module")
def env(tmp_path_factory, dataset):
    eng = db.connect(f"sqlite:///{tmp_path_factory.mktemp('db')}/t.db")
    counts = seed_from_generated(eng, dataset)
    assert counts["supplier-order.v1"] == len(dataset["orders"])
    keys = {
        "admin": auth.issue_key(eng, "t-admin", "admin"),
        "site": auth.issue_key(eng, "t-site", "site", factory_id=dataset["factories"][0]["factory_id"]),
        "ingest": auth.issue_key(eng, "t-ingest", "ingest"),
    }
    return {"client": TestClient(create_app(eng)), "keys": keys, "eng": eng,
            "site_fac": dataset["factories"][0]["factory_id"], "n_fac": len(dataset["factories"])}


def _h(env, role):
    return {"X-API-Key": env["keys"][role]}


def test_no_key_is_401(env):
    assert env["client"].get("/api/dashboard").status_code == 401


def test_ingest_rejects_invalid_batch_whole(env):
    c = env["client"]
    good = json.loads(open("contracts/examples/supplier-order.sample.jsonl").readline())
    bad = {**good, "order_id": "ord_bad", "market": "mars"}
    r = c.post("/api/ingest/supplier-order.v1", json=[good, bad], headers=_h(env, "ingest"))
    assert r.status_code == 422 and r.json()["detail"]["invalid"] == 1
    # 하나라도 틀리면 좋은 행도 안 들어간다
    assert not any(o["order_id"] == "ord_bad" for o in db.load_rows(env["eng"], "supplier-order.v1"))


def test_ingest_key_cannot_read(env):
    assert env["client"].get("/api/dashboard", headers=_h(env, "ingest")).status_code == 403


def test_site_cannot_run_scoring(env):
    assert env["client"].post("/api/score/run", headers=_h(env, "site")).status_code == 403


def test_score_run_then_dashboard_scoped_by_role(env):
    c = env["client"]
    before = c.get("/api/dashboard", headers=_h(env, "admin"))
    assert before.status_code == 404  # 채점 전

    r = c.post("/api/score/run", headers=_h(env, "admin"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_scored"] >= body["n_train"] >= 40

    admin = c.get("/api/dashboard", headers=_h(env, "admin")).json()
    assert admin["role"] == "admin" and len(admin["factories"]) == env["n_fac"]
    assert admin["coverage"]["total"] == env["n_fac"] and admin["coverage"]["c"] == 0

    site = c.get("/api/dashboard", headers=_h(env, "site")).json()
    assert site["role"] == "site"
    assert {r["fac"] for r in site["rows"]} == {env["site_fac"]}
    assert [f["id"] for f in site["factories"]] == [env["site_fac"]]

    scores = c.get("/api/scores", headers=_h(env, "site")).json()["rows"]
    assert scores and all(s["factory_id"] == env["site_fac"] for s in scores)
    assert set(scores[0]) >= {"schema_version", "predicted_escape_ppm", "recommended_action"}


def test_index_serves_live_page(env):
    html = env["client"].get("/").text
    assert "__LIVE__" in html and "__DATA__" not in html
