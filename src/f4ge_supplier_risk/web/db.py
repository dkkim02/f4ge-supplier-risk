"""저장소 — SQLite 한 파일, SQLAlchemy Core.

계약 행은 **받은 그대로**(JSON) 보관하고 조회 키만 열로 뽑는다. 채점은 저장된 계약 행을 다시 읽어
ingestion.derive 로 레코드를 만든다 — 그래서 스키마가 바뀌어도 원본은 남는다.
서버 한 대 · 공장 12곳 · 오더 수천 건 규모라 SQLite 로 충분하다. 옮길 일이 생기면 URL 만 바꾼다.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine

DEFAULT_URL = os.environ.get("SR_DATABASE_URL", "sqlite:///data/supplier_risk.db")

meta = sa.MetaData()

api_keys = sa.Table(
    "api_keys", meta,
    sa.Column("key_hash", sa.String(64), primary_key=True),
    sa.Column("label", sa.String(64), nullable=False),
    sa.Column("role", sa.String(16), nullable=False),  # admin | site | ingest
    sa.Column("factory_id", sa.String(64)),  # site 는 자기 공장 하나
    sa.Column("created_at", sa.String(32), nullable=False),
    sa.Column("revoked", sa.Boolean, nullable=False, default=False),
)
products = sa.Table(
    "products", meta,
    sa.Column("product_code", sa.String(32), primary_key=True),
    sa.Column("nominal_cycle_sec", sa.Float, nullable=False),
    sa.Column("material_per_unit", sa.Float, nullable=False),
    sa.Column("difficulty", sa.Float, nullable=False, default=0.0),
    sa.Column("label", sa.String(64)),
)
factories = sa.Table(
    "factories", meta,
    sa.Column("factory_id", sa.String(64), primary_key=True),
    sa.Column("label", sa.String(64)),
    sa.Column("product_code", sa.String(32)),
    sa.Column("has_mes", sa.Boolean, nullable=False, default=False),  # 자체 MES → FactoryOS 연동
    sa.Column("region", sa.String(64)),  # 위치(공장 카드에 표시). 없으면 표시하지 않는다
    sa.Column("profile", sa.Text),  # 보고 프로필 JSON — 집계 주기·수량 단위·불량코드 체계·채우는 필드 (09-08 저녁)
)


def _contract_table(name: str, *extra: sa.Column) -> sa.Table:
    return sa.Table(
        name, meta,
        sa.Column("order_id", sa.String(64), nullable=False, index=True),
        sa.Column("factory_id", sa.String(64), nullable=False, index=True),
        *extra,
        sa.Column("payload", sa.Text, nullable=False),
        sa.Column("ingested_at", sa.String(32), nullable=False),
    )


orders = _contract_table("orders", sa.Column("ordered_at", sa.String(32), nullable=False))
factory_reports = _contract_table("factory_reports", sa.Column("seq", sa.Integer, nullable=False))
fai_reports = _contract_table("fai_reports")
cell_daily = _contract_table("cell_daily", sa.Column("day_index", sa.Integer, nullable=False))
erp_daily = _contract_table("erp_daily", sa.Column("day_index", sa.Integer, nullable=False))
quality_outcomes = _contract_table("quality_outcomes", sa.Column("label_available_at", sa.String(32)))
sa.Index("ix_orders_pk", orders.c.order_id, unique=True)
sa.Index("ix_reports_pk", factory_reports.c.order_id, factory_reports.c.seq, unique=True)
sa.Index("ix_fai_pk", fai_reports.c.order_id, unique=True)
sa.Index("ix_cell_pk", cell_daily.c.order_id, cell_daily.c.day_index, unique=True)
sa.Index("ix_erp_pk", erp_daily.c.order_id, erp_daily.c.day_index, unique=True)
sa.Index("ix_outcome_pk", quality_outcomes.c.order_id, unique=True)

scores = sa.Table(
    "scores", meta,
    sa.Column("run_id", sa.String(32), nullable=False, index=True),
    sa.Column("order_id", sa.String(64), nullable=False, index=True),
    sa.Column("factory_id", sa.String(64), nullable=False, index=True),
    sa.Column("payload", sa.Text, nullable=False),  # supplier-risk-score.v1 행
)
# 공장 프로필 — run 마다 공장별 검출률 d 와 90% 구간을 쌓는다 (2026-09-11, [[모델_방향_결정]] §3 ③).
# 점수 행에도 d 가 있지만 오더 단위라 시계열로 읽기 어렵다. 여기는 (run, 공장) 하나에 행 하나.
# ⚠ 「검출률이 나아지고 있나」는 표본 ×2 전에는 주장하지 않는다(현행 d 최소 seed +0.007). 저장만 한다.
factory_profile = sa.Table(
    "factory_profile", meta,
    sa.Column("run_id", sa.String(32), nullable=False, index=True),
    sa.Column("factory_id", sa.String(64), nullable=False, index=True),
    sa.Column("scored_at", sa.String(32), nullable=False),
    sa.Column("detection", sa.Float, nullable=False),        # d = 1/(1+κ)
    sa.Column("detection_q05", sa.Float, nullable=False),    # Gamma 사후분포 90% 구간
    sa.Column("detection_q95", sa.Float, nullable=False),
    sa.Column("kappa", sa.Float, nullable=False),
    sa.Column("n_events", sa.Integer, nullable=False),       # κ 갱신에 쓰인 입고검사 불량 건수
)
score_runs = sa.Table(
    "score_runs", meta,
    sa.Column("run_id", sa.String(32), primary_key=True),
    sa.Column("scored_at", sa.String(32), nullable=False),
    sa.Column("n_train", sa.Integer), sa.Column("n_scored", sa.Integer),
    sa.Column("dashboard", sa.Text),  # 화면이 읽는 JSON 전체
    sa.Column("note", sa.Text),
)

CONTRACT_TABLES = {
    "supplier-order.v1": orders, "factory-report.v1": factory_reports, "fai-report.v1": fai_reports,
    "cell-daily.v1": cell_daily, "erp-daily.v1": erp_daily, "order-quality-outcome.v1": quality_outcomes,
}


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(url: str = DEFAULT_URL) -> Engine:
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    eng = sa.create_engine(url, future=True)
    meta.create_all(eng)
    _migrate(eng)
    return eng


def _migrate(eng: Engine) -> None:
    """create_all 은 기존 표에 열을 더하지 않는다. 09-08 저녁 이전 DB 에 `factories.region` 을 붙인다."""
    with eng.begin() as cx:
        cols = {r[1] for r in cx.exec_driver_sql("PRAGMA table_info(factories)")} if eng.dialect.name == "sqlite" else None
        if cols is not None and "region" not in cols:
            cx.exec_driver_sql("ALTER TABLE factories ADD COLUMN region VARCHAR(64)")
        if cols is not None and "profile" not in cols:
            cx.exec_driver_sql("ALTER TABLE factories ADD COLUMN profile TEXT")


def _extra_cols(name: str, row: dict[str, Any]) -> dict[str, Any]:
    if name == "supplier-order.v1":
        return {"ordered_at": row["ordered_at"]}
    if name == "factory-report.v1":
        return {"seq": int(row["seq"])}
    if name in ("cell-daily.v1", "erp-daily.v1"):
        return {"day_index": int(row["day_index"])}
    if name == "order-quality-outcome.v1":
        return {"label_available_at": row.get("label_available_at")}
    return {}


def upsert_rows(eng: Engine, contract: str, rows: list[dict[str, Any]]) -> int:
    """같은 키의 행은 새 것으로 덮는다 — 계약 행은 상태 스냅샷이라 마지막 것이 정본이다."""
    table = CONTRACT_TABLES[contract]
    key_cols = [c.name for c in table.columns if c.name not in ("payload", "ingested_at", "factory_id")]
    stamp = now_iso()
    with eng.begin() as cx:
        for r in rows:
            vals = {"order_id": r["order_id"], "factory_id": r["factory_id"], **_extra_cols(contract, r),
                    "payload": json.dumps(r, ensure_ascii=False), "ingested_at": stamp}
            cond = sa.and_(*[table.c[k] == vals[k] for k in key_cols])
            cx.execute(table.delete().where(cond))
            cx.execute(table.insert().values(**vals))
    return len(rows)


def load_rows(eng: Engine, contract: str, factory_id: str | None = None) -> list[dict[str, Any]]:
    table = CONTRACT_TABLES[contract]
    q = sa.select(table.c.payload)
    if factory_id:
        q = q.where(table.c.factory_id == factory_id)
    with eng.connect() as cx:
        return [json.loads(p) for (p,) in cx.execute(q)]


def load_products(eng: Engine) -> list[dict[str, Any]]:
    with eng.connect() as cx:
        return [dict(r._mapping) for r in cx.execute(sa.select(products))]


def load_factories(eng: Engine) -> list[dict[str, Any]]:
    with eng.connect() as cx:
        return [dict(r._mapping) for r in cx.execute(sa.select(factories))]


def upsert_master(eng: Engine, table: sa.Table, rows: list[dict[str, Any]], key: str) -> None:
    with eng.begin() as cx:
        for r in rows:
            cx.execute(table.delete().where(table.c[key] == r[key]))
            cx.execute(table.insert().values(**r))


def save_run(eng: Engine, run_id: str, score_rows: list[dict[str, Any]], dashboard: dict[str, Any],
             n_train: int, note: str = "") -> None:
    with eng.begin() as cx:
        cx.execute(score_runs.insert().values(
            run_id=run_id, scored_at=now_iso(), n_train=n_train, n_scored=len(score_rows),
            dashboard=json.dumps(dashboard, ensure_ascii=False), note=note))
        if score_rows:
            cx.execute(scores.insert(), [
                {"run_id": run_id, "order_id": s["order_id"], "factory_id": s["factory_id"],
                 "payload": json.dumps(s, ensure_ascii=False)} for s in score_rows])


def latest_run(eng: Engine) -> dict[str, Any] | None:
    with eng.connect() as cx:
        r = cx.execute(sa.select(score_runs).order_by(score_runs.c.scored_at.desc()).limit(1)).first()
        return dict(r._mapping) if r else None


def save_profile(eng: Engine, run_id: str, scored_at: str, explain: dict[str, Any]) -> int:
    """`two_stage.explain()` 결과에서 공장별 d 행을 쌓는다. run 하나에 공장 수만큼."""
    rows = [
        {"run_id": run_id, "factory_id": fid, "scored_at": scored_at,
         "detection": float(explain["detection"][fid]),
         "detection_q05": float(explain["detection_q05"][fid]),
         "detection_q95": float(explain["detection_q95"][fid]),
         "kappa": float(explain["kappa"][fid]),
         "n_events": int(max(explain["kappa_obs"].get(fid, 0), 0))}
        for fid in explain["detection"]
    ]
    if rows:
        with eng.begin() as cx:
            cx.execute(factory_profile.insert(), rows)
    return len(rows)


def load_profile(eng: Engine, factory_id: str | None = None) -> list[dict[str, Any]]:
    """공장별 d 이력 — 시간순."""
    q = sa.select(factory_profile).order_by(factory_profile.c.scored_at, factory_profile.c.factory_id)
    if factory_id:
        q = q.where(factory_profile.c.factory_id == factory_id)
    with eng.connect() as cx:
        return [dict(r._mapping) for r in cx.execute(q)]
