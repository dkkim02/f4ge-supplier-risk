"""API key 인증 — 역할 셋.

  admin   전 공장을 본다. 채점 실행·키 발급.
  site    자기 공장 하나만 — **서버에서** factory_id 로 잘라 준다. 화면 필터가 아니다.
  ingest  계약 행을 넣는다(FactoryOS 연동용). 읽기 없음.
키는 해시만 저장한다. 발급 시 한 번만 원문을 보여 준다.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import sqlalchemy as sa
from fastapi import Header, HTTPException
from sqlalchemy.engine import Engine

from f4ge_supplier_risk.web import db

ROLES = ("admin", "site", "ingest")


@dataclass(frozen=True)
class Principal:
    label: str
    role: str
    factory_id: str | None

    def can_read(self, factory_id: str) -> bool:
        return self.role == "admin" or (self.role == "site" and self.factory_id == factory_id)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def issue_key(eng: Engine, label: str, role: str, factory_id: str | None = None) -> str:
    if role not in ROLES:
        raise ValueError(f"role 은 {ROLES} 중 하나")
    if role == "site" and not factory_id:
        raise ValueError("site 키에는 factory_id 가 필요하다")
    key = f"srk_{role}_{secrets.token_urlsafe(24)}"
    with eng.begin() as cx:
        cx.execute(db.api_keys.insert().values(
            key_hash=_hash(key), label=label, role=role, factory_id=factory_id,
            created_at=db.now_iso(), revoked=False))
    return key


def resolve(eng: Engine, key: str | None) -> Principal:
    if not key:
        raise HTTPException(401, "X-API-Key 헤더가 필요하다")
    with eng.connect() as cx:
        r = cx.execute(sa.select(db.api_keys).where(db.api_keys.c.key_hash == _hash(key))).first()
    if not r or r.revoked:
        raise HTTPException(401, "키가 없거나 폐기됐다")
    return Principal(label=r.label, role=r.role, factory_id=r.factory_id)


def make_dependency(eng: Engine, *allowed: str):
    def dep(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Principal:
        p = resolve(eng, x_api_key)
        if allowed and p.role not in allowed:
            raise HTTPException(403, f"이 작업은 {allowed} 역할만 할 수 있다")
        return p
    return dep
