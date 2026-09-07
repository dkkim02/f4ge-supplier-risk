"""전체 파이프라인 — 파일이 나오고, 라벨 시간이 지켜지는가."""

import json
from datetime import UTC, datetime

from f4ge_supplier_risk.generator.pipeline import OUT_FILES, generate


def _rows(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _dt(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def test_generate_writes_every_file(cfg, tmp_path):
    counts = generate(cfg, tmp_path)
    for name in OUT_FILES:
        assert (tmp_path / name).exists(), name
    assert counts["orders"] == cfg["scale"]["months"] * cfg["scale"]["orders_per_month"]


def test_private_fields_never_leak(cfg, tmp_path):
    """`_ordered_dt` 같은 생성 편의 필드가 모델 쪽으로 새면 안 된다."""
    generate(cfg, tmp_path)
    for row in _rows(tmp_path / "orders.jsonl"):
        assert not any(k.startswith("_") for k in row)


def test_label_arrives_after_shipping_and_scales_with_market(cfg, tmp_path):
    """라벨 지연은 시장마다 다르다. 유럽이 가장 늦다."""
    generate(cfg, tmp_path)
    orders = {o["order_id"]: o for o in _rows(tmp_path / "orders.jsonl")}
    by_market: dict[str, list[float]] = {}
    for o in _rows(tmp_path / "quality_outcomes.jsonl"):
        assert _dt(o["label_available_at"]) > _dt(o["shipped_at"])
        gap = (_dt(o["arrived_at"]) - _dt(o["shipped_at"])).days
        by_market.setdefault(orders[o["order_id"]]["market"], []).append(gap)
    med = {k: sorted(v)[len(v) // 2] for k, v in by_market.items()}
    assert med["us_west"] < med["us_east"] < med["eu"]


def test_ground_truth_is_a_separate_file(cfg, tmp_path):
    """진실 잠재값은 모델이 보는 파일 어디에도 섞이면 안 된다."""
    generate(cfg, tmp_path)
    leaky = {"internal_defect_rate_true", "escape_rate_true", "state_t", "reported_vs_true_gap"}
    for name in (
        "orders.jsonl",
        "factory_reports.jsonl",
        "quality_outcomes.jsonl",
        "order_meta.jsonl",
        "fai_reports.jsonl",
    ):
        for row in _rows(tmp_path / name):
            assert not (leaky & set(row)), name
