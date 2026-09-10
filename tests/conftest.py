import pytest

from f4ge_supplier_risk import config
from f4ge_supplier_risk.generator import latent, masters, orders, production


# 2026-09-10 부터 configs/generator.yaml 은 보고 편향을 **통제**한다(편향 0). 그래서
# 편향 메커니즘 자체를 검증하는 테스트는 여기서 현실 값을 되돌려 쓴다 — 통제 세계에서
# "보고가 축소된다" 를 검증하면 항상 실패하고, 기댓값을 고치면 메커니즘 검증이 사라진다.
REALISTIC_BIAS = {
    "report_bias_mean": 0.5,
    "report_bias_sd": 0.25,
    "bias_sensitivity": 0.5,
    "mes_input_bias_range": [0.55, 1.0],
}


def _with_bias(c):
    return {**c, "assumptions": {**c["assumptions"], **REALISTIC_BIAS}}


@pytest.fixture(scope="session")
def cfg():
    c = config.load("configs/generator.yaml")
    c["scale"]["months"] = 6  # 테스트는 작게
    return c


@pytest.fixture(scope="session")
def cfg_biased(cfg):
    """현실 편향을 되돌린 설정 — 편향 메커니즘 검증 전용."""
    return _with_bias(cfg)


def _table(cfg):
    from f4ge_supplier_risk.features.build import build
    from f4ge_supplier_risk.generator.pipeline import build_dataset

    c = dict(cfg)
    c["scale"] = {**cfg["scale"], "months": 24}
    return build(build_dataset(c))


@pytest.fixture(scope="session")
def table(cfg):
    """피처 테이블 — 모델 테스트가 쓴다."""
    return _table(cfg)


@pytest.fixture(scope="session")
def table_biased(cfg_biased):
    return _table(cfg_biased)


@pytest.fixture(scope="session")
def dataset(cfg):
    """생성 레코드 전체(파일 8개 그대로) — 계약 검증이 쓴다."""
    from f4ge_supplier_risk.generator.pipeline import build_dataset

    return build_dataset(cfg)


def _truth_map(cfg):
    from f4ge_supplier_risk.generator.pipeline import build_dataset

    c = dict(cfg)
    c["scale"] = {**cfg["scale"], "months": 24}
    return {r["order_id"]: r for r in build_dataset(c)["ground_truth"]}


@pytest.fixture(scope="session")
def truth_map(cfg):
    """오더별 진실 잠재값 — 사유 분해 검증에 쓴다."""
    return _truth_map(cfg)


@pytest.fixture(scope="session")
def truth_map_biased(cfg_biased):
    return _truth_map(cfg_biased)


def _world(cfg):
    products = masters.build_products(cfg)
    factories = masters.build_factories(cfg, products)
    order_rows = orders.build_orders(cfg, factories, products)
    fac = {f["factory_id"]: f for f in factories}
    prod = {p["product_id"]: p for p in products}
    states = {
        f["factory_id"]: latent.factory_state_series(cfg, f["factory_id"], 200) for f in factories
    }
    built = []
    for o in order_rows[:60]:
        f, p = fac[o["factory_id"]], prod[o["product_id"]]
        wk = int((o["_ordered_dt"] - orders.EPOCH).days // 7)
        lat = latent.order_latents(cfg, o, f, p, float(states[f["factory_id"]][wk]))
        built.append((o, f, p, lat, production.build_truth(cfg, o, f, lat)))
    return built


@pytest.fixture(scope="session")
def world(cfg):
    return _world(cfg)


@pytest.fixture(scope="session")
def world_biased(cfg_biased):
    """현실 편향 세계 — 보고 축소 검증 전용."""
    return _world(cfg_biased)
