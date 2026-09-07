import pytest

from f4ge_supplier_risk import config
from f4ge_supplier_risk.generator import latent, masters, orders, production


@pytest.fixture(scope="session")
def cfg():
    c = config.load("configs/generator.yaml")
    c["scale"]["months"] = 6  # 테스트는 작게
    return c


@pytest.fixture(scope="session")
def table(cfg):
    """피처 테이블 — 모델 테스트가 쓴다."""
    from f4ge_supplier_risk.features.build import build
    from f4ge_supplier_risk.generator.pipeline import build_dataset

    c = dict(cfg)
    c["scale"] = {**cfg["scale"], "months": 24}
    return build(build_dataset(c))


@pytest.fixture(scope="session")
def truth_map(cfg):
    """오더별 진실 잠재값 — 사유 분해 검증에 쓴다."""
    from f4ge_supplier_risk.generator.pipeline import build_dataset

    c = dict(cfg)
    c["scale"] = {**cfg["scale"], "months": 24}
    return {r["order_id"]: r for r in build_dataset(c)["ground_truth"]}


@pytest.fixture(scope="session")
def world(cfg):
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
