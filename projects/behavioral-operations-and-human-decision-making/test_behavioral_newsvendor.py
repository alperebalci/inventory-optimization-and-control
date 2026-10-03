import numpy as np
from behavioral_newsvendor import NewsvendorEconomics, rational_normal_order, pull_to_center, bounded_human_override, realized_profit, compare_policies


def test_high_margin_orders_above_mean():
    econ = NewsvendorEconomics(price=10, unit_cost=3, salvage=1)
    assert econ.critical_ratio > 0.5
    assert rational_normal_order(100, 20, econ) > 100


def test_pull_to_center_moves_toward_mean():
    assert pull_to_center(130, 100, 0.5) == 115


def test_override_is_bounded():
    assert bounded_human_override(100, 140, 0.10) == 110
    assert bounded_human_override(100, 60, 0.10) == 90


def test_rational_policy_has_lowest_monte_carlo_regret_among_shrunk_policies():
    econ = NewsvendorEconomics(price=10, unit_cost=3, salvage=1)
    rows = compare_policies(100, 20, econ, n=120000, seed=7)
    assert rows[0]["regret"] == 0
    assert all(r["regret"] >= -0.05 for r in rows[1:])


def test_profit_vector_shape():
    econ = NewsvendorEconomics(price=8, unit_cost=4, salvage=1)
    out = realized_profit(10, np.array([5, 10, 15]), econ)
    assert out.shape == (3,)
