from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class NewsvendorEconomics:
    price: float
    unit_cost: float
    salvage: float = 0.0

    @property
    def underage_cost(self) -> float:
        return self.price - self.unit_cost

    @property
    def overage_cost(self) -> float:
        return self.unit_cost - self.salvage

    @property
    def critical_ratio(self) -> float:
        den = self.underage_cost + self.overage_cost
        if den <= 0 or self.underage_cost < 0 or self.overage_cost < 0:
            raise ValueError("Economics must imply non-negative underage/overage costs")
        return self.underage_cost / den


def rational_normal_order(mean: float, std: float, economics: NewsvendorEconomics) -> float:
    if std < 0:
        raise ValueError("std must be non-negative")
    if std == 0:
        return float(mean)
    return float(mean + std * norm.ppf(economics.critical_ratio))


def pull_to_center(rational_order: float, demand_mean: float, bias_strength: float) -> float:
    if not 0 <= bias_strength <= 1:
        raise ValueError("bias_strength must lie in [0, 1]")
    return float((1 - bias_strength) * rational_order + bias_strength * demand_mean)


def anchor_adjust(anchor: float, rational_order: float, adjustment_fraction: float) -> float:
    if not 0 <= adjustment_fraction <= 1:
        raise ValueError("adjustment_fraction must lie in [0, 1]")
    return float(anchor + adjustment_fraction * (rational_order - anchor))


def bounded_human_override(recommendation: float, human_order: float, max_relative_override: float = 0.15) -> float:
    if recommendation < 0 or human_order < 0 or max_relative_override < 0:
        raise ValueError("Orders and override bound must be non-negative")
    delta = max_relative_override * max(recommendation, 1.0)
    return float(np.clip(human_order, recommendation - delta, recommendation + delta))


def realized_profit(order: float, demand: np.ndarray, economics: NewsvendorEconomics) -> np.ndarray:
    q = max(float(order), 0.0)
    d = np.asarray(demand, dtype=float)
    sales = np.minimum(q, d)
    leftover = np.maximum(q - d, 0.0)
    return economics.price * sales + economics.salvage * leftover - economics.unit_cost * q


def compare_policies(mean: float, std: float, economics: NewsvendorEconomics, bias_strengths=(0.25, 0.5, 0.75), n=100000, seed=2026):
    rng = np.random.default_rng(seed)
    demand = np.maximum(rng.normal(mean, std, n), 0.0)
    q_star = rational_normal_order(mean, std, economics)
    optimal_profit = float(realized_profit(q_star, demand, economics).mean())
    rows = [{"policy": "rational", "order": q_star, "mean_profit": optimal_profit, "regret": 0.0}]
    for b in bias_strengths:
        q = pull_to_center(q_star, mean, b)
        p = float(realized_profit(q, demand, economics).mean())
        rows.append({"policy": f"pull_to_center_{b:.2f}", "order": q, "mean_profit": p, "regret": optimal_profit - p})
    return rows
