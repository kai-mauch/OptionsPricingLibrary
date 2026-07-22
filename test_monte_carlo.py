"""
Unit tests for Monte Carlo pricing. Main check: as the number of
simulated paths grows, the Monte Carlo price should converge to the
Black-Scholes price for European options. A seed is used everywhere
so results are reproducible instead of flaky.
"""

import pytest
from PricingModels.monte_carlo import monte_carlo_call, monte_carlo_put
from PricingModels.black_scholes import black_scholes_call, black_scholes_put
from options import EuropeanCall, EuropeanPut


def test_monte_carlo_call_converges_to_black_scholes():
    bs_price = black_scholes_call(S=100, K=105, T=1, sigma=0.2, r=0.05)
    mc_price = monte_carlo_call(S=100, K=105, T=1, sigma=0.2, r=0.05,
                                 n_paths=500_000, seed=42)
    # Monte Carlo has sampling error, so the tolerance is much looser
    # than the binomial convergence test.
    assert mc_price == pytest.approx(bs_price, abs=0.10)


def test_monte_carlo_put_converges_to_black_scholes():
    bs_price = black_scholes_put(S=100, K=105, T=1, sigma=0.2, r=0.05)
    mc_price = monte_carlo_put(S=100, K=105, T=1, sigma=0.2, r=0.05,
                                n_paths=500_000, seed=42)
    assert mc_price == pytest.approx(bs_price, abs=0.10)


def test_monte_carlo_same_seed_is_reproducible():
    price_1 = monte_carlo_call(S=100, K=105, T=1, sigma=0.2, r=0.05, n_paths=10_000, seed=7)
    price_2 = monte_carlo_call(S=100, K=105, T=1, sigma=0.2, r=0.05, n_paths=10_000, seed=7)
    assert price_1 == price_2


def test_monte_carlo_more_paths_reduces_variance():
    """
    Run the same pricing twice at low path counts with different
    seeds -- the spread between them should shrink once we use a much
    higher path count. Confirms error actually decreases with more
    simulations rather than the function just returning a constant.
    """
    low_a = monte_carlo_call(S=100, K=105, T=1, sigma=0.2, r=0.05, n_paths=200, seed=1)
    low_b = monte_carlo_call(S=100, K=105, T=1, sigma=0.2, r=0.05, n_paths=200, seed=2)

    high_a = monte_carlo_call(S=100, K=105, T=1, sigma=0.2, r=0.05, n_paths=200_000, seed=1)
    high_b = monte_carlo_call(S=100, K=105, T=1, sigma=0.2, r=0.05, n_paths=200_000, seed=2)

    assert abs(high_a - high_b) < abs(low_a - low_b)


def test_european_call_dispatches_to_monte_carlo():
    option = EuropeanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)
    direct = monte_carlo_call(S=100, K=105, T=1, sigma=0.2, r=0.05, n_paths=10_000, seed=99)
    assert option.price(method="monte_carlo", n_paths=10_000, seed=99) == direct


def test_european_put_dispatches_to_monte_carlo():
    option = EuropeanPut(S=100, K=105, T=1, sigma=0.2, r=0.05)
    direct = monte_carlo_put(S=100, K=105, T=1, sigma=0.2, r=0.05, n_paths=10_000, seed=99)
    assert option.price(method="monte_carlo", n_paths=10_000, seed=99) == direct