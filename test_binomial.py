"""
Unit tests for binomial tree pricing. The main check: as the number
of steps grows, the binomial price should converge to the
Black-Scholes price for European options.
"""

import pytest
from PricingModels.binomial import binomial_call, binomial_put
from PricingModels.black_scholes import black_scholes_call, black_scholes_put


def test_binomial_call_converges_to_black_scholes():
    bs_price = black_scholes_call(S=100, K=105, T=1, sigma=0.2, r=0.05)
    tree_price = binomial_call(S=100, K=105, T=1, sigma=0.2, r=0.05, steps=500)
    assert tree_price == pytest.approx(bs_price, abs=0.05)


def test_binomial_put_converges_to_black_scholes():
    bs_price = black_scholes_put(S=100, K=105, T=1, sigma=0.2, r=0.05)
    tree_price = binomial_put(S=100, K=105, T=1, sigma=0.2, r=0.05, steps=500)
    assert tree_price == pytest.approx(bs_price, abs=0.05)


def test_american_put_worth_at_least_european_put():
    """
    American puts have early-exercise value, so they should never be
    priced lower than the equivalent European put.
    """
    european = binomial_put(S=100, K=105, T=1, sigma=0.2, r=0.05, steps=200, american=False)
    american = binomial_put(S=100, K=105, T=1, sigma=0.2, r=0.05, steps=200, american=True)
    assert american >= european


def test_american_put_exceeds_european_when_deep_itm():
    """
    Strict inequality, unlike the >= check above: a deep ITM put has
    real early-exercise value, so this proves the early-exercise
    branch actually fires rather than just never hurting the price.
    """
    european = binomial_put(S=50, K=100, T=1, sigma=0.2, r=0.05, steps=200, american=False)
    american = binomial_put(S=50, K=100, T=1, sigma=0.2, r=0.05, steps=200, american=True)
    assert american > european


def test_american_call_equals_european_call_no_dividends():
    """
    With no dividends, early exercise of a call is never optimal, so
    American and European call prices should match.
    """
    european = binomial_call(S=100, K=105, T=1, sigma=0.2, r=0.05, steps=200, american=False)
    american = binomial_call(S=100, K=105, T=1, sigma=0.2, r=0.05, steps=200, american=True)
    assert american == pytest.approx(european, abs=1e-6)