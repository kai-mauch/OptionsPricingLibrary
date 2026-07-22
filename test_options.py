"""
Unit tests for the AmericanCall / AmericanPut classes: confirms the
dispatcher wires correctly to the binomial engine with american=True,
and that Black-Scholes is correctly rejected.
"""

import pytest
from options import AmericanCall, AmericanPut, EuropeanCall, EuropeanPut


def test_american_put_exceeds_european_put_when_deep_itm():
    european = EuropeanPut(S=50, K=100, T=1, sigma=0.2, r=0.05)
    american = AmericanPut(S=50, K=100, T=1, sigma=0.2, r=0.05)
    assert american.price(method="binomial", steps=200) > european.price(method="binomial", steps=200)


def test_american_call_equals_european_call_no_dividends():
    european = EuropeanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)
    american = AmericanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)
    assert american.price(method="binomial", steps=200) == pytest.approx(
        european.price(method="binomial", steps=200), abs=1e-6
    )


def test_american_defaults_to_binomial_method():
    """
    Unlike European classes (which default to black_scholes),
    American classes should default to binomial since Black-Scholes
    has no closed form for early exercise.
    """
    option = AmericanPut(S=50, K=100, T=1, sigma=0.2, r=0.05)
    assert option.price() == option.price(method="binomial")


def test_american_call_rejects_black_scholes():
    option = AmericanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)
    with pytest.raises(ValueError):
        option.price(method="black_scholes")


def test_american_put_rejects_black_scholes():
    option = AmericanPut(S=100, K=105, T=1, sigma=0.2, r=0.05)
    with pytest.raises(ValueError):
        option.price(method="black_scholes")


def test_american_put_unimplemented_monte_carlo_raises():
    option = AmericanPut(S=100, K=105, T=1, sigma=0.2, r=0.05)
    with pytest.raises(NotImplementedError):
        option.price(method="monte_carlo")