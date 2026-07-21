"""
Unit tests for Black-Scholes pricing, checked against a known
hand-computed benchmark value.
"""

import pytest
from options import EuropeanCall, EuropeanPut
from PricingModels.black_scholes import black_scholes_call


def test_black_scholes_call_known_value():
    """
    S=100, K=105, T=1, sigma=0.2, r=0.05
    Hand-computed / widely cited benchmark: price ~= 8.02
    """
    price = black_scholes_call(S=100, K=105, T=1, sigma=0.2, r=0.05)
    assert price == pytest.approx(8.02, abs=0.01)


def test_european_call_dispatches_to_black_scholes():
    """
    Confirms the EuropeanCall.price(method="black_scholes") wrapper
    returns the same value as calling the pricer function directly.
    """
    option = EuropeanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)
    direct = black_scholes_call(S=100, K=105, T=1, sigma=0.2, r=0.05)
    assert option.price(method="black_scholes") == direct

def test_european_put_dispatches_to_black_scholes():
    option = EuropeanPut(S=100, K=105, T=1, sigma=0.2, r=0.05)
    price = option.price(method="black_scholes")
    assert price > 0

def test_deep_itm_call_approaches_intrinsic_value():
    """
    A deep in-the-money call with near-zero time value should price
    close to its intrinsic value: S - K (discounted slightly by rate).
    """
    option = EuropeanCall(S=200, K=50, T=0.01, sigma=0.2, r=0.05)
    price = option.price(method="black_scholes")
    intrinsic = 200 - 50
    assert price == pytest.approx(intrinsic, rel=0.01)


def test_invalid_spot_price_raises():
    with pytest.raises(ValueError):
        EuropeanCall(S=-10, K=105, T=1, sigma=0.2, r=0.05)


def test_unimplemented_method_raises():
    option = EuropeanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)
    with pytest.raises(NotImplementedError):
        option.price(method="binomial")