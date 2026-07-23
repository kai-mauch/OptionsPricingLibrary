"""
Unit tests for Greeks: closed-form Black-Scholes formulas, numerical
finite-difference Greeks, and the .greeks() dispatcher on Option
classes.
"""

import pytest
from greeks import black_scholes_greeks, numerical_greeks
from options import EuropeanCall, EuropeanPut, AmericanCall, AmericanPut


# --- Closed-form sanity checks ---------------------------------------------

def test_deep_itm_call_delta_near_one():
    g = black_scholes_greeks(S=200, K=50, T=0.5, sigma=0.2, r=0.05, option_type="call")
    assert g["delta"] == pytest.approx(1.0, abs=0.01)


def test_deep_otm_call_delta_near_zero():
    g = black_scholes_greeks(S=50, K=200, T=0.5, sigma=0.2, r=0.05, option_type="call")
    assert g["delta"] == pytest.approx(0.0, abs=0.01)


def test_put_delta_equals_call_delta_minus_one():
    """
    Put-call parity implies delta_put = delta_call - 1 for the same
    S, K, T, sigma, r.
    """
    call_greeks = black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="call")
    put_greeks = black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="put")
    assert put_greeks["delta"] == pytest.approx(call_greeks["delta"] - 1, abs=1e-9)


def test_gamma_same_for_call_and_put():
    call_greeks = black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="call")
    put_greeks = black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="put")
    assert call_greeks["gamma"] == pytest.approx(put_greeks["gamma"], abs=1e-9)


def test_vega_positive():
    """Higher volatility should always increase option value."""
    g = black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="call")
    assert g["vega"] > 0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="straddle")


# --- Numerical vs closed-form agreement ------------------------------------

def test_numerical_binomial_greeks_match_closed_form():
    """
    Binomial with enough steps should approximate Black-Scholes
    closely, so numerical Greeks computed via binomial should be
    close to the exact closed-form Black-Scholes Greeks.
    """
    option = EuropeanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)
    closed_form = black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="call")
    numerical = numerical_greeks(option, method="binomial", steps=300)

    assert numerical["delta"] == pytest.approx(closed_form["delta"], abs=0.01)
    assert numerical["vega"] == pytest.approx(closed_form["vega"], abs=0.5)
    assert numerical["rho"] == pytest.approx(closed_form["rho"], abs=0.5)


def test_numerical_monte_carlo_greeks_match_closed_form():
    """
    Common random numbers (fixed seed across bumped re-pricings) keep
    Monte Carlo finite differences from being swamped by sampling
    noise, so this should also land close to the closed-form value.
    """
    option = EuropeanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)
    closed_form = black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="call")
    numerical = numerical_greeks(option, method="monte_carlo", n_paths=200_000)

    assert numerical["delta"] == pytest.approx(closed_form["delta"], abs=0.05)


# --- .greeks() dispatcher on Option classes --------------------------------

def test_european_call_greeks_dispatches_to_black_scholes():
    option = EuropeanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)
    direct = black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="call")
    assert option.greeks(method="black_scholes") == direct


def test_european_put_greeks_dispatches_to_black_scholes():
    option = EuropeanPut(S=100, K=105, T=1, sigma=0.2, r=0.05)
    direct = black_scholes_greeks(S=100, K=105, T=1, sigma=0.2, r=0.05, option_type="put")
    assert option.greeks(method="black_scholes") == direct


def test_american_greeks_rejects_black_scholes():
    """
    AmericanCall/Put should raise the same ValueError for .greeks()
    as they do for .price(), since greeks() reuses that validation.
    """
    option = AmericanPut(S=100, K=105, T=1, sigma=0.2, r=0.05)
    with pytest.raises(ValueError):
        option.greeks(method="black_scholes")


def test_american_greeks_via_binomial_works():
    option = AmericanPut(S=100, K=105, T=1, sigma=0.2, r=0.05)
    result = option.greeks(method="binomial", steps=200)
    assert set(result.keys()) == {"delta", "gamma", "vega", "theta", "rho"}
    # An American put's delta should be negative (price falls as S rises).
    assert result["delta"] < 0