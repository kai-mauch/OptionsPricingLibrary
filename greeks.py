"""
Greeks: sensitivities of an option's price to its inputs.

Two ways to get them, and this module offers both:

1. Closed-form (Black-Scholes only) -- exact analytical formulas.
   Fast, but only valid for European options priced via Black-Scholes.

2. Numerical / finite-difference (any pricing method) -- bump one
   input by a small amount, re-price, and measure the change. Works
   for Black-Scholes, Binomial, or Monte Carlo, and for American
   options too, since it only depends on `.price()` existing.
"""

import math
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. Closed-form Black-Scholes Greeks
# ---------------------------------------------------------------------------

def black_scholes_greeks(S, K, T, sigma, r, option_type="call"):
    """
    Analytical Greeks for a European option under Black-Scholes.

    Returns a dict: {delta, gamma, vega, theta, rho}

    Notes on units/convention:
    - vega is the price change per 1.0 (100 percentage points) change
      in sigma. Divide by 100 if you want "price change per 1% vol".
    - theta is the price change per 1.0 (one full year) of time
      passing. Divide by 365 for "price change per calendar day".
    - rho is the price change per 1.0 (100 percentage points) change
      in the interest rate. Divide by 100 for "price change per 1%".
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    # Same d1/d2 used in the pricing formula itself -- every Greek
    # below is just a derivative of the Black-Scholes price formula
    # with respect to one of its inputs (S, sigma, T, or r), and
    # those derivatives all reduce back down to expressions built out
    # of d1, d2, and the normal distribution. We're not re-deriving
    # calculus here at runtime, just plugging into formulas that were
    # already worked out on paper.
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # phi(d1): the standard normal PROBABILITY DENSITY at d1 (bell
    # curve height, not cumulative probability). This shows up
    # because differentiating N(d1) with respect to S, sigma, or T
    # -- via the chain rule -- always produces phi(d1) times some
    # other factor. It's the common thread running through gamma,
    # vega, and part of theta.
    pdf_d1 = norm.pdf(d1)

    # --- Gamma: rate of change of delta as S moves ---
    # Gamma is the SECOND derivative of price with respect to S (how
    # curved the price curve is). It's identical for calls and puts
    # because call and put prices only differ by a straight line
    # (put-call parity: call - put = S - K*e^-rT), and the second
    # derivative of a straight line is zero -- so whatever curvature
    # exists is shared by both.
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))

    # --- Vega: sensitivity to volatility ---
    # Also identical for calls and puts, for the same put-call-parity
    # reason as gamma: the S - K*e^-rT term in parity doesn't depend
    # on sigma at all, so d(call)/d(sigma) = d(put)/d(sigma).
    vega = S * pdf_d1 * math.sqrt(T)

    if option_type == "call":
        # Delta: probability-weighted sensitivity to a $1 move in S.
        # N(d1) is roughly (not exactly) the risk-neutral probability
        # the call finishes in-the-money -- which is why deep ITM
        # calls have delta near 1 (near-certain exercise) and deep
        # OTM calls have delta near 0 (near-certain no exercise).
        delta = norm.cdf(d1)

        # Theta: price change per year as time passes (T decreases).
        # Two components, added together:
        #   1) -(S*phi(d1)*sigma) / (2*sqrt(T))
        #      "Time value decay" -- the option loses extrinsic value
        #      as less time remains for the stock to move favorably.
        #      Always negative (a call constantly bleeds time value).
        #   2) -r*K*e^(-rT)*N(d2)
        #      Cost-of-carry on the strike: since exercising a call
        #      means paying K at expiry, the present value of that
        #      future payment (K*e^-rT) itself decays toward K as
        #      time passes, and this term captures that effect.
        theta = (
            -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        )

        # Rho: sensitivity to the risk-free rate. Higher rates make
        # holding a call (deferring the K payment) more attractive
        # relative to owning the stock outright, so call rho > 0.
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)

    else:  # put
        # Put delta = call delta - 1 (direct consequence of put-call
        # parity: differentiate both sides of call - put = S - K*e^-rT
        # with respect to S, and rearrange). Always negative: put
        # value falls as the stock rises.
        delta = norm.cdf(d1) - 1

        # Same time-decay term as the call (that part doesn't depend
        # on option_type), but the cost-of-carry term flips sign and
        # uses N(-d2) instead of N(d2), since a put holder RECEIVES K
        # at exercise rather than paying it -- so the discounting
        # effect on that future cash flow works in the opposite
        # direction.
        theta = (
            -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        )

        # Rho is negative for puts: higher rates make deferring the
        # K receipt (holding the put) less attractive, so put value
        # falls as rates rise.
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ---------------------------------------------------------------------------
# 2. Numerical (finite-difference) Greeks -- works for any pricing method
# ---------------------------------------------------------------------------

def _bumped_copy(option, **overrides):
    """
    Create a new option of the SAME class as `option`, with one or
    more parameters replaced. E.g. _bumped_copy(option, S=101) makes
    a fresh option identical to `option` except S=101.

    Why a new object instead of mutating `option` in place: mutating
    and restoring `option.S` around every bump would work, but is
    fragile (easy to forget to restore it, or to leave it in a bad
    state if an exception fires mid-calculation). Building a
    throwaway copy is simpler to reason about and can't leak state.
    """
    params = dict(S=option.S, K=option.K, T=option.T, sigma=option.sigma, r=option.r)
    params.update(overrides)
    return type(option)(**params)


def numerical_greeks(option, method="binomial", h=0.01, **kwargs):
    """
    Estimate Greeks via central finite differences, by bumping each
    input up and down and re-pricing with `option.price()`.

    This works for ANY method ("black_scholes", "binomial",
    "monte_carlo") and any option style (European or American),
    because it only relies on `.price()` -- it never looks inside the
    pricing model itself. That's the whole point of this function: it
    treats the pricer as a black box and just watches how its output
    responds to small nudges in the input.

    The math behind "central" differences: for a smooth function f,
    the true derivative f'(x) is approximated by
        f'(x) ~= (f(x + h) - f(x - h)) / (2h)
    This is more accurate than the simpler "forward difference"
        f'(x) ~= (f(x + h) - f(x)) / h
    because the central version's error shrinks with h^2, while the
    forward version's error only shrinks with h. Gamma (a SECOND
    derivative) uses a related three-point formula:
        f''(x) ~= (f(x+h) - 2*f(x) + f(x-h)) / h^2

    h : RELATIVE bump size (fraction of each parameter's value), not
        an absolute amount. Default 0.01 = bump each input by 1%.

        Why relative and not a tiny fixed number: binomial-tree
        prices are piecewise-LINEAR in S, with kinks wherever the
        strike falls between lattice nodes. A tiny absolute bump
        (e.g. 0.001 when S=100) often lands on the exact same linear
        segment as the unbumped price, so the finite difference just
        measures that segment's local slope -- which can differ
        noticeably from the option's true sensitivity. A bump that's
        a meaningful fraction of S crosses enough lattice segments to
        average out to something close to the true derivative. The
        same reasoning applies more mildly to Monte Carlo (sampling
        noise) and is harmless for the smooth Black-Scholes formula.
    """
    # For Monte Carlo, use the SAME random draws across all the bumped
    # re-pricings ("common random numbers"). Without this, each call
    # draws fresh random paths, and the sampling noise between calls
    # swamps the actual sensitivity we're trying to measure -- the
    # finite difference would be dominated by randomness, not signal.
    # Fixing the seed means the only thing that differs between the
    # "up" and "down" price calls is the parameter we intentionally
    # bumped, exactly like we want.
    if method == "monte_carlo":
        kwargs.setdefault("seed", 12345)

    # Small local helper so the five blocks below can just write
    # price(some_option) instead of repeating .price(method=..., **kwargs)
    # every time -- keeps the method/kwargs plumbing in one place.
    def price(opt):
        return opt.price(method=method, **kwargs)

    S, sigma, T, r = option.S, option.sigma, option.T, option.r

    # --- Delta & Gamma: both come from bumping S ---
    # We only need three prices (down, middle, up) to get both the
    # first derivative (delta) and second derivative (gamma) of price
    # with respect to S, so we compute them together here rather than
    # pricing S twice.
    h_S = h * S
    opt_S_up = _bumped_copy(option, S=S + h_S)
    opt_S_down = _bumped_copy(option, S=S - h_S)
    price_up, price_mid, price_down = price(opt_S_up), price(option), price(opt_S_down)

    # Central difference, first derivative: dPrice/dS
    delta = (price_up - price_down) / (2 * h_S)
    # Central difference, second derivative: d^2Price/dS^2
    gamma = (price_up - 2 * price_mid + price_down) / (h_S ** 2)

    # --- Vega: bump sigma, hold everything else fixed ---
    h_sigma = h * sigma
    opt_sigma_up = _bumped_copy(option, sigma=sigma + h_sigma)
    opt_sigma_down = _bumped_copy(option, sigma=sigma - h_sigma)
    vega = (price(opt_sigma_up) - price(opt_sigma_down)) / (2 * h_sigma)

    # --- Theta: bump T, hold everything else fixed ---
    # Guard against T - h_T going to zero or negative for short-dated
    # options (you can't price an option with negative time to
    # expiry) by capping the step at half of T.
    h_T = min(h * T, T / 2)
    opt_T_up = _bumped_copy(option, T=T + h_T)
    opt_T_down = _bumped_copy(option, T=T - h_T)
    # dPrice/dT tells us how price changes as MORE time remains.
    # Theta is conventionally defined the opposite way -- how price
    # changes as time PASSES (T shrinks toward expiry) -- which is
    # just the negative of that, hence the leading minus sign.
    theta = -(price(opt_T_up) - price(opt_T_down)) / (2 * h_T)

    # --- Rho: bump r, hold everything else fixed ---
    # Guard against r being at or near zero, where a purely relative
    # bump (h * r) would shrink to virtually nothing and the finite
    # difference would be dominated by floating-point noise instead
    # of signal -- so we floor the bump at 1% in absolute terms.
    h_r = h * max(abs(r), 0.01)
    opt_r_up = _bumped_copy(option, r=r + h_r)
    opt_r_down = _bumped_copy(option, r=r - h_r)
    rho = (price(opt_r_up) - price(opt_r_down)) / (2 * h_r)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}