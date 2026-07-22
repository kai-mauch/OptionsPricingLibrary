"""
Monte Carlo pricing for European options.

Idea: simulate many possible terminal stock prices under the
risk-neutral measure, compute the option payoff on each simulated
path, average those payoffs, then discount back to today. As the
number of simulated paths grows, this average converges to the
Black-Scholes price for European options (law of large numbers).
"""

import numpy as np


def _simulate_terminal_prices(S, T, sigma, r, n_paths, seed):
    """
    Simulate terminal stock prices ST using the risk-neutral
    Geometric Brownian Motion solution:

        ST = S * exp[ (r - sigma^2/2) * T + sigma * sqrt(T) * Z ]

    where Z ~ Standard Normal. This is the exact (not approximate)
    solution to the GBM stochastic differential equation, so we can
    jump straight to expiry in one step per path -- no need to
    simulate intermediate time steps for European options.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)  # NumPy array with length n_paths

    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z   # calculation is performed on each Z value in the array, called vectorization

    terminal_prices = S * np.exp(drift + diffusion)  # also a NumPy array
    return terminal_prices


def monte_carlo_call(S, K, T, sigma, r, n_paths=10000, seed=None):
    """
    Price a European call via Monte Carlo simulation.

    n_paths : number of simulated price paths. Higher = more accurate,
              slower. Error shrinks proportional to 1/sqrt(n_paths).
    seed : optional RNG seed for reproducible results (useful in tests).
    """
    terminal_prices = _simulate_terminal_prices(S, T, sigma, r, n_paths, seed)

    payoffs = np.maximum(terminal_prices - K, 0.0) # NumPy array

    discounted_expected_payoff = np.exp(-r * T) * payoffs.mean()
    return discounted_expected_payoff


def monte_carlo_put(S, K, T, sigma, r, n_paths=10000, seed=None):
    """
    Price a European put via Monte Carlo simulation.
    """
    terminal_prices = _simulate_terminal_prices(S, T, sigma, r, n_paths, seed)

    payoffs = np.maximum(K - terminal_prices, 0.0)

    discounted_expected_payoff = np.exp(-r * T) * payoffs.mean()
    return discounted_expected_payoff