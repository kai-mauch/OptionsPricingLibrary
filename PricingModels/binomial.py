"""
Cox-Ross-Rubinstein (CRR) binomial tree pricing.
Supports both European and American options.
"""

import math


def _binomial_price(S, K, T, sigma, r, steps, option_type, american):
    """
    Shared engine for binomial pricing. Builds a CRR tree and
    backward-induces the option value from expiry to today.
    """
    dt = T / steps                          # length of one time step
    u = math.exp(sigma * math.sqrt(dt))     # up-branch multiplier
    d = 1 / u                               # down-branch multiplier (CRR: d = 1/u keeps the tree recombining)
    p = (math.exp(r * dt) - d) / (u - d)    # risk-neutral probability of an up-move
    discount = math.exp(-r * dt)            # one-step discount factor

    ''' Remember that u and d are constant, as volatility (sigma) is assumed
    to be constant, and each time step has the same length dt.
    '''

    # Terminal stock prices at expiry.
    stock_prices = []
    for up_moves in range(steps + 1):
        down_moves = steps - up_moves

        up_multiplier = u ** up_moves
        down_multiplier = d ** down_moves

        stock_price = (S * up_multiplier * down_multiplier)
        stock_prices.append(stock_price)

    # Payoff at each terminal node.
    values = []
    for stock_price in stock_prices:
        if option_type == "call":
            payoff = max(stock_price - K, 0.0)

        elif option_type == "put":
            payoff = max(K - stock_price, 0.0)

        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        values.append(payoff)

    # Step backward through the tree, one layer at a time.
    for time_step in range(steps - 1, -1, -1):
        option_values_at_previous_step = []

        for number_of_up_moves in range(time_step + 1):
            down_option_value = values[number_of_up_moves]
            up_option_value = values[number_of_up_moves + 1]

            expected_future_value = (p * up_option_value + (1 - p) * down_option_value)

            continuation_value = discount * expected_future_value

            if american is True:
                number_of_down_moves = time_step - number_of_up_moves

                stock_price_at_node = (S * (u ** number_of_up_moves) * (d ** number_of_down_moves))

                if option_type == "call":
                    exercise_value = max(stock_price_at_node - K, 0.0)

                else:
                    exercise_value = max(K - stock_price_at_node, 0.0)

                option_value_at_node = max(continuation_value, exercise_value)

            else:
                option_value_at_node = continuation_value

            option_values_at_previous_step.append(option_value_at_node)

        values = option_values_at_previous_step

    return values[0]


def binomial_call(S, K, T, sigma, r, steps=100, american=False):
    """
    Price a call option using a binomial tree.

    steps : number of time steps in the tree. Higher = more accurate,
            slower. Converges toward the Black-Scholes price as
            steps approach infinity (for European options).
    american : if True, allows early exercise at every node.
    """
    return _binomial_price(S, K, T, sigma, r, steps, option_type="call", american=american)


def binomial_put(S, K, T, sigma, r, steps=100, american=False):
    """
    Price a put option using a binomial tree.
    """
    return _binomial_price(S, K, T, sigma, r, steps, option_type="put", american=american)