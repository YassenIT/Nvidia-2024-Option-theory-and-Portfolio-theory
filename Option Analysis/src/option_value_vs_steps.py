import numpy as np
import matplotlib.pyplot as plt


def crr_binomial_tree_american_option(S, K, T, r, q, sigma, steps, option_type):
    """
    Cox-Ross-Rubinstein (CRR) Binomial Tree model for American options.

    Parameters:
    -----------
    S : float
        Current underlying asset price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annual)
    q : float
        Dividend yield (annual)
    sigma : float
        Volatility (annual)
    steps : int
        Number of time steps in the binomial tree
    option_type : str
        'call' for call option, 'put' for put option

    Returns:
    --------
    float
        The calculated option price at root node
    """
    dt = T / steps  # Time step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize asset price tree
    asset_tree = np.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            asset_tree[j, i] = S * (u ** j) * (d ** (i - j))

    # Initialize option price tree
    option_tree = np.zeros((steps + 1, steps + 1))
    for j in range(steps + 1):
        if option_type == 'call':
            option_tree[j, steps] = max(0, asset_tree[j, steps] - K)
        elif option_type == 'put':
            option_tree[j, steps] = max(0, K - asset_tree[j, steps])

    # Backward induction to calculate option price
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            if option_type == 'call':
                early_exercise = asset_tree[j, i] - K
            elif option_type == 'put':
                early_exercise = K - asset_tree[j, i]
            hold_value = np.exp(-r * dt) * (p * option_tree[j + 1, i + 1] + (1 - p) * option_tree[j, i + 1])
            option_tree[j, i] = max(early_exercise, hold_value)

    return option_tree[0, 0]  # Return the root option price


def plot_option_value_vs_steps(S, K, T, r, q, sigma, option_type):
    """
    Plot option value vs. number of steps.

    Parameters:
    -----------
    S : float
        Current underlying asset price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annual)
    q : float
        Dividend yield (annual)
    sigma : float
        Volatility (annual)
    option_type : str
        'call' for call option, 'put' for put option
    """
    steps_range = range(10, 501, 10)  # Steps from 10 to 500, increment by 10
    option_values = []

    for steps in steps_range:
        option_price = crr_binomial_tree_american_option(S, K, T, r, q, sigma, steps, option_type)
        option_values.append(option_price)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.scatter(steps_range, option_values, color='black')  # Use scatter plot with black dots
    plt.xlabel('Number of Steps')
    plt.ylabel('Option Value')
    plt.ylim(2, 2.99)  # Set y-axis range from 12 to 13
    plt.yticks(np.arange(2, 2.99, 0.03))  # Y-axis ticks with 0.03 changes
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Input Parameters
    S = 147.37  # Underlying asset price
    K = 157.5  # Strike price
    T = 14 / 365  # Time to expiration in years
    r = 4.75 / 100  # Risk-free interest rate
    q = 0.03  # Dividend yield
    sigma = 55 / 100  # Volatility
    option_type = 'call'  # Choose 'call' or 'put'

    # Plot the graph
    plot_option_value_vs_steps(S, K, T, r, q, sigma, option_type)
