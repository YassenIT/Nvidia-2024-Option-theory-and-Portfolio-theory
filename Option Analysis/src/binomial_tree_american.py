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
    tuple
        (asset_tree, option_tree, option_price)
        - asset_tree: numpy array of asset prices at each node
        - option_tree: numpy array of option values at each node
        - option_price: the calculated option price at root node
    """
    dt = T / steps  # Time step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor (swapped)
    d = 1 / u  # Down factor (swapped)
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

    return asset_tree, option_tree, option_tree[0, 0]  # Return the root option price


def plot_tree_christmas_style(asset_tree, option_tree, steps):
    """
    Plot the binomial tree in a Christmas tree style.

    Parameters:
    -----------
    asset_tree : numpy array
        Asset prices at each node
    option_tree : numpy array
        Option values at each node
    steps : int
        Number of time steps
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    x_offsets = np.arange(0, steps + 1) * 2  # Horizontal spacing
    center_y = 0  # Center vertically for the root
    y_offsets = [center_y + np.arange(0, i + 1) * 15 - (i * 7) for i in range(steps + 1)]  # Adjust positions for triangular shape

    for i in range(steps + 1):
        for j in range(i + 1):
            x_pos = x_offsets[i]
            y_pos = y_offsets[i][j]
            ax.text(x_pos, y_pos, f"{asset_tree[j, i]:.2f}\n{option_tree[j, i]:.2f}",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightblue'))

    # Draw tree-like structure
    for i in range(steps):
        for j in range(i + 1):
            x_start = x_offsets[i]
            x_end = x_offsets[i + 1]
            y_start = y_offsets[i][j]
            y_end_up = y_offsets[i + 1][j + 1]  # Connect upward
            y_end_down = y_offsets[i + 1][j]  # Connect downward

            ax.plot([x_start, x_end], [y_start, y_end_up], color='green', linewidth=1.5)
            ax.plot([x_start, x_end], [y_start, y_end_down], color='red', linewidth=1.5)

    ax.set_xlim(-1, steps * 2 + 2)
    ax.set_ylim(-steps * 10 - 5, steps * 10 + 5)
    ax.axis('off')
    plt.show()


if __name__ == "__main__":
    # Input Parameters
    S = 147.37  # Underlying asset price
    K = 141.00  # Strike price
    T = 14 / 365  # Time to expiration in years
    r = 4.75 / 100  # Risk-free interest rate
    q = 0.03  # Dividend yield
    sigma = 48 / 100  # Volatility
    steps = 5  # Number of steps
    option_type = 'call'

    asset_tree, option_tree, option_price = crr_binomial_tree_american_option(S, K, T, r, q, sigma, steps, option_type)

    print(f"The calculated price of the {option_type} option is: {option_price:.2f}")

    # Plot the tree
    plot_tree_christmas_style(asset_tree, option_tree, steps)
