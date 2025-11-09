import numpy as np
import matplotlib.pyplot as plt


def crr_binomial_tree_real_option(S, T, r, sigma, steps, growth_rate=None, growth_cost=None, reduction_rate=None, cost_saving=None):
    """
    CRR Binomial Tree model for Real Options with growth and reduction options.

    Parameters:
    -----------
    S : float
        Present value of the project (e.g., drug patent value in Million SEK)
    T : float
        Time horizon in years
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility (annual)
    steps : int
        Number of steps in the binomial tree
    growth_rate : float, optional
        Increase in profit (e.g., 0.25 for 25%)
    growth_cost : float, optional
        Cost to acquire the growth option (as a fraction of present value)
    reduction_rate : float, optional
        Reduction in operations (e.g., 0.2 for 20%)
    cost_saving : float, optional
        Cost saving (as a fraction of reduced operations)

    Returns:
    --------
    tuple
        (asset_tree, option_tree, option_value)
        - asset_tree: numpy array of asset values at each node
        - option_tree: numpy array of option values at each node
        - option_value: the calculated option value at root node
    """
    dt = T / steps  # Time step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize asset price tree
    asset_tree = np.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            asset_tree[j, i] = S * (u ** j) * (d ** (i - j))

    # Initialize option price tree
    option_tree = np.zeros((steps + 1, steps + 1))

    # Backward induction with growth and reduction options
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continue_value = np.exp(-r * dt) * (p * option_tree[j + 1, i + 1] + (1 - p) * option_tree[j, i + 1])

            growth_value = -np.inf
            if growth_rate and growth_cost:
                growth_value = (1 + growth_rate) * asset_tree[j, i] - growth_cost * S

            reduction_value = -np.inf
            if reduction_rate and cost_saving:
                reduction_value = asset_tree[j, i] - cost_saving * (reduction_rate * asset_tree[j, i])

            option_tree[j, i] = max(continue_value, growth_value, reduction_value)

    return asset_tree, option_tree, option_tree[0, 0]  # Return the root option price


def plot_tree(asset_tree, option_tree, steps):
    """
    Plot the binomial tree.

    Parameters:
    -----------
    asset_tree : numpy array
        Asset values at each node
    option_tree : numpy array
        Option values at each node
    steps : int
        Number of time steps
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    x_offsets = np.arange(0, steps + 1) * 2  # Horizontal spacing
    y_offsets = [np.arange(0, i + 1) * 15 - (i * 7) for i in range(steps + 1)]  # Adjust positions for triangular shape

    for i in range(steps + 1):
        for j in range(i + 1):
            x_pos = x_offsets[i]
            y_pos = y_offsets[i][j]
            ax.text(x_pos, y_pos, f"{asset_tree[j, i]:.2f}\n{option_tree[j, i]:.2f}",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightblue'))

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
    S = 36069000  # Present value of the drug patent in Million SEK
    T = 10  # Time horizon in years
    r = 5 / 100  # Risk-free interest rate
    sigma = 30 / 100  # Volatility
    steps = 10  # Number of steps in the binomial tree

    # Case 1: Option to Buy a Competitor
    growth_rate = 0.25  # Increase in profit (25%)
    growth_cost = 0.2  # Cost to acquire competitor (20% of present value)
    asset_tree, option_tree, growth_option_value = crr_binomial_tree_real_option(S, T, r, sigma, steps, growth_rate=growth_rate, growth_cost=growth_cost)
    print(f"The value of the growth option is: {growth_option_value:.2f} Million SEK")
    plot_tree(asset_tree, option_tree, steps)

    # Case 2: Option to Reduce Operations
    reduction_rate = 0.2  # Reduction in operations (20%)
    cost_saving = 0.3  # Cost saving (30% of reduced operations)
    asset_tree, option_tree, reduction_option_value = crr_binomial_tree_real_option(S, T, r, sigma, steps, reduction_rate=reduction_rate, cost_saving=cost_saving)
    print(f"The value of the reduction option is: {reduction_option_value:.2f} Million SEK")
    plot_tree(asset_tree, option_tree, steps)
