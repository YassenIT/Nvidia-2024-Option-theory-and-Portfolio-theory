from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def black_scholes(S, K, T, r, q, sigma, option_type):
    """
    Black-Scholes formula for European options.

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

    Returns:
    --------
    float
        The Black-Scholes option price
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price


def vega(S, K, T, r, sigma):
    """
    Vega calculation (partial derivative of price with respect to volatility).

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
    sigma : float
        Volatility (annual)

    Returns:
    --------
    float
        Vega value
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


def calculate_delta(S, K, T, r, sigma):
    """
    Delta calculation for call option.

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
    sigma : float
        Volatility (annual)

    Returns:
    --------
    float
        Delta value
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def plot_black_scholes_analysis(S, K, T, r, q, sigma, option_type='call'):
    """
    Create comprehensive visualizations for Black-Scholes model.

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
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Black-Scholes {option_type.capitalize()} Option Analysis', fontsize=16, fontweight='bold')

    # 1. Option Price vs. Underlying Asset Price
    ax1 = axes[0, 0]
    S_range = np.linspace(S * 0.5, S * 1.5, 100)
    prices = [black_scholes(s, K, T, r, q, sigma, option_type) for s in S_range]
    ax1.plot(S_range, prices, 'b-', linewidth=2, label=f'{option_type.capitalize()} Option')
    ax1.axvline(x=S, color='r', linestyle='--', linewidth=1, label=f'Current Price: ${S:.2f}')
    ax1.axvline(x=K, color='g', linestyle='--', linewidth=1, label=f'Strike Price: ${K:.2f}')
    ax1.set_xlabel('Underlying Asset Price ($)', fontsize=11)
    ax1.set_ylabel('Option Price ($)', fontsize=11)
    ax1.set_title('Option Price vs. Underlying Price', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Option Price vs. Volatility
    ax2 = axes[0, 1]
    sigma_range = np.linspace(0.1, 1.5, 100)
    prices_vol = [black_scholes(S, K, T, r, q, sig, option_type) for sig in sigma_range]
    ax2.plot(sigma_range * 100, prices_vol, 'purple', linewidth=2)
    ax2.axvline(x=sigma * 100, color='r', linestyle='--', linewidth=1, label=f'Current Vol: {sigma*100:.1f}%')
    ax2.set_xlabel('Volatility (%)', fontsize=11)
    ax2.set_ylabel('Option Price ($)', fontsize=11)
    ax2.set_title('Option Price vs. Volatility', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Option Price vs. Time to Expiration
    ax3 = axes[1, 0]
    T_range = np.linspace(0.01, T * 3, 100)
    prices_time = [black_scholes(S, K, t, r, q, sigma, option_type) for t in T_range]
    ax3.plot(T_range * 365, prices_time, 'orange', linewidth=2)
    ax3.axvline(x=T * 365, color='r', linestyle='--', linewidth=1, label=f'Current: {T*365:.0f} days')
    ax3.set_xlabel('Time to Expiration (days)', fontsize=11)
    ax3.set_ylabel('Option Price ($)', fontsize=11)
    ax3.set_title('Option Price vs. Time to Expiration', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Greeks (Delta and Vega) vs. Underlying Price
    ax4 = axes[1, 1]
    delta_values = [calculate_delta(s, K, T, r, sigma) for s in S_range]
    vega_values = [vega(s, K, T, r, sigma) for s in S_range]

    ax4_twin = ax4.twinx()
    line1 = ax4.plot(S_range, delta_values, 'b-', linewidth=2, label='Delta')
    line2 = ax4_twin.plot(S_range, vega_values, 'g-', linewidth=2, label='Vega')

    ax4.axvline(x=S, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(x=K, color='orange', linestyle='--', linewidth=1, alpha=0.5)

    ax4.set_xlabel('Underlying Asset Price ($)', fontsize=11)
    ax4.set_ylabel('Delta', fontsize=11, color='b')
    ax4_twin.set_ylabel('Vega', fontsize=11, color='g')
    ax4.set_title('Greeks: Delta and Vega', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='g')
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.show()


def print_black_scholes_summary(S, K, T, r, q, sigma, option_type='call'):
    """
    Print a detailed summary of Black-Scholes calculations.

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
    price = black_scholes(S, K, T, r, q, sigma, option_type)
    delta = calculate_delta(S, K, T, r, sigma)
    vega_val = vega(S, K, T, r, sigma)

    print("\n" + "="*60)
    print(f"BLACK-SCHOLES {option_type.upper()} OPTION ANALYSIS")
    print("="*60)
    print("\nINPUT PARAMETERS:")
    print(f"  Underlying Asset Price (S):  ${S:.2f}")
    print(f"  Strike Price (K):            ${K:.2f}")
    print(f"  Time to Expiration (T):      {T*365:.0f} days ({T:.4f} years)")
    print(f"  Risk-Free Rate (r):          {r*100:.2f}%")
    print(f"  Dividend Yield (q):          {q*100:.2f}%")
    print(f"  Volatility (σ):              {sigma*100:.2f}%")
    print(f"  Option Type:                 {option_type.capitalize()}")

    print("\nRESULTS:")
    print(f"  Option Price:                ${price:.2f}")
    print(f"  Delta (Δ):                   {delta:.4f}")
    print(f"  Vega (ν):                    {vega_val:.4f}")

    # Calculate intrinsic and time value
    if option_type == 'call':
        intrinsic_value = max(0, S - K)
    else:
        intrinsic_value = max(0, K - S)
    time_value = price - intrinsic_value

    print(f"\n  Intrinsic Value:             ${intrinsic_value:.2f}")
    print(f"  Time Value:                  ${time_value:.2f}")
    print(f"  Moneyness:                   ", end="")

    if option_type == 'call':
        if S > K:
            print("In-the-Money (ITM)")
        elif S == K:
            print("At-the-Money (ATM)")
        else:
            print("Out-of-the-Money (OTM)")
    else:
        if S < K:
            print("In-the-Money (ITM)")
        elif S == K:
            print("At-the-Money (ATM)")
        else:
            print("Out-of-the-Money (OTM)")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Input Parameters
    S = 147.37  # Underlying asset price
    K = 157.5  # Strike price
    T = 14 / 365  # Time to expiration in years
    r = 4.75 / 100  # Risk-free interest rate
    q = 0.03  # Dividend yield
    sigma = 55 / 100  # Volatility
    option_type = 'call'  # Choose 'call' or 'put'

    # Print detailed summary
    print_black_scholes_summary(S, K, T, r, q, sigma, option_type)

    # Create comprehensive graphs
    plot_black_scholes_analysis(S, K, T, r, q, sigma, option_type)
