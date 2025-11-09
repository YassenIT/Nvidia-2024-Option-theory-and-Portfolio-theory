import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_call_price(S, K, T, r, sigma):
    """
    Black-Scholes formula for call option price.

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
        Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


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


def implied_volatility(market_price, S, K, T, r, tol=1e-5, max_iter=100):
    """
    Calculate implied volatility using Newton-Raphson method.

    Parameters:
    -----------
    market_price : float
        Observed market price of the option
    S : float
        Current underlying asset price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annual)
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    float
        Implied volatility (returns NaN if no solution found)
    """
    sigma = 0.5  # Initial guess

    for i in range(max_iter):
        price = black_scholes_call_price(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        price_diff = market_price - price

        if abs(price_diff) < tol:
            return sigma

        sigma += price_diff / v

    return np.nan  # Return NaN if no solution is found


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


if __name__ == "__main__":
    # Provided data
    data = pd.DataFrame({
        'Strike': [138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 152, 152.5, 154, 155, 156, 157.5, 158],
        'MarketPrice': [8.6, 7.8, 6.6, 5.7, 4.76, 3.86, 3.1, 2.37, 1.79, 1.25, 0.86, 0.57, 0.37, 0.15, 0.13, 0.07, 0.04, 0.03, 0.02]
    })

    expiry = 14 / 365  # Time to expiry in years
    asset_price = 147.37  # Current asset price

    # Interest rates from 3% to 13% with step of 2%
    interest_rates = np.arange(0.03, 0.14, 0.02)

    # Colors for plotting
    dot_colors = plt.cm.viridis(np.linspace(0, 1, len(interest_rates)))

    # Plotting Implied Volatility vs. Strike Prices
    plt.figure(figsize=(10, 6))
    for rate, color in zip(interest_rates, dot_colors):
        temp_data = data.copy()
        temp_data['ImpliedVolatility'] = temp_data.apply(
            lambda row: implied_volatility(row['MarketPrice'], asset_price, row['Strike'], expiry, rate), axis=1
        )

        plt.scatter(temp_data['Strike'], temp_data['ImpliedVolatility'], color=color, label=f'{rate*100:.0f}% Interest Rate')

    plt.title('Implied Volatility vs. Strike Prices by Interest Rates')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.show()

    # Plotting Implied Volatility vs. Delta
    plt.figure(figsize=(10, 6))
    for rate, color in zip(interest_rates, dot_colors):
        temp_data = data.copy()
        temp_data['ImpliedVolatility'] = temp_data.apply(
            lambda row: implied_volatility(row['MarketPrice'], asset_price, row['Strike'], expiry, rate), axis=1
        )

        temp_data['Delta'] = temp_data.apply(
            lambda row: calculate_delta(asset_price, row['Strike'], expiry, rate, row['ImpliedVolatility']), axis=1
        )

        plt.scatter(temp_data['Delta'], temp_data['ImpliedVolatility'], color=color, label=f'{rate*100:.0f}% Interest Rate')

    plt.title('Implied Volatility vs. Delta by Interest Rates')
    plt.xlabel('Delta')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.show()
