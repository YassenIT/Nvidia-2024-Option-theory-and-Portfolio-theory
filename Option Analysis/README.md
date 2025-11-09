# Options Pricing Analysis

This project contains Python implementations of various options pricing models and analysis tools, including binomial trees, Black-Scholes formula, implied volatility calculations, and real options valuation.

## Project Structure

```
Option Analysis/
├── src/
│   ├── binomial_tree_american.py      # American option pricing using CRR binomial tree
│   ├── black_scholes.py               # Black-Scholes formula for European options
│   ├── implied_volatility.py          # Implied volatility calculation and analysis
│   ├── real_options.py                # Real options valuation (growth/reduction)
│   └── option_value_vs_steps.py       # Convergence analysis of binomial tree
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Scripts Description

### 1. Binomial Tree American Option (`binomial_tree_american.py`)

Implements the Cox-Ross-Rubinstein (CRR) binomial tree model for pricing American options (both call and put).

**Features:**
- Calculates option prices using binomial tree method
- Handles early exercise for American options
- Visualizes the binomial tree in a Christmas tree style
- Returns asset tree, option tree, and final option price

**Usage:**
```bash
python src/binomial_tree_american.py
```

### 2. Black-Scholes Model (`black_scholes.py`)

Implements the Black-Scholes formula for European options pricing.

**Features:**
- Calculates call and put option prices
- Includes Vega calculation (sensitivity to volatility)
- Includes Delta calculation (sensitivity to underlying price)

**Usage:**
```bash
python src/black_scholes.py
```

### 3. Implied Volatility Analysis (`implied_volatility.py`)

Calculates implied volatility from market prices using Newton-Raphson method.

**Features:**
- Computes implied volatility from observed market prices
- Analyzes volatility smile/skew patterns
- Plots implied volatility vs. strike prices
- Plots implied volatility vs. delta
- Tests sensitivity to different interest rates

**Usage:**
```bash
python src/implied_volatility.py
```

### 4. Real Options Valuation (`real_options.py`)

Values real options for strategic decisions using binomial tree approach.

**Features:**
- Models growth options (e.g., acquiring a competitor)
- Models reduction options (e.g., downsizing operations)
- Visualizes decision trees
- Applicable to project valuation and strategic planning

**Usage:**
```bash
python src/real_options.py
```

### 5. Option Value vs Steps Analysis (`option_value_vs_steps.py`)

Analyzes the convergence of binomial tree option prices as the number of steps increases.

**Features:**
- Tests option price convergence
- Plots option value vs. number of steps (10 to 500)
- Helps determine optimal number of steps for pricing

**Usage:**
```bash
python src/option_value_vs_steps.py
```

## Key Parameters

All scripts use standard options pricing parameters:

- **S**: Underlying asset price
- **K**: Strike price
- **T**: Time to expiration (in years)
- **r**: Risk-free interest rate (annual)
- **q**: Dividend yield (annual)
- **sigma**: Volatility (annual)
- **steps**: Number of time steps in binomial tree
- **option_type**: 'call' or 'put'

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **scipy**: Statistical functions (norm.cdf, norm.pdf, optimization)
- **pandas**: Data manipulation (for implied volatility analysis)

## Examples

### Pricing a Call Option

```python
from src.binomial_tree_american import crr_binomial_tree_american_option

S = 147.37  # Current price
K = 141.00  # Strike price
T = 14/365  # 14 days to expiration
r = 0.0475  # 4.75% interest rate
q = 0.03    # 3% dividend yield
sigma = 0.48  # 48% volatility
steps = 5

asset_tree, option_tree, price = crr_binomial_tree_american_option(
    S, K, T, r, q, sigma, steps, 'call'
)
print(f"Call option price: ${price:.2f}")
```

### Calculating Implied Volatility

```python
from src.implied_volatility import implied_volatility

market_price = 8.6
S = 147.37
K = 138
T = 14/365
r = 0.0475

iv = implied_volatility(market_price, S, K, T, r)
print(f"Implied volatility: {iv*100:.2f}%")
```

## Notes

- The binomial tree method converges to the Black-Scholes price as the number of steps increases
- American options can be exercised early, while European options cannot
- Implied volatility represents the market's expectation of future volatility
- Real options provide a framework for valuing flexibility in strategic decisions

## License

This project is for educational and research purposes.
