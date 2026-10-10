# Stochastic Volatility: Option Pricing and Hedging Model Risk

A Python-based quantitative finance project investigating how stochastic volatility affects option pricing, implied-volatility skew, and Black-Scholes delta hedging.

**Research question:** How does Black-Scholes hedging performance change when the underlying asset follows Heston stochastic-volatility dynamics?

## Project Overview

The project follows three stages:

1. **Monte Carlo Foundations:** Validate GBM simulations against analytical Black-Scholes pricing, examine convergence and antithetic variance reduction, and price an arithmetic-average Asian option.
2. **Heston Stochastic Volatility:** Simulate correlated stock and variance processes, price European options across strikes, and recover the implied-volatility skew.
3. **Delta Hedging and Model Risk:** Implement self-financing Black-Scholes delta hedging under both Black-Scholes and Heston dynamics, comparing rebalancing frequencies, random-seed robustness, and volatility-of-volatility sensitivity.

## Key Results

- **Monte Carlo efficiency:** Antithetic variates reduced pricing standard error by **29.7%** at equal payoff-evaluation counts.
- **Implied-volatility skew:** Heston-implied volatility declined from **22.88% to 17.26%** across strikes 80 to 120.
- **Hedging model risk:** Daily hedging RMSE increased from **0.4362 under Black-Scholes to 1.5558 under Heston**, a **3.57x increase**.
- **Robustness:** The RMSE ratio averaged **3.53x across 10 random seeds**, with an across-seed standard deviation of 0.07.
- **Parameter sensitivity:** Increasing Heston volatility of volatility from 0.0 to 0.4 raised daily hedging RMSE from approximately **0.43 to 1.97**.

## Technologies

Python, NumPy, SciPy, Matplotlib, Monte Carlo simulation, Black-Scholes, Heston stochastic volatility.

## Getting Started

Install dependencies:

`pip install -r requirements.txt`

Run the scripts in order:

```bash
python 01_monte_carlo_foundations.py
python 02_heston_pricing.py
python 03_model_risk_hedging.py
```

Run validation tests:

`python -m pytest -q`

## Limitations

The experiments use illustrative, uncalibrated risk-neutral parameters. Hedging assumes frictionless trading and excludes transaction costs. Monte Carlo confidence intervals capture sampling uncertainty but not time-discretisation bias.

The results demonstrate model risk in a controlled simulation rather than empirical trading performance.
