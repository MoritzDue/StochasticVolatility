from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class HestonParams:
    v0: float = 0.04
    theta: float = 0.04
    kappa: float = 2.0
    xi: float = 0.30
    rho: float = -0.70


def _scalar_or_array(value):
    value = np.asarray(value)
    return float(value) if value.ndim == 0 else value


def bs_call_price(spot, strike, rate, vol, maturity):
    """European call price under Black-Scholes, with no dividends."""
    s, k, v = np.broadcast_arrays(spot, strike, vol)
    s, k, v = s.astype(float), k.astype(float), v.astype(float)

    if maturity <= 0:
        return _scalar_or_array(np.maximum(s - k, 0.0))

    safe_v = np.maximum(v, 1e-12)
    sqrt_t = np.sqrt(maturity)

    d1 = (
        np.log(s / k)
        + (rate + 0.5 * safe_v**2) * maturity
    ) / (safe_v * sqrt_t)

    d2 = d1 - safe_v * sqrt_t

    stochastic_price = (
        s * norm.cdf(d1)
        - k * np.exp(-rate * maturity) * norm.cdf(d2)
    )

    deterministic_price = np.exp(-rate * maturity) * np.maximum(
        s * np.exp(rate * maturity) - k, 0.0
    )

    return _scalar_or_array(
        np.where(v > 0, stochastic_price, deterministic_price)
    )


def bs_call_delta(spot, strike, rate, vol, maturity):
    """Black-Scholes call delta, supporting scalar or array inputs."""
    s, k, v = np.broadcast_arrays(spot, strike, vol)
    s, k, v = s.astype(float), k.astype(float), v.astype(float)

    if maturity <= 0:
        return _scalar_or_array(
            np.where(s > k, 1.0, np.where(s < k, 0.0, 0.5))
        )

    safe_v = np.maximum(v, 1e-12)

    d1 = (
        np.log(s / k)
        + (rate + 0.5 * safe_v**2) * maturity
    ) / (safe_v * np.sqrt(maturity))

    stochastic_delta = norm.cdf(d1)
    deterministic_delta = (s * np.exp(rate * maturity) > k).astype(float)

    return _scalar_or_array(
        np.where(v > 0, stochastic_delta, deterministic_delta)
    )


def mc_summary(discounted_payoffs):
    """Return MC estimate, standard error and approximate 95% CI."""
    x = np.asarray(discounted_payoffs, dtype=float)

    if x.shape[0] < 2:
        raise ValueError("At least two independent observations are required.")

    estimate = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    half_width = 1.96 * se

    return estimate, se, estimate - half_width, estimate + half_width


def simulate_gbm_terminal(
    spot, rate, vol, maturity, n_paths, rng
):
    """Exact terminal GBM simulation."""
    z = rng.standard_normal(n_paths)

    return spot * np.exp(
        (rate - 0.5 * vol**2) * maturity
        + vol * np.sqrt(maturity) * z
    )


def simulate_gbm_paths(
    spot, rate, vol, maturity, n_steps, n_paths, rng
):
    """Exact GBM simulation on a discrete monitoring grid."""
    dt = maturity / n_steps
    drift = (rate - 0.5 * vol**2) * dt
    diffusion = vol * np.sqrt(dt)

    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = spot

    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        paths[t + 1] = paths[t] * np.exp(drift + diffusion * z)

    return paths


def simulate_heston(
    spot, rate, maturity, params, n_steps, n_paths, rng,
    store_paths=False
):
    """
    Risk-neutral Heston simulation.

    Uses full-truncation Euler for the raw variance process
    and log-Euler for the stock process.

    Returns terminal stock prices by default.
    If store_paths=True, returns (stock_paths, variance_paths).
    """
    if not -1.0 <= params.rho <= 1.0:
        raise ValueError("Heston correlation must lie between -1 and 1.")

    if min(params.v0, params.theta, params.kappa, params.xi) < 0:
        raise ValueError("Heston variance parameters must be non-negative.")

    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)
    orthogonal_weight = np.sqrt(1.0 - params.rho**2)

    stock = np.full(n_paths, spot, dtype=float)
    raw_variance = np.full(n_paths, params.v0, dtype=float)

    if store_paths:
        stock_paths = np.empty((n_steps + 1, n_paths))
        variance_paths = np.empty((n_steps + 1, n_paths))

        stock_paths[0] = stock
        variance_paths[0] = params.v0

    for t in range(n_steps):
        variance = np.maximum(raw_variance, 0.0)

        z_var = rng.standard_normal(n_paths)
        z_independent = rng.standard_normal(n_paths)
        z_stock = (
            params.rho * z_var
            + orthogonal_weight * z_independent
        )

        stock *= np.exp(
            (rate - 0.5 * variance) * dt
            + np.sqrt(variance) * sqrt_dt * z_stock
        )

        raw_variance += (
            params.kappa * (params.theta - variance) * dt
            + params.xi * np.sqrt(variance) * sqrt_dt * z_var
        )

        if store_paths:
            stock_paths[t + 1] = stock
            variance_paths[t + 1] = np.maximum(raw_variance, 0.0)

    if store_paths:
        return stock_paths, variance_paths

    return stock


def heston_call_mc(
    spot, strikes, rate, maturity, params,
    n_steps=252, n_paths=100_000, seed=42
):
    """Price calls across strikes using the same Heston terminal paths."""
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))

    terminal_stock = simulate_heston(
        spot=spot,
        rate=rate,
        maturity=maturity,
        params=params,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=np.random.default_rng(seed),
    )

    discounted_payoffs = np.exp(-rate * maturity) * np.maximum(
        terminal_stock[:, None] - strikes[None, :], 0.0
    )

    return mc_summary(discounted_payoffs)


def implied_volatility(price, spot, strike, rate, maturity):
    """Invert Black-Scholes call price; return NaN for invalid prices."""
    lower_bound = max(spot - strike * np.exp(-rate * maturity), 0.0)
    upper_bound = spot

    if (
        not np.isfinite(price)
        or price < lower_bound - 1e-8
        or price >= upper_bound
    ):
        return np.nan

    if price <= lower_bound + 1e-10:
        return 0.0

    objective = lambda vol: (
        bs_call_price(spot, strike, rate, vol, maturity) - price
    )

    try:
        return brentq(objective, 1e-8, 5.0, xtol=1e-12)
    except ValueError:
        return np.nan


def delta_hedge(
    stock_paths,
    initial_capital,
    strike,
    rate,
    maturity,
    hedge_vol,
    rebalance_every=1,
    variance_paths=None,
    use_current_vol=False,
):
    """
    Self-financing stock/cash hedge for a short European call.

    Positive terminal error = hedging portfolio exceeds the payoff.

    use_current_vol=True inserts sqrt(instantaneous variance)
    into the BS delta; this is NOT the Heston-model delta.
    """
    n_steps = stock_paths.shape[0] - 1
    n_paths = stock_paths.shape[1]
    dt = maturity / n_steps

    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1.")

    if use_current_vol and variance_paths is None:
        raise ValueError("variance_paths required for current-vol hedging.")

    initial_vol = (
        np.sqrt(variance_paths[0])
        if use_current_vol
        else hedge_vol
    )

    delta = np.broadcast_to(
        bs_call_delta(
            stock_paths[0], strike, rate, initial_vol, maturity
        ),
        (n_paths,),
    ).copy()

    cash = np.full(n_paths, initial_capital) - delta * stock_paths[0]
    growth = np.exp(rate * dt)

    for t in range(1, n_steps + 1):
        cash *= growth

        if t < n_steps and t % rebalance_every == 0:
            remaining = maturity - t * dt

            vol = (
                np.sqrt(variance_paths[t])
                if use_current_vol
                else hedge_vol
            )

            new_delta = bs_call_delta(
                stock_paths[t], strike, rate, vol, remaining
            )

            cash -= (new_delta - delta) * stock_paths[t]
            delta = new_delta

    terminal_portfolio = delta * stock_paths[-1] + cash
    payoff = np.maximum(stock_paths[-1] - strike, 0.0)

    return terminal_portfolio - payoff


def hedge_metrics(errors):
    errors = np.asarray(errors)
    return {
        "mean": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }
