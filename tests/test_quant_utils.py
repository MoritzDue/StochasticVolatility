import numpy as np

from quant_utils import (
    HestonParams,
    bs_call_delta,
    bs_call_price,
    delta_hedge,
    hedge_metrics,
    implied_volatility,
    mc_summary,
    simulate_gbm_paths,
    simulate_heston,
)


def test_black_scholes_price_and_delta():
    price = bs_call_price(100, 100, 0.05, 0.20, 1.0)
    delta = bs_call_delta(100, 100, 0.05, 0.20, 1.0)

    assert np.isclose(price, 10.4505835722, atol=1e-8)
    assert np.isclose(delta, 0.6368306512, atol=1e-8)


def test_implied_volatility_round_trip():
    for strike in (80, 100, 120):
        price = bs_call_price(100, strike, 0.05, 0.20, 1.0)
        recovered_vol = implied_volatility(
            price, 100, strike, 0.05, 1.0
        )

        assert np.isclose(recovered_vol, 0.20, atol=1e-8)


def test_mc_summary():
    observations = np.array([1.0, 2.0, 3.0, 4.0])

    estimate, se, lower, upper = mc_summary(observations)

    assert np.isclose(estimate, 2.5)
    assert np.isclose(
        se,
        observations.std(ddof=1) / np.sqrt(4),
    )
    assert lower < estimate < upper


def test_heston_zero_vol_of_vol_matches_bs_price():
    params = HestonParams(
        v0=0.04,
        theta=0.04,
        kappa=2.0,
        xi=0.0,
        rho=-0.70,
    )

    terminal = simulate_heston(
        spot=100,
        rate=0.05,
        maturity=1.0,
        params=params,
        n_steps=64,
        n_paths=40_000,
        rng=np.random.default_rng(123),
    )

    payoffs = np.exp(-0.05) * np.maximum(
        terminal - 100, 0.0
    )

    estimate, se, _, _ = mc_summary(payoffs)
    analytical = bs_call_price(100, 100, 0.05, 0.20, 1.0)

    assert abs(estimate - analytical) < 4 * se


def test_heston_risk_neutral_stock_expectation():
    terminal = simulate_heston(
        spot=100,
        rate=0.05,
        maturity=1.0,
        params=HestonParams(),
        n_steps=64,
        n_paths=20_000,
        rng=np.random.default_rng(7),
    )

    estimate = terminal.mean()
    se = terminal.std(ddof=1) / np.sqrt(len(terminal))
    expected = 100 * np.exp(0.05)

    assert abs(estimate - expected) < 4 * se


def test_daily_bs_hedging_improves_on_monthly():
    paths = simulate_gbm_paths(
        spot=100,
        rate=0.05,
        vol=0.20,
        maturity=1.0,
        n_steps=252,
        n_paths=3_000,
        rng=np.random.default_rng(42),
    )

    initial_capital = bs_call_price(
        100, 100, 0.05, 0.20, 1.0
    )

    daily_errors = delta_hedge(
        paths, initial_capital, 100, 0.05, 1.0, 0.20,
        rebalance_every=1,
    )

    monthly_errors = delta_hedge(
        paths, initial_capital, 100, 0.05, 1.0, 0.20,
        rebalance_every=21,
    )

    assert (
        hedge_metrics(daily_errors)["rmse"]
        < hedge_metrics(monthly_errors)["rmse"]
    )


def test_deterministic_hedge_is_self_financing():
    times = np.linspace(0.0, 1.0, 13)

    stock_paths = np.repeat(
        (100 * np.exp(0.05 * times))[:, None],
        3,
        axis=1,
    )

    initial_capital = bs_call_price(
        100, 90, 0.05, 0.0, 1.0
    )

    errors = delta_hedge(
        stock_paths=stock_paths,
        initial_capital=initial_capital,
        strike=90,
        rate=0.05,
        maturity=1.0,
        hedge_vol=0.0,
        rebalance_every=3,
    )

    assert np.max(np.abs(errors)) < 1e-10
