"""Checks for sv.py. Run with:  python -m pytest -q"""
import numpy as np
import pytest
from scipy.integrate import quad

from sv import (BATES, HESTON, K, R, S0, SVModel, _cumulants, bs_call, bs_delta, bs_gamma,
                char_fn, cos_call, gbm_paths, hedge_pnl, histogram_bins, implied_variance,
                implied_vol, jump_quadrature, local_cov, min_var_hedge, paired_difference,
                simulate, vega_ratio)


def reference_call(S, strike, T, v, m):
    """Slow but independent Heston/Bates price and delta by direct numerical integration
    (Gil-Pelaez inversion), used to check the COS pricer."""
    x = np.log(S / strike)
    cf = lambda u: char_fn(np.array(u), T, R, v, m)[0]
    forward = np.real(cf(-1j))
    P1 = 0.5 + quad(lambda u: np.real(np.exp(1j * u * x) * cf(u - 1j) / (1j * u)) / forward,
                    0, np.inf, limit=5000)[0] / np.pi
    P2 = 0.5 + quad(lambda u: np.real(np.exp(1j * u * x) * cf(u + 0j) / (1j * u)),
                    0, np.inf, limit=5000)[0] / np.pi
    return S * P1 - strike * np.exp(-R * T) * P2, P1


def test_black_scholes_textbook_value():
    assert bs_call(100, 100, 1, 0.05, 0.2) == pytest.approx(10.4505835722, abs=1e-8)
    assert bs_delta(100, 100, 1, 0.05, 0.2) == pytest.approx(0.6368306512, abs=1e-8)


def test_implied_vol_round_trip():
    strikes = np.array([70, 90, 100, 110, 140])
    prices = bs_call(100, strikes, 0.5, 0.05, 0.25)
    np.testing.assert_allclose(implied_vol(prices, 100, strikes, 0.5, 0.05), 0.25, atol=1e-8)


# Includes the edge cases the hedging loop actually reaches: 1-day options and
# zero or near-zero variance (full-truncation Euler can put a path at v = 0).
@pytest.mark.parametrize("model", [HESTON, BATES], ids=["heston", "bates"])
@pytest.mark.parametrize("T", [1 / 252, 2 / 252, 0.1, 1.0, 1.5])
@pytest.mark.parametrize("v", [0.0, 1e-4, 0.002, 0.04, 0.15])
def test_cos_matches_numerical_integration(model, T, v):
    for S in (80.0, 100.0, 120.0):
        price, delta, _, _ = cos_call(S, 100, T, R, v, model)
        ref_price, ref_delta = reference_call(S, 100, T, v, model)
        assert price == pytest.approx(ref_price, abs=1e-5)
        assert delta == pytest.approx(ref_delta, abs=1e-5)


@pytest.mark.parametrize("model", [HESTON, BATES], ids=["heston", "bates"])
def test_no_arbitrage_bounds_on_a_wide_grid(model):
    S = np.array([50, 70, 85, 95, 100, 105, 115, 130, 160, 250.0])
    for T in (1 / 252, 0.02, 0.25, 1.25):
        for v in (0.0, 1e-4, 0.04, 0.5):
            price, delta, _, gamma = cos_call(S, K, T, R, v, model)
            intrinsic = np.maximum(S - K * np.exp(-R * T), 0)
            assert np.all(price >= intrinsic - 1e-6) and np.all(price <= S)
            assert np.all(delta >= -1e-6) and np.all(delta <= 1 + 1e-6)
            assert np.all(gamma >= -1e-6)


def test_cumulants_equal_integrated_variance_when_volvol_vanishes():
    # With xi -> 0 the variance path is deterministic, so Var(log S_T) = int_0^T v_t dt.
    # The old closed-form cumulants underestimated this several times over at 1 day.
    m = SVModel(v0=0.04, theta=0.04, kappa=2.0, xi=1e-4, rho=-0.7)
    for T in (1 / 252, 0.1, 1.0):
        for v in (0.0, 1e-4, 0.04):
            c2 = _cumulants(T, R, np.array([v]), m, 0)[1][0]
            exact = m.theta * T + (v - m.theta) * (1 - np.exp(-m.kappa * T)) / m.kappa
            assert c2 == pytest.approx(exact, rel=1e-4)


@pytest.mark.parametrize("v", [1e-4, 0.002, 0.04])
def test_gamma_near_expiry_matches_finite_differences(v):
    # Gamma is sharply peaked for a 1-day option at low variance; Richardson-extrapolated
    # central differences of the delta are needed for a fair reference.
    S, T = np.array([98.0, 99.5, 100.0, 100.5, 102.0]), 1 / 252
    def fd(h):
        return (cos_call(S + h, K, T, R, v, HESTON)[1] - cos_call(S - h, K, T, R, v, HESTON)[1]) / (2 * h)
    reference = (4 * fd(0.005) - fd(0.01)) / 3
    np.testing.assert_allclose(cos_call(S, K, T, R, v, HESTON)[3], reference, atol=1e-5)


def test_expired_option_is_rejected():
    with pytest.raises(ValueError):
        cos_call(100.0, K, 0.0, R, 0.04, HESTON)


@pytest.mark.parametrize("model", [HESTON, BATES], ids=["heston", "bates"])
def test_cos_greeks_match_finite_differences(model):
    S, v, T, h = 103.0, 0.05, 0.7, 1e-4
    _, delta, vega, gamma = cos_call(S, K, T, R, v, model)
    fd_delta = (cos_call(S + h, K, T, R, v, model)[0] - cos_call(S - h, K, T, R, v, model)[0]) / (2 * h)
    fd_vega = (cos_call(S, K, T, R, v + h, model)[0] - cos_call(S, K, T, R, v - h, model)[0]) / (2 * h)
    fd_gamma = (cos_call(S + h, K, T, R, v, model)[1] - cos_call(S - h, K, T, R, v, model)[1]) / (2 * h)
    assert delta == pytest.approx(fd_delta, abs=1e-5)
    assert vega == pytest.approx(fd_vega, rel=1e-4)
    assert gamma == pytest.approx(fd_gamma, rel=1e-4)


@pytest.mark.parametrize("model", [HESTON, BATES], ids=["heston", "bates"])
def test_shocked_prices_match_repricing(model):
    spots, shocks = np.array([90.0, 100.0, 115.0]), np.log([0.8, 0.95, 1.1])
    shocked = cos_call(spots, K, 0.5, R, 0.04, model, shocks=shocks)[4]
    for row, y in zip(shocked, shocks):
        np.testing.assert_allclose(row, cos_call(spots * np.exp(y), K, 0.5, R, 0.04, model)[0],
                                   atol=1e-10)


def test_gamma_formula_black_scholes():
    h = 1e-4
    fd = (bs_delta(100 + h, K, 1, R, 0.2) - bs_delta(100 - h, K, 1, R, 0.2)) / (2 * h)
    assert bs_gamma(100, K, 1, R, 0.2) == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize("xi", [0.0, 1e-8, 1e-5])
def test_vanishing_vol_of_vol_gives_black_scholes(xi):
    # v0 = theta, so as xi -> 0 Heston must reduce to BS with 20% vol: exactly at xi = 0
    # (the old xi^-2 form divided by zero there), and with an O(xi) skew correction otherwise.
    almost_bs = SVModel(v0=0.04, theta=0.04, kappa=2.0, xi=xi, rho=-0.7)
    for T in (1 / 252, 1.0):
        strikes = np.array([80.0, 100.0, 120.0])
        price, delta, _, gamma = cos_call(S0, strikes, T, R, 0.04, almost_bs)
        tol = 1e-10 + 3 * xi
        np.testing.assert_allclose(price, bs_call(S0, strikes, T, R, 0.2), atol=tol)
        np.testing.assert_allclose(delta, bs_delta(S0, strikes, T, R, 0.2), atol=tol)
        np.testing.assert_allclose(gamma, bs_gamma(S0, strikes, T, R, 0.2), atol=tol)


@pytest.mark.parametrize("model", [HESTON, BATES], ids=["heston", "bates"])
def test_monte_carlo_agrees_with_cos(model):
    ST = simulate(model, 100_000, seed=3, substeps=1, keep_paths=False)
    # risk-neutral forward
    assert abs(ST.mean() - S0 * np.exp(R)) < 4 * ST.std() / np.sqrt(len(ST))
    # ATM call price
    payoff = np.exp(-R) * np.maximum(ST - K, 0)
    assert abs(payoff.mean() - cos_call(S0, K, 1.0, R, model.v0, model)[0]) < 4 * payoff.std() / 316


def test_hedge_is_self_financing_without_risk():
    # Zero volatility: stock grows at r, the call is worth S - K e^{-rT}, holding one
    # share replicates it exactly, so the hedge error must be zero.
    Z = np.zeros((252, 3))
    S = gbm_paths(0.0, Z)
    premium = S0 - K * np.exp(-R)
    pnl, _ = hedge_pnl(S, premium, stock_pos=np.ones((252, 3)), every=5)
    np.testing.assert_allclose(pnl, 0, atol=1e-10)


def test_daily_bs_hedge_beats_monthly_in_bs_world():
    S = gbm_paths(0.2, np.random.default_rng(0).standard_normal((252, 2000)))
    tau = 1 - np.arange(252) / 252
    delta = bs_delta(S[:-1], K, tau[:, None], R, 0.2)
    premium = bs_call(S0, K, 1.0, R, 0.2)
    daily = hedge_pnl(S, premium, delta, every=1)[0]
    monthly = hedge_pnl(S, premium, delta, every=21)[0]
    assert np.sqrt(np.mean(daily**2)) < 0.5 * np.sqrt(np.mean(monthly**2))
    assert abs(daily.mean()) < 0.05


def test_unwind_costs_are_charged_at_expiry():
    # Zero volatility, hold one share throughout: the only costs are buying it at
    # t = 0 and selling it at T, so total cost = c * (S0 * e^{rT} + S_T).
    S = gbm_paths(0.0, np.zeros((252, 2)))
    premium = S0 - K * np.exp(-R)
    pnl, costs = hedge_pnl(S, premium, np.ones((252, 2)), cost_stock=0.001)
    expected = 0.001 * (S0 * np.exp(R) + S[-1])
    np.testing.assert_allclose(costs, expected, rtol=1e-10)
    np.testing.assert_allclose(pnl, -expected, atol=1e-10)


def test_recorded_jumps_match_poisson_intensity():
    _, _, _, jumps = simulate(BATES, 4000, seed=5, n_days=50, substeps=2)
    expected = BATES.lam * 1.0        # the 50 days still span the 1-year horizon
    total = jumps.sum(axis=0)
    assert abs(total.mean() - expected) < 4 * np.sqrt(expected / 4000)
    _, _, _, none = simulate(HESTON, 100, seed=5, n_days=10)
    assert none.sum() == 0


@pytest.mark.parametrize("model", [HESTON, BATES], ids=["heston", "bates"])
def test_implied_variance_round_trip(model):
    # Includes starting points far from the answer and high variances, where clipped
    # Newton used to bounce between the bounds (seen at v ~ 0.13 from a 0.04 start).
    S = np.array([60.0, 80.0, 100.0, 120.0, 75.5, 100.0])
    v_true = np.array([0.01, 0.04, 0.09, 0.2, 0.132, 0.6])
    for T in (0.25, 1.0):
        target, _, true_vega, _ = cos_call(S, K, T, R, v_true, model)
        identified = true_vega > 0.1   # deep OTM the price does not pin down v
        assert identified.sum() >= 5
        for start in (0.0, 0.04, 1.0):
            v, price, _, _, residual = implied_variance(target, S, K, T, R, model, start)
            np.testing.assert_allclose(v[identified], v_true[identified], atol=1e-7)
            assert np.abs(residual).max() < 1e-8


def test_implied_variance_reports_when_no_variance_fits():
    # A price below the model price at v = 0 cannot be matched: v = 0, positive residual
    target = cos_call(100.0, K, 0.5, R, 0.0, HESTON)[0] - 0.05
    v, _, _, _, residual = implied_variance(np.array([target]), 100.0, K, 0.5, R, HESTON, 0.04)
    assert v[0] == 0.0 and residual[0] == pytest.approx(0.05, abs=1e-8)


def test_vega_ratio_guards_noise():
    vega = np.array([10.0, 2e-4, -1e-4, 5.0])
    hedge = np.array([20.0, -8e-5, 1e-4, 0.0])
    np.testing.assert_allclose(vega_ratio(vega, hedge), [0.5, 0.0, 0.0, 0.0])


def test_paired_difference_and_bins():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(500)
    b = 2 * a
    boot = rng.integers(0, 500, (200, 500))
    diff, lo, hi = paired_difference(a, b, boot)
    assert diff < 0 and lo <= diff <= hi < 0
    bins = histogram_bins([a, b, np.array([-50.0, 40.0])])
    assert bins[0] == -50.0 and bins[-1] == 40.0


def _call_and_hedge(model, S, v, jump_sizes):
    """(delta, vega, value change per jump size) of the sold call (T = 0.5) and hedge option (T = 0.75)."""
    c, c_delta, c_vega, _, c_jumped = cos_call(S, K, 0.5, R, v, model, shocks=jump_sizes)
    h, h_delta, h_vega, _, h_jumped = cos_call(S, K, 0.75, R, v, model, shocks=jump_sizes)
    return (c_delta, c_vega, c_jumped - c), (h_delta, h_vega, h_jumped - h)


def test_min_var_hedge_without_jumps_is_min_var_delta_and_delta_vega():
    # With lam = 0 the market is driven by two Brownian motions only, so the stock-only
    # min-var hedge is delta + rho xi vega / S, and stock + option replicates the call:
    # the delta-vega hedge.
    S, v = np.array([85.0, 100.0, 115.0]), np.array([0.02, 0.04, 0.09])
    jump_sizes, jump_w = jump_quadrature(HESTON)
    call, hedge = _call_and_hedge(HESTON, S, v, jump_sizes)
    stock_only, a, q = min_var_hedge(S, v, HESTON, call, hedge, jump_sizes, jump_w)
    np.testing.assert_allclose(stock_only, call[0] + HESTON.rho * HESTON.xi * call[1] / S,
                               atol=1e-10)
    np.testing.assert_allclose(q, call[1] / hedge[1], atol=1e-10)
    np.testing.assert_allclose(a, call[0] - q * hedge[0], atol=1e-10)


def test_min_var_hedge_with_jumps_minimises_local_variance():
    # With jumps the hedge is no longer delta-vega neutral, but no nearby position
    # may give a smaller local variance of (call - a * stock - q * option).
    S, v = np.array([90.0, 100.0, 110.0]), np.array([0.03, 0.032, 0.05])
    jump_sizes, jump_w = jump_quadrature(BATES)
    call, hedge = _call_and_hedge(BATES, S, v, jump_sizes)
    stock = (1.0, 0.0, S * (np.exp(jump_sizes)[:, None] - 1))
    stock_only, a, q = min_var_hedge(S, v, BATES, call, hedge, jump_sizes, jump_w)

    def cov(x, y):
        return local_cov(x, y, S, v, BATES, jump_w)

    def residual_variance(a, q):
        return (cov(call, call) + a**2 * cov(stock, stock) + q**2 * cov(hedge, hedge)
                - 2 * a * cov(stock, call) - 2 * q * cov(hedge, call) + 2 * a * q * cov(stock, hedge))

    best = residual_variance(a, q)
    for da, dq in [(1e-3, 0), (-1e-3, 0), (0, 1e-2), (0, -1e-2), (1e-3, -1e-2)]:
        assert np.all(residual_variance(a + da, q + dq) > best)
    for da in (1e-3, -1e-3):
        assert np.all(residual_variance(stock_only + da, 0) > residual_variance(stock_only, 0))
    # jumps make the option hedge differ from plain delta-vega neutrality (at and above
    # the money by about 0.1 options here; at S = 90 the two happen to nearly coincide)
    assert np.abs(q - call[1] / hedge[1]).max() > 0.05
