"""
Model code used by all three scripts.

  - Black-Scholes formulas and implied volatility
  - Heston / Bates characteristic function and a COS pricer (price, delta, vega, gamma)
  - Heston calibration to an implied-vol surface
  - Monte Carlo simulation of Heston / Bates paths
  - P&L of a self-financing hedge of a short call
  - Jump-aware minimum-variance hedge ratios (the Part 3 oracle)

Bates = Heston + lognormal jumps in the stock. Setting lam=0 gives plain Heston,
so one class covers both models.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm


# Contract and market used everywhere
S0, K, T, R = 100.0, 100.0, 1.0, 0.05
N_DAYS = 252


@dataclass
class SVModel:
    v0: float         # initial variance
    theta: float      # long-run variance
    kappa: float      # mean-reversion speed
    xi: float         # vol of vol
    rho: float        # correlation stock / variance shocks
    lam: float = 0.0  # jump intensity (jumps per year)
    mu_j: float = 0.0  # mean of log jump size
    sig_j: float = 0.0  # std of log jump size

    def jump_mean(self):
        """E[e^J] - 1, the average relative jump size."""
        return np.exp(self.mu_j + 0.5 * self.sig_j**2) - 1


# The two worlds used in the project. Bates has the same expected annual variance of log-returns as Heston,
# 0.032 + lam * E[J^2] = 0.032 + 0.4 * (0.1^2 + 0.1^2) = 0.04. (Its ATM implied vol is
# not exactly 20%: jumps and the different variance path change the smile.)
HESTON = SVModel(v0=0.04, theta=0.04, kappa=2.0, xi=0.3, rho=-0.7)
BATES = SVModel(v0=0.032, theta=0.032, kappa=2.0, xi=0.3, rho=-0.7,
                lam=0.4, mu_j=-0.10, sig_j=0.10)


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    return norm.cdf(_d1(S, K, T, r, sigma))


def bs_vega(S, K, T, r, sigma):
    return S * norm.pdf(_d1(S, K, T, r, sigma)) * np.sqrt(T)


def bs_gamma(S, K, T, r, sigma):
    return norm.pdf(_d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))


def implied_vol(price, S, K, T, r, n_iter=60):
    """Vectorised bisection. Returns NaN where the price is outside the no-arbitrage bounds."""
    price, S, K = np.broadcast_arrays(*(np.asarray(a, float) for a in (price, S, K)))
    lo = np.full(price.shape, 1e-4)
    hi = np.full(price.shape, 4.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_high = bs_call(S, K, T, r, mid) > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    iv = 0.5 * (lo + hi)
    intrinsic = np.maximum(S - K * np.exp(-r * T), 0)
    return np.where((price > intrinsic) & (price < S), iv, np.nan)


# ---------------------------------------------------------------------------
# Heston / Bates pricing with the COS method (Fang & Oosterlee, 2008)
# ---------------------------------------------------------------------------
def char_fn(u, T, r, v, m, n_jumps=None):
    """Characteristic function of log(S_T / S_t) given current variance v.

    Uses the "little Heston trap" form (Albrecher et al., 2007), which is stable
    for long maturities. Also returns D, because d(phi)/dv = D * phi (needed for vega).
    If n_jumps is given, the CF is conditional on exactly that many jumps by T;
    otherwise the jumps are the usual compound Poisson process.

    Written without dividing by xi^2 (using b - d = -xi^2 (iu + u^2) / (b + d)), so it
    stays accurate as xi -> 0, where Heston becomes Black-Scholes with deterministic variance.
    Requires kappa > 0.
    """
    iu = 1j * u
    q = iu + u**2
    b = m.kappa - m.rho * m.xi * iu
    d = np.sqrt(b**2 + m.xi**2 * q)
    beta = -q / (b + d)                      # (b - d) / xi^2
    g = m.xi**2 * beta / (b + d)             # (b - d) / (b + d)
    e = np.exp(-d * T)
    D = beta * (1 - e) / (1 - g * e)
    w = beta / (b + d) * (1 - e) / (1 - g)   # log((1 - g e) / (1 - g)) = log(1 + xi^2 w)
    C = m.kappa * m.theta * (beta * T - 2 * w * _log1p_ratio(m.xi**2 * w))
    compensator = -iu * m.lam * m.jump_mean() * T
    if n_jumps is None:
        jumps = m.lam * T * (np.exp(iu * m.mu_j - 0.5 * m.sig_j**2 * u**2) - 1) + compensator
    else:
        jumps = n_jumps * (iu * m.mu_j - 0.5 * m.sig_j**2 * u**2) + compensator
    return np.exp(iu * r * T + C + D * v + jumps), D


def _log1p_ratio(z):
    """log(1 + z) / z for complex z, accurate also for tiny z (numpy's log1p is not)."""
    small = np.abs(z) < 1e-4
    z_safe = np.where(small, 1.0, z)
    return np.where(small, 1 - z / 2 + z**2 / 3 - z**3 / 4, np.log1p(z_safe) / z_safe)


def _cumulants(T, r, v, m, n_jumps, h=1e-2):
    """Mean and variance of log(S_T/S_t) given n_jumps, used for the COS integration range.

    Taken from the characteristic function itself by central differences of log(phi):
    log phi(u) = i c1 u - c2 u^2 / 2 + O(u^3). The closed-form Heston cumulants suffer
    from cancellation at short maturities (they underestimate c2 several times over at
    1 day and low variance), which makes the COS range too narrow.
    """
    log_phi = np.log(char_fn(np.array([h, -h]), T, r, v[:, None], m, n_jumps)[0])
    c1 = (log_phi[:, 0] - log_phi[:, 1]).imag / (2 * h)
    c2 = -(log_phi[:, 0] + log_phi[:, 1]).real / h**2
    return c1, c2


def cos_call(S, K, T, r, v, m, shocks=None, N=128, L=12):
    """European call price, delta (dC/dS), variance vega (dC/dv) and gamma under Heston / Bates.

    S, K and v can be arrays (e.g. one value per simulated path); T is a scalar.
    If `shocks` (log-returns y_1..y_n) is given, also returns the call prices after an
    instant move S -> S * exp(y_k) with v unchanged, shape (n, ...). The characteristic
    function does not depend on S, so these scenario prices are almost free.

    For Bates we price conditional on the number of jumps n and average with
    Poisson weights. The no-jump term is very narrow at short maturities while
    the jump terms are wide, so giving each its own COS range is much more
    accurate than one range for the whole mixture. The Poisson sum is cut at
    1e-10 of probability: a 1e-6 cut leaves price errors of order 1e-5, because
    the dropped many-jump terms are priced very differently from the rest.
    """
    if T <= 0:
        raise ValueError("cos_call needs T > 0; use the payoff at expiry")
    shocks = np.zeros(0) if shocks is None else np.asarray(shocks, float)
    if m.lam == 0:
        result = _cos_call(S, K, T, r, v, m, 0, N, L, shocks)
    else:
        mean_jumps = m.lam * T
        result = None
        n = 0
        weight = np.exp(-mean_jumps)
        cumulative = 0.0
        while cumulative < 1 - 1e-10:
            terms = _cos_call(S, K, T, r, v, m, n, N if n == 0 else 64, L, shocks)
            result = [weight * x for x in terms] if result is None else \
                [t + weight * x for t, x in zip(result, terms)]
            cumulative += weight
            n += 1
            weight *= mean_jumps / n
    return tuple(result) if len(shocks) else tuple(result[:4])


def _cos_call(S, K, T, r, v, m, n_jumps, N, L, shocks):
    """COS price, delta, vega, gamma and shocked prices given exactly n_jumps jumps.

    The put is priced with COS (more stable than pricing the call directly)
    and the call follows from put-call parity (gamma and vega are the same for both).
    """
    S, K, v = (np.asarray(a, float) for a in np.broadcast_arrays(S, K, v))
    shape = S.shape
    S, K, v = S.ravel(), K.ravel(), v.ravel()

    c1, c2 = _cumulants(T, r, v, m, n_jumps)
    a = (c1 - L * np.sqrt(c2))[:, None]
    b = (c1 + L * np.sqrt(c2))[:, None]

    k = np.arange(N)
    u = k * np.pi / (b - a)                      # shape (n_points, N)
    phi, D = char_fn(u, T, r, v[:, None], m, n_jumps)

    weight = np.where(k == 0, 0.5, 1.0) * 2 / (b - a)
    rotation = np.exp(-1j * u * a)
    terms = weight * np.real(phi * rotation)
    vega_terms = weight * np.real(D * phi * rotation)
    u_safe = np.where(k == 0, 1.0, u)
    disc = np.exp(-r * T)

    exp_a = np.exp(a)
    one_plus_u2 = 1 + u**2

    def put_coefficients(x):
        """Cosine coefficients of the put payoff K * (1 - e^{x+z})^+, non-zero for z < -x.

        Returns the payoff coefficients, chi (for delta), cos(w) (for gamma) and whether the
        kink lies inside the range. Each transcendental is evaluated once: this function
        dominates the run time of Part 3.
        """
        d = np.clip(-x[:, None], a, b)
        w = u * (d - a)
        cos_w, sin_w, exp_d = np.cos(w), np.sin(w), np.exp(d)
        chi = (exp_d * (cos_w + u * sin_w) - exp_a) / one_plus_u2
        psi = np.where(k == 0, d - a, sin_w / u_safe)
        return psi - np.exp(x)[:, None] * chi, chi, cos_w, (d > a) & (d < b)

    x = np.log(S / K)
    payoff, chi, cos_w, inside = put_coefficients(x)
    put = disc * K * np.sum(terms * payoff, axis=1)
    delta = 1 - disc * np.sum(terms * chi, axis=1)
    vega = disc * K * np.sum(vega_terms * payoff, axis=1)
    gamma = disc * K * np.sum(terms * cos_w * inside, axis=1) / S**2
    call = put + S - K * disc

    shocked = np.empty((len(shocks), S.size))
    for i, y in enumerate(shocks):
        put_y = disc * K * np.sum(terms * put_coefficients(x + y)[0], axis=1)
        shocked[i] = put_y + S * np.exp(y) - K * disc

    return [call.reshape(shape), delta.reshape(shape), vega.reshape(shape),
            gamma.reshape(shape), shocked.reshape((len(shocks),) + shape)]


def implied_variance(target, S, K, T, r, m, v_start, v_max=1.0, tol=1e-9, max_iter=60):
    """Spot variance v in [0, v_max] at which model m reprices a call to `target`.

    The model's version of an implied vol. The call price increases with v, so the root is
    bracketed: Newton steps are taken when they stay inside the current bracket and
    bisection is used otherwise (plain clipped Newton can oscillate between the bounds).
    If even v = 0 gives a price above the target, no v >= 0 fits: v = 0 is returned and
    the mismatch shows up in the residual. Likewise v = v_max if the target is above the
    price at v_max.

    Returns v, the model price, delta and vega at v, and the residual (price - target).
    Where the option has almost no vega (deep out of the money), the price barely depends
    on v, so v is not identified even though the residual is tiny; callers should check
    the vega.
    """
    target = np.asarray(target, float)
    n = target.size
    S = np.broadcast_to(np.asarray(S, float), target.shape)
    lo, hi = np.zeros(n), np.full(n, v_max)
    p_lo = cos_call(S, K, T, r, lo, m)[0]
    p_hi = cos_call(S, K, T, r, hi, m)[0]
    below, above = target <= p_lo, target >= p_hi
    v = np.clip(np.asarray(v_start, float) * np.ones(n), 0.0, v_max)
    v = np.where(below, 0.0, np.where(above, v_max, v))
    done = below | above
    for _ in range(max_iter):
        price, delta, vega, _ = cos_call(S, K, T, r, v, m)
        err = price - target
        done |= np.abs(err) < tol
        if done.all():
            break
        hi = np.where(~done & (err > 0), v, hi)
        lo = np.where(~done & (err < 0), v, lo)
        newton = v - err / np.where(vega > 0, vega, np.inf)
        inside = (vega > 0) & (newton > lo) & (newton < hi)
        v = np.where(done, v, np.where(inside, newton, 0.5 * (lo + hi)))
    price, delta, vega, _ = cos_call(S, K, T, r, v, m)
    return v, price, delta, vega, price - target


def model_ivs(m, S, strikes, maturities, r):
    """Black-Scholes implied vols of model m's call prices, one array per maturity."""
    return {T_: implied_vol(cos_call(S, strikes, T_, r, m.v0, m)[0], S, strikes, T_, r)
            for T_ in maturities}


# Starting point and box for Heston calibration (v0, theta, kappa, xi, rho)
CAL_START = [0.04, 0.04, 2.0, 0.3, -0.7]
CAL_LOWER = [0.001, 0.001, 0.1, 0.05, -0.99]
CAL_UPPER = [0.5, 0.5, 10.0, 2.0, 0.5]


def calibrate_heston(market_iv, S, strikes, r, start=CAL_START):
    """Least-squares fit of Heston's five parameters to a surface of implied vols.

    `market_iv` maps maturity -> implied vols at `strikes`. The objective is the
    difference in implied vol (not price), so every option counts about equally.
    Returns the fitted SVModel and the residuals in vol units.
    """
    maturities = list(market_iv)

    def iv_errors(params):
        model = model_ivs(SVModel(*params), S, strikes, maturities, r)
        return np.concatenate([model[T_] - market_iv[T_] for T_ in maturities])

    fit = least_squares(iv_errors, start, bounds=(CAL_LOWER, CAL_UPPER))
    return SVModel(*fit.x), fit.fun


def vega_ratio(vega, hedge_vega, min_vega=1e-3):
    """Number of hedge options that make the book vega-neutral.

    Deep out of the money both vegas are ~1e-4 and the COS values can even be slightly
    negative, so their ratio is noise. Where the hedge option has (almost) no vega there
    is no vega left to hedge either, and no option is held.
    """
    ok = hedge_vega > min_vega
    return np.where(ok, np.maximum(vega, 0.0) / np.where(ok, hedge_vega, 1.0), 0.0)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def simulate(m, n_paths, seed, n_days=N_DAYS, substeps=4, keep_paths=True):
    """Risk-neutral Heston / Bates paths on a daily grid.

    Full-truncation Euler for the variance, log-Euler for the stock, and
    `substeps` steps per day to keep discretisation error small.
    Returns daily stock S, variance V, the daily stock shocks Z (so the same
    randomness can drive a Black-Scholes world for a like-for-like comparison)
    and the number of jumps on each day (all zero for Heston).
    If keep_paths=False, only the terminal stock prices are returned.
    """
    rng = np.random.default_rng(seed)
    dt = T / (n_days * substeps)
    log_s = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, m.v0)
    if keep_paths:
        S = np.empty((n_days + 1, n_paths))
        V = np.empty((n_days + 1, n_paths))
        Z = np.zeros((n_days, n_paths))
        jumps = np.zeros((n_days, n_paths), dtype=int)
        S[0], V[0] = S0, m.v0

    for day in range(n_days):
        for _ in range(substeps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            vp = np.maximum(v, 0)
            log_s += (R - m.lam * m.jump_mean() - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
            if m.lam > 0:
                n_jumps = rng.poisson(m.lam * dt, n_paths)
                log_s += n_jumps * m.mu_j + np.sqrt(n_jumps) * m.sig_j * rng.standard_normal(n_paths)
                if keep_paths:
                    jumps[day] += n_jumps
            v += (m.kappa * (m.theta - vp) * dt
                  + m.xi * np.sqrt(vp * dt) * (m.rho * z1 + np.sqrt(1 - m.rho**2) * z2))
            if keep_paths:
                Z[day] += z1
        if keep_paths:
            S[day + 1] = np.exp(log_s)
            V[day + 1] = np.maximum(v, 0)

    if not keep_paths:
        return np.exp(log_s)
    return S, V, Z / np.sqrt(substeps), jumps


def gbm_paths(sigma, Z):
    """Exact Black-Scholes paths driven by given daily standard-normal shocks Z."""
    dt = T / Z.shape[0]
    log_returns = (R - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_s = np.vstack([np.zeros(Z.shape[1]), np.cumsum(log_returns, axis=0)])
    return S0 * np.exp(log_s)


# ---------------------------------------------------------------------------
# Hedging
# ---------------------------------------------------------------------------
def hedge_pnl(S, premium, stock_pos, opt_pos=None, opt_price=None, opt_value_T=None,
              every=1, cost_stock=0.0, cost_opt=0.0):
    """Terminal P&L of selling one call for `premium` and hedging it.

    stock_pos, opt_pos, opt_price have shape (n_days, n_paths) and hold the target
    holdings (and hedge-option prices) at the start of each day. The book is only
    rebalanced every `every` days and is unwound at expiry. Cash earns the risk-free rate.
    Costs are proportional: cost_stock * traded stock value, cost_opt * traded option value,
    charged on every trade including the final unwind.

    Returns (pnl, costs), both per path and valued at expiry.
    pnl = hedge portfolio - call payoff, so negative means the hedge lost money.
    """
    n_days, n_paths = stock_pos.shape
    growth = np.exp(R * T / n_days)
    cash = np.full(n_paths, float(premium))
    costs = np.zeros(n_paths)
    stock = np.zeros(n_paths)
    option = np.zeros(n_paths)

    for day in range(n_days):
        if day % every == 0:
            trade = stock_pos[day] - stock
            fee = cost_stock * np.abs(trade) * S[day]
            cash -= trade * S[day]
            stock = stock_pos[day]
            if opt_pos is not None:
                trade = opt_pos[day] - option
                fee += cost_opt * np.abs(trade) * opt_price[day]
                cash -= trade * opt_price[day]
                option = opt_pos[day]
            cash -= fee
            costs += fee
        cash *= growth
        costs *= growth

    value = cash + stock * S[-1]
    unwind = cost_stock * np.abs(stock) * S[-1]
    if opt_pos is not None:
        value += option * opt_value_T
        unwind += cost_opt * np.abs(option) * opt_value_T
    return value - unwind - np.maximum(S[-1] - K, 0), costs + unwind


def jump_quadrature(m, n=7):
    """Gauss-Hermite nodes (log jump sizes) and weights for expectations over J ~ N(mu_j, sig_j^2)."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    return m.mu_j + m.sig_j * nodes, weights / weights.sum()


def local_cov(x, y, S, v, m, jump_w):
    """Covariance per unit time of the value changes of two instruments under Heston / Bates.

    Each instrument is (delta, variance vega, value change for each jump size). The
    diffusion part follows from Ito's lemma; the jump part is lam * E[dX dY | jump].
    """
    dx, vx, jx = x
    dy, vy, jy = y
    diffusion = v * (S**2 * dx * dy + m.rho * m.xi * S * (dx * vy + vx * dy) + m.xi**2 * vx * vy)
    jumps = m.lam * np.sum(jump_w[:, None] * jx * jy, axis=0)
    return diffusion + jumps


def min_var_hedge(S, v, m, call, hedge, jump_sizes, jump_w, min_vega=1e-3):
    """Positions that minimise the local variance of (call - a * stock - q * option).

    `call` and `hedge` are (delta, variance vega, value change for each jump size), each
    per path. Returns the stock-only position, and the stock and option positions of the
    stock + option hedge. Under the pricing measure the hedge error is a martingale, so
    minimising each day's variance minimises the variance at expiry.

    Without jumps this reduces to the min-var delta (delta + rho xi vega / S) and to the
    delta-vega hedge. Where the hedge option has no usable vega the 2x2 system is
    near-singular and its solution is numerical noise; there stock only is used.
    """
    stock = (1.0, 0.0, S * (np.exp(jump_sizes)[:, None] - 1))
    ss, hh = local_cov(stock, stock, S, v, m, jump_w), local_cov(hedge, hedge, S, v, m, jump_w)
    sh = local_cov(stock, hedge, S, v, m, jump_w)
    sc, hc = local_cov(stock, call, S, v, m, jump_w), local_cov(hedge, call, S, v, m, jump_w)
    stock_only = sc / ss
    det = ss * hh - sh**2
    usable = (hedge[1] > min_vega) & (det > 1e-10 * ss * hh)
    safe_det = np.where(usable, det, 1.0)
    a = np.where(usable, (sc * hh - sh * hc) / safe_det, stock_only)
    q = np.where(usable, (ss * hc - sh * sc) / safe_det, 0.0)
    return stock_only, a, q


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def summary(pnl):
    """RMSE, mean, standard deviation and 5% expected shortfall (mean of the worst 5%, as a loss)."""
    worst = np.sort(pnl)[: max(1, int(0.05 * len(pnl)))]
    return {"rmse": rmse(pnl), "mean": float(np.mean(pnl)), "std": float(np.std(pnl)),
            "es5": float(-worst.mean())}


def bootstrap_indices(n_paths, n_boot=1000, seed=0):
    return np.random.default_rng(seed).integers(0, n_paths, (n_boot, n_paths))


def rmse_interval(pnl, boot):
    """95% percentile-bootstrap interval for the RMSE."""
    return tuple(np.percentile([rmse(pnl[b]) for b in boot], [2.5, 97.5]))


def paired_difference(a, b, boot):
    """RMSE(a) - RMSE(b) on the same paths, with a 95% paired-bootstrap interval.

    Resampling the same paths for both hedges keeps their correlation, so the interval
    is much tighter than comparing two separate intervals.
    """
    diffs = [rmse(a[i]) - rmse(b[i]) for i in boot]
    return rmse(a) - rmse(b), *np.percentile(diffs, [2.5, 97.5])


class _Tee:
    """Writes to the console and to a file at the same time."""

    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


def log_output(path):
    """From now on, copy everything printed to the console into `path` as well.

    Calling it again (e.g. rerunning a cell in an IDE) starts a fresh file instead of
    writing to two files at once.
    """
    if isinstance(sys.stdout, _Tee):
        sys.stdout.file.close()
        sys.stdout = sys.stdout.console
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, open(path, "w", encoding="utf-8"))


def histogram_bins(series, n=90):
    """Bins spanning every value of every series, so no outcome is silently dropped."""
    lo = min(float(np.min(s)) for s in series)
    hi = max(float(np.max(s)) for s in series)
    return np.linspace(lo, hi, n)
