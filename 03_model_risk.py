# %% [markdown]
# # Part 3 - Model risk in a Bates market

# %%
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from sv import (BATES, K, N_DAYS, R, S0, T, SVModel, bootstrap_indices, bs_delta, cos_call,
                hedge_pnl, histogram_bins, implied_variance, implied_vol, jump_quadrature,
                log_output, min_var_hedge, paired_difference, rmse, rmse_interval, simulate,
                summary, vega_ratio)

parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=int, default=4000)
parser.add_argument("--seed", type=int, default=7)
args, _ = parser.parse_known_args()

Path("figures").mkdir(exist_ok=True)
N_PATHS = args.paths
T_HEDGE = 1.25
MIN_VEGA = 1e-3

default_run = (args.paths, args.seed) == (4000, 7)
log_output("results/03_model_risk.txt" if default_run else
           f"results/03_model_risk_paths{args.paths}_seed{args.seed}.txt")

jump_sizes, jump_w = jump_quadrature(BATES)

# %% 1. Create market option prices
strikes = np.arange(80.0, 125.0, 5.0)
maturities = [0.25, 0.5, 1.0, 1.5]
market_iv = {T_: implied_vol(cos_call(S0, strikes, T_, R, BATES.v0, BATES)[0], S0, strikes, T_, R)
             for T_ in maturities}

# %% 2. Calibrate the Heston model
def heston_ivs(params):
    m = SVModel(*params)
    return {T_: implied_vol(cos_call(S0, strikes, T_, R, m.v0, m)[0], S0, strikes, T_, R)
            for T_ in maturities}


def iv_errors(params):
    model = heston_ivs(params)
    return np.concatenate([model[T_] - market_iv[T_] for T_ in maturities])


start = [0.04, 0.04, 2.0, 0.3, -0.7]
lower = [0.001, 0.001, 0.1, 0.05, -0.99]
upper = [0.5, 0.5, 10.0, 2.0, 0.5]
fit = least_squares(iv_errors, start, bounds=(lower, upper))
heston_cal = SVModel(*fit.x)
fitted_iv = heston_ivs(fit.x)

print("Calibrated Heston:  v0={:.4f} theta={:.4f} kappa={:.3f} xi={:.3f} rho={:.3f}".format(*fit.x))
print(f"True Bates (diffusion part):  v0={BATES.v0:.4f} theta={BATES.theta:.4f} "
      f"kappa={BATES.kappa:.3f} xi={BATES.xi:.3f} rho={BATES.rho:.3f}, "
      f"plus jumps lam={BATES.lam} mu={BATES.mu_j:.2f} sig={BATES.sig_j:.2f}")
print(f"Feller condition 2*kappa*theta > xi^2: "
      f"{2 * heston_cal.kappa * heston_cal.theta:.3f} vs {heston_cal.xi ** 2:.3f}")
print("Fit error by maturity (RMSE in vol points):")
for T_ in maturities:
    print(f"  T = {T_:<5} {100 * rmse(fitted_iv[T_] - market_iv[T_]):.3f}")
print(f"  all      {100 * rmse(iv_errors(fit.x)):.3f}")

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, T_ in zip(axes, maturities):
    ax.plot(strikes, 100 * market_iv[T_], "o", label="market (Bates)")
    ax.plot(strikes, 100 * fitted_iv[T_], "-", label="calibrated Heston")
    ax.set_title(f"T = {T_}")
    ax.set_xlabel("Strike")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Implied vol (%)")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/03_calibration.png", dpi=150)

# %% 3. Simulate paths and calculate hedges
S, V, _, n_jumps = simulate(BATES, N_PATHS, seed=args.seed)
days = np.arange(N_DAYS)
tau = T - days / N_DAYS
tau_h = T_HEDGE - days / N_DAYS
shape = (N_DAYS, N_PATHS)

keys = ["c_price", "c_delta", "c_vega", "h_price", "h_delta", "h_vega",
        "mv_stock", "mv2_stock", "mv2_q",
        "hc_delta", "hc_vega", "hh_delta", "hh_vega",
        "v_implied", "inv_residual", "iv"]
g = {key: np.empty(shape) for key in keys}


v_hat = np.full(N_PATHS, heston_cal.v0)
iv_prev = np.full(N_PATHS, 0.2)
for i in days:
    g["c_price"][i], g["c_delta"][i], g["c_vega"][i], _, c_jumped = \
        cos_call(S[i], K, tau[i], R, V[i], BATES, shocks=jump_sizes)
    g["h_price"][i], g["h_delta"][i], g["h_vega"][i], _, h_jumped = \
        cos_call(S[i], K, tau_h[i], R, V[i], BATES, shocks=jump_sizes)

    call = (g["c_delta"][i], g["c_vega"][i], c_jumped - g["c_price"][i])
    hedge = (g["h_delta"][i], g["h_vega"][i], h_jumped - g["h_price"][i])
    g["mv_stock"][i], g["mv2_stock"][i], g["mv2_q"][i] = \
        min_var_hedge(S[i], V[i], BATES, call, hedge, jump_sizes, jump_w, MIN_VEGA)

    v_hat, _, g["hh_delta"][i], g["hh_vega"][i], g["inv_residual"][i] = \
        implied_variance(g["h_price"][i], S[i], K, tau_h[i], R, heston_cal, v_hat)
    _, g["hc_delta"][i], g["hc_vega"][i], _ = cos_call(S[i], K, tau[i], R, v_hat, heston_cal)
    g["v_implied"][i] = v_hat

    iv = implied_vol(g["c_price"][i], S[i], K, tau[i], R)
    iv_prev = np.where(np.isnan(iv), iv_prev, iv)
    g["iv"][i] = iv_prev

h_value_T = cos_call(S[-1], K, T_HEDGE - T, R, V[-1], BATES)[0]
premium = g["c_price"][0, 0]
atm_iv = g["iv"][0, 0]

# %% 4. Compare hedging strategies
def option_hedge(delta, vega, h_delta, h_vega):
    q = vega_ratio(vega, h_vega, MIN_VEGA)
    return delta - q * h_delta, q


cal_stock, cal_q = option_hedge(g["hc_delta"], g["hc_vega"], g["hh_delta"], g["hh_vega"])
true_stock, true_q = option_hedge(g["c_delta"], g["c_vega"], g["h_delta"], g["h_vega"])
cal_mv = g["hc_delta"] + heston_cal.rho * heston_cal.xi * g["hc_vega"] / S[:-1]

strategies = {
    f"BS delta, fixed {100 * atm_iv:.1f}% vol": (bs_delta(S[:-1], K, tau[:, None], R, atm_iv), None),
    "BS delta, today's implied vol":   (bs_delta(S[:-1], K, tau[:, None], R, g["iv"]), None),
    "Calibrated Heston delta":         (g["hc_delta"], None),
    "Calibrated Heston min-var delta": (cal_mv, None),
    "Calibrated Heston delta-vega":    (cal_stock, cal_q),
    "Oracle min-var delta":            (g["mv_stock"], None),
    "Oracle delta-vega":               (true_stock, true_q),
    "Oracle min-var + option":         (g["mv2_stock"], g["mv2_q"]),
}

results = {}
print(f"\nDaily hedging in the Bates market, {N_PATHS} paths (seed {args.seed}), "
      f"premium {premium:.3f}")
print(f"{'strategy':<34}{'RMSE':>8}{'mean':>9}{'std':>8}{'ES 5%':>8}")
for name, (stock, q) in strategies.items():
    opt_price = g["h_price"] if q is not None else None
    pnl, _ = hedge_pnl(S, premium, stock, q, opt_price, h_value_T if q is not None else None)
    results[name] = pnl
    s = summary(pnl)
    print(f"{name:<34}{s['rmse']:8.3f}{s['mean']:+9.3f}{s['std']:8.3f}{s['es5']:8.3f}")

boot = bootstrap_indices(N_PATHS)
print("\n95% bootstrap intervals for RMSE (jump losses make these wide and skewed;")
print("the paired differences below are the reliable comparisons):")
for name, pnl in results.items():
    lo, hi = rmse_interval(pnl, boot)
    print(f"  {name:<34}[{lo:.3f}, {hi:.3f}]")

comparisons = {
    "Model knowledge, objective fixed": [
        ("Calibrated Heston min-var delta", "Oracle min-var delta"),
        ("Calibrated Heston delta-vega", "Oracle delta-vega")],
    "Objective, true model known": [
        ("Oracle delta-vega", "Oracle min-var + option"),
        ("Oracle min-var delta", "Oracle min-var + option")],
    "Both together": [
        ("Calibrated Heston delta-vega", "Oracle min-var + option")],
}
print("\nPaired differences in RMSE (first minus second, 95% paired-bootstrap interval):")
for heading, pairs in comparisons.items():
    print(f"  {heading}:")
    for a, b in pairs:
        diff, lo, hi = paired_difference(results[a], results[b], boot)
        print(f"    {a} - {b}: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# %% 5. Check variance estimates
too_dear = g["inv_residual"] > 1e-6
too_cheap = g["inv_residual"] < -1e-6
failed = too_dear | too_cheap
print(f"\nHeston variance inversion: no v in [0, 1] reprices the hedge option on "
      f"{failed.mean():.2%} of path-days ({failed.any(axis=0).mean():.1%} of paths); "
      f"model too expensive at v = 0 on {too_dear.mean():.2%}, too cheap at v = 1 on "
      f"{too_cheap.mean():.2%}")
if failed.any():
    print(f"  on those days: mean |model - market| price {np.abs(g['inv_residual'][failed]).mean():.3f}, "
          f"max {np.abs(g['inv_residual']).max():.3f}; true variance there averages "
          f"{V[:-1][failed].mean():.4f} vs {V[:-1].mean():.4f} overall")
print(f"Its implied variance vs the true diffusive variance: correlation "
      f"{np.corrcoef(g['v_implied'].ravel(), V[:-1].ravel())[0, 1]:.3f}, "
      f"mean {g['v_implied'].mean():.4f} vs {V[:-1].mean():.4f}")

IDENTIFIED_VEGA = 0.1
errors, identified = [], []
for i in days[::21]:
    v_rec = implied_variance(g["h_price"][i], S[i], K, tau_h[i], R, BATES, 0.04)[0]
    errors.append(np.abs(v_rec - V[i]))
    identified.append(g["h_vega"][i] > IDENTIFIED_VEGA)
errors, identified = np.array(errors), np.array(identified)
print(f"Variance recovered from the hedge option price with the true model (every 21st day): "
      f"max error {errors[identified].max():.1e} where the hedge option's variance vega > "
      f"{IDENTIFIED_VEGA}; not identified (vega below that) on {1 - identified.mean():.2%} "
      f"of sampled path-days")

# %% 6. Compare paths with and without jumps
has_jump = n_jumps.sum(axis=0) > 0
print(f"\nPaths with at least one jump: {has_jump.mean():.1%} "
      f"(expected 1 - exp(-lam T) = {1 - np.exp(-BATES.lam * T):.1%})")
print(f"  ({has_jump.sum()} paths with a jump, {(~has_jump).sum()} without)")
for name in ["Calibrated Heston min-var delta", "Oracle min-var delta",
             "Calibrated Heston delta-vega", "Oracle delta-vega", "Oracle min-var + option"]:
    pnl = results[name]
    print(f"  {name:<34} RMSE with jumps {rmse(pnl[has_jump]):.3f}, "
          f"without {rmse(pnl[~has_jump]):.3f}")
print(f"No option held: oracle min-var {(g['mv2_q'] == 0).mean():.2%} of path-days, "
      f"oracle delta-vega {(true_q == 0).mean():.2%}, calibrated delta-vega "
      f"{(cal_q == 0).mean():.2%} (of which on inversion-failure days "
      f"{((cal_q == 0) & failed).mean():.2%})")
print(f"Average option holding over the year: calibrated delta-vega {cal_q.mean():.3f}, "
      f"oracle delta-vega {true_q.mean():.3f}, oracle min-var {g['mv2_q'].mean():.3f}")

# %% Plot the results
fig, ax = plt.subplots(figsize=(8, 4.6))
names = list(results)
values = [rmse(results[n]) for n in names]
errors = np.abs(np.array([rmse_interval(results[n], boot) for n in names]).T - values)
ax.barh(names, values, color=["#8da0cb"] * 2 + ["#fc8d62"] * 3 + ["#66c2a5"] * 3,
        xerr=errors, capsize=2, ecolor="0.3")
upper = np.array(values) + errors[1]
for y, (val, end) in enumerate(zip(values, upper)):
    ax.text(end + 0.03, y, f"{val:.2f}", va="center", fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.12 * upper.max())
ax.set_xlabel("RMSE of hedging P&L at expiry, 95% CI")
ax.set_title("Hedging a short 1y ATM call in the Bates (jump) market, daily")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/03_model_risk.png", dpi=150)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
shown = [names[0], "Calibrated Heston delta-vega", "Oracle delta-vega", "Oracle min-var + option"]
bins = histogram_bins([results[n] for n in shown], 120)
for name in shown:
    ax.hist(results[name], bins=bins, histtype="step", density=True, lw=1.4, label=name)
ax.set_yscale("log")
ax.set_xlabel("Hedging P&L at expiry")
ax.set_ylabel("Density (log scale)")
ax.set_title("Hedging P&L in the Bates market (log density)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/03_pnl_tails.png", dpi=150)
