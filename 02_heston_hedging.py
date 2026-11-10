# %% [markdown]
# # Part 2 - Heston hedging
#   A short 1-year ATM call is hedged daily on simulated Heston paths, and on Black-Scholes
#   paths driven by the same stock shocks for a like-for-like reference.
#     1. Stock-only hedges: BS delta (fixed and implied vol), Heston delta, min-var delta.
#     2. Continuous-time benchmark: the Ito-isometry error floor for any stock-only hedge.
#     3. Stock + option hedges (delta-gamma, delta-vega) and rebalancing frequency.
#     4. Proportional transaction costs, daily vs weekly.


# %%
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sv import (HESTON, K, N_DAYS, R, S0, T, bootstrap_indices, bs_call, bs_delta, bs_gamma,
                bs_vega, cos_call, gbm_paths, hedge_pnl, histogram_bins, implied_vol,
                log_output, paired_difference, rmse, rmse_interval, simulate, summary,
                vega_ratio)

parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=int, default=4000)
parser.add_argument("--seed", type=int, default=1)
args, _ = parser.parse_known_args()

Path("figures").mkdir(exist_ok=True)
N_PATHS = args.paths


default_run = (args.paths, args.seed) == (4000, 1)
log_output("results/02_heston_hedging.txt" if default_run else
           f"results/02_heston_hedging_paths{args.paths}_seed{args.seed}.txt")
SIGMA = 0.20
T_HEDGE = 1.25


S, V, Z, _ = simulate(HESTON, N_PATHS, seed=args.seed)
S_bs = gbm_paths(SIGMA, Z)

# %% Calculate option Greeks
days = np.arange(N_DAYS)
tau = T - days / N_DAYS
tau_h = T_HEDGE - days / N_DAYS

shape = (N_DAYS, N_PATHS)
keys = ("price", "delta", "vega", "gamma", "h_price", "h_delta", "h_vega", "h_gamma")
heston = {key: np.empty(shape) for key in keys}
for i in days:
    heston["price"][i], heston["delta"][i], heston["vega"][i], heston["gamma"][i] = \
        cos_call(S[i], K, tau[i], R, V[i], HESTON)
    heston["h_price"][i], heston["h_delta"][i], heston["h_vega"][i], heston["h_gamma"][i] = \
        cos_call(S[i], K, tau_h[i], R, V[i], HESTON)
h_value_T = cos_call(S[-1], K, T_HEDGE - T, R, V[-1], HESTON)[0]


Sb, tb, thb = S_bs[:-1], tau[:, None], tau_h[:, None]
bs_world = {
    "delta": bs_delta(Sb, K, tb, R, SIGMA),
    "vega": bs_vega(Sb, K, tb, R, SIGMA),
    "gamma": bs_gamma(Sb, K, tb, R, SIGMA),
    "h_price": bs_call(Sb, K, thb, R, SIGMA),
    "h_delta": bs_delta(Sb, K, thb, R, SIGMA),
    "h_vega": bs_vega(Sb, K, thb, R, SIGMA),
    "h_gamma": bs_gamma(Sb, K, thb, R, SIGMA),
}
bs_h_value_T = bs_call(S_bs[-1], K, T_HEDGE - T, R, SIGMA)


def option_hedge(g, greek):
    """Stock + second option, chosen so the book has zero delta and zero `greek`.

    The ratio is guarded: deep out of the money both Greeks are tiny and the numerical
    values are noise, and there is then nothing left to hedge (see sv.vega_ratio).
    """
    q = vega_ratio(g[greek], g["h_" + greek], 1e-3 if greek == "vega" else 1e-7)
    return g["delta"] - q * g["h_delta"], q


# %% Hedging strategies
premium_heston = cos_call(S0, K, T, R, HESTON.v0, HESTON)[0]
premium_bs = bs_call(S0, K, T, R, SIGMA)
mv_delta = heston["delta"] + HESTON.rho * HESTON.xi * heston["vega"] / S[:-1]
bs_fixed_delta = bs_delta(S[:-1], K, tb, R, SIGMA)


# BS delta at today's implied vol: what a Black-Scholes desk would actually run. The implied
# vol of the sold call is NaN where its price leaves the BS bounds (deep in or out of the
# money near expiry); there the desk keeps yesterday's vol (20% on day 0).
iv = implied_vol(heston["price"], S[:-1], K, tb, R)
for i in days:
    previous = iv[i - 1] if i > 0 else np.full(N_PATHS, SIGMA)
    iv[i] = np.where(np.isnan(iv[i]), previous, iv[i])
bs_iv_delta = bs_delta(S[:-1], K, tb, R, iv)


for world, g in [("BS world", bs_world), ("Heston world", heston)]:
    for greek in ("gamma", "vega"):
        print(f"{world}, {greek}-neutral: no option held (guard) on "
              f"{(option_hedge(g, greek)[1] == 0).mean():.2%} of path-days")


def option_strategy(paths, premium, g, value_T, greek):
    stock, q = option_hedge(g, greek)
    return paths, premium, stock, q, g["h_price"], value_T


strategies = {
    "BS world: BS delta":            (S_bs, premium_bs, bs_world["delta"], None, None, None),
    "BS world: delta-gamma":         option_strategy(S_bs, premium_bs, bs_world, bs_h_value_T, "gamma"),
    "BS world: delta-vega":          option_strategy(S_bs, premium_bs, bs_world, bs_h_value_T, "vega"),
    "Heston world: BS delta (20%)":  (S, premium_heston, bs_fixed_delta, None, None, None),
    "Heston world: BS delta (impl.)": (S, premium_heston, bs_iv_delta, None, None, None),
    "Heston world: Heston delta":    (S, premium_heston, heston["delta"], None, None, None),
    "Heston world: min-var delta":   (S, premium_heston, mv_delta, None, None, None),
    "Heston world: delta-gamma":     option_strategy(S, premium_heston, heston, h_value_T, "gamma"),
    "Heston world: delta-vega":      option_strategy(S, premium_heston, heston, h_value_T, "vega"),
}


def run(name, every=1, cost_stock=0.0, cost_opt=0.0):
    paths, premium, stock, opt, opt_price, opt_T = strategies[name]
    return hedge_pnl(paths, premium, stock, opt, opt_price, opt_T, every, cost_stock, cost_opt)


# %% Daily hedging results
daily = {name: run(name)[0] for name in strategies}
print(f"Daily rebalancing, {N_PATHS} paths. P&L = hedge portfolio - payoff at expiry.\n")
print(f"{'strategy':<32}{'RMSE':>8}{'mean':>9}{'std':>8}{'ES 5%':>8}")
for name, pnl in daily.items():
    s = summary(pnl)
    print(f"{name:<32}{s['rmse']:8.3f}{s['mean']:+9.3f}{s['std']:8.3f}{s['es5']:8.3f}")
print(f"(Monte Carlo s.e. of a mean is about std / {np.sqrt(N_PATHS):.0f}: under the pricing "
      "measure every hedge has mean zero up to simulation and discretisation error.)")

bs_only = rmse(daily["BS world: BS delta"])
heston_bs = rmse(daily["Heston world: BS delta (20%)"])
print(f"\nBS delta RMSE, Heston world / BS world: {heston_bs / bs_only:.2f}x")


boot = bootstrap_indices(N_PATHS)
print("\n95% bootstrap intervals for RMSE:")
for name, pnl in daily.items():
    lo, hi = rmse_interval(pnl, boot)
    print(f"  {name:<32}[{lo:.3f}, {hi:.3f}]")


print("\nPaired differences in RMSE (same paths), with 95% paired-bootstrap intervals:")
for a, b in [("Heston world: Heston delta", "Heston world: BS delta (20%)"),
             ("Heston world: BS delta (impl.)", "Heston world: BS delta (20%)"),
             ("Heston world: min-var delta", "Heston world: BS delta (impl.)"),
             ("Heston world: delta-vega", "Heston world: min-var delta"),
             ("Heston world: delta-gamma", "Heston world: min-var delta")]:
    diff, lo, hi = paired_difference(daily[a], daily[b], boot)
    print(f"  {a} - {b}: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# %% Continuous-time hedging error


def continuous_variance(a):
    """Per-path integrand of the Ito isometry, summed over the daily grid."""
    gap = (a - heston["delta"]) * S[:-1]
    nu = heston["vega"]
    rate = V[:-1] * (gap**2 - 2 * HESTON.rho * HESTON.xi * gap * nu + HESTON.xi**2 * nu**2)
    return np.sum(np.exp(2 * R * tau)[:, None] * rate, axis=0) / N_DAYS


print("\nStock-only hedges in the Heston world: continuous-time theory vs daily simulation")
print(f"{'strategy':<32}{'continuous (95% CI)':>24}{'daily, simulated':>18}{'floor share':>13}")
stock_only = [("Heston world: BS delta (20%)", bs_fixed_delta),
              ("Heston world: BS delta (impl.)", bs_iv_delta),
              ("Heston world: Heston delta", heston["delta"]),
              ("Heston world: min-var delta", mv_delta)]
continuous = {}
floor_variance = continuous_variance(mv_delta).mean()
for name, a in stock_only:
    per_path = continuous_variance(a)
    continuous[name] = np.sqrt(per_path.mean())
    lo, hi = np.percentile([np.sqrt(per_path[b].mean()) for b in boot], [2.5, 97.5])
    print(f"{name:<32}{continuous[name]:9.3f} [{lo:.3f}, {hi:.3f}]{rmse(daily[name]):18.3f}"
          f"{floor_variance / per_path.mean():13.0%}")


print("Extra error from daily rebalancing, if it were uncorrelated with the rest:")
for name, value in continuous.items():
    print(f"  {name:<32}{np.sqrt(max(rmse(daily[name])**2 - value**2, 0)):.3f}"
          f"   (daily BS delta in the BS world: {bs_only:.3f})")

# %%
print("\nHedge ratios at t = 0 (S = 100, v = 0.04):")
print(f"  BS delta (20% vol): {bs_delta(S0, K, T, R, SIGMA):.4f}")
print(f"  BS delta (implied): {bs_iv_delta[0, 0]:.4f}")
print(f"  Heston delta:       {heston['delta'][0, 0]:.4f}")
print(f"  min-variance delta: {mv_delta[0, 0]:.4f}")
print(f"  average over all paths and days: BS {bs_fixed_delta.mean():.4f}, "
      f"BS implied {bs_iv_delta.mean():.4f}, Heston {heston['delta'].mean():.4f}, "
      f"min-var {mv_delta.mean():.4f}")

# %%
fig, ax = plt.subplots(figsize=(8, 4.6))
names = list(daily)
values = [rmse(daily[n]) for n in names]
colors = ["#8da0cb" if n.startswith("BS world") else "#fc8d62" for n in names]
errors = np.abs(np.array([rmse_interval(daily[n], boot) for n in names]).T - values)
ax.barh(names, values, color=colors, xerr=errors, capsize=2, ecolor="0.3")
upper = np.array(values) + errors[1]
for y, (val, end) in enumerate(zip(values, upper)):
    ax.text(end + 0.02, y, f"{val:.2f}", va="center", fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.12 * upper.max())
ax.set_xlabel("RMSE of hedging P&L at expiry (per call, premium about 10.4), 95% CI")
ax.set_title("Daily hedging of a short 1y ATM call")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/02_hedge_comparison.png", dpi=150)

# %% Rebalancing frequency
frequencies = [63, 21, 10, 5, 2, 1]
curves = ["BS world: BS delta", "Heston world: BS delta (20%)", "Heston world: BS delta (impl.)",
          "Heston world: min-var delta", "Heston world: delta-vega"]
scaling = {name: [rmse(run(name, every=f)[0]) for f in frequencies] for name in curves}
n_trades = [int(np.ceil(N_DAYS / f)) for f in frequencies]

print("\nRMSE vs number of rebalances per year")
print(f"{'strategy':<32}" + "".join(f"{n:>7}" for n in n_trades))
for name in curves:
    print(f"{name:<32}" + "".join(f"{x:7.3f}" for x in scaling[name]))

fine = np.array(n_trades) >= 51
slope = np.polyfit(np.log(np.array(n_trades)[fine]),
                   np.log(np.array(scaling["BS world: BS delta"])[fine]), 1)[0]
print(f"log-log slope in the BS world, 51-252 rebalances: {slope:.2f} (theory -0.5)")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for name in curves:
    ax.loglog(n_trades, scaling[name], "o-", label=name)
floor = continuous["Heston world: min-var delta"]
ax.axhline(floor, color="grey", ls="--", lw=1)
ax.text(n_trades[-1], floor * 0.93, "continuous-time floor, any stock-only hedge",
        fontsize=8, color="grey", va="top", ha="right")
ax.set_xticks(n_trades, [str(n) for n in n_trades])
ax.set_yticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
ax.minorticks_off()
ax.set_xlabel("Rebalances per year")
ax.set_ylabel("RMSE of hedging P&L")
ax.set_title("More frequent rebalancing cannot remove volatility risk")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("figures/02_rebalancing.png", dpi=150)

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.2))
shown = ["BS world: BS delta", "Heston world: BS delta (20%)", "Heston world: delta-vega"]
bins = histogram_bins([daily[n] for n in shown], 120)
for name in shown:
    ax.hist(daily[name], bins=bins, histtype="step", density=True, lw=1.6, label=name)
ax.set_xlabel("Hedging P&L at expiry")
ax.set_ylabel("Density")
ax.set_title("Daily hedging P&L distributions")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/02_pnl_distributions.png", dpi=150)

# %% Transaction costs


with_costs = {}
print("\nWith costs (stock 2 bp, options 1% of price)")
print(f"{'strategy':<32}{'every':>6}{'RMSE':>8}{'mean':>9}{'std':>8}{'mean cost':>11}")
for name in ["Heston world: BS delta (20%)", "Heston world: BS delta (impl.)",
             "Heston world: min-var delta", "Heston world: delta-vega"]:
    for every in (1, 5):
        pnl, costs = run(name, every, cost_stock=0.0002, cost_opt=0.01)
        with_costs[name, every] = pnl
        s = summary(pnl)
        print(f"{name:<32}{every:>6}{s['rmse']:8.3f}{s['mean']:+9.3f}{s['std']:8.3f}"
              f"{costs.mean():11.3f}")

# Weekly vs daily delta-vega with costs, paired on the same paths
weekly, daily_dv = with_costs["Heston world: delta-vega", 5], with_costs["Heston world: delta-vega", 1]
diff, lo, hi = paired_difference(weekly, daily_dv, boot)
std_diffs = [weekly[b].std() - daily_dv[b].std() for b in boot]
std_lo, std_hi = np.percentile(std_diffs, [2.5, 97.5])
print("\nDelta-vega with costs, weekly minus daily (95% paired-bootstrap interval):")
print(f"  RMSE: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")
print(f"  std:  {weekly.std() - daily_dv.std():+.3f} [{std_lo:+.3f}, {std_hi:+.3f}]")
