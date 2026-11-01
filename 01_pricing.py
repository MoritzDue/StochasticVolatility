# %% [markdown]
# # Part 1 - Pricing in the two worlds
#   1. Monte Carlo prices vs the COS (Fourier) prices for Heston and Bates.
#   2. The implied-volatility smiles each world produces.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sv import BATES, HESTON, R, S0, cos_call, implied_vol, log_output, simulate

Path("figures").mkdir(exist_ok=True)
log_output("results/01_pricing.txt")     # everything printed is also saved here
strikes = np.arange(80.0, 125.0, 5.0)

# %% 1. Monte Carlo vs COS at T = 1
n_paths = 500_000
print(f"European calls, T = 1, {n_paths:,} paths, 252 steps")
print(f"{'model':<8}{'strike':>7}{'COS':>10}{'MC':>10}{'95% CI half-width':>20}{'z':>7}")
for name, model, seed in [("Heston", HESTON, 1), ("Bates", BATES, 2)]:
    ST = simulate(model, n_paths, seed=seed, substeps=1, keep_paths=False)
    payoffs = np.exp(-R) * np.maximum(ST[:, None] - strikes[None, :], 0)
    mc = payoffs.mean(axis=0)
    se = payoffs.std(axis=0) / np.sqrt(n_paths)
    exact = cos_call(S0, strikes, 1.0, R, model.v0, model)[0]
    for k, e, p, s in zip(strikes[::2], exact[::2], mc[::2], se[::2]):
        print(f"{name:<8}{k:7.0f}{e:10.4f}{p:10.4f}{1.96 * s:20.4f}{(p - e) / s:7.1f}")
    fwd_z = (ST.mean() - S0 * np.exp(R)) / (ST.std() / np.sqrt(n_paths))
    print(f"{name:<8} forward E[S_T] vs S0 e^rT: z = {fwd_z:.1f}")


# %% 2. Implied volatility smiles
maturities = [0.25, 0.5, 1.0, 1.5]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (name, model) in zip(axes, [("Heston", HESTON), ("Bates (Heston + jumps)", BATES)]):
    for T_ in maturities:
        prices = cos_call(S0, strikes, T_, R, model.v0, model)[0]
        ax.plot(strikes, 100 * implied_vol(prices, S0, strikes, T_, R), "o-", ms=3,
                label=f"T = {T_}")
    ax.set_title(name)
    ax.set_xlabel("Strike")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Black-Scholes implied vol (%)")
axes[0].legend()
fig.suptitle("Implied volatility smiles of the two model worlds")
fig.tight_layout()
fig.savefig("figures/01_smiles.png", dpi=150)

print("\nATM / 80-strike / 120-strike implied vols (%)")
for name, model in [("Heston", HESTON), ("Bates", BATES)]:
    for T_ in (0.25, 1.0):
        p = cos_call(S0, [80, 100, 120], T_, R, model.v0, model)[0]
        iv = 100 * implied_vol(p, S0, np.array([80, 100, 120]), T_, R)
        print(f"{name:<7} T={T_:<5} 80: {iv[0]:.2f}  100: {iv[1]:.2f}  120: {iv[2]:.2f}  "
              f"skew (80 - 120): {iv[0] - iv[2]:.2f}")
