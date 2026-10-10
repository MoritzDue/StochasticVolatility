from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quant_utils import (
    HestonParams,
    bs_call_price,
    implied_volatility,
    mc_summary,
    simulate_heston,
)


S0 = 100.0
R = 0.05
T = 1.0
BS_VOL = 0.20

PARAMS = HestonParams(
    v0=0.04,
    theta=0.04,
    kappa=2.0,
    xi=0.30,
    rho=-0.70,
)

N_STEPS = 252
N_PATHS = 100_000
SEED = 42

STRIKES = np.arange(80.0, 125.0, 5.0)

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def main():
    # 1. Simulate risk-neutral Heston terminal prices

    terminal = simulate_heston(
        spot=S0,
        rate=R,
        maturity=T,
        params=PARAMS,
        n_steps=N_STEPS,
        n_paths=N_PATHS,
        rng=np.random.default_rng(SEED),
    )

    # 2. Risk-neutral martingale and variance sanity checks

    simulated_forward = terminal.mean()
    forward_se = terminal.std(ddof=1) / np.sqrt(N_PATHS)
    theoretical_forward = S0 * np.exp(R * T)

    feller_lhs = 2 * PARAMS.kappa * PARAMS.theta
    feller_rhs = PARAMS.xi**2

    print("1. Heston simulation checks")
    print(f"Simulated E[S_T]: {simulated_forward:.5f}")
    print(f"Theoretical E[S_T]: {theoretical_forward:.5f}")
    print(
        f"95% MC CI for E[S_T]: "
        f"[{simulated_forward - 1.96 * forward_se:.5f}, "
        f"{simulated_forward + 1.96 * forward_se:.5f}]"
    )
    print(
        f"Feller condition: 2*kappa*theta = {feller_lhs:.4f}, "
        f"xi^2 = {feller_rhs:.4f}"
    )
    print(
        f"Feller condition satisfied: "
        f"{feller_lhs >= feller_rhs}"
    )

    # 3. European call pricing across strikes

    discounted_payoffs = np.exp(-R * T) * np.maximum(
        terminal[:, None] - STRIKES[None, :], 0.0
    )

    heston_prices, heston_se, lower, upper = mc_summary(
        discounted_payoffs
    )

    bs_prices = bs_call_price(
        S0, STRIKES, R, BS_VOL, T
    )

    print("\n2. European call pricing")
    print(
        f"{'Strike':>8} {'Heston':>12} {'MC SE':>10} "
        f"{'BS 20%':>12} {'95% MC CI':>24}"
    )

    for k, price, se, bs, lo, hi in zip(
        STRIKES,
        heston_prices,
        heston_se,
        bs_prices,
        lower,
        upper,
    ):
        print(
            f"{k:8.0f} {price:12.4f} {se:10.4f} "
            f"{bs:12.4f} [{lo:.4f}, {hi:.4f}]"
        )

    # 4. Convert Heston prices into BS implied volatilities

    implied_vols = np.array([
        implied_volatility(price, S0, strike, R, T)
        for price, strike in zip(heston_prices, STRIKES)
    ])

    print("\n3. Black-Scholes implied volatilities")
    for strike, iv in zip(STRIKES, implied_vols):
        print(f"K = {strike:.0f}: IV = {iv:.2%}")

    if not np.all(np.isfinite(implied_vols)):
        print(
            "Warning: at least one MC price could not be "
            "converted into a valid implied volatility."
        )

    # 5. Visualise Heston pricing and implied-volatility skew

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5)
    )

    axes[0].errorbar(
        STRIKES,
        heston_prices,
        yerr=1.96 * heston_se,
        fmt="o-",
        capsize=3,
        label="Heston MC ± 95% CI",
    )
    axes[0].plot(
        STRIKES,
        bs_prices,
        "k--",
        label="Black-Scholes, 20% vol",
    )
    axes[0].set_xlabel("Strike")
    axes[0].set_ylabel("European call price")
    axes[0].set_title("Call prices across strikes")
    axes[0].legend()

    axes[1].plot(
        STRIKES,
        implied_vols,
        "o-",
        label="Heston-implied volatility",
    )
    axes[1].axhline(
        BS_VOL,
        color="black",
        linestyle="--",
        label="Constant BS volatility",
    )
    axes[1].set_xlabel("Strike")
    axes[1].set_ylabel("Black-Scholes implied volatility")
    axes[1].set_title("Stochastic-volatility skew")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES / "02_heston_skew.png", dpi=160)
    plt.show()

    print("\nParameters are illustrative, not market-calibrated.")
    print(
        "Confidence intervals quantify MC sampling error, "
        "not Heston discretisation bias."
    )


if __name__ == "__main__":
    main()
