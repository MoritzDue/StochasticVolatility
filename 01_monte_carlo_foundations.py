
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quant_utils import (
    bs_call_price,
    mc_summary,
    simulate_gbm_paths,
    simulate_gbm_terminal,
)


S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.20
T = 1.0

N_STEPS = 252
N_PATHS = 100_000
SEED = 42

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def discounted_call(z):
    terminal = S0 * np.exp(
        (R - 0.5 * SIGMA**2) * T
        + SIGMA * np.sqrt(T) * z
    )
    return np.exp(-R * T) * np.maximum(terminal - K, 0.0)


def main():
    # 1. Validate GBM simulation

    rng = np.random.default_rng(SEED)

    terminal = simulate_gbm_terminal(
        S0, R, SIGMA, T, N_PATHS, rng
    )

    log_returns = np.log(terminal / S0)

    theoretical_mean = (R - 0.5 * SIGMA**2) * T
    theoretical_std = SIGMA * np.sqrt(T)

    print("1. GBM validation")
    print(
        f"Log-return mean: {log_returns.mean():.5f} "
        f"(theory: {theoretical_mean:.5f})"
    )
    print(
        f"Log-return std:  {log_returns.std(ddof=1):.5f} "
        f"(theory: {theoretical_std:.5f})"
    )

    sample_paths = simulate_gbm_paths(
        S0, R, SIGMA, T, N_STEPS, 20,
        np.random.default_rng(SEED + 1),
    )

    plt.figure(figsize=(9, 4))
    plt.plot(np.linspace(0, T, N_STEPS + 1), sample_paths)
    plt.xlabel("Time (years)")
    plt.ylabel("Stock price")
    plt.title("Geometric Brownian motion: 20 simulated paths")
    plt.tight_layout()
    plt.savefig(FIGURES / "01_gbm_paths.png", dpi=160)
    plt.show()

    # 2. Monte Carlo European call vs analytical Black-Scholes

    discounted_payoffs = (
        np.exp(-R * T) * np.maximum(terminal - K, 0.0)
    )

    mc_price, mc_se, lower, upper = mc_summary(
        discounted_payoffs
    )
    bs_price = bs_call_price(S0, K, R, SIGMA, T)

    print("\n2. European call pricing")
    print(f"Black-Scholes: {bs_price:.5f}")
    print(f"Monte Carlo:   {mc_price:.5f}")
    print(f"MC SE:         {mc_se:.5f}")
    print(f"95% CI:        [{lower:.5f}, {upper:.5f}]")

    # 3. Monte Carlo convergence

    path_counts = np.array([
        100, 300, 1_000, 3_000, 10_000, 30_000, 100_000
    ])

    estimates = []
    standard_errors = []

    for n in path_counts:
        estimate, se, _, _ = mc_summary(discounted_payoffs[:n])
        estimates.append(estimate)
        standard_errors.append(se)

    plt.figure(figsize=(9, 4))
    plt.errorbar(
        path_counts,
        estimates,
        yerr=1.96 * np.array(standard_errors),
        fmt="o-",
        capsize=3,
        label="MC estimate ± 95% CI",
    )
    plt.axhline(
        bs_price,
        color="black",
        linestyle="--",
        label="Analytical Black-Scholes",
    )
    plt.xscale("log")
    plt.xlabel("Number of simulated paths")
    plt.ylabel("Call price")
    plt.title("Monte Carlo convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "01_mc_convergence.png", dpi=160)
    plt.show()

    # 4. Antithetic variates: equal payoff-evaluation budget

    n_pairs = 50_000

    iid_z = np.random.default_rng(123).standard_normal(
        2 * n_pairs
    )
    paired_z = np.random.default_rng(456).standard_normal(
        n_pairs
    )

    iid_payoffs = discounted_call(iid_z)

    # Each pair average is one independent observation.
    antithetic_pair_means = 0.5 * (
        discounted_call(paired_z)
        + discounted_call(-paired_z)
    )

    iid_price, iid_se, _, _ = mc_summary(iid_payoffs)
    anti_price, anti_se, _, _ = mc_summary(
        antithetic_pair_means
    )

    print("\n3. Antithetic variance reduction")
    print(f"Independent MC: {iid_price:.5f}, SE = {iid_se:.5f}")
    print(f"Antithetic MC:  {anti_price:.5f}, SE = {anti_se:.5f}")
    print(
        f"SE reduction at equal payoff evaluations: "
        f"{100 * (1 - anti_se / iid_se):.1f}%"
    )

    # 5. Path-dependent arithmetic-average Asian call

    n_asian_paths = 20_000
    dt = T / N_STEPS
    asian_rng = np.random.default_rng(789)

    stock = np.full(n_asian_paths, S0)
    running_sum = np.zeros(n_asian_paths)

    for _ in range(N_STEPS):
        z = asian_rng.standard_normal(n_asian_paths)
        stock *= np.exp(
            (R - 0.5 * SIGMA**2) * dt
            + SIGMA * np.sqrt(dt) * z
        )
        running_sum += stock

    arithmetic_average = running_sum / N_STEPS

    asian_payoffs = np.exp(-R * T) * np.maximum(
        arithmetic_average - K, 0.0
    )
    european_payoffs = np.exp(-R * T) * np.maximum(
        stock - K, 0.0
    )

    asian_price, asian_se, _, _ = mc_summary(asian_payoffs)
    european_price, _, _, _ = mc_summary(european_payoffs)

    print("\n4. Path-dependent pricing")
    print(f"Asian call MC:    {asian_price:.5f} (SE {asian_se:.5f})")
    print(f"European call MC: {european_price:.5f}")
    print("Asian average uses 252 daily observations, excluding t=0.")


if __name__ == "__main__":
    main()
