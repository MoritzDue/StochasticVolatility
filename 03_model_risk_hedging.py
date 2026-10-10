
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quant_utils import (
    HestonParams,
    bs_call_price,
    delta_hedge,
    hedge_metrics,
    heston_call_mc,
    simulate_gbm_paths,
    simulate_heston,
)


# Parameters
S0, K, R, T, BS_VOL = 100.0, 100.0, 0.05, 1.0, 0.20
PARAMS = HestonParams(v0=0.04, theta=0.04, kappa=2.0, xi=0.30, rho=-0.70)

N_STEPS = 252
N_HEDGE_PATHS = 20_000
N_PRICE_PATHS = 200_000
SEED = 42

FREQUENCIES = {"Monthly": 21, "Weekly": 5, "Daily": 1}
ROBUSTNESS_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 111)
SENSITIVITY_SEEDS = (101, 202, 303, 404, 505)
XI_VALUES = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def report(label, errors):
    stats = hedge_metrics(errors)
    print(f"{label:<12} Mean = {stats['mean']:+.4f}, RMSE = {stats['rmse']:.4f}")
    return stats["rmse"]


def main():
    # 1. Model-consistent initial option prices
    bs_price = bs_call_price(S0, K, R, BS_VOL, T)

    prices, price_se, _, _ = heston_call_mc(
        S0, K, R, T, PARAMS, N_STEPS, N_PRICE_PATHS, seed=999
    )
    heston_price = float(prices[0])

    print("1. INITIAL OPTION PRICES")
    print(f"Black-Scholes: {bs_price:.5f}")
    print(f"Heston MC:     {heston_price:.5f} (SE = {price_se[0]:.5f})")

    # 2. Simulate hedging paths under both models
    bs_paths = simulate_gbm_paths(
        S0, R, BS_VOL, T, N_STEPS, N_HEDGE_PATHS,
        np.random.default_rng(SEED)
    )

    heston_paths, variance_paths = simulate_heston(
        S0, R, T, PARAMS, N_STEPS, N_HEDGE_PATHS,
        np.random.default_rng(SEED), store_paths=True
    )

    # 3. Delta hedging: monthly, weekly and daily
    bs_rmse, heston_rmse = [], []

    print("\n2. BLACK-SCHOLES DELTA HEDGING")

    for name, frequency in FREQUENCIES.items():
        bs_errors = delta_hedge(
            bs_paths, bs_price, K, R, T, BS_VOL, frequency
        )
        heston_errors = delta_hedge(
            heston_paths, heston_price, K, R, T, BS_VOL, frequency
        )

        print(f"\n{name} rebalancing")
        bs_rmse.append(report("BS market", bs_errors))
        heston_rmse.append(report("Heston market", heston_errors))

    daily_ratio = heston_rmse[-1] / bs_rmse[-1]

    print("\n3. MAIN MODEL-RISK RESULT")
    print(f"Daily BS RMSE:     {bs_rmse[-1]:.4f}")
    print(f"Daily Heston RMSE: {heston_rmse[-1]:.4f}")
    print(f"Heston / BS ratio: {daily_ratio:.2f}x")

    # 4. Hedging RMSE comparison
    x = np.arange(len(FREQUENCIES))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, bs_rmse, width, label="BS market")
    plt.bar(x + width / 2, heston_rmse, width, label="Heston market")
    plt.xticks(x, list(FREQUENCIES))
    plt.ylabel("Terminal hedging RMSE")
    plt.title("Black-Scholes Delta Hedging: Model Risk")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "03_hedging_model_risk.png", dpi=160)
    plt.show()

    del bs_paths, heston_paths, variance_paths

    # 5. Robustness across random seeds
    ratios = []
    bs_seed_rmses = []

    print("\n4. RANDOM-SEED ROBUSTNESS")

    for seed in ROBUSTNESS_SEEDS:
        bs_paths = simulate_gbm_paths(
            S0, R, BS_VOL, T, N_STEPS, 5_000,
            np.random.default_rng(seed)
        )
        heston_paths, _ = simulate_heston(
            S0, R, T, PARAMS, N_STEPS, 5_000,
            np.random.default_rng(seed), store_paths=True
        )

        bs_errors = delta_hedge(bs_paths, bs_price, K, R, T, BS_VOL)
        heston_errors = delta_hedge(
            heston_paths, heston_price, K, R, T, BS_VOL
        )

        bs_error_rmse = hedge_metrics(bs_errors)["rmse"]
        heston_error_rmse = hedge_metrics(heston_errors)["rmse"]

        bs_seed_rmses.append(bs_error_rmse)
        ratios.append(heston_error_rmse / bs_error_rmse)

    ratios = np.array(ratios)

    print(f"Mean Heston / BS RMSE ratio: {ratios.mean():.3f}x")
    print(f"Across-seed standard deviation: {ratios.std(ddof=1):.3f}")

    del bs_paths, heston_paths

    # 6. Volatility-of-volatility sensitivity
    mean_rmses, std_rmses = [], []

    print("\n5. VOL-OF-VOL SENSITIVITY")

    for xi in XI_VALUES:
        params_xi = replace(PARAMS, xi=float(xi))

        # xi=0 and v0=theta reduce Heston to constant volatility.
        if xi == 0:
            initial_price = bs_price
        else:
            prices, _, _, _ = heston_call_mc(
                S0, K, R, T, params_xi, N_STEPS, 100_000,
                seed=777 + int(round(1000 * xi))
            )
            initial_price = float(prices[0])

        seed_rmses = []

        for seed in SENSITIVITY_SEEDS:
            stock_paths, _ = simulate_heston(
                S0, R, T, params_xi, N_STEPS, 5_000,
                np.random.default_rng(seed), store_paths=True
            )

            errors = delta_hedge(
                stock_paths, initial_price, K, R, T, BS_VOL
            )
            seed_rmses.append(hedge_metrics(errors)["rmse"])

        mean_rmse = np.mean(seed_rmses)
        std_rmse = np.std(seed_rmses, ddof=1)

        mean_rmses.append(mean_rmse)
        std_rmses.append(std_rmse)

        print(f"xi = {xi:.2f}: RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")

    # 7. Vol-of-vol sensitivity plot
    plt.figure(figsize=(9, 5))
    plt.errorbar(
        XI_VALUES, mean_rmses, yerr=std_rmses,
        marker="o", capsize=4,
        label="Heston: mean RMSE ± 1 across-seed SD"
    )
    plt.axhline(
        np.mean(bs_seed_rmses), color="black", linestyle="--",
        label="BS market: daily RMSE"
    )
    plt.xlabel("Volatility of volatility (xi)")
    plt.ylabel("Daily hedging RMSE")
    plt.title("Impact of Vol-of-Vol on Delta Hedging")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "03_vol_of_vol_sensitivity.png", dpi=160)
    plt.show()

   

if __name__ == "__main__":
    main()
