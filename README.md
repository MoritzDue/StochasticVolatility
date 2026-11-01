# Hedging options when volatility is stochastic

A simulation study of why delta hedging fails when volatility moves, what fixes it, and what a well-calibrated but wrong model costs.

## Key findings

1. **How much of the stock-only hedging error is volatility risk, and how much is the choice of delta?** Daily Black-Scholes delta hedging is 3.5x worse in a Heston market than in a Black-Scholes one. In a stochastic-volatility market the "right" delta depends on the objective: the Heston delta ∂C/∂S is not the error-minimising stock position, the minimum-variance delta is. The Itô isometry gives each stock-only hedge's error under continuous rebalancing and the floor that no stock-only hedge can beat (RMSE 1.31). That floor is 87% of the continuous-time variance for BS delta at today's implied vol, 77% at a fixed 20% vol and only 51% for the Heston delta, which is worse than the naive BS delta (paired difference Heston minus fixed-vol BS delta: +0.35 [+0.31, +0.38]). The rest is the choice of delta.
2. **A second option removes the volatility risk; controls separate this from the gamma effect.** Delta-vega hedging cuts RMSE from 1.37 (best stock-only hedge) to 0.23 in the Heston world, against 0.26 for the same hedge in a constant-vol world. A gamma-neutral hedge, which fixes rebalancing error but not volatility risk, only gets to 0.89. With a Heston market and a second option priced by Heston the market is complete, so this part confirms the mechanics rather than testing a hypothesis. With costs (daily rebalancing), delta-vega's advantage over the min-var delta shrinks from 1.14 to 0.62 (RMSE 0.76 vs 1.38), and its error is now mostly a predictable cost (mean −0.64) rather than a wide spread (std 0.41).
3. **Extension: in a jump market, how much is knowing the true model worth, and how much is the hedging objective worth?** A Heston model calibrated to a Bates market fits vanilla prices to 0.08 vol points with the wrong parameters (κ = 3.8 vs 2, ξ = 0.46 vs 0.30). For a fixed objective, knowing the true model is worth little: calibrated minus oracle RMSE is +0.10 [+0.06, +0.13] for the stock-only min-var delta, and for delta-vega the calibrated hedger is actually slightly *better*, −0.08 [−0.11, −0.06]: its inflated vol-of-vol makes it hold more of the hedge option (0.84 vs 0.76 on average), which plausibly covers part of the jump risk (a hypothesis, not tested here). The objective matters far more: for a fixed (true) model, switching the oracle from delta-vega to jump-aware minimum variance cuts RMSE by 0.47 [0.28, 0.71], from 0.92 to 0.45. The calibrated hedger cannot make that switch, because its model has no jumps.

## Setup

- Short one 1-year ATM call (S0 = K = 100, r = 5%), hedged daily over 252 trading days. 4,000 paths in Part 2 and, by default, in Part 3 (`--paths`, `--seed` to change).
- **Heston world:** v0 = θ = 0.04, κ = 2, ξ = 0.3, ρ = −0.7.
- **Black-Scholes world:** σ = 20%, driven by the same stock shocks as the Heston world, so comparisons are paired.
- **Bates world** (Heston + lognormal jumps, Part 3): v0 = θ = 0.032, κ = 2, ξ = 0.3, ρ = −0.7, jump intensity λ = 0.4 per year, log jump size N(−10%, 10%²). Expected annual variance of log-returns is 0.032 + λ·E[J²] = 0.04, as in the Heston world (the ATM implied vol is not exactly 20%).
- **Option hedges** use a second call (strike 100, expiry 1.25 years), traded at model prices.
- **Hedging error** = value of the self-financing hedge portfolio at expiry minus the call payoff. I report RMSE (the premium is about 10.4), mean and 5% expected shortfall, with 95% bootstrap intervals over paths (1,000 resamples). Every conclusion that compares two hedges rests on a paired bootstrap difference on the same paths, not on overlapping marginal intervals.

## Methodology

**Pricing.** Prices, deltas, variance vegas and gammas come from the COS Fourier method (Fang & Oosterlee, 2008). The Heston characteristic function uses the "little Heston trap" form, written without dividing by ξ², so it reduces exactly to Black-Scholes at ξ = 0. Bates prices are a Poisson mixture over the number of jumps, each term with its own integration range. The pricer also returns prices after an instant spot shock from the same characteristic-function evaluation, which the jump-aware hedge needs.

**Numerical edge cases.** The hedging loop reaches 1-day options and paths with zero variance (the Euler scheme truncates at 0), so the pricer is tested there explicitly:

- **Integration range:** the COS range is set from the mean and variance of the log-return. These are taken from the characteristic function by central differences. The usual closed-form Heston cumulants lose precision at short maturities (they understate the variance several times over for a 1-day option at low variance), which made the range too narrow.
- **Poisson cutoff:** the Bates sum is cut at 1e-10 of probability mass. A 1e-6 cut leaves price errors of order 1e-5, because the dropped many-jump terms are priced very differently from the rest.
- **Tests** (`test_sv.py`):
  - prices and deltas match independent numerical integration to 1e-5 for T from 1 day to 1.5 years and v from 0 to 0.15
  - no-arbitrage bounds hold on a wide grid
  - gamma, vega and the shocked prices match finite differences or direct repricing
  - the cumulants match the integrated variance as ξ → 0
  - the model converges to Black-Scholes at rate O(ξ)
  - Monte Carlo agrees with COS
  - the implied-variance solver recovers v (where the option has vega) from any start, and reports when no v fits
  - the hedge accounting is exactly self-financing, and the costs and recorded jumps are correct
  - the jump-aware minimum-variance hedge reduces to the min-var delta and the delta-vega hedge without jumps, and with jumps no nearby position has a smaller local variance

**Simulation.** Full-truncation Euler for the variance and log-Euler for the stock, with 4 steps per day. Jumps are simulated as compound Poisson with the risk-neutral compensator, and the number of jumps on each path and day is recorded.

**Monte Carlo check (Part 1).** All strikes are priced on the same paths, so their z-scores move together and count as one test per model; the forward E[S_T] is reported alongside.

**Stock-only hedges (Part 2).** BS delta at a fixed 20%, BS delta at today's implied vol of the sold call (what a Black-Scholes desk would actually use), the Heston delta ∂C/∂S, and the minimum-variance delta Δ + ρξ(∂C/∂v)/S, which minimises the instantaneous variance of the hedged book.

**Continuous-time benchmark (Part 2).** For a stock position *a* against a short Heston call, Itô's lemma and the Heston PDE make the discounted hedge error a martingale with increments (a − Δ) dS − 𝒱 dv, where 𝒱 = ∂C/∂v. The Itô isometry gives its exact second moment under continuous rebalancing:

$$E[\text{error}_T^2] = E\int_0^T e^{2r(T-t)}\, v\left[(a-\Delta)^2S^2 - 2\rho\xi(a-\Delta)S\,\mathcal{V} + \xi^2\mathcal{V}^2\right]dt .$$

At the minimum-variance delta this becomes E∫e^{2r(T−t)} ξ²(1−ρ²) v 𝒱² dt, a floor for every stock-only hedge. I estimate the integral along the simulated paths with a left Riemann sum on the daily grid, and report a bootstrap interval. The gap between this continuous-time error and the simulated daily error is the cost of rebalancing only once a day. The two are not exactly orthogonal, so I don't split the daily error into additive parts; sqrt(daily² − continuous²) is reported only as a rough consistency check against the daily error in the BS world. The "floor share" (floor variance ÷ a hedge's continuous-time variance) is the clean split between volatility risk and the choice of delta, because both terms are continuous-time quantities.

**Option hedges (Part 2).** Stock plus a second call, chosen so the book is delta-neutral and either vega-neutral or gamma-neutral. Both are run in both worlds. In the BS world there is no volatility risk, so any gain there is from gamma. In the Heston world, a gamma-neutral hedge still carries vega. Deep out of the money both options' Greeks are ~1e-4 and the numerical values can even be slightly negative, so their ratio is noise; where the hedge option has no usable vega (or gamma) the book holds no option, and the script reports how often (about 1% of path-days).

**Costs.** Costs are close to a fixed charge, so they mostly move the mean. The cost table reports mean and standard deviation next to RMSE (RMSE² = mean² + std²) so a predictable cost is not read as extra risk.

**Model risk with jumps (Part 3, extension).** The market follows Bates. Two hedgers trade at the same market prices on the same paths:

- **Calibrated Heston hedger:** sees only option prices. It calibrates Heston once, at t = 0, to implied vols at 9 strikes × 4 maturities. Each day it backs out its variance state from the market price of the hedge option (`implied_variance`: Newton steps safeguarded by bisection on a bracket [0, 1], since the price increases with v). When no v ≥ 0 reprices that option, it uses v = 0, and the script reports how often this happens. Because the root is bracketed, a remaining mismatch means no variance fits, not that the solver failed.
- **Oracle:** knows the true Bates dynamics and parameters and reads the true variance. It is an idealised benchmark, not a feasible trader. Perfect model knowledge pays off only together with the right objective: the oracle's delta-vega hedge is not optimal under jumps, so the calibrated hedger can beat it; the jump-aware minimum-variance oracle is the true benchmark. Its variance input is not itself privileged: under the true model the hedge option's price determines v wherever that option has vega, and the script checks that inverting it recovers the true variance on those path-days. Deep out of the money the price hardly depends on v, so v is not identified there; but then neither option has vega and the hedges hardly depend on v either. The script reports the share of such path-days. Its real advantage is knowing the dynamics, in particular the jumps, which vanilla prices on one date did not reveal.

The oracle hedges two ways:

- **Greek-neutral:** zero delta and vega.
- **Minimum-variance:** each day it chooses stock *a* and options *q* to minimise the local variance of call − a·stock − q·option. The covariance per unit time between instruments is the Itô diffusion term plus λ·E[ΔX·ΔY], where ΔX is the exact repriced change after a jump. The expectation over jump size uses 7-point Gauss–Hermite quadrature. Under the pricing measure the hedge error is a martingale with orthogonal increments, so minimising each day's variance minimises the variance at expiry (up to discrete-time effects). With no jumps it reduces to the minimum-variance delta and the delta-vega hedge, to machine precision. Where the hedge option has no usable vega the 2×2 system is near-singular, and the oracle then hedges with stock only.

**Decomposition (Part 3).** Paired differences answer two separate questions: the value of knowing the true model for a fixed objective (calibrated vs oracle, delta-vega and stock-only min-var), and the value of the jump-aware objective for a fixed, true model (oracle delta-vega vs oracle min-var + option).

The jump analysis splits paths by whether a jump actually occurred in the simulation.

## Results

### Part 1: validation and smiles (`01_pricing.py`)

![smiles](figures/01_smiles.png)

| Model, T = 1 | Strike | COS | Monte Carlo | z-score |
|---|---:|---:|---:|---:|
| Heston | 80 / 100 / 120 | 25.045 / 10.394 / 2.333 | 25.078 / 10.414 / 2.343 | 1.4 / 1.2 / 1.2 |
| Bates | 80 / 100 / 120 | 25.037 / 10.382 / 2.248 | 24.995 / 10.345 / 2.236 | −1.7 / −2.2 / −1.5 |

500,000 paths. Forward check z: Heston 1.5, Bates −1.1 (the strike z-scores share paths, so this is one test per model; the common shift across strikes points to sampling noise in the level of S_T rather than a bias that grows with strike).

Skew (80-strike minus 120-strike implied vol) at 3 months: Heston 8.4, Bates 10.6 vol points. At 1 year: Heston 5.7, Bates 6.0. Jumps steepen the short-dated skew, as expected.

### Part 2: hedging in the Heston world (`02_heston_hedging.py`)

![hedge comparison](figures/02_hedge_comparison.png)

| Daily hedge | RMSE | 95% CI | Mean | ES 5% |
|---|---:|---:|---:|---:|
| BS world, BS delta | 0.446 | [0.431, 0.458] | −0.003 | 1.016 |
| BS world, delta-gamma | 0.058 | [0.051, 0.065] | +0.002 | 0.104 |
| BS world, delta-vega | 0.255 | [0.244, 0.267] | +0.001 | 0.621 |
| Heston world, BS delta (20%) | 1.544 | [1.500, 1.588] | −0.019 | 3.946 |
| Heston world, BS delta (today's implied vol) | 1.471 | [1.435, 1.505] | −0.026 | 3.516 |
| Heston world, Heston delta | 1.890 | [1.852, 1.930] | −0.024 | 4.349 |
| Heston world, min-var delta | 1.369 | [1.335, 1.402] | −0.029 | 3.278 |
| Heston world, delta-gamma | 0.885 | [0.855, 0.917] | +0.001 | 1.862 |
| Heston world, delta-vega | 0.228 | [0.213, 0.241] | −0.004 | 0.608 |

Hedge ratios at t = 0: BS delta 0.637, BS delta at implied vol 0.637, Heston delta 0.692, min-var delta 0.610.

| Paired difference in RMSE (Heston world) | Difference | 95% CI |
|---|---:|---:|
| Heston delta − BS delta (20%) | +0.346 | [+0.309, +0.381] |
| BS delta (implied) − BS delta (20%) | −0.074 | [−0.101, −0.048] |
| Min-var delta − BS delta (implied) | −0.102 | [−0.116, −0.088] |
| Delta-vega − min-var delta | −1.141 | [−1.177, −1.109] |
| Delta-gamma − min-var delta | −0.484 | [−0.523, −0.442] |

| Stock-only hedge, Heston world | Continuous-time, Itô (95% CI) | Daily, simulated | Floor share of variance |
|---|---:|---:|---:|
| BS delta (20%) | 1.485 [1.475, 1.495] | 1.544 | 77% |
| BS delta (implied) | 1.401 [1.393, 1.410] | 1.471 | 87% |
| Heston delta | 1.829 [1.818, 1.841] | 1.890 | 51% |
| Min-var delta | 1.306 [1.299, 1.315] | 1.369 | 100% |

Interpretation: the Heston delta ∂C/∂S over-hedges (0.69 vs 0.61 at t = 0). With ρ = −0.7 a fall in the stock tends to come with a rise in variance, which raises the call's value and offsets part of the stock move; the min-var delta builds this in through the ρξ𝒱/S term, the Heston delta ignores it. A BS delta happens to sit closer to the min-var delta, so it does better than the "correct" model's delta. Daily rebalancing adds roughly the same error (≈0.41–0.48) in both worlds, so the extra error in the Heston world is almost entirely continuous-time volatility risk, not discretisation.

![rebalancing](figures/02_rebalancing.png)

| RMSE by rebalances per year | 4 | 12 | 26 | 51 | 126 | 252 |
|---|---:|---:|---:|---:|---:|---:|
| BS world, BS delta | 3.230 | 1.991 | 1.375 | 0.970 | 0.624 | 0.446 |
| Heston world, BS delta (20%) | 3.820 | 2.569 | 2.065 | 1.803 | 1.599 | 1.544 |
| Heston world, BS delta (implied) | 3.755 | 2.526 | 2.018 | 1.742 | 1.536 | 1.471 |
| Heston world, min-var delta | 3.638 | 2.422 | 1.916 | 1.638 | 1.436 | 1.369 |
| Heston world, delta-vega | 1.135 | 0.851 | 0.621 | 0.450 | 0.321 | 0.228 |

Fitted log-log slope in the BS world over 51–252 rebalances: −0.49 (theory −0.5, an asymptotic rate, so the coarse frequencies are left out of the fit).

| With costs (2 bp stock, 1% option price, incl. unwind) | Rebalance | RMSE | Mean | Std | Mean cost |
|---|---:|---:|---:|---:|---:|
| BS delta (20%) | daily / weekly | 1.564 / 1.810 | −0.141 / −0.124 | 1.558 / 1.806 | 0.121 / 0.070 |
| BS delta (implied) | daily / weekly | 1.482 / 1.746 | −0.150 / −0.124 | 1.475 / 1.741 | 0.123 / 0.071 |
| Min-var delta | daily / weekly | 1.382 / 1.645 | −0.153 / −0.126 | 1.374 / 1.640 | 0.123 / 0.071 |
| Delta-vega | daily / weekly | 0.763 / 0.641 | −0.642 / −0.415 | 0.413 / 0.489 | 0.638 / 0.399 |

![pnl distributions](figures/02_pnl_distributions.png)

### Part 3 (extension): model risk in a jump market (`03_model_risk.py`)

![calibration](figures/03_calibration.png)

Calibrated Heston: v0 = 0.0394, θ = 0.0399, κ = 3.786, ξ = 0.456, ρ = −0.707 (true diffusion part: v0 = θ = 0.032, κ = 2, ξ = 0.3, ρ = −0.7, plus jumps). Fit RMSE 0.076 vol points (worst maturity T = 0.5, 0.112). The calibration absorbs the jumps into a higher variance level and a higher vol-of-vol. Feller condition holds (0.302 > 0.208).

Variance inversion: no v in [0, 1] fits the hedge option on 5.94% of path-days (48.9% of paths), always because the model is too expensive even at v = 0; mean mismatch 0.085 when it fails, on days with true variance around 0.0039 (vs 0.032 overall). Implied vs true variance: correlation 0.994, mean 0.041 vs 0.032. Oracle variance recovered from prices to within 7.5e-9 where the hedge option's variance vega exceeds 0.1; not identified on 0.83% of sampled path-days. No option held: oracle 0.29%, calibrated 1.42% of path-days (0.03% on inversion-failure days, when the calibrated model believes v = 0).

![model risk](figures/03_model_risk.png)

| Daily hedge in the Bates market | RMSE | 95% CI | Mean | ES 5% |
|---|---:|---:|---:|---:|
| BS delta, fixed ATM implied vol (19.8%) | 2.247 | [2.126, 2.381] | +0.012 | 6.836 |
| BS delta, today's implied vol | 2.181 | [2.062, 2.315] | +0.004 | 6.520 |
| Calibrated Heston delta | 2.656 | [2.545, 2.781] | +0.001 | 7.571 |
| Calibrated Heston min-var delta | 2.053 | [1.930, 2.190] | +0.005 | 6.108 |
| Calibrated Heston delta-vega | 0.839 | [0.626, 1.099] | +0.011 | 2.142 |
| Oracle min-var delta | 1.957 | [1.859, 2.068] | +0.003 | 5.412 |
| Oracle delta-vega | 0.920 | [0.724, 1.166] | +0.014 | 2.616 |
| Oracle min-var + option | 0.452 | [0.425, 0.484] | −0.003 | 1.021 |

| Paired difference in RMSE (first − second) | Difference | 95% CI |
|---|---:|---:|
| *Model knowledge, objective fixed* | | |
| Calibrated min-var delta − oracle min-var delta | +0.095 | [+0.061, +0.134] |
| Calibrated delta-vega − oracle delta-vega | −0.081 | [−0.112, −0.059] |
| *Objective, true model known* | | |
| Oracle delta-vega − oracle min-var + option | +0.468 | [+0.276, +0.707] |
| Oracle min-var delta − oracle min-var + option | +1.505 | [+1.410, +1.613] |
| *Both together* | | |
| Calibrated delta-vega − oracle min-var + option | +0.387 | [+0.180, +0.639] |

| By actual jumps (33.4% of paths had one; expected 33.0%) | RMSE, no jump | RMSE, ≥ 1 jump |
|---|---:|---:|
| Calibrated Heston min-var delta | 1.424 | 2.927 |
| Oracle min-var delta | 1.475 | 2.670 |
| Calibrated Heston delta-vega | 0.294 | 1.391 |
| Oracle delta-vega | 0.317 | 1.528 |
| Oracle min-var + option | 0.390 | 0.554 |

![tails](figures/03_pnl_tails.png)

## Limitations

- **Simulated markets only:** one parameter set per world, no real data. The Bates market stands in for reality but is still a model.
- **Model prices:** options trade at model prices, and costs are a simple proportional charge. Rebalancing is on a fixed schedule.
- **Measure:** the minimum-variance hedges minimise variance under the pricing measure. With risk premia on volatility or jumps, the real-world optimal hedge would differ.
- **Continuous-time benchmark:** it is estimated on the daily grid along simulated paths, so it carries Monte Carlo error and a small discretisation error.
- **Calibration:** the calibrated hedger calibrates once and never recalibrates. Daily recalibration would change its results, in either direction.
- **Oracle:** the oracle is an idealised benchmark, not an achievable strategy. Knowing the true model only guarantees the best result when it is combined with the right objective: the jump-aware minimum-variance oracle is the benchmark, while the oracle's delta-vega hedge is not optimal under jumps and is in fact slightly beaten by the calibrated Heston delta-vega hedge (−0.08 [−0.11, −0.06], one seed).
- **Statistics:** jump losses make the Part 3 RMSE distribution heavy-tailed, so marginal bootstrap intervals are wide and the paired differences are the reliable comparisons. Rerun with another `--seed` to check that conclusions do not hinge on one set of paths. Bootstrap intervals measure Monte Carlo noise only, not parameter uncertainty; the calibration itself is deterministic (it fits model prices at t = 0).
- **Complete-market benchmark:** in Part 2 the hedge option trades at Heston prices in a Heston market, so delta-vega hedging is perfect in continuous time by construction. Part 3 is where it is tested against a market it does not describe.
- **Simulation scheme:** full-truncation Euler is validated against COS prices, but Andersen's QE scheme would be more accurate for larger ξ.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q                       # ~15 s
python 01_pricing.py                      # ~15 s
python 02_heston_hedging.py               # ~2 min at 4,000 paths
python 02_heston_hedging.py --paths 1000  # quick pass
python 03_model_risk.py                   # ~20 min at 4,000 paths (about 0.3 s per path)
python 03_model_risk.py --paths 1000      # ~5 min, for a quick pass
python 03_model_risk.py --seed 8          # robustness check on a second set of paths
```

Run the scripts from this folder; figures are written to `figures/` and everything each script prints is saved to `results/` (runs with non-default `--paths` or `--seed` get their own file). All seeds are fixed. The scripts use `# %%` cells, so they can also be run cell by cell in VS Code or Spyder.

```
sv.py                  model code: Black-Scholes, Heston/Bates COS pricer, simulation, hedge P&L,
                       jump-aware minimum-variance hedge
01_pricing.py          Part 1: pricer vs Monte Carlo, implied-vol smiles
02_heston_hedging.py   Part 2: hedging in the Heston world, continuous-time benchmark
03_model_risk.py       Part 3: calibrated Heston vs oracle hedging in a Bates market
test_sv.py             numerical, accounting and hedge-ratio tests
results/               console output of each script
figures/               figures used in this README
```

## References

- Heston (1993), A closed-form solution for options with stochastic volatility, *Review of Financial Studies*.
- Bates (1996), Jumps and stochastic volatility, *Review of Financial Studies*.
- Albrecher, Mayer, Schoutens & Tistaert (2007), The little Heston trap, *Wilmott*.
- Fang & Oosterlee (2008), A novel pricing method for European options based on Fourier-cosine series expansions, *SIAM J. Sci. Comput.*
- Andersen (2008), Simple and efficient simulation of the Heston stochastic volatility model, *Journal of Computational Finance*.
- Schweizer (1991), Option hedging for semimartingales, *Stochastic Processes and their Applications*.
- Gatheral (2006), *The Volatility Surface*.
