# LGCM calibration — design

Date: 2026-10-09. Branch: `lgcm-calibration`.

## Goal

Calibrate the full Linear Gaussian Collateral Model (domestic d = collateral currency, foreign f) to
market quotes, document the procedure in `docs/CollateralDiscounting.tex`, and verify that the
calibrated model delivers the input quotes, by closed forms and by Monte Carlo. Deliver tests, a
marimo notebook on calibration results, parameter impact and collateral convexity, plus
`requirements.txt` and `environment.yml`.

Inputs: swap rates (OIS or Libor) in d and f, FX spot, FX forwards (outrights or points), resetting
currency basis (CB) swap spreads, ATM swaption normal vols in d and f, ATM FX option Black vols,
optional CB spread normal vols (e.g. historical).

The discount curve calibration must also run on its own with a deterministic model (no convexity
adjustments, no option targets).

## Conventions

- **Index per currency**: `OIS` (kappa = 1) or `Libor` with the model's constant multiplicative
  funding spread, kappa = 1 + theta * spread (marked). Floating leg periods use the index tenor
  theta; `1 + theta L_j = kappa Z_{j-1}/Z_j` (native collateral).
- **Swap quotes**: spot-starting, par rate = float leg / fixed annuity, fixed leg frequency of its
  own, native collateral.
- **FX forwards**: d-collateralised, outright or points (`outright = spot + points / pip_factor`).
  Model-free: `x^{df|d}_{0,T} = x_0 Z^{f|d}_{0,T} / Z^{d|d}_{0,T}`. Spot lag ignored (spot date 0).
- **Resetting CB swaps**: spot-starting, period theta (= f index tenor, and d index tenor if d is
  Libor), spread `B^{f/d|d,r}_{0,0;N} = sum_j P_j (one_Lr_j - one_L_j) / A`, with
  `P_j = Z^{f|d}_{0,T_j}`, `one_Lr_j = kappa_d P_{j-1}/P_j`,
  `one_L_j = kappa_f Z^{f|f}_{j-1}/Z^{f|f}_j * K_j`, `A = theta sum P_j`, and `K_j` the Libor
  collateral convexity (1 in a deterministic model).
- **Swaptions**: ATM (strike = forward) normal vol of a native swap, expiry T0, tenor, fixed and float
  frequency equal to the index tenor.
- **FX options**: ATM Black vol of the FX spot at expiry. The LGCM total variance does not depend on
  the collateral, so the vol quote is collateral free; ATM is the d-collateral forward.
- **CB vols**: ATM normal vol of the forward resetting CB spread `B_{T0;tenor}` under its annuity
  measure.

## Architecture

New modules in `python/collateraldiscounting/`:

| module | responsibility |
|---|---|
| `curves.py` | `PiecewiseFlatCurve` (log discount factor linear between pillars, flat extrapolation of the last forward), `MarketCurves` with the `InitialCurves` interface (`x0`, `kappa_d`, `kappa_f`, `kappa(ccy)`, `Z0(kind, T)` for dd/ff/fd/qz, `fx_forward0`), built from Z^{d\|d}, Z^{f\|f} and the collateral basis curve Q = Z^{f\|d}/Z^{f\|f} |
| `quotes.py` | quote dataclasses (`IndexSpec`, `SwapQuote`, `FXForwardQuote`, `CBSwapQuote`, `SwaptionQuote`, `FXVolQuote`, `CBVolQuote`), the `MarketQuotes` container, `example_quotes()` (realistic USD/EUR market) |
| `instruments.py` | model values of every quote type: `swap_rate`, `fx_forward`, `cb_spread`, swaption exact price and implied normal vol (1 factor, closed form), generalised Gaussian swaption variance (any number of factors, fixed and float schedules), `fx_atm_vol`, CB vol (approx and exact) |
| `curve_calibration.py` | `ConvexityProvider` protocol (`libor_convexity(a, b)` = K seen from 0), `DeterministicConvexity` (K = 1, the DCB model), `LGCMConvexity(model)`; `bootstrap_native_curve(quotes, index)`, `bootstrap_collateral_basis_curve(curves, fx_forwards, cb_swaps, convexity)`; `calibrate_curves(quotes, convexity=DeterministicConvexity())` |
| `calibration.py` | `ModelSpec` (factor structure per block: mean reversions and fixed weights, correlation matrix, marked sigma^c), `calibrate(quotes, spec) -> Calibration` (model, curves, diagnostics), the bootstrapping steps, `CalibrationError` |
| `calibration_check.py` | `analytic_report(cal, quotes)` and `mc_report(cal, quotes, n_paths, seed)`: rows (instrument, market, model, error, stderr) |

Changes to existing modules:

- `schemes.py`: `ExactScheme.simulate_dates(sim, dates, ...)` returning the state at every date of a
  grid in one pass (transitions computed once).
- `montecarlo.py`: `Simulator.simulate_dates`.
- `payoffs.py`: `CurrencyBasisSwaption` (value `A (omega (B - K))^+` in f under d collateral).

`InitialCurves` stays as it is (the existing tests use it); `MarketCurves` is duck-type compatible.

### Curve calibration abstraction

The curve bootstrap only sees the model through a `ConvexityProvider`. The native curves are
model-free. The collateral basis curve takes the FX forwards (model-free) up to the last forward
pillar, then each CB swap pillar beyond it fixes the level of the flat q-forward on its segment by a
1D root search, with `K_j` from the provider. `calibrate_curves(quotes)` with the default
`DeterministicConvexity` is the deterministic collateral basis (DCB) market practice; the LGCM passes
`LGCMConvexity(model)`.

## Calibration algorithm

Model: blocks d, f, c with `n_d, n_f, n_c` factors (1 by default) and the FX driver. Mean reversions
and the full correlation matrix are marked. Within a block, the spot vols are
`sigma^{a,i}_t = lambda^a_t w^{a,i}` with fixed weights `w`; the scalar `lambda^a` is piecewise
constant on the block's calibration expiries and bootstrapped. The model knots are the union of all
expiries; flat extrapolation after the last expiry of each block.

1. **Native curves**: bootstrap Z^{d\|d}, Z^{f\|f} from swap quotes (sequential, exact).
2. **lambda^d, lambda^f**: per swaption expiry, solve on the segment so that the model ATM normal
   vol matches. Initial guess from the Gaussian approximation; the segment variance is quadratic in
   lambda so the guess is closed form. Refinement by brentq on the exact closed-form price for one
   factor; with several factors the Gaussian approximation is the pricer (flagged in diagnostics).
   Raise `CalibrationError` on a variance squeeze.
3. **(Q, lambda^c) fixed point**: (a) bootstrap Q with `LGCMConvexity`; (b) if CB vols are given,
   bootstrap lambda^c per CB vol expiry (Gaussian approximation guess, exact 2D quadrature refinement
   when n_f = n_c = 1), else keep the marked sigma^c; repeat until the change in lambda^c and Q is
   below tolerance. Iteration count is reported.
4. **lambda^x**: per FX expiry, the FX total variance is exactly quadratic in sigma^x on the segment;
   solve in closed form, `CalibrationError` if no non-negative root (variance squeeze).

## Verification

- **Analytic**: reprice every quote with `instruments.py`. Curve quotes and FX vols exact (< 1e-10);
  swaption vols exact for 1 factor; CB vols exact (2D quadrature) with the approximation gap
  reported.
- **Monte Carlo** (exact LGCM scheme, antithetic, fixed seeds): swap rates, FX forwards, CB spreads
  (f Libor leg under d collateral valued at each fixing by `simulate_dates`), swaption, FX option
  and CB swaption implied vols. Control variates (known discount factor means) where needed. Pass if
  within 4 standard errors.

## Tests (`tests/test_calibration.py`)

- Native and collateral basis curves reprice their quotes (deterministic and LGCM convexity).
- Round trip: quotes generated by a known model are calibrated back to its parameters.
- Swaption closed form vs numerical quadrature; Gaussian approximation converges to it as vol -> 0.
- Generalised Gaussian loadings equal `gaussian_approx.approx_native` for equal schedules.
- DCB limit: sigma^c = 0 gives K = 1 and the deterministic curve.
- Variance squeeze raises `CalibrationError`.
- Multi-factor block (2 d factors) calibrates to its swaption targets (Gaussian approximation).
- Reduced Monte Carlo repricing of all quote types within 4 s.e.

## Notes (`docs/CollateralDiscounting.tex`)

Rewrite "The Linear Gaussian Collateral Model: Calibration" in place: quotes and conventions;
curve bootstrap with a convexity provider (DCB as the special case); swaption calibration
(Proposition: exact one factor price; Gaussian guess); model-free FX forward curve; joint CB
spread / sigma^c bootstrap (Algorithm, with the fixed point and its measured iteration count); FX
vol constraint (Proposition: quadratic in sigma^x, variance squeeze condition). Replace the blank
theorem items. Verification remark with measured repricing errors (analytic and MC) and a remark
quantifying collateral convexity (LGCM vs DCB curves calibrated to the same quotes). Must compile
with no new errors or warnings.

## Notebook and environment

- `python/notebooks/lgcm_calibration.py` (marimo): quotes, calibrated curves and vol term
  structures, analytic and MC repricing tables, sliders (rho^{fc}, sigma^c scale, mean reversions)
  with live recalibration, collateral convexity panels (K_j, FX forward under f vs d collateral,
  Z^{d\|f}, Z^{f\|d} shift vs DCB).
- `requirements.txt`, `environment.yml` (repo root): python 3.12, numpy, scipy, sympy, pytest,
  matplotlib, marimo.

## Out of scope

Non-native swaptions, smiles, spot lag and calendars, calibration of correlations or mean
reversions, Euler/QGCM calibration.
