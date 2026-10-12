# Overnight-indexed cashflows alongside Libor — design

Date: 2026-10-11. Branch: `ois-cashflows` (off `main` after the LGCM calibration merge, #1).

## Goal

Since the Libor transition, GBP, JPY, CHF and USD swaps and cross-currency swaps pay compounded
overnight (RFR) coupons: SONIA, TONA, SOFR, €STR. The notes and the code currently model
Libor-type coupons (fixed at the start of the period) for every non-native cashflow. Extend both
to compounded overnight coupons alongside Libor, integrated throughout the notes, with:

- the **exact** model of daily compounding, and
- **continuous compounding** as an approximation, with its error order measured.

Add example markets GBPUSD (SONIA vs SOFR) and USDJPY (TONA vs SOFR, FX quoted inverted), and
keep EURUSD (EURIBOR 3M).

## Coupon model

Accrual period [a, b], overnight grid a = t_0 < t_1 < ... < t_n = b with steps delta_k (the
model's time axis; day counts and calendars are out of scope: a uniform step delta = 1/365 by
default, the last step shortened to land on b).

| coupon | factor C paid at b (coupon = C - 1, plus spread theta s) | fixing |
|---|---|---|
| Libor (IBOR) | kappa / Z^{e\|e}_{a,b} | at a |
| daily compounded RFR (exact) | prod_k 1 / Z^{e\|e}_{t_k, t_{k+1}} | each day |
| continuously compounded RFR (approximation) | exp(int_a^b r^{e\|e}_u du) | continuously |

The overnight fixing is the one-step simple rate of the native curve, 1 + delta_k R_k = 1 / Z^{e|e}_{t_k,t_{k+1}}.
The spread is added simply (spread-exclusive compounding, the ISDA convention for RFR swaps and
cross-currency swaps). Payment lags, lookbacks and lockouts are out of scope.

Libor is the one-step case (n = 1) times the funding spread kappa; so one formula covers the three
coupons. Under native collateral, all three are worth Z^{e|e}_{0,a} kappa - Z^{e|e}_{0,b} (kappa = 1 for RFR): model-free,
so native swaps are unchanged.

## Collateral convexity of a compounded coupon (new theory)

Foreign coupon, domestic collateral, paid at b. Value at t <= a:
V_t = Z^{f|d}_{t,b} E^{f|d,b}_t[C]. Define K^C_t(a, b) = E^{f|d,b}_t[C] / (C-forward under native
collateral). In the LGCM ln C is affine in the Gaussian state, and the change of measure from
Q^{f|f,b} to Q^{f|d,b} is driven by Sigma^c_{.,b}, so K^C = exp(-Cov(ln C, int Sigma^c_{u,b} dW_u))
(to be proved in the notes, with the exact covariance kernel). Expected forms (sign conventions to
be verified against `market.libor_convexity`):

- Libor: ln K = -int_t^a (Sigma^f_{u,b} - Sigma^f_{u,a}) . Sigma^c_{u,b} du (existing formula);
- daily: ln K = -sum_k int_t^{t_k} (Sigma^f_{u,t_{k+1}} - Sigma^f_{u,t_k}) . Sigma^c_{u,b} du;
- continuous: ln K = -[int_t^a (Sigma^f_{u,b} - Sigma^f_{u,a}) . Sigma^c_{u,b} du + int_a^b Sigma^f_{u,b} . Sigma^c_{u,b} du]
  (the second term is the accrual-period, in-arrears covariance absent from Libor).

Also: valuation inside a partially accrued period (a < t < b, the accrued part known at t); the
error of the continuous approximation versus daily compounding (order in delta, measured
numerically, not asserted); the size of the in-arrears term versus the Libor term.

## Products affected

- Resetting CCS (foreign RFR + spread vs domestic RFR on a reset notional, domestic collateral):
  spread B with K^C_j in place of the Libor K_j. The domestic resetting leg stays model-free.
- Forward-starting resetting CB spread and its vol: Gaussian approximation (loadings with the
  compounded coupon) and exact 2D quadrature (state functions at T0 including partially accrued
  periods if T0 is inside a period — by construction T0 is a period start, so not needed).
- Non-native swap rate Y^{f|d} with RFR coupons (Gaussian approximation and exact quadrature;
  `payoffs.SwapPayoff` for "fd" swaps).
- Native RFR swaps and swaptions: unchanged (model-free curves; the existing exact one-factor
  swaption covers OIS coupons since the float leg telescopes).

## Code changes (package `python/convexity/`)

- `curves.Index`: `compounding` in {"daily", "continuous"} for kind "OIS" (default "daily"),
  `on_step` (default 1/365). IBOR unchanged.
- `market.py`: `compounded_convexity(m, t, a, b, compounding, step)` (closed form, LGCM) next to
  `libor_convexity`; a single dispatcher `coupon_convexity(m, t, index, a, b)`.
- `pricers.Pricer`: `coupon_factor(index, collateral, a, b)` = forward value of C (native: Z ratio
  times kappa; non-native: times K^C). `forward_rate` becomes a thin wrapper. `LGCMPricer` caches
  per coupon type. Fixes the current behaviour where an OIS foreign index under domestic
  collateral silently gets the Libor K.
- `instruments.ResettingCCS`, `gaussian_approx` (`_ingredients`, `_rates`, `approx`, `exact`),
  `payoffs.SwapPayoff.convexities`: take the coupon type from the index.
- FX quotation inversion: `FXForward(..., inverted=False)` quoting 1/x (outright or points) and
  `MarketQuotes.fx_spot_quote` / `fx_inverted` for the spot; `FXOption` vol unchanged (ln x
  Gaussian: the ATM Black vol of 1/x equals that of x; proved in the notes).
- `market_data.example_quotes(market="EURUSD" | "GBPUSD" | "USDJPY")`: rounded illustrative
  quotes (not live data); GBPUSD and USDJPY with RFR swaps and RFR resetting CCS.
- `calibration_check.mc_report`: compounded coupons valued on daily paths (`simulate_dates` over
  each accrual period); continuous coupons through the daily product as delta -> 0 (no new
  simulator state).
- Notebook: market selector; Libor vs daily vs continuous convexity on the same market; impact on
  calibrated Z^{f|d}.

## Notes (integrated throughout `docs/CollateralDiscounting.tex`)

Generalise in place, keeping labels stable where possible:
- the convexity adjustment to forward rates (HJM framework section and deterministic-volatility
  section): Libor, daily-compounded and continuous coupons; exact vs approximation with the
  measured error order;
- currency basis swaplets and spreads (resetting CCS with RFR legs);
- Gaussian par spread approximations (loadings for compounded coupons);
- the calibration section: instrument definitions, CCS formula with K^C, FX inversion, the three
  example markets, verification remarks with measured numbers;
- a remark on native vs non-native: compounded coupons are model-free under native collateral.

Every statement labelled; exact vs asymptotic vs numerical observation distinguished; numbers
measured. Compiles with 0 errors and only the three known warnings.

## Verification and tests

- K^C closed forms vs an independent benchmark: brute-force Gaussian expectation from the joint law
  of the state on the daily grid (transitions / moment ODEs) and Monte Carlo (4 s.e.).
- Libor = one-step case; native telescoping (model-free) for all coupon types.
- Continuous vs daily: convergence in delta, order measured.
- CCS spread and CB vol with RFR coupons: analytic vs exact quadrature vs MC.
- Calibration round trip on EURUSD, GBPUSD and USDJPY; inverted FX quotes give the same curve as
  direct quotes; MC repricing of all quotes within 4 s.e. with negative controls (Libor K used for an
  RFR CCS is detected).
- Existing 111 tests keep passing.

## Out of scope

Payment lags, lookbacks, lockouts, observation shifts, calendars and day counts; term RFR rates;
non-resetting CCS calibration; multi-factor f or c blocks for CB vols.
