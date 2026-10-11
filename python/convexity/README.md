# Linear Gaussian Collateral Model — numerical companion

Checks of the derivations in `docs/CollateralDiscounting.tex` (Quasi-Gaussian and Linear Gaussian
Collateral Models), and the calibration of the LGCM to market quotes.

Environment (from the repository root): `conda env create -f environment.yml` (Python 3.12), or
`pip install -r requirements.txt` in an existing environment. Run the tests from `python/`:

```
python -m pytest convexity -q                  # full suite (~1 min)
python -m pytest convexity -q -m "not slow"    # without the Monte Carlo repricing tests
```

| file | contents |
|---|---|
| `lgcm.py` | piecewise-constant LGCM (blocks d, f, c = f→d, x); closed-form covariance integrals `covint` and a brute-force quadrature benchmark; `y_t`, `v^{f,c}_t`; ZCB reconstruction exponents |
| `benchmarks.py` | exact Gaussian benchmark of the coupon collateral convexity (`coupon_value_exact`: state augmentation of the exact transitions, independent of the closed forms) |
| `transitions.py` | exact one-step transitions of the state under Q^{d,RN}, the T-forward measure and the jumping numeraire measure |
| `moments.py` | moment ODEs of the state SDE — an independent benchmark that does not use the covariance integrals |
| `market.py` | `InitialCurves` (Z^{d\|d}_0, Z^{f\|f}_0, Z^{f\|d}_0, x_0, funding spreads) coupon types `Coupon` (Libor, daily or continuously compounded overnight; `coupon_grid`) and the LGCM closed forms: Libor, compounded-coupon (`compounded_convexity`) and FX collateral convexities, Z^{d\|f}_0, FX total variance, Black and Bachelier |
| `schemes.py` | `MCScheme` base class and its schemes: `EulerScheme(dt, y_mode)` for the QGCM with stochastic local vol and simulated `y_t`, `v^{f,c}_t`; `ExactScheme(dt)` with exact Gaussian transitions for the LGCM |
| `montecarlo.py` | `Simulator(model, kappa, floor, scheme)`: QGCM local vol, reconstruction formulas and deflators; path generation is delegated to the scheme. Vectorised over paths, antithetic |
| `payoffs.py` | `Payoff` hierarchy and `price(sim, payoffs, curves, n_paths, seed)` |
| `mc_verification.py` | Monte Carlo verification of the approximations; `python -m convexity.mc_verification` prints the full report (~2.5 min) |
| `gaussian_approx.py` | Gaussian approximations of the forward swap rate Y^{f\|d} and the resetting currency basis spread B^{f/d\|d,r}, and their exact benchmark by 2D Gauss–Hermite quadrature |
| `tests/test_lgcm.py` | the checks quoted in the verification remarks of the notes |
| `tests/test_montecarlo.py` | reduced Monte Carlo checks (4 standard error tolerance, fixed seeds) |
| `tests/test_payoffs.py` | Monte Carlo prices of all payoffs against LGCM closed forms, model-free prices in the QGCM, put-call parities |
| `curves.py` | `Index` (OIS with daily or continuous compounding, or IBOR with funding spread, kappa = 1 + theta s; `.coupon`), curve keys `DiscountCurveKey(currency, collateral)` / `ForwardCurveKey`, `PiecewiseFlatCurve`, `SpreadCurve` (Z^{f\|d} = Z^{f\|f} x collateral basis curve), `FunctionCurve`, `LGCMSetup` (roles d, f and collateral indices), `CurveSet` (duck-type compatible with `InitialCurves`) |
| `instruments.py` | calibration instruments (pure data): `Swap`, `FXForward`, `ResettingCCS`, `Swaption`, `FXOption`, `CurrencyBasisSwaption`; `Quote`, `MarketQuotes` |
| `pricers.py` | `DeterministicPricer` (DCB: K = 1, no options), `LGCMPricer` (Libor convexities, exact one factor swaption, Gaussian swaption variance with separate fixed/float schedules, FX vol, CB vol exact and Gaussian) |
| `curve_calibration.py` | generic sequential curve bootstrap with a convexity provider; `calibrate_curves(quotes)` is the DCB market practice |
| `calibration.py` | `ModelSpec`, `BlockSpec`, `default_spec`, `calibrate(quotes, spec) -> Calibration`: Algorithm "LGCM calibration" of the notes |
| `market_data.py` | `example_setup()`, `example_quotes(market)` for `MARKETS` = EURUSD (USD SOFR / EUR ESTR, EURIBOR 3M), GBPUSD, USDJPY (inverted quotation): illustrative, not live data |
| `calibration_check.py` | `analytic_report(cal)` and `mc_report(cal, n_paths, seed, compounding_step=None)`: every quote repriced in closed form and by Monte Carlo (rows with market, model, error, standard error); compounded RFR coupons valued on paths by `CouponPaths` (fixing dates streamed, exact controls), `coupon_convexity_mc` for one coupon |
| `schemes.py` (`ExactScheme.simulate_dates`), `montecarlo.py` (`Simulator.simulate_dates`) | the LGCM state at every date of a grid along the same paths, in one pass; `stream` dates are passed to `on_date` without being stored |
| `tests/test_ois.py` | compounded RFR coupons: closed forms against the exact benchmark, coupon-aware curves, pricers and calibration, the three markets, inverted FX, Monte Carlo of compounded coupons |
| `tests/test_curves.py` | curves, instruments and the deterministic curve bootstrap |
| `tests/test_calibration.py` | LGCM pricer, calibration and repricing checks (the Monte Carlo ones are marked `slow`) |
| `repo.py` | bonds and repo (McCloud, *Repo convexity*): `RepoModel` (LGCM with block f = bond curve, c = discounting basis, no d factor, zero FX vol), `BondCurves`; `repo_convexity` (ln K^{repo}), `maturity_adjustment`, `forwardness_adjustment`, `repo_rate` (ACT/360 or ACT/365 accrual), `repo_discount_factor`; McCloud's Hull-White closed forms `hw_*`; `CouponBond`, `coupon_repo_rate` (frozen weights) and `coupon_repo_rate_exact` (Gauss–Hermite); `mc_repo_rate` |
| `tests/test_repo.py` | the checks of Remark "Verification" of section "Bonds, repo and repo convexity" |

## What is checked

| notes | test | benchmark |
|---|---|---|
| Algorithm "Covariance integrals" | `test_kernels_match_quadrature`, `test_covariance_integrals_match_brute_force` | adaptive quadrature (rel. err. < 1e-12; measured 7e-16) |
| Theorem "Quasi-Gaussian Collateral Model" | `test_qgcm_no_arbitrage_identities`, `test_qgcm_identities_detect_a_missing_term` | martingale identities for Z^{d\|d}, x Z^{f\|d}, x Z^{f\|f} from the moment ODEs (< 1e-13) |
| Libor collateral convexity (eqs. 1807/1813 corrected) | `test_libor_collateral_convexity` | exact Gaussian expectation |
| Proposition "Joint law of the state" | `test_lgcm_joint_law` | moment ODEs |
| Propositions "Exact transition" / "Change of numéraire" | `test_rn_transitions_match_moment_odes`, `test_martingale_identities_under_rn_tforward_and_jumping` | moment ODEs; martingale identities under three measures |
| Lemma on the law of ẑ under Q^{f\|d,T0} | `test_zhat_is_centred_with_covariance_yhat_under_forward_measure` | exponential tilt of the RN law |
| Gaussian approximations | `test_gaussian_approximations_and_convergence_order`, `test_single_swaplet_loadings` | exact quadrature; error O(λ²) under vol scaling λ (measured, not proved) |
| Proposition "Consistent families" | `test_consistent_exponential_families` | sympy |
| Proposition "Model-free curves", Algorithm "Curve bootstrap with a convexity provider" | `test_deterministic_bootstrap_recovers_flat_curves`, `test_fx_forward_curve_is_model_free`, `test_points_and_outrights_give_the_same_curve`, `test_points_with_other_pip_factor_give_the_same_curve`, `test_native_curves_only_from_swaps`, `test_partial_bootstrap_targets_only` | flat curves recovered exactly; Z^{f\|d} up to the last FX forward identical with and without convexity |
| Proposition "Exact one factor swaption" | `test_native_swaption_exact_closed_form_matches_quadrature`, `test_lgcm_pricer_swaption_dispatch_both_blocks` | adaptive quadrature of E[V^+] (rel. 1e-9) |
| Proposition "Gaussian swaption variance with separate fixed and floating schedules" | `test_gaussian_swaption_variance_converges_to_exact`, `test_gaussian_variance_matches_gaussian_approx_native`, `test_multifactor_gaussian_variance_vs_brute_force_and_pricer_fallback` | exact price: error O(λ²) (measured ratios 4.0002, 4.0010, not proved); `gaussian_approx.approx_native` (1e-12); quadrature loadings (1e-8) |
| Resetting CB spread with Libor convexity K_j, FX vol | `test_resetting_ccs_without_convexity`, `test_lgcm_pricer_ccs_and_cb_vol_match_gaussian_approx`, `test_lgcm_pricer_convexity_and_fx_vol` | `gaussian_approx` (independent code), `market.libor_convexity`, `fx_total_variance` |
| Algorithm "LGCM calibration" | `test_calibration_reprices_all_quotes`, `test_round_trip_recovers_parameters`, `test_input_order_does_not_matter`, `test_cb_vols_not_moved_by_later_expiries`, `test_two_factor_domestic_block` | all 65 quotes repriced (measured max errors 2e-16, FX forwards 1.4e-12 pips); round trip to 1e-8; fixed point in 5 iterations |
| DCB limit, collateral convexity of the curves | `test_dcb_limit`, `test_collateral_convexity_moves_the_long_end` | sigma^c = 0 gives the DCB curve (1e-14) |
| Variance squeeze (Proposition "Calibration constraints"), missing quotes | `test_variance_squeeze_raises`, `test_swaption_squeeze_at_later_expiry_raises`, `test_zero_cb_vol_raises_with_label`, `test_cb_vol_squeeze_at_later_expiry_raises`, `test_duplicate_expiry_raises`, `test_calibrate_without_fd_quotes_raises`, `test_marked_blocks_are_reported` | `CalibrationError` naming the instrument; `diagnostics["marked"]` |
| Remark "Verification" (analytic) | `test_analytic_report`, `test_check_row_z_without_stderr` | Gaussian gaps: swaptions 0 < approx/exact - 1 < 1e-2, CB vols < 2e-4 |
| Remark "Verification" (Monte Carlo, `slow`) | `test_mc_report_within_4_stderr`, `test_mc_report_stderr_bounded`, `test_mc_rejects_missing_collateral_convexity`, `test_mc_rejects_gaussian_swaption_approximation`, `test_cb_vol_control_has_zero_mean` | 4 s.e. with fixed seeds; negative controls: K = 1 CCS spread > 10 s.e. away (measured 19.7 / 26.1 / 35.6 at 5 / 7 / 10Y), Gaussian swaption > 4 s.e. away (7.8 / 7.0 at USD 15Y / 20Y x 10Y) |
| `simulate_dates` | `test_simulate_dates_martingales`, `test_simulate_dates_single_date_is_simulate`, `test_simulate_dates_rejects_empty`, `test_simulate_dates_time_zero_only`, `test_mc_report_ccs_with_fixings_at_time_zero_only`, `test_cv_mean_and_stderr_reduces_variance` | martingale identities; path-by-path equality with `simulate` (1e-12) |
| Proposition "LGCM repo convexity" | `test_repo_convexity_matches_quadrature`, `test_repo_to_maturity_is_libor_convexity`, `test_boundary_conditions` | nested quadrature (6e-15 rel.); T = e is the Libor collateral convexity (1e-16 absolute) |
| Proposition "Hull–White repo convexity" (McCloud (30), (33), (34), (38), (39), (41), (43)) | `test_hull_white_matches_mccloud`, `test_zero_mean_reversion_limit`, `test_hull_white_*_interpolation`, `test_hull_white_forward_spread_and_extrapolation`, `test_interpolation_fails_beyond_hull_white` | McCloud's closed forms (1e-11); polynomial limit at theta = kappa = 0; the interpolations fail in a two-factor piecewise model (implied theta 0.023 vs 0.030) |
| Proposition "Coupon repo rate", Definition and Remark "Frozen weights" | `test_single_cashflow_is_zero_coupon`, `test_zero_basis_gives_bond_forward`, `test_frozen_weights_error_order`, `test_mc_coupon_repo_rate`, `test_coupon_on_window_boundaries_exact_vs_mc` | Gauss–Hermite (converged to 1e-14); frozen weights with the control E[D/D] = 1 keep the b = 0 identity (1e-13), error O(lambda^4) (measured ratios 16.2, 16.0, not proved); exact and frozen weights both within 4 s.e. of MC at vol scale 3 |
| Proposition "Overnight repo" | `test_spot_overnight_repo_is_bond_rate`, `test_forward_overnight_repo_limit` | spot: O(h^2); forward: Hull–White closed-form limit |
| Monte Carlo of the zero-coupon repo rate | `test_mc_zero_coupon_repo_rate` | 4 s.e. (seeds 1–4, 2^18 paths); bond forward (K = 1) rejected at 7.8 to 9.8 s.e. |
| Compounded overnight (RFR) coupon convexity | `test_closed_form_matches_exact_gaussian_benchmark`, `test_partially_accrued_period`, `test_benchmark_native_coupon_is_model_free`, `test_benchmark_convexity_does_not_depend_on_the_state_at_t`, `test_vectorised_daily_convexity_matches_scalar_oracle` | `benchmarks.coupon_value_exact` (state augmentation of the exact transitions, independent code): measured max abs(Delta ln K) 4.4e-16 |
| Coupon types: Libor is the one step daily coupon, continuous is the step -> 0 limit | `test_libor_is_the_one_step_daily_coupon`, `test_coupon_grid_lands_on_b`, `test_continuous_is_the_limit_of_daily_with_order_one`, `test_compounded_and_libor_convexities_differ_by_the_accrual_term` | continuous vs daily error ratio about 2 per halving of the step: O(step), measured, not proved |
| Coupon-aware curves, pricers, swap and CB inputs | `test_index_coupon`, `test_curve_set_and_initial_curves_coupon`, `test_rfr_ccs_uses_the_compounded_convexity_not_libor`, `test_gaussian_approx_and_exact_forward_consistency`, `test_ingredients_use_the_curves_coupon`, `test_swap_convexities_pick_up_the_curves_coupon`, `test_cb_straddle_inputs_use_the_curves_coupon`, `test_initial_curves_paths_keep_libor` | the coupon of `curves.coupon("f")` reaches K0/KT, swap convexities and CB straddle inputs |
| Example markets, inverted FX | `test_calibration_reprices_every_market`, `test_rfr_markets_use_compounded_coupons`, `test_default_market_is_eurusd_and_not_inverted`, `test_inverted_forward_quote_is_reciprocal`, `test_inverted_fx_quotes_give_the_same_curve`, `test_fx_option_vol_is_invariant_under_inversion`, `test_mc_inverted_forward_se_is_propagated` | all three markets reprice (closed form); USDJPY inverted quotes give the same curve as the direct pair |
| Compounded coupons on Monte Carlo paths (`slow` where marked) | `test_simulate_dates_streams_dates_in_order_without_storing`, `test_coupon_paths_match_a_hand_computation_on_stored_states`, `test_coupon_paths_value_the_continuous_coupon_on_its_daily_grid`, `test_with_compounding_step_changes_only_the_foreign_coupon_step`, `test_mc_daily_coupon_convexity_one_period`, `test_coupon_path_control_means`, `test_mc_reprices_rfr_ccs_and_cb_vols`, `test_mc_rfr_negative_controls` | 4 s.e. with fixed seeds; 2^17 paths, seed 7, weekly step on paths: max abs(z) 2.22 (EURUSD), 3.10 (GBPUSD), 2.70 (USDJPY) |

The test model (`default_model`) has two domestic factors, one foreign factor, one collateral
basis factor and the FX factor. Its parameters are piecewise constant, with zero, 1e-7 and
negative mean reversions on some intervals, and it uses a full 5×5 correlation matrix.

## Calibration

`calibration.calibrate` implements Algorithm "LGCM calibration" of the notes: native curves from swaps;
lambda^d, lambda^f from ATM swaptions (exact one factor price); a fixed point between the Z^{f|d} curve
(FX forwards and resetting CCS with the LGCM Libor convexities) and lambda^c (CB vols, exact 2D
quadrature); sigma^x from ATM FX vols in closed form. Mean reversions, correlations and factor weights
are marked (`ModelSpec`). The calibration takes one swaption quote per expiry and currency (one tenor, not a
swaption matrix) and one CB vol and one FX vol per expiry; two quotes with the same expiry raise a
`ValueError` naming both. A block without option quotes keeps its marked level, listed in
`cal.diagnostics["marked"]`; without FX forward or resetting CCS quotes the calibration raises
`CalibrationError` (Z^{f|d} cannot be built). Run from `python/`:

```python
from convexity.market_data import example_quotes
from convexity.calibration import calibrate
from convexity.calibration_check import analytic_report, mc_report
from convexity.curve_calibration import calibrate_curves
from convexity.instruments import Swaption, FXOption, CurrencyBasisSwaption

cal = calibrate(example_quotes())                # ~5.5 s; cal.model, cal.curves, cal.diagnostics
rows = analytic_report(cal)                      # every quote repriced in closed form
rows_mc = mc_report(cal, n_paths=2**17, seed=7)  # ~6 s; each row has a standard error
cal.levels("c")                                  # knots and lambda^c

# DCB market practice: curves only, no vols, no convexity
dcb, report = calibrate_curves(example_quotes().without(Swaption, FXOption, CurrencyBasisSwaption))
```

Measured on the example market: 5 fixed point iterations, 5.5 s; max analytic repricing error 2.3e-16
(FX forwards 1.45e-12 pips); Monte Carlo (2^17 paths, seed 7) all 65 quotes within 4 s.e., max |z| = 2.22.
The 30Y Z^{f|d} zero rate of the LGCM is 3.75 bp above the DCB one calibrated to the same quotes.

The marimo notebook `python/notebooks/lgcm_calibration.py` shows the calibration results, the
repricing tables, the impact of the marked parameters and the collateral convexity. From the
repository root: `marimo edit python/notebooks/lgcm_calibration.py` (or `marimo run` for the app view).

## Coupon types and markets

A foreign index coupon is a `market.Coupon(kind, step)`, with `kind`

- `"libor"`: `kappa / Z^{f|f}_{a,b}` fixed at `a` (kappa kept outside the convexity);
- `"daily"`: exact daily compounding, prod 1/Z^{f|f}_{t_k,t_{k+1}} on `coupon_grid(a, b, step)` (default step 1/365); Libor is its one step case;
- `"continuous"`: exp(int_a^b r^{f|f}), the step -> 0 limit. It is an approximation of the daily coupon with error O(step) (measured ratio about 2 per halving of the step, not proved).

`Index(..., compounding, on_step)` sets the coupon of an OIS index; `Index.coupon`, `CurveSet.coupon(role)` and `Pricer.coupon_convexity(a, b)` expose it, and the swap, CCS and CB-vol machinery uses that coupon. The closed form is `Coupon.convexity(m, t, a, b)` (`market.compounded_convexity` for the compounded kinds); `benchmarks.coupon_value_exact` is the independent check. Libor and daily K differ by about 1e-7 relative per period, so the calibrated Z^{f|d} moves by only about -0.001 bp when switching coupon type: the in-arrears term is negligible in these markets. K - 1 reaches about 1.8e-4 (GBPUSD, USDJPY) and 2.2e-4 (EURUSD) at 30Y.

Monte Carlo cannot resolve Libor against compounded K: the difference (~1e-7 relative) is far below the standard error of the runs here. The MC rows therefore test the compounded coupon valuation on paths (exact controls) and reject K = 1 (negative controls), not the Libor / compounded distinction, which rests on the exact benchmark. `mc_report(cal, ..., compounding_step=1/52)` compounds on a weekly grid on the paths, and `with_compounding_step(cal, 1/52)` makes the model side use the same step.

`example_quotes(market)` for `market` in `MARKETS` = EURUSD (Libor-type foreign coupon), GBPUSD and USDJPY (daily compounded RFR). USDJPY is quoted inverted (`FXForward(..., inverted=True)`, `MarketQuotes.fx_inverted`): quotes are units of foreign per domestic, `MarketQuotes.x0` is the model spot (domestic per foreign). Run from `python/`:

```python
from convexity.market import Coupon, InitialCurves
from convexity.benchmarks import coupon_value_exact
from convexity.lgcm import default_model
from convexity.market_data import example_quotes, MARKETS
from convexity.calibration import calibrate
from convexity.calibration_check import mc_report, with_compounding_step

m, curves = default_model(), InitialCurves()
for c in (Coupon("libor"), Coupon("daily"), Coupon("continuous")):
    print(c.kind, c.convexity(m, 0.0, 10.0, 10.25) - 1)           # K - 1 of a 3M coupon at 10Y
print(coupon_value_exact(m, curves, Coupon("daily"), 10.0, 10.25) - 1)   # exact benchmark

for mk in MARKETS:
    q = example_quotes(mk)
    cal = calibrate(q)                                               # ~6 s
    print(mk, q.fx_inverted, q.x0, cal.curves.coupon("f").kind)

cal = calibrate(example_quotes("USDJPY"))
rows = mc_report(with_compounding_step(cal, 1/52), n_paths=2**17, seed=7, compounding_step=1/52)
```

Output (the `K - 1` lines are for `default_model`):

```
libor 0.00017963350458449767
daily 0.00017974700646350428
continuous 0.0001797479505707411
0.00017974700646350428
EURUSD False 1.1 libor
GBPUSD False 1.27 daily
USDJPY True 0.006666666666666667 daily
```

## Monte Carlo verification

QGCM volatility used for the stochastic-vol tests: a floored linear local volatility in each factor,
`sigma^{a,i}_t = sigmabar^{a,i}_t max(1 + kappa z^{a,i}_t, 0.05)`. It is separable, so the QGCM
theorem applies, and `kappa = 0` is the LGCM. Simulation is under Q^{d,RN} with the deflator
`exp(-int r^{d|d})`. Speed: 2^18 antithetic paths over 5Y take about 12 s with the Euler scheme
(dt = 0.02) and 2 s with the exact LGCM scheme.

Results of `python -m convexity.mc_verification` (2^18 paths, measured):

| check | result |
|---|---|
| No-arbitrage identities (`Z^{d\|d}`, `x Z^{f\|d}`, `x Z^{f\|f}`) at T=5, T'=15 | within 1.1 s.e. for LGCM exact, LGCM Euler and QGCM kappa=15. With `y_mode="frozen"` (QGCM with the LGCM y, v) they fail by 7 to 14 s.e. The stochastic y_t, v^{f,c}_t states of the theorem are needed |
| LGCM 5Yx10Y `Y^{f\|d}` normal vol (scale 1) | MC 131.33 ± 0.25 bp, exact quadrature 131.19, Gaussian approx 131.62 |
| LGCM `Y^{f\|d}` (scale 2) | MC 259.54 ± 0.56, exact 259.22, approx 262.58: the approximation error (+1.3%) is resolved by MC at 5.6 s.e. |
| LGCM `B^{f/d\|d,r}` normal vol (scale 1) | exact 16.285 bp. Four independent seeds give 16.294, 16.279, 16.290, 16.274 (± 0.03); approx 16.276 |
| QGCM native swap `Y^{d\|d}`, kappa = 0 / 10 / 20 | MC 68.55 / 68.69 / 70.22 (± 0.13–0.15) bp. LGCM formula 68.51 for all kappa (rejected at kappa=20, 11 s.e.). Frozen gradient with E^A[y_{T0}]: 68.56 / 68.42 / 69.65, better but still 4 s.e. off at kappa=20. MC smile at kappa=20: 56.7 / 66.5 / 75.5 bp at -100 / ATM / +100 bp, a skew no Gaussian approximation can produce |
| Libor collateral convexity at S = 5Y, 10Y | LGCM: MC 4.41e-5 ± 0.45e-5 vs formula 4.95e-5, and 1.71e-4 ± 0.11e-4 vs 1.80e-4. QGCM kappa = 10, 20: within 1.3 s.e. of the LGCM formula, so the deterministic convexity is not visibly affected by this local vol |

The estimator for the Libor convexity uses the identity E[x Z^{f|d} / deflator] = 1 as a control
variate (standard error divided by 27) and common random numbers between QGCM and LGCM.

## Payoffs

```
Payoff                      payment in `ccy` under `collateral` at `expiry`
├── DiscountFactor          Z_{T,Tm}
├── ForwardDiscountFactor   Z_{S,T2}/Z_{S,T1} paid at S (convexity) or at T1
├── SwapPayoff              forward-starting swap T0 < T1 < ... < Tn
│   ├── Swap                omega A (Y - K)
│   └── Swaption            A (omega (Y - K))^+
└── FXPayoff                payment in d, on the FX spot x^{df}_T
    ├── FXForward           x_T - K
    └── FXOption            (omega (x_T - K))^+
```

Each payoff has `value(sim, res, curves)` (value at expiry on every path) and `lgcm_price(model, curves)`.
The LGCM price is exact for all payoffs except swaptions, where it is the Bachelier price with the
Gaussian approximation. Prices are E^{d,RN}[D V_T], with the deflator `Simulator.deflator(res, curves, ccy,
collateral)`: exp(-int r^{d|d}) for d collateral, exp(-int (r^{d|d} - q)) for f collateral, times
x_T/x_0 for f payments. Supported: discount factors and swaps on Z^{d|d}, Z^{f|f}, Z^{f|d}; FX payoffs
under d or f collateral. For "fd" swaps in the QGCM, the Libor collateral convexity is the deterministic
LGCM one (an approximation; section 4 of the report finds it accurate).

```python
from convexity.lgcm import default_model
from convexity.montecarlo import Simulator
from convexity.schemes import EulerScheme
from convexity.market import InitialCurves
from convexity.payoffs import Swaption, FXOption, price

m, curves = default_model(), InitialCurves()
sim = Simulator(m, kappa=15.0, scheme=EulerScheme(dt=0.02))     # QGCM; ExactScheme() for the LGCM
pay = [Swaption(5, 40, 0.25, 0.042, "f", "d"), FXOption(5, 1.1, collateral="f")]
price(sim, pay, curves, n_paths=2**18)                           # [(pv, stderr), ...]
```

Measured (2^18 paths, seed 7): in the LGCM all 18 payoffs of the test set agree with their closed
forms within 2 standard errors, including the FX forward and the Black price under foreign
collateral, which use the FX collateral convexity (0.15% at 5Y) and Z^{d|f}_0. In the QGCM
(kappa = 15) the model-free prices stay within 2.4 standard errors. Swaptions leave the Gaussian
approximation by 7 to 16 standard errors (smile), and forward discount factors paid at fixing by
about 3 standard errors (convexity under stochastic vol). Timing: 18 payoffs on 2^18 paths in 5 s
(exact scheme) and 15 s (Euler, dt = 0.02).

## Repo convexity

`repo.py` implements section "Bonds, repo and repo convexity" of the notes (McCloud, *Repo convexity*,
arXiv:1905.03316). The repo rate on the bond maturing at T over [s, e] is
1 + delta f = (pbar_{0,s}/pbar_{0,e}) K^{repo}_0(s, e, T): a forward collateral convexity, with the bond curve as the
native curve and the OIS curve as the collateral. Run from `python/`:

```python
from convexity import repo
rm = repo.RepoModel.hull_white(theta=0.05, sigma=0.01, kappa=0.4, eps=0.008, rho=-0.5)
cv = repo.BondCurves.flat(rbar=0.035, b=-0.002)
repo.repo_rate(rm, cv, 1.0, 1.25, 10.0)                                  # 3M repo in 1Y on a 10Y bond, ACT/360
repo.coupon_repo_rate_exact(rm, cv, repo.CouponBond.fixed(10.0, 0.05, 2), 1.0, 1.25)
repo.mc_repo_rate(rm, cv, 10.0, 1.0, 1.25, n_paths=2**18)               # (rate, stderr)
```

The notebook `python/notebooks/repo_convexity.py` shows the impacts: `marimo edit python/notebooks/repo_convexity.py`.
