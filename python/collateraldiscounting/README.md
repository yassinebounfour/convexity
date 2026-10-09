# Linear Gaussian Collateral Model — numerical companion

Checks of the derivations in `docs/CollateralDiscounting.tex` (Quasi-Gaussian and Linear Gaussian
Collateral Models). Run from `python/`:

```
python -m pytest collateraldiscounting -q
```

| file | contents |
|---|---|
| `lgcm.py` | piecewise-constant LGCM (blocks d, f, c = f→d, x); closed-form covariance integrals `covint` and a brute-force quadrature benchmark; `y_t`, `v^{f,c}_t`; ZCB reconstruction exponents |
| `transitions.py` | exact one-step transitions of the state under Q^{d,RN}, the T-forward measure and the jumping numeraire measure |
| `moments.py` | moment ODEs of the state SDE — an independent benchmark that does not use the covariance integrals |
| `market.py` | `InitialCurves` (Z^{d\|d}_0, Z^{f\|f}_0, Z^{f\|d}_0, x_0, funding spreads) and the LGCM closed forms: Libor and FX collateral convexities, Z^{d\|f}_0, FX total variance, Black and Bachelier |
| `schemes.py` | `MCScheme` base class and its schemes: `EulerScheme(dt, y_mode)` for the QGCM with stochastic local vol and simulated `y_t`, `v^{f,c}_t`; `ExactScheme(dt)` with exact Gaussian transitions for the LGCM |
| `montecarlo.py` | `Simulator(model, kappa, floor, scheme)`: QGCM local vol, reconstruction formulas and deflators; path generation is delegated to the scheme. Vectorised over paths, antithetic |
| `payoffs.py` | `Payoff` hierarchy and `price(sim, payoffs, curves, n_paths, seed)` |
| `mc_verification.py` | Monte Carlo verification of the approximations; `python -m collateraldiscounting.mc_verification` prints the full report (~2.5 min) |
| `gaussian_approx.py` | Gaussian approximations of the forward swap rate Y^{f\|d} and the resetting currency basis spread B^{f/d\|d,r}, and their exact benchmark by 2D Gauss–Hermite quadrature |
| `tests/test_lgcm.py` | the checks quoted in the verification remarks of the notes |
| `tests/test_montecarlo.py` | reduced Monte Carlo checks (4 standard error tolerance, fixed seeds) |
| `tests/test_payoffs.py` | Monte Carlo prices of all payoffs against LGCM closed forms, model-free prices in the QGCM, put-call parities |

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

The test model (`default_model`) has two domestic factors, one foreign factor, one collateral
basis factor and the FX factor. Its parameters are piecewise constant, with zero, 1e-7 and
negative mean reversions on some intervals, and it uses a full 5×5 correlation matrix.

## Monte Carlo verification

QGCM volatility used for the stochastic-vol tests: a floored linear local volatility in each factor,
`sigma^{a,i}_t = sigmabar^{a,i}_t max(1 + kappa z^{a,i}_t, 0.05)`. It is separable, so the QGCM
theorem applies, and `kappa = 0` is the LGCM. Simulation is under Q^{d,RN} with the deflator
`exp(-int r^{d|d})`. Speed: 2^18 antithetic paths over 5Y take about 12 s with the Euler scheme
(dt = 0.02) and 2 s with the exact LGCM scheme.

Results of `python -m collateraldiscounting.mc_verification` (2^18 paths, measured):

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
from collateraldiscounting.lgcm import default_model
from collateraldiscounting.montecarlo import Simulator
from collateraldiscounting.schemes import EulerScheme
from collateraldiscounting.market import InitialCurves
from collateraldiscounting.payoffs import Swaption, FXOption, price

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
