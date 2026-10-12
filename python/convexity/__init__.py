"""Numerical companion of docs/CollateralDiscounting.tex: the Linear Gaussian Collateral Model.

Modules
-------
lgcm             model, covariance integrals, reconstruction formulas
market           initial curves, funding spreads, Coupon (Libor, daily, continuous), LGCM closed forms (convexities, FX, Black/Bachelier)
benchmarks       exact Gaussian benchmark of the coupon collateral convexity (independent of the closed forms)
transitions      exact one-step transitions under the RN, T-forward and jumping numeraire measures
moments          moment ODEs of the state SDE (independent benchmark)
gaussian_approx  Gaussian approximations of swap rates / currency basis spreads and their exact benchmark
correlations     terminal correlations of log-FX and swap (CMS) rates: Gaussian approximation, exact benchmark, Monte Carlo
schemes          Monte Carlo schemes: MCScheme (base), EulerScheme (QGCM/LGCM), ExactScheme (LGCM, simulate_dates)
montecarlo       Simulator (model, QGCM local vol, reconstruction, deflators) delegating to a scheme
payoffs          Payoff hierarchy (discount factors, swaps, swaptions, FX forwards/options) and pricer
mc_verification  Monte Carlo verification report of the approximations
curves           indices, curve keys, piecewise flat / hybrid (flat + smooth) / spread curves, LGCMSetup, CurveSet
instruments      calibration instruments (swaps, rate futures, FX forwards, resetting CCS, swaptions, FX options, CB swaptions), quotes
pricers          DeterministicPricer (DCB) and LGCMPricer (collateral convexity of the foreign coupon, Libor or compounded; exact and Gaussian option vols)
curve_calibration  sequential curve bootstrap with a convexity provider; calibrate_curves (DCB)
curve_fitting    nested global least squares curve fit on hybrid curves; calibrate_curves_global, GlobalFit
calibration      ModelSpec and calibrate: the LGCM calibration algorithm
market_data      illustrative EURUSD, GBPUSD, USDJPY (inverted) example markets (not live data)
calibration_check  analytic and Monte Carlo repricing of the calibration quotes
figures          plotly figures and LaTeX tables of the notes (python -m convexity.figures)
"""
