"""Numerical companion of docs/CollateralDiscounting.tex: the Linear Gaussian Collateral Model.

Modules
-------
lgcm             model, covariance integrals, reconstruction formulas
market           initial curves, funding spreads, LGCM closed forms (convexities, FX, Black/Bachelier)
transitions      exact one-step transitions under the RN, T-forward and jumping numeraire measures
moments          moment ODEs of the state SDE (independent benchmark)
gaussian_approx  Gaussian approximations of swap rates / currency basis spreads and their exact benchmark
schemes          Monte Carlo schemes: MCScheme (base), EulerScheme (QGCM/LGCM), ExactScheme (LGCM, simulate_dates)
montecarlo       Simulator (model, QGCM local vol, reconstruction, deflators) delegating to a scheme
payoffs          Payoff hierarchy (discount factors, swaps, swaptions, FX forwards/options) and pricer
mc_verification  Monte Carlo verification report of the approximations
curves           indices, curve keys, piecewise flat / spread curves, LGCMSetup, CurveSet
instruments      calibration instruments (swaps, FX forwards, resetting CCS, swaptions, FX options, CB swaptions), quotes
pricers          DeterministicPricer (DCB) and LGCMPricer (Libor convexities, exact and Gaussian option vols)
curve_calibration  sequential curve bootstrap with a convexity provider; calibrate_curves (DCB)
calibration      ModelSpec and calibrate: the LGCM calibration algorithm
market_data      illustrative USD/EUR example market (not live data)
calibration_check  analytic and Monte Carlo repricing of the calibration quotes
"""
