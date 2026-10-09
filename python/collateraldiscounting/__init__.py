"""Numerical companion of docs/CollateralDiscounting.tex: the Linear Gaussian Collateral Model.

Modules
-------
lgcm             model, covariance integrals, reconstruction formulas
market           initial curves, funding spreads, LGCM closed forms (convexities, FX, Black/Bachelier)
transitions      exact one-step transitions under the RN, T-forward and jumping numeraire measures
moments          moment ODEs of the state SDE (independent benchmark)
gaussian_approx  Gaussian approximations of swap rates / currency basis spreads and their exact benchmark
schemes          Monte Carlo schemes: MCScheme (base), EulerScheme (QGCM/LGCM), ExactScheme (LGCM)
montecarlo       Simulator (model, QGCM local vol, reconstruction, deflators) delegating to a scheme
payoffs          Payoff hierarchy (discount factors, swaps, swaptions, FX forwards/options) and pricer
mc_verification  Monte Carlo verification report of the approximations
"""
