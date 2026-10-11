# Convexity

Notes on collateral discounting and collateral convexity, with a Python companion that checks the
derivations numerically.

| path | contents |
|---|---|
| `docs/CollateralDiscounting.tex` | HJM framework with multiple currencies and stochastic collateral; the Quasi-Gaussian and Linear Gaussian Collateral Models (QGCM / LGCM); convexity adjustments, Gaussian approximations of swap rates and currency basis spreads, simulation, calibration |
| `docs/smfart.cls`, `docs/smfbook.cls` | Société Mathématique de France document classes (`CollateralDiscounting.tex` itself uses `amsart`) |
| `python/convexity/` | numerical companion: LGCM, closed-form covariance integrals, exact transitions, moment-ODE benchmark, Gaussian approximations, a vectorised Monte Carlo simulator (QGCM and LGCM), a payoff hierarchy, compounded overnight (RFR) coupon convexities with an exact benchmark, and an LGCM calibration with three example markets (EURUSD, GBPUSD, USDJPY with inverted FX quotation). See its `README.md` |

## Usage

```bash
cd docs && pdflatex -interaction=nonstopmode CollateralDiscounting.tex   # twice, for refs
cd python && python -m pytest convexity -q                   # ~75 s
cd python && python -m convexity.mc_verification             # Monte Carlo report, ~2.5 min
```

Requires a LaTeX distribution with `amsart`, and Python with numpy, scipy and sympy.

LaTeX build files (`.aux`, `.log`, `.pdf`, ...) and Python caches are ignored (`.gitignore`).

Moved from the `smile-dynamic` repository.
