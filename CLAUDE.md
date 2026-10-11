# Collateral discounting and convexity research

## Project overview

Personal research notes on collateral discounting: multi-currency HJM models with stochastic
collateral, collateral and quanto convexity adjustments, and the Quasi-Gaussian / Linear Gaussian
Collateral Models (QGCM / LGCM). Python experiments check the derivations numerically.

The working pattern is: **derive in LaTeX, then verify in Python.** A result is not settled until
it has been checked symbolically (sympy) or numerically (an exact benchmark, finite differences,
quadrature or Monte Carlo). Derivations in these notes have had sign errors, double counted terms
and wrong integration bounds that only surfaced that way.

- `docs/CollateralDiscounting.tex`: the notes (`amsart`, ~5.5k lines). It is a working notebook, with
  `TODO`/`FIXME`/`??` markers in places. Verification remarks quote measured numbers.
- `python/convexity/`: the numerical companion (see its `README.md`). Run the tests with
  `cd python && python -m pytest convexity -q`.

## Conventions

- **Distinguish exact from asymptotic from conjectural.** Gaussian approximations are asymptotic in
  the volatility. State the order and demonstrate the convergence rate numerically rather than
  asserting it.
- **Label statements** as Theorem / Proposition / Lemma / Remark. Do not present a numerical
  observation as a proved statement.
- **Verify before asserting.** Prefer an exact benchmark (no-arbitrage identities, moment ODEs,
  quadrature) over Monte Carlo. When quoting an error, quote the measured number.
- Monte Carlo tests use a 4 standard error tolerance with fixed seeds. Use control variates where
  the noise hides the effect (see `mc_verification.qgcm_libor_convexity`).
- Notation: blocks `d` (domestic rates), `f` (foreign rates), `c` (collateral basis f→d), `x` (FX).
  The state uses the Cheyette convention of Theorem "Quasi-Gaussian Collateral Model".

## Building

```bash
cd docs && pdflatex -interaction=nonstopmode CollateralDiscounting.tex   # twice, for refs
```

It currently compiles with 0 errors. The known warnings are two multiply defined labels
(`eq:cb_r_swaplet`, `eq:forward_cb_non_resetting`) and one undefined reference (`eq:cb_r_swaplet2`).
