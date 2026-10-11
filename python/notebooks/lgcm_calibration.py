import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    _py = str(Path(__file__).resolve().parent.parent)
    if _py not in sys.path:
        sys.path.insert(0, _py)

    import functools
    from dataclasses import replace

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from collateraldiscounting.calibration import BlockSpec, CalibrationError, ModelSpec, calibrate, default_spec
    from collateraldiscounting.calibration_check import analytic_report, mc_report
    from collateraldiscounting.curve_calibration import calibrate_curves
    from collateraldiscounting.instruments import (CurrencyBasisSwaption, FXForward, FXOption, ResettingCCS, Swap,
                                                   Swaption)
    from collateraldiscounting.market import fx_collateral_convexity, libor_convexity
    from collateraldiscounting.market_data import example_quotes

    # colour-blind-safe palette (Okabe-Ito), one colour per object across all panels
    C = {"lgcm": "#0072B2", "dcb": "#D55E00", "d": "#009E73", "f": "#CC79A7", "c": "#E69F00", "x": "#7F7F7F",
         "zfd": "#000000", "zdf": "#56B4E9", "grey": "#666666"}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})
    return (BlockSpec, C, CalibrationError, CurrencyBasisSwaption, FXForward, FXOption, ModelSpec, ResettingCCS, Swap, Swaption,
            analytic_report, calibrate, calibrate_curves, default_spec, example_quotes, functools,
            fx_collateral_convexity, libor_convexity, mc_report, mo, np, plt, replace)


@app.cell
def _(mo):
    mo.md(r"""
    # Linear Gaussian Collateral Model: calibration

    A two-currency market, **USD** (domestic `d`, collateral SOFR) and **EUR** (foreign `f`, collateral
    ESTR, index EURIBOR 3M), with the collateral basis `c` (f to d) and the FX rate `x` (USD per EUR).
    The quotes are rounded broker-screen style numbers, not live data.

    **What is calibrated to what** (algorithm "LGCM calibration" of the notes):

    1. $Z^{d|d}$, $Z^{f|f}$ from the swaps (model-free bootstrap).
    2. $\lambda^d$, $\lambda^f$ (piecewise constant) from the ATM swaption normal vols, expiry by expiry.
    3. A **fixed point**: $Z^{f|d}$ from the FX forwards and resetting CCS (using the LGCM Libor
       convexities, which depend on $\sigma^c$), then $\lambda^c$ from the currency-basis spread vols, repeated
       until neither moves.
    4. $\sigma^x$ from the ATM FX vols (the total variance is quadratic in $\sigma^x$ on each segment).
    5. Model output, not an input: the off-market curve $Z^{d|f}$ (USD cash flows collateralised in EUR).

    **Exact versus approximate.** With one factor per block the swaption vols are priced exactly
    (quadrature); the Gaussian approximation of the swaption variance is only used as the starting guess
    and is asymptotic in the volatility, its error is shown in the `approx` column below. The multi-factor
    case uses the Gaussian approximation. Mean reversions and correlations are *marked* inputs, not calibrated.
    The Monte Carlo check is a statistical statement: model minus market should lie within 4 standard errors.
    """)
    return


@app.cell
def _(CurrencyBasisSwaption, FXForward, FXOption, ResettingCCS, Swap, Swaption, example_quotes, mo):
    MQ = example_quotes()

    def _display(inst, v):
        if isinstance(inst, Swap):
            return "swap rate (%)", v * 100
        if isinstance(inst, FXForward):
            return "FX forward points (pips)", v
        if isinstance(inst, ResettingCCS):
            return "CCS basis (bp)", v * 1e4
        if isinstance(inst, (Swaption, CurrencyBasisSwaption)):
            return "normal vol (bp/yr)", v * 1e4
        if isinstance(inst, FXOption):
            return "Black vol (%)", v * 100
        return "", v

    _rows = []
    for _q in MQ.quotes:
        _unit, _val = _display(_q.instrument, _q.value)
        _rows.append({"label": _q.instrument.label, "type": type(_q.instrument).__name__, "unit": _unit,
                      "value": round(_val, 4)})
    mo.vstack([mo.md(f"## Market quotes\n\nFX spot {MQ.fx_spot:.4f} USD per EUR; {len(MQ.quotes)} quotes."),
               mo.ui.table(_rows, page_size=12, selection=None)])
    return (MQ,)


@app.cell
def _(mo):
    rho = mo.ui.slider(-0.9, 0.9, step=0.1, value=-0.4, label=r"$\rho^{fc}$ (corr. EUR rates / basis)", debounce=True)
    cbscale = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="CB vol scale", debounce=True)
    beta_c = mo.ui.slider(0.01, 0.5, step=0.01, value=0.10, label=r"$\beta^c$", debounce=True)
    beta_f = mo.ui.slider(-0.02, 0.2, step=0.01, value=0.02, label=r"$\beta^f$", debounce=True)
    run_mc = mo.ui.run_button(label="Run Monte Carlo check")
    run_impact = mo.ui.run_button(label="Run parameter-impact table")
    npaths = mo.ui.dropdown({"2^14": 2**14, "2^15": 2**15, "2^16": 2**16}, value="2^15", label="MC paths")
    mo.vstack([mo.md("## Controls\n\nThe sliders recalibrate the whole model (about 5.5 s) when released "
                     "(results are cached per parameter tuple, so returning to earlier values is instant). "
                     "The Monte Carlo check and the parameter-impact table (up to 5 recalibrations) run only on their buttons."),
               mo.hstack([rho, cbscale], justify="start"), mo.hstack([beta_c, beta_f], justify="start"),
               mo.hstack([run_mc, npaths, run_impact], justify="start")])
    return beta_c, beta_f, cbscale, npaths, rho, run_impact, run_mc


@app.cell
def _(BlockSpec, CurrencyBasisSwaption, MQ, calibrate, calibrate_curves, default_spec, functools,
      fx_collateral_convexity, np, replace):
    # No slider dependencies in this cell: the caches below survive slider releases.
    def build(rho_fc, scale, bc, bf):
        """ModelSpec and quotes for the parameter tuple (rho^{fc}, CB vol scale, beta^c, beta^f).
        Raises ValueError when the correlation matrix is not positive definite."""
        spec0 = default_spec()
        R = spec0.corr.copy()
        R[1, 2] = R[2, 1] = rho_fc          # order d, f, c, x
        spec = replace(spec0, corr=R, c=BlockSpec((bc,), level=spec0.c.level),
                       f=BlockSpec((bf,), level=spec0.f.level))
        vals = [q.value * scale if isinstance(q.instrument, CurrencyBasisSwaption) else q.value
                for q in MQ.quotes]
        return spec, MQ.with_values(vals)

    @functools.lru_cache(maxsize=32)
    def calibrated(rho_fc, scale, bc, bf):
        """Calibration for the parameter tuple (cached on the tuple)."""
        spec, quotes = build(rho_fc, scale, bc, bf)
        return calibrate(quotes, spec)

    dcb, _ = calibrate_curves(MQ)

    @functools.lru_cache(maxsize=32)
    def impact(rho_fc, scale, bc, bf):
        """(30Y Z^{f|d} zero-rate shift vs DCB in bp, lambda^c at 10Y in bp, 10Y FX collateral convexity in bp)."""
        cal_ = calibrated(rho_fc, scale, bc, bf)
        shift = (-np.log(cal_.curves.Z0("fd", 30.0)) + np.log(dcb.Z0("fd", 30.0))) / 30.0 * 1e4
        k, lam = cal_.levels("c")
        lam10 = lam[min(np.searchsorted(k, 10.0 - 1e-9, side="right") - 1, len(lam) - 1)]
        return float(shift), float(lam10 * 1e4), float((fx_collateral_convexity(cal_.model, 10.0) - 1) * 1e4)

    return build, calibrated, dcb, impact


@app.cell
def _(CalibrationError, beta_c, beta_f, build, calibrated, cbscale, mo, rho):
    try:
        build(rho.value, cbscale.value, beta_c.value, beta_f.value)    # only the spec construction is guarded
    except ValueError as _e:
        mo.stop(True, mo.callout(mo.md(f"rho^fc = {rho.value:.1f} is infeasible with the other (fixed) correlations: "
                                       f"{_e}. Choose a value in about [-0.7, 0.6]."), kind="danger"))
    try:
        cal = calibrated(rho.value, cbscale.value, beta_c.value, beta_f.value)
    except CalibrationError as _e:     # e.g. CB vol scale 0: a variance squeeze; other errors stay visible
        mo.stop(True, mo.callout(mo.md(f"The calibration fails for these parameters: {_e}"), kind="danger"))
    _d = cal.diagnostics
    mo.md(f"""## Calibration

    Calibrated in **{_d['seconds']:.1f} s** (cached: a repeated parameter tuple is not recalibrated);
    fixed point ($Z^{{f|d}}$, $\\sigma^c$) converged in
    **{_d['fixed_point_iterations']}** iterations (max change per iteration:
    {', '.join(f'{c:.1e}' for c in _d['fixed_point_changes'])}).
    Parameters: $\\rho^{{fc}}$ = {rho.value:.1f}, CB vol scale = {cbscale.value:.1f}, $\\beta^c$ = {beta_c.value:.2f},
    $\\beta^f$ = {beta_f.value:.2f}. DCB = the same quotes with deterministic collateral basis (no convexity).
    """)
    return (cal,)


@app.cell
def _(C, cal, dcb, mo, np, plt):
    _T = np.linspace(0.25, 30, 120)

    def _zr(curves, kind):
        return np.array([-np.log(curves.Z0(kind, t)) / t for t in _T]) * 100

    def _fwd_q(curves):          # instantaneous collateral basis forward q = -d/dT log(Z^{fd}/Z^{ff})
        h = 1e-3
        return np.array([-(np.log(curves.Z0("fd", t + h) / curves.Z0("ff", t + h))
                           - np.log(curves.Z0("fd", t) / curves.Z0("ff", t))) / h for t in _T]) * 1e4

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    _a = ax[0, 0]
    for _k, _lab, _col, _ls in [("dd", "$Z^{d|d}$ USD, SOFR", C["d"], "-"), ("ff", "$Z^{f|f}$ EUR, ESTR", C["f"], "-"),
                                ("fd", "$Z^{f|d}$ EUR, SOFR coll.", C["zfd"], "-"),
                                ("df", "$Z^{d|f}$ USD, ESTR coll.", C["zdf"], "-")]:
        _a.plot(_T, _zr(cal.curves, _k), color=_col, ls=_ls, label=_lab + " (LGCM)")
    for _k, _col in [("fd", C["dcb"])]:
        _a.plot(_T, _zr(dcb, _k), color=_col, ls=(0, (4, 2)), lw=2, zorder=5, label="$Z^{f|d}$ (DCB, vermillion dashed)")
    _a.set(title="Zero rates by currency and collateral", xlabel="maturity (years)", ylabel="continuously compounded zero rate (%)")
    _a.legend(fontsize=8, frameon=False)
    _a = ax[0, 1]
    _a.plot(_T, _fwd_q(cal.curves), color=C["lgcm"], label="LGCM")
    _a.plot(_T, _fwd_q(dcb), color=C["dcb"], ls="--", label="DCB")
    _a.set(title="Instantaneous collateral basis forward $q$", xlabel="maturity (years)", ylabel="$q$ (bp)")
    _a.legend(frameon=False)
    _a = ax[1, 0]
    for _b, _col in [("d", C["d"]), ("f", C["f"]), ("c", C["c"])]:
        _k, _lam = cal.levels(_b)
        _a.step(_k, np.r_[_lam, _lam[-1]] * 1e4, where="post", color=_col, label=rf"$\lambda^{_b}$")
    _a.set(title=r"Calibrated volatility levels $\lambda^d,\lambda^f,\lambda^c$", xlabel="time (years)",
           ylabel=r"$\lambda$ (bp/yr)", xlim=(0, 25))
    _a.legend(frameon=False)
    _a = ax[1, 1]
    _k, _sx = cal.levels("x")
    _a.step(_k, np.r_[_sx, _sx[-1]] * 100, where="post", color=C["x"], label=r"$\sigma^x$")
    _a.set(title="Calibrated FX volatility $\\sigma^x$", xlabel="time (years)", ylabel=r"$\sigma^x$ (%)", xlim=(0, 25))
    _a.legend(frameon=False)
    mo.vstack([mo.md("## Calibrated curves and volatilities"), fig])
    return


@app.cell
def _():
    def unit_scale(kind):
        if kind == "Swap":
            return "%", 100
        if kind in ("Swaption", "CurrencyBasisSwaption"):
            return "bp", 1e4
        if kind == "FXOption":
            return "vol pt", 100
        if kind == "FXForward":
            return "pips", 1
        return "bp", 1e4

    return (unit_scale,)


@app.cell
def _(analytic_report, cal, mo, np, unit_scale):
    _rows = []
    for _r in analytic_report(cal):
        _u, _s = unit_scale(_r.kind)
        _rows.append({"instrument": _r.label, "unit": _u, "market": round(_r.market * _s, 4),
                      "model": round(_r.model * _s, 4), "error": float(f"{_r.error * _s:.2e}"),
                      "Gaussian approx. error": (None if np.isnan(_r.approx) else round((_r.approx - _r.market) * _s, 4))})
    _worst = max(abs(r["error"]) for r in _rows)
    mo.vstack([mo.md(f"""## Analytic repricing

    The model reprices every quote with the *exact* pricer; the largest absolute error (in the unit of the row) is
    **{_worst:.1e}**. The last column is the error of the **Gaussian approximation** for swaption and
    CB vols, which is asymptotic in the volatility (not exact), so it is nonzero.
    """), mo.ui.table(_rows, page_size=15, selection=None)])
    return


@app.cell
def _(C, cal, mc_report, mo, npaths, np, plt, run_mc, unit_scale):
    mo.stop(not run_mc.value, mo.md("## Monte Carlo check\n\nPress *Run Monte Carlo check* above. "
                                    "Note: the result is cleared whenever the model is recalibrated (a slider "
                                    "is released), because this cell depends on the calibration and the button "
                                    "resets after each run; press the button again for the new model."))
    _t = {}
    _rows = mc_report(cal, n_paths=npaths.value, timings=_t)
    _zs = np.nan_to_num(np.array([r.z for r in _rows]))
    _ok = np.abs(_zs) <= 4
    _fig, _ax = plt.subplots(figsize=(8, 0.17 * len(_rows) + 1.5), constrained_layout=True)
    _y = np.arange(len(_rows))
    _ax.barh(_y, _zs, color=np.where(_ok, C["lgcm"], C["dcb"]))
    for _v in (-4, 4):
        _ax.axvline(_v, color=C["grey"], ls="--", lw=1)
    _ax.set(title="Monte Carlo repricing: z = (model - market) / s.e.", xlabel="z-score (dimensionless)",
            yticks=_y, yticklabels=[r.label for r in _rows], ylim=(len(_rows) - 0.5, -0.5))
    _ax.tick_params(axis="y", labelsize=7)
    _tab = []
    for r in _rows:
        _u, _sc = unit_scale(r.kind)
        _tab.append({"instrument": r.label, "unit": _u, "market": round(r.market * _sc, 4),
                     "MC value": round(r.model * _sc, 4), "s.e.": float(f"{r.stderr * _sc:.2e}"),
                     "z": None if np.isnan(r.z) else round(r.z, 2)})
    _bad = [r.label for r, ok in zip(_rows, _ok) if not ok and not np.isnan(r.z)]
    mo.vstack([mo.md(f"""## Monte Carlo check

    {npaths.value} paths, exact scheme; simulation {_t['simulate']:.1f} s. The check passes when |z| <= 4.
    Rows outside: **{', '.join(_bad) if _bad else 'none'}**. Instruments with no sampling error (z undefined,
    e.g. deterministic cash flows) are not plotted as outliers. The result is cleared when a slider is
    released (see above); press the button again.
    """), _fig, mo.ui.table(_tab, page_size=10, selection=None)])
    return


@app.cell
def _(C, cal, dcb, fx_collateral_convexity, libor_convexity, mo, np, plt):
    _m = cal.model
    _j = np.arange(0, 120)
    _K = np.array([libor_convexity(_m, 0.0, 0.25 * j, 0.25 * (j + 1)) - 1 for j in _j]) * 1e4
    _T = np.linspace(0.25, 30, 120)
    _zr = lambda cv, k: np.array([-np.log(cv.Z0(k, t)) / t for t in _T])
    _dfd = (_zr(cal.curves, "fd") - _zr(dcb, "fd")) * 1e4
    _fxc = np.array([fx_collateral_convexity(_m, t) - 1 for t in _T]) * 1e4
    _zdf = np.array([cal.curves.Z0("df", t) for t in _T])
    _off = np.array([cal.curves.x0 * cal.curves.Z0("ff", t) / cal.curves.fx_forward0(t) for t in _T])
    _dzdf = (-np.log(_zdf) / _T + np.log(_off) / _T) * 1e4

    fig2, ax2 = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    _a = ax2[0, 0]
    _a.plot(0.25 * _j, _K, color=C["lgcm"], marker=".", ms=3, lw=1)
    _a.set(title="Libor convexity $K_j-1$ (quarterly EURIBOR 3M periods)", xlabel="period start (years)",
           ylabel="$K_j-1$ (bp)")
    _a = ax2[0, 1]
    _a.plot(_T, _dfd, color=C["lgcm"])
    _a.axhline(0, color=C["dcb"], ls="--", label="DCB (zero by definition)")
    _a.set(title="$Z^{f|d}$ zero rate: LGCM minus DCB", xlabel="maturity (years)", ylabel="difference (bp)")
    _a.legend(frameon=False)
    _a = ax2[1, 0]
    _a.plot(_T, _fxc, color=C["lgcm"], label=r"LGCM: $x^{df|f}/x^{df|d}-1$")
    _a.axhline(0, color=C["dcb"], ls="--", label="DCB (no convexity)")
    _a.set(title="FX forward: EUR collateral vs USD collateral", xlabel="maturity (years)",
           ylabel="relative difference (bp)")
    _a.legend(frameon=False)
    _a = ax2[1, 1]
    _a.plot(_T, _dzdf, color=C["lgcm"])
    _a.axhline(0, color=C["dcb"], ls="--", label="DCB off-market curve $x_0 Z^{f|f}/F$")
    _a.set(title="$Z^{d|f}$ zero rate: LGCM minus DCB off-market curve", xlabel="maturity (years)",
           ylabel="difference (bp)")
    _a.legend(frameon=False)

    def _at(arr, T):
        return float(arr[np.argmin(abs(_T - T))])

    mo.vstack([mo.md(f"""## Collateral convexity

    All numbers are computed from the calibrated model. Under the current parameters:

    - 30Y $Z^{{f|d}}$ zero-rate shift, LGCM minus DCB: **{_at(_dfd, 30):.2f} bp**
    - 10Y FX forward collateral convexity $x^{{df|f}}/x^{{df|d}}-1$: **{fx_collateral_convexity(_m, 10.0) * 1e4 - 1e4:.2f} bp**
      (30Y: {fx_collateral_convexity(_m, 30.0) * 1e4 - 1e4:.2f} bp)
    - 10Y $Z^{{d|f}}$ zero-rate difference to the DCB off-market curve: **{_at(_dzdf, 10):.2f} bp**
    - Largest Libor convexity $K_j-1$ over 30Y: **{_K[np.argmax(abs(_K))]:.2f} bp** at {0.25 * _j[np.argmax(abs(_K))]:.2f}Y

    The DCB curves ignore the stochastic basis; the differences are the convexity adjustments that the LGCM
    adds. They are exact closed forms of the calibrated Gaussian model (no Monte Carlo noise).
    """), fig2])
    return


@app.cell
def _(beta_c, beta_f, build, cbscale, impact, mo, run_impact):
    mo.stop(not run_impact.value, mo.md("## Parameter impact: $\\rho^{fc}$\n\nPress *Run parameter-impact table* above "
                                        "(about 5.5 s per uncached row)."))
    _rows = []
    for _r in (-0.8, -0.6, -0.4, 0.0, 0.4):
        try:
            build(_r, cbscale.value, beta_c.value, beta_f.value)    # only the spec construction is guarded
        except ValueError as _e:
            _rows.append({"rho^{fc}": _r, "30Y Z^{f|d} shift vs DCB (bp)": f"n/a ({_e})",
                          "lambda^c at 10Y (bp/yr)": "n/a", "10Y FX fwd collateral convexity (bp)": "n/a"})
            continue
        _s, _l, _f = impact(_r, cbscale.value, beta_c.value, beta_f.value)   # genuine errors propagate
        _rows.append({"rho^{fc}": _r, "30Y Z^{f|d} shift vs DCB (bp)": round(_s, 3),
                      "lambda^c at 10Y (bp/yr)": round(_l, 3), "10Y FX fwd collateral convexity (bp)": round(_f, 3)})
    mo.vstack([mo.md(f"""## Parameter impact: $\\rho^{{fc}}$

    Recalibration for several values of the marked correlation, other parameters as in the controls
    (CB vol scale {cbscale.value:.1f}, $\\beta^c$ {beta_c.value:.2f}, $\\beta^f$ {beta_f.value:.2f}). -0.8 is infeasible: with the
    other fixed correlations the matrix is not positive definite (the feasible range is about [-0.7, 0.6]).
    The table runs only when its button is pressed. Results are cached per parameter tuple in a cache that is
    independent of the sliders, so a repeated tuple (also the one used by the main calibration) is instant;
    a new tuple costs about 5.5 s per row. The market quotes are the same in every row: $\\rho^{{fc}}$ changes what
    the *same* quotes imply for the basis curve and convexities.
    """), mo.ui.table(_rows, selection=None)])
    return


if __name__ == "__main__":
    app.run()
