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
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from convexity.calibration import BlockSpec, CalibrationError, ModelSpec, calibrate, default_spec
    from convexity.calibration_check import analytic_report, mc_report
    from convexity.curve_calibration import calibrate_curves
    from convexity.instruments import (CurrencyBasisSwaption, FXForward, FXOption, ResettingCCS, Swap,
                                                   Swaption)
    from convexity.curves import Index
    from convexity.market import Coupon, LIBOR, fx_collateral_convexity
    from convexity.market_data import MARKETS, example_quotes

    # colour-blind-safe palette (Okabe-Ito), one colour per object across all panels
    C = {"lgcm": "#0072B2", "dcb": "#D55E00", "d": "#009E73", "f": "#CC79A7", "c": "#E69F00", "x": "#7F7F7F",
         "zfd": "#000000", "zdf": "#56B4E9", "grey": "#666666"}
    DASH = {"-": "solid", "--": "dash", ":": "dot"}

    def style(fig, width=1000, height=380, legend_y=1.14):
        """Common look: white background, light grid, legend above the panels."""
        fig.update_layout(template="simple_white", width=width, height=height, font=dict(size=12),
                          margin=dict(l=65, r=20, t=80, b=55), hovermode="x unified",
                          legend=dict(orientation="h", x=0, y=legend_y, xanchor="left", yanchor="bottom"))
        fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", ticks="outside")
        fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8", ticks="outside")
        return fig

    def panel_legends(fig, cells, dy=0.03):
        """One legend above each panel: cells = {legend name: (xaxis name, yaxis name)}."""
        for name, (xa, ya) in cells.items():
            fig.update_layout({name: dict(orientation="h", x=fig.layout[xa].domain[0], y=fig.layout[ya].domain[1] + dy,
                                          xanchor="left", yanchor="bottom", font=dict(size=11))})
        return fig

    def line(x, y, color, name, dash="-", width=2, show=True, **kw):
        return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, dash=DASH.get(dash, dash), width=width),
                          name=name, showlegend=show, **kw)
    return (BlockSpec, C, CalibrationError, Coupon, CurrencyBasisSwaption, FXForward, FXOption, Index, LIBOR, MARKETS,
            ModelSpec, ResettingCCS, Swap, Swaption, analytic_report, calibrate, calibrate_curves, default_spec,
            example_quotes, functools, fx_collateral_convexity, go, line, make_subplots, mc_report, mo, np,
            panel_legends, replace, style)


@app.cell
def _(mo):
    mo.md(r"""
    # Linear Gaussian Collateral Model: calibration

    A two-currency market, **USD** (domestic `d`, collateral SOFR) and a foreign currency `f` chosen below,
    with the collateral basis `c` (f to d) and the FX rate `x` (the model quantity is always USD per unit of
    foreign). Three example markets: **EURUSD** (collateral ESTR, floating index EURIBOR 3M, a Libor-type
    coupon fixed at the period start), **GBPUSD** (SONIA) and **USDJPY** (TONA), whose floating index is a
    *compounded overnight* coupon. The quotes are rounded broker-screen style numbers, not live data.

    **What is calibrated to what** (algorithm "LGCM calibration" of the notes):

    1. $Z^{d|d}$, $Z^{f|f}$ from the swaps (model-free bootstrap).
    2. $\lambda^d$, $\lambda^f$ (piecewise constant) from the ATM swaption normal vols, expiry by expiry.
    3. A **fixed point**: $Z^{f|d}$ from the FX forwards and resetting CCS (using the LGCM collateral
       convexity of the foreign coupon, Libor or compounded, which depends on $\sigma^c$), then $\lambda^c$
       from the currency-basis spread vols, repeated until neither moves.
    4. $\sigma^x$ from the ATM FX vols (the total variance is quadratic in $\sigma^x$ on each segment).
    5. Model output, not an input: the off-market curve $Z^{d|f}$ (USD cash flows collateralised in the foreign currency).

    **Exact versus approximate.** With one factor per block the swaption vols are priced exactly
    (quadrature); the Gaussian approximation of the swaption variance is only used as the starting guess
    and is asymptotic in the volatility, its error is shown in the `approx` column below. The multi-factor
    case uses the Gaussian approximation. Mean reversions and correlations are *marked* inputs, not calibrated.
    The Monte Carlo check is a statistical statement: model minus market should lie within 4 standard errors.
    """)
    return


@app.cell
def _(MARKETS, mo):
    market = mo.ui.dropdown(list(MARKETS), value="EURUSD", label="Market")
    market
    return (market,)


@app.cell
def _(CurrencyBasisSwaption, FXForward, FXOption, ResettingCCS, Swap, Swaption, market, mo, quotes_for):
    MQ = quotes_for(market.value)

    def _display(inst, v):
        if isinstance(inst, Swap):
            return "swap rate (%)", v * 100
        if isinstance(inst, FXForward):
            return ("FX forward points (1/100 per USD, inverted quote)" if inst.inverted else "FX forward points (pips)"), v
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
    _s = MQ.setup
    _fccy = _s.foreign_collateral.currency
    _cp = _s.foreign_index.coupon
    _cpdesc = ("a Libor-type coupon (fixed at the period start)" if _cp.kind == "libor" else
               "a compounded overnight coupon, " +
               ("daily compounding on a grid of step 1/365 (exact in the model)" if _cp.kind == "daily"
                else "continuous compounding (an approximation)"))
    if MQ.fx_inverted:
        _fx = (f"FX spot (quoted, inverted): **{MQ.fx_spot:.4f} {_fccy} per USD**. The model works with "
               f"x = USD per {_fccy}, so x_0 = 1/{MQ.fx_spot:.4f} = {MQ.x0:.6f}; the forward points in the table are "
               f"quoted in the same inverted convention and converted internally.")
    else:
        _fx = f"FX spot (quoted): **{MQ.fx_spot:.4f} USD per {_fccy}** (equal to the model x_0)."
    mo.vstack([mo.md(f"## Market quotes: {market.value}\n\nForeign collateral {_s.foreign_collateral.name}, floating index "
                     f"{_s.foreign_index.name}, {_cpdesc}.\n\n{_fx} {len(MQ.quotes)} quotes."),
               mo.ui.table(_rows, page_size=12, selection=None)])
    return (MQ,)


@app.cell
def _(mo):
    rho = mo.ui.slider(-0.9, 0.9, step=0.1, value=-0.4, label=r"$\rho^{fc}$ (corr. foreign rates / basis)", debounce=True)
    cbscale = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="CB vol scale", debounce=True)
    beta_c = mo.ui.slider(0.01, 0.5, step=0.01, value=0.10, label=r"$\beta^c$", debounce=True)
    beta_f = mo.ui.slider(-0.02, 0.2, step=0.01, value=0.02, label=r"$\beta^f$", debounce=True)
    run_mc = mo.ui.run_button(label="Run Monte Carlo check")
    run_impact = mo.ui.run_button(label="Run parameter-impact table")
    run_libor = mo.ui.run_button(label="Recalibrate with the Libor coupon (RFR markets)")
    npaths = mo.ui.dropdown({"2^14": 2**14, "2^15": 2**15, "2^16": 2**16}, value="2^15", label="MC paths")
    mo.vstack([mo.md("## Controls\n\nThe sliders recalibrate the whole model (about 5.5 s) when released "
                     "(results are cached per parameter tuple, so returning to earlier values is instant). "
                     "Switching the market back and forth is cached too. The Monte Carlo check, the parameter-impact table "
                     "(up to 5 recalibrations) and the Libor-coupon recalibration run only on their buttons."),
               mo.hstack([rho, cbscale], justify="start"), mo.hstack([beta_c, beta_f], justify="start"),
               mo.hstack([run_mc, npaths, run_impact, run_libor], justify="start")])
    return beta_c, beta_f, cbscale, npaths, rho, run_impact, run_libor, run_mc


@app.cell
def _(BlockSpec, CurrencyBasisSwaption, Index, calibrate, calibrate_curves, default_spec, example_quotes, functools,
      fx_collateral_convexity, np, replace):
    # No slider (or market) dependencies in this cell: the caches below survive slider releases and market
    # switches; every cached function is keyed on the market name and the parameter tuple.
    @functools.lru_cache(maxsize=8)
    def quotes_for(mkt):
        return example_quotes(mkt)

    def build(mkt, rho_fc, scale, bc, bf):
        """ModelSpec and quotes of market `mkt` for the parameter tuple (rho^{fc}, CB vol scale, beta^c, beta^f).
        Raises ValueError when the correlation matrix is not positive definite."""
        MQ_ = quotes_for(mkt)
        spec0 = default_spec()
        R = spec0.corr.copy()
        R[1, 2] = R[2, 1] = rho_fc          # order d, f, c, x
        spec = replace(spec0, corr=R, c=BlockSpec((bc,), level=spec0.c.level),
                       f=BlockSpec((bf,), level=spec0.f.level))
        vals = [q.value * scale if isinstance(q.instrument, CurrencyBasisSwaption) else q.value
                for q in MQ_.quotes]
        return spec, MQ_.with_values(vals)

    @functools.lru_cache(maxsize=48)
    def calibrated(mkt, rho_fc, scale, bc, bf):
        """Calibration for the market and parameter tuple (cached on the key)."""
        spec, quotes = build(mkt, rho_fc, scale, bc, bf)
        return calibrate(quotes, spec)

    @functools.lru_cache(maxsize=8)
    def dcb_for(mkt):
        return calibrate_curves(quotes_for(mkt))[0]

    @functools.lru_cache(maxsize=32)
    def libor_calibrated(mkt, rho_fc, scale, bc, bf):
        """The same quotes calibrated with the foreign OIS index replaced by a Libor-type index (same name and
        tenor, zero spread), i.e. the Libor coupon K used for the RFR index (a deliberate mis-specification)."""
        spec, quotes = build(mkt, rho_fc, scale, bc, bf)
        s = quotes.setup
        lib = Index(s.foreign_index.name, s.foreign_index.currency, "IBOR", s.foreign_index.tenor, 0.0)
        return calibrate(replace(quotes, setup=replace(s, foreign_index=lib)), spec)

    @functools.lru_cache(maxsize=96)
    def impact(mkt, rho_fc, scale, bc, bf):
        """(30Y Z^{f|d} zero-rate shift vs DCB in bp, lambda^c at 10Y in bp, 10Y FX collateral convexity in bp)."""
        cal_ = calibrated(mkt, rho_fc, scale, bc, bf)
        dcb_ = dcb_for(mkt)
        shift = (-np.log(cal_.curves.Z0("fd", 30.0)) + np.log(dcb_.Z0("fd", 30.0))) / 30.0 * 1e4
        k, lam = cal_.levels("c")
        lam10 = lam[min(np.searchsorted(k, 10.0 - 1e-9, side="right") - 1, len(lam) - 1)]
        return float(shift), float(lam10 * 1e4), float((fx_collateral_convexity(cal_.model, 10.0) - 1) * 1e4)

    return build, calibrated, dcb_for, impact, libor_calibrated, quotes_for


@app.cell
def _(CalibrationError, beta_c, beta_f, build, calibrated, cbscale, dcb_for, market, mo, rho):
    try:
        build(market.value, rho.value, cbscale.value, beta_c.value, beta_f.value)    # only the spec construction is guarded
    except ValueError as _e:
        mo.stop(True, mo.callout(mo.md(f"rho^fc = {rho.value:.1f} is infeasible with the other (fixed) correlations: "
                                       f"{_e}. Choose a value in about [-0.7, 0.6]."), kind="danger"))
    try:
        cal = calibrated(market.value, rho.value, cbscale.value, beta_c.value, beta_f.value)
    except CalibrationError as _e:     # e.g. CB vol scale 0: a variance squeeze; other errors stay visible
        mo.stop(True, mo.callout(mo.md(f"The calibration fails for these parameters: {_e}"), kind="danger"))
    _d = cal.diagnostics
    dcb = dcb_for(market.value)
    mo.md(f"""## Calibration: {market.value}

    Calibrated in **{_d['seconds']:.1f} s** (cached: a repeated parameter tuple is not recalibrated);
    fixed point ($Z^{{f|d}}$, $\\sigma^c$) converged in
    **{_d['fixed_point_iterations']}** iterations (max change per iteration:
    {', '.join(f'{c:.1e}' for c in _d['fixed_point_changes'])}).
    Parameters: $\\rho^{{fc}}$ = {rho.value:.1f}, CB vol scale = {cbscale.value:.1f}, $\\beta^c$ = {beta_c.value:.2f},
    $\\beta^f$ = {beta_f.value:.2f}. DCB = the same quotes with deterministic collateral basis (no convexity).
    """)
    return cal, dcb


@app.cell
def _(C, MQ, cal, dcb, go, line, make_subplots, mo, np, panel_legends, style):
    _T = np.linspace(0.25, 30, 120)

    def _zr(curves, kind):
        return np.array([-np.log(curves.Z0(kind, t)) / t for t in _T]) * 100

    def _fwd_q(curves):          # instantaneous collateral basis forward q = -d/dT log(Z^{fd}/Z^{ff})
        h = 1e-3
        return np.array([-(np.log(curves.Z0("fd", t + h) / curves.Z0("ff", t + h))
                           - np.log(curves.Z0("fd", t) / curves.Z0("ff", t))) / h for t in _T]) * 1e4

    fig = make_subplots(2, 2, vertical_spacing=0.3, horizontal_spacing=0.1, subplot_titles=[
        "Zero rates by currency and collateral", "Instantaneous collateral basis forward q",
        "Calibrated volatility levels λ<sup>d</sup>, λ<sup>f</sup>, λ<sup>c</sup>", "Calibrated FX volatility σ<sup>x</sup>"])
    _fc, _fcol = MQ.setup.foreign_collateral.currency, MQ.setup.foreign_collateral.name.split("_")[-1]
    for _k, _lab, _col in [("dd", "Z<sup>d|d</sup> USD, SOFR", C["d"]), ("ff", f"Z<sup>f|f</sup> {_fc}, {_fcol}", C["f"]),
                           ("fd", f"Z<sup>f|d</sup> {_fc}, SOFR coll.", C["zfd"]), ("df", f"Z<sup>d|f</sup> USD, {_fcol} coll.", C["zdf"])]:
        fig.add_trace(line(_T, _zr(cal.curves, _k), _col, _lab + " (LGCM)"), 1, 1)
    fig.add_trace(line(_T, _zr(dcb, "fd"), C["dcb"], "Z<sup>f|d</sup> (DCB)", "--", 2.5), 1, 1)
    fig.add_trace(line(_T, _fwd_q(cal.curves), C["lgcm"], "q (LGCM)"), 1, 2)
    fig.add_trace(line(_T, _fwd_q(dcb), C["dcb"], "q (DCB)", "--"), 1, 2)
    for _b, _col in [("d", C["d"]), ("f", C["f"]), ("c", C["c"])]:
        _k, _lam = cal.levels(_b)
        fig.add_trace(line(_k, np.r_[_lam, _lam[-1]] * 1e4, _col, f"λ<sup>{_b}</sup>", line_shape="hv", legend="legend3"), 2, 1)
    _k, _sx = cal.levels("x")
    fig.add_trace(line(_k, np.r_[_sx, _sx[-1]] * 100, C["x"], "σ<sup>x</sup>", line_shape="hv", legend="legend4"), 2, 2)
    for _r, _c, _xt, _yt in [(1, 1, "maturity (years)", "zero rate, cont. comp. (%)"), (1, 2, "maturity (years)", "q (bp)"),
                             (2, 1, "time (years)", "λ (bp/yr)"), (2, 2, "time (years)", "σ<sup>x</sup> (%)")]:
        fig.update_xaxes(title_text=_xt, row=_r, col=_c, **({"range": [0, 25]} if _r == 2 else {}))
        fig.update_yaxes(title_text=_yt, row=_r, col=_c)
    style(fig, width=1100, height=860)
    # top row: one legend over both panels (colours are per object across the figure); bottom row: one per panel
    fig.update_layout(margin=dict(t=110), legend=dict(x=0, y=fig.layout.yaxis.domain[1] + 0.06))
    panel_legends(fig, {"legend3": ("xaxis3", "yaxis3"), "legend4": ("xaxis4", "yaxis4")}, dy=0.035)
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
def _(C, Coupon, cal, go, market, mc_report, mo, npaths, np, run_mc, style, unit_scale):
    mo.stop(not run_mc.value, mo.md("## Monte Carlo check\n\nPress *Run Monte Carlo check* above. "
                                    "Note: the result is cleared whenever the model is recalibrated (a slider "
                                    "is released), because this cell depends on the calibration and the button "
                                    "resets after each run; press the button again for the new model."))
    _t = {}
    _ois = cal.curves.setup.foreign_index.kind == "OIS"
    _step = 1 / 52 if _ois else None      # weekly compounding on the paths and in the model (daily paths are slow)
    _rows = mc_report(cal, n_paths=npaths.value, timings=_t, compounding_step=_step)
    _gapw = (max(abs(cal.curves.coupon("f").convexity(cal.model, 0.0, 0.25 * j, 0.25 * j + 0.25)
                     - Coupon("daily", _step).convexity(cal.model, 0.0, 0.25 * j, 0.25 * j + 0.25)) for j in range(120))
             if _ois else 0.0)
    _zs = np.nan_to_num(np.array([r.z for r in _rows]))
    _ok = np.abs(_zs) <= 4
    _fig = go.Figure(go.Bar(x=_zs, y=[r.label for r in _rows], orientation="h", showlegend=False,
                            marker_color=np.where(_ok, C["lgcm"], C["dcb"]), hovertemplate="%{y}: z = %{x:.2f}<extra></extra>"))
    for _v in (-4, 4):
        _fig.add_vline(x=_v, line=dict(color=C["grey"], dash="dash", width=1))
    _fig.update_yaxes(autorange="reversed", tickfont=dict(size=9))
    _fig.update_xaxes(title_text="z-score (dimensionless)")
    _fig.update_layout(title=dict(text="Monte Carlo repricing: z = (model − market) / s.e.", x=0))
    style(_fig, width=800, height=int(13 * len(_rows) + 140))
    _fig.update_layout(hovermode="closest", margin=dict(t=50))
    _tab = []
    for r in _rows:
        _u, _sc = unit_scale(r.kind)
        _tab.append({"instrument": r.label, "unit": _u, "market": round(r.market * _sc, 4),
                     "MC value": round(r.model * _sc, 4), "s.e.": float(f"{r.stderr * _sc:.2e}"),
                     "z": None if np.isnan(r.z) else round(r.z, 2)})
    _bad = [r.label for r, ok in zip(_rows, _ok) if not ok and not np.isnan(r.z)]
    mo.vstack([mo.md(f"""## Monte Carlo check

    {market.value}: {npaths.value} paths, exact scheme; simulation {_t['simulate']:.1f} s. The check passes when |z| <= 4.
    {"The foreign overnight coupon is compounded **weekly** (step 1/52) on the paths and in the model: daily paths are slow, and the closed form is exact for any step (the quotes stay those calibrated with the daily coupon, so the weekly coupon differs from the daily one by an O(step) amount: max |K_weekly - K_daily| over the quarterly periods to 30Y is " + f"{_gapw:.1e}" + "; compare it with the s.e. column below)." if _ois else "The foreign index is a Libor-type coupon (no compounding grid)."}
    Rows outside: **{', '.join(_bad) if _bad else 'none'}**. Instruments with no sampling error (z undefined,
    e.g. deterministic cash flows) are not plotted as outliers. The result is cleared when a slider is
    released (see above); press the button again.
    """), _fig, mo.ui.table(_tab, page_size=10, selection=None)])
    return

@app.cell
def _(C, cal, dcb, fx_collateral_convexity, line, make_subplots, mo, np, panel_legends, style):
    _m = cal.model
    _j = np.arange(0, 120)
    _cpn = cal.curves.coupon("f")     # the coupon of the foreign floating index of the market
    _K = np.array([_cpn.convexity(_m, 0.0, 0.25 * j, 0.25 * (j + 1)) - 1 for j in _j]) * 1e4
    _T = np.linspace(0.25, 30, 120)
    _zr = lambda cv, k: np.array([-np.log(cv.Z0(k, t)) / t for t in _T])
    _dfd = (_zr(cal.curves, "fd") - _zr(dcb, "fd")) * 1e4
    _fxc = np.array([fx_collateral_convexity(_m, t) - 1 for t in _T]) * 1e4
    _zdf = np.array([cal.curves.Z0("df", t) for t in _T])
    _off = np.array([cal.curves.x0 * cal.curves.Z0("ff", t) / cal.curves.fx_forward0(t) for t in _T])
    _dzdf = (-np.log(_zdf) / _T + np.log(_off) / _T) * 1e4

    fig2 = make_subplots(2, 2, vertical_spacing=0.3, horizontal_spacing=0.1, subplot_titles=[
        f"Coupon convexity K<sub>j</sub> − 1 ({_cpn.kind}, quarterly periods)", "Z<sup>f|d</sup> zero rate: LGCM minus DCB",
        "FX forward: foreign collateral vs USD collateral", "Z<sup>d|f</sup> zero rate: LGCM minus DCB off-market curve"])
    fig2.add_trace(line(0.25 * _j, _K, C["lgcm"], "K<sub>j</sub> − 1", width=1, marker=dict(size=4)), 1, 1)
    fig2.data[-1].mode = "lines+markers"
    fig2.add_trace(line(_T, _dfd, C["lgcm"], "LGCM", legend="legend2"), 1, 2)
    fig2.add_trace(line(_T, _fxc, C["lgcm"], "LGCM: x<sup>df|f</sup>/x<sup>df|d</sup> − 1", legend="legend3"), 2, 1)
    fig2.add_trace(line(_T, _dzdf, C["lgcm"], "LGCM", legend="legend4"), 2, 2)
    for _r, _c, _lg, _lab in [(1, 2, "legend2", "DCB (zero by definition)"), (2, 1, "legend3", "DCB (no convexity)"),
                              (2, 2, "legend4", "DCB off-market curve x<sub>0</sub> Z<sup>f|f</sup>/F")]:
        fig2.add_trace(line(_T, np.zeros_like(_T), C["dcb"], _lab, "--", legend=_lg), _r, _c)
    for _r, _c, _xt, _yt in [(1, 1, "period start (years)", "K<sub>j</sub> − 1 (bp)"), (1, 2, "maturity (years)", "difference (bp)"),
                             (2, 1, "maturity (years)", "relative difference (bp)"), (2, 2, "maturity (years)", "difference (bp)")]:
        fig2.update_xaxes(title_text=_xt, row=_r, col=_c)
        fig2.update_yaxes(title_text=_yt, row=_r, col=_c)
    style(fig2, width=1100, height=820)
    panel_legends(fig2, {'legend': ('xaxis', 'yaxis'), 'legend2': ('xaxis2', 'yaxis2'), 'legend3': ('xaxis3', 'yaxis3'), 'legend4': ('xaxis4', 'yaxis4')}, dy=0.035)

    def _at(arr, T):
        return float(arr[np.argmin(abs(_T - T))])

    mo.vstack([mo.md(f"""## Collateral convexity

    All numbers are computed from the calibrated model. Under the current parameters:

    - 30Y $Z^{{f|d}}$ zero-rate shift, LGCM minus DCB: **{_at(_dfd, 30):.2f} bp**
    - 10Y FX forward collateral convexity $x^{{df|f}}/x^{{df|d}}-1$: **{fx_collateral_convexity(_m, 10.0) * 1e4 - 1e4:.2f} bp**
      (30Y: {fx_collateral_convexity(_m, 30.0) * 1e4 - 1e4:.2f} bp)
    - 10Y $Z^{{d|f}}$ zero-rate difference to the DCB off-market curve: **{_at(_dzdf, 10):.2f} bp**
    - Largest {_cpn.kind} coupon convexity $K_j-1$ over 30Y: **{_K[np.argmax(abs(_K))]:.2f} bp** at {0.25 * _j[np.argmax(abs(_K))]:.2f}Y

    The DCB curves ignore the stochastic basis; the differences are the convexity adjustments that the LGCM
    adds. They are exact closed forms of the calibrated Gaussian model (no Monte Carlo noise).
    """), fig2])
    return


@app.cell
def _(C, Coupon, LIBOR, cal, line, make_subplots, market, mo, np, panel_legends, style):
    _m = cal.model
    _j = np.arange(0, 120)
    _a0, _b0 = 0.25 * _j, 0.25 * (_j + 1)

    def _Km1(coupon):
        return np.array([coupon.convexity(_m, 0.0, a, b) for a, b in zip(_a0, _b0)]) - 1.0

    _KL, _KD, _KC = _Km1(LIBOR), _Km1(Coupon("daily")), _Km1(Coupon("continuous"))
    _d_dl, _d_cd = _KD - _KL, _KC - _KD
    _rel = np.abs(_d_dl[1:]) / np.abs(_KL[1:])
    _steps = (1 / 12, 1 / 52, 1 / 365)
    _gap = [float(np.max(np.abs(_KC - _Km1(Coupon("daily", h))))) for h in _steps]
    _own = cal.curves.setup.foreign_index.coupon.kind

    fig3 = make_subplots(1, 3, horizontal_spacing=0.08, subplot_titles=[
        "Coupon convexity K − 1 per quarterly period", "Daily minus Libor", "Continuous minus daily"])
    fig3.add_trace(line(_a0, _KL * 1e4, C["lgcm"], "Libor", width=3.5), 1, 1)
    fig3.add_trace(line(_a0, _KD * 1e4, C["dcb"], "daily (1/365)", "--"), 1, 1)
    fig3.add_trace(line(_a0, _KC * 1e4, C["d"], "continuous", ":", 1.5), 1, 1)
    fig3.add_trace(line(_a0, _d_dl * 1e8, C["dcb"], "daily − Libor", legend="legend2"), 1, 2)
    fig3.add_trace(line(_a0, _d_cd * 1e8, C["d"], "continuous − daily", legend="legend3"), 1, 3)
    for _c, _yt in [(1, "K − 1 (bp)"), (2, "K<sup>daily</sup> − K<sup>Libor</sup> (10<sup>−8</sup>)"),
                    (3, "K<sup>cont</sup> − K<sup>daily</sup> (10<sup>−8</sup>)")]:
        fig3.update_xaxes(title_text="period start (years)", row=1, col=_c)
        fig3.update_yaxes(title_text=_yt, row=1, col=_c)
        if _c > 1:
            fig3.add_hline(y=0, line=dict(color=C["grey"], width=1), row=1, col=_c)
    style(fig3, width=1250, height=430)
    fig3.update_layout(margin=dict(t=115))
    panel_legends(fig3, {"legend": ("xaxis", "yaxis"), "legend2": ("xaxis2", "yaxis2"), "legend3": ("xaxis3", "yaxis3")}, dy=0.13)

    mo.vstack([mo.md(f"""## Coupon comparison on {market.value}: Libor, daily and continuous

    The three coupons evaluated on the *same* calibrated model, $K-1$ per quarterly period $[a, a+0.25]$ for
    $a$ up to 30Y, computed with `Coupon(kind).convexity`: **Libor** is $(1+\\theta L)$ fixed at the start,
    **daily** is the exact compounded overnight coupon on a daily grid (exact in this model), and **continuous**
    is its $\\exp\\int r$ limit, an *approximation* with an $O(\\text{{step}})$ error. This market's own index uses the
    **{_own}** coupon; the other two are what-ifs for comparison. The curves are indistinguishable in the first
    panel, hence the differences on the right (note the scale $10^{{-8}}$).

    - largest $K-1$ (Libor): **{_KL.max():.3e}** ({_KL.max() * 1e4:.3f} bp) at {_a0[np.argmax(_KL)]:.2f}Y; 30Y period: {_KL[-1]:.3e}
    - largest |daily $-$ Libor|: **{np.max(np.abs(_d_dl)):.2e}** (at most {np.max(_rel):.1e} relative to $K-1$; the first period $a=0$ is excluded because $K=1$ for every coupon there, so the relative difference is undefined)
    - largest |continuous $-$ daily|: **{np.max(np.abs(_d_cd)):.2e}**
    - $O(\\text{{step}})$ check: max |continuous $-$ daily(step)| for step = 1/12, 1/52, 1/365:
      {_gap[0]:.2e}, {_gap[1]:.2e}, {_gap[2]:.2e}; ratios {_gap[0] / _gap[1]:.1f} and {_gap[1] / _gap[2]:.1f}
      (step ratios {_steps[0] / _steps[1]:.1f} and {_steps[1] / _steps[2]:.1f}).
    """), fig3])
    return


@app.cell
def _(C, beta_c, beta_f, cal, cbscale, go, libor_calibrated, line, market, mo, np, rho, run_libor, style):
    _kind = cal.curves.setup.foreign_index.kind
    mo.stop(_kind != "OIS", mo.md(f"## Libor versus compounded coupon in the calibration\n\n{market.value} has a Libor-type "
                                  "floating index, so there is no compounded-coupon variant to compare with; select GBPUSD "
                                  "or USDJPY."))
    mo.stop(not run_libor.value, mo.md(f"## Libor versus compounded coupon in the calibration: {market.value}\n\n"
                                       "Press *Recalibrate with the Libor coupon* above (about 5.5 s the first time, cached "
                                       "per market and parameter tuple afterwards)."))
    _lib = libor_calibrated(market.value, rho.value, cbscale.value, beta_c.value, beta_f.value)
    _T = np.linspace(0.25, 30, 120)
    _zr = lambda c: np.array([-np.log(c.curves.Z0("fd", t)) / t for t in _T]) * 1e4
    _sh = _zr(_lib) - _zr(cal)       # Libor-coupon calibration minus compounded-coupon calibration, bp
    _fig = go.Figure(line(_T, _sh, C["lgcm"], "Libor − compounded", show=False))
    _fig.add_hline(y=0, line=dict(color=C["grey"], width=1))
    _fig.update_layout(title=dict(text="Z<sup>f|d</sup> zero rate: Libor coupon minus compounded coupon", x=0))
    _fig.update_xaxes(title_text="maturity (years)")
    _fig.update_yaxes(title_text="difference (bp)")
    style(_fig, width=720, height=380)
    _fig.update_layout(margin=dict(t=50))
    mo.vstack([mo.md(f"""## Libor versus compounded coupon in the calibration: {market.value}

    The same quotes are calibrated twice: with the market's own compounded overnight coupon (the exact daily
    coupon) and with the Libor coupon $K$ in its place (the floating index replaced by a Libor-type index of the
    same tenor and zero spread; this is the mis-specification the compounded-coupon pricers fix).
    The $Z^{{f|d}}$ zero-rate shift (Libor-coupon calibration minus compounded-coupon calibration) is **{_sh[-1]:.4f} bp at 30Y** (largest absolute shift {np.max(np.abs(_sh)):.4f} bp).
    That is a measurement, not a bound: the coupon type barely matters for the collateral convexity in these
    markets, because $K-1$ itself is small (see above) and the two coupons differ by only a relative amount of
    $10^{{-8}}$ to $10^{{-7}}$ (about $2\\cdot10^{{-8}}$ in $\\ln K$ on the calibrated GBPUSD model, $10^{{-7}}$ on the test model of the notes; see the coupon comparison above), far below any market tick. The point of the panel is this robustness, not a correction.
    """), _fig])
    return


@app.cell
def _(beta_c, beta_f, build, cbscale, impact, market, mo, run_impact):
    mo.stop(not run_impact.value, mo.md("## Parameter impact: $\\rho^{fc}$\n\nPress *Run parameter-impact table* above "
                                        "(about 5.5 s per uncached row)."))
    _rows = []
    for _r in (-0.8, -0.6, -0.4, 0.0, 0.4):
        try:
            build(market.value, _r, cbscale.value, beta_c.value, beta_f.value)    # only the spec construction is guarded
        except ValueError as _e:
            _rows.append({"rho^{fc}": _r, "30Y Z^{f|d} shift vs DCB (bp)": f"n/a ({_e})",
                          "lambda^c at 10Y (bp/yr)": "n/a", "10Y FX fwd collateral convexity (bp)": "n/a"})
            continue
        _s, _l, _f = impact(market.value, _r, cbscale.value, beta_c.value, beta_f.value)   # genuine errors propagate
        _rows.append({"rho^{fc}": _r, "30Y Z^{f|d} shift vs DCB (bp)": round(_s, 3),
                      "lambda^c at 10Y (bp/yr)": round(_l, 3), "10Y FX fwd collateral convexity (bp)": round(_f, 3)})
    mo.vstack([mo.md(f"""## Parameter impact on {market.value}: $\\rho^{{fc}}$

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
