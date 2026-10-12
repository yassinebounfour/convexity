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

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from convexity import repo

    # colour-blind-safe palette (Okabe-Ito), one colour per object across all panels
    C = {"bond": "#0072B2", "repo": "#D55E00", "ois": "#009E73", "M": "#CC79A7", "F": "#E69F00",
         "exact": "#000000", "approx": "#56B4E9", "grey": "#666666"}
    DASH = {"-": "solid", "--": "dash", ":": "dot"}

    def style(fig, width=1000, height=380):
        """Common look: white background, light grid, legend above the panels."""
        fig.update_layout(template="simple_white", width=width, height=height, font=dict(size=12),
                          margin=dict(l=60, r=20, t=80, b=55), hovermode="x unified",
                          legend=dict(orientation="h", x=0, y=1.14, xanchor="left", yanchor="bottom"))
        fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", ticks="outside")
        fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8", ticks="outside")
        return fig

    def panel_legends(fig, n):
        """One horizontal legend above each of the n panels (traces pick theirs with legend="legend2", ...)."""
        for k in range(1, n + 1):
            x0 = fig.layout[f"xaxis{k if k > 1 else ''}"].domain[0]
            fig.update_layout({f"legend{k if k > 1 else ''}": dict(orientation="h", x=x0, y=1.14, xanchor="left",
                                                                  yanchor="bottom", font=dict(size=11))})
        return fig

    def line(x, y, color, name, dash="-", show=True, group=None, **kw):
        return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, dash=DASH.get(dash, dash), width=2),
                          name=name, showlegend=show, legendgroup=group or name, **kw)
    return C, go, line, make_subplots, mo, np, panel_legends, repo, style


@app.cell
def _(mo):
    mo.md(r"""
    # Repo convexity

    McCloud (*Repo convexity*, 2019) explains the basis between **repo** discount factors (implied from repo
    rates) and **bond** discount factors (stripped from bond prices) as a convexity: the repo is discounted at
    the derivative (OIS) rate $r = \bar r + b$, while the bond is priced at its own rate $\bar r$.
    In the notes (section "Bonds, repo and repo convexity"), the repo rate is a forward collateral convexity:

    $$1 + \delta f^{eT}_{ts} = \frac{\bar p^s_t}{\bar p^e_t}\,K^{\mathrm{repo}}_t(s,e,T),\qquad
    \ln K = -\int_t^s(\bar\Sigma_{u,e}-\bar\Sigma_{u,s})\cdot\Sigma^b_{u,e}du-\int_s^e(\bar\Sigma_{u,e}-\bar\Sigma_{u,T})\cdot\Sigma^b_{u,e}du .$$

    Model: Hull–White bond rate $(\theta,\sigma)$ and basis $(\kappa,\varepsilon)$ with correlation $\rho$; flat
    bond curve 3.5%, basis −20bp. Parameters are illustrative, not calibrated. Rates are ACT/360.
    "Convexity" below is the repo rate minus the bond forward rate, in bp.
    """)
    return


@app.cell
def _(mo):
    rho = mo.ui.slider(-0.9, 0.9, step=0.1, value=-0.5, label="ρ (bond rate, basis)")
    sigma = mo.ui.slider(0.002, 0.03, step=0.001, value=0.01, label="σ bond rate vol")
    eps = mo.ui.slider(0.001, 0.02, step=0.001, value=0.008, label="ε basis vol")
    theta = mo.ui.slider(0.01, 0.5, step=0.01, value=0.05, label="θ bond mean reversion")
    kappa = mo.ui.slider(0.05, 2.0, step=0.05, value=0.4, label="κ basis mean reversion")
    mo.vstack([mo.md("**Parameters**"), mo.hstack([rho, sigma, eps]), mo.hstack([theta, kappa])])
    return eps, kappa, rho, sigma, theta


@app.cell
def _(eps, kappa, repo, rho, sigma, theta):
    rm = repo.RepoModel.hull_white(theta.value, sigma.value, kappa.value, eps.value, rho.value)
    cv = repo.BondCurves.flat(rbar=0.035, b=-0.002)

    def conv_bp(model, s, e, T):
        fwd = (cv.Zbond(s) / cv.Zbond(e) - 1.0) / repo.accrual(s, e)
        return 1e4 * (repo.repo_rate(model, cv, s, e, T) - fwd)
    return conv_bp, cv, rm


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Where the convexity comes from: forwardness, term, residual maturity

    McCloud's three intervals: forwardness $\tau = s - t$, repo term $e - s$, residual maturity $\mu = T - e$
    of the bond. The convexity splits into the **maturity adjustment** $M$ (absorbed in the repo discount factors
    $\hat p^{eT} = \bar p^e e^{-M^{eT}}$) and the **forwardness adjustment** $F$ (on the rate only).
    """)
    return


@app.cell
def _(C, conv_bp, go, line, make_subplots, np, panel_legends, repo, rm, style):
    _fig = make_subplots(1, 3, horizontal_spacing=0.08, subplot_titles=[
        "vs forwardness (bond 10Y after e)", "vs residual maturity (3M repo)", "split, 3M repo on a bond 10Y after s"])
    _tau = np.linspace(0.0, 10.0, 41)
    for _d, _ls in [(1 / 12, ":"), (0.25, "--"), (1.0, "-")]:
        _fig.add_trace(line(_tau, [conv_bp(rm, t, t + _d, t + _d + 10.0) for t in _tau], C["repo"],
                            f"term {12 * _d:.0f}M", _ls), 1, 1)
    _mu = np.linspace(0.0, 30.0, 61)
    for _s, _ls in [(0.0, "-"), (2.0, "--"), (5.0, ":")]:
        _fig.add_trace(line(_mu, [conv_bp(rm, _s, _s + 0.25, _s + 0.25 + m) for m in _mu], C["repo"],
                            f"start {_s:.0f}Y", _ls, legend="legend2"), 1, 2)
    _starts = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    _e, _T = 0.25, 10.0
    _Mp = [1e4 * (repo.maturity_adjustment(rm, 0, s + _e, s + _T) - repo.maturity_adjustment(rm, 0, s, s + _T)) / repo.accrual(s, s + _e)
           for s in _starts]
    _Fp = [1e4 * repo.forwardness_adjustment(rm, 0, s, s + _e, s + _T) / repo.accrual(s, s + _e) if s > 0 else 0.0
           for s in _starts]
    _lab = [f"{s:.0f}Y" for s in _starts]
    _fig.add_trace(go.Bar(x=_lab, y=_Mp, name="maturity M", marker_color=C["M"], legend="legend3"), 1, 3)
    _fig.add_trace(go.Bar(x=_lab, y=_Fp, name="forwardness F", marker_color=C["F"], legend="legend3"), 1, 3)
    _fig.update_layout(barmode="group", bargap=0.3)
    _fig.add_hline(y=0, line=dict(color=C["grey"], width=1), row=1, col=3)
    _fig.update_xaxes(title_text="forward start s (years)", row=1, col=1)
    _fig.update_xaxes(title_text="T − e (years)", row=1, col=2)
    _fig.update_xaxes(title_text="forward start", row=1, col=3)
    _fig.update_yaxes(title_text="convexity (bp)", row=1, col=1)
    _fig.update_yaxes(title_text="contribution (bp, ≈ ln K / δ)", row=1, col=3)
    style(_fig, width=1250)
    panel_legends(_fig, 3)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Sensitivity to the parameters

    The convexity is linear in $\rho\sigma\varepsilon$ and vanishes at $\rho = 0$. The mean reversions set its
    shape: $\kappa$ how quickly a basis shock decays, $\theta$ how much of a bond rate shock reaches long bonds.
    """)
    return


@app.cell
def _(C, conv_bp, eps, line, make_subplots, np, panel_legends, repo, sigma, style, theta):
    _fig = make_subplots(1, 2, horizontal_spacing=0.1,
                         subplot_titles=["3M repo in 2Y on a 12Y bond", "vs θ (ρ = −0.5, κ = 0.4)"])
    _rhos = np.linspace(-0.9, 0.9, 19)
    for _k, _ls in [(0.1, ":"), (0.4, "--"), (1.5, "-")]:
        _fig.add_trace(line(_rhos, [conv_bp(repo.RepoModel.hull_white(theta.value, sigma.value, _k, eps.value, r), 2.0, 2.25, 12.25)
                                    for r in _rhos], C["repo"], f"κ = {_k}", _ls), 1, 1)
    _fig.add_hline(y=0, line=dict(color=C["grey"], width=1), row=1, col=1)
    _ths = np.linspace(0.01, 0.5, 25)
    for _mu, _ls in [(2.0, ":"), (10.0, "--"), (30.0, "-")]:
        _fig.add_trace(line(_ths, [conv_bp(repo.RepoModel.hull_white(th, sigma.value, 0.4, eps.value, -0.5), 2.0, 2.25, 2.25 + _mu)
                                   for th in _ths], C["repo"], f"T − e = {_mu:.0f}Y", _ls, legend="legend2"), 1, 2)
    _fig.update_xaxes(title_text="ρ", row=1, col=1)
    _fig.update_xaxes(title_text="θ", row=1, col=2)
    _fig.update_yaxes(title_text="convexity (bp)", row=1, col=1)
    _fig.update_yaxes(title_text="convexity (bp)", row=1, col=2)
    style(_fig)
    panel_legends(_fig, 2)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Bond curve, repo curve, and extrapolating the repo curve

    The infinite-maturity repo curve $\hat p^e = \bar p^e e^{-M^{e\infty}}$ differs from the bond curve by the
    maturity adjustment. If repo quotes stop at $E$, McCloud's (39) extrapolates the forward spread
    $\hat f - \bar f$ using the bond forwards and $(\theta, \kappa)$ only. In Hull–White it is exact (dashed line on
    the solid line).
    """)
    return


@app.cell
def _(C, cv, eps, kappa, line, make_subplots, np, panel_legends, repo, rho, rm, sigma, style, theta):
    _e = np.linspace(0.05, 30.0, 120)
    _pb = np.array([float(cv.Zbond(x)) for x in _e])
    _ph = np.array([repo.repo_discount_factor(rm, cv, x, x + 1e3) for x in _e])
    _po = np.array([float(cv.Zois(x)) for x in _e])
    _fig = make_subplots(1, 2, horizontal_spacing=0.1, subplot_titles=["zero rates", "instantaneous forward spread f̂ − f̄"])
    for _p, _k, _lab in [(_pb, "bond", "bond curve p̄"), (_ph, "repo", "repo curve p̂ (T = ∞)"), (_po, "ois", "OIS curve p")]:
        _fig.add_trace(line(_e, -1e2 * np.log(_p) / _e, C[_k], _lab), 1, 1)
    _P = (rho.value, sigma.value, eps.value, theta.value, kappa.value)
    _sp = [1e4 * repo.hw_forward_spread(*_P, 0.0, x) for x in _e]
    _E = 2.0
    _xe = _e[_e >= _E]
    _ext = [1e4 * repo.hw_extrapolate(repo.hw_forward_spread(*_P, 0.0, _E), 0.0, _E, x, theta.value, kappa.value) for x in _xe]
    _fig.add_trace(line(_e, _sp, C["repo"], "model spread", legend="legend2"), 1, 2)
    _fig.add_trace(line(_xe, _ext, C["exact"], f"extrapolated from E = {_E:.0f}Y", "--", legend="legend2"), 1, 2)
    _fig.add_vline(x=_E, line=dict(color=C["grey"], width=1), row=1, col=2)
    _fig.update_xaxes(title_text="maturity (years)", row=1, col=1)
    _fig.update_xaxes(title_text="maturity (years)", row=1, col=2)
    _fig.update_yaxes(title_text="zero rate (%)", row=1, col=1)
    _fig.update_yaxes(title_text="spread (bp)", row=1, col=2)
    style(_fig)
    panel_legends(_fig, 2)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Coupon bonds

    Repo on a 10Y bond with semi-annual coupons over [1.2Y, 1.6Y] (the coupon at 1.5Y falls in the window and is
    passed through to the bond's owner, discounted at $r$). The gap to the bond forward has two parts:

    - the **ρ = 0 part** (grey): the passed-through coupon is discounted at $r$ while the bonds are priced at $\bar r$, so
      the basis moves the rate even without any correlation;
    - the **convexity** (the rest), driven by the correlation of the bond rate and the basis.

    Exact: Gauss–Hermite quadrature (Proposition "Coupon repo rate"). Approximation: frozen weights with the control
    $\mathbb E[D/D] = 1$ (Definition "Frozen weights"); its error in the rate grows like vol⁴ (measured, not proved).
    """)
    return


@app.cell
def _(C, cv, eps, go, kappa, np, repo, rm, sigma, style, theta):
    _cps = np.linspace(0.0, 0.08, 9)
    _s, _e = 1.2, 1.6
    _fwd = (cv.Zbond(_s) / cv.Zbond(_e) - 1.0) / repo.accrual(_s, _e)
    _r0 = repo.RepoModel.hull_white(theta.value, sigma.value, kappa.value, eps.value, 0.0)
    _ex, _ap, _z0 = [], [], []
    for _c in _cps:
        _b = repo.CouponBond.fixed(10.0, _c, 2) if _c > 0 else repo.CouponBond((10.0,), (1.0,))
        _base = repo.coupon_repo_rate_exact(_r0, cv, _b, _s, _e)
        _z0.append(1e4 * (_base - _fwd))
        _ex.append(1e4 * (repo.coupon_repo_rate_exact(rm, cv, _b, _s, _e) - _base))
        _ap.append(1e4 * (repo.coupon_repo_rate(rm, cv, _b, _s, _e) - repo.coupon_repo_rate(_r0, cv, _b, _s, _e)))
    _gap = max(abs(a - x) for a, x in zip(_ap, _ex))
    _x = 1e2 * _cps
    _fig = go.Figure([
        go.Scatter(x=_x, y=_z0, mode="lines+markers", line=dict(color=C["grey"], width=2), marker=dict(symbol="square", size=7),
                   name="ρ = 0 part (basis on the passed-through coupon)"),
        go.Scatter(x=_x, y=_ex, mode="lines+markers", line=dict(color=C["exact"], width=2), marker=dict(size=7),
                   name="convexity, exact (quadrature)"),
        go.Scatter(x=_x, y=_ap, mode="lines", line=dict(color=C["approx"], width=2, dash="dash"), name="convexity, frozen weights")])
    _fig.add_hline(y=0, line=dict(color=C["grey"], width=1))
    _fig.update_layout(title=dict(text=f"repo rate minus bond forward, 10Y coupon bond (frozen weights off by at most {_gap:.4f}bp)",
                                  font=dict(size=13), x=0, y=0.97))
    _fig.update_xaxes(title_text="coupon (%)")
    _fig.update_yaxes(title_text="contribution (bp)")
    style(_fig, width=800, height=440)
    _fig.update_layout(margin=dict(t=110), legend=dict(orientation="v", x=0, y=1.02, yanchor="bottom"),
                       title=dict(y=0.98))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Monte Carlo check

    Exact LGCM simulation, $2^{18}$ antithetic paths. The control variates have exact means: $\mathbb E[D_{0,e}] = p_{0,e}$,
    and the same repo discounted at $\bar r$ (bond collateral) is worth $\bar p_{0,s}$.
    """)
    return


@app.cell
def _(cv, mo, repo, rm):
    _rows = []
    for _lab, _bond, _s, _e in [("ZCB 10Y, 3M in 1Y", 10.0, 1.0, 1.25), ("ZCB 30Y, 1Y in 5Y", 30.0, 5.0, 6.0),
                                ("10Y 5% s.a., [1.2Y, 1.6Y]", repo.CouponBond.fixed(10.0, 0.05, 2), 1.2, 1.6)]:
        _r, _se = repo.mc_repo_rate(rm, cv, _bond, _s, _e, n_paths=2**18, seed=11)
        _cf = (repo.repo_rate(rm, cv, _s, _e, _bond) if not isinstance(_bond, repo.CouponBond)
               else repo.coupon_repo_rate_exact(rm, cv, _bond, _s, _e))
        _rows.append({"repo": _lab, "formula (%)": f"{1e2 * _cf:.5f}", "MC (%)": f"{1e2 * _r:.5f}",
                      "s.e. (bp)": f"{1e4 * _se:.4f}", "z": f"{(_r - _cf) / _se:+.2f}"})
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Bond-collateralised derivatives: which curve?

    With the collateral financed by rolling **overnight** repo, the collateral earns the bond rate $\bar r$
    (Proposition "Overnight repo") and the discount curve is the **bond curve** $\bar p$ (Proposition "Discount curve
    of bond-collateralised derivatives"). Discounting on the extrapolated **term repo** curve $\hat p$, as McCloud §4
    suggests, assumes term financing locked at inception. Below: the PV of a 10Y annual fixed leg of 1% on 100m notional
    under the three curves (fixed flows only, so no index convexity enters).
    """)
    return


@app.cell
def _(cv, mo, np, repo, rm):
    _T = np.arange(1.0, 11.0)
    _pv = {"bond curve p̄": sum(1e6 * float(cv.Zbond(t)) for t in _T),
           "term repo curve p̂": sum(1e6 * repo.repo_discount_factor(rm, cv, t, t + 1e3) for t in _T),
           "OIS p": sum(1e6 * float(cv.Zois(t)) for t in _T)}
    _ref = _pv["bond curve p̄"]
    mo.ui.table([{"discount curve": k, "PV": f"{v:,.0f}", "vs bond curve": f"{v - _ref:+,.0f}"} for k, v in _pv.items()],
                selection=None)
    return


if __name__ == "__main__":
    app.run()
