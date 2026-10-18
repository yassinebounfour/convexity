"""Figures and tables of the notes: global curve fit, correlations, caplets.

Run from python/:  python -m convexity.figures [curves] [correlations] [caplets]   (all groups: about 2.5 min)
Plotly figures exported as vector PDFs (kaleido) and LaTeX table fragments (\\input by docs/CollateralDiscounting.tex) to docs/figures/.
Every number shown in these figures and tables is computed here.
"""
import dataclasses
import pathlib
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .calibration import calibrate, default_spec
from .correlations import LogFX, SwapRate, correlation_approx, correlation_exact, correlation_mc
from .curve_calibration import calibrate_curves
from .curve_fitting import GlobalFit, calibrate_curves_global, unit
from .curves import SpreadCurve
from .instruments import CurrencyBasisSwaption, FXOption, Swap, Swaption
from .lgcm import default_model
from .market import LIBOR, Coupon, InitialCurves, black, caplet_total_variance
from .montecarlo import Simulator, cv_mean_and_stderr
from .schemes import ExactScheme
from .market_data import MARKETS, example_quotes
from .pricers import DeterministicPricer

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "figures"
# categorical slots 1-4 of the validated reference palette, in fixed order; each series also has its own
# dash pattern or marker symbol, so identity never rests on colour alone (print, CVD)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FONT = dict(family="Times New Roman, serif", size=19, color=INK)
W, H = 1000, 380          # px; the PDFs are vector and included at \textwidth


def _layout(fig, legend_y=1.18, **kw):
    fig.update_layout(template="simple_white", font=FONT, width=kw.pop("width", W), height=kw.pop("height", H),
                      margin=dict(l=75, r=15, t=kw.pop("top", 90), b=80), plot_bgcolor="white",
                      legend=dict(orientation="h", x=0, y=legend_y, xanchor="left", yanchor="bottom",
                                  font=dict(size=17)), **kw)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor=MUTED, tickcolor=MUTED, ticks="outside", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=MUTED, tickcolor=MUTED, ticks="outside", zeroline=False)
    for ann in fig.layout.annotations:           # subplot titles: left aligned, plain ink
        ann.update(font=dict(size=18, color=INK))
    return fig


def _line(x, y, color, name, dash="solid", width=2.6, show=True, group=None, **kw):
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, dash=dash, width=width), name=name,
                      showlegend=show, legendgroup=group or name, **kw)


def _curve_quotes(market="EURUSD", futures=False):
    return example_quotes(market, futures).without(Swaption, FXOption, CurrencyBasisSwaption)


def _forward(curve, t):
    h = curve.spread if isinstance(curve, SpreadCurve) else curve
    return h.instantaneous_forward(t) if hasattr(h, "instantaneous_forward") else h.forward(t)


def _save(fig, name):
    fig.write_image(OUT / name)


def _write(name, text):
    (OUT / name).write_text(text)


# ---------------------------------------------------------------------- global curve fit
def fig_curve_forwards():
    """Instantaneous forwards, bootstrap against hybrid (EURUSD with the SOFR futures, DCB)."""
    mq = _curve_quotes(futures=True)
    boot, _ = calibrate_curves(mq)
    glob, _ = calibrate_curves_global(mq)
    t = np.linspace(1e-3, 35.0, 3501)
    s = mq.setup
    panels = [(s.key("dd"), "Z<sup>d|d</sup> (USD, SOFR): r<sup>d|d</sup><sub>0,t</sub>"),
              (s.key("ff"), "Z<sup>f|f</sup> (EUR, ESTR): r<sup>f|f</sup><sub>0,t</sub>"),
              (s.key("fd"), "collateral basis: q<sup>f|f→d</sup><sub>0,t</sub>")]
    fig = make_subplots(1, 3, subplot_titles=[p[1] for p in panels], horizontal_spacing=0.07)
    for k, (key, _) in enumerate(panels, start=1):
        fb = np.array([float(_forward(boot[key], x)) for x in t])
        fg = _forward(glob[key], t)
        fig.add_trace(_line(t, 100 * fb, ORANGE, "bootstrap (flat forwards)", "dash", 1.6, show=k == 1), 1, k)
        fig.add_trace(_line(t, 100 * fg, BLUE, "global fit (hybrid curve)", show=k == 1), 1, k)
        pillars = sorted({q.instrument.pillar for q in mq.for_target(key)})
        lo = 100 * min(fb.min(), fg.min())
        fig.add_trace(go.Scatter(x=pillars, y=np.full(len(pillars), lo), mode="markers", name="pillars",
                                 marker=dict(symbol="line-ns-open", size=9, color=MUTED, line=dict(width=1.5)),
                                 showlegend=k == 1, legendgroup="pillars"), 1, k)
        fig.update_xaxes(title_text="t (years)", row=1, col=k)
    fig.update_yaxes(title_text="forward (%)", row=1, col=1)
    _save(_layout(fig), "curve_forwards.pdf")


def fig_curve_futures_locality():
    """(a) USD residuals with futures; (b) forward response to a 1 bp bump of the 10Y USD swap."""
    mq = _curve_quotes(futures=True)
    key = mq.setup.key("dd")
    boot, _ = calibrate_curves(mq)
    glob, _ = calibrate_curves_global(mq)
    usd = [q for q in mq.quotes if q.instrument.target(mq.setup) == key and q.instrument.pillar <= 2.0 + 1e-9]
    usd.sort(key=lambda q: (isinstance(q.instrument, Swap), q.instrument.pillar))
    pb, pg = DeterministicPricer(boot), DeterministicPricer(glob)
    rb = [(q.instrument.quote(pb) - q.value) / unit(q.instrument) for q in usd]
    rg = [(q.instrument.quote(pg) - q.value) / unit(q.instrument) for q in usd]
    labels = [q.instrument.label.replace("USD_SOFR ", "").replace("future ", "fut ") for q in usd]

    q0 = _curve_quotes()
    bump = q0.with_values([q.value + (1e-4 if isinstance(q.instrument, Swap) and q.instrument.index.name == "USD_SOFR"
                                       and q.instrument.end == 10 else 0.0) for q in q0.quotes])
    t = np.linspace(1e-3, 30.0, 3001)
    k0 = q0.setup.key("dd")
    fwd = lambda c: np.array([float(_forward(c[k0], x)) for x in t])
    resp = {name: 1e4 * (fwd(f(bump)[0]) - fwd(f(q0)[0]))
            for name, f in (("bootstrap", calibrate_curves), ("global", calibrate_curves_global))}

    fig = make_subplots(1, 2, column_widths=[0.52, 0.48], horizontal_spacing=0.16,
                        subplot_titles=["(a) USD quotes up to 2Y, with SOFR futures", "(b) 10Y USD swap bumped by 1 bp"])
    fig.add_trace(go.Scatter(x=rb, y=labels, mode="markers", name="bootstrap", legendgroup="b",
                             marker=dict(symbol="square", size=9, color=ORANGE)), 1, 1)
    fig.add_trace(go.Scatter(x=rg, y=labels, mode="markers", name="global fit", legendgroup="g",
                             marker=dict(symbol="circle", size=9, color=BLUE)), 1, 1)
    fig.add_vline(x=0.0, line=dict(color=MUTED, width=1), row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="model − market (bp of rate)", row=1, col=1)
    fig.add_trace(_line(t, resp["bootstrap"], ORANGE, "bootstrap", "dash", 1.6, show=False, group="b"), 1, 2)
    fig.add_trace(_line(t, resp["global"], BLUE, "global fit", show=False, group="g"), 1, 2)
    fig.add_hline(y=0.0, line=dict(color=MUTED, width=1), row=1, col=2)
    fig.update_xaxes(title_text="t (years)", row=1, col=2)
    fig.update_yaxes(title_text="change of r<sup>d|d</sup><sub>0,t</sub> (bp)", row=1, col=2)
    _save(_layout(fig, height=420), "curve_futures_locality.pdf")


def table_curve_fit():
    """Per market: DCB fit and LGCM calibration, bootstrap against global fit."""
    rows = []
    for mk in MARKETS:
        mq = _curve_quotes(mk)
        t0 = time.perf_counter(); calibrate_curves(mq); tb = time.perf_counter() - t0
        t0 = time.perf_counter(); g, rep = calibrate_curves_global(mq); tg = time.perf_counter() - t0
        full = example_quotes(mk)
        cb = calibrate(full)
        cg = calibrate(full, dataclasses.replace(default_spec(), curves=GlobalFit()))
        dcb_b, dcb_g = calibrate_curves(full)[0], calibrate_curves_global(full)[0]
        z = lambda c, T=30.0: -np.log(c.Z0("fd", T)) / T * 1e4
        dvol = max(np.abs(cg.levels(b)[1] / cb.levels(b)[1] - 1).max() for b in "dfcx")
        rows.append(f"{mk} & {rep.max_abs_residual():.1e} & {', '.join(str(l['nfev']) for l in rep.levels)} & "
                    f"{1e3 * tb:.0f} / {1e3 * tg:.0f} & {cb.diagnostics['fixed_point_iterations']} / "
                    f"{cg.diagnostics['fixed_point_iterations']} & {cb.diagnostics['seconds']:.1f} / "
                    f"{cg.diagnostics['seconds']:.1f} & {100 * dvol:.2f} & "
                    f"{z(cb.curves) - z(dcb_b):.2f} / {z(cg.curves) - z(dcb_g):.2f} \\\\")
    _write("table_curve_fit.tex", "\n".join([
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r" & \multicolumn{3}{c}{DCB curves} & \multicolumn{4}{c}{LGCM calibration} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        r"market & max res.\ (bp) & evaluations & ms (B / G) & iterations & s (B / G) & max $|\Delta\lambda|$ (\%) & 30Y conv.\ (bp) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def table_zigzag():
    """A natural cubic spline of the forward through FREE knot values, fitted exactly to the USD swaps of the
    example (flat forward to 1Y), against the hybrid curve fitted to the same swaps."""
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares
    mq = _curve_quotes()
    swaps = mq.for_target(mq.setup.key("dd"))
    knots = np.array([1.0, 2, 3, 5, 7, 10, 15, 20, 30])
    grid = np.linspace(0, 30, 30 * 365 + 1)

    def Z_of(x):
        g = CubicSpline(knots, x, bc_type="natural")
        fwd = np.where(grid <= 1.0, x[0], g(np.clip(grid, 1.0, 30.0)))
        I = np.r_[0.0, np.cumsum(0.5 * (fwd[1:] + fwd[:-1]) * np.diff(grid))]
        return lambda T: np.exp(-np.interp(T, grid, I))

    def res(x):
        Z = Z_of(x)
        return [(lambda fl, A: fl / A)(*q.instrument.legs(Z)) - q.value for q in swaps]
    x = least_squares(res, np.full(9, 0.04), xtol=1e-15, ftol=1e-15, gtol=1e-15).x
    r = calibrate_curves_global(mq)[0][mq.setup.key("dd")].instantaneous_forward(knots)
    fmt = lambda v: " & ".join(f"{100 * u:.2f}" for u in v)
    _write("table_zigzag.tex", "\n".join([
        r"\begin{tabular}{l" + "c" * len(knots) + "}", r"\toprule",
        "knot (years) & " + " & ".join(f"{k:g}" for k in knots) + r" \\", r"\midrule",
        r"spline through knot values (\%) & " + fmt(x) + r" \\",
        r"hybrid curve (\%) & " + fmt(r) + r" \\", r"\bottomrule", r"\end{tabular}"]))


# ---------------------------------------------------------------------- correlations
def _models():
    cal = calibrate(example_quotes())
    return [("test model (2 d factors)", default_model(), InitialCurves()), ("calibrated EURUSD (1 factor per block)", cal.model, cal.curves)]


def fig_corr_term_structure(models):
    T = np.r_[0.5, 1:20.5:1.0]
    Tmc = [1.0, 5.0, 10.0, 15.0]
    series = [(lambda t: (LogFX(t), SwapRate("d", t, 10.0)), BLUE, "solid", "circle-open", "ln x<sub>T</sub>, Y<sup>d</sup><sub>T</sub>"),
              (lambda t: (LogFX(t), SwapRate("f", t, 10.0)), ORANGE, "dash", "square-open", "ln x<sub>T</sub>, Y<sup>f</sup><sub>T</sub>"),
              (lambda t: (SwapRate("d", t, 10.0), SwapRate("f", t, 10.0)), AQUA, "dashdot", "triangle-up-open",
               "Y<sup>d</sup><sub>T</sub>, Y<sup>f</sup><sub>T</sub>")]
    fig = make_subplots(1, 2, shared_yaxes=True, horizontal_spacing=0.05, subplot_titles=[m[0] for m in models])
    for k, (_, m, cv) in enumerate(models, start=1):
        for mk, color, dash, symbol, label in series:
            ex = [correlation_exact(m, cv, *mk(t)).corr for t in T]
            ap = [correlation_approx(m, cv, *mk(t)).corr for t in T]
            mc = [correlation_mc(m, cv, *mk(t), 2**15, seed=1, control=True) for t in Tmc]
            fig.add_trace(_line(T, ex, color, label + " (exact)", dash, show=k == 1, group=label), 1, k)
            fig.add_trace(go.Scatter(x=T, y=ap, mode="markers", marker=dict(color=color, size=4), name="approximation",
                                     showlegend=False, legendgroup=label), 1, k)
            fig.add_trace(go.Scatter(x=Tmc, y=[r.corr for r in mc], mode="markers", showlegend=False, legendgroup=label,
                                     error_y=dict(array=[4 * r.stderr for r in mc], color=color, thickness=1.5, width=0),
                                     marker=dict(symbol=symbol, size=10, color=color, line=dict(width=2))), 1, k)
        fig.update_xaxes(title_text="expiry T (years), 10Y swaps", row=1, col=k)
    fig.update_yaxes(title_text="correlation under Q<sup>d,RN</sup>", row=1, col=1)
    _save(_layout(fig), "corr_term_structure.pdf")


def fig_corr_tenor_intertemporal(models):
    M = np.r_[1.0:30.5:1.0]
    T1 = np.r_[0.5, 1:15.5:0.5]
    fig = make_subplots(1, 2, horizontal_spacing=0.1, subplot_titles=[
        "(a) Y<sup>d</sup><sub>5</sub> of tenor M against Y<sup>d</sup><sub>5</sub> (10Y)",
        "(b) Y<sup>d</sup><sub>T₁</sub> (10Y) or ln x<sub>T₁</sub> against Y<sup>d</sup><sub>5</sub> (10Y)"])
    ref = SwapRate("d", 5.0, 10.0)
    for (title, m, cv), color, dash in zip(models, (BLUE, ORANGE), ("solid", "dash")):
        short = title.split(" (")[0].replace("calibrated ", "")
        fig.add_trace(_line(M, [correlation_exact(m, cv, SwapRate("d", 5.0, x), ref).corr for x in M], color,
                            short + ": Y<sup>d</sup>", dash, group=short), 1, 1)
        fig.add_trace(_line(T1, [correlation_exact(m, cv, SwapRate("d", x, 10.0), ref).corr for x in T1], color,
                            short, dash, show=False, group=short), 1, 2)
        fig.add_trace(_line(T1, [correlation_exact(m, cv, LogFX(x), ref).corr for x in T1], color,
                            short + ": ln x<sub>T₁</sub>", "dot", 1.8), 1, 2)
    fig.add_vline(x=5.0, line=dict(color=MUTED, width=1), row=1, col=2)
    fig.update_xaxes(title_text="tenor M (years)", row=1, col=1)
    fig.update_xaxes(title_text="T₁ (years)", row=1, col=2)
    fig.update_yaxes(title_text="correlation (exact, Q<sup>d,RN</sup>)", row=1, col=1)
    _save(_layout(fig, legend_y=1.17, top=110), "corr_tenor_intertemporal.pdf")


PAIRS = [(LogFX(5.0), SwapRate("d", 5.0, 10.0), "ln x<sub>5</sub>, Y<sup>d</sup><sub>5</sub>", r"$\ln x_5$, $Y^d_5$"),
         (LogFX(5.0), SwapRate("f", 5.0, 10.0), "ln x<sub>5</sub>, Y<sup>f</sup><sub>5</sub>", r"$\ln x_5$, $Y^f_5$"),
         (SwapRate("d", 5.0, 2.0), SwapRate("d", 5.0, 10.0), "Y<sup>d</sup><sub>5</sub> (2Y), Y<sup>d</sup><sub>5</sub>",
          r"$Y^d_5$ ($2$Y), $Y^d_5$"),
         (SwapRate("d", 5.0, 10.0), SwapRate("f", 5.0, 10.0), "Y<sup>d</sup><sub>5</sub>, Y<sup>f</sup><sub>5</sub>", r"$Y^d_5$, $Y^f_5$"),
         (SwapRate("d", 2.0, 10.0), SwapRate("d", 5.0, 10.0), "Y<sup>d</sup><sub>2</sub>, Y<sup>d</sup><sub>5</sub>", r"$Y^d_2$, $Y^d_5$"),
         (LogFX(2.0), SwapRate("d", 5.0, 10.0), "ln x<sub>2</sub>, Y<sup>d</sup><sub>5</sub>", r"$\ln x_2$, $Y^d_5$")]
# swap rates: 10Y tenor, annual fixed, quarterly floating, unless stated


def fig_corr_convergence():
    lam = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
    cv = InitialCurves()
    styles = [(BLUE, "circle", "solid"), (ORANGE, "square", "dash"), (AQUA, "triangle-up", "dashdot"),
              (YELLOW, "diamond", "dot"), (BLUE, "triangle-down", "dot"), (ORANGE, "cross", "dashdot")]
    fig = make_subplots(1, 2, shared_yaxes=True, horizontal_spacing=0.05, subplot_titles=[
        "(a) approximation − exact (Q<sup>d,RN</sup>)", "(b) Q<sup>d|d,15</sup> − Q<sup>d,RN</sup> (exact)"])
    for (a, b, label, _), (color, symbol, dash) in zip(PAIRS, styles):
        ms = [default_model(s) for s in lam]
        ex = np.array([correlation_exact(m, cv, a, b).corr for m in ms])
        ap = np.array([correlation_approx(m, cv, a, b).corr for m in ms])
        tf = np.array([correlation_exact(m, cv, a, b, "Tfwd", 15.0).corr for m in ms])
        for k, y in ((1, np.abs(ap - ex)), (2, np.abs(tf - ex))):
            fig.add_trace(go.Scatter(x=lam, y=np.maximum(y, 1e-12), mode="lines+markers", name=label, legendgroup=label,
                                     showlegend=k == 1, line=dict(color=color, dash=dash, width=1.8),
                                     marker=dict(symbol=symbol, size=7, color=color)), 1, k)
    for k in (1, 2):
        fig.add_trace(_line(lam, 2e-4 * lam ** 2, MUTED, "∝ λ²", "dot", 1.2, show=k == 1, group="ref"), 1, k)
        fig.update_xaxes(type="log", title_text="vol scale λ", tickvals=list(lam), ticktext=[f"{x:g}" for x in lam], row=1, col=k)
    fig.update_yaxes(type="log", range=[-9, -2], exponentformat="power", title_text="|difference of correlations|", row=1, col=1)
    fig.update_yaxes(type="log", range=[-9, -2], exponentformat="power", row=1, col=2)
    _save(_layout(fig, legend_y=1.17, top=110), "corr_convergence.pdf")


def table_corr_verification():
    m, cv = default_model(), InitialCurves()
    rows = []
    for a, b, _, label in PAIRS:
        ex = correlation_exact(m, cv, a, b)
        e = [correlation_approx(default_model(s), cv, a, b).corr - correlation_exact(default_model(s), cv, a, b).corr
             for s in (0.5, 1.0, 2.0)]
        p = correlation_mc(m, cv, a, b, 2**17, seed=1)
        c = correlation_mc(m, cv, a, b, 2**17, seed=1, control=True)
        ct = correlation_mc(m, cv, a, b, 2**17, seed=1, control=True, measure="Tfwd", Tstar=15.0)
        ext = correlation_exact(m, cv, a, b, "Tfwd", 15.0)
        ratio = f"{e[1] / e[0]:.2f}, {e[2] / e[1]:.2f}" if abs(e[1]) > 1e-12 else "exact"
        rows.append(f"{label} & {ex.corr:.5f} & {e[1]:+.1e} & {ratio} & {(p.corr - ex.corr) / p.stderr:+.2f} & {p.stderr:.1e} & "
                    f"{(c.corr - ex.corr) / c.stderr:+.2f} & {c.stderr:.1e} & {(correlation_approx(m, cv, a, b).corr - c.corr) / c.stderr:+.0f} & "
                    f"{(ct.corr - ext.corr) / ct.stderr:+.2f} \\\\")
    _write("table_corr_verification.tex", "\n".join([
        r"\begin{tabular}{lccccccccc}", r"\toprule",
        r" & & & & \multicolumn{2}{c}{MC plain} & \multicolumn{3}{c}{MC control} & $Q^{d|d,15}$ \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(lr){10-10}",
        r"pair & exact & approx $-$ exact & ratio $\lambda\times2$ & $z$ & s.e. & $z$ & s.e. & approx ($z$) & $z$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


# ---------------------------------------------------------------------- caplets
def fig_caplet_vol_ratio(models):
    """sqrt(w_RFR / w_Libor): vol of the compounded (daily) caplet over that of the Libor caplet on the same
    period, domestic block, against the start date; Ho-Lee reference sqrt(1 + theta / (3 a))."""
    starts = np.r_[0.25, 0.5, 0.75, 1:20.5:0.5]
    fig = make_subplots(1, 2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=["(a) 3M periods", "(b) 1Y periods"])
    for k, th in enumerate((0.25, 1.0), start=1):
        for (title, m, _), color, dash in zip(models, (BLUE, ORANGE), ("solid", "dash")):
            r = [np.sqrt(caplet_total_variance(m, "d", a, a + th, Coupon("daily")) / caplet_total_variance(m, "d", a, a + th, LIBOR))
                 for a in starts]
            fig.add_trace(_line(starts, r, color, title.split(" (")[0], dash, show=k == 1, group=title), 1, k)
        fig.add_trace(_line(starts, np.sqrt(1 + th / (3 * starts)), MUTED, "Ho–Lee: √(1 + θ/(3a))", "dot", 1.8,
                            show=k == 1, group="hl"), 1, k)
        fig.update_xaxes(type="log", title_text="period start a (years)", tickvals=[0.25, 0.5, 1, 2, 5, 10, 20],
                         ticktext=["0.25", "0.5", "1", "2", "5", "10", "20"], row=1, col=k)
    fig.update_yaxes(title_text="vol ratio √(w<sub>RFR</sub>/w<sub>Libor</sub>)", row=1, col=1)
    _save(_layout(fig), "caplet_vol_ratio.pdf")


def table_caplet_verification():
    """ATM caplets on the test model, weekly compounding: closed form, Monte Carlo with exact-mean controls
    (D C and D), z-score, and the z-score of the price computed with the Libor variance (fixing at a)."""
    m, cv = default_model(), InitialCurves()
    sim = Simulator(m, scheme=ExactScheme(0.25))
    rows = []
    for block, a, b in (("d", 0.25, 0.5), ("d", 2.0, 2.25), ("d", 10.0, 10.25), ("f", 1.0, 1.25), ("f", 5.0, 6.0)):
        cp = Coupon("daily", 1 / 52)
        g, kind = cp.grid(a, b), block * 2
        st = sim.simulate_dates(list(g), 2**18, seed=5)
        C = np.exp(sum(-np.log(sim.discount(st(t0), cv, kind, t1)) for t0, t1 in zip(g[:-1], g[1:])))
        D = sim.deflator(st(b), cv, block, block)
        Za, Zb = cv.Z0(kind, a), cv.Z0(kind, b)
        F = Za / Zb
        est, se = cv_mean_and_stderr(D * np.maximum(C - F, 0.0), [D * C, D], [Za, Zb])
        w, wl = caplet_total_variance(m, block, a, b, cp), caplet_total_variance(m, block, a, b, LIBOR)
        closed, libor = Zb * black(F, F, w), Zb * black(F, F, wl)
        rows.append(f"${block}$ & $[{a:g}, {b:g}]$ & {np.sqrt(w / wl):.4f} & {1e4 * closed:.4f} & {1e4 * est:.4f} & "
                    f"{1e4 * se:.4f} & {(est - closed) / se:+.2f} & {(est - libor) / se:+.1f} \\\\")
    _write("table_caplet_verification.tex", "\n".join([
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"block & period & $\sqrt{w_{\mathrm{RFR}}/w_{\mathrm{Libor}}}$ & closed form (bp) & MC (bp) & s.e. (bp) & $z$ & $z$ with $w_{\mathrm{Libor}}$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


GROUPS = ("curves", "correlations", "caplets")


def main(groups=GROUPS):
    """Regenerate the figures and tables of the given groups (default: all)."""
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if "curves" in groups:
        fig_curve_forwards()
        fig_curve_futures_locality()
        table_curve_fit()
        table_zigzag()
    models = _models() if {"correlations", "caplets"} & set(groups) else None
    if "correlations" in groups:
        fig_corr_term_structure(models)
        fig_corr_tenor_intertemporal(models)
        fig_corr_convergence()
        table_corr_verification()
    if "caplets" in groups:
        fig_caplet_vol_ratio(models)
        table_caplet_verification()
    print(f"figures and tables written to {OUT} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or GROUPS)
