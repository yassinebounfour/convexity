"""Global curve calibration: hybrid interpolation and nested non-linear least squares.

The alternative to the sequential bootstrap of curve_calibration (which is kept unchanged, for comparison).

Interpolation (curves.HybridCurve): each curve is parametrised by its instantaneous forwards, piecewise
flat on the short end (nodes at the futures or FX forward maturities) and a cubic spline on the long end
(knots at the swap or CCS pillars beyond the short end), continuous at the junction and flat after the
last knot. A curve off native collateral is Z^{e|e} times a hybrid collateral basis curve
(curves.SpreadCurve), as for the bootstrap (Remark "Which curve to interpolate").

Calibration (Algorithm "Nested global curve calibration" of the notes): the target curves are grouped
into nested curve sets by dependency level (level 0: the native curves Z^{d|d}, Z^{f|f}; level 1:
Z^{f|d}, whose quotes read the level 0 curves). The parameters of all curves of a level are found
jointly by Levenberg-Marquardt on the residuals
    (model quote - market quote) / unit      (unit: 1 bp of rate, 0.01 of a futures price, 1 FX pip)
with the curves of the outer levels frozen, plus an optional curvature penalty sqrt(smoothing) * 1e4 *
(g'') in L^2 on each spline section. nested=False solves all levels in a single problem.
With one quote per node the system is square and the fit is exact; with more quotes than nodes (e.g.
1Y and 2Y swaps inside a 2Y futures strip) it is a least squares compromise and the residuals are
reported; with fewer quotes than parameters a positive smoothing is required.

The model only enters through the pricer, as for the bootstrap: DeterministicPricer (DCB) or LGCMPricer.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .curve_calibration import LINEAR, CalibrationError, _target_order
from .curves import CurveSet, DiscountCurveKey, HybridCurve, SpreadCurve
from .instruments import FXForward, RateFuture
from .pricers import DeterministicPricer

SHORT_END = (RateFuture, FXForward)     # instruments whose pillars are flat nodes


@dataclass(frozen=True)
class CurveSpec:
    """Nodes of a HybridCurve: flat forwards up to flat_nodes[-1], cubic spline through spline_knots."""
    flat_nodes: tuple
    spline_knots: tuple = ()

    def build(self):
        return HybridCurve(self.flat_nodes, self.spline_knots)


def default_curve_spec(quotes):
    """Flat nodes at the pillars of the short end instruments (futures, FX forwards), or at the first
    pillar if there are none; spline knots at the remaining pillars beyond the last flat node."""
    pillars = sorted({round(q.instrument.pillar, 12) for q in quotes})
    short = sorted({round(q.instrument.pillar, 12) for q in quotes if isinstance(q.instrument, SHORT_END)})
    short = short or pillars[:1]
    return CurveSpec(tuple(short), tuple(p for p in pillars if p > short[-1] + 1e-10))


def unit(inst):
    """Residual unit of a quote: 1 bp of rate or spread, 0.01 of a futures price (1 bp of rate), 1 FX pip."""
    if isinstance(inst, RateFuture):
        return 1e-2
    if isinstance(inst, FXForward):
        return 1.0 if inst.quotation == "points" else 1.0 / inst.pip_factor
    return 1e-4


@dataclass
class FitRow:
    quote: object
    model: float

    @property
    def label(self):
        return self.quote.instrument.label

    @property
    def residual(self):
        """(model - market) in units of unit(): bp, futures bp, pips."""
        return (self.model - self.quote.value) / unit(self.quote.instrument)


@dataclass
class FitReport:
    used: list = field(default_factory=list)        # every quote enters the objective
    checks: list = field(default_factory=list)      # always empty (kept for the BootstrapReport interface)
    rows: list = field(default_factory=list)
    levels: list = field(default_factory=list)      # per solve: dict(targets, n_params, n_quotes, nfev, cost)

    def max_abs_residual(self):
        return max((abs(r.residual) for r in self.rows), default=0.0)


def _hybrid(curve):
    return curve.spread if isinstance(curve, SpreadCurve) else curve


def _install(key, spec, curves):
    """The HybridCurve of key in curves, created (or replaced if its nodes differ) from spec. An existing
    curve with the same nodes is kept, so its parameters warm start the solve."""
    native = DiscountCurveKey(key.currency, curves.setup.collateral_of(key.currency).name)
    if key in curves:
        c, h = curves[key], _hybrid(curves[key])
        if (isinstance(h, HybridCurve) and np.array_equal(h.flat_nodes, spec.flat_nodes)
                and np.array_equal(h.spline_knots, spec.spline_knots)
                and (key == native or c.base is curves[native])):
            return h
    h = spec.build()
    curves[key] = h if key == native else SpreadCurve(curves[native], h)
    return h


def _levels(order, groups, setup):
    level = {}
    for t in order:            # dependency order: the levels of the dependencies are known
        deps = set().union(*(q.instrument.dependencies(setup) for q in groups[t])) & set(order)
        level[t] = 1 + max((level[d] for d in deps if d != t), default=-1)
    return [[t for t in order if level[t] == k] for k in range(max(level.values()) + 1)]


def _solve(keys, groups, hybrids, pricer, smoothing, weights, report):
    quotes = [q for k in keys for q in groups[k]]
    sizes = [hybrids[k].n_params for k in keys]
    cuts = np.cumsum(sizes)[:-1]
    w = np.array([weights(q) if weights else 1.0 for q in quotes])
    units = np.array([unit(q.instrument) for q in quotes])
    market = np.array([q.value for q in quotes])
    n_par = int(sum(sizes))
    n_spline = sum(len(hybrids[k].spline_knots) for k in keys)
    if n_par > len(quotes) and smoothing <= 0:
        raise CalibrationError(f"{', '.join(map(str, keys))}: {n_par} parameters for {len(quotes)} quotes, "
                               "use a positive smoothing or fewer nodes")

    def set_x(x):
        for k, xk in zip(keys, np.split(x, cuts)):
            hybrids[k].set_params(xk)

    def residuals(x):
        set_x(x)
        r = w * (np.array([q.instrument.quote(pricer) for q in quotes]) - market) / units
        if smoothing > 0 and n_spline:
            r = np.r_[r, np.sqrt(smoothing) * 1e4 * np.concatenate([hybrids[k].roughness() for k in keys])]
        return r

    x0 = np.concatenate([hybrids[k].params for k in keys])
    m = len(residuals(x0))
    res = least_squares(residuals, x0, method="lm" if m >= n_par else "trf",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200 * (n_par + 1))
    if res.status <= 0:
        raise CalibrationError(f"{', '.join(map(str, keys))}: least squares failed ({res.message})")
    set_x(res.x)
    report.used += quotes
    report.rows += [FitRow(q, q.instrument.quote(pricer)) for q in quotes]
    report.levels.append(dict(targets=list(keys), n_params=n_par, n_quotes=len(quotes),
                              nfev=int(res.nfev), cost=float(res.cost)))


def fit_curves(quotes, curves, pricer, specs=None, targets=None, smoothing=0.0, nested=True, weights=None,
               report=None):
    """Fit the target curves (default: all) of the linear quotes into `curves`, in place.
    specs: {CurveKey: CurveSpec} (default_curve_spec for the others); weights: quote -> weight."""
    report = report if report is not None else FitReport()
    order, groups = _target_order(quotes.of_type(*LINEAR), quotes.setup)
    order = [t for t in order if targets is None or t in targets]
    specs = specs or {}
    hybrids = {k: _install(k, specs.get(k) or default_curve_spec(groups[k]), curves) for k in order}
    for keys in (_levels(order, groups, quotes.setup) if nested else [order]):
        _solve(keys, groups, hybrids, pricer, smoothing, weights, report)
    return report


def calibrate_curves_global(quotes, pricer_factory=DeterministicPricer, **kwargs):
    """Global counterpart of curve_calibration.calibrate_curves: (curves, report)."""
    curves = CurveSet(quotes.setup, quotes.x0)
    report = fit_curves(quotes, curves, pricer_factory(curves), **kwargs)
    return curves, report


@dataclass(frozen=True)
class GlobalFit:
    """Curve method of calibration.ModelSpec (`curves`): fit_curves with these options instead of the
    bootstrap."""
    specs: dict = None
    smoothing: float = 0.0
    nested: bool = True

    def fit(self, quotes, curves, pricer, targets=None):
        return fit_curves(quotes, curves, pricer, self.specs, targets, self.smoothing, self.nested)
