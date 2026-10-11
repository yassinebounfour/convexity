"""Generic sequential bootstrap of discount curves from linear instruments.

Each linear instrument (Swap, FXForward, ResettingCCS) declares the CurveKey it calibrates, its pillar
and the curves it depends on. Target curves are built in dependency order. On each curve, the quotes
are sorted by (priority, pillar); each quote adds a node at its pillar, solved by a 1D root search of
instrument.quote(pricer) - market value. A quote whose pillar is not beyond the last node of its
curve is not used, and is reported as a check. Native discount curves have flat forwards; a curve
off native collateral is Z^{e|e} times a flat-forward collateral basis curve (Remark "Which curve to
interpolate" in the notes).

The model only enters through the pricer (Libor collateral convexities of the resetting CCS):
DeterministicPricer is the DCB market practice, LGCMPricer the LGCM.
"""
from dataclasses import dataclass, field

from scipy.optimize import brentq

from .curves import CurveSet, DiscountCurveKey, PiecewiseFlatCurve, SpreadCurve
from .instruments import FXForward, ResettingCCS, Swap
from .pricers import DeterministicPricer

LINEAR = (Swap, FXForward, ResettingCCS)


class CalibrationError(Exception):
    """A calibration target cannot be matched (e.g. variance squeeze, no root)."""


@dataclass
class BootstrapReport:
    used: list = field(default_factory=list)     # quotes that set a node
    checks: list = field(default_factory=list)   # quotes inside an already built section


def root_in_bracket(f, guess, lower=None, step=None, max_expand=60):
    """Root of f near guess, bracketed by expanding [guess - step, guess + step] (lower bound optional)."""
    step = step if step is not None else max(abs(guess), 1e-3)

    def bounds(s):
        lo = guess - s if lower is None else max(guess - s, lower)
        return lo, guess + s

    lo, hi = bounds(step)
    flo, fhi = f(lo), f(hi)
    for _ in range(max_expand):
        if flo * fhi <= 0:
            return brentq(f, lo, hi, xtol=1e-15, maxiter=500)
        step *= 2
        lo, hi = bounds(step)
        flo, fhi = f(lo), f(hi)
    raise CalibrationError(f"no root found around {guess} (f({lo:.3g}) = {flo:.3g}, f({hi:.3g}) = {fhi:.3g})")


def _new_curve(key, curves):
    native = DiscountCurveKey(key.currency, curves.setup.collateral_of(key.currency).name)
    return PiecewiseFlatCurve() if key == native else SpreadCurve(curves[native])


def _target_order(quotes, setup):
    groups = {}
    for q in quotes:
        groups.setdefault(q.instrument.target(setup), []).append(q)
    deps = {t: set().union(*(q.instrument.dependencies(setup) for q in qs)) - {t} for t, qs in groups.items()}
    for t, ds in deps.items():
        for d in ds - set(groups):
            raise ValueError(f"no quotes for {d}, needed by the quotes of {t}")
    order, done = [], set()
    while len(order) < len(groups):
        ready = [t for t in groups if t not in done and deps[t] <= done]
        if not ready:
            raise ValueError(f"cyclic curve dependencies: {deps}")
        for t in sorted(ready, key=str):
            order.append(t)
            done.add(t)
    return order, groups


def _bootstrap_curve(key, quotes, curves, pricer, report):
    curve = _new_curve(key, curves)
    curves[key] = curve
    for q in sorted(quotes, key=lambda q: (q.instrument.priority, q.instrument.pillar)):
        T = q.instrument.pillar
        if curve.pillars and T <= curve.pillars[-1] + 1e-10:
            report.checks.append(q)
            continue
        guess = curve.forwards[-1] if curve.forwards else 0.0
        curve.add_node(T, guess)

        def residual(x):
            curve.set_last(x)
            return q.instrument.quote(pricer) - q.value
        curve.set_last(root_in_bracket(residual, guess, step=0.01))
        report.used.append(q)


def bootstrap(quotes, curves, pricer, targets=None, report=None):
    """Bootstrap the target curves (default: all) of the linear quotes into `curves`, in place."""
    report = report if report is not None else BootstrapReport()
    order, groups = _target_order(quotes.of_type(*LINEAR), quotes.setup)
    for key in order:
        if targets is None or key in targets:
            _bootstrap_curve(key, groups[key], curves, pricer, report)
    return report


def calibrate_curves(quotes, pricer_factory=DeterministicPricer):
    """Discount curves of all targets of the linear quotes. Default: DCB model (no convexity, no vols).
    pricer_factory(curves) -> Pricer, e.g. lambda c: LGCMPricer(model, c)."""
    curves = CurveSet(quotes.setup, quotes.fx_spot)
    report = bootstrap(quotes, curves, pricer_factory(curves))
    return curves, report
