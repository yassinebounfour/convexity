"""Calibration of the Linear Gaussian Collateral Model (section "The Linear Gaussian Collateral Model:
Calibration" of the notes).

Model: blocks d, f, c with n_d, n_f, n_c factors and the FX driver. Mean reversions and correlations
are marked. In each block sigma^{a,i}_t = lambda^a_t w^{a,i} with fixed weights; lambda^a and sigma^x
are piecewise constant on the block's calibration expiries (flat extrapolation). The model knots are
the union of the expiries.

Algorithm "LGCM calibration":
1. native curves Z^{d|d}, Z^{f|f} from swaps (model-free bootstrap, or the global fit of curve_fitting
   if ModelSpec.curves is a GlobalFit; the same choice is used for Z^{f|d} in step 3);
2. lambda^d, lambda^f from ATM swaption normal vols, expiry by expiry: Gaussian approximation guess
   (the segment variance is v0 + lambda^2 v1), then the exact one-factor price;
3. fixed point: Z^{f|d} from FX forwards and resetting CCS with the LGCM collateral convexities of
   the foreign coupon (Libor or compounded), then
   lambda^c from CB spread vols (Gaussian guess, exact 2D quadrature), until lambda^c and Z^{f|d}
   no longer move;
4. sigma^x from ATM FX vols: the total variance is quadratic in sigma^x on the segment (exact).
Finally the off-market curve Z^{d|f} is added as a model output.
"""
from dataclasses import dataclass, field
from math import sqrt
import time

import numpy as np

from .curve_calibration import BootstrapReport, CalibrationError, bootstrap, root_in_bracket
from .curves import CurveSet, FunctionCurve, HybridCurve
from .instruments import CurrencyBasisSwaption, FXOption, Swaption
from .lgcm import Model
from .market import fx_total_variance, lgcm_Zdf0
from .pricers import DeterministicPricer, LGCMPricer, native_swaption_variance

__all__ = ["BlockSpec", "ModelSpec", "default_spec", "Calibration", "calibrate", "CalibrationError"]


@dataclass(frozen=True)
class BlockSpec:
    """Factors of one block: constant mean reversions, fixed vol weights, initial / marked level lambda."""
    betas: tuple = (0.03,)
    weights: tuple = None
    level: float = 0.01

    def __post_init__(self):
        if self.weights is None:
            object.__setattr__(self, "weights", (1.0,) * len(self.betas))
        if len(self.weights) != len(self.betas):
            raise ValueError("one weight per factor")

    @property
    def n(self):
        return len(self.betas)


@dataclass(frozen=True)
class ModelSpec:
    d: BlockSpec
    f: BlockSpec
    c: BlockSpec
    corr: np.ndarray            # order d..., f..., c..., x
    fx_level: float = 0.10
    horizon: float = 60.0
    exact: bool = True
    tol: float = 1e-12
    max_iter: int = 30
    curves: object = None       # None: sequential bootstrap; curve_fitting.GlobalFit: nested global fit

    def __post_init__(self):
        n = self.d.n + self.f.n + self.c.n + 1
        R = np.asarray(self.corr, float)
        if R.shape != (n, n) or not np.allclose(R, R.T) or not np.allclose(np.diag(R), 1):
            raise ValueError(f"corr must be a {n}x{n} correlation matrix")
        if np.linalg.eigvalsh(R).min() <= 0:
            raise ValueError("corr must be positive definite")


def default_spec():
    """One factor per block. Correlations (d, f, c, x) as in the test model of the notes."""
    corr = np.array([[1.0, 0.6, -0.2, 0.25],
                     [0.6, 1.0, -0.4, 0.3],
                     [-0.2, -0.4, 1.0, 0.35],
                     [0.25, 0.3, 0.35, 1.0]])
    return ModelSpec(BlockSpec((0.03,), level=0.009), BlockSpec((0.02,), level=0.008),
                     BlockSpec((0.10,), level=0.003), corr)


@dataclass
class Calibration:
    model: Model
    curves: CurveSet
    quotes: object
    spec: ModelSpec
    report: BootstrapReport
    diagnostics: dict = field(default_factory=dict)

    def pricer(self, exact=None):
        return LGCMPricer(self.model, self.curves, self.spec.exact if exact is None else exact)

    def levels(self, block):
        """Knots and lambda^block (sigma^x for "x") on each piece."""
        m = self.model
        if block == "x":
            return m.knots, m.sig[m.ix].copy()
        i = m.idx[block][0]
        return m.knots, m.sig[i] / getattr(self.spec, block).weights[0]


# ---------------------------------------------------------------------- helpers
def _set_level(model, spec, block, t0, t1, lam):
    """lambda^block = lam on the pieces of [t0, t1) (t1 = inf: flat extrapolation after the last expiry).
    Pieces after t1 are left alone: a CB vol at T0 also reads sigma^c on [T0, T0 + tenor) through the
    coupon convexities, so overwriting the tail would move the targets already calibrated."""
    k0 = int(np.searchsorted(model.knots, t0 - 1e-12))
    k1 = int(np.searchsorted(model.knots, t1 - 1e-12)) if np.isfinite(t1) else len(model.knots) - 1
    if block == "x":
        model.sig[model.ix][k0:k1] = lam
        return
    for i, w in zip(model.idx[block], getattr(spec, block).weights):
        model.sig[i][k0:k1] = lam * w


def _segments(quotes):
    """(quote, t0, t1): quote k sets the level on [T_{k-1}, T_k), the last one on [T_{n-1}, inf)."""
    T = [q.instrument.expiry for q in quotes]
    return zip(quotes, [0.0] + T[:-1], T[:-1] + [np.inf])


def _by_expiry(quotes):
    qs = sorted(quotes, key=lambda q: q.instrument.pillar)
    for a, b in zip(qs[:-1], qs[1:]):
        if abs(a.instrument.pillar - b.instrument.pillar) < 1e-10:
            raise ValueError(f"two quotes with the same expiry: {a.instrument.label}, {b.instrument.label}")
    return qs


def _build_model(spec, knots):
    n = len(knots) - 1
    factors = [(blk, np.full(n, b.level * w), np.full(n, beta))
               for blk, b in (("d", spec.d), ("f", spec.f), ("c", spec.c))
               for beta, w in zip(b.betas, b.weights)]
    return Model(knots, factors, spec.corr, np.full(n, spec.fx_level))


def _squeeze(q, detail):
    return CalibrationError(f"variance squeeze on {q.instrument.label} (market {q.value:.6g}): {detail}")


# ---------------------------------------------------------------------- steps
def _bootstrap_swaptions(model, spec, curves, quotes, block, diag):
    nf = getattr(spec, block).n
    exact = spec.exact and nf == 1
    if not exact and quotes:
        diag["approximations"].append(f"{block} swaptions: Gaussian approximation ({nf} factors)")
    key = curves.setup.key(block * 2)
    Z = lambda T: curves.Z(key, T)
    pricer = LGCMPricer(model, curves, exact=True)
    for q, t0, t1 in _segments(quotes):
        u, T0 = q.instrument.underlying, q.instrument.expiry

        def var(lam):
            _set_level(model, spec, block, t0, t1, lam)
            return native_swaption_variance(model, model.idx[block], u, Z)
        v0, v1 = var(0.0), var(1.0) - var(0.0)
        target = q.value ** 2 * T0
        if v0 >= target:
            raise _squeeze(q, f"variance {v0:.3g} already delivered before {t0}")
        lam = sqrt((target - v0) / v1)
        if exact:
            def residual(x):
                _set_level(model, spec, block, t0, t1, x)
                return pricer.swaption_normal_vol(q.instrument) - q.value
            lam = root_in_bracket(residual, lam, lower=0.0, step=0.1 * lam)
        _set_level(model, spec, block, t0, t1, lam)


def _bootstrap_cb_vols(model, spec, curves, quotes):
    approx, exact = LGCMPricer(model, curves, exact=False), LGCMPricer(model, curves, exact=True)
    for q, t0, t1 in _segments(quotes):
        def residual(pricer):
            def f(x):
                _set_level(model, spec, "c", t0, t1, x)
                return pricer.cb_normal_vol(q.instrument) - q.value
            return f
        # warm start: the level of the previous fixed-point iteration (read before the probe at 0)
        k0, (i0, w0) = int(np.searchsorted(model.knots, t0 - 1e-12)), (model.idx["c"][0], spec.c.weights[0])
        guess = max(model.sig[i0][k0] / w0, 1e-4)
        excess = residual(approx)(0.0)
        if excess >= 0:
            where = ("the foreign rates block alone delivers" if t0 == 0 else
                     f"lambda^c before {t0:g} and the foreign rates block deliver")
            raise _squeeze(q, f"with lambda^c = 0 on [{t0:g}, {q.instrument.expiry:g}), {where} a CB vol "
                              f"of {q.value + excess:.4g} (Gaussian approximation)")
        _set_level(model, spec, "c", t0, t1, guess)
        lam = root_in_bracket(residual(approx), guess, lower=0.0, step=0.5 * guess)
        if spec.exact:
            lam = root_in_bracket(residual(exact), lam, lower=0.0, step=0.05 * lam)
        _set_level(model, spec, "c", t0, t1, lam)


def _bootstrap_fx(model, quotes):
    for q, t0, t1 in _segments(quotes):
        T = q.instrument.expiry

        def w(s):
            _set_level(model, None, "x", t0, t1, s)
            return fx_total_variance(model, T)
        w0, wp, wm = w(0.0), w(1.0), w(-1.0)
        a, b, c = 0.5 * (wp + wm) - w0, 0.5 * (wp - wm), w0 - q.value ** 2 * T
        disc = b * b - 4 * a * c
        root = (-b + sqrt(disc)) / (2 * a) if disc >= 0 else -1.0
        if root < 0:
            raise _squeeze(q, f"no sigma^x >= 0 on [{t0}, {T}] (rates and basis deliver {w0 / T:.4g} of variance per year)")
        _set_level(model, None, "x", t0, t1, root)


def _marked(spec, swaptions, cb_vols, fx_vols):
    """Blocks left at their marked level because no quote targets them."""
    out = [f"lambda^{b} marked at level={getattr(spec, b).level}: no {b} swaption quotes"
           for b in "df" if not swaptions[b]]
    if not cb_vols:
        out.append(f"lambda^c marked at level={spec.c.level}: no CB vol quotes")
    if not fx_vols:
        out.append(f"sigma^x marked at fx_level={spec.fx_level}: no FX option quotes")
    return out


def _snapshot(model, curves):
    fd = curves[curves.setup.key("fd")]
    state = fd.spread.params if isinstance(fd.spread, HybridCurve) else fd.forwards
    return np.r_[model.sig[model.idx["c"][0]], state]


def _fit_curves(spec, quotes, curves, pricer, targets):
    if spec.curves is None:
        return bootstrap(quotes, curves, pricer, targets=targets)
    return spec.curves.fit(quotes, curves, pricer, targets=targets)


# ---------------------------------------------------------------------- driver
def calibrate(quotes, spec=None):
    spec = spec or default_spec()
    setup = quotes.setup
    t_start = time.perf_counter()
    swaptions = {b: _by_expiry([q for q in quotes.of_type(Swaption) if q.instrument.target(setup) == b]) for b in "df"}
    cb_vols = _by_expiry(quotes.of_type(CurrencyBasisSwaption))
    fx_vols = _by_expiry(quotes.of_type(FXOption))
    expiries = [q.instrument.pillar for q in swaptions["d"] + swaptions["f"] + cb_vols + fx_vols]
    if expiries and max(expiries) >= spec.horizon:
        raise ValueError(f"expiries must be before the horizon {spec.horizon}")
    if not quotes.for_target(setup.key("fd")):
        raise CalibrationError("Z^{f|d} needs FX forward or resetting CCS quotes (none given): the "
                               "calibration bootstraps it in the fixed point with lambda^c")
    model = _build_model(spec, np.unique(np.r_[0.0, expiries, spec.horizon]))
    diag = {"approximations": [], "marked": _marked(spec, swaptions, cb_vols, fx_vols)}

    # 1. native curves
    curves = CurveSet(setup, quotes.x0)
    report = _fit_curves(spec, quotes, curves, DeterministicPricer(curves), [setup.key("dd"), setup.key("ff")])
    # 2. rates vols
    for b in "df":
        _bootstrap_swaptions(model, spec, curves, swaptions[b], b, diag)
    # 3. Z^{f|d} and lambda^c
    fd_report, history, prev = None, [], None
    for it in range(1, spec.max_iter + 1):
        fd_report = _fit_curves(spec, quotes, curves, LGCMPricer(model, curves, spec.exact), [setup.key("fd")])
        if cb_vols:
            _bootstrap_cb_vols(model, spec, curves, cb_vols)
        snap = _snapshot(model, curves)
        change = np.inf if prev is None else float(np.abs(snap - prev).max())
        history.append(change)
        prev = snap
        if not cb_vols or change < spec.tol:
            break
    else:
        raise CalibrationError(f"Z^{{f|d}} / sigma^c fixed point did not converge: changes {history}")
    report.used += fd_report.used
    report.checks += fd_report.checks
    # 4. FX vol
    _bootstrap_fx(model, fx_vols)
    # model output: Z^{d|f}
    curves[setup.key("df")] = FunctionCurve(lambda T: lgcm_Zdf0(model, curves, T) if T > 0 else 1.0)
    diag.update(fixed_point_iterations=it, fixed_point_changes=history,
                seconds=time.perf_counter() - t_start)
    return Calibration(model, curves, quotes, spec, report, diag)
