"""Hybrid curves, rate futures and the nested global curve fit (curve_fitting), against the bootstrap."""
import dataclasses
import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad

from convexity.curve_calibration import calibrate_curves
from convexity.curve_fitting import (CurveSpec, GlobalFit, calibrate_curves_global, default_curve_spec,
                                     fit_curves, unit)
from convexity.curves import CurveSet, HybridCurve, Index, LGCMSetup, PiecewiseFlatCurve, SpreadCurve
from convexity.curve_calibration import CalibrationError
from convexity.instruments import (CurrencyBasisSwaption, FXForward, FXOption, MarketQuotes, Quote, RateFuture,
                                   ResettingCCS, Swap, Swaption)
from convexity.market_data import MARKETS, example_quotes
from convexity.pricers import DeterministicPricer

SOFR = Index("USD_SOFR", "USD")
ESTR = Index("EUR_ESTR", "EUR")
E3M = Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.0010)
SETUP = LGCMSetup(SOFR, ESTR, foreign_index=E3M)
CURVE = HybridCurve([0.25, 0.5, 1.0], [2, 5, 10, 30], [0.04, 0.042, 0.039, 0.037, 0.036, 0.04, 0.035])
BREAKS = [0.25, 0.5, 1, 2, 5, 10, 30]


def _quad(f, a, b):
    """Adaptive quadrature; the roundoff warning (requested 1e-13 not certified) is silenced, the
    assertions carry their own tolerances."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        return quad(f, a, b, points=[p for p in BREAKS if a < p < b] or None, limit=200, epsabs=1e-15, epsrel=1e-13)[0]


# ---------------------------------------------------------------------- HybridCurve
def test_hybrid_integral_matches_quadrature():
    for T in (0.1, 0.25, 0.7, 1.0, 1.5, 3.0, 7.0, 10.0, 22.0, 30.0, 45.0):
        assert CURVE.integral(T) == pytest.approx(_quad(CURVE.instantaneous_forward, 0, T), abs=1e-14)
    assert CURVE.Z(0.0) == 1.0
    np.testing.assert_allclose(CURVE.Z(np.array([3.0, 7.0])), [CURVE.Z(3.0), CURVE.Z(7.0)], rtol=1e-15)


def test_hybrid_parameters_are_flat_forwards_then_section_averages():
    x = CURVE.params
    np.testing.assert_allclose(CURVE.instantaneous_forward([0.1, 0.4, 0.9]), x[:3], rtol=1e-15)
    t = np.r_[1.0, CURVE.spline_knots]
    avg = [(CURVE.integral(b) - CURVE.integral(a)) / (b - a) for a, b in zip(t[:-1], t[1:])]
    np.testing.assert_allclose(avg, x[3:], rtol=1e-13)


def test_hybrid_forward_continuity_and_smoothness():
    r, h = CURVE.instantaneous_forward, 1e-7
    assert r(1.0 + 1e-12) == pytest.approx(0.039, abs=1e-10)               # continuous at the junction
    for k in (2.0, 5.0, 10.0, 30.0):                                        # C^1 at the knots
        assert r(k + h) == pytest.approx(r(k - h), abs=1e-9)
        left, right = (r(k) - r(k - h)) / h, (r(k + h) - r(k)) / h
        assert right == pytest.approx(left, abs=1e-5)
    assert r(31.0) == r(45.0) == pytest.approx(r(30.0), abs=1e-15)          # flat extrapolation
    assert (r(30.0) - r(30.0 - h)) / h == pytest.approx(0.0, abs=1e-5)      # r'(k_n) = 0


def test_hybrid_without_knots_is_piecewise_flat():
    h = HybridCurve([0.5, 1.0, 3.0], params=[0.02, 0.03, 0.025])
    p = PiecewiseFlatCurve([0.5, 1.0, 3.0], [0.02, 0.03, 0.025])
    T = np.linspace(0.0, 10.0, 41)
    np.testing.assert_allclose(h.Z(T), p.Z(T), rtol=1e-15)
    assert h.roughness().size == 0
    with pytest.raises(ValueError):
        HybridCurve([], [1.0])
    with pytest.raises(ValueError):
        HybridCurve([1.0], [0.5])


def test_roughness_is_the_l2_norm_of_the_forward_slope():
    dr = lambda t: CURVE._spline(t, 2)
    exact = sum(_quad(lambda t: dr(t) ** 2, a, b) for a, b in zip([1, 2, 5, 10], [2, 5, 10, 30]))
    assert float(CURVE.roughness() @ CURVE.roughness()) == pytest.approx(exact, rel=1e-12)


def test_spline_minimises_the_forward_slope():
    """Proposition "Smooth forward section": for every C^2 perturbation phi of I vanishing at the knots
    with phi'(s_m) = 0 (same section averages, same junction condition), int I'' phi'' = 0, so
    int (I'' + phi'')^2 = int (I'')^2 + int (phi'')^2 >= int (I'')^2. Tested on sin^3 bumps of each
    section (C^2, zero value, slope and curvature at both ends) and on phi = (t - a)^2 (t - k_n) on the
    last section, whose slope at k_n is not zero: there the natural condition I''(k_n) = 0 kills the
    boundary term. Negative control: a spline clamped at k_n (I''(k_n) != 0) fails the orthogonality."""
    from scipy.interpolate import CubicSpline
    knots = np.r_[1.0, CURVE.spline_knots]

    def bump2(t, a, b):          # second derivative of sin^3(pi (t - a)/(b - a))
        u, w = np.pi * (t - a) / (b - a), np.pi / (b - a)
        return w * w * (6 * np.sin(u) * np.cos(u) ** 2 - 3 * np.sin(u) ** 3)

    def cross(d2, phi2, sections):
        return sum(_quad(lambda t: d2(t) * phi2(t), a, b) for a, b in sections)
    secs = list(zip(knots[:-1], knots[1:]))
    d2 = lambda t: CURVE._spline(t, 2)
    scale = np.sqrt(float(CURVE.roughness() @ CURVE.roughness()))
    for a, b in secs:
        phi2 = lambda t, a=a, b=b: bump2(t, a, b) if a <= t <= b else 0.0
        assert abs(cross(d2, phi2, secs)) < 1e-12 * scale * np.sqrt(_quad(lambda t: phi2(t) ** 2, a, b))
    # last section: phi = (t - a)^2 (t - b) has phi(a) = phi'(a) = phi(b) = 0 and phi'(b) != 0
    a, b = secs[-1]
    phi2 = lambda t: 2 * (t - b) + 4 * (t - a)
    assert abs(cross(d2, phi2, [secs[-1]])) < 1e-12 * scale * 20
    clamped = CubicSpline(CURVE._spline.x, CURVE._spline(CURVE._spline.x), bc_type=((1, 0.039), (1, 0.0)))
    assert abs(cross(lambda t: clamped(t, 2), phi2, [secs[-1]])) > 1e-6


# ---------------------------------------------------------------------- RateFuture
def test_rate_future_quote():
    cs = CurveSet(SETUP, 1.1, {SETUP.key("dd"): CURVE})
    pr = DeterministicPricer(cs)
    f = RateFuture(SOFR, 0.5, 0.75, convexity=1.5e-4)
    fwd = (CURVE.Z(0.5) / CURVE.Z(0.75) - 1) / 0.25
    assert f.quote(pr) == pytest.approx(100 * (1 - fwd - 1.5e-4), abs=1e-12)
    assert f.target(SETUP) == SETUP.key("dd") and f.pillar == 0.75 and f.label == "USD_SOFR future 6M-9M"
    assert unit(f) == 1e-2
    with pytest.raises(ValueError):
        RateFuture(SOFR, 0.5, 0.5)


# ---------------------------------------------------------------------- global fit
SPECS = {SETUP.key("dd"): CurveSpec(tuple(0.25 * np.arange(1, 9)), (3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)),
         SETUP.key("ff"): CurveSpec((1.0,), (2.0, 3.0, 5.0, 10.0, 30.0)),
         SETUP.key("fd"): CurveSpec((0.25, 0.5, 1.0), (2.0, 5.0, 10.0, 30.0))}


def reference_curves():
    rng = np.random.default_rng(11)
    cs = CurveSet(SETUP, 1.1)
    for key, base in ((SETUP.key("dd"), 0.035), (SETUP.key("ff"), 0.022)):
        s = SPECS[key]
        cs[key] = HybridCurve(s.flat_nodes, s.spline_knots, base + rng.normal(scale=0.004, size=len(s.flat_nodes) + len(s.spline_knots)))
    s = SPECS[SETUP.key("fd")]
    cs[SETUP.key("fd")] = SpreadCurve(cs[SETUP.key("ff")], HybridCurve(
        s.flat_nodes, s.spline_knots, -0.001 + rng.normal(scale=0.0005, size=7)))
    return cs


def reference_quotes(cs):
    pr = DeterministicPricer(cs)
    insts = ([RateFuture(SOFR, a, a + 0.25, 1e-5 * a) for a in 0.25 * np.arange(8)]
             + [Swap(SOFR, T) for T in (3, 5, 7, 10, 15, 20, 30)]
             + [Swap(E3M, T) for T in (1, 2, 3, 5, 10, 30)]
             + [FXForward(T, "points") for T in (0.25, 0.5, 1.0)] + [ResettingCCS(T) for T in (2, 5, 10, 30)])
    return MarketQuotes(SETUP, 1.1, [Quote(i, i.quote(pr)) for i in insts])


@pytest.mark.parametrize("nested", [True, False])
def test_global_fit_round_trip(nested):
    ref = reference_curves()
    mq = reference_quotes(ref)
    cs, rep = calibrate_curves_global(mq, specs=SPECS, nested=nested)
    for key in SPECS:
        a, b = cs[key], ref[key]
        a, b = (a.spread, b.spread) if isinstance(a, SpreadCurve) else (a, b)
        np.testing.assert_allclose(a.params, b.params, atol=1e-11)
    T = np.linspace(0.05, 40, 60)
    for kind in ("dd", "ff", "fd"):
        np.testing.assert_allclose(cs.Z0(kind, T), ref.Z0(kind, T), rtol=1e-12)
    assert rep.max_abs_residual() < 1e-9 and len(rep.used) == len(mq.quotes) and not rep.checks
    assert [len(l["targets"]) for l in rep.levels] == ([2, 1] if nested else [3])


def test_default_spec_uses_short_end_instruments():
    mq = reference_quotes(reference_curves())
    s = default_curve_spec(mq.for_target(SETUP.key("dd")))
    assert s == SPECS[SETUP.key("dd")]
    assert default_curve_spec(mq.for_target(SETUP.key("ff"))) == SPECS[SETUP.key("ff")]   # first pillar flat
    assert default_curve_spec(mq.for_target(SETUP.key("fd"))) == SPECS[SETUP.key("fd")]


def curve_quotes(market="EURUSD", futures=False):
    return example_quotes(market, futures).without(Swaption, FXOption, CurrencyBasisSwaption)


def test_flat_spec_reproduces_the_bootstrap():
    """With flat nodes at every pillar the global fit is the bootstrap: two independent algorithms."""
    mq = curve_quotes()
    boot, _ = calibrate_curves(mq)
    specs = {k: CurveSpec(tuple(boot[k].pillars)) for k in boot.keys()}
    glob, rep = calibrate_curves_global(mq, specs=specs)
    T = np.linspace(0.05, 40, 80)
    for kind in ("dd", "ff", "fd"):
        np.testing.assert_allclose(glob.Z0(kind, T), boot.Z0(kind, T), rtol=1e-12)


@pytest.mark.parametrize("market", MARKETS)
def test_example_markets_exact_fit_and_smooth_forwards(market):
    mq = curve_quotes(market)
    cs, rep = calibrate_curves_global(mq)
    assert rep.max_abs_residual() < 1e-8                                    # bp / pips
    for kind in ("dd", "ff", "fd"):
        c = cs[mq.setup.key(kind)]
        h = c.spread if isinstance(c, SpreadCurve) else c
        for k in h.spline_knots[:-1]:
            assert h.instantaneous_forward(k + 1e-7) == pytest.approx(h.instantaneous_forward(k - 1e-7), abs=1e-8)
    boot, _ = calibrate_curves(mq)                                          # same model-free short end
    for q in mq.of_type(FXForward):
        T = q.instrument.maturity
        assert cs.Z0("fd", T) == pytest.approx(boot.Z0("fd", T), rel=1e-12)


def test_futures_in_bootstrap_and_least_squares():
    """Futures inside the 1Y and 2Y swaps: the bootstrap fits the futures and leaves the swaps as checks
    (0.050 and 0.026 bp off); the least squares spreads the inconsistency over all USD quotes."""
    mq = curve_quotes(futures=True)
    boot, brep = calibrate_curves(mq)
    pr = DeterministicPricer(boot)
    assert [q.instrument.label for q in brep.checks] == ["USD_SOFR swap 1Y", "USD_SOFR swap 2Y"]
    off = [(q.instrument.quote(pr) - q.value) / 1e-4 for q in brep.checks]
    np.testing.assert_allclose(off, [0.0501, 0.0263], atol=5e-4)
    for q in mq.of_type(RateFuture):
        assert q.instrument.quote(pr) == pytest.approx(q.value, abs=1e-12)
    glob, rep = calibrate_curves_global(mq)
    usd = [r for r in rep.rows if r.quote.instrument.target(mq.setup) == mq.setup.key("dd")]
    assert len(usd) == 17 and rep.levels[0]["n_params"] == rep.levels[0]["n_quotes"] - 2
    assert 0.01 < max(abs(r.residual) for r in usd) < 0.05
    # first order conditions of the least squares: J^T r = 0 (J by central differences)
    h = glob[mq.setup.key("dd")]
    gp, x0 = DeterministicPricer(glob), h.params.copy()
    res = lambda: np.array([(q.instrument.quote(gp) - q.value) / unit(q.instrument) for q in (r.quote for r in usd)])
    r0, J = res(), []
    for i in range(len(x0)):
        e = np.zeros_like(x0); e[i] = 1e-7
        h.set_params(x0 + e); up = res()
        h.set_params(x0 - e); dn = res()
        J.append((up - dn) / 2e-7)
    h.set_params(x0)
    assert np.abs(np.array(J) @ r0).max() < 1e-6 * np.abs(np.array(J)).max()


def test_underdetermined_needs_smoothing_and_smoothing_trades_fit_for_roughness():
    mq = curve_quotes()
    key = mq.setup.key("ff")
    specs = {key: CurveSpec((1.0,), (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0))}
    with pytest.raises(CalibrationError, match="smoothing"):
        calibrate_curves_global(mq, specs=specs, targets=[key])
    out = []
    for s in (1e-6, 1e-2, 1.0):
        cs, rep = calibrate_curves_global(mq, specs=specs, targets=[key], smoothing=s)
        rough = cs[key].roughness()
        out.append((float(rough @ rough), rep.max_abs_residual()))
    assert out[0][0] > out[1][0] > out[2][0] and out[0][1] < out[1][1] < out[2][1]


def test_lgcm_calibration_with_global_curves():
    mq = example_quotes()
    from convexity.calibration import calibrate, default_spec
    cal = calibrate(mq, dataclasses.replace(default_spec(), curves=GlobalFit()))
    pr = cal.pricer()
    for q in mq.quotes:
        tol = 1e-8 if isinstance(q.instrument, FXForward) else 1e-12
        assert q.instrument.quote(pr) == pytest.approx(q.value, abs=tol), q.instrument.label
    assert cal.diagnostics["fixed_point_iterations"] <= 10
    assert isinstance(cal.curves[mq.setup.key("fd")].spread, HybridCurve)


def test_spline_through_forward_knot_values_zig_zags():
    """Negative control for the choice of Definition "Hybrid forward curve": a natural cubic spline of the
    forward through FREE knot values (flat forward to 1Y, flat after 30Y), fitted exactly to the same USD
    swaps, oscillates (signs of successive knot differences alternate), while the hybrid curve does not."""
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares
    mq = curve_quotes()
    swaps = mq.for_target(mq.setup.key("dd"))
    knots = np.array([1.0, 2, 3, 5, 7, 10, 15, 20, 30])
    grid = np.linspace(0, 30, 30 * 365 + 1)

    def Z_of(x):
        g = CubicSpline(knots, x, bc_type="natural")
        fwd = np.where(grid <= 1.0, x[0], g(np.clip(grid, 1.0, 30.0)))
        I = np.r_[0.0, np.cumsum(0.5 * (fwd[1:] + fwd[:-1]) * np.diff(grid))]
        return lambda T: np.exp(-np.interp(T, grid, I))
    res = least_squares(lambda x: [q.instrument.legs(Z_of(x))[0] / q.instrument.legs(Z_of(x))[1] - q.value
                                   for q in swaps], np.full(9, 0.04), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    assert np.abs(res.fun).max() < 1e-8                      # 1e-4 bp: an exact fit
    d = np.sign(np.diff(res.x))
    assert (d[1:] != d[:-1]).sum() >= 6                     # sign changes of the knot differences
    cs, _ = calibrate_curves_global(mq)
    r = cs[mq.setup.key("dd")].instantaneous_forward(knots)
    d = np.sign(np.diff(r))
    assert (d[1:] != d[:-1]).sum() <= 2
    print("forward knot values (%)", np.round(res.x * 100, 2), "hybrid", np.round(r * 100, 2))


def test_mc_report_rejects_futures():
    from convexity.calibration_check import mc_report
    from types import SimpleNamespace
    q = Quote(RateFuture(SOFR, 0.0, 0.25), 95.0)
    with pytest.raises(ValueError, match="marked"):
        mc_report(SimpleNamespace(quotes=None), quotes=[q])
