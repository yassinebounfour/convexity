"""LGCM pricer (Task 4), calibration (Task 5), repricing checks (Task 6)."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from collateraldiscounting import gaussian_approx as ga
from collateraldiscounting.lgcm import Model
from collateraldiscounting.curves import Index, LGCMSetup, CurveSet, PiecewiseFlatCurve, SpreadCurve
from collateraldiscounting.instruments import (Swap, ResettingCCS, Swaption, FXOption, CurrencyBasisSwaption)
from collateraldiscounting.pricers import (LGCMPricer, DeterministicPricer, native_swaption_variance,
                                           native_swaption_exact_vol)
from collateraldiscounting.market import libor_convexity, fx_total_variance

SOFR = Index("USD_SOFR", "USD")
ESTR = Index("EUR_ESTR", "EUR")
E3M = Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.0010)
SETUP = LGCMSetup(SOFR, ESTR, foreign_index=E3M)


def small_model(scale=1.0):
    knots = [0, 1, 3, 7, 15, 60]
    n = len(knots) - 1
    ramp = np.linspace(1.0, 0.8, n)
    R = np.array([[1, .6, -.2, .25], [.6, 1, -.4, .3], [-.2, -.4, 1, .35], [.25, .3, .35, 1]])
    return Model(knots, [("d", scale * 0.009 * ramp, np.full(n, 0.03)), ("f", scale * 0.008 * ramp, np.full(n, 0.02)),
                         ("c", scale * 0.003 * ramp, np.full(n, 0.10))], R, scale * 0.08 * ramp)


def curves():
    cs = CurveSet(SETUP, 1.1)
    cs[SETUP.key("dd")] = PiecewiseFlatCurve([2, 10, 50], [0.042, 0.038, 0.04])
    cs[SETUP.key("ff")] = PiecewiseFlatCurve([2, 10, 50], [0.02, 0.026, 0.027])
    cs[SETUP.key("fd")] = SpreadCurve(cs[SETUP.key("ff")], PiecewiseFlatCurve([5, 50], [-0.001, -0.0015]))
    return cs


def test_native_swaption_exact_closed_form_matches_quadrature():
    m, cs = small_model(), curves()
    swap = Swap(E3M, 15.0, 1.0, 5.0)
    Z = lambda T: cs.Z0("ff", T)
    (i,) = m.idx["f"]
    T0 = swap.start
    tf, cf, tx, wx = swap.cashflows()
    K = swap.legs(Z)[0] / swap.legs(Z)[1]
    y = m.covint((i, "M", T0), (i, "M", T0), 0.0, T0)
    t = np.r_[tf, tx]
    c = np.r_[cf, -K * wx]
    P = Z(t) / Z(T0)
    G = np.array([m.G(i, T0, T) for T in t])
    V = lambda z: (c * P * np.exp(-G * z - 0.5 * G * G * y)).sum()
    s = np.sqrt(y)
    price = Z(T0) * quad(lambda z: max(V(z), 0.0) * norm.pdf(z / s) / s, -12 * s, 12 * s, limit=400,
                          epsabs=1e-14, epsrel=1e-12)[0]
    A0 = (wx * Z(tx)).sum()
    vol_quad = price / (A0 * np.sqrt(T0 / (2 * np.pi)))
    assert native_swaption_exact_vol(m, i, swap, Z) == pytest.approx(vol_quad, rel=1e-9)


def test_gaussian_swaption_variance_converges_to_exact():
    """Relative error of the Gaussian vol is O(scale^2) (asymptotic in the volatility)."""
    cs = curves()
    swap = Swap(E3M, 15.0, 1.0, 5.0)
    Z = lambda T: cs.Z0("ff", T)
    errs = []
    for scale in (0.5, 1.0, 2.0):
        m = small_model(scale)
        (i,) = m.idx["f"]
        exact = native_swaption_exact_vol(m, i, swap, Z)
        approx = np.sqrt(native_swaption_variance(m, [i], swap, Z) / 5.0)
        errs.append(abs(approx / exact - 1))
    assert errs[1] / errs[0] == pytest.approx(4, rel=0.15) and errs[2] / errs[1] == pytest.approx(4, rel=0.15)


def test_gaussian_variance_matches_gaussian_approx_native():
    """Same schedule on both legs: identical to gaussian_approx.approx_native (independent code)."""
    m, cs = small_model(), curves()
    swap = Swap(E3M, 15.0, 0.25, 5.0)          # kappa_f on both sides (SwapSetup reads curves.kappa_f)
    s = ga.SwapSetup(T0=5.0, theta=0.25, n=40, curves=cs)
    ref = ga.approx_native(m, s, ccy="f")
    v = native_swaption_variance(m, m.idx["f"], swap, lambda T: cs.Z0("ff", T))
    assert v == pytest.approx(ref["V"], rel=1e-12)


def test_lgcm_pricer_ccs_and_cb_vol_match_gaussian_approx():
    m, cs = small_model(), curves()
    pr = LGCMPricer(m, cs, exact=False)
    cbs = CurrencyBasisSwaption.make(5.0, 2.0)
    a = ga.approx(m, ga.SwapSetup(T0=5.0, theta=0.25, n=8, curves=cs))
    assert cbs.underlying.quote(pr) == pytest.approx(a["B0"], rel=1e-13)
    assert cbs.quote(pr) == pytest.approx(np.sqrt(a["VB"] / 5.0), rel=1e-13)
    ex = ga.exact(m, ga.SwapSetup(T0=5.0, theta=0.25, n=8, curves=cs))
    assert cbs.quote(LGCMPricer(m, cs)) == pytest.approx(ex["B"]["atm_normal_vol"], rel=1e-13)


def test_lgcm_pricer_convexity_and_fx_vol():
    m, cs = small_model(), curves()
    pr = LGCMPricer(m, cs)
    assert pr.libor_convexity(10.0, 10.25) == libor_convexity(m, 0.0, 10.0, 10.25)
    assert pr.libor_convexity(10.0, 10.25) > 1.0
    old = pr.libor_convexity(10.0, 10.25)
    m.sig[m.idx["c"][0]][:] *= 2                       # the cache must see model changes
    new = pr.libor_convexity(10.0, 10.25)
    assert new != old and new == libor_convexity(m, 0.0, 10.0, 10.25)
    assert FXOption(5.0).quote(pr) == pytest.approx(np.sqrt(fx_total_variance(m, 5.0) / 5.0), rel=1e-14)
    # LGCM CCS spread differs from the DCB one by the convexity only
    ccs = ResettingCCS(20.0)
    assert ccs.quote(pr) < ccs.quote(DeterministicPricer(cs))


def test_lgcm_pricer_swaption_dispatch_both_blocks():
    """Swaption.quote -> LGCMPricer picks the right block, native curve and T0 normalisation."""
    m, cs = small_model(), curves()
    for index, role, kind in ((E3M, "f", "ff"), (SOFR, "d", "dd")):
        sw = Swaption.make(index, 5.0, 10.0)
        u = sw.underlying
        Z = lambda T, kind=kind: cs.Z0(kind, T)
        (i,) = m.idx[role]
        assert sw.quote(LGCMPricer(m, cs, exact=True)) == pytest.approx(native_swaption_exact_vol(m, i, u, Z), rel=1e-13)
        assert sw.quote(LGCMPricer(m, cs, exact=False)) == pytest.approx(
            np.sqrt(native_swaption_variance(m, [i], u, Z) / u.start), rel=1e-13)
    # the d and f vols really differ (a wrong block or curve cannot pass by accident)
    pr = LGCMPricer(m, cs)
    assert abs(Swaption.make(E3M, 5.0, 10.0).quote(pr) / Swaption.make(SOFR, 5.0, 10.0).quote(pr) - 1) > 0.01


def two_factor_f_model():
    knots = [0, 1, 3, 7, 15, 60]
    n = len(knots) - 1
    ramp = np.linspace(1.0, 0.8, n)
    # drivers: d, f1, f2, c, FX
    R = np.array([[1, .3, .2, -.2, .25], [.3, 1, .5, -.4, .3], [.2, .5, 1, -.1, .1],
                  [-.2, -.4, -.1, 1, .35], [.25, .3, .1, .35, 1]])
    return Model(knots, [("d", 0.009 * ramp, np.full(n, 0.03)), ("f", 0.008 * ramp, np.full(n, 0.02)),
                         ("f", 0.005 * ramp, np.full(n, 0.15)), ("c", 0.003 * ramp, np.full(n, 0.10))],
                 R, 0.08 * ramp)


def test_multifactor_gaussian_variance_vs_brute_force_and_pricer_fallback():
    m, cs = two_factor_f_model(), curves()
    swap = Swap(E3M, 15.0, 1.0, 5.0)
    Z = lambda T: cs.Z0("ff", T)
    T0 = swap.start
    tf, cf, tx, wx = swap.cashflows()
    fl, A = swap.legs(Z)
    K = fl / A
    t, c = np.r_[tf, tx], np.r_[cf, -K * wx]
    P = Z(t) / Z(T0)
    fac = m.idx["f"]
    assert len(fac) == 2
    # explicit loadings: dS/dz_i = -sum_m c_m P_m G_i(T0, t_m) / A with G by quadrature of M
    Gq = lambda i, T: quad(lambda v: m.M(i, T0, v), T0, T, epsabs=1e-14, epsrel=1e-13,
                           points=[k for k in m.knots if T0 < k < T] or None)[0]
    q = np.array([-(c * P * np.array([Gq(i, T) for T in t])).sum() / (A / Z(T0)) for i in fac])
    y = np.array([[m.covint_brute((i, "M", T0), (j, "M", T0), 0.0, T0) for j in fac] for i in fac])
    assert abs(y[0, 1]) > 0.1 * np.sqrt(y[0, 0] * y[1, 1])            # the cross term matters
    V = float(q @ y @ q)
    assert native_swaption_variance(m, fac, swap, Z) == pytest.approx(V, rel=1e-8)
    # the cross term is not negligible: dropping it changes the answer
    assert abs((q ** 2 * np.diag(y)).sum() / V - 1) > 1e-3
    sw = Swaption.make(E3M, 5.0, 10.0)
    for exact in (True, False):
        assert sw.quote(LGCMPricer(m, cs, exact=exact)) == pytest.approx(np.sqrt(V / T0), rel=1e-8)


# ---------------------------------------------------------------------- Task 5: model calibration
import dataclasses
from collateraldiscounting.calibration import calibrate, default_spec, BlockSpec, ModelSpec, CalibrationError
from collateraldiscounting.curve_calibration import calibrate_curves
from collateraldiscounting.market_data import example_quotes
from collateraldiscounting.instruments import Quote, MarketQuotes, FXForward

MQ = example_quotes()


@pytest.fixture(scope="module")
def cal():
    return calibrate(MQ)


def test_calibration_reprices_all_quotes(cal):
    pr = cal.pricer()
    for q in MQ.quotes:
        tol = 1e-12 if isinstance(q.instrument, (Swap, FXForward, ResettingCCS)) else 1e-10
        if isinstance(q.instrument, FXForward) and q.instrument.quotation == "points":
            tol = 1e-8
        assert q.instrument.quote(pr) == pytest.approx(q.value, abs=tol), q.instrument.label
    assert cal.diagnostics["fixed_point_iterations"] <= 10


def test_round_trip_recovers_parameters(cal):
    """Quotes generated by the calibrated model are calibrated back to the same model."""
    pr = cal.pricer()
    synth = MQ.with_values([q.instrument.quote(pr) for q in MQ.quotes])
    spec2 = dataclasses.replace(cal.spec, c=dataclasses.replace(cal.spec.c, level=0.02))   # other start
    cal2 = calibrate(synth, spec2)
    for i in range(cal.model.nr + 1):
        np.testing.assert_allclose(cal2.model.sig[i], cal.model.sig[i], rtol=1e-8)
    T = np.linspace(0.1, 40, 30)
    np.testing.assert_allclose(cal2.curves.Z0("fd", T), cal.curves.Z0("fd", T), rtol=1e-12)


def test_input_order_does_not_matter(cal):
    shuffled = dataclasses.replace(MQ, quotes=list(np.random.default_rng(3).permutation(np.array(MQ.quotes, dtype=object))))
    cal2 = calibrate(shuffled)
    for i in range(cal.model.nr + 1):
        np.testing.assert_allclose(cal2.model.sig[i], cal.model.sig[i], rtol=1e-12)


def test_dcb_limit():
    """Without CB vols and with sigma^c = 0, K = 1 and the LGCM curve is the DCB curve."""
    mq = MQ.without(CurrencyBasisSwaption)
    spec = dataclasses.replace(default_spec(), c=BlockSpec((0.1,), level=0.0))
    cal0 = calibrate(mq, spec)
    dcb, _ = calibrate_curves(mq)
    T = np.linspace(0.1, 40, 30)
    np.testing.assert_allclose(cal0.curves.Z0("fd", T), dcb.Z0("fd", T), rtol=1e-14)
    assert cal0.diagnostics["fixed_point_iterations"] == 1


def test_collateral_convexity_moves_the_long_end(cal):
    """Same quotes: the LGCM Z^{f|d} differs from the DCB one beyond the FX forwards, not before."""
    dcb, _ = calibrate_curves(MQ)
    assert cal.curves.Z0("fd", 0.9) == pytest.approx(dcb.Z0("fd", 0.9), rel=1e-14)
    assert abs(cal.curves.Z0("fd", 30.0) / dcb.Z0("fd", 30.0) - 1) > 1e-4


def test_variance_squeeze_raises():
    low_fx = [Quote(q.instrument, 0.01) if isinstance(q.instrument, FXOption) and q.instrument.expiry == 10 else q
              for q in MQ.quotes]
    with pytest.raises(CalibrationError, match="FX option 10Y"):
        calibrate(dataclasses.replace(MQ, quotes=low_fx))


def test_duplicate_expiry_raises():
    dup = MQ.quotes + [q for q in MQ.quotes if isinstance(q.instrument, FXOption)][:1]
    with pytest.raises(ValueError, match="FX option"):
        calibrate(dataclasses.replace(MQ, quotes=dup))


def test_two_factor_domestic_block():
    spec = default_spec()
    corr = np.eye(5)
    corr[2:, 2:] = spec.corr[1:, 1:]
    corr[0, 1] = corr[1, 0] = 0.3
    corr[0, 2:] = corr[2:, 0] = spec.corr[0, 1:]
    corr[1, 2:] = corr[2:, 1] = 0.5 * spec.corr[0, 1:]
    spec2 = dataclasses.replace(spec, d=BlockSpec((0.03, 0.5), (1.0, 0.8), 0.006), corr=corr)
    cal2 = calibrate(MQ, spec2)
    pr = cal2.pricer()
    for q in MQ.of_type(Swaption):
        assert q.instrument.quote(pr) == pytest.approx(q.value, abs=1e-12)
    assert "d swaptions: Gaussian approximation (2 factors)" in cal2.diagnostics["approximations"]
    k, lam = cal2.levels("d")
    np.testing.assert_allclose(cal2.model.sig[1], 0.8 * cal2.model.sig[0])


def test_cb_vols_not_moved_by_later_expiries(cal):
    """A CB vol at T0 reads sigma^c on [T0, T0 + tenor) through the Libor convexities K_{T0,j}, i.e. the
    pieces of the next expiries. Each expiry sets only its own segment and the fixed point resolves the
    coupling: every CB vol is repriced to rounding (overwriting the tail left errors of 4e-11)."""
    pr = cal.pricer()
    for q in MQ.of_type(CurrencyBasisSwaption):
        assert q.instrument.quote(pr) == pytest.approx(q.value, abs=1e-14), q.instrument.label
