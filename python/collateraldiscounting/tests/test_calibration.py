"""LGCM pricer (Task 4), calibration (Task 5), repricing checks (Task 6)."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from collateraldiscounting import gaussian_approx as ga
from collateraldiscounting.lgcm import Model
from collateraldiscounting.curves import Index, LGCMSetup, CurveSet, PiecewiseFlatCurve, SpreadCurve
from collateraldiscounting.instruments import (Swap, ResettingCCS, Swaption, FXOption, CurrencyBasisSwaption,
                                               schedule)
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
    m.sig[m.idx["c"][0]][:] *= 2                       # the cache must see model changes
    assert pr.libor_convexity(10.0, 10.25) == libor_convexity(m, 0.0, 10.0, 10.25)
    assert FXOption(5.0).quote(pr) == pytest.approx(np.sqrt(fx_total_variance(m, 5.0) / 5.0), rel=1e-14)
    # LGCM CCS spread differs from the DCB one by the convexity only
    ccs = ResettingCCS(20.0)
    assert ccs.quote(pr) < ccs.quote(DeterministicPricer(cs))
