"""Compounded overnight coupons: convexity closed forms, exact benchmark (Task 1), pricer (Task 2),
markets and inverted FX (Task 3), Monte Carlo (Task 4)."""
import numpy as np
import pytest

from convexity.lgcm import default_model
from convexity.market import (InitialCurves, Coupon, LIBOR, coupon_grid, libor_convexity, compounded_convexity,
                              _compounded_convexity_scalar)
from convexity.benchmarks import coupon_value_exact, coupon_expectation_exact
from convexity.curves import Index, LGCMSetup, CurveSet

M, CV = default_model(), InitialCurves()


def test_coupon_grid_lands_on_b():
    g = coupon_grid(1.0, 1.25, 1 / 52)
    assert g[0] == 1.0 and g[-1] == 1.25 and np.all(np.diff(g) > 0)
    assert np.all(np.diff(g)[:-1] == pytest.approx(1 / 52)) and np.diff(g)[-1] <= 1 / 52 + 1e-15
    np.testing.assert_array_equal(coupon_grid(1.0, 1.25, 1.0), [1.0, 1.25])


def test_libor_is_the_one_step_daily_coupon():
    for t in (0.0, 1.0):
        assert Coupon("daily", 10.0).convexity(M, t, 5.0, 5.25) == pytest.approx(libor_convexity(M, t, 5.0, 5.25), rel=1e-14)
        assert LIBOR.convexity(M, t, 5.0, 5.25) == libor_convexity(M, t, 5.0, 5.25)


@pytest.mark.parametrize("coupon", [LIBOR, Coupon("daily", 1 / 52), Coupon("daily", 1 / 12)])
@pytest.mark.parametrize("t", [0.0, 2.0])
def test_closed_form_matches_exact_gaussian_benchmark(coupon, t):
    a, b = 5.0, 5.25
    assert coupon.convexity(M, t, a, b) == pytest.approx(coupon_value_exact(M, CV, coupon, a, b, t), rel=1e-11)


def test_partially_accrued_period():
    """t strictly inside the accrual period: only the days after t carry convexity."""
    c = Coupon("daily", 1 / 52)
    a, b, t = 5.0, 5.25, 5.0 + 5 / 52 + 1e-9
    assert c.convexity(M, t, a, b) == pytest.approx(coupon_value_exact(M, CV, c, a, b, t), rel=1e-11)
    assert Coupon("daily", 1 / 52).convexity(M, b, a, b) == 1.0


def test_continuous_is_the_limit_of_daily_with_order_one():
    a, b = 10.0, 10.25
    kc = Coupon("continuous").convexity(M, 0.0, a, b)
    errs = [abs(np.log(Coupon("daily", h).convexity(M, 0.0, a, b) / kc)) for h in (1 / 26, 1 / 52, 1 / 104)]
    assert errs[0] / errs[1] == pytest.approx(2, rel=0.1) and errs[1] / errs[2] == pytest.approx(2, rel=0.1)


def test_compounded_and_libor_convexities_differ_by_the_accrual_term():
    a, b = 10.0, 10.25
    kl, kc = LIBOR.convexity(M, 0.0, a, b), Coupon("continuous").convexity(M, 0.0, a, b)
    assert kl != kc
    extra = -M.I(M.Sig("f", b), M.Sig("c", b), a, b)
    assert np.log(kc / kl) == pytest.approx(extra, rel=1e-12)


# ---------------------------------------------------------------- sanity checks of the benchmark itself
@pytest.mark.parametrize("coupon", [LIBOR, Coupon("daily", 1 / 52)])
@pytest.mark.parametrize("t", [0.0, 2.0, 5.0 + 5 / 52 + 1e-9])
def test_benchmark_native_coupon_is_model_free(coupon, t):
    """E^{f|f,b}_t[C] = Z^{f|f}_{t,a'}/Z^{f|f}_{t,b} (a' = first fixing after t): native telescoping."""
    a, b = 5.0, 5.25
    g = coupon_grid(a, b, 10.0 if coupon.kind == "libor" else coupon.step)
    a1 = g[g > t][0] if t > a else a
    for z0 in (None, np.array([0.01, -0.02, 0.015, -0.005])):
        ex = coupon_expectation_exact(M, CV, coupon, a, b, t, "f", z0)
        zt = np.zeros(M.nr) if z0 is None else z0
        ga, ca = M.lnZ("ff", t, a1)
        gb, cb = M.lnZ("ff", t, b)
        native = CV.Z0("ff", a1) / CV.Z0("ff", b) * np.exp(ga @ zt + ca - gb @ zt - cb)
        assert ex == pytest.approx(native, rel=1e-12)


def test_benchmark_convexity_does_not_depend_on_the_state_at_t():
    """K_t is deterministic in the LGCM: the ratio is the same for any z_t."""
    c, a, b, t = Coupon("daily", 1 / 12), 5.0, 5.25, 2.0
    z0 = np.array([0.01, -0.02, 0.015, -0.005])
    k0 = coupon_expectation_exact(M, CV, c, a, b, t, "d") / coupon_expectation_exact(M, CV, c, a, b, t, "f")
    k1 = coupon_expectation_exact(M, CV, c, a, b, t, "d", z0) / coupon_expectation_exact(M, CV, c, a, b, t, "f", z0)
    assert k1 == pytest.approx(k0, rel=1e-12)
    assert k0 == pytest.approx(coupon_value_exact(M, CV, c, a, b, t), rel=1e-12)


def test_vectorised_daily_convexity_matches_scalar_oracle():
    for t, a, b in ((0.0, 10.0, 10.25), (2.0, 5.0, 5.25), (5.1, 5.0, 5.25), (0.0, 29.9, 30.2)):
        assert compounded_convexity(M, t, a, b) == pytest.approx(_compounded_convexity_scalar(M, t, a, b), rel=1e-13)


# ---------------------------------------------------------------- indices
def test_index_coupon():
    assert Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.001).coupon == LIBOR
    assert Index("GBP_SONIA", "GBP").coupon == Coupon("daily", 1 / 365)
    assert Index("GBP_SONIA", "GBP", compounding="continuous").coupon == Coupon("continuous", 1 / 365)
    with pytest.raises(ValueError):
        Index("X", "GBP", compounding="weekly")
    with pytest.raises(ValueError):
        Index("X", "GBP", "IBOR", 0.25, 0.0, compounding="continuous")   # compounding is for OIS only
    with pytest.raises(ValueError):
        Index("X", "GBP", on_step=0.0)
    with pytest.raises(ValueError):
        Coupon("weekly")


def test_curve_set_and_initial_curves_coupon():
    sofr, estr = Index("USD_SOFR", "USD"), Index("EUR_ESTR", "EUR", compounding="continuous")
    cs = CurveSet(LGCMSetup(sofr, estr), 1.1)
    assert cs.coupon("d") == Coupon("daily") and cs.coupon("f") == Coupon("continuous")
    cs3 = CurveSet(LGCMSetup(sofr, estr, foreign_index=Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.001)), 1.1)
    assert cs3.coupon("f") == LIBOR
    assert CV.coupon("d") == LIBOR and CV.coupon("f") == LIBOR


# ---------------------------------------------------------------------- Task 2: coupon-aware pricers
from convexity.curves import PiecewiseFlatCurve, SpreadCurve
from convexity.instruments import ResettingCCS, CurrencyBasisSwaption
from convexity.pricers import LGCMPricer, DeterministicPricer
from convexity import gaussian_approx as ga

SOFR = Index("USD_SOFR", "USD")
SONIA = Index("GBP_SONIA", "GBP", on_step=1 / 52)
SETUP_RFR = LGCMSetup(SOFR, SONIA)                         # foreign index = SONIA (compounded)


def rfr_curves():
    cs = CurveSet(SETUP_RFR, 1.27)
    cs[SETUP_RFR.key("dd")] = PiecewiseFlatCurve([2, 10, 50], [0.042, 0.038, 0.04])
    cs[SETUP_RFR.key("ff")] = PiecewiseFlatCurve([2, 10, 50], [0.040, 0.037, 0.039])
    cs[SETUP_RFR.key("fd")] = SpreadCurve(cs[SETUP_RFR.key("ff")], PiecewiseFlatCurve([5, 50], [-0.0005, -0.0008]))
    return cs


def test_rfr_ccs_uses_the_compounded_convexity_not_libor():
    cs = rfr_curves()
    pr = LGCMPricer(M, cs)
    ccs = ResettingCCS(20.0)
    Tj = ccs.schedule(SETUP_RFR)
    P, Zf = cs.Z0("fd", Tj), cs.Z0("ff", Tj)
    K = np.array([SONIA.coupon.convexity(M, 0.0, a, b) for a, b in zip(Tj[:-1], Tj[1:])])
    B = ((P[:-1] / P[1:] - Zf[:-1] / Zf[1:] * K) * P[1:]).sum() / (0.25 * P[1:].sum())
    assert ccs.quote(pr) == pytest.approx(B, rel=1e-13)
    KL = np.array([libor_convexity(M, 0.0, a, b) for a, b in zip(Tj[:-1], Tj[1:])])
    BL = ((P[:-1] / P[1:] - Zf[:-1] / Zf[1:] * KL) * P[1:]).sum() / (0.25 * P[1:].sum())
    assert abs(ccs.quote(pr) - BL) > 1e-7                   # the Libor K is not used (negative control)


def test_gaussian_approx_and_exact_forward_consistency():
    """Martingale / pricer consistency only (shares _ingredients, so it does not by itself prove the coupon
    is used: see test_ingredients_use_the_curves_coupon for that)."""
    cs = rfr_curves()
    s = ga.SwapSetup(T0=5.0, theta=0.25, n=8, curves=cs)
    a = ga.approx(M, s)
    cbs = CurrencyBasisSwaption.make(5.0, 2.0)
    assert cbs.underlying.quote(LGCMPricer(M, cs, exact=False)) == pytest.approx(a["B0"], rel=1e-13)
    # the exact quadrature reprices the forward under the annuity measure (martingale check)
    ex = ga.exact(M, s)
    assert ex["B"]["mean"] == pytest.approx(a["B0"], abs=1e-14)
    assert ex["Y"]["mean"] == pytest.approx(a["Y0"], abs=1e-14)


def test_ingredients_use_the_curves_coupon():
    s = ga.SwapSetup(T0=5.0, theta=0.25, n=8, curves=rfr_curves())
    _, _, _, _, K0, KT = ga._ingredients(M, s)
    pairs = list(zip(s.Tj[:-1], s.Tj[1:]))
    for K, t in ((K0, 0.0), (KT, s.T0)):
        sonia = np.array([SONIA.coupon.convexity(M, t, a, b) for a, b in pairs])
        libor = np.array([libor_convexity(M, t, a, b) for a, b in pairs])
        np.testing.assert_allclose(K[1:], sonia, rtol=1e-14)
        # discriminating: the gap between the compounded and the Libor K is resolved by this threshold
        assert np.all(np.abs(np.log(K[1:]) - np.log(libor)) > 1e-9)


def test_swap_convexities_pick_up_the_curves_coupon():
    from convexity.payoffs import Swap
    sw = Swap(5.0, 8, 0.25, 0.03, ccy="f", collateral="d")
    pairs = list(zip(sw.Tj[:-1], sw.Tj[1:]))
    for t in (0.0, 2.0):
        sonia = np.array([SONIA.coupon.convexity(M, t, a, b) for a, b in pairs])
        libor = np.array([libor_convexity(M, t, a, b) for a, b in pairs])
        np.testing.assert_allclose(sw.convexities(M, t, rfr_curves()), sonia, rtol=1e-14)
        np.testing.assert_allclose(sw.convexities(M, t), libor, rtol=1e-14)             # default: Libor
        np.testing.assert_allclose(sw.convexities(M, t, CV), libor, rtol=1e-14)         # InitialCurves: Libor
        assert np.all(np.abs(np.log(sonia) - np.log(libor)) > 1e-9)


def test_cb_straddle_inputs_use_the_curves_coupon():
    from types import SimpleNamespace
    from convexity.calibration_check import cb_straddle_inputs
    from convexity.montecarlo import Simulator, mean_and_stderr
    from convexity.schemes import ExactScheme
    cs = rfr_curves()
    cal = SimpleNamespace(curves=cs, model=M, pricer=lambda: LGCMPricer(M, cs))
    inst = CurrencyBasisSwaption.make(5.0, 2.0)
    q = SimpleNamespace(instrument=inst)
    sim = Simulator(M, scheme=ExactScheme(0.25))
    st = sim.simulate_dates([5.0], 2**14, seed=5)
    D, V, A0, T0 = cb_straddle_inputs(sim, st, cal, q)
    # rebuild the pathwise control by hand with the SONIA K seen from T0 and with the Libor K
    u = inst.underlying
    Tj = u.schedule(SETUP_RFR)
    res = st(T0)
    one = np.ones(res.n_paths)
    P = np.stack([one] + [sim.discount(res, cs, "fd", t) for t in Tj[1:]], axis=1)
    Zf = np.stack([one] + [sim.discount(res, cs, "ff", t) for t in Tj[1:]], axis=1)
    B0, th = u.quote(cal.pricer()), SONIA.tenor

    def control(Kfun):
        K = np.array([Kfun(M, T0, a, b) for a, b in zip(Tj[:-1], Tj[1:])])
        AB = ((P[:, :-1] / P[:, 1:] - Zf[:, :-1] / Zf[:, 1:] * K) * P[:, 1:]).sum(1)
        return AB - th * P[:, 1:].sum(1) * B0
    np.testing.assert_allclose(V, control(SONIA.coupon.convexity), rtol=1e-12, atol=1e-14)
    assert np.max(np.abs(V - control(lambda m, t, a, b: libor_convexity(m, t, a, b)))) > 1e-9
    mu, se = mean_and_stderr(D * V)                        # zero-mean control identity with the compounded K
    assert abs(mu) < 4 * se


def test_initial_curves_paths_keep_libor():
    s = ga.SwapSetup(T0=5.0, theta=0.25, n=8, curves=CV)
    _, _, _, _, K0, _ = ga._ingredients(M, s)
    _, _, _, _, K0, KT = ga._ingredients(M, s)
    for K, t in ((K0, 0.0), (KT, s.T0)):
        np.testing.assert_allclose(K[1:], [libor_convexity(M, t, a, b) for a, b in zip(s.Tj[:-1], s.Tj[1:])], rtol=1e-15)


# ---------------------------------------------------------------------- Task 3: inverted FX, RFR markets
from functools import lru_cache

from convexity.instruments import FXForward, MarketQuotes, Quote, Swap
from convexity.curve_calibration import calibrate_curves
from convexity.calibration import calibrate
from convexity.market_data import example_quotes, MARKETS
from convexity.calibration_check import analytic_report, mc_report
from convexity.pricers import DeterministicPricer


@lru_cache(maxsize=None)
def _cal(market):
    return calibrate(example_quotes(market))


def test_inverted_fx_quotes_give_the_same_curve():
    mq = example_quotes("USDJPY")
    assert mq.fx_inverted and mq.x0 == pytest.approx(1 / mq.fx_spot)
    cs, _ = calibrate_curves(mq)
    # direct outrights generated from the inverted-quote curve rebuild the same Z^{f|d}
    pr = DeterministicPricer(cs)
    direct = [Quote(FXForward(q.instrument.maturity), FXForward(q.instrument.maturity).quote(pr))
              if isinstance(q.instrument, FXForward) else q for q in mq.quotes]
    cs2, _ = calibrate_curves(MarketQuotes(mq.setup, mq.x0, direct))
    T = np.linspace(0.05, 30, 40)
    np.testing.assert_allclose(cs2.Z0("fd", T), cs.Z0("fd", T), rtol=1e-12)


@pytest.mark.parametrize("market", MARKETS)
def test_calibration_reprices_every_market(market):
    cal = _cal(market)
    assert cal.diagnostics["seconds"] < 30
    rows = analytic_report(cal)
    for r in rows:
        tol = 1e-8 if r.kind == "FXForward" else 1e-10
        assert abs(r.error) < tol, (market, r.label, r.error)


def test_rfr_markets_use_compounded_coupons():
    for market in ("GBPUSD", "USDJPY"):
        mq = example_quotes(market)
        assert mq.setup.foreign_index.kind == "OIS" and mq.setup.foreign_index.coupon.kind == "daily"


def test_default_market_is_eurusd_and_not_inverted():
    mq = example_quotes()
    assert not mq.fx_inverted and mq.x0 == mq.fx_spot == 1.10 and mq.setup.foreign_index.kind == "IBOR"


@pytest.mark.parametrize("quotation", ["outright", "points"])
def test_inverted_forward_quote_is_reciprocal(quotation):
    mq = example_quotes("USDJPY")
    cs, _ = calibrate_curves(mq)
    pr = DeterministicPricer(cs)
    T = 0.5
    F = FXForward(T).quote(pr)
    inv = FXForward(T, quotation, 100.0, inverted=True).quote(pr)
    expect = 1 / F if quotation == "outright" else (1 / F - 1 / cs.x0) * 100.0
    assert inv == pytest.approx(expect, rel=1e-14)
    assert FXForward(T, quotation, 100.0, True).label.startswith("FX forward") and "inverted" in FXForward(T, quotation, 100.0, True).label


def test_fx_option_vol_is_invariant_under_inversion():
    """ln x_T Gaussian => y = 1/x_T lognormal with the same total variance w, so the ATM Black vol of y
    (strike E[y], price E[(y - E y)^+] / E[y] = 2 N(sqrt(w)/2) - 1) equals that of x. Checked by quadrature."""
    from scipy.integrate import quad
    from scipy.stats import norm
    w = 0.09 ** 2 * 2.0
    y = lambda z: np.exp(w / 2 - np.sqrt(w) * z)       # 1/x with ln x = -w/2 + sqrt(w) z
    Ey = np.exp(w)                                      # E[1/x]
    zstar = (w / 2 - np.log(Ey)) / np.sqrt(w)           # y > Ey iff z < zstar
    price = quad(lambda z: (y(z) - Ey) * norm.pdf(z), -np.inf, zstar, epsabs=1e-14, epsrel=1e-13)[0] / Ey
    assert 2 * norm.ppf(0.5 * (price + 1)) == pytest.approx(np.sqrt(w), rel=1e-9)


def test_mc_inverted_forward_se_is_propagated():
    """Quoted model value 1/F_hat with s.e. s_F / F^2 (delta method), points: pip * s_F / F^2."""
    cal = _cal("USDJPY")
    fxs = [q for q in cal.quotes.quotes if isinstance(q.instrument, FXForward)]
    q = fxs[-1]
    T = q.instrument.maturity
    row = mc_report(cal, n_paths=2**13, seed=3, quotes=[q])[0]
    pr = cal.pricer(exact=True)
    direct = FXForward(T)
    qd = Quote(direct, direct.quote(pr))
    rd = mc_report(cal, n_paths=2**13, seed=3, quotes=[qd])[0]
    F, sF = rd.model, rd.stderr
    x0, pip = cal.curves.x0, q.instrument.pip_factor
    assert row.model == pytest.approx((1 / F - 1 / x0) * pip, rel=1e-9)
    assert row.stderr == pytest.approx(pip * sF / F**2, rel=1e-3)
    assert abs(row.z) < 4
