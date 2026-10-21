"""Caplets on Libor-type and compounded overnight coupons in the LGCM: closed form against independent
benchmarks (joint law of the exact transitions, Ho-Lee limit, Monte Carlo)."""
import numpy as np
import pytest

from convexity.correlations import joint_law
from convexity.curves import CurveSet, Index, LGCMSetup
from convexity.instruments import Caplet
from convexity.lgcm import Model, default_model
from convexity.market import LIBOR, Coupon, InitialCurves, black, caplet_total_variance
from convexity.montecarlo import Simulator, cv_mean_and_stderr
from convexity.pricers import DeterministicPricer, LGCMPricer
from convexity.schemes import ExactScheme

M, CV = default_model(), InitialCurves()


def ho_lee(sig=0.01):
    return Model([0.0, 60.0], [("d", [sig], [0.0])], np.eye(2), [0.1])


@pytest.mark.parametrize("a,th", [(0.25, 0.25), (1.0, 0.25), (5.0, 1.0)])
def test_ho_lee_limit(a, th):
    """Constant vol, no mean reversion: w_Libor = sig^2 th^2 a, w_continuous = sig^2 th^2 (a + th/3)."""
    m, s = ho_lee(), 0.01
    assert caplet_total_variance(m, "d", a, a + th, LIBOR) == pytest.approx(s * s * th * th * a, rel=1e-12)
    assert caplet_total_variance(m, "d", a, a + th, Coupon("continuous")) == pytest.approx(s * s * th * th * (a + th / 3), rel=1e-12)


@pytest.mark.parametrize("block", ["d", "f"])
@pytest.mark.parametrize("step", [1 / 12, 1 / 52])
def test_daily_variance_matches_the_transition_law(block, step):
    """ln C = sum_k -ln Z_{t_k,t_{k+1}} is linear in the stacked states at the fixing dates; its variance from
    the joint law of the exact transitions (independent of the covariance integrals) equals the closed form."""
    a, b = 1.0, 1.25
    cp = Coupon("daily", step)
    g = cp.grid(a, b)
    dates, _, cov, ix = joint_law(M, list(g[:-1]))
    n, v = ix["n"], np.zeros(len(dates) * ix["n"])
    for t0, t1 in zip(g[:-1], g[1:]):
        coef, _ = M.lnZ(block * 2, t0, t1)
        k = dates.index(float(t0))
        v[k * n:k * n + M.nr] -= coef
    assert caplet_total_variance(M, block, a, b, cp) == pytest.approx(v @ cov @ v, rel=1e-13)


def test_libor_is_the_one_step_daily_caplet_and_continuous_is_the_limit():
    a, b = 1.0, 1.25
    assert caplet_total_variance(M, "d", a, b, Coupon("daily", b - a)) == pytest.approx(
        caplet_total_variance(M, "d", a, b, LIBOR), rel=1e-14)
    wc = caplet_total_variance(M, "d", a, b, Coupon("continuous"))
    e = [abs(caplet_total_variance(M, "d", a, b, Coupon("daily", h)) - wc) for h in (1 / 12, 1 / 52, 1 / 365)]
    assert e[0] > e[1] > e[2]
    assert e[0] / e[1] == pytest.approx((1 / 12) / (1 / 52), rel=0.15)        # O(step)


def test_in_accrual_variance_is_positive_and_spot_libor_caplet_has_none():
    for a, b in ((0.0, 0.25), (1.0, 1.25), (10.0, 11.0)):
        assert caplet_total_variance(M, "d", a, b, Coupon("daily")) > caplet_total_variance(M, "d", a, b, LIBOR)
    assert caplet_total_variance(M, "d", 0.0, 0.25, LIBOR) == 0.0


@pytest.mark.parametrize("block,a,b", [("d", 0.25, 0.5), ("d", 2.0, 2.25), ("f", 1.0, 1.25)])
def test_monte_carlo_prices_the_rfr_caplet_and_rejects_the_libor_variance(block, a, b):
    """Exact LGCM scheme, weekly compounding, ATM: MC within 4 s.e. of the closed form; the same caplet priced
    with the Libor variance (fixing at a) is rejected (measured 7.6 to 155 s.e.)."""
    cp = Coupon("daily", 1 / 52)
    g, kind = cp.grid(a, b), block * 2
    sim = Simulator(M, scheme=ExactScheme(0.25))
    st = sim.simulate_dates(list(g), 2**18, seed=5)
    C = np.exp(sum(-np.log(sim.discount(st(t0), CV, kind, t1)) for t0, t1 in zip(g[:-1], g[1:])))
    D = sim.deflator(st(b), CV, block, block)              # payment in the block's currency, native collateral
    Za, Zb = CV.Z0(kind, a), CV.Z0(kind, b)
    F = Za / Zb
    est, se = cv_mean_and_stderr(D * np.maximum(C - F, 0.0), [D * C, D], [Za, Zb])   # exact means of the controls
    closed = Zb * black(F, F, caplet_total_variance(M, block, a, b, cp))
    libor = Zb * black(F, F, caplet_total_variance(M, block, a, b, LIBOR))
    assert abs(est - closed) < 4 * se
    assert abs(est - libor) > 7 * se


SOFR = Index("USD_SOFR", "USD")
ESTR = Index("EUR_ESTR", "EUR")
E3M = Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.0010)


def test_caplet_instrument_and_pricer():
    from convexity.market_data import example_quotes
    from convexity.calibration import calibrate
    cal = calibrate(example_quotes())
    pr = cal.pricer()
    rfr, ibor = Caplet(SOFR, 1.0, 1.25), Caplet(E3M, 1.0, 1.25)
    for cap, coupon, blk, key in ((rfr, Coupon("daily"), "d", "dd"), (ibor, LIBOR, "f", "ff")):
        F, Zb = pr.caplet_forward(cap)
        Z = lambda T: cal.curves.Z0(key, T)
        assert F == pytest.approx(cap.index.kappa * Z(1.0) / Z(1.25), rel=1e-14)
        w = caplet_total_variance(cal.model, blk, 1.0, 1.25, coupon)
        assert cap.quote(pr) == pytest.approx(Zb * black(F, F, w), rel=1e-14)
    otm = Caplet(SOFR, 1.0, 1.25, strike=0.06)
    assert 0 < otm.quote(pr) < rfr.quote(pr)
    assert Caplet(E3M, 0.0, 0.25, strike=0.0).quote(pr) == pytest.approx(        # spot Libor caplet: intrinsic
        cal.curves.Z0("ff", 0.25) * (E3M.kappa / cal.curves.Z0("ff", 0.25) - 1.0), rel=1e-12)
    assert rfr.target(cal.curves.setup) == "d" and rfr.label == "USD_SOFR caplet 1Y-1.25Y ATM"
    with pytest.raises(NotImplementedError):
        rfr.quote(DeterministicPricer(cal.curves))
    with pytest.raises(ValueError):
        Caplet(SOFR, 1.0, 1.0)
