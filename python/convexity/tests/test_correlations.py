"""Correlations of log-FX and swap (CMS) rates: Gaussian approximation, exact benchmark, Monte Carlo."""
import numpy as np
import pytest

from convexity import gaussian_approx as ga
from convexity.correlations import (LogFX, SwapRate, correlation_approx, correlation_exact, correlation_mc,
                                    joint_law, state_covariance, weighted_correlation)
from convexity.lgcm import default_model
from convexity.market import InitialCurves
from convexity.montecarlo import Simulator, SimResult
from convexity.schemes import ExactScheme

M, CV = default_model(), InitialCurves()
PAIRS = [(LogFX(5.0), SwapRate("d", 5.0, 10.0)),                 # FX vs CMS, same expiry
         (LogFX(5.0), SwapRate("f", 5.0, 10.0)),
         (SwapRate("d", 5.0, 2.0), SwapRate("d", 5.0, 10.0)),     # two tenors, same expiry
         (SwapRate("d", 5.0, 10.0), SwapRate("f", 5.0, 10.0)),    # two currencies
         (SwapRate("d", 2.0, 10.0), SwapRate("d", 5.0, 10.0)),    # intertemporal
         (LogFX(2.0), SwapRate("d", 5.0, 10.0))]                  # intertemporal FX vs CMS
IDS = [f"{a.label}-{b.label}" for a, b in PAIRS]


def test_state_covariance_matches_the_transition_law():
    """Closed-form covariance integrals across dates against the joint law of the exact transitions."""
    for T1, T2 in ((5.0, 5.0), (2.0, 5.0), (5.0, 2.0), (0.7, 12.0)):
        dates, _, cov, ix = joint_law(M, [T1, T2])
        n, pos = ix["n"], list(range(M.nr)) + [ix["iZX"]]
        a, b = dates.index(T1) * n, dates.index(T2) * n
        J = cov[np.ix_([a + p for p in pos], [b + p for p in pos])]
        np.testing.assert_allclose(state_covariance(M, T1, T2), J, rtol=0, atol=1e-16)


def test_swap_gradient_is_the_swaption_gradient():
    s = ga.SwapSetup(T0=5.0, theta=0.25, n=40, curves=CV)
    for ccy in "df":
        ref = ga.approx_native(M, s, ccy=ccy)
        X = SwapRate(ccy, 5.0, 10.0, fixed_tenor=0.25, float_tenor=0.25)
        g = X.gradient(M, CV)
        np.testing.assert_allclose(g[M.idx[ccy]], ref["q"], rtol=1e-12)
        assert np.count_nonzero(g) == len(M.idx[ccy])
        assert X.forward(CV) == pytest.approx(ref["Y0"], rel=1e-13)
        assert correlation_approx(M, CV, X, X).var1 == pytest.approx(ref["V"], rel=1e-12)


def test_swap_value_matches_the_simulator_reconstruction():
    """SwapRate.value against swap legs on the discount factors of Simulator.discount (independent code)."""
    X = SwapRate("d", 5.0, 10.0)
    z = np.random.default_rng(2).normal(scale=0.01, size=(7, M.nr))
    res = SimResult(5.0, z, np.zeros(7), np.broadcast_to(M.Y(5.0), (7, M.nr, M.nr)),
                    np.broadcast_to(M.vfc(5.0), (7, M.nr)), np.zeros(7), np.zeros(7))
    sim = Simulator(M, scheme=ExactScheme())
    tf, cf, tx, wx = X.cashflows(CV)
    fl = sum(c * sim.discount(res, CV, "dd", t) for c, t in zip(cf, tf))
    A = sum(w * sim.discount(res, CV, "dd", t) for w, t in zip(wx, tx))
    np.testing.assert_allclose(X.value(M, CV, {"z": z, "x": np.zeros(7)}), fl / A, rtol=1e-13)


def test_linear_observables_are_exact():
    for a, b in ((LogFX(2.0), LogFX(5.0)), (LogFX(5.0), LogFX(5.0))):
        assert correlation_exact(M, CV, a, b).corr == pytest.approx(correlation_approx(M, CV, a, b).corr, abs=1e-12)
    assert correlation_approx(M, CV, LogFX(3.0), LogFX(3.0)).corr == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("pair", PAIRS, ids=IDS)
def test_quadrature_is_converged(pair):
    a, b = pair
    assert correlation_exact(M, CV, a, b, n=12).corr == pytest.approx(correlation_exact(M, CV, a, b, n=24).corr, abs=1e-12)


def test_gaussian_approximation_error_is_second_order():
    """Under vol scaling lambda the approximation error scales like lambda^2 (measured, not proved)."""
    for a, b in PAIRS[:5]:
        err = [correlation_approx(default_model(s), CV, a, b).corr - correlation_exact(default_model(s), CV, a, b).corr
               for s in (0.5, 1.0, 2.0)]
        assert err[1] / err[0] == pytest.approx(4.0, rel=0.03) and err[2] / err[1] == pytest.approx(4.0, rel=0.03)
        assert abs(err[1]) < 3e-4


@pytest.mark.parametrize("pair", PAIRS, ids=IDS)
@pytest.mark.parametrize("measure", ["RN", "Tfwd"])
def test_mc_with_control_matches_exact_and_rejects_the_approximation(pair, measure):
    a, b = pair
    kw = dict(measure=measure, Tstar=15.0 if measure == "Tfwd" else None)
    ex = correlation_exact(M, CV, a, b, **kw)
    mc = correlation_mc(M, CV, a, b, 2**16, seed=1, control=True, **kw)
    assert abs(mc.corr - ex.corr) < 4 * mc.stderr
    ap = correlation_approx(M, CV, a, b)
    # negative control under RN (s.e. ~3e-6); the T*-forward weights raise the s.e. to ~4e-5, about the
    # size of the approximation error, so there the MC only confirms the exact value
    if measure == "RN":
        assert abs(ap.corr - mc.corr) > 10 * mc.stderr


@pytest.mark.parametrize("pair", PAIRS[:3], ids=IDS[:3])
def test_plain_mc_matches_exact(pair):
    a, b = pair
    ex = correlation_exact(M, CV, a, b)
    mc = correlation_mc(M, CV, a, b, 2**16, seed=2)
    assert abs(mc.corr - ex.corr) < 4 * mc.stderr and mc.stderr < 1e-2


def test_measures_differ_at_second_order():
    a, b = PAIRS[0]
    d = [correlation_exact(default_model(s), CV, a, b, "Tfwd", 15.0).corr - correlation_exact(default_model(s), CV, a, b).corr
         for s in (1.0, 2.0)]
    assert d[1] / d[0] == pytest.approx(4.0, rel=0.05) and 1e-6 < abs(d[0]) < 1e-3


def test_stderr_is_calibrated():
    """z-scores over 60 seeds (2^12 paths): standard deviation close to 1, plain and with control."""
    a, b = PAIRS[0]
    ex = correlation_exact(M, CV, a, b).corr
    for control in (False, True):
        z = [(lambda r: (r.corr - ex) / r.stderr)(correlation_mc(M, CV, a, b, 2**12, seed=s, control=control))
             for s in range(60)]
        assert 0.75 < np.std(z) < 1.3 and abs(np.mean(z)) < 0.4


def test_weighted_correlation():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=1000), rng.normal(size=1000)
    y = 0.6 * x + 0.8 * y
    assert weighted_correlation(x, y, antithetic=False)[0] == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-12)
    w = rng.integers(1, 4, size=1000)
    assert weighted_correlation(x, y, w, antithetic=False)[0] == pytest.approx(
        np.corrcoef(np.repeat(x, w), np.repeat(y, w))[0, 1], rel=1e-12)


def test_validation():
    with pytest.raises(ValueError):
        SwapRate("x", 5.0, 10.0)
    with pytest.raises(ValueError):
        correlation_exact(M, CV, *PAIRS[0], measure="Tfwd", Tstar=3.0)
    with pytest.raises(ValueError):
        correlation_mc(M, CV, *PAIRS[0], 2**8, measure="Tfwd")
