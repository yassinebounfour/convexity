"""Checks of the LGCM / QGCM derivations in docs/CollateralDiscounting.tex.

Run from python/ with ``python -m pytest convexity -q``.

Each test corresponds to a verification remark of the notes. The benchmarks are exact
(no-arbitrage identities, moment ODEs, quadrature), so tolerances are near machine precision,
except for the Gaussian approximations which are asymptotic.
"""
import itertools

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import quad

from convexity.lgcm import default_model, E, Phi1, Phi2
from convexity import moments, transitions, gaussian_approx


@pytest.fixture(scope="module")
def m():
    return default_model()


def eexp(a, mX, P, c0=0.0):
    """E[exp(a.X + c0)] for X ~ N(mX, P)."""
    return np.exp(a @ mX + 0.5 * a @ P @ a + c0)


# --------------------------------------------------------------------------- covariance integrals
@pytest.mark.parametrize("a,b,D", [(0, 0, 1.0), (1e-9, 0.3, 2.0), (-0.02, 0.6, 10.0), (0.05, 0.05, 30.0),
                                   (2.0, 1e-6, 5.0), (0.7, -0.7, 3.0), (1e-4, -1e-4, 40.0), (-0.3, 0.1, 0.25)])
def test_kernels_match_quadrature(a, b, D):
    p1 = quad(lambda s: E(a, s), 0, D, epsabs=1e-15, epsrel=1e-13)[0]
    p2 = quad(lambda s: E(a, s) * E(b, s), 0, D, epsabs=1e-15, epsrel=1e-13)[0]
    assert Phi1(a, D) == pytest.approx(p1, rel=1e-13)
    assert Phi2(a, b, D) == pytest.approx(p2, rel=1e-13)


def test_covariance_integrals_match_brute_force(m):
    """Algorithm 'Covariance integrals' vs nested adaptive quadrature (G itself by quadrature)."""
    T = [0.7, 3.0, 12.0]
    loads = [(0, "M"), (1, "G"), (2, "G"), (3, "M"), (1, "M"), (4, "1"), (3, "G")]
    worst = 0.0
    for (i, ti), (j, tj) in itertools.combinations_with_replacement(loads, 2):
        for s, t in [(0, 0.7), (0.3, 2.6), (1.1, 11.0)]:
            la = (i, ti, max(t, T[(i + j) % 3]))
            lb = (j, tj, max(t, T[(2 * i + j) % 3]))
            exact = m.covint_brute(la, lb, s, t)
            worst = max(worst, abs(m.covint(la, lb, s, t) - exact) / abs(exact))
    assert worst < 1e-12


def test_forward_tenor_amortising_factor(m):
    """G_{t,T;theta} = G_{t,T+theta} - G_{t,T} = M_{t,T} G_{T,T+theta} (footnote in the LGCM section)."""
    for i in range(m.nr):
        for t, T, th in [(0.3, 2.2, 0.25), (1.0, 7.5, 0.5), (0.0, 14.0, 1.0)]:
            assert m.G(i, t, T + th) - m.G(i, t, T) == pytest.approx(m.M(i, t, T) * m.G(i, T, T + th), rel=1e-11)


# --------------------------------------------------------------------------- QGCM theorem (moment ODEs)
def _aoa_errors(m, t, T, drop=None):
    mX, P, Y, v = moments.propagate(m, t, drop)
    ix = moments.indices(m)
    n, nr = ix["n"], m.nr
    out = {}
    a_, c_ = m.lnZ("dd", t, T, Y, v)
    a = np.zeros(n); a[:nr] = a_; a[ix["iID"]] = -1
    out["Zdd"] = eexp(a, mX, P, c_) - 1
    a_, c_ = m.lnZ("fd", t, T, Y, v)
    a = np.zeros(n); a[:nr] = a_; a[ix["iID"]] = -1; a[ix["iZX"]] = 1
    out["xZfd"] = eexp(a, mX, P, c_) - 1
    a_, c_ = m.lnZ("ff", t, T, Y, v)
    a = np.zeros(n); a[:nr] = a_; a[ix["iID"]] = -1; a[ix["iIQ"]] = 1; a[ix["iZX"]] = 1
    out["xZff"] = eexp(a, mX, P, c_) - 1
    return out, (mX, P, Y, v)


@pytest.mark.parametrize("t", [0.8, 2.5, 7.0])
def test_qgcm_no_arbitrage_identities(m, t):
    """Theorem QGCM: the three martingale identities under Q^{d,RN}, with y, v from their ODEs."""
    for T in [t + 0.25, t + 3.0, t + 15.0]:
        errs, (mX, P, Y, v) = _aoa_errors(m, t, T)
        assert max(abs(e) for e in errs.values()) < 1e-13
    # the ODE solutions for y and v^{f,c} agree with the covariance integrals
    assert np.allclose(Y, m.Y(t), rtol=1e-9, atol=1e-14)
    assert np.allclose(v, m.vfc(t), rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("drop", ["quanto_f", "ycf", "v_in_fx"])
def test_qgcm_identities_detect_a_missing_term(m, drop):
    """Dropping any one drift term breaks the identity by orders of magnitude more than its tolerance."""
    errs, _ = _aoa_errors(m, 2.5, 10.0, drop)
    assert abs(errs["xZfd"]) > 1e-6


def test_libor_collateral_convexity(m):
    """E^{f|d,S+th}[1+th L^f_S] / (1+th L^{f|f}_{0,S}) = exp(-int Sigma^f_{u,S;th}.Sigma^c_{u,S+th} du).

    The older form with Sigma^c_{u,S;th} (eqs 1807/1813 before correction) misses most of it.
    """
    for S in [0.8, 2.5, 7.0]:
        th = 0.25
        Tp = S + th
        mX, P, Y, v = moments.propagate(m, S)
        ix = moments.indices(m)
        a1, c1 = m.lnZ("fd", S, Tp, Y, v)
        a2, c2 = m.lnZ("ff", S, Tp, Y, v)
        a = np.zeros(ix["n"]); a[:m.nr] = a1 - a2; a[ix["iID"]] = -1; a[ix["iZX"]] = 1
        lhs = eexp(a, mX, P, c1 - c2)
        I = lambda A, B: m.I(A, B, 0, S)
        good = np.exp(-(I(m.Sig("f", Tp), m.Sig("c", Tp)) - I(m.Sig("f", S), m.Sig("c", Tp))))
        old = np.exp(-(I(m.Sig("f", Tp), m.Sig("c", Tp)) - I(m.Sig("f", S), m.Sig("c", Tp))
                       - I(m.Sig("f", Tp), m.Sig("c", S)) + I(m.Sig("f", S), m.Sig("c", S))))
        assert lhs == pytest.approx(good, rel=1e-13)
        assert abs(lhs / old - 1) > 0.5 * abs(good - 1)


# --------------------------------------------------------------------------- joint law (LGCM spec)
@pytest.mark.parametrize("t", [0.8, 7.0])
def test_lgcm_joint_law(m, t):
    """Proposition 'Joint law of the state': mean and covariance from covariance integrals."""
    mO, PO, _, _ = moments.propagate(m, t)
    nr, X = m.nr, m.ix
    J = lambda la, lb: m.covint(la, lb, 0, t)
    mean = np.zeros(nr)
    for i in range(nr):
        blk = m.blocks[i]
        src = {"d": m.idx["d"], "f": m.idx["f"], "c": m.idx["c"] + m.idx["f"]}[blk]
        mean[i] = sum(J((i, "M", t), (j, "G", t)) for j in src) - (J((i, "M", t), (X, "1", 0)) if blk in "fc" else 0)
    SDF, Sd = m.SigDF(t), m.Sig("d", t)
    assert np.abs(mean - mO[:nr]).max() < 1e-15
    assert abs(-0.5 * m.I(SDF, SDF, 0, t) + m.I(SDF, Sd, 0, t) - mO[nr]) < 1e-15
    assert np.abs(PO[:nr, :nr] - m.Y(t)).max() < 1e-15
    covzx = np.array([m.I([(1.0, i, "M", t)], SDF, 0, t) for i in range(nr)])
    assert np.abs(PO[:nr, nr] - covzx).max() < 1e-15
    assert abs(PO[nr, nr] - m.I(SDF, SDF, 0, t)) < 1e-15


# --------------------------------------------------------------------------- transitions (item 4)
GRID = np.round(np.arange(0, 3.0001, 0.25), 10)


def test_rn_transitions_match_moment_odes(m):
    mX, P, ix = transitions.propagate(m, GRID, "RN")
    mO, PO, _, _ = moments.propagate(m, GRID[-1])
    sel = list(range(m.nr)) + [ix["iZX"], ix["iID"], ix["iIQ"]]
    assert np.abs(mX[sel] - mO).max() < 1e-15
    assert np.abs(P[np.ix_(sel, sel)] - PO).max() < 1e-15


def test_martingale_identities_under_rn_tforward_and_jumping(m):
    t = GRID[-1]
    Y, v = m.Y(t), m.vfc(t)
    nr = m.nr
    errs = []
    mX, P, ix = transitions.propagate(m, GRID, "RN")
    n = ix["n"]
    for T in [t + 0.5, t + 10]:
        for kind, extra in [("dd", {}), ("fd", {"iZX": 1}), ("ff", {"iZX": 1, "iIQ": 1})]:
            a_, c_ = m.lnZ(kind, t, T, Y, v)
            a = np.zeros(n); a[:nr] = a_; a[ix["iID"]] = -1
            for k, val in extra.items():
                a[ix[k]] = val
            errs.append(eexp(a, mX, P, c_) - 1)
    Ts = 20.0
    mX, P, ix = transitions.propagate(m, GRID, "Tfwd", Ts)
    aS, cS = m.lnZ("dd", t, Ts, Y, v)
    for T in [t + 0.5, t + 10]:
        a_, c_ = m.lnZ("dd", t, T, Y, v)
        a = np.zeros(n); a[:nr] = a_ - aS
        errs.append(eexp(a, mX, P, c_ - cS) - 1)
    a_, c_ = m.lnZ("fd", t, Ts, Y, v)
    a = np.zeros(n); a[:nr] = a_ - aS; a[ix["iZX"]] = 1
    errs.append(eexp(a, mX, P, c_ - cS) - 1)
    mX, P, ix = transitions.propagate(m, GRID, "jump")
    for T in [t + 0.5, t + 10]:
        for kind, fx in [("dd", 0), ("fd", 1)]:
            a_, c_ = m.lnZ(kind, t, T, Y, v)
            a = np.zeros(n); a[:nr] = a_; a[ix["iL"]] = -1; a[ix["iZX"]] = fx
            errs.append(eexp(a, mX, P, c_) - 1)
    assert max(abs(e) for e in errs) < 1e-13


def test_measure_shifts_are_material(m):
    mR, _, _ = transitions.propagate(m, GRID, "RN")
    mJ, _, _ = transitions.propagate(m, GRID, "jump")
    mT, _, _ = transitions.propagate(m, GRID, "Tfwd", 20.0)
    assert np.abs(mR - mJ).max() > 1e-5
    assert np.abs(mR - mT).max() > 1e-3


# --------------------------------------------------------------------------- Gaussian approximations (item 1)
def test_zhat_is_centred_with_covariance_yhat_under_forward_measure(m):
    """Lemma: zhat_{T0} ~ N(0, yhat_{T0}) under Q^{f|d,T0} (exponential tilt of the RN law)."""
    T0 = 5.0
    mX, P, ix = transitions.propagate(m, np.linspace(0, T0, 21), "RN")
    a = np.zeros(ix["n"]); a[ix["iID"]] = -1; a[ix["iZX"]] = 1
    mt = mX + P @ a
    (f,), (c,) = m.idx["f"], m.idx["c"]
    assert abs(mt[f] + m.vfc(T0)[f]) < 1e-15 and abs(mt[c]) < 1e-15
    assert np.abs(P[np.ix_([f, c], [f, c])] - m.Y(T0)[np.ix_([f, c], [f, c])]).max() < 1e-15


def test_gaussian_approximations_and_convergence_order():
    rel = {}
    for scale in [0.5, 1.0]:
        mm = default_model(scale=scale)
        ap = gaussian_approx.approx(mm)
        ex = gaussian_approx.exact(mm)
        # the exact pricer is a martingale under the annuity measure
        assert ex["Y"]["mean"] == pytest.approx(ap["Y0"], abs=1e-14)
        assert ex["B"]["mean"] == pytest.approx(ap["B0"], abs=1e-14)
        rel[scale] = (np.sqrt(ap["VY"] / ex["Y"]["var"]) - 1, np.sqrt(ap["VB"] / ex["B"]["var"]) - 1)
    # measured: 3.3e-3 (Y) and -6.2e-4 (B) at scale 1, about 4x smaller at scale 0.5
    assert abs(rel[1.0][0]) < 5e-3 and abs(rel[1.0][1]) < 1e-3
    for k in range(2):
        assert 3.0 < rel[1.0][k] / rel[0.5][k] < 5.0


def test_single_swaplet_loadings():
    """n = 1: q^c_B = (1 + theta L^r)/theta G^c_{T0,T1} and q^f_B = B G^f_{T0,T1}."""
    mm = default_model()
    s = gaussian_approx.SwapSetup(n=1)
    ap = gaussian_approx.approx(mm, s)
    (f,), (c,) = mm.idx["f"], mm.idx["c"]
    P = s.Zfd0(s.Tj) / s.Zfd0(s.T0)
    one_Lr = s.kappa_d * P[0] / P[1]
    assert ap["qB"][1] == pytest.approx(one_Lr / s.theta * mm.G(c, s.T0, s.Tj[1]), rel=1e-14)
    assert ap["qB"][0] == pytest.approx(ap["B0"] * mm.G(f, s.T0, s.Tj[1]), rel=1e-14)


# --------------------------------------------------------------------------- consistent families (item 5)
def _exponentials(expr, tau):
    e = sp.expand(sp.powsimp(sp.expand(expr)))
    return {sp.powsimp(term / term.as_independent(tau)[0]) for term in e.as_ordered_terms()}


def test_consistent_exponential_families():
    tau = sp.symbols("tau", positive=True)
    b1, b2, bf, bc = sp.symbols("beta1 beta2 beta_f beta_c", positive=True)
    z1, z2, zc, v, ycc, ycf = sp.symbols("z1 z2 z_c v y_cc y_cf")
    y = sp.Matrix(2, 2, sp.symbols("y11 y12 y21 y22"))
    M = lambda b: sp.exp(-b * tau)
    G = lambda b: (1 - sp.exp(-b * tau)) / b
    native = sum(M(bi) * (zi + sum(y[i, j] * G(bj) for j, bj in enumerate([b1, b2])))
                 for i, (bi, zi) in enumerate([(b1, z1), (b2, z2)]))
    assert _exponentials(native, tau) == {M(b1), M(b2), sp.exp(-2 * b1 * tau), sp.exp(-2 * b2 * tau),
                                          sp.exp(-b1 * tau - b2 * tau)}
    basis = M(bc) * (zc + ycc * G(bc) + ycf * G(bf)) + M(bf) * (v + ycf * G(bc))
    assert _exponentials(basis, tau) == {M(bc), M(bf), sp.exp(-2 * bc * tau), sp.exp(-bc * tau - bf * tau)}
