"""Checks of section "Bonds, repo and repo convexity" of docs/CollateralDiscounting.tex.

Run from python/ with ``python -m pytest convexity -q``.

The benchmarks are exact (nested quadrature, the Libor collateral convexity, McCloud's Hull-White
closed forms, Gauss-Hermite quadrature) except the Monte Carlo checks (4 standard errors, fixed
seeds) and the frozen-weights coupon approximation, which is asymptotic (measured order).
"""
import math

import numpy as np
import pytest

from convexity import repo
from convexity.market import libor_convexity

HW = dict(theta=0.05, sigma=0.01, kappa=0.4, eps=0.008, rho=-0.5)
CURVES = repo.BondCurves.flat(rbar=0.035, b=-0.002)
DATES = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.5, 10.0), (0.0, 2.0, 2.25, 7.0), (0.5, 1.3, 2.1, 7.5),
         (0.0, 5.0, 6.0, 30.0)]


@pytest.fixture(scope="module")
def hw():
    return repo.RepoModel.hull_white(**HW)


@pytest.fixture(scope="module")
def pw():
    """Two bond factors and one basis factor, piecewise constant, with zero, tiny and negative mean reversions."""
    knots = [0, 1, 3, 7, 15, 40]
    bond = [([0.010, 0.012, 0.009, 0.011, 0.010], [0.05, 0.0, 0.03, -0.01, 0.02]),
            ([0.006, 0.005, 0.007, 0.006, 0.005], [0.8, 0.6, 1e-7, 0.5, 0.7])]
    basis = [([0.004, 0.006, 0.005, 0.004, 0.003], [0.3, 0.5, 0.2, 0.4, 0.3])]
    R = [[1.0, 0.3, -0.4], [0.3, 1.0, 0.2], [-0.4, 0.2, 1.0]]
    return repo.RepoModel(knots, bond, basis, R)


def brute_ln_K(rm, t, s, e, T):
    """Proposition "LGCM repo convexity" by nested adaptive quadrature (lgcm.Model.covint_brute)."""
    m = rm.lgcm

    def I(A, B, a, b):
        return sum(ca * cb * m.covint_brute((i, ta, Ta), (j, tb, Tb), a, b)
                   for ca, i, ta, Ta in A for cb, j, tb, Tb in B)
    sc = rm.basis_vol(e)
    return -(I(rm.bond_vol(e) + rm.bond_vol(s, -1.0), sc, t, s) + I(rm.bond_vol(e) + rm.bond_vol(T, -1.0), sc, s, e))


# --------------------------------------------------------------------------- zero-coupon repo convexity
@pytest.mark.parametrize("t,s,e,T", DATES)
def test_repo_convexity_matches_quadrature(pw, t, s, e, T):
    assert repo.repo_convexity(pw, t, s, e, T) == pytest.approx(brute_ln_K(pw, t, s, e, T), rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("a,b", [(1.0, 1.5), (2.0, 3.0), (5.0, 10.0)])
def test_repo_to_maturity_is_libor_convexity(pw, a, b):
    """T = e: the repo convexity is the Libor collateral convexity (this fixes the sign of Sigma^c)."""
    assert repo.repo_convexity(pw, 0.0, a, b, b) == pytest.approx(math.log(libor_convexity(pw.lgcm, 0.0, a, b)), rel=1e-12)


def test_boundary_conditions(pw):
    assert abs(repo.maturity_adjustment(pw, 0.0, 3.0, 3.0)) < 1e-18               # M^{ee} = 0 (up to rounding)
    assert repo.maturity_adjustment(pw, 1.0, 1.0, 7.0) == 0.0                     # M^{tT}_t = 0
    assert abs(repo.forwardness_adjustment(pw, 1.0, 1.0, 2.0, 7.0)) < 1e-18       # F^{eT}_{tt} = 0
    C = repo.repo_convexity(pw, 0.5, 1.3, 2.1, 7.5)
    M = repo.maturity_adjustment(pw, 0.5, 2.1, 7.5) - repo.maturity_adjustment(pw, 0.5, 1.3, 7.5)
    assert C == pytest.approx(M + repo.forwardness_adjustment(pw, 0.5, 1.3, 2.1, 7.5), rel=1e-13)


def test_no_basis_correlation_gives_bond_forward():
    rm = repo.RepoModel.hull_white(**{**HW, "rho": 0.0})
    s, e = 1.3, 2.1
    fwd = (CURVES.Zbond(s) / CURVES.Zbond(e) - 1.0) / repo.accrual(s, e)
    assert repo.repo_rate(rm, CURVES, s, e, 7.5) == pytest.approx(fwd, rel=1e-15)


def test_repo_discount_factor_to_maturity_is_bond(pw):
    assert repo.repo_discount_factor(pw, CURVES, 4.0, 4.0) == pytest.approx(float(CURVES.Zbond(4.0)), rel=1e-15)


def test_day_count_only_quotes_the_rate(hw):
    s, e, T = 1.3, 2.1, 7.5
    assert repo.accrual(s, e, "ACT/360") == pytest.approx((e - s) * 365.0 / 360.0, rel=1e-15)
    assert repo.accrual(s, e, "ACT/365") == pytest.approx(e - s, rel=1e-15)
    r360 = repo.repo_rate(hw, CURVES, s, e, T, "ACT/360")
    r365 = repo.repo_rate(hw, CURVES, s, e, T, "ACT/365")
    assert r360 * 365.0 / 360.0 == pytest.approx(r365, rel=1e-14)
    with pytest.raises(ValueError, match="day count"):
        repo.accrual(s, e, "30/360")


@pytest.mark.parametrize("t,s,e,T", [(0.0, 2.0, 1.0, 5.0), (0.0, 1.0, 2.0, 1.5), (1.5, 1.0, 2.0, 5.0),
                                     (0.0, 1.0, 1.0, 5.0), (-1.0, 0.0, 1.0, 2.0)])
def test_dates_are_validated(hw, t, s, e, T):
    with pytest.raises(ValueError, match="t <= s < e <= T"):
        repo.repo_convexity(hw, t, s, e, T)


def test_model_inputs_are_validated():
    with pytest.raises(ValueError, match="R must be"):
        repo.RepoModel([0, 10], [([0.01], [0.1])], [([0.01], [0.1])], [[1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        repo.RepoModel([0, 10], [([0.01], [0.1])], [([0.01], [0.1])], [[1.0, 1.2], [1.2, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        repo.RepoModel([0, 10], [], [([0.01], [0.1])], np.eye(1))


# --------------------------------------------------------------------------- McCloud's Hull-White closed forms
P = (HW["rho"], HW["sigma"], HW["eps"], HW["theta"], HW["kappa"])


@pytest.mark.parametrize("t,s,e,T", DATES)
def test_hull_white_matches_mccloud(hw, t, s, e, T):
    """Proposition "Hull-White repo convexity": the LGCM formula is McCloud's (30), (33), (34)."""
    assert repo.repo_convexity(hw, t, s, e, T) == pytest.approx(repo.hw_convexity(*P, t, s, e, T), rel=1e-11, abs=1e-18)
    assert repo.maturity_adjustment(hw, t, e, T) == pytest.approx(repo.hw_M(*P, t, e, T), rel=1e-11, abs=1e-18)
    if s > t:
        assert repo.forwardness_adjustment(hw, t, s, e, T) == pytest.approx(repo.hw_F(*P, t, s, e, T), rel=1e-11)


def test_hull_white_beyond_horizon(hw):
    """Model.piece clamps to the last piece: constant parameters extend past the horizon (200)."""
    assert repo.repo_convexity(hw, 0.0, 2.0, 3.0, 300.0) == pytest.approx(repo.hw_convexity(*P, 0.0, 2.0, 3.0, 300.0), rel=1e-11)


@pytest.mark.parametrize("theta,kappa", [(1e-3, 1e-3), (1e-3, 0.5), (0.3, 1e-3), (-0.02, 0.4)])
def test_hull_white_small_and_negative_mean_reversion(theta, kappa):
    rm = repo.RepoModel.hull_white(theta, 0.01, kappa, 0.008, -0.5)
    for t, s, e, T in DATES[1:]:
        assert repo.repo_convexity(rm, t, s, e, T) == pytest.approx(
            repo.hw_convexity(-0.5, 0.01, 0.008, theta, kappa, t, s, e, T), rel=1e-7)


def test_zero_mean_reversion_limit():
    """theta = kappa = 0: ln K = -rho sigma eps [(e-s)((e-t)^2 - (e-s)^2)/2 - (T-e)(e-s)^2/2]."""
    rho, sig, eps = -0.5, 0.01, 0.008
    rm = repo.RepoModel.hull_white(0.0, sig, 0.0, eps, rho)
    for t, s, e, T in DATES:
        poly = -rho * sig * eps * ((e - s) * ((e - t) ** 2 - (e - s) ** 2) / 2 - (T - e) * (e - s) ** 2 / 2)
        assert repo.repo_convexity(rm, t, s, e, T) == pytest.approx(poly, rel=1e-12, abs=1e-18)
    with pytest.raises(ValueError, match="repo_convexity"):
        repo.hw_B(1.0, 1.0, 1.0, 0.0, 0.4)


def test_hull_white_maturity_interpolation(hw):
    """McCloud (41): hat p^{eT} = (pbar^e)^w (hat p^{e,inf})^{1-w}, w = exp(-theta (T-e)): exact in Hull-White."""
    for e, mu in [(1.0, 2.0), (2.0, 10.0), (5.0, 25.0)]:
        w = repo.hw_maturity_weight(HW["theta"], mu)
        p_inf = repo.repo_discount_factor(hw, CURVES, e, e + 1e3)
        p_eT = repo.repo_discount_factor(hw, CURVES, e, e + mu)
        assert p_eT == pytest.approx(float(CURVES.Zbond(e)) ** w * p_inf ** (1 - w), rel=1e-13)


def test_hull_white_forwardness_interpolation(hw):
    """McCloud (43): F(tau) = (1 - exp(-(theta+kappa) tau)) F(inf) at fixed e-s, T-e: exact in Hull-White."""
    d, mu = 1.0, 5.0
    F_inf = repo.forwardness_adjustment(hw, 0.0, 150.0, 150.0 + d, 150.0 + d + mu)
    for tau in [0.5, 1.0, 3.0, 6.0]:
        w = repo.hw_forwardness_weight(HW["theta"], HW["kappa"], tau)
        assert repo.forwardness_adjustment(hw, 0.0, tau, tau + d, tau + d + mu) == pytest.approx((1 - w) * F_inf, rel=1e-12)


def test_hull_white_forward_spread_and_extrapolation(hw):
    """McCloud (38): hat f^e - bar f^e = d/de M^{e,inf}_0; (39) extrapolates it from E."""
    h = 1e-5
    for e in [0.5, 2.0, 7.0]:
        dM = (repo.maturity_adjustment(hw, 0.0, e + h, 1e3) - repo.maturity_adjustment(hw, 0.0, e - h, 1e3)) / (2 * h)
        assert dM == pytest.approx(repo.hw_forward_spread(*P, 0.0, e), rel=1e-7)
    E = 2.0
    sE = repo.hw_forward_spread(*P, 0.0, E)
    for e in [2.0, 5.0, 20.0]:
        assert repo.hw_extrapolate(sE, 0.0, E, e, HW["theta"], HW["kappa"]) == pytest.approx(
            repo.hw_forward_spread(*P, 0.0, e), rel=1e-13)


def test_interpolation_fails_beyond_hull_white(pw):
    """Remark "Beyond Hull-White": no single geometric weight in the two-factor piecewise model.
    Measured: implied theta 0.0232 (mu = 2) vs 0.0296 (mu = 20); F(tau)/F(30) = 0.18, 0.31, 1.02, 1.53, 1.38
    at tau = 0.5, 1, 3, 6, 10 (not monotone, above 1)."""
    e, Minf = 2.0, repo.maturity_adjustment(pw, 0.0, 2.0, 60.0)
    th = [-math.log(1 - repo.maturity_adjustment(pw, 0.0, e, e + mu) / Minf) / mu for mu in (2.0, 20.0)]
    assert th[1] / th[0] - 1 > 0.2
    Finf = repo.forwardness_adjustment(pw, 0.0, 30.0, 31.0, 36.0)
    ratios = [repo.forwardness_adjustment(pw, 0.0, tau, tau + 1, tau + 6) / Finf for tau in (0.5, 1, 3, 6, 10)]
    assert max(ratios) > 1.2 and np.any(np.diff(ratios) < 0)


# --------------------------------------------------------------------------- coupon bonds
BOND = repo.CouponBond.fixed(10.0, 0.05, 2)      # 10Y semi-annual 5%: coupons at 1.5 and 2.0 fall in (1.3, 2.1]


def test_coupon_bond_construction():
    assert BOND.times[0] == pytest.approx(0.5) and BOND.times[-1] == pytest.approx(10.0) and len(BOND.times) == 20
    assert BOND.cashflows[-1] == pytest.approx(1.025) and BOND.cashflows[0] == pytest.approx(0.025)
    with pytest.raises(ValueError, match="same length"):
        repo.CouponBond((1.0, 2.0), (1.0,))
    with pytest.raises(ValueError, match="increasing"):
        repo.CouponBond((2.0, 1.0), (0.05, 1.05))
    with pytest.raises(ValueError, match="after the repo end"):
        repo.coupon_repo_rate(repo.RepoModel.hull_white(**HW), CURVES, repo.CouponBond((1.0, 2.0), (0.05, 1.05)), 1.3, 2.1)


def test_ois_forward_law_is_normalised(pw):
    """E^{RN}[D_{0,s}] = p_{0,s}: the tilt a = e_zx - e_Id has a.m + a'Pa/2 = 0."""
    from convexity import transitions
    m = pw.lgcm
    mX, P, ix = transitions.propagate(m, [0.0, 1.0, 2.5], "RN")
    a = np.zeros(ix["n"]); a[ix["iZX"]], a[ix["iID"]] = 1.0, -1.0
    assert abs(a @ mX + 0.5 * a @ P @ a) < 1e-15


@pytest.mark.parametrize("s,e,T", [(1.3, 2.1, 7.5), (0.0, 0.5, 3.0), (2.0, 3.0, 3.0)])
def test_single_cashflow_is_zero_coupon(hw, s, e, T):
    """A one-cashflow bond: both the exact and the frozen-weights rates are Proposition "Repo rate"."""
    b = repo.CouponBond((T,), (1.0,))
    zc = repo.repo_rate(hw, CURVES, s, e, T)
    assert repo.coupon_repo_rate_exact(hw, CURVES, b, s, e) == pytest.approx(zc, rel=1e-13)
    assert repo.coupon_repo_rate(hw, CURVES, b, s, e) == pytest.approx(zc, rel=1e-13)


def test_zero_basis_gives_bond_forward():
    """b = 0 (eps = 0, flat zero basis): Y_s = 1, so 1 + delta f = pbar_{0,s}/pbar_{0,e} for any coupons."""
    rm = repo.RepoModel.hull_white(0.05, 0.01, 0.4, 0.0, -0.5)
    cv = repo.BondCurves.flat(rbar=0.035, b=0.0)
    fwd = (cv.Zbond(1.3) / cv.Zbond(2.1) - 1.0) / repo.accrual(1.3, 2.1)
    assert repo.coupon_repo_rate_exact(rm, cv, BOND, 1.3, 2.1) == pytest.approx(fwd, rel=1e-13)


def test_gauss_hermite_converged(hw):
    a = repo.coupon_repo_rate_exact(hw, CURVES, BOND, 1.3, 2.1, n_gh=40)
    b = repo.coupon_repo_rate_exact(hw, CURVES, BOND, 1.3, 2.1, n_gh=60)
    assert abs(a - b) < 1e-14


def test_frozen_weights_error_order():
    """Asymptotic, not proved. Measured: errors 3.73e-4, 2.31e-5, 1.44e-6 at vol scale 4, 2, 1
    (ratios 16.2, 16.0): O(lambda^4) absolute in the rate."""
    errs = []
    for sc in [4.0, 2.0, 1.0]:
        rm = repo.RepoModel.hull_white(0.05, 0.01 * sc, 0.4, 0.008 * sc, -0.5)
        errs.append(repo.coupon_repo_rate(rm, CURVES, BOND, 1.3, 2.1) - repo.coupon_repo_rate_exact(rm, CURVES, BOND, 1.3, 2.1))
    r = [errs[0] / errs[1], errs[1] / errs[2]]
    assert all(14.0 < x < 18.0 for x in r), (errs, r)


# --------------------------------------------------------------------------- Monte Carlo
MC_HW = repo.RepoModel.hull_white(0.05, 0.03, 0.4, 0.024, -0.5)   # vol scale 3: the convexity is ~1.2bp


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_mc_zero_coupon_repo_rate(seed):
    """Measured (2^18 paths): z = -1.39, +0.57, +0.45, -0.23 against Proposition "Repo rate";
    the bond forward (K = 1) is rejected at -9.81, -7.75, -7.94, -8.65 standard errors."""
    s, e, T = 1.3, 2.1, 7.5
    r, se = repo.mc_repo_rate(MC_HW, CURVES, T, s, e, n_paths=2**18, seed=seed)
    assert abs(r - repo.repo_rate(MC_HW, CURVES, s, e, T)) < 4 * se
    no_conv = (CURVES.Zbond(s) / CURVES.Zbond(e) - 1.0) / repo.accrual(s, e)
    assert abs(r - no_conv) > 4 * se


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_mc_coupon_repo_rate(seed):
    """Measured (2^18 paths): z = -0.88, +1.42, +0.72, -0.57 against the exact quadrature; the frozen-weights
    approximation (-1.17bp off at this vol scale) is rejected at -8.47, -6.09, -6.82, -8.14 standard errors."""
    r, se = repo.mc_repo_rate(MC_HW, CURVES, BOND, 1.3, 2.1, n_paths=2**18, seed=seed)
    assert abs(r - repo.coupon_repo_rate_exact(MC_HW, CURVES, BOND, 1.3, 2.1)) < 4 * se
    assert abs(r - repo.coupon_repo_rate(MC_HW, CURVES, BOND, 1.3, 2.1)) > 4 * se


def test_coupon_on_window_boundaries_exact_vs_mc():
    """Coupons exactly at s (to the seller) and at e (passed through, not returned).
    Measured: z = -3.03 at seed 5 (a tail draw: over 40 seeds at 2^16 paths the z have mean 0.13, std 1.06)."""
    b = repo.CouponBond((1.0, 2.0, 3.0, 4.0), (0.06, 0.06, 0.06, 1.06))
    r, se = repo.mc_repo_rate(MC_HW, CURVES, b, 1.0, 2.0, n_paths=2**18, seed=5)
    assert abs(r - repo.coupon_repo_rate_exact(MC_HW, CURVES, b, 1.0, 2.0)) < 4 * se


# --------------------------------------------------------------------------- overnight repo (bond-collateralised derivatives)
def test_spot_overnight_repo_is_bond_rate(hw):
    """Proposition "Overnight repo": ln K_0(0, h, T) = O(h^2), so the spot overnight repo rate is rbar_0."""
    k = [repo.repo_convexity(hw, 0.0, 0.0, h, 10.0) for h in (1 / 365, 0.5 / 365)]
    assert k[0] / k[1] == pytest.approx(4.0, rel=1e-2)
    assert abs(k[0]) * 365 < 1e-6


def test_forward_overnight_repo_limit(hw):
    """ln K_0(s, s+h, T)/h -> -rho sigma eps (E(theta, s) - E(theta + kappa, s))/kappa != 0 (Hull-White)."""
    from convexity.lgcm import E
    s = 3.0
    lim = -HW["rho"] * HW["sigma"] * HW["eps"] * (E(HW["theta"], s) - E(HW["theta"] + HW["kappa"], s)) / HW["kappa"]
    for h, tol in [(1 / 365, 2e-2), (0.1 / 365, 2e-3)]:
        assert repo.repo_convexity(hw, 0.0, s, s + h, 10.0) / h == pytest.approx(lim, rel=tol)
    assert lim > 1e-4
