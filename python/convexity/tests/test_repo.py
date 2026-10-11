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
