"""Bonds, repo and repo convexity (McCloud, "Repo convexity", arXiv:1905.03316, 2019).

Numerical companion of section "Bonds, repo and repo convexity" of docs/CollateralDiscounting.tex.

Two curves under one risk-neutral measure: the bond curve pbar (discount rate rbar) and the
derivative (OIS) curve p (rate r = rbar + b, b the discounting basis). They are the LGCM blocks
f (bond curve, Z^{f|f} = pbar) and c (collateral basis f->d = b, Z^{f|d} = p), with no d factor
and zero FX vol. Only RepoModel and BondCurves know this mapping.

Repo over [s, e] on the bond maturing at T, agreed at t <= s < e <= T: one unit of cash at s for
1/pbar^T_s bonds, returned at e against 1 + delta f, discounted at r. Then
    1 + delta f = (pbar_{t,s} / pbar_{t,e}) K^{repo}_t(s, e, T)      (Proposition "Repo rate")
Model time is in ACT/365 years; the accrual delta = accrual(s, e, day_count) only quotes the rate.

Main entry points
-----------------
RepoModel, BondCurves                 model (bond and basis factors) and initial curves
repo_convexity                        ln K^{repo}, Proposition "LGCM repo convexity"
maturity_adjustment, forwardness_adjustment   McCloud's M and F (Corollary "Maturity and forwardness")
repo_rate, repo_discount_factor       quoted repo rate, repo discount factor hat p^{eT}_0
hw_*                                  McCloud's Hull-White closed forms, written independently of lgcm
CouponBond, coupon_repo_rate, coupon_repo_rate_exact   repo on a coupon bond: frozen weights, Gauss-Hermite
mc_repo_rate                          Monte Carlo benchmark (exact LGCM scheme)
"""
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .lgcm import Model

ACCRUAL = {"ACT/360": 365.0 / 360.0, "ACT/365": 1.0}


def accrual(s, e, day_count="ACT/360"):
    """Accrual fraction of [s, e] (model time in ACT/365 years)."""
    if day_count not in ACCRUAL:
        raise ValueError(f"day count {day_count!r}: use one of {sorted(ACCRUAL)}")
    return (e - s) * ACCRUAL[day_count]


def _check_dates(t, s, e, T):
    if not (0.0 <= t <= s < e <= T):
        raise ValueError(f"need 0 <= t <= s < e <= T, got t={t}, s={s}, e={e}, T={T}")


# ---------------------------------------------------------------------- model and curves
class RepoModel:
    """LGCM with bond factors (block f) and basis factors (block c), no d factor, zero FX vol."""

    def __init__(self, knots, bond_factors, basis_factors, R):
        """bond_factors, basis_factors: lists of (sig[k], beta[k]) on the pieces of `knots`;
        R: correlation of the bond drivers then the basis drivers."""
        nf, nc = len(bond_factors), len(basis_factors)
        if nf == 0 or nc == 0:
            raise ValueError("need at least one bond factor and one basis factor")
        R = np.asarray(R, float)
        if R.shape != (nf + nc, nf + nc):
            raise ValueError(f"R must be {nf + nc}x{nf + nc}, got {R.shape}")
        RR = np.eye(nf + nc + 1)
        RR[:-1, :-1] = R
        if np.any(np.linalg.eigvalsh(RR) <= 0):
            raise ValueError("correlation matrix is not positive definite")
        factors = [("f", s, b) for s, b in bond_factors] + [("c", s, b) for s, b in basis_factors]
        self.lgcm = Model(knots, factors, RR, np.zeros(len(knots) - 1))

    @classmethod
    def hull_white(cls, theta, sigma, kappa, eps, rho, horizon=200.0):
        """McCloud (27): d rbar = theta(.) dt + sigma dx, d b = kappa(.) dt + eps dy, dx dy = rho dt."""
        return cls([0.0, horizon], [([sigma], [theta])], [([eps], [kappa])], [[1.0, rho], [rho, 1.0]])

    def bond_vol(self, T, coef=1.0):
        """Loading of the bond curve vol Sigma-bar_{u,T}."""
        return self.lgcm.Sig("f", T, coef)

    def basis_vol(self, T, coef=1.0):
        """Loading of the basis vol Sigma^b_{u,T} = Sigma^p_{u,T} - Sigma-bar_{u,T} (block c)."""
        return self.lgcm.Sig("c", T, coef)


@dataclass
class BondCurves:
    """Initial bond curve pbar_{0,T} = exp(-int_rbar0(T)) and OIS curve p_{0,T} = pbar_{0,T} exp(-int_b0(T)).

    Duck-type compatible with market.InitialCurves for the Monte Carlo: "ff" = bond, "fd" = OIS,
    "qz" = OIS / bond, "dd" = OIS (unused: no d factor), x0 = 1."""
    int_rbar0: Callable
    int_b0: Callable
    x0: float = 1.0

    @classmethod
    def flat(cls, rbar=0.035, b=-0.002):
        return cls(lambda T: rbar * np.asarray(T, float), lambda T: b * np.asarray(T, float))

    def Zbond(self, T):
        return np.exp(-self.int_rbar0(T))

    def Zois(self, T):
        return np.exp(-self.int_rbar0(T) - self.int_b0(T))

    def Z0(self, kind, T):
        if kind == "ff":
            return self.Zbond(T)
        if kind in ("fd", "dd"):
            return self.Zois(T)
        if kind == "qz":
            return np.exp(-self.int_b0(T))
        raise ValueError(kind)

    def fx_forward0(self, T):
        return self.x0 * self.Z0("fd", T) / self.Z0("dd", T)


# ---------------------------------------------------------------------- repo convexity (zero-coupon)
def repo_convexity(rm, t, s, e, T):
    """ln K^{repo}_t(s, e, T) = -int_t^s (Sb_{u,e} - Sb_{u,s}).S^b_{u,e} du - int_s^e (Sb_{u,e} - Sb_{u,T}).S^b_{u,e} du."""
    _check_dates(t, s, e, T)
    m, sb = rm.lgcm, rm.basis_vol(e)
    fwd = m.I(rm.bond_vol(e) + rm.bond_vol(s, -1.0), sb, t, s)
    win = m.I(rm.bond_vol(e) + rm.bond_vol(T, -1.0), sb, s, e)
    return -(fwd + win)


def maturity_adjustment(rm, t, e, T):
    """M^{eT}_t = C^{eT}_{tt} (McCloud (17)); M^{tT}_t = 0."""
    return 0.0 if e == t else repo_convexity(rm, t, t, e, T)


def forwardness_adjustment(rm, t, s, e, T):
    """F^{eT}_{ts} = C^{eT}_{ts} - (M^{eT}_t - M^{sT}_t) (McCloud (17))."""
    return repo_convexity(rm, t, s, e, T) - (maturity_adjustment(rm, t, e, T) - maturity_adjustment(rm, t, s, T))


def repo_rate(rm, curves, s, e, T, day_count="ACT/360"):
    """Repo rate f^{eT}_{0s} agreed at 0: 1 + delta f = (pbar_{0,s}/pbar_{0,e}) K^{repo}_0(s, e, T)."""
    gross = curves.Zbond(s) / curves.Zbond(e) * math.exp(repo_convexity(rm, 0.0, s, e, T))
    return float((gross - 1.0) / accrual(s, e, day_count))


def repo_discount_factor(rm, curves, e, T):
    """hat p^{eT}_0 = pbar_{0,e} exp(-M^{eT}_0) (McCloud (20) with L = 0)."""
    return float(curves.Zbond(e) * math.exp(-maturity_adjustment(rm, 0.0, e, T)))
