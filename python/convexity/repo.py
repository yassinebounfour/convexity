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

from . import transitions
from .lgcm import Model
from .montecarlo import Simulator, cv_mean_and_stderr
from .schemes import ExactScheme

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


# ---------------------------------------------------------------------- McCloud's Hull-White closed forms
def _ex(x):
    """1 - e^{-x}."""
    return -math.expm1(-x)


def _check_mr(theta, kappa):
    if theta * kappa * (theta + kappa) == 0:
        raise ValueError("McCloud's closed forms need theta, kappa, theta + kappa != 0: use repo_convexity "
                         "(stable at zero mean reversion)")


def hw_B(tau, delta, mu, theta, kappa):
    """McCloud (33): C^{eT}_{ts} = rho sigma eps B[s-t, e-s, T-e; theta, kappa]."""
    _check_mr(theta, kappa)
    th, ka = theta, kappa
    a = _ex(th * mu) * (_ex(th * (tau + delta)) / th - _ex((th + ka) * (tau + delta)) / (th + ka))
    b = _ex(th * (delta + mu)) * (_ex(th * tau) / th - math.exp(-ka * delta) * _ex((th + ka) * tau) / (th + ka))
    return (a - b) / (th * ka)


def hw_convexity(rho, sigma, eps, theta, kappa, t, s, e, T):
    _check_dates(t, s, e, T)
    return rho * sigma * eps * hw_B(s - t, e - s, T - e, theta, kappa)


def hw_M(rho, sigma, eps, theta, kappa, t, e, T):
    """McCloud (34), maturity adjustment."""
    _check_mr(theta, kappa)
    th, ka = theta, kappa
    return rho * sigma * eps / (th * ka) * _ex(th * (T - e)) * (_ex(th * (e - t)) / th - _ex((th + ka) * (e - t)) / (th + ka))


def hw_F(rho, sigma, eps, theta, kappa, t, s, e, T):
    """McCloud (34), forwardness adjustment."""
    _check_mr(theta, kappa)
    th, ka = theta, kappa
    return -rho * sigma * eps / (th * ka * (th + ka)) * _ex(th * (T - s)) * _ex(ka * (e - s)) * _ex((th + ka) * (s - t))


def hw_forward_spread(rho, sigma, eps, theta, kappa, t, e):
    """McCloud (38): hat f^e_t - bar f^e_t for the infinite-maturity repo curve."""
    _check_mr(theta, kappa)
    return rho * sigma * eps / (theta * kappa) * math.exp(-theta * (e - t)) * _ex(kappa * (e - t))


def hw_extrapolate(spread_E, t, E, e, theta, kappa):
    """McCloud (39): hat f^e - bar f^e from the spread observed at E (e >= E)."""
    return spread_E * math.exp(-theta * (e - E)) * _ex(kappa * (e - t)) / _ex(kappa * (E - t))


def hw_maturity_weight(theta, mu):
    """McCloud (41): hat p^{eT} = (bar p^e)^w (hat p^{e,inf})^{1-w}, w = exp(-theta (T - e))."""
    return math.exp(-theta * mu)


def hw_forwardness_weight(theta, kappa, tau):
    """McCloud (43): weight exp(-(theta + kappa)(s - t)) of the spot-starting gross rate."""
    return math.exp(-(theta + kappa) * tau)


# ---------------------------------------------------------------------- coupon bonds
@dataclass(frozen=True)
class CouponBond:
    """Fixed cashflows c_k at T_k (the last one includes the principal)."""
    times: tuple
    cashflows: tuple

    def __post_init__(self):
        T, c = np.asarray(self.times, float), np.asarray(self.cashflows, float)
        if T.ndim != 1 or T.size == 0 or T.shape != c.shape:
            raise ValueError("times and cashflows must be non-empty and of the same length")
        if np.any(np.diff(T) <= 0) or T[0] <= 0 or np.any(c <= 0):
            raise ValueError("times must be increasing and positive, cashflows positive")

    @classmethod
    def fixed(cls, maturity, coupon, freq=1, notional=1.0):
        n = int(round(maturity * freq))
        times = tuple(t for t in (maturity - k / freq for k in range(n - 1, -1, -1)) if t > 1e-12)
        flows = tuple(notional * coupon / freq + (notional if i == len(times) - 1 else 0.0) for i in range(len(times)))
        return cls(times, flows)

    def price0(self, curves):
        return float(sum(c * curves.Zbond(T) for T, c in zip(self.times, self.cashflows)))


def _check_bond(bond, s, e):
    """The bond must mature at or after e (a last flow at e is redeemed and passed through)."""
    if max(bond.times) < e:
        raise ValueError(f"the bond must mature at or after the repo end e = {e}, got maturity {max(bond.times)}")
    _check_dates(0.0, s, e, max(bond.times))


def _ois_forward_law(rm, s):
    """Mean and covariance of the rate state z_s under the OIS s-forward measure Q^{p,s}.

    The deflator of a payment at s is p_{0,s} exp(z^x_s - int r^d) (Simulator.deflator, ccy f under d
    collateral), so Q^{p,s} tilts the risk-neutral law of the state by a = e_{zx} - e_{Id}."""
    m = rm.lgcm
    if s <= 0:
        return np.zeros(m.nr), np.zeros((m.nr, m.nr))
    grid = np.unique(np.r_[0.0, [k for k in m.knots if 0 < k < s], s])
    mX, P, ix = transitions.propagate(m, grid, "RN")
    a = np.zeros(ix["n"])
    a[ix["iZX"]], a[ix["iID"]] = 1.0, -1.0
    return (mX + P @ a)[: m.nr], P[: m.nr, : m.nr]


def _coupon_terms(rm, curves, bond, s, e):
    """Y(z) = sum_k num_k(z) / sum_j den_j(z) with PV_0 = p_{0,s} E^{p,s}[Y(z_s)] = p_{0,e} (1 + delta f).

    Proposition "Coupon repo rate": conditional on F_s, the bonds bought at s are worth P^ex_e at e
    (flows after e), whose OIS-discounted value at s is sum_k c_k p^e_s (pbar^{T_k}_s / pbar^e_s) K^{repo}_s(s, e, T_k);
    coupons in (s, e] are passed through at T_k, worth c_k p^{T_k}_s; a coupon at s belongs to the seller.
    Each term is (coef, alpha, const) for coef exp(alpha.z + const)."""
    m = rm.lgcm
    Y, v = m.Y(s), m.vfc(s)

    def ln(kind, T):        # ln Z^{kind}_{s,T} = a.z + c
        a, c = m.lnZ(kind, s, T, Y, v)
        Z0 = curves.Zbond if kind == "ff" else curves.Zois
        return a, c + math.log(Z0(T) / Z0(s))

    num, den = [], []
    a_pe, c_pe = ln("fd", e)
    a_be, c_be = ln("ff", e)
    for T, c in zip(bond.times, bond.cashflows):
        if T <= s:
            continue
        a_b, c_b = ln("ff", T)
        den.append((c, a_b, c_b))
        if T > e:
            lk = repo_convexity(rm, s, s, e, T)
            num.append((c, a_pe + a_b - a_be, c_pe + c_b - c_be + lk))
        else:
            a_p, c_p = ln("fd", T)
            num.append((c, a_p, c_p))
    return num, den


def _gross(curves, s, e, EY):
    return float(curves.Zois(s) * EY / curves.Zois(e))


def coupon_repo_rate(rm, curves, bond, s, e, day_count="ACT/360"):
    """Frozen-weights approximation (asymptotic, Remark "Frozen weights"): the basket sum_j den_j is replaced
    by the lognormal with the same mean and the loading sum_j w_j beta_j, w_j its weights at the mean, and the
    same approximation of E[den/den] = 1 is used as a control: E[Y] ~ 1 + FW(num) - FW(den). The control
    makes it exact when num = den (b = 0) and for a single cash flow."""
    _check_bond(bond, s, e)
    num, den = _coupon_terms(rm, curves, bond, s, e)
    mu, P = _ois_forward_law(rm, s)
    dm = np.array([c * math.exp(b @ mu + d + 0.5 * b @ P @ b) for c, b, d in den])
    Dbar = dm.sum()
    bbar = sum(w * b for w, (_, b, _) in zip(dm / Dbar, den))

    def fw(terms):          # frozen-weights E[sum_k terms_k / den]
        return sum(c * math.exp(a @ mu + d + 0.5 * bbar @ P @ bbar + 0.5 * (a - bbar) @ P @ (a - bbar))
                   for c, a, d in terms) / Dbar
    EY = 1.0 + fw(num) - fw(den)
    return (_gross(curves, s, e, EY) - 1.0) / accrual(s, e, day_count)


def coupon_repo_rate_exact(rm, curves, bond, s, e, n_gh=40, day_count="ACT/360"):
    """Exact up to Gauss-Hermite quadrature (tensor grid of n_gh^nr nodes) in the rate state z_s under Q^{p,s}."""
    _check_bond(bond, s, e)
    num, den = _coupon_terms(rm, curves, bond, s, e)
    mu, P = _ois_forward_law(rm, s)
    x, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    nr = len(mu)
    X = np.stack(np.meshgrid(*([x] * nr), indexing="ij"), -1).reshape(-1, nr)
    W = np.prod(np.stack(np.meshgrid(*([w] * nr), indexing="ij"), -1).reshape(-1, nr), axis=1)
    lam, U = np.linalg.eigh(P)
    Z = mu + X @ (U * np.sqrt(np.clip(lam, 0, None))).T
    N = sum(c * np.exp(Z @ a + d) for c, a, d in num)
    D = sum(c * np.exp(Z @ b + d) for c, b, d in den)
    return (_gross(curves, s, e, float(W @ (N / D))) - 1.0) / accrual(s, e, day_count)


# ---------------------------------------------------------------------- Monte Carlo
def mc_repo_rate(rm, curves, bond, s, e, n_paths=2**16, seed=7, day_count="ACT/360"):
    """Repo rate by Monte Carlo (exact LGCM scheme, antithetic). `bond` is a CouponBond or a maturity T.

    Control variates with exact means: E[D_{0,e}] = p_{0,e}, and the same repo discounted at rbar
    (the bond-collateralised repo) is worth pbar_{0,s}. Returns (rate, stderr)."""
    if not isinstance(bond, CouponBond):
        bond = CouponBond((float(bond),), (1.0,))
    _check_bond(bond, s, e)
    sim = Simulator(rm.lgcm, scheme=ExactScheme())
    flows = list(zip(bond.times, bond.cashflows))
    paths = sim.simulate_dates([s, e] + [T for T, _ in flows if s < T <= e], n_paths, seed=seed)
    rs, re_ = paths(s), paths(e)
    Pbar_s = sum(c * sim.discount(rs, curves, "ff", T) for T, c in flows if T > s)
    Pex_e = sum(c * sim.discount(re_, curves, "ff", T) for T, c in flows if T > e)
    De, Dbe = sim.deflator(re_, curves, "f", "d"), sim.deflator(re_, curves, "f", "f")
    x, xbar = De * Pex_e / Pbar_s, Dbe * Pex_e / Pbar_s
    for T, c in flows:
        if s < T <= e:
            rT = paths(T)
            x = x + sim.deflator(rT, curves, "f", "d") * c / Pbar_s
            xbar = xbar + sim.deflator(rT, curves, "f", "f") * c / Pbar_s
    pe = float(curves.Zois(e))
    g, se = cv_mean_and_stderr(x / pe, [De / pe, xbar / pe], [1.0, float(curves.Zbond(s)) / pe])
    d = accrual(s, e, day_count)
    return (g - 1.0) / d, se / d
