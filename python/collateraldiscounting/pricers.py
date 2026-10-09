"""Pricers: the model view of the calibration instruments.

Pricer               curves, forward rates and the interface for option quotes
DeterministicPricer  deterministic collateral basis (DCB) model: Libor collateral convexity K = 1,
                     no volatility (option quotes raise NotImplementedError)
LGCMPricer           LGCM closed forms: exact one-factor swaptions, Gaussian multi-factor variance, CB and FX vols
"""
from math import sqrt, pi

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from . import gaussian_approx as ga
from .curves import DiscountCurveKey
from .market import libor_convexity, fx_total_variance


class Pricer:
    def __init__(self, curves):
        self.curves = curves

    @property
    def setup(self):
        return self.curves.setup

    def libor_convexity(self, a, b):
        """K = (1 + theta L^{f|d}_{0,a}) / (1 + theta L^{f|f}_{0,a}) for the foreign index on [a, b]."""
        raise NotImplementedError

    def forward_rate(self, key, a, b):
        """L_{0,a;b-a} of the index key.index under key.collateral:
        1 + (b-a) L = kappa Z^{e|e}_{0,a} / Z^{e|e}_{0,b} * K, K = 1 under native collateral."""
        s = self.setup
        idx = s.index_by_name(key.index)
        native = DiscountCurveKey(idx.currency, s.collateral_of(idx.currency).name)
        ratio = idx.kappa * self.curves.Z(native, a) / self.curves.Z(native, b)
        if key.collateral != native.collateral:
            if (idx.currency, key.collateral) != (s.foreign, s.domestic_collateral.name):
                raise NotImplementedError(f"{key}: only the foreign index under domestic collateral")
            ratio *= self.libor_convexity(a, b)
        return (ratio - 1.0) / (b - a)

    def _no_vol(self, inst):
        raise NotImplementedError(f"{type(self).__name__} has no volatility: cannot quote {inst.label}")

    def swaption_normal_vol(self, swaption):
        self._no_vol(swaption)

    def fx_black_vol(self, option):
        self._no_vol(option)

    def cb_normal_vol(self, option):
        self._no_vol(option)


class DeterministicPricer(Pricer):
    """Deterministic collateral basis model: no convexity adjustment, no option quotes."""

    def libor_convexity(self, a, b):
        return 1.0


def native_swaption_variance(model, factors, swap, Z):
    """Normal variance of the forward swap rate at T0 under its annuity measure, Gaussian approximation
    (Remark "Native swaps"; separate fixed and float schedules): q^T y_{T0} q with
    q = (-sum_m cf_m G_m P_m + Y sum_k w_k G_k P_k) / A on the given factors of the block."""
    T0 = swap.start
    tf, cf, tx, wx = swap.cashflows()
    Pf, Px = Z(tf) / Z(T0), Z(tx) / Z(T0)
    A = (wx * Px).sum()
    Y = (cf * Pf).sum() / A
    q = np.array([(-(cf * Pf * [model.G(i, T0, T) for T in tf]).sum()
                   + Y * (wx * Px * [model.G(i, T0, T) for T in tx]).sum()) / A for i in factors])
    y = np.array([[model.covint((i, "M", T0), (j, "M", T0), 0.0, T0) for j in factors] for i in factors])
    return float(q @ y @ q)


def native_swaption_exact_vol(model, i, swap, Z):
    """ATM normal vol from the exact one-factor price (Proposition "Exact one factor swaption").
    Under Q^{T0}, z = z^{a,i}_{T0} ~ N(0, y) and Z_{T0,T} = P(T) exp(-G z - G^2 y/2). The swap value at
    the ATM strike K, V(z) = sum_m c_m Z_{T0,t_m}, is increasing in z; with V(z*) = 0,
    E^{T0}[V^+] = sum_m c_m P_m N(-(z* + G_m y)/sqrt(y))."""
    T0 = swap.start
    tf, cf, tx, wx = swap.cashflows()
    fl, A0 = swap.legs(Z)
    K = fl / A0
    t, c = np.r_[tf, tx], np.r_[cf, -K * wx]
    P = Z(t) / Z(T0)
    G = np.array([model.G(i, T0, T) for T in t])
    y = model.covint((i, "M", T0), (i, "M", T0), 0.0, T0)
    s = sqrt(y)
    V = lambda z: float((c * P * np.exp(-G * z - 0.5 * G * G * y)).sum())
    lo, hi = -12 * s, 12 * s
    if not V(lo) < 0 < V(hi):
        raise ValueError(f"{swap.label}: swap value not increasing in the state, no exact closed form")
    zs = brentq(V, lo, hi, xtol=1e-14 * s)
    price = Z(T0) * float((c * P * norm.cdf(-(zs + G * y) / s)).sum())
    return price / (A0 * sqrt(T0 / (2 * pi)))


class LGCMPricer(Pricer):
    """LGCM closed forms. exact=True: one-factor native swaptions by the exact closed form and CB vols
    by exact 2D quadrature (gaussian_approx.exact); otherwise, or with several factors in the block,
    the Gaussian approximations. CB vols need one f and one c factor (gaussian_approx)."""

    def __init__(self, model, curves, exact=True):
        super().__init__(curves)
        self.model, self.exact = model, exact
        self._K, self._v = {}, None

    def _version(self):
        m = self.model
        return hash(tuple(a.tobytes() for a in m.sig + m.beta) + (m.R.tobytes(),))

    def libor_convexity(self, a, b):
        v = self._version()
        if v != self._v:
            self._K, self._v = {}, v
        if (a, b) not in self._K:
            self._K[(a, b)] = libor_convexity(self.model, 0.0, a, b)
        return self._K[(a, b)]

    def swaption_normal_vol(self, swaption):
        u = swaption.underlying
        role = self.setup.role(u.index.currency)
        key = self.setup.key(role * 2)
        Z = lambda T: self.curves.Z(key, T)
        factors = self.model.idx[role]
        if self.exact and len(factors) == 1:
            return native_swaption_exact_vol(self.model, factors[0], u, Z)
        return sqrt(native_swaption_variance(self.model, factors, u, Z) / u.start)

    def fx_black_vol(self, option):
        return sqrt(fx_total_variance(self.model, option.expiry) / option.expiry)

    def _cb_setup(self, option):
        u = option.underlying
        th = self.setup.foreign_index.tenor
        return ga.SwapSetup(T0=u.start, theta=th, n=len(u.schedule(self.setup)) - 1, curves=self.curves)

    def cb_normal_vol(self, option):
        m = self.model
        if len(m.idx["f"]) != 1 or len(m.idx["c"]) != 1:
            raise NotImplementedError("CB vols need exactly one f and one c factor")
        s = self._cb_setup(option)
        if self.exact:
            return float(ga.exact(m, s)["B"]["atm_normal_vol"])
        return sqrt(ga.approx(m, s)["VB"] / s.T0)
