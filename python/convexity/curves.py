"""Indices, curve keys, discount curves and the curve set of the LGCM calibration.

A curve is identified by a CurveKey. DiscountCurveKey(currency, collateral) is Z^{currency|collateral},
the collateral being named by its overnight index: with d = USD, f = EUR,
    Z^{d|d} = Z[USD|USD_SOFR], Z^{f|f} = Z[EUR|EUR_ESTR], Z^{f|d} = Z[EUR|USD_SOFR], Z^{d|f} = Z[USD|EUR_ESTR].
ForwardCurveKey(index, collateral) identifies the projection of an IBOR index under a collateral.

CurveSet maps keys to curves. It also exposes the role-based interface of market.InitialCurves
(Z0(kind, T) for kind in dd/ff/fd/df/qz, kappa_d, kappa_f, kappa(role), fx_forward0, x0), so that the
simulator, the payoffs and gaussian_approx run unchanged on calibrated curves.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .market import Coupon, LIBOR


def _out(x, T):
    return float(x) if np.ndim(T) == 0 else x


@dataclass(frozen=True)
class Index:
    """Rate index. OIS: kappa = 1. IBOR: constant multiplicative funding spread
    kappa = 1 + tenor * spread, i.e. 1 + tenor L_j = kappa Z_{j-1}/Z_j under native collateral
    (kappa = 1 - theta S^{div} in the notes).
    An OIS index pays a compounded overnight coupon: compounding = "daily" (exact, on a grid of step
    on_step) or "continuous" (approximation); see market.Coupon."""
    name: str
    currency: str
    kind: str = "OIS"
    tenor: float = 0.25
    spread: float = 0.0
    compounding: str = "daily"
    on_step: float = 1 / 365

    def __post_init__(self):
        if self.kind not in ("OIS", "IBOR"):
            raise ValueError(f"{self.name}: kind must be OIS or IBOR, got {self.kind!r}")
        if self.kind == "OIS" and self.spread != 0.0:
            raise ValueError(f"{self.name}: an OIS index has no funding spread")
        if self.tenor <= 0:
            raise ValueError(f"{self.name}: tenor must be positive")
        if self.compounding not in ("daily", "continuous"):
            raise ValueError(f"{self.name}: compounding must be daily or continuous, got {self.compounding!r}")
        if not self.on_step > 0:
            raise ValueError(f"{self.name}: on_step must be positive")
        if self.kind == "IBOR" and (self.compounding, self.on_step) != ("daily", 1 / 365):
            raise ValueError(f"{self.name}: compounding and on_step apply to OIS indices only")

    @property
    def coupon(self):
        """market.Coupon of the index: Libor for IBOR, compounded overnight for OIS."""
        return LIBOR if self.kind == "IBOR" else Coupon(self.compounding, self.on_step)

    @property
    def kappa(self):
        return 1.0 + self.tenor * self.spread


class CurveKey:
    """Hashable identifier of a curve (subclasses are frozen dataclasses)."""


@dataclass(frozen=True)
class DiscountCurveKey(CurveKey):
    currency: str
    collateral: str          # name of the collateral overnight index

    def __str__(self):
        return f"Z[{self.currency}|{self.collateral}]"


@dataclass(frozen=True)
class ForwardCurveKey(CurveKey):
    index: str
    collateral: str

    def __str__(self):
        return f"L[{self.index}|{self.collateral}]"


# ---------------------------------------------------------------------- curves
class Curve(ABC):
    @abstractmethod
    def Z(self, T):
        """Discount factor at T >= 0 (float for a scalar T, array for an array)."""

    def zero_rate(self, T):
        T = np.asarray(T, float)
        return _out(-np.log(self.Z(T)) / T, T)

    def forward(self, T, h=1e-6):
        """Instantaneous forward rate (central difference of ln Z)."""
        T = np.asarray(T, float)
        lo = np.maximum(T - h, 0.0)
        return _out(-(np.log(self.Z(T + h)) - np.log(self.Z(lo))) / (T + h - lo), T)


class PiecewiseFlatCurve(Curve):
    """Flat instantaneous forwards f_k on (t_{k-1}, t_k], t_0 = 0; the last forward is extrapolated."""

    def __init__(self, pillars=(), forwards=()):
        self.pillars = [float(t) for t in pillars]
        self.forwards = [float(f) for f in forwards]
        if len(self.pillars) != len(self.forwards) or np.any(np.diff([0.0] + self.pillars) <= 0):
            raise ValueError("pillars must be increasing and positive, one forward per pillar")

    def add_node(self, t, f):
        if t <= (self.pillars[-1] if self.pillars else 0.0) + 1e-12:
            raise ValueError(f"node {t} is not beyond the last pillar")
        self.pillars.append(float(t))
        self.forwards.append(float(f))

    def set_last(self, f):
        self.forwards[-1] = float(f)

    def integral(self, T):
        T = np.asarray(T, float)
        if not self.pillars:
            return np.zeros_like(T)
        t, f = np.array(self.pillars), np.array(self.forwards)
        left = np.concatenate([[0.0], t[:-1]])
        cum = np.concatenate([[0.0], np.cumsum(f * (t - left))])
        k = np.minimum(np.searchsorted(t, T, side="left"), len(t) - 1)
        return cum[k] + f[k] * (T - left[k])

    def Z(self, T):
        return _out(np.exp(-self.integral(T)), T)


class SpreadCurve(Curve):
    """Z = Z^{base} Q with Q piecewise flat: Z^{f|d} = Z^{f|f} Q^{f|f->d}. The node API acts on Q."""

    def __init__(self, base, spread=None):
        self.base, self.spread = base, spread if spread is not None else PiecewiseFlatCurve()

    @property
    def pillars(self):
        return self.spread.pillars

    @property
    def forwards(self):
        return self.spread.forwards

    def add_node(self, t, f):
        self.spread.add_node(t, f)

    def set_last(self, f):
        self.spread.set_last(f)

    def Z(self, T):
        return _out(np.asarray(self.base.Z(T)) * np.asarray(self.spread.Z(T)), T)


class FunctionCurve(Curve):
    """Curve given by a scalar function T -> Z (e.g. the model-implied off-market curve Z^{d|f})."""

    def __init__(self, fn):
        self.fn = fn

    def Z(self, T):
        T = np.asarray(T, float)
        if T.ndim == 0:
            return float(self.fn(float(T)))
        return np.array([self.fn(float(t)) for t in T.ravel()]).reshape(T.shape)


# ---------------------------------------------------------------------- setup and curve set
@dataclass(frozen=True)
class LGCMSetup:
    """Roles of the LGCM: d = domestic (the collateral currency of the cross-currency market), f = foreign.
    The collaterals are overnight indices; the indices are those of the floating legs (default: the
    collateral index). pair = foreign + domestic, x = units of domestic per unit of foreign."""
    domestic_collateral: Index
    foreign_collateral: Index
    domestic_index: Index = None
    foreign_index: Index = None

    def __post_init__(self):
        for role in ("domestic", "foreign"):
            if getattr(self, f"{role}_index") is None:
                object.__setattr__(self, f"{role}_index", getattr(self, f"{role}_collateral"))
            col, idx = getattr(self, f"{role}_collateral"), getattr(self, f"{role}_index")
            if col.kind != "OIS":
                raise ValueError(f"collateral {col.name} must be an OIS index")
            if idx.currency != col.currency:
                raise ValueError(f"{idx.name} is not in the {role} currency {col.currency}")
        if self.domestic == self.foreign:
            raise ValueError("domestic and foreign currencies must differ")
        if self.domestic_index.kind == "IBOR" and self.domestic_index.tenor != self.foreign_index.tenor:
            raise ValueError("resetting CCS: the domestic IBOR tenor must equal the foreign index tenor")

    @property
    def domestic(self):
        return self.domestic_collateral.currency

    @property
    def foreign(self):
        return self.foreign_collateral.currency

    @property
    def pair(self):
        return self.foreign + self.domestic

    def role(self, currency):
        if currency == self.domestic:
            return "d"
        if currency == self.foreign:
            return "f"
        raise ValueError(f"{currency} is neither {self.domestic} nor {self.foreign}")

    def collateral_of(self, currency):
        return self.domestic_collateral if self.role(currency) == "d" else self.foreign_collateral

    def index_by_name(self, name):
        for idx in (self.domestic_index, self.foreign_index, self.domestic_collateral, self.foreign_collateral):
            if idx.name == name:
                return idx
        raise KeyError(name)

    def key(self, kind):
        """DiscountCurveKey of kind = (currency role)(collateral role), e.g. "fd" = Z^{f|d}."""
        ccy = {"d": self.domestic, "f": self.foreign}
        col = {"d": self.domestic_collateral.name, "f": self.foreign_collateral.name}
        return DiscountCurveKey(ccy[kind[0]], col[kind[1]])


class CurveSet:
    """Curves by CurveKey, with the role-based interface of market.InitialCurves."""

    def __init__(self, setup, x0, curves=None):
        self.setup, self.x0 = setup, float(x0)
        self._curves = dict(curves or {})

    def __getitem__(self, key):
        return self._curves[key]

    def __setitem__(self, key, curve):
        self._curves[key] = curve

    def __contains__(self, key):
        return key in self._curves

    def keys(self):
        return self._curves.keys()

    def Z(self, key, T):
        return self._curves[key].Z(T)

    # ------------------------------------------------ InitialCurves interface
    @property
    def kappa_d(self):
        return self.setup.domestic_index.kappa

    @property
    def kappa_f(self):
        return self.setup.foreign_index.kappa

    def kappa(self, ccy):
        return {"d": self.kappa_d, "f": self.kappa_f}[ccy]

    def Z0(self, kind, T):
        if kind == "qz":
            return self.Z0("fd", T) / self.Z0("ff", T)
        key = self.setup.key(kind)
        if key not in self:
            raise ValueError(f"{key} is not in the curve set" + (" (model output, set after calibration)" if kind == "df" else ""))
        return self.Z(key, T)

    def coupon(self, role):
        """market.Coupon of the floating index of role "d" or "f"."""
        return {"d": self.setup.domestic_index, "f": self.setup.foreign_index}[role].coupon

    def fx_forward0(self, T):
        """x^{df|d}_{0,T} (d collateral)."""
        return self.x0 * self.Z0("fd", T) / self.Z0("dd", T)
