"""Calibration instruments and market quotes.

An Instrument is pure data: it declares what it calibrates (`target`: a CurveKey for linear
instruments, a model block "d", "f", "c" or "x" for options), its `pillar` and the curves it reads
besides its target (`dependencies`), and computes its model quote from a Pricer (`quote`). Linear
instruments are priced from the curves and the collateral convexities of the foreign
coupon (Libor or compounded) given by the pricer;
options delegate to the pricer. Values are in the instrument's quotation: par rate, FX outright or
points, CB spread, normal vol (swaptions, CB spreads) or Black vol (FX).

Conventions: spot date 0; swaps are collateralised in their own currency; FX forwards and resetting
CCS are collateralised in the domestic currency; CCS periods are the foreign index tenor.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from .curves import DiscountCurveKey, ForwardCurveKey


def schedule(start, end, tenor):
    """start, start + tenor, ..., end (a whole number of periods)."""
    n = int(round((end - start) / tenor))
    if n < 1 or abs(start + n * tenor - end) > 1e-9:
        raise ValueError(f"[{start}, {end}] is not a whole number of periods of {tenor}")
    return start + tenor * np.arange(n + 1)


def _fmt(t):
    return f"{t * 12:.0f}M" if t < 1 else f"{t:g}Y"


class Instrument(ABC):
    priority = 0          # bootstrapping order between instruments of the same curve (lower first)

    @property
    @abstractmethod
    def pillar(self):
        """Date determined by the instrument (maturity or expiry)."""

    @abstractmethod
    def target(self, setup):
        """CurveKey (linear instruments) or model block in {"d", "f", "c", "x"} (options)."""

    def dependencies(self, setup):
        """Curves read by the instrument besides its target."""
        return set()

    @abstractmethod
    def quote(self, pricer):
        """Model quote, in the instrument's quotation."""

    @property
    @abstractmethod
    def label(self):
        """Short name for reports."""


# ---------------------------------------------------------------------- linear
@dataclass(frozen=True)
class Swap(Instrument):
    """Fixed vs floating swap on [start, end] under native collateral: fixed leg every `fixed_tenor`,
    floating leg on the index tenor. Par rate = sum_i (kappa Z_{t_{i-1}} - Z_{t_i}) / sum_k tau Z_{t_k}."""
    index: object
    end: float
    fixed_tenor: float = 1.0
    start: float = 0.0

    @property
    def pillar(self):
        return self.end

    @property
    def label(self):
        lead = f"{_fmt(self.start)}x" if self.start > 0 else ""
        return f"{self.index.name} swap {lead}{_fmt(self.end - self.start)}"

    def target(self, setup):
        return DiscountCurveKey(self.index.currency, setup.collateral_of(self.index.currency).name)

    def cashflows(self):
        """Float dates tf with coefficients cf (float leg = sum cf Z(tf)), fixed dates tx with weights wx."""
        tf = schedule(self.start, self.end, self.index.tenor)
        cf = np.zeros(len(tf))
        cf[:-1] += self.index.kappa
        cf[1:] -= 1.0
        tx = schedule(self.start, self.end, self.fixed_tenor)[1:]
        return tf, cf, tx, np.full(len(tx), self.fixed_tenor)

    def legs(self, Z):
        tf, cf, tx, wx = self.cashflows()
        return float((cf * Z(tf)).sum()), float((wx * Z(tx)).sum())

    def quote(self, pricer):
        key = self.target(pricer.setup)
        fl, A = self.legs(lambda T: pricer.curves.Z(key, T))
        return fl / A


@dataclass(frozen=True)
class FXForward(Instrument):
    """FX forward on the model pair (units of domestic per foreign) under domestic collateral:
    x^{df|d}_{0,T} = x_0 Z^{f|d}_{0,T} / Z^{d|d}_{0,T}. quotation "outright" or "points" = (F - x_0) pip_factor.
    With `inverted` the market quotes the reciprocal pair (units of foreign per domestic, e.g. USDJPY
    with d = USD, f = JPY): outright 1/F, points (1/F - 1/x_0) pip_factor. `to_outright` and `quoted`
    convert between the quote and the model outright F (x_0 the model spot)."""
    maturity: float
    quotation: str = "outright"
    pip_factor: float = 1e4
    inverted: bool = False

    def __post_init__(self):
        if self.quotation not in ("outright", "points"):
            raise ValueError(self.quotation)

    @property
    def pillar(self):
        return self.maturity

    @property
    def label(self):
        tags = ([self.quotation] if self.quotation == "points" else []) + (["inverted"] if self.inverted else [])
        return f"FX forward {_fmt(self.maturity)}" + (f" ({', '.join(tags)})" if tags else "")

    def target(self, setup):
        return setup.key("fd")

    def dependencies(self, setup):
        return {setup.key("dd")}

    def outright(self, pricer):
        s, c = pricer.setup, pricer.curves
        return c.x0 * c.Z(s.key("fd"), self.maturity) / c.Z(s.key("dd"), self.maturity)

    def quoted(self, F, x0):
        """Quote (in the quoted convention) of the model outright F."""
        out, spot = (1.0 / F, 1.0 / x0) if self.inverted else (F, x0)
        return out if self.quotation == "outright" else (out - spot) * self.pip_factor

    def to_outright(self, value, x0):
        """Model outright F implied by a quote."""
        spot = 1.0 / x0 if self.inverted else x0
        out = value if self.quotation == "outright" else spot + value / self.pip_factor
        return 1.0 / out if self.inverted else out

    def quote(self, pricer):
        return self.quoted(self.outright(pricer), pricer.curves.x0)


@dataclass(frozen=True)
class ResettingCCS(Instrument):
    """Resetting cross-currency basis swap on [start, end] under domestic collateral, periods theta =
    foreign index tenor: foreign index + B against the domestic index on a notional reset at each fixing.
        B = sum_j P_j (1 + theta L^r_j - (1 + theta L^{f|d}_j)) / (theta sum_j P_j),  P_j = Z^{f|d}_{0,T_j},
        1 + theta L^r_j = kappa^d P_{j-1}/P_j,  1 + theta L^{f|d}_j = kappa^f Z^{f|f}_{j-1}/Z^{f|f}_j K_j,
    K_j the collateral convexity of the foreign coupon (Libor or compounded) given by the pricer."""
    end: float
    start: float = 0.0
    priority = 1

    @property
    def pillar(self):
        return self.end

    @property
    def label(self):
        lead = f"{_fmt(self.start)}x" if self.start > 0 else ""
        return f"resetting CCS {lead}{_fmt(self.end - self.start)}"

    def target(self, setup):
        return setup.key("fd")

    def dependencies(self, setup):
        return {setup.key("ff")}

    def schedule(self, setup):
        return schedule(self.start, self.end, setup.foreign_index.tenor)

    def components(self, pricer):
        """Annuity A = theta sum P_j and spread B."""
        s = pricer.setup
        Tj, th = self.schedule(s), s.foreign_index.tenor
        P = pricer.curves.Z(s.key("fd"), Tj)
        one_Lr = s.domestic_index.kappa * P[:-1] / P[1:]
        fkey = ForwardCurveKey(s.foreign_index.name, s.domestic_collateral.name)
        one_L = 1.0 + th * np.array([pricer.forward_rate(fkey, a, b) for a, b in zip(Tj[:-1], Tj[1:])])
        A = th * P[1:].sum()
        return float(A), float(((one_Lr - one_L) * P[1:]).sum() / A)

    def quote(self, pricer):
        return self.components(pricer)[1]


# ---------------------------------------------------------------------- options
@dataclass(frozen=True)
class Swaption(Instrument):
    """ATM swaption on a native forward-starting swap, quoted as a normal vol."""
    underlying: Swap

    @classmethod
    def make(cls, index, expiry, tenor, fixed_tenor=1.0):
        return cls(Swap(index, expiry + tenor, fixed_tenor, expiry))

    @property
    def expiry(self):
        return self.underlying.start

    @property
    def pillar(self):
        return self.expiry

    @property
    def label(self):
        u = self.underlying
        return f"{u.index.name} swaption {_fmt(self.expiry)}x{_fmt(u.end - u.start)}"

    def target(self, setup):
        return setup.role(self.underlying.index.currency)

    def quote(self, pricer):
        return pricer.swaption_normal_vol(self)


@dataclass(frozen=True)
class FXOption(Instrument):
    """ATM FX option at `expiry`, quoted as a Black vol (collateral free in the LGCM). Unchanged by
    inverting the quoted pair: ln x_T is Gaussian, so 1/x_T is lognormal with the same total variance
    and the ATM Black vol of 1/x equals that of x (proved in the notes; checked by quadrature in
    test_fx_option_vol_is_invariant_under_inversion)."""
    expiry: float

    @property
    def pillar(self):
        return self.expiry

    @property
    def label(self):
        return f"FX option {_fmt(self.expiry)}"

    def target(self, setup):
        return "x"

    def quote(self, pricer):
        return pricer.fx_black_vol(self)


@dataclass(frozen=True)
class CurrencyBasisSwaption(Instrument):
    """ATM option on the forward resetting CB spread, quoted as a normal vol under the annuity measure."""
    underlying: ResettingCCS

    @classmethod
    def make(cls, expiry, tenor):
        return cls(ResettingCCS(expiry + tenor, expiry))

    @property
    def expiry(self):
        return self.underlying.start

    @property
    def pillar(self):
        return self.expiry

    @property
    def label(self):
        u = self.underlying
        return f"CB vol {_fmt(self.expiry)}x{_fmt(u.end - u.start)}"

    def target(self, setup):
        return "c"

    def quote(self, pricer):
        return pricer.cb_normal_vol(self)


# ---------------------------------------------------------------------- quotes
@dataclass(frozen=True)
class Quote:
    instrument: Instrument
    value: float


@dataclass
class MarketQuotes:
    setup: object
    fx_spot: float          # the QUOTED spot (units of foreign per domestic if fx_inverted)
    quotes: list
    fx_inverted: bool = False

    def __post_init__(self):
        """The spot and every FX forward must be quoted in the same convention: a mismatch is otherwise
        not detected by the curve calibration (EURUSD with only fx_inverted flipped calibrates, with the
        Z^{f|d} zero rate off by about -43 bp at 0.5Y)."""
        bad = [q.instrument.label for q in self.quotes
               if isinstance(q.instrument, FXForward) and q.instrument.inverted != self.fx_inverted]
        if bad:
            raise ValueError(f"FX spot quoted with fx_inverted={self.fx_inverted} but forwards {bad} have "
                             f"inverted={not self.fx_inverted}: quote the spot and forwards in the same convention")

    @property
    def x0(self):
        """Model spot: units of domestic per unit of foreign."""
        return 1.0 / self.fx_spot if self.fx_inverted else self.fx_spot

    def of_type(self, *types):
        return [q for q in self.quotes if isinstance(q.instrument, types)]

    def for_target(self, target):
        return [q for q in self.quotes if q.instrument.target(self.setup) == target]

    def with_values(self, values):
        return replace(self, quotes=[Quote(q.instrument, float(v)) for q, v in zip(self.quotes, values, strict=True)])

    def without(self, *types):
        return replace(self, quotes=[q for q in self.quotes if not isinstance(q.instrument, types)])
