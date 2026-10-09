"""Payoffs for the Monte Carlo simulator, as a class hierarchy.

    Payoff                      (abstract) payment in `ccy` under `collateral` at `expiry`
    ├── DiscountFactor          Z_{T,Tm}                  (holding a zero coupon bond until T)
    ├── ForwardDiscountFactor   Z_{S,T2}/Z_{S,T1} paid at S, or Z_{S,T2} (paid at T1)
    ├── SwapPayoff              (abstract) forward-starting swap on T0 < T1 < ... < Tn
    │   ├── Swap                omega A_{T0} (Y_{T0} - K)
    │   └── Swaption            A_{T0} (omega (Y_{T0} - K))^+
    └── FXPayoff                (abstract) payment in d, observed on the FX spot at expiry
        ├── FXForward           x_T - K
        └── FXOption            (omega (x_T - K))^+

Each payoff gives its value at expiry on every path (`value`). The time-0 price in `ccy` is
E^{d,RN}[D V_T], with the deflator D of `Simulator.deflator`. `lgcm_price` returns the closed-form
LGCM price when it exists (exact for all but swaptions, for which it is the Gaussian approximation).

Supported (currency, collateral) pairs: discount factors and swaps on Z^{d|d}, Z^{f|f}, Z^{f|d};
FX payoffs in d under d or f collateral.

QGCM caveat: the foreign Libor under domestic collateral L^{f|d} is not a function of the state in
the QGCM. Swaps and swaptions in "fd" use the deterministic LGCM convexity, which is then an
approximation; the Monte Carlo study (mc_verification, section 4) finds it accurate.
"""
from abc import ABC, abstractmethod
from math import sqrt

import numpy as np

from . import gaussian_approx as ga
from .market import (InitialCurves, libor_convexity, lgcm_discount0, lgcm_fx_forward0,
                     fx_total_variance, black, bachelier)
from .montecarlo import mean_and_stderr

RATE_KINDS = ("dd", "ff", "fd")


class Payoff(ABC):
    def __init__(self, expiry, ccy, collateral):
        self.expiry, self.ccy, self.collateral = float(expiry), ccy, collateral

    @property
    def kind(self):
        return self.ccy + self.collateral

    @abstractmethod
    def value(self, sim, res, curves):
        """Value at expiry, in `ccy`, on every path."""

    def pv_paths(self, sim, res, curves):
        return sim.deflator(res, curves, self.ccy, self.collateral) * self.value(sim, res, curves)

    def lgcm_price(self, model, curves):
        raise NotImplementedError


# ---------------------------------------------------------------------- discount factors
class DiscountFactor(Payoff):
    """Value at T of the zero coupon bond paying 1 at `maturity`. Price Z_{0,maturity} in any model."""

    def __init__(self, expiry, maturity, ccy="d", collateral="d"):
        super().__init__(expiry, ccy, collateral)
        if self.kind not in RATE_KINDS:
            raise NotImplementedError(f"no reconstruction formula for Z^{{{ccy}|{collateral}}}")
        self.maturity = maturity

    def value(self, sim, res, curves):
        return sim.discount(res, curves, self.kind, self.maturity)

    def lgcm_price(self, model, curves):
        return float(curves.Z0(self.kind, self.maturity))


class ForwardDiscountFactor(Payoff):
    """Forward discount factor Z_{S,T1,T2} = Z_{S,T2}/Z_{S,T1} fixed at S = expiry.

    paid_at="fixing": paid at S, a convexity-adjusted claim;
    paid_at="T1":     paid at T1 (value Z_{S,T2} at S), whose price is Z_{0,T2} in any model.
    """

    def __init__(self, expiry, T1, T2, ccy="d", collateral="d", paid_at="fixing"):
        super().__init__(expiry, ccy, collateral)
        if self.kind not in RATE_KINDS:
            raise NotImplementedError(f"no reconstruction formula for Z^{{{ccy}|{collateral}}}")
        if paid_at not in ("fixing", "T1"):
            raise ValueError(paid_at)
        self.T1, self.T2, self.paid_at = T1, T2, paid_at

    def value(self, sim, res, curves):
        z2 = sim.discount(res, curves, self.kind, self.T2)
        return z2 if self.paid_at == "T1" else z2 / sim.discount(res, curves, self.kind, self.T1)

    def lgcm_price(self, model, curves):
        Z0 = lambda T: float(curves.Z0(self.kind, T))
        if self.paid_at == "T1":
            return Z0(self.T2)
        # under the S-forward measure, the Cheyette state of the curve is N(0, y_S): see Lemma "zhat forward law"
        S = self.expiry
        mask = {"dd": model.mask("d"), "ff": model.mask("f"), "fd": model.mask("f", "c")}[self.kind]
        g1, g2, Y = model.Gvec(S, self.T1) * mask, model.Gvec(S, self.T2) * mask, model.Y(S)
        return Z0(S) * Z0(self.T2) / Z0(self.T1) * float(np.exp(-g1 @ Y @ (g2 - g1)))


# ---------------------------------------------------------------------- swaps
class SwapPayoff(Payoff):
    """Forward-starting swap: fixed leg K against Libor on T0 < T1 < ... < Tn, accruals theta."""

    def __init__(self, start, n, theta, strike, ccy="d", collateral="d", payer=True):
        super().__init__(start, ccy, collateral)
        if self.kind not in RATE_KINDS:
            raise NotImplementedError(f"swaps in {ccy} under {collateral} collateral")
        self.n, self.theta, self.strike, self.omega = n, theta, strike, (1.0 if payer else -1.0)
        self.Tj = start + theta * np.arange(n + 1)

    def underlying(self, sim, res, curves):
        """Annuity A_{T0} = sum theta Z_{T0,Tj} and swap rate Y_{T0} on every path."""
        P = np.stack([sim.discount(res, curves, self.kind, T) for T in self.Tj[1:]], axis=1)
        nat = self.ccy * 2
        Zn = np.stack([sim.discount(res, curves, nat, T) for T in self.Tj], axis=1)
        one_L = curves.kappa(self.ccy) * Zn[:, :-1] / Zn[:, 1:] * self.convexities(sim.m, self.expiry)
        A = self.theta * P.sum(1)
        return A, ((one_L - 1) * P).sum(1) / A        # theta_j L_j = (1 + theta_j L_j) - 1

    def convexities(self, model, t):
        """(1+th L^{e|c}) / (1+th L^{e|e}) per period, seen from t (LGCM formula; 1 for native swaps)."""
        if self.kind != "fd":
            return np.ones(self.n)
        return np.array([libor_convexity(model, t, a, b) for a, b in zip(self.Tj[:-1], self.Tj[1:])])

    def forward(self, model, curves):
        """Time-0 annuity (spot, undiscounted to T0) and forward swap rate."""
        P = curves.Z0(self.kind, self.Tj[1:])
        Zn = curves.Z0(self.ccy * 2, self.Tj)
        one_L = curves.kappa(self.ccy) * Zn[:-1] / Zn[1:] * self.convexities(model, 0.0)
        A = self.theta * P.sum()
        return A, float(((one_L - 1) * P).sum() / A)


class Swap(SwapPayoff):
    def value(self, sim, res, curves):
        A, Y = self.underlying(sim, res, curves)
        return self.omega * A * (Y - self.strike)

    def lgcm_price(self, model, curves):
        A, Y = self.forward(model, curves)
        return self.omega * A * (Y - self.strike)


class Swaption(SwapPayoff):
    def value(self, sim, res, curves):
        A, Y = self.underlying(sim, res, curves)
        return A * np.maximum(self.omega * (Y - self.strike), 0.0)

    def approx_variance(self, model, curves):
        """Normal variance of Y_{T0} under the annuity measure, Gaussian approximation (notes, LGCM)."""
        s = ga.SwapSetup(T0=self.expiry, theta=self.theta, n=self.n, curves=curves)
        if self.kind == "fd":
            return ga.approx(model, s)["VY"]
        return ga.approx_native(model, s, ccy=self.ccy)["V"]

    def lgcm_price(self, model, curves):
        """Bachelier price with the Gaussian approximation (an approximation, not exact)."""
        A, Y = self.forward(model, curves)
        vol = sqrt(self.approx_variance(model, curves) / self.expiry)
        return A * bachelier(Y, self.strike, vol, self.expiry, call=self.omega > 0)


# ---------------------------------------------------------------------- FX
class FXPayoff(Payoff):
    """Payment in d at expiry, function of the FX spot x^{df}_T (units of d per unit of f)."""

    def __init__(self, expiry, strike, collateral="d"):
        super().__init__(expiry, "d", collateral)
        self.strike = strike

    def lgcm_forward_and_discount(self, model, curves):
        return (lgcm_fx_forward0(model, curves, self.expiry, self.collateral),
                lgcm_discount0(model, curves, self.expiry, "d", self.collateral))


class FXForward(FXPayoff):
    def value(self, sim, res, curves):
        return sim.fx_spot(res, curves) - self.strike

    def lgcm_price(self, model, curves):
        F, Z = self.lgcm_forward_and_discount(model, curves)
        return Z * (F - self.strike)


class FXOption(FXPayoff):
    def __init__(self, expiry, strike, collateral="d", call=True):
        super().__init__(expiry, strike, collateral)
        self.call = call

    def value(self, sim, res, curves):
        x = sim.fx_spot(res, curves)
        return np.maximum((x - self.strike) if self.call else (self.strike - x), 0.0)

    def lgcm_price(self, model, curves):
        """Black formula: forward x^{df|c}_{0,T}, discount Z^{d|c}_{0,T}, collateral-independent variance."""
        F, Z = self.lgcm_forward_and_discount(model, curves)
        return Z * black(F, self.strike, fx_total_variance(model, self.expiry), self.call)


# ---------------------------------------------------------------------- pricer
def price(sim, payoffs, curves=None, n_paths=2**16, seed=0, antithetic=True):
    """Monte Carlo prices of a list of payoffs. Paths are simulated once per distinct expiry and
    shared by all payoffs with that expiry. Returns a list of (price, standard error)."""
    curves = curves or InitialCurves()
    out = [None] * len(payoffs)
    for T in sorted({p.expiry for p in payoffs}):
        res = sim.simulate(T, n_paths, seed=seed, antithetic=antithetic)
        for k, p in enumerate(payoffs):
            if p.expiry == T:
                out[k] = mean_and_stderr(p.pv_paths(sim, res, curves), antithetic)
    return out
