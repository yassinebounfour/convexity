"""Initial market data of the LGCM / QGCM and the closed-form LGCM values used as benchmarks.

Curve kinds (currency | collateral): "dd" = Z^{d|d}, "ff" = Z^{f|f}, "fd" = Z^{f|d}, "df" = Z^{d|f}.
The model takes Z^{d|d}_0, Z^{f|f}_0, Z^{f|d}_0 and x_0 as inputs. Z^{d|f}_0 is an output of the
model (collateral convexity of FX forwards) and is given here in closed form for the LGCM.
"""
from dataclasses import dataclass
from math import log, sqrt

import numpy as np
from scipy.stats import norm


@dataclass
class InitialCurves:
    x0: float = 1.10            # FX spot: units of d per unit of f
    kappa_d: float = 1.0004     # 1 - theta S^{d|d,div} (constant multiplicative funding spread)
    kappa_f: float = 1.0006     # 1 - theta S^{f|f,div}

    # integrated initial forward curves int_0^T r_{0,u} du
    @staticmethod
    def int_rd0(T):
        return 0.025 * T + 0.008 * (T - 3 * (1 - np.exp(-T / 3)))

    @staticmethod
    def int_rf0(T):
        return 0.03 * T + 0.01 * (T - 4 * (1 - np.exp(-T / 4)))

    @staticmethod
    def int_q0(T):              # collateral basis q^{f|f->d}_{0,.}
        return -0.002 * T - 0.005 * (1 - np.exp(-T / 5))

    def Z0(self, kind, T):
        """Initial discount factor Z^{kind}_{0,T} for kind in {"dd", "ff", "fd", "qz"} (qz = Z^{f|d}/Z^{f|f})."""
        if kind == "dd":
            return np.exp(-self.int_rd0(T))
        if kind == "ff":
            return np.exp(-self.int_rf0(T))
        if kind == "fd":
            return np.exp(-self.int_rf0(T) - self.int_q0(T))
        if kind == "qz":
            return np.exp(-self.int_q0(T))
        raise ValueError(f"{kind}: Z^{{d|f}}_0 is a model output, see lgcm_Zdf0")

    def kappa(self, ccy):
        return {"d": self.kappa_d, "f": self.kappa_f}[ccy]

    def fx_forward0(self, T):
        """x^{df|d}_{0,T}: FX forward under domestic collateral (a model input)."""
        return self.x0 * self.Z0("fd", T) / self.Z0("dd", T)


# ---------------------------------------------------------------------- LGCM closed forms
def libor_convexity(m, t, a, b):
    """(1+th L^{f|d}_{t,a;b-a}) / (1+th L^{f|f}_{t,a;b-a}) = exp(-int_t^a Sigma^f_{u,a;th}.Sigma^c_{u,b} du), LGCM."""
    if a <= t:
        return 1.0
    return float(np.exp(-(m.I(m.Sig("f", b), m.Sig("c", b), t, a) - m.I(m.Sig("f", a), m.Sig("c", b), t, a))))


def fx_collateral_convexity(m, T):
    """x^{df|f}_{0,T} / x^{df|d}_{0,T} = exp(-int_0^T Sigma^{df}_{u,T}.Sigma^{d->f}_{u,T} du), Sigma^{d->f} = -Sigma^c."""
    return float(np.exp(m.I(m.SigDF(T), m.Sig("c", T), 0.0, T)))


def lgcm_fx_forward0(m, curves, T, collateral):
    x = curves.fx_forward0(T)
    return x if collateral == "d" else x * fx_collateral_convexity(m, T)


def lgcm_Zdf0(m, curves, T):
    """Z^{d|f}_{0,T} = x_0 Z^{f|f}_{0,T} / x^{df|f}_{0,T}."""
    return curves.x0 * curves.Z0("ff", T) / lgcm_fx_forward0(m, curves, T, "f")


def lgcm_discount0(m, curves, T, ccy, collateral):
    kind = ccy + collateral
    return lgcm_Zdf0(m, curves, T) if kind == "df" else float(curves.Z0(kind, T))


def fx_total_variance(m, T):
    """w^{df}_0(T) = int_0^T |Sigma^{df}_{u,T}|^2 du (independent of the collateral)."""
    return m.I(m.SigDF(T), m.SigDF(T), 0.0, T)


def black(F, K, w, call=True):
    """Undiscounted Black price with total variance w."""
    s = sqrt(w)
    d1 = (log(F / K) + 0.5 * w) / s
    d2 = d1 - s
    return F * norm.cdf(d1) - K * norm.cdf(d2) if call else K * norm.cdf(-d2) - F * norm.cdf(-d1)


def bachelier(F, K, vol, T, call=True):
    s = vol * sqrt(T)
    d = (F - K) / s
    return (F - K) * norm.cdf(d) + s * norm.pdf(d) if call else (K - F) * norm.cdf(-d) + s * norm.pdf(d)
