"""Initial market data of the LGCM / QGCM and the closed-form LGCM values used as benchmarks.

Curve kinds (currency | collateral): "dd" = Z^{d|d}, "ff" = Z^{f|f}, "fd" = Z^{f|d}, "df" = Z^{d|f}.
The model takes Z^{d|d}_0, Z^{f|f}_0, Z^{f|d}_0 and x_0 as inputs. Z^{d|f}_0 is an output of the
model (collateral convexity of FX forwards) and is given here in closed form for the LGCM.
"""
from dataclasses import dataclass
from math import log, sqrt

import numpy as np
from scipy.stats import norm

from . import lgcm


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

    def coupon(self, role):
        """Coupon type of the role's floating index: always Libor for the synthetic curves."""
        return LIBOR


# ---------------------------------------------------------------------- LGCM closed forms
def libor_convexity(m, t, a, b):
    """(1+th L^{f|d}_{t,a;b-a}) / (1+th L^{f|f}_{t,a;b-a}) = exp(-int_t^a Sigma^f_{u,a;th}.Sigma^c_{u,b} du), LGCM."""
    if a <= t:
        return 1.0
    return float(np.exp(-(m.I(m.Sig("f", b), m.Sig("c", b), t, a) - m.I(m.Sig("f", a), m.Sig("c", b), t, a))))


def coupon_grid(a, b, step):
    """Overnight grid a = t_0 < ... < t_n = b with steps `step` (the last one shortened)."""
    n = max(int(np.ceil((b - a) / step - 1e-9)), 1)
    return np.r_[a + step * np.arange(n), b]


def _compounded_convexity_scalar(m, t, a, b, compounding="daily", step=1 / 365):
    """Scalar loop version of compounded_convexity (test oracle)."""
    sc = m.Sig("c", b)
    if compounding == "continuous":
        x = m.I(m.Sig("f", b), sc, max(t, a), b) if b > max(t, a) else 0.0
        if t < a:
            x += m.I(m.Sig("f", b), sc, t, a) - m.I(m.Sig("f", a), sc, t, a)
        return float(np.exp(-x))
    if compounding != "daily":
        raise ValueError(compounding)
    g = coupon_grid(a, b, step)
    x = sum(m.I(m.Sig("f", t1), sc, t, t0) - m.I(m.Sig("f", t0), sc, t, t0)
            for t0, t1 in zip(g[:-1], g[1:]) if t0 > t)
    return float(np.exp(-x))


def _daily_log_convexity(m, t, a, b, step):
    """-ln K_t of the daily coupon, vectorised over the grid (same value as the scalar loop).

    With Sigma^f_{u,t_{k+1}} - Sigma^f_{u,t_k} = M(u,t_k) G(t_k,t_{k+1}) per factor and the cocycle
    M(u,t_k) = M(u,a') M(a',t_k), a' = max(t, a):
        -ln K = sum_{i in f, j in c} sum_{k: t_k > t} G_i(t_k,t_{k+1}) M_i(a',t_k) [A_ij + J_ij(t_k)],
    A_ij = covint((i,M,a'), (j,G,b), t, a') (one closed form per pair, zero for t >= a) and
    J_ij(t_k) = int_{a'}^{t_k} M_i(u,a') G_j(u,b) rho sig_i sig_j du, accumulated over the grid refined by
    the knots in closed form (the p + q E(beta, e-u) loadings of Model.covint, vectorised kernels)."""
    g = coupon_grid(a, b, step)
    fix = g[:-1][g[:-1] > t]
    if fix.size == 0:
        return 0.0
    a1 = max(t, a)
    pts = np.unique(np.r_[a1, fix, b, m.knots[(m.knots > a1) & (m.knots < b)]])
    s, e = pts[:-1], pts[1:]
    D = e - s
    pk = m.pieces(0.5 * (s + e))
    kfix = np.searchsorted(pts, fix)                         # position of each fixing in pts
    step_of = np.searchsorted(fix, s, side="right") - 1      # fixing step containing [s, e]
    x = 0.0
    for i in m.idx["f"]:
        bi = m.beta[i][pk]
        dBi = m.Bcum_vec(i, pts) - m.Bcum_vec(i, a1)          # int_{a'}^{pts} beta_i
        mk = np.exp(-dBi[kfix])                              # M_i(a', t_k)
        # G_i(t_k, t_{k+1}) = sum over [s, e] in the step of M_i(t_k, s) E(beta_i, e - s)
        # ([t, t_0] when t is inside the period belongs to no remaining step)
        inside = step_of >= 0
        gk = np.bincount(step_of[inside], (np.exp(-(dBi[:-1] - dBi[kfix][step_of])) * lgcm.E_vec(bi, D))[inside],
                         minlength=fix.size)
        pa = np.exp(dBi[1:]); qa = -bi * pa                  # loading M_i(u, a') = pa + qa E(beta_i, e - u)
        for j in m.idx["c"]:
            bj = m.beta[j][pk]
            dBj = m.Bcum_vec(j, pts) - m.Bcum_vec(j, a1)
            # G_j(s_l, b) = exp(dBj(s_l)) sum_{l' >= l} exp(-dBj(s_l')) E(beta_j, D_l')
            R = np.r_[np.cumsum((np.exp(-dBj[:-1]) * lgcm.E_vec(bj, D))[::-1])[::-1], 0.0]
            pb = np.exp(dBj[1:]) * R[1:]; qb = 1.0 - bj * pb   # loading G_j(u, b) = pb + qb E(beta_j, e - u)
            val = (pa * pb * D + pa * qb * lgcm.Phi1_vec(bj, D) + qa * pb * lgcm.Phi1_vec(bi, D)
                   + qa * qb * lgcm.Phi2_vec(bi, bj, D)) * m.R[i, j] * m.sig[i][pk] * m.sig[j][pk]
            J = np.r_[0.0, np.cumsum(val)][kfix]
            A = m.covint((i, "M", a1), (j, "G", b), t, a1) if t < a1 else 0.0
            x += float(np.sum(gk * mk * (A + J)))
    return x


def compounded_convexity(m, t, a, b, compounding="daily", step=1 / 365):
    """K_t = E^{f|d,b}_t[C] / E^{f|f,b}_t[C] for the foreign overnight factor C compounded on [a, b] and paid
    at b under domestic collateral, LGCM (Proposition "Collateral convexity of compounded coupons"):
      daily (exact, C = prod 1/Z^{f|f}_{t_k,t_{k+1}} on coupon_grid(a, b, step)):
        ln K = -sum_{k: t_k > t} int_t^{t_k} (Sigma^f_{u,t_{k+1}} - Sigma^f_{u,t_k}) . Sigma^c_{u,b} du;
      continuous (C = exp(int_a^b r^{f|f}), the step -> 0 limit):
        ln K = -int_t^a (Sigma^f_{u,b} - Sigma^f_{u,a}) . Sigma^c_{u,b} du - int_{max(t,a)}^b Sigma^f_{u,b} . Sigma^c_{u,b} du.
    For t inside the period only the fixings after t carry convexity (the realised part cancels).
    The daily case is vectorised (_daily_log_convexity); _compounded_convexity_scalar is the oracle."""
    if compounding == "daily":
        return float(np.exp(-_daily_log_convexity(m, t, a, b, step)))
    return _compounded_convexity_scalar(m, t, a, b, compounding, step)


@dataclass(frozen=True)
class Coupon:
    """Foreign coupon factor C on [a, b] paid at b: "libor" (kappa / Z^{f|f}_{a,b}, fixed at a; kappa kept
    outside the convexity), "daily" (exact daily compounding prod 1/Z^{f|f}_{t_k,t_{k+1}} on
    coupon_grid(a, b, step)), "continuous" (exp(int_a^b r^{f|f}), the step -> 0 limit of daily)."""
    kind: str = "libor"
    step: float = 1 / 365

    def __post_init__(self):
        if self.kind not in ("libor", "daily", "continuous"):
            raise ValueError(self.kind)
        if not self.step > 0:
            raise ValueError(f"step must be positive, got {self.step}")

    def grid(self, a, b):
        """Fixing grid of the coupon: [a, b] for Libor, coupon_grid(a, b, step) for daily."""
        if self.kind == "continuous":
            raise ValueError("a continuous coupon has no fixing grid")
        return np.array([a, b], float) if self.kind == "libor" else coupon_grid(a, b, self.step)

    def convexity(self, m, t, a, b):
        """K_t = E^{f|d,b}_t[C] / E^{f|f,b}_t[C] in the LGCM."""
        if self.kind == "libor":
            return libor_convexity(m, t, a, b)
        return compounded_convexity(m, t, a, b, self.kind, self.step)


LIBOR = Coupon("libor")


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
