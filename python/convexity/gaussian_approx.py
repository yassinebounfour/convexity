"""Gaussian approximations of forward swap rates and resetting currency basis spreads in the LGCM.

Implements Propositions "Gaussian approximation of forward swap rates" and "Gaussian approximation
of currency basis spreads" of docs/CollateralDiscounting.tex, and the exact benchmark: by Lemma
"zhat forward law", zhat_{T0} = (z^f + v^{f,c}, z^c)_{T0} ~ N(0, yhat_{T0}) under Q^{f|d,T0}, and Y, B
are explicit functions of zhat (Lemma "par spread state functions"), so the exact moments under the
annuity measure follow from 2D Gauss-Hermite quadrature.

The model must have exactly one foreign and one collateral basis factor (2D quadrature).
"""
from dataclasses import dataclass, field
from math import sqrt, pi
import numpy as np

from .market import InitialCurves, libor_convexity


@dataclass
class SwapSetup:
    """Swap schedule T0 < T0 + theta < ... < T0 + n theta, with the initial curves and funding spreads."""
    T0: float = 5.0
    theta: float = 0.25
    n: int = 40
    curves: InitialCurves = field(default_factory=InitialCurves)
    Tj: np.ndarray = field(init=False)

    def __post_init__(self):
        self.Tj = self.T0 + self.theta * np.arange(self.n + 1)

    @property
    def kappa_d(self):
        return self.curves.kappa_d

    @property
    def kappa_f(self):
        return self.curves.kappa_f

    def Zdd0(self, T):
        return self.curves.Z0("dd", T)

    def Zff0(self, T):
        return self.curves.Z0("ff", T)

    def Zfd0(self, T):
        return self.curves.Z0("fd", T)


def _ingredients(m, s):
    (f,), (c,) = m.idx["f"], m.idx["c"]
    T0, Tj, th = s.T0, s.Tj, s.theta
    Y = m.Y(T0)
    Ys = Y[np.ix_([f, c], [f, c])]
    vf = m.vfc(T0)[f]
    Gf = np.array([m.G(f, T0, T) for T in Tj])
    Gc = np.array([m.G(c, T0, T) for T in Tj])

    def K(t, j):   # Libor collateral convexity, period j (fix T_{j-1}, pay T_j), seen from t
        return libor_convexity(m, t, Tj[j - 1], Tj[j])

    K0 = np.array([np.nan] + [K(0.0, j) for j in range(1, s.n + 1)])
    KT = np.array([np.nan] + [K(T0, j) for j in range(1, s.n + 1)])
    return Ys, vf, Gf, Gc, K0, KT


def _rates(s, P, Zff, Kv):
    """P[..., j] = Z^{f|d}_{., T_j}, Zff[..., j] = Z^{f|f}_{., T_j}, j = 0..n, normalised at T0."""
    th = s.theta
    one_L = s.kappa_f * Zff[..., :-1] / Zff[..., 1:] * Kv[1:]
    one_Lr = s.kappa_d * P[..., :-1] / P[..., 1:]
    A = th * P[..., 1:].sum(-1)
    Y = ((one_L - 1) * P[..., 1:]).sum(-1) / A
    B = ((one_Lr - one_L) * P[..., 1:]).sum(-1) / A
    return A, Y, B, (one_L - 1) / th, (one_Lr - 1) / th


def approx(m, s=None):
    """Forward Y0, B0, approximate variances and loadings (qY, qB) on (zhat^f, z^c)."""
    s = s or SwapSetup()
    Ys, vf, Gf, Gc, K0, KT = _ingredients(m, s)
    th = s.theta
    P = s.Zfd0(s.Tj) / s.Zfd0(s.T0)
    Zff = s.Zff0(s.Tj) / s.Zff0(s.T0)
    A, Y, B, L, Lr = _rates(s, P, Zff, K0)
    Pj = P[1:]
    qY = np.array([(Pj * (1 + th * Y) * Gf[1:] - Pj * (1 + th * L) * Gf[:-1]).sum(),
                   (th * Pj * (Y - L) * Gc[1:]).sum()]) / A
    qB = np.array([(th * Pj * (B * Gf[1:] - (Lr - L) * Gf[:-1])).sum(),
                   (Pj * ((1 + th * L + th * B) * Gc[1:] - (1 + th * Lr) * Gc[:-1])).sum()]) / A
    return dict(Y0=Y, B0=B, VY=qY @ Ys @ qY, VB=qB @ Ys @ qB, qY=qY, qB=qB, yhat=Ys)


def exact(m, s=None, nodes=160):
    """Exact annuity-measure mean, variance and ATM-implied normal vol of Y_{T0} and B_{T0}."""
    s = s or SwapSetup()
    Ys, vf, Gf, Gc, K0, KT = _ingredients(m, s)
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w)
    Lc = np.linalg.cholesky(Ys)
    zf = Lc[0, 0] * X1
    zc = Lc[1, 0] * X1 + Lc[1, 1] * X2
    lnP = (-np.multiply.outer(zf, Gf) - np.multiply.outer(zc, Gc)
           - 0.5 * (Ys[0, 0] * Gf ** 2 + 2 * Ys[0, 1] * Gf * Gc + Ys[1, 1] * Gc ** 2))
    P = s.Zfd0(s.Tj) / s.Zfd0(s.T0) * np.exp(lnP)
    Zff = s.Zff0(s.Tj) / s.Zff0(s.T0) * np.exp(-np.multiply.outer(zf - vf, Gf) - 0.5 * Ys[0, 0] * Gf ** 2)
    A, Y, B, _, _ = _rates(s, P, Zff, KT)
    EA = (W * A).sum()
    out = {}
    for name, X in (("Y", Y), ("B", B)):
        m1 = (W * A * X).sum() / EA
        m2 = (W * A * X ** 2).sum() / EA
        atm = (W * A * np.maximum(X - m1, 0)).sum() / EA
        out[name] = dict(mean=m1, var=m2 - m1 ** 2, atm_normal_vol=atm * sqrt(2 * pi / s.T0))
    return out


def approx_native(m, s=None, y=None, ccy="d"):
    """Gaussian approximation of the native swap rate Y^{e|e}, e = ccy (Remark "Native swaps").

    Returns Y0, the loadings q^e and the variance q^T y q. By default y = y^{e,e}_{T0} of the LGCM;
    in the QGCM one can pass E^A[y^{e,e}_{T0}] (frozen gradient with stochastic y).
    """
    s = s or SwapSetup()
    blk = m.idx[ccy]
    th = s.theta
    P = s.curves.Z0(ccy * 2, s.Tj) / s.curves.Z0(ccy * 2, s.T0)
    one_L = s.curves.kappa(ccy) * P[:-1] / P[1:]
    A = th * P[1:].sum()
    Y0 = ((one_L - 1) * P[1:]).sum() / A
    G = np.array([[m.G(i, s.T0, T) for T in s.Tj] for i in blk])           # (Ne, n+1)
    q = (P[1:] * ((1 + th * Y0) * G[:, 1:] - one_L * G[:, :-1])).sum(1) / A
    y = m.Y(s.T0)[np.ix_(blk, blk)] if y is None else y
    return dict(Y0=Y0, q=q, V=q @ y @ q)
