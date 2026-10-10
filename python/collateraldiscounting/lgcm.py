"""Linear Gaussian Collateral Model (LGCM) with piecewise constant parameters.

Numerical companion of docs/CollateralDiscounting.tex (sections "The Linear Gaussian Collateral
Model: Specification" and "Implementation").

Blocks: d (domestic native rates), f (foreign native rates), c (collateral basis f->d), x (FX).
Parameters are piecewise constant on `knots`; the correlated drivers W~ have constant correlation R.
The state z uses the notes' (Cheyette) convention, Definition "LGCM":
    dz = (drift - beta z) dt + sigma dW~   under Q^{d,RN}.

Main entry points
-----------------
E, Phi1, Phi2       kernels of the covariance integral algorithm (Proposition "Covariance integrals")
Model.covint        closed-form J_{s,t}[l_a, l_b] for loadings M, G or 1 (Algorithm "Covariance integrals")
Model.covint_brute  the same by nested adaptive quadrature (benchmark only)
Model.Y, Model.vfc  the deterministic y_t and v^{f,c}_t
Model.lnZ           reconstruction exponents of Z^{d|d}, Z^{f|f}, Z^{f|d} (Theorem "QGCM")
default_model       the 2+1+1+FX test model used throughout the verification remarks of the notes
"""
import math
import numpy as np
from math import factorial
from numpy.polynomial.legendre import leggauss

GL_X, GL_W = leggauss(64)
_PHI2_K = np.arange(30)
_PHI2_C = np.array([[1.0 / (factorial(m + 1) * factorial(n + 1) * (m + n + 3)) for n in range(30)] for m in range(30)])


# ---------------------------------------------------------------- scalar kernels
def E(b, D):
    """int_0^D e^{-b s} ds, stable."""
    x = b * D
    if abs(x) < 1e-8:
        return D * (1 - x / 2 + x * x / 6)
    return -math.expm1(-x) / b


def Phi1(b, D):
    """int_0^D E(b,s) ds = (D - E(b,D))/b."""
    if abs(b) * D < 0.1:
        return sum((-b) ** n * D ** (n + 2) / factorial(n + 2) for n in range(20))
    return (D - E(b, D)) / b


def Phi2(a, b, D):
    """int_0^D E(a,s) E(b,s) ds."""
    if max(abs(a), abs(b)) * D < 0.5:
        # sum_{m,n<30} (-a)^m (-b)^n D^{m+n+3} / ((m+1)! (n+1)! (m+n+3)), as u^T C v
        return D ** 3 * float(np.power(-a * D, _PHI2_K) @ _PHI2_C @ np.power(-b * D, _PHI2_K))
    if min(abs(a), abs(b), abs(a + b)) * D > 0.05:
        return (D - E(a, D) - E(b, D) + E(a + b, D)) / (a * b)
    s = 0.5 * D * (GL_X + 1)
    return 0.5 * D * sum(w * E(a, si) * E(b, si) for w, si in zip(GL_W, s))


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, knots, factors, R, sigx):
        """factors: list of (block, sig[k], beta[k]) for rate factors; FX is the last driver."""
        self.knots = np.asarray(knots, float)
        self.blocks = [f[0] for f in factors]
        self.sig = [np.asarray(f[1], float) for f in factors] + [np.asarray(sigx, float)]
        self.beta = [np.asarray(f[2], float) for f in factors] + [np.zeros(len(knots) - 1)]
        self.R = np.asarray(R, float)
        self.nr = len(factors)
        self.ix = self.nr                     # index of FX driver
        self.idx = {b: [i for i, bb in enumerate(self.blocks) if bb == b] for b in "dfc"}
        # cumulative int beta on knots
        self.Bk = [np.concatenate([[0], np.cumsum(be * np.diff(self.knots))]) for be in self.beta]

    def piece(self, t):
        return min(int(np.searchsorted(self.knots, t, side="right")) - 1, len(self.knots) - 2)

    def Bcum(self, i, t):
        k = self.piece(t)
        return self.Bk[i][k] + self.beta[i][k] * (t - self.knots[k])

    def M(self, i, t, T):
        return math.exp(-(self.Bcum(i, T) - self.Bcum(i, t)))

    def G(self, i, t, T):
        if T <= t:
            return 0.0
        pts = [t] + [k for k in self.knots if t < k < T] + [T]
        return sum(self.M(i, t, a) * E(self.beta[i][self.piece(a)], b - a) for a, b in zip(pts[:-1], pts[1:]))

    # -- loadings ell(u) for (factor, type, T)
    def load(self, i, typ, T, u):
        if typ == "1":
            return 1.0
        return self.M(i, u, T) if typ == "M" else self.G(i, u, T)

    def pq(self, i, typ, T, b):
        """ell(u) = p + q * E(beta, b-u) on a piece ending at b (T >= b)."""
        k = self.piece(b - 1e-14)
        be = self.beta[i][k]
        if typ == "1":
            return 1.0, 0.0
        if typ == "M":
            m = self.M(i, b, T)
            return m, -be * m
        g = self.G(i, b, T)
        return g, 1.0 - be * g

    def covint(self, la, lb, s, t):
        """int_s^t rho sig_a sig_b ell_a(u) ell_b(u) du, closed form (item 3)."""
        (i, ta, Ta), (j, tb, Tb) = la, lb
        if t <= s:
            return 0.0
        pts = [s] + [k for k in self.knots if s < k < t] + [t]
        tot = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            k = self.piece(0.5 * (a + b))
            D = b - a
            pa, qa = self.pq(i, ta, Ta, b)
            pb, qb = self.pq(j, tb, Tb, b)
            ba, bb = self.beta[i][k], self.beta[j][k]
            val = pa * pb * D + pa * qb * Phi1(bb, D) + qa * pb * Phi1(ba, D) + qa * qb * Phi2(ba, bb, D)
            tot += self.R[i, j] * self.sig[i][k] * self.sig[j][k] * val
        return tot

    def covint_brute(self, la, lb, s, t):
        from scipy.integrate import quad
        (i, ta, Ta), (j, tb, Tb) = la, lb
        pts = [s] + [k for k in self.knots if s < k < t] + [t]
        tot = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            k = self.piece(0.5 * (a + b))

            def Gb(ii, u, T):  # brute-force G via quadrature of M
                return quad(lambda v: self.M(ii, u, v), u, T, limit=200,
                            points=[kk for kk in self.knots if u < kk < T][:50], epsabs=1e-14, epsrel=1e-13)[0]

            def L(ii, typ, T, u):
                return 1.0 if typ == "1" else (self.M(ii, u, T) if typ == "M" else Gb(ii, u, T))
            f = lambda u: L(i, ta, Ta, u) * L(j, tb, Tb, u)
            tot += self.R[i, j] * self.sig[i][k] * self.sig[j][k] * quad(f, a, b, epsabs=1e-15, epsrel=1e-12)[0]
        return tot

    # -- sums of covints for vector loadings: list of (coef, factor, type, T)
    def I(self, A, B, s, t):
        return sum(ca * cb * self.covint((i, ta, Ta), (j, tb, Tb), s, t)
                   for ca, i, ta, Ta in A for cb, j, tb, Tb in B)

    def Sig(self, block, T, coef=1.0):
        """loading of Sigma^{block}_{u,T}."""
        return [(coef, i, "G", T) for i in self.idx[block]]

    def SigDF(self, T):
        return self.Sig("d", T) + self.Sig("f", T, -1) + self.Sig("c", T, -1) + [(1.0, self.ix, "1", 0.0)]

    # -- deterministic y and v^{fc}
    def Y(self, t):
        n = self.nr
        return np.array([[self.covint((i, "M", t), (j, "M", t), 0, t) for j in range(n)] for i in range(n)])

    def vfc(self, t):
        v = np.zeros(self.nr)
        for i in self.idx["f"]:
            v[i] = sum(self.covint((i, "M", t), (j, "G", t), 0, t) for j in self.idx["c"])
        return v

    def Gvec(self, t, T):
        return np.array([self.G(i, t, T) for i in range(self.nr)])

    def mask(self, *blocks):
        m = np.zeros(self.nr)
        for b in blocks:
            m[self.idx[b]] = 1
        return m

    # -- reconstruction exponents: ln(Z_{t,T}/Z_{0,t,T}) = a.z + c  (z = rate state, Cheyette)
    def lnZ(self, kind, t, T, Y=None, v=None):
        Y = self.Y(t) if Y is None else Y
        v = self.vfc(t) if v is None else v
        G = self.Gvec(t, T)
        if kind == "dd":
            g = G * self.mask("d")
            return -g, -0.5 * g @ Y @ g
        if kind == "ff":
            g = G * self.mask("f")
            return -g, -0.5 * g @ Y @ g
        if kind == "fd":   # Z^{f|d} = Z^{f|f} Q, state (z^f + v^{fc}, z^c)
            g = G * self.mask("f", "c")
            return -g, -g @ v - 0.5 * g @ Y @ g
        raise ValueError(kind)


def default_model(scale=1.0):
    knots = [0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 60]
    n = len(knots) - 1
    rng = np.random.default_rng(7)
    def ts(base):
        return base * (1 + 0.3 * rng.uniform(-1, 1, n))
    factors = [
        ("d", scale * ts(0.010), ts(0.05)),                       # d1
        ("d", scale * ts(0.008), np.r_[0.0, 1e-7, ts(0.6)[2:]]), # d2: zero and tiny mean reversion
        ("f", scale * ts(0.012), np.r_[ts(0.03)[:4], -0.02 * np.ones(n - 4)]),  # negative MR at long end
        ("c", scale * ts(0.004), ts(0.15)),
    ]
    R = np.array([[1.0, 0.3, 0.6, -0.2, 0.25],
                  [0.3, 1.0, 0.2, 0.1, -0.1],
                  [0.6, 0.2, 1.0, -0.4, 0.3],
                  [-0.2, 0.1, -0.4, 1.0, 0.35],
                  [0.25, -0.1, 0.3, 0.35, 1.0]])
    assert np.all(np.linalg.eigvalsh(R) > 0)
    sigx = scale * ts(0.12)
    return Model(knots, factors, R, sigx)
