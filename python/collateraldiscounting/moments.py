"""Exact moments of the LGCM state from the moment ODEs of its linear SDE.

This is an independent benchmark for Theorem "Quasi-Gaussian Collateral Model" (deterministic
volatility case): the drifts of the theorem are integrated as ODEs for the mean and covariance of

    X = (z^{rates}, z^x, int_0^t (r^{d|d}_u - r^{d|d}_{0,u}) du, int_0^t (q_u - q_{0,u}) du),

together with the ODEs for y_t and v^{f,c}_t. No covariance integral is used for the dynamics.

`drop` switches off one term of the drift, to show that the no-arbitrage tests are sensitive to it:
"quanto_f" (quanto drift of z^f), "ycf" (term y^{c,f} 1 in the drift of z^c), "v_in_fx" (term
1^T v^{f,c} in the drift of z^x).
"""
import numpy as np
from scipy.integrate import solve_ivp


def _indices(m):
    nr = m.nr
    return nr + 3, nr, nr + 1, nr + 2   # n, iZX, iID, iIQ


def propagate(m, t, drop=None):
    """Return (mean, cov, Y, v) of X at time t under Q^{d,RN}."""
    nr, ix = m.nr, m.ix
    d, f, c = m.idx["d"], m.idx["f"], m.idx["c"]
    n, iZX, iID, iIQ = _indices(m)
    one = lambda idx: np.isin(np.arange(nr), idx).astype(float)

    def unpack(u):
        mX = u[:n]
        P = u[n:n + n * n].reshape(n, n)
        Y = u[n + n * n:n + n * n + nr * nr].reshape(nr, nr)
        v = u[n + n * n + nr * nr:]
        return mX, P, Y, v

    def rhs(_, u, k):
        mX, P, Y, v = unpack(u)
        sig = np.array([m.sig[i][k] for i in range(nr + 1)])
        be = np.array([m.beta[i][k] for i in range(nr)])
        Rs = m.R * np.outer(sig, sig)              # instantaneous covariance of sigma dW~
        S = np.zeros((n, nr + 1))
        S[np.arange(nr), np.arange(nr)] = 1
        S[iZX, ix] = 1
        A = np.zeros((n, n))
        b = np.zeros(n)
        A[np.arange(nr), np.arange(nr)] = -be
        xq = Rs[:nr, ix]                            # sigma_z^T sigma^x per factor
        b[d] = (Y @ one(d))[d]
        b[f] = (Y @ one(f))[f] - (0 if drop == "quanto_f" else 1) * xq[f]
        b[c] = (Y @ one(c))[c] + (0 if drop == "ycf" else 1) * (Y @ one(f))[c] - xq[c]
        A[iZX, d], A[iZX, f], A[iZX, c] = 1, -1, -1
        b[iZX] = -(0 if drop == "v_in_fx" else 1) * v[f].sum() - 0.5 * Rs[ix, ix]
        A[iID, d] = 1
        A[iIQ, c] = 1
        b[iIQ] = v[f].sum()
        dm = A @ mX + b
        dP = A @ P + P @ A.T + S @ Rs @ S.T
        dY = Rs[:nr, :nr] - be[:, None] * Y - Y * be[None, :]
        dv = np.zeros(nr)
        dv[f] = (Y @ one(c))[f] - be[f] * v[f]
        return np.concatenate([dm, dP.ravel(), dY.ravel(), dv])

    u = np.zeros(n + n * n + nr * nr + nr)
    for k in range(len(m.knots) - 1):
        a, bnd = m.knots[k], min(m.knots[k + 1], t)
        if bnd <= a:
            break
        u = solve_ivp(rhs, (a, bnd), u, args=(k,), method="DOP853", rtol=1e-12, atol=1e-15).y[:, -1]
    return unpack(u)


def indices(m):
    """Positions of z^x, int r^d and int q in X."""
    _, iZX, iID, iIQ = _indices(m)
    return dict(n=m.nr + 3, iZX=iZX, iID=iID, iIQ=iIQ)
