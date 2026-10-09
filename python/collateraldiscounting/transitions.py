"""Exact Gaussian one-step transitions X_t = F X_s + mu + eps of the LGCM state.

Implements Propositions "Exact transition under the risk-neutral measure" and "Change of numeraire"
of docs/CollateralDiscounting.tex. X = (z^{rates}, z^x, int(r^d - r^d_0), int(q - q_0), L) where L is
the stochastic part of the log jumping numeraire. measure in {"RN", "Tfwd", "jump"}.
"""
import numpy as np
def transition(m, s, t, measure="RN", Tstar=None):
    nr = m.nr; n = nr + 4; iZX, iID, iIQ, iL = nr, nr + 1, nr + 2, nr + 3
    d, f, c = m.idx["d"], m.idx["f"], m.idx["c"]
    Ys, vs = m.Y(s), m.vfc(s); G = m.Gvec(s, t)
    F = np.eye(n); mu = np.zeros(n)
    for i in range(nr): F[i, i] = m.M(i, s, t)
    # noise loadings eps_k = int ell_k . dW^{RN}
    ell = {i: [(1.0, i, "M", t)] for i in range(nr)}
    ell[iZX] = m.SigDF(t); ell[iID] = m.Sig("d", t); ell[iIQ] = m.Sig("c", t); ell[iL] = []
    # RN means of rate factors
    for i in range(nr):
        blk = m.blocks[i]
        src = {"d": d, "f": f, "c": c + f}[blk]
        mu[i] = sum(m.covint((i, "M", t), (j, "G", t), s, t) for j in src) + m.M(i, s, t) * sum(Ys[i, j] * G[j] for j in src)
        if blk in "fc":
            mu[i] -= m.covint((i, "M", t), (m.ix, "1", 0.0), s, t)
    gd, gfc = G * m.mask("d"), G * m.mask("f", "c")
    Cxx = m.I(ell[iZX], ell[iZX], s, t)
    # FX: exact under Q^{d|d,t}, then shift to RN
    F[iZX, :nr] += gd - gfc
    mu[iZX] = -gfc @ vs - 0.5 * gfc @ Ys @ gfc + 0.5 * gd @ Ys @ gd - 0.5 * Cxx + m.I(ell[iZX], m.Sig("d", t), s, t)
    # int (r^d - r^d_0)
    F[iID, :nr] += gd; mu[iID] = 0.5 * gd @ Ys @ gd + 0.5 * m.I(m.Sig("d", t), m.Sig("d", t), s, t)
    # int (q - q_0)
    gc, gf = G * m.mask("c"), G * m.mask("f")
    F[iIQ, :nr] += gc
    mu[iIQ] = (gf @ vs + 0.5 * gc @ Ys @ gc + gc @ Ys @ gf
               + 0.5 * m.I(m.Sig("c", t), m.Sig("c", t), s, t) + m.I(m.Sig("f", t), m.Sig("c", t), s, t)
               - m.I(m.Sig("c", t), [(1.0, m.ix, "1", 0.0)], s, t))
    # log jumping numeraire accumulator (stochastic part of -ln Z^{dd}_{s,t})
    F[iL, :nr] += gd; mu[iL] = 0.5 * gd @ Ys @ gd
    # change of measure: kernel Sigma^d_{.,T}
    T = {"RN": None, "Tfwd": Tstar, "jump": t}[measure]
    if T is not None:
        for k in range(n):
            mu[k] -= m.I(ell[k], m.Sig("d", T), s, t)
    C = np.array([[m.I(ell[a], ell[b], s, t) for b in range(n)] for a in range(n)])
    return F, mu, C, dict(iZX=iZX, iID=iID, iIQ=iIQ, iL=iL, n=n)

def propagate(m, grid, measure, Tstar=None):
    F, mu, C, ix = transition(m, grid[0], grid[1], measure, Tstar)
    mX = np.zeros(ix["n"]); P = np.zeros((ix["n"],) * 2)
    for s, t in zip(grid[:-1], grid[1:]):
        F, mu, C, ix = transition(m, s, t, measure, Tstar)
        mX = F @ mX + mu; P = F @ P @ F.T + C
    return mX, P, ix
