"""Exact Gaussian benchmarks that do not use the covariance-kernel formulas of market.py.

coupon_value_exact: K_t = E^{f|d,b}_t[C] / E^{f|f,b}_t[C] for a coupon factor C on [a, b] paid at b,
computed from the joint Gaussian law of (state, log deflator, log C) obtained by stepping the exact
transitions (transitions.transition, measure "RN", i.e. under Q^{d,RN}) over the fixing grid of the
coupon, with ln C = sum_k -ln Z^{f|f}_{t_k,t_{k+1}} accumulated as an extra state (each term is affine
in z_{t_k}, Model.lnZ("ff", t_k, t_{k+1})). The value at t of the f payment C at b under collateral
col is E^{d,RN}_t[D_{t,b} C] with D_{t,b} the f-payment deflator of montecarlo.Simulator.deflator,
    D_{t,b} = exp(-int_t^b r^{d|d}) [exp(+int_t^b q) if col = f] x_b / x_t,
built from the initial curves and the state increments; E[exp(w.X)] = exp(w.mean + w.P.w/2). Dividing by
Z^{f|col}_{t,b} gives E^{f|col,b}_t[C].

Conditioning on F_t. The exact transitions from t use only the model parameters after t and the
deterministic y_s, v^{f,c}_s (s >= t), and the state enters linearly: E_t[exp(w.X_b)] =
exp(affine(z_t)). The state at t is an argument (default z_t = 0); the accumulators (z^x, int r^d,
int q, L, ln C) start at 0 because only their increments over [t, b] enter. The curves are the
curves seen from 0 (Z_{0,t,T} = Z_{0,T}/Z_{0,t}), consistent with the reconstruction Model.lnZ. That K_t
does not depend on z_t (it is deterministic in the LGCM) is checked numerically in the tests by
passing a non-zero z_t.

Partially accrued periods (a < t < b): the factors 1/Z^{f|f}_{t_k,t_{k+1}} with t_k <= t are
F_t-measurable, so they are known at t and cancel in K_t; only the fixings t_k > t are random and are
the ones accumulated here. The Libor coupon uses the one-step grid [a, b] (its kappa is kept outside
K). The continuous coupon is not benchmarked here: it is the step -> 0 limit of the daily coupon,
tested by convergence.
"""
import numpy as np

from . import transitions


def _remaining_fixings(coupon, a, b, t):
    g = coupon.grid(a, b)
    return [float(s) for s in g[:-1] if s > t]


def coupon_expectation_exact(m, curves, coupon, a, b, t=0.0, collateral="d", z0=None):
    """E^{f|collateral,b}_t[C_t] with C_t the part of the coupon factor not yet fixed at t, given the
    rates state z_t = z0 (default 0), by state augmentation of the exact transitions."""
    if collateral not in ("d", "f"):
        raise ValueError(collateral)
    fix = _remaining_fixings(coupon, a, b, t)
    if not fix:
        return 1.0
    nr = m.nr
    zt = np.zeros(nr) if z0 is None else np.asarray(z0, float)
    pts = [t] + fix + [b]
    n0 = nr + 4
    iS = n0                                   # ln C accumulator
    mX = np.zeros(n0 + 1); mX[:nr] = zt
    P = np.zeros((n0 + 1, n0 + 1))
    lnC_det = 0.0                             # initial-curve part of ln C
    for s, u in zip(pts[:-1], pts[1:]):
        if s in fix:                          # fixing at s of the step [s, u]: ln C += -ln Z^{f|f}_{s,u}
            g, c = m.lnZ("ff", s, u)
            A = np.eye(n0 + 1); A[iS, :nr] -= g
            mX = A @ mX; mX[iS] -= c; P = A @ P @ A.T
            lnC_det -= np.log(curves.Z0("ff", u) / curves.Z0("ff", s))
        F, mu, C, ix = transitions.transition(m, s, u, "RN")
        Fe = np.eye(n0 + 1); Fe[:n0, :n0] = F
        mue = np.r_[mu, 0.0]
        Ce = np.zeros((n0 + 1, n0 + 1)); Ce[:n0, :n0] = C
        mX = Fe @ mX + mue; P = Fe @ P @ Fe.T + Ce
    # log of D_{t,b} C: deterministic part from the curves, stochastic part w.X_b
    w = np.zeros(n0 + 1)
    w[ix["iZX"]] += 1.0; w[ix["iID"]] -= 1.0; w[iS] += 1.0
    det = curves.Z0("dd", b) / curves.Z0("dd", t) * curves.fx_forward0(b) / curves.fx_forward0(t)
    if collateral == "f":
        w[ix["iIQ"]] += 1.0
        det *= curves.Z0("qz", t) / curves.Z0("qz", b)
    value = det * np.exp(lnC_det + w @ mX + 0.5 * w @ P @ w)
    kind = "f" + collateral
    g, c = m.lnZ(kind, t, b)
    Ztb = curves.Z0(kind, b) / curves.Z0(kind, t) * np.exp(g @ zt + c)
    return float(value / Ztb)


def coupon_value_exact(m, curves, coupon, a, b, t=0.0):
    """K_t = E^{f|d,b}_t[C] / E^{f|f,b}_t[C], with the native expectation given by telescoping
    (model free): E^{f|f,b}_t[C_t] = Z^{f|f}_{t,a'} / Z^{f|f}_{t,b}, a' the first fixing after t (z_t = 0)."""
    fix = _remaining_fixings(coupon, a, b, t)
    if not fix:
        return 1.0
    ga, ca = m.lnZ("ff", t, fix[0])
    gb, cb = m.lnZ("ff", t, b)
    native = curves.Z0("ff", fix[0]) / curves.Z0("ff", b) * np.exp(ca - cb)
    return coupon_expectation_exact(m, curves, coupon, a, b, t, "d") / float(native)
