"""Monte Carlo discretisation schemes for the QGCM / LGCM.

A scheme generates the terminal state (a `SimResult`) of a `Simulator` under Q^{d,RN}. All paths
are simulated at once with numpy arrays of shape (n_paths, ...).

EulerScheme  Euler scheme for the SDE of Theorem "Quasi-Gaussian Collateral Model", with an
             exponential integrator for the mean reversion and trapezoidal integrals of the short
             rates. y_t and v^{f,c}_t are simulated alongside the state, so stochastic (local)
             volatility is supported. Discretisation bias is O(dt).
ExactScheme  LGCM only: exact Gaussian transitions of Proposition "Exact transition under the
             risk-neutral measure" (transitions.py). No discretisation bias for any step size.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from . import transitions


@dataclass
class SimResult:
    """Terminal values at time T, one row per path (antithetic pairs adjacent)."""
    T: float
    z: np.ndarray        # (P, nr)  rates factors (Cheyette convention)
    zx: np.ndarray       # (P,)     log FX moneyness z^x = ln(x_T / x^{df|d}_{0,T})
    Y: np.ndarray        # (P, nr, nr) covariance process y_T (deterministic for the LGCM)
    v: np.ndarray        # (P, nr)  v^{f,c}_T (non zero on f factors only)
    Id: np.ndarray       # (P,)     int_0^T (r^{d|d}_u - r^{d|d}_{0,u}) du
    Iq: np.ndarray       # (P,)     int_0^T (q_u - q_{0,u}) du
    antithetic: bool = True

    @property
    def n_paths(self):
        return self.z.shape[0]


class PathStates(dict):
    """SimResult by date, the same paths at every date. states(t) looks the date up rounded to 1e-10."""

    @staticmethod
    def key(t):
        return round(float(t), 10)

    def __call__(self, t):
        return self[self.key(t)]


class MCScheme(ABC):
    """Base class: time grid and (antithetic) Gaussian draws."""

    def __init__(self, dt):
        self.dt = dt

    def grid(self, model, T):
        """Uniform grid of step <= dt, refined to contain the parameter knots."""
        knots = [k for k in model.knots if 0 < k < T]
        g = np.union1d(np.linspace(0, T, max(int(np.ceil(T / self.dt)), 1) + 1), knots)
        return g[np.concatenate([[True], np.diff(g) > 1e-12])]

    @staticmethod
    def normals(rng, n_paths, dim, antithetic):
        if antithetic:
            h = rng.standard_normal((n_paths // 2, dim))
            return np.stack([h, -h], axis=1).reshape(-1, dim)
        return rng.standard_normal((n_paths, dim))

    @abstractmethod
    def simulate(self, sim, T, n_paths, rng, antithetic) -> SimResult:
        """Simulate the state of `sim` (a Simulator) from 0 to T."""

    def simulate_dates(self, sim, dates, n_paths, rng, antithetic) -> PathStates:
        """Joint state at every date of `dates` along the same paths."""
        raise NotImplementedError(f"{type(self).__name__} does not record intermediate dates")


class EulerScheme(MCScheme):
    """Euler scheme for the QGCM (and the LGCM as kappa = 0).

    y_mode="frozen" (diagnostic) replaces the simulated y_t, v_t by their deterministic LGCM values
    in the drifts and in the reconstruction, i.e. ignores the extra QGCM state variables. It is only
    correct for kappa = 0.
    """

    def __init__(self, dt=0.02, y_mode="simulated"):
        super().__init__(dt)
        if y_mode not in ("simulated", "frozen"):
            raise ValueError(y_mode)
        self.y_mode = y_mode

    def simulate(self, sim, T, P, rng, antithetic):
        m, nr, ix = sim.m, sim.m.nr, sim.m.ix
        d, f, c = m.idx["d"], m.idx["f"], m.idx["c"]
        sd = np.isin(np.arange(nr), d).astype(float)
        sf = np.isin(np.arange(nr), f).astype(float)
        sc = np.isin(np.arange(nr), c).astype(float)
        # drift selectors: z^d_i <- sum_{j in d} y_ij ; z^f_i <- sum_{j in f} y_ij ; z^c_i <- sum_{j in c u f} y_ij
        S = np.zeros((nr, nr))
        for i in range(nr):
            S[i, {"d": d, "f": f, "c": c + f}[m.blocks[i]]] = 1.0
        quanto = (sf + sc) * m.R[:nr, ix]           # rho^{(a,i),x} on f and c factors
        Rr = m.R[:nr, :nr]

        z = np.zeros((P, nr)); zx = np.zeros(P); Y = np.zeros((P, nr, nr)); v = np.zeros((P, nr))
        Id = np.zeros(P); Iq = np.zeros(P)
        grid = self.grid(m, T)
        frozen = self.y_mode == "frozen"
        if frozen:
            Yf = {t: m.Y(t) for t in grid}
            vf = {t: m.vfc(t) for t in grid}
        for t0, t1 in zip(grid[:-1], grid[1:]):
            h = t1 - t0
            k = m.piece(0.5 * (t0 + t1))
            if frozen:
                Y = np.broadcast_to(Yf[t0], (P, nr, nr))
                v = np.broadcast_to(vf[t0], (P, nr))
            sbar = np.array([m.sig[i][k] for i in range(nr)])
            sx = m.sig[ix][k]
            be = np.array([m.beta[i][k] for i in range(nr)])
            sig = sim.local_vol(sbar, z)                                         # (P, nr)
            dW = self.normals(rng, P, nr + 1, antithetic) @ sim.chol.T * np.sqrt(h)
            decay = np.exp(-be * h)
            drift = np.einsum("pij,ij->pi", Y, S) - quanto * sig * sx            # (P, nr)
            vdrift = np.einsum("pij,j->pi", Y, sc) * sf                          # sum_{j in c} y_ij on f rows
            xdrift_old = z @ (sd - sf - sc) - v @ sf
            iq_old = z @ sc + v @ sf
            id_old = z @ sd
            z_new = decay * z + drift * h + sig * dW[:, :nr]
            v_new = decay * v + vdrift * h
            Y = np.exp(-(be[:, None] + be[None, :]) * h) * Y + Rr * np.einsum("pi,pj->pij", sig, sig) * h
            xdrift_new = z_new @ (sd - sf - sc) - v_new @ sf
            zx = zx + 0.5 * (xdrift_old + xdrift_new) * h - 0.5 * sx * sx * h + sx * dW[:, nr]
            Id = Id + 0.5 * (id_old + z_new @ sd) * h
            Iq = Iq + 0.5 * (iq_old + z_new @ sc + v_new @ sf) * h
            z, v = z_new, v_new
        if frozen:
            Y, v = np.broadcast_to(Yf[T], (P, nr, nr)), np.broadcast_to(vf[T], (P, nr))
        return SimResult(T, z, zx, Y, v, Id, Iq, antithetic)


class ExactScheme(MCScheme):
    """Exact Gaussian transitions of the LGCM (kappa = 0 only)."""

    def __init__(self, dt=0.25):
        super().__init__(dt)

    def simulate(self, sim, T, P, rng, antithetic):
        if not sim.is_lgcm:
            raise ValueError("the exact scheme only applies to the LGCM (kappa = 0)")
        m, nr = sim.m, sim.m.nr
        X = None
        for t0, t1 in zip(*(lambda g: (g[:-1], g[1:]))(self.grid(m, T))):
            F, mu, C, ixs = transitions.transition(m, t0, t1, "RN")
            w, U = np.linalg.eigh(C)                 # C may be singular (no noise on L)
            root = U * np.sqrt(np.clip(w, 0, None))
            if X is None:
                X = np.zeros((P, ixs["n"]))
            X = X @ F.T + mu + self.normals(rng, P, ixs["n"], antithetic) @ root.T
        Yd = np.broadcast_to(m.Y(T), (P, nr, nr))
        vd = np.broadcast_to(m.vfc(T), (P, nr))
        return SimResult(T, X[:, :nr], X[:, ixs["iZX"]], Yd, vd, X[:, ixs["iID"]], X[:, ixs["iIQ"]], antithetic)

    def simulate_dates(self, sim, dates, P, rng, antithetic):
        """Joint state at every date of `dates` (>= 0) along the same paths. The grid is the grid of
        `simulate` up to the last date refined with the dates; the transitions are exact, so the law
        at each date is that of `simulate` and the joint law is that of the Gaussian Markov state.
        The state layout (z, z^x, int r^d, int q) is read from the indices of transitions.transition."""
        if not sim.is_lgcm:
            raise ValueError("the exact scheme only applies to the LGCM (kappa = 0)")
        dates = np.unique(np.asarray(dates, float))
        if dates.size == 0:
            raise ValueError("simulate_dates needs at least one date")
        if dates[0] < 0:
            raise ValueError("dates must be >= 0")
        m, nr = sim.m, sim.m.nr
        want = {PathStates.key(t) for t in dates}
        g = self.grid(m, max(dates[-1], 1e-10))
        near = np.abs(g[:, None] - dates[None, :]).min(1) < 1e-10      # grid points that are dates up to 1e-10
        g = np.union1d(g[~near], dates)
        g = g[np.concatenate([[True], np.diff(g) > 1e-10])]
        steps = [transitions.transition(m, t0, t1, "RN") for t0, t1 in zip(g[:-1], g[1:])]
        # only t = 0 requested (no step): the layout is that of any transition
        ixs = (steps[0] if steps else transitions.transition(m, 0.0, 1e-10, "RN"))[3]
        X = np.zeros((P, ixs["n"]))
        out = PathStates()

        def record(t, X):
            Yd = np.broadcast_to(m.Y(t), (P, nr, nr))
            vd = np.broadcast_to(m.vfc(t), (P, nr))
            out[PathStates.key(t)] = SimResult(float(t), X[:, :nr].copy(), X[:, ixs["iZX"]].copy(), Yd, vd,
                                               X[:, ixs["iID"]].copy(), X[:, ixs["iIQ"]].copy(), antithetic)
        if 0.0 in want:
            record(0.0, X)
        for t1, (F, mu, C, _) in zip(g[1:], steps):
            w, U = np.linalg.eigh(C)                 # C may be singular (no noise on L)
            X = X @ F.T + mu + self.normals(rng, P, ixs["n"], antithetic) @ (U * np.sqrt(np.clip(w, 0, None))).T
            if PathStates.key(t1) in want:
                record(t1, X)
        return out
