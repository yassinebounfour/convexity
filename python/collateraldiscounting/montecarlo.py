"""Vectorised Monte Carlo simulator for the QGCM and the LGCM.

The `Simulator` holds the model, the QGCM volatility specification and the reconstruction formulas;
path generation is delegated to an `MCScheme` (schemes.py): `EulerScheme` (QGCM and LGCM) or
`ExactScheme` (LGCM). Simulation is under Q^{d,RN}; payoffs are valued with the deflators below
(see payoffs.py).

QGCM volatility specification (separable, so the theorem applies): for each rates factor
    sigma^{a,i}_t = sigmabar^{a,i}_t * max(1 + kappa^{a,i} z^{a,i}_t, floor),
where sigmabar is the model's piecewise constant spot vol. kappa = 0 gives the LGCM. The FX spot
vol is deterministic.
"""
import numpy as np

from .schemes import EulerScheme, ExactScheme, MCScheme, SimResult   # noqa: F401  (re-exported)


class Simulator:
    def __init__(self, model, kappa=None, floor=0.05, scheme=None):
        self.m = model
        nr = model.nr
        self.kappa = np.zeros(nr) if kappa is None else np.broadcast_to(np.asarray(kappa, float), (nr,)).copy()
        self.floor = floor
        self.chol = np.linalg.cholesky(model.R)
        self.is_lgcm = not np.any(self.kappa)
        self.scheme = scheme if scheme is not None else EulerScheme()

    def local_vol(self, sbar, z):
        """sigma^{a,i}_t on every path, shape (P, nr)."""
        return sbar * np.maximum(1.0 + self.kappa * z, self.floor)

    def simulate(self, T, n_paths, seed=0, antithetic=True):
        if antithetic and n_paths % 2:
            raise ValueError("n_paths must be even with antithetic sampling")
        return self.scheme.simulate(self, T, n_paths, np.random.default_rng(seed), antithetic)

    # ------------------------------------------------------------------ reconstruction
    def lnZ(self, res, kind, T):
        """ln(Z^{kind}_{t,T} / Z^{kind}_{0,t,T}) on every path, kind in {"dd", "ff", "fd"} (Theorem QGCM)."""
        m = self.m
        g = m.Gvec(res.T, T) * {"dd": m.mask("d"), "ff": m.mask("f"), "fd": m.mask("f", "c")}[kind]
        out = -res.z @ g - 0.5 * np.einsum("i,pij,j->p", g, res.Y, g)
        if kind == "fd":
            out -= res.v @ g
        return out

    def discount(self, res, curves, kind, T):
        """Z^{kind}_{t,T} on every path (t = res.T), kind in {"dd", "ff", "fd"}."""
        return curves.Z0(kind, T) / curves.Z0(kind, res.T) * np.exp(self.lnZ(res, kind, T))

    def fx_spot(self, res, curves):
        return curves.fx_forward0(res.T) * np.exp(res.zx)

    def deflator(self, res, curves, ccy, collateral):
        """D such that the time-0 value, in currency `ccy`, of a payment V_T in `ccy` at T = res.T under
        `collateral` is E^{d,RN}[D V_T].

        The d-value of the payment accrues at r^{d|d} (d collateral) or r^{d|f} = r^{d|d} - q^{f->d}
        (f collateral), and f payments are converted at x_T and back at x_0.
        """
        T = res.T
        D = curves.Z0("dd", T) * np.exp(-res.Id)                     # exp(-int r^{d|d})
        if collateral == "f":
            D = D / curves.Z0("qz", T) * np.exp(res.Iq)              # * exp(+int q)
        elif collateral != "d":
            raise ValueError(collateral)
        if ccy == "f":
            D = D * self.fx_spot(res, curves) / curves.x0
        elif ccy != "d":
            raise ValueError(ccy)
        return D


# ---------------------------------------------------------------------- estimators
def mean_and_stderr(x, antithetic=True):
    """Mean and standard error, averaging antithetic pairs first."""
    if antithetic:
        x = 0.5 * (x[0::2] + x[1::2])
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def ratio_and_stderr(num, den, antithetic=True, n_batches=50):
    """E[num]/E[den] with a batch-means standard error."""
    if antithetic:
        num, den = 0.5 * (num[0::2] + num[1::2]), 0.5 * (den[0::2] + den[1::2])
    nb = np.array_split(np.arange(len(num)), n_batches)
    r = np.array([num[b].sum() / den[b].sum() for b in nb])
    return num.sum() / den.sum(), r.std(ddof=1) / np.sqrt(n_batches)
