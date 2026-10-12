"""Terminal correlations of log-FX and swap (CMS) rates in the LGCM: Gaussian approximation, exact benchmark
and Monte Carlo estimator.

Implements section "Correlations of FX and swap rates" of docs/CollateralDiscounting.tex.

Observables (each a function of the LGCM state at its date):
    LogFX(T)                                  ln x_T = ln x^{df|d}_{0,T} + z^x_T  (exactly affine in the state)
    SwapRate(ccy, T0, tenor, ...)             native forward swap rate Y^{e|e}_{T0} fixing at T0 (the CMS fixing),
                                              e = "d" or "f", an explicit function of z^e_{T0}
Any pair can be correlated: log-FX vs CMS at the same expiry, swap rates of different tenors at the same
expiry, and intertemporal pairs (different dates), in one or both currencies.

* correlation_approx: Gaussian approximation (Proposition "Gaussian approximation of terminal correlations").
  A swap rate is replaced by its frozen-weights linearisation Y0 + q . z_{T0} (the gradient of the swaption
  approximation); the state covariances across dates are closed-form covariance integrals,
      Cov(z^i_{T1}, z^j_{T2}) = int_0^{T1^T2} rho sig_i sig_j M_i(u,T1) M_j(u,T2) du,
      Cov(z^x_{T1}, z^j_{T2}) = int_0^{T1^T2} Sigma^{df}_{u,T1} . (M_j(u,T2) sig_j e_j) du,
      Cov(z^x_{T1}, z^x_{T2}) = int_0^{T1^T2} Sigma^{df}_{u,T1} . Sigma^{df}_{u,T2} du.
  The result does not depend on the measure (the measures differ by deterministic drifts).
* correlation_exact: exact correlation under Q^{d,RN} ("RN") or the domestic T*-forward measure ("Tfwd"):
  the joint Gaussian law of the state components at the dates (transitions.transition, independent of the
  covariance integrals above), then tensor Gauss-Hermite quadrature of the observables.
* correlation_mc: Monte Carlo with the exact LGCM scheme under Q^{d,RN}, reweighted by D_t Z^{d|d}_{t,T*} / Z_{0,T*}
  for the T*-forward measure; standard error by the delta method (influence function of the correlation,
  antithetic pairs averaged first).
"""
from dataclasses import dataclass
from math import sqrt

import numpy as np

from . import transitions
from .instruments import schedule
from .montecarlo import Simulator
from .schemes import ExactScheme


# ---------------------------------------------------------------------- observables
@dataclass(frozen=True)
class LogFX:
    """ln x^{df}_T: log FX spot (units of domestic per foreign) at T."""
    expiry: float

    @property
    def date(self):
        return float(self.expiry)

    @property
    def label(self):
        return f"ln x({self.expiry:g}Y)"

    def components(self, model):
        """State components it depends on: "x" = z^x."""
        return ["x"]

    def value(self, model, curves, state):
        return np.log(curves.fx_forward0(self.expiry)) + state["x"]

    def gradient(self, model, curves):
        """Loadings on (z_1..z_nr, z^x) at the date (exact: ln x is affine in the state)."""
        g = np.zeros(model.nr + 1)
        g[-1] = 1.0
        return g


@dataclass(frozen=True)
class SwapRate:
    """Native forward swap rate of currency ccy ("d" or "f") fixing at `expiry`, on [expiry, expiry + tenor]:
    fixed leg every fixed_tenor, floating leg every float_tenor with the curves' funding spread kappa,
    Y = sum_i (kappa Z_{T0,t_{i-1}} - Z_{T0,t_i}) / sum_k tau_k Z_{T0,t_k}."""
    ccy: str
    expiry: float
    tenor: float
    fixed_tenor: float = 1.0
    float_tenor: float = 0.25

    def __post_init__(self):
        if self.ccy not in ("d", "f"):
            raise ValueError(f"ccy must be 'd' or 'f' (native swap rates), got {self.ccy!r}")
        if self.expiry <= 0:
            raise ValueError("the swap rate must fix after 0")

    @property
    def date(self):
        return float(self.expiry)

    @property
    def label(self):
        return f"Y{self.ccy}({self.expiry:g}Y x {self.tenor:g}Y)"

    def components(self, model):
        return list(model.idx[self.ccy])

    def cashflows(self, curves):
        T0, T1 = self.expiry, self.expiry + self.tenor
        tf = schedule(T0, T1, self.float_tenor)
        cf = np.zeros(len(tf))
        cf[:-1] += curves.kappa(self.ccy)
        cf[1:] -= 1.0
        tx = schedule(T0, T1, self.fixed_tenor)[1:]
        return tf, cf, tx, np.full(len(tx), self.fixed_tenor)

    def _exponents(self, model, t):
        """ln(Z_{T0,t} / Z_{0,T0,t}) = a . z + c for each date t (LGCM reconstruction, Theorem QGCM)."""
        T0, kind = self.expiry, self.ccy * 2
        Y, v = model.Y(T0), model.vfc(T0)
        ac = [model.lnZ(kind, T0, s, Y, v) for s in t]
        return np.array([a for a, _ in ac]), np.array([c for _, c in ac])

    def value(self, model, curves, state):
        """Y on each row of the rate state z (N, nr) at the expiry."""
        tf, cf, tx, wx = self.cashflows(curves)
        kind = self.ccy * 2
        out = []
        for t, coef in ((tf, cf), (tx, wx)):
            P = curves.Z0(kind, t) / curves.Z0(kind, self.expiry)
            a, c = self._exponents(model, t)
            out.append((np.exp(state["z"] @ a.T + c) * (coef * P)).sum(1))
        return out[0] / out[1]

    def gradient(self, model, curves):
        """Frozen-weights gradient q on (z, z^x) at the expiry: dY/dz at z = 0 with the weights of the initial
        forward curve, q = (-sum cf G P + Y0 sum w G P) / A (as in pricers.native_swaption_variance)."""
        tf, cf, tx, wx = self.cashflows(curves)
        kind, T0 = self.ccy * 2, self.expiry
        Pf, Px = curves.Z0(kind, tf) / curves.Z0(kind, T0), curves.Z0(kind, tx) / curves.Z0(kind, T0)
        A = (wx * Px).sum()
        Y0 = (cf * Pf).sum() / A
        mask = model.mask(self.ccy)
        Gf = np.array([model.Gvec(T0, t) * mask for t in tf])          # (n_f, nr)
        Gx = np.array([model.Gvec(T0, t) * mask for t in tx])
        q = (-(cf * Pf) @ Gf + Y0 * (wx * Px) @ Gx) / A
        return np.r_[q, 0.0]

    def forward(self, curves):
        tf, cf, tx, wx = self.cashflows(curves)
        Z = lambda t: curves.Z0(self.ccy * 2, t)
        return float((cf * Z(tf)).sum() / (wx * Z(tx)).sum())


# ---------------------------------------------------------------------- Gaussian approximation
def _loadings(model, T):
    """Noise loadings of (z_1..z_nr, z^x) at T, as lists of (coef, factor, type, T) for model.I."""
    return [[(1.0, i, "M", T)] for i in range(model.nr)] + [model.SigDF(T)]


def state_covariance(model, T1, T2):
    """C[a, b] = Cov(s_a(T1), s_b(T2)), s = (z_1..z_nr, z^x), in closed form (covariance integrals)."""
    l1, l2 = _loadings(model, T1), _loadings(model, T2)
    t = min(T1, T2)
    return np.array([[model.I(a, b, 0.0, t) for b in l2] for a in l1])


@dataclass
class CorrelationResult:
    corr: float
    cov: float
    var1: float
    var2: float
    stderr: float = float("nan")

    @property
    def vol1(self):
        return sqrt(self.var1)

    @property
    def vol2(self):
        return sqrt(self.var2)


def correlation_approx(model, curves, X1, X2):
    """Gaussian approximation of Corr(X1, X2) (measure independent)."""
    g1, g2 = X1.gradient(model, curves), X2.gradient(model, curves)
    c12 = float(g1 @ state_covariance(model, X1.date, X2.date) @ g2)
    v1 = float(g1 @ state_covariance(model, X1.date, X1.date) @ g1)
    v2 = float(g2 @ state_covariance(model, X2.date, X2.date) @ g2)
    return CorrelationResult(c12 / sqrt(v1 * v2), c12, v1, v2)


# ---------------------------------------------------------------------- exact benchmark
def joint_law(model, dates, measure="RN", Tstar=None):
    """Mean and covariance of the stacked state (X_{t_1}, ..., X_{t_K}) of transitions.transition at the
    sorted distinct dates, under measure "RN" or "Tfwd" (Tstar >= the last date). Exact for any step."""
    dates = sorted(set(float(t) for t in dates))
    grid = [0.0] + dates
    K = len(dates)
    means, covs = [], {}
    mX, P = None, None
    for k, (s, t) in enumerate(zip(grid[:-1], grid[1:])):
        F, mu, C, ix = transitions.transition(model, s, t, measure, Tstar)
        if mX is None:
            mX, P = np.zeros(ix["n"]), np.zeros((ix["n"], ix["n"]))
        mX, P = F @ mX + mu, F @ P @ F.T + C
        for a in range(k):                                  # Cov(X_a, X_k) = Cov(X_a, X_{k-1}) F^T
            covs[a, k] = covs[a, k - 1] @ F.T
        covs[k, k] = P
        means.append(mX)
    n = len(means[0])
    mean = np.concatenate(means)
    cov = np.zeros((K * n, K * n))
    for (a, b), c in covs.items():
        cov[a * n:(a + 1) * n, b * n:(b + 1) * n] = c
        cov[b * n:(b + 1) * n, a * n:(a + 1) * n] = c.T
    return dates, mean, cov, ix


def gauss_expectation(mean, cov, fn, n=16, tol=1e-13):
    """E[fn(X)] for X ~ N(mean, cov) by tensor Gauss-Hermite quadrature on the range of cov (eigenvalues
    below tol * max are dropped). fn maps an (N, d) array of points to (N,) or (N, k) values."""
    w, U = np.linalg.eigh(cov)
    keep = w > tol * max(w.max(), 1e-300)
    root = U[:, keep] * np.sqrt(w[keep])
    x, wt = np.polynomial.hermite.hermgauss(n)
    x, wt = np.sqrt(2.0) * x, wt / np.sqrt(np.pi)
    r = root.shape[1]
    grids = np.meshgrid(*([x] * r), indexing="ij")
    xi = np.stack([g.ravel() for g in grids], axis=1)
    W = np.prod(np.stack(np.meshgrid(*([wt] * r), indexing="ij")).reshape(r, -1), axis=0)
    vals = fn(mean + xi @ root.T)
    return np.tensordot(W, vals, axes=(0, 0))


def correlation_exact(model, curves, X1, X2, measure="RN", Tstar=None, n=16):
    """Exact Corr(X1, X2) under Q^{d,RN} or the domestic Tstar-forward measure (Gauss-Hermite, n points per
    dimension on the joint law of the state components the observables depend on)."""
    if measure == "Tfwd" and (Tstar is None or Tstar < max(X1.date, X2.date) - 1e-12):
        raise ValueError("Tfwd needs Tstar >= the dates of the observables")
    dates, mean, cov, ix = joint_law(model, [X1.date, X2.date], measure, Tstar)
    nX = ix["n"]
    coords = []                                     # (date, component) of the reduced Gaussian vector
    for X in (X1, X2):
        for c in X.components(model):
            if (X.date, c) not in coords:
                coords.append((X.date, c))
    pos = [dates.index(d) * nX + (ix["iZX"] if c == "x" else c) for d, c in coords]
    mu, C = mean[pos], cov[np.ix_(pos, pos)]

    def state(X, pts):
        z = np.zeros((len(pts), model.nr))
        x = np.zeros(len(pts))
        for k, (d, c) in enumerate(coords):
            if d == X.date:
                if c == "x":
                    x = pts[:, k]
                else:
                    z[:, c] = pts[:, k]
        return {"z": z, "x": x}

    def fn(pts):
        a, b = X1.value(model, curves, state(X1, pts)), X2.value(model, curves, state(X2, pts))
        return np.stack([a, b], axis=1)
    m1, m2 = gauss_expectation(mu, C, fn, n)

    def central(pts):
        v = fn(pts) - [m1, m2]
        return np.stack([v[:, 0] * v[:, 1], v[:, 0] ** 2, v[:, 1] ** 2], axis=1)
    c12, v1, v2 = gauss_expectation(mu, C, central, n)
    return CorrelationResult(float(c12 / sqrt(v1 * v2)), float(c12), float(v1), float(v2))


# ---------------------------------------------------------------------- Monte Carlo
def _influence(x, y, om):
    mx, my = np.mean(om * x), np.mean(om * y)
    dx, dy = x - mx, y - my
    sxx, syy, sxy = np.mean(om * dx * dx), np.mean(om * dy * dy), np.mean(om * dx * dy)
    rho = sxy / sqrt(sxx * syy)
    xs, ys = dx / sqrt(sxx), dy / sqrt(syy)
    return rho, om * (xs * ys - 0.5 * rho * (xs * xs + ys * ys)), (sxy, sxx, syy)


def _stderr(psi, antithetic):
    if antithetic:
        psi = 0.5 * (psi[0::2] + psi[1::2])
    return float(psi.std(ddof=1) / sqrt(len(psi)))


def weighted_correlation(x, y, w=None, antithetic=True):
    """Self-normalised weighted sample correlation and its delta-method standard error: with omega = w/mean(w)
    and standardised x~, y~, the influence function is psi = omega (x~ y~ - rho (x~^2 + y~^2) / 2), averaged
    over antithetic pairs before taking its standard deviation. Returns (rho, stderr, (cov, var_x, var_y))."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    om = np.ones_like(x) if w is None else np.asarray(w, float) / np.mean(w)
    rho, psi, mom = _influence(x, y, om)
    return float(rho), _stderr(psi, antithetic), tuple(float(v) for v in mom)


def correlation_mc(model, curves, X1, X2, n_paths=2**17, seed=1, measure="RN", Tstar=None, dt=0.25, control=False):
    """Monte Carlo Corr(X1, X2) with the exact LGCM scheme (simulated under Q^{d,RN}); measure "Tfwd" reweights
    by D_t Z^{d|d}_{t,Tstar} (t the last date), whose RN mean is Z^{d|d}_{0,Tstar}.

    control=True: control variate by the linearised observables L_k = g_k . (z, z^x) on the same paths, whose
    correlation is exactly correlation_approx (they are linear in the Gaussian state, under either measure):
    rho = rho^MC(X1, X2) - rho^MC(L1, L2) + rho^approx, standard error from psi_X - psi_L."""
    sim = Simulator(model, scheme=ExactScheme(dt))
    dates = sorted({X1.date, X2.date})
    st = sim.simulate_dates(dates, n_paths, seed)
    vals, lins = [], []
    for X in (X1, X2):
        res = st(X.date)
        vals.append(X.value(model, curves, {"z": res.z, "x": res.zx}))
        lins.append(np.c_[res.z, res.zx] @ X.gradient(model, curves))
    w = None
    if measure == "Tfwd":
        if Tstar is None or Tstar < dates[-1] - 1e-12:
            raise ValueError("Tfwd needs Tstar >= the dates of the observables")
        res = st(dates[-1])
        w = sim.deflator(res, curves, "d", "d") * sim.discount(res, curves, "dd", Tstar)
    elif measure != "RN":
        raise ValueError(measure)
    om = np.ones(n_paths) if w is None else w / np.mean(w)
    rho, psi, (c, v1, v2) = _influence(vals[0], vals[1], om)
    if not control:
        return CorrelationResult(float(rho), c, v1, v2, _stderr(psi, True))
    rho_l, psi_l, _ = _influence(lins[0], lins[1], om)
    ap = correlation_approx(model, curves, X1, X2)
    return CorrelationResult(float(rho - rho_l + ap.corr), c, v1, v2, _stderr(psi - psi_l, True))
