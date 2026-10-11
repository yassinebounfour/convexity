"""Do the calibrated model's delivered quotes match the market? Closed forms and Monte Carlo.

analytic_report  every quote repriced with the LGCM pricer (exact where available), and the Gaussian
                 approximation for swaptions and CB vols (column `approx`).
mc_report        every quote repriced by Monte Carlo under Q^{d,RN} with the exact LGCM scheme: one
                 pass over the union of the needed dates, deflators of Simulator.deflator (the time-0
                 value of V_T in ccy under a collateral is E[D V_T]).

Conventions of the Monte Carlo values (column `model`, same quotation as the market quote):
* swaps and FX forwards: the MC checks only the martingale property of the deflators against the
  curves (E[D_t] = Z_{0,t}, E[D x_T] = Z x^{fwd}); they do not test the calibrated vols. Their closed
  forms are repriced analytically to ~1e-16 (analytic_report).
* resetting CCS: tests the Libor collateral convexity K_{0,j} of the pricer against the model's
  dynamics (a K = 1 spread is rejected by many standard errors).
* swaptions and CB vols: the quoted "normal vol" of the pricer is a *price*, price / (A0 sqrt(T0/2pi))
  (exact one factor price, resp. exact 2D quadrature under the annuity measure), not a Gaussian
  variance. The MC straddle price is mapped by the same linear map, so the comparison is a price
  comparison with an exact standard error (no Gaussian assumption on the MC side).
* FX options: Black vol implied from the MC ATM call price by exact inversion of
  Z F (2 N(vol sqrt(T)/2) - 1) (ln x_T is exactly Gaussian under Q^{d|d,T} in the LGCM); the standard
  error is the price standard error divided by the vega (delta method; relative s.e. ~ 1e-3).
Every control variate has a mean that is an exact no-arbitrage identity of the model (stated at each use).
"""
from dataclasses import dataclass
from math import copysign, inf, nan, pi, sqrt
import time

import numpy as np
from scipy.stats import norm

from .instruments import CurrencyBasisSwaption, FXForward, FXOption, ResettingCCS, Swap, Swaption
from .montecarlo import Simulator, cv_mean_and_stderr, mean_and_stderr
from .schemes import ExactScheme


@dataclass
class CheckRow:
    label: str
    kind: str
    market: float
    model: float
    stderr: float = nan
    approx: float = nan

    @property
    def error(self):
        return self.model - self.market

    @property
    def z(self):
        """error / stderr; +-inf for a nonzero error with stderr 0, nan for 0 / 0 or without stderr."""
        if self.stderr == 0:
            return copysign(inf, self.error) if self.error != 0 else nan
        return self.error / self.stderr


def analytic_report(cal):
    exact, approx = cal.pricer(exact=True), cal.pricer(exact=False)
    rows = []
    for q in cal.quotes.quotes:
        inst = q.instrument
        a = inst.quote(approx) if isinstance(inst, (Swaption, CurrencyBasisSwaption)) else nan
        rows.append(CheckRow(inst.label, type(inst).__name__, q.value, inst.quote(exact), approx=a))
    return rows


# ---------------------------------------------------------------------- Monte Carlo valuations
def _dates(inst, setup):
    if isinstance(inst, Swap):
        tf, _, tx, _ = inst.cashflows()
        return set(tf) | set(tx)
    if isinstance(inst, FXForward):
        return {inst.maturity}
    if isinstance(inst, ResettingCCS):
        return set(inst.schedule(setup)[:-1])
    return {inst.expiry}


def _swap(sim, st, cal, q):
    """Deflator / curve martingale check (no vol dependence). Native swap, collateral e: float leg sum_m c_m Z^{e|e}_{0,t_m} (kappa at t_{i-1}, -1 at t_i),
    fixed leg sum_k w_k Z_{0,t_k}. Value at the market rate: E[sum c_m D_{t_m} - S sum w_k D_{t_k}]
    with E[D^{e|e}_t] = Z^{e|e}_{0,t}; par = S + value / A0."""
    u, role = q.instrument, cal.curves.setup.role(q.instrument.index.currency)
    tf, cf, tx, wx = u.cashflows()
    D = lambda t: sim.deflator(st(t), cal.curves, role, role)
    pv = sum(c * D(t) for t, c in zip(tf, cf)) - q.value * sum(w * D(t) for t, w in zip(tx, wx))
    A0 = (wx * cal.curves.Z0(role * 2, tx)).sum()
    mu, se = mean_and_stderr(pv)
    return q.value + mu / A0, se / A0


def _fx_forward(sim, st, cal, q):
    """Deflator / curve martingale check (no vol dependence): E[D^{d|d}_T (x_T - F)] =
    Z^{d|d}_{0,T} (x^{df|d}_{0,T} - F)."""
    inst, T = q.instrument, q.instrument.maturity
    res, c = st(T), cal.curves
    F_mkt = q.value if inst.quotation == "outright" else c.x0 + q.value / inst.pip_factor
    mu, se = mean_and_stderr(sim.deflator(res, c, "d", "d") * (sim.fx_spot(res, c) - F_mkt))
    F, sF = F_mkt + mu / c.Z0("dd", T), se / c.Z0("dd", T)
    return (F, sF) if inst.quotation == "outright" else ((F - c.x0) * inst.pip_factor, sF * inst.pip_factor)


def _ccs(sim, st, cal, q):
    """Resetting CCS under d collateral, in f units. The foreign Libor of period [a, b] fixes at a and
    pays at b: E[D^{f|d}_a Z^{f|d}_{a,b} kappa^f / Z^{f|f}_{a,b}] = P_b (1 + theta L^{f|d}_{0,a}), with
    the state at the fixing date a (deterministic for a = 0). Hence
        theta sum_j P_j L^{f|d}_j = E[X],  X = sum_j D_a Z^{f|d}_{a,b} (kappa^f / Z^{f|f}_{a,b} - 1),
    and B = (sum_j (kappa^d P_{j-1} - P_j) - E[X]) / (theta sum_j P_j).
    Controls: C1 = sum_j D_a Z^{f|d}_{a,b} and C2 = sum_j D_a with E[C1] = sum_j P_j, E[C2] =
    sum_j P_{j-1} (E[D^{f|d}_a Z^{f|d}_{a,b}] = Z^{f|d}_{0,b}, martingale property of f under d collateral)."""
    c, s = cal.curves, cal.curves.setup
    Tj = q.instrument.schedule(s)
    th = s.foreign_index.tenor
    P = c.Z0("fd", Tj)
    X = C1 = C2 = 0.0
    for a, b in zip(Tj[:-1], Tj[1:]):
        res = st(a)
        D = sim.deflator(res, c, "f", "d")
        Pab = sim.discount(res, c, "fd", b)
        X = X + D * Pab * (s.foreign_index.kappa / sim.discount(res, c, "ff", b) - 1.0)
        C1, C2 = C1 + D * Pab, C2 + D
    mu, se = cv_mean_and_stderr(X, [C1, C2], [P[1:].sum(), P[:-1].sum()])
    A0 = th * P[1:].sum()
    resetting = (s.domestic_index.kappa * P[:-1] - P[1:]).sum()
    return (resetting - mu) / A0, se / A0


def _straddle_vol(D, V, A0, T0):
    """Pricer convention: vol = E[D V^+] / (A0 sqrt(T0/2pi)) (a price, not a Gaussian variance).
    E[D V^+] = 0.5 E[D |V|] + 0.5 E[D V] and E[D V] = 0 exactly (V is the value at T0 of a tradable
    worth 0 at time 0, the ATM strike being the forward), used as a control variate."""
    mu, se = cv_mean_and_stderr(0.5 * D * np.abs(V), [D * V], [0.0])
    k = A0 * sqrt(T0 / (2 * pi))
    return mu / k, se / k


def _swaption(sim, st, cal, q):
    """V = sum_m c_m Z^{e|e}_{T0,t_m} - K sum_k w_k Z^{e|e}_{T0,t_k} at the ATM strike K = fl0 / A0 on the
    pricer's curve: E[D^{e|e}_{T0} V] = fl0 - K A0 = 0."""
    u = q.instrument.underlying
    role = cal.curves.setup.role(u.index.currency)
    res = st(u.start)
    tf, cf, tx, wx = u.cashflows()
    Z0 = lambda T: cal.curves.Z0(role * 2, T)
    fl0, A0 = u.legs(Z0)
    K = fl0 / A0
    disc = lambda t: sim.discount(res, cal.curves, role * 2, t) if t > u.start else np.ones(res.n_paths)
    V = sum(c * disc(t) for t, c in zip(tf, cf)) - K * sum(w * disc(t) for t, w in zip(tx, wx))
    return _straddle_vol(sim.deflator(res, cal.curves, role, role), V, A0, u.start)


def _fx_option(sim, st, cal, q):
    """ATM call (x_T - F)^+ under d collateral, F = x^{df|d}_{0,T}: E[D^{d|d}_T (x_T - F)] = 0 is the
    control, and the call price Z F (2 N(vol sqrt(T)/2) - 1) (exact: ln x_T Gaussian under Q^{d|d,T})
    is inverted exactly."""
    T, c = q.instrument.expiry, cal.curves
    res = st(T)
    F, Z = c.fx_forward0(T), c.Z0("dd", T)
    D, x = sim.deflator(res, c, "d", "d"), sim.fx_spot(res, c)
    mu, se = cv_mean_and_stderr(0.5 * D * np.abs(x - F), [D * (x - F)], [0.0])
    vol = 2 * norm.ppf(0.5 * (mu / (Z * F) + 1)) / sqrt(T)
    vega = Z * F * sqrt(T) * norm.pdf(0.5 * vol * sqrt(T))
    return vol, se / vega


def cb_straddle_inputs(sim, st, cal, q):
    """Pathwise inputs of the CB-vol check at T0: (D, AB - A B0, A0, T0) with D = D^{f|d}_{T0}.
    A_{T0} = theta sum_j Z^{f|d}_{T0,T_j} and 1 + theta L^{f|d}_{T0,j} = kappa^f Z^{f|f}_{T0,j-1} /
    Z^{f|f}_{T0,j} K_{T0,j} (collateral convexity of the foreign coupon seen from T0, coupon.convexity(m, T0, a, b)).
    A B is the value of a tradable, so E[D (AB - A B0)] = 0 with B0 the time-0 spread of the pricer
    (convexities K_{0,j}); this control-mean identity is checked by the tests."""
    c, s, m = cal.curves, cal.curves.setup, cal.model
    u = q.instrument.underlying
    T0, Tj, th = u.start, u.schedule(s), s.foreign_index.tenor
    res = st(T0)
    one = np.ones(res.n_paths)
    P = np.stack([one] + [sim.discount(res, c, "fd", t) for t in Tj[1:]], axis=1)
    Zf = np.stack([one] + [sim.discount(res, c, "ff", t) for t in Tj[1:]], axis=1)
    cp = c.coupon("f")
    K = np.array([cp.convexity(m, T0, a, b) for a, b in zip(Tj[:-1], Tj[1:])])
    one_L = s.foreign_index.kappa * Zf[:, :-1] / Zf[:, 1:] * K
    one_Lr = s.domestic_index.kappa * P[:, :-1] / P[:, 1:]
    A = th * P[:, 1:].sum(1)
    AB = ((one_Lr - one_L) * P[:, 1:]).sum(1)
    B0 = u.quote(cal.pricer())
    A0 = th * c.Z0("fd", Tj[1:]).sum()
    return sim.deflator(res, c, "f", "d"), AB - A * B0, A0, T0


def _cb_vol(sim, st, cal, q):
    """Option on the forward resetting CB spread B_{T0}, in f units under d collateral: the ATM straddle
    0.5 E[D A |B - B0|] = A0 E^A[(B - B0)^+] is the price behind the exact (2D quadrature) CB vol
    (see cb_straddle_inputs for the zero-mean control D A (B - B0))."""
    return _straddle_vol(*cb_straddle_inputs(sim, st, cal, q))


_MC = {Swap: _swap, FXForward: _fx_forward, ResettingCCS: _ccs, Swaption: _swaption, FXOption: _fx_option,
       CurrencyBasisSwaption: _cb_vol}


def mc_report(cal, n_paths=2**16, seed=1, quotes=None, dt=0.25, timings=None):
    """Monte Carlo repricing of `quotes` (default: all) on one set of paths. `timings`, if a dict,
    receives the simulation and valuation times in seconds."""
    quotes = cal.quotes.quotes if quotes is None else quotes
    sim = Simulator(cal.model, scheme=ExactScheme(dt))
    dates = sorted(set().union(*(_dates(q.instrument, cal.curves.setup) for q in quotes)))
    t0 = time.perf_counter()
    st = sim.simulate_dates(dates, n_paths, seed)
    t1 = time.perf_counter()
    rows = []
    for q in quotes:
        val, se = _MC[type(q.instrument)](sim, st, cal, q)
        rows.append(CheckRow(q.instrument.label, type(q.instrument).__name__, q.value, float(val), float(se)))
    if timings is not None:
        timings.update(simulate=t1 - t0, value=time.perf_counter() - t1, n_dates=len(dates))
    return rows
