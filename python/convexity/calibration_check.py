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
* resetting CCS: tests the collateral convexity K_{0,j} of the foreign coupon (Libor, or compounded
  overnight on the fixing grid of the coupon, see CouponPaths) of the pricer against the model's
  dynamics (a K = 1 spread is rejected by many standard errors). The compounded coupons are valued on
  paths: the fixing dates are streamed through the simulation (not stored), so a long daily grid
  costs transitions, not memory. mc_report(..., compounding_step=h) values the coupon with step h on
  the paths and in the model (with_compounding_step); the closed form is exact for any step.
* swaptions and CB vols: the quoted "normal vol" of the pricer is a *price*, price / (A0 sqrt(T0/2pi))
  (exact one factor price, resp. exact 2D quadrature under the annuity measure), not a Gaussian
  variance. The MC straddle price is mapped by the same linear map, so the comparison is a price
  comparison with an exact standard error (no Gaussian assumption on the MC side).
* FX options: Black vol implied from the MC ATM call price by exact inversion of
  Z F (2 N(vol sqrt(T)/2) - 1) (ln x_T is exactly Gaussian under Q^{d|d,T} in the LGCM); the standard
  error is the price standard error divided by the vega (delta method; relative s.e. ~ 1e-3).
Every control variate has a mean that is an exact no-arbitrage identity of the model (stated at each use).
"""
from collections import defaultdict
from dataclasses import dataclass, replace
from math import copysign, inf, nan, pi, sqrt
import time

import numpy as np
from scipy.stats import norm

from .curves import CurveSet
from .instruments import CurrencyBasisSwaption, FXForward, FXOption, ResettingCCS, Swap, Swaption
from .market import coupon_grid
from .montecarlo import Simulator, cv_mean_and_stderr, mean_and_stderr
from .schemes import ExactScheme, PathStates


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
        return set()                        # streamed through CouponPaths
    return {inst.expiry}


def _periods(inst, setup):
    Tj = inst.schedule(setup)
    return list(zip(Tj[:-1], Tj[1:]))


class CouponPaths:
    """Pathwise values of foreign coupons paid in f at b under d collateral, accumulated while the paths
    are simulated (an `on_date` observer of simulate_dates; register the periods with add(), simulate
    cp.dates(), read sums() and means()).

    Coupon factor on [a, b] with fixing grid a = t_0 < ... < t_n = b (coupon.grid; the continuous coupon
    is valued as its daily product on coupon_grid(a, b, coupon.step), an O(step) approximation):
    C = prod_k 1/Z^{f|f}_{t_k,t_{k+1}} (times kappa^f for Libor, n = 1). The time-0 value of C paid at b is
    E[D^{f|d}_b C]. Write C = C_{<n-1} / Z^{f|f}_{s,b}, s = t_{n-1}, with C_{<n-1} = prod_{k<n-1} and
    1/Z^{f|f}_{s,b} both known at s. By the tower property and the martingale property of the deflated
    f zero-coupon bond under d collateral, E_s[D^{f|d}_b] = D^{f|d}_s Z^{f|d}_{s,b}, so
        E[D^{f|d}_b C] = E[Y],   Y = D^{f|d}_s Z^{f|d}_{s,b} C_{<n-1} / Z^{f|f}_{s,b}
    exactly (no discretisation: the exact scheme samples the joint law of the state at the fixing
    dates), without simulating the last overnight step. For Libor (n = 1) this is the fixing-date
    valuation D_a Z^{f|d}_{a,b} / Z^{f|f}_{a,b}.

    Controls with exactly known means (sums over the periods of a basket):
      C1 = D^{f|d}_s Z^{f|d}_{s,b},  E = P_b = Z^{f|d}_{0,b} (same martingale property);
      C2 = D^{f|d}_s A_{<n-1},       E = P_a = Z^{f|d}_{0,a}, A_{<n-1} = prod_{k<n-1} 1/Z^{f|d}_{t_k,t_{k+1}} the
           f account rolled at the d-collateralised rate: it telescopes by the same martingale
           property (E_{t_k}[D^{f|d}_{t_{k+1}}] = D^{f|d}_{t_k} Z^{f|d}_{t_k,t_{k+1}}). Y / C2 is a product of
           basis factors Z^{f|d}/Z^{f|f} only, which removes the FX x rates noise of the compounding.
           For Libor (n = 1) C2 = D^{f|d}_a;
      C3 = D^{f|f}_s C_{<n-1},       E = Z^{f|f}_{0,a}: under f collateral the rolled overnight account
           telescopes (E_{t_k}[D^{f|f}_{t_{k+1}}] / Z^{f|f}_{t_k,t_{k+1}} = D^{f|f}_{t_k}), the native
           (model-free) value of the compounded coupon. With C1, C2 it removes the remaining noise not
           driven by the collateral basis (it halves the CCS s.e., measured on EURUSD and GBPUSD).
    Periods shared by several baskets (CCS of different maturities) are simulated once."""

    def __init__(self, sim, curves, coupon=None):
        self.sim, self.curves = sim, curves
        self.coupon = curves.coupon("f") if coupon is None else coupon
        self._events = defaultdict(list)     # date key -> [(period, k, next fixing)]
        self._periods = {}                   # (a, b) -> number of fixings n
        self._baskets = {}                   # name -> list of periods
        self._C = {}                         # running (C_{<k}, A_{<k}) per period
        self._sums = {}
        self._dates = {}                     # date key -> date

    def grid(self, a, b):
        c = self.coupon
        return coupon_grid(a, b, c.step) if c.kind == "continuous" else c.grid(a, b)

    def add(self, name, periods):
        periods = [(float(a), float(b)) for a, b in periods]
        self._baskets[name] = periods
        for p in periods:
            if p in self._periods:
                continue
            g = self.grid(*p)
            self._periods[p] = len(g) - 1
            for k, (t0, t1) in enumerate(zip(g[:-1], g[1:])):
                self._events[PathStates.key(t0)].append((p, k, float(t1)))
                self._dates.setdefault(PathStates.key(t0), float(t0))

    def dates(self):
        """Fixing dates t_0 .. t_{n-1} of every registered period (one per date key)."""
        return sorted(self._dates.values())

    def __call__(self, res):
        sim, c = self.sim, self.curves
        for p, k, t1 in self._events.get(PathStates.key(res.T), ()):
            b = p[1]
            n = self._periods[p]
            if k == 0:
                self._C[p] = (np.ones(res.n_paths), np.ones(res.n_paths))
            Cp, Ap = self._C[p]
            if k < n - 1:
                self._C[p] = (Cp / sim.discount(res, c, "ff", t1), Ap / sim.discount(res, c, "fd", t1))
                continue
            Dfd = sim.deflator(res, c, "f", "d")
            Psb = sim.discount(res, c, "fd", b)
            vals = dict(Y=Dfd * Psb * Cp / sim.discount(res, c, "ff", b), C1=Dfd * Psb, C2=Dfd * Ap,
                        C3=sim.deflator(res, c, "f", "f") * Cp)
            del self._C[p]
            for name, periods in self._baskets.items():
                if p in periods:
                    acc = self._sums.setdefault(name, {})
                    for key, v in vals.items():
                        acc[key] = acc[key] + v if key in acc else np.array(v, float)

    def sums(self, name):
        """Pathwise sums over the basket's periods of Y, C1, C2, C3 (arrays of n_paths)."""
        return self._sums[name]

    def means(self, name):
        """Exact means of the controls C1, C2, C3 of the basket."""
        a, b = np.array(self._baskets[name]).T
        c = self.curves
        return dict(C1=float(np.sum(c.Z0("fd", b))), C2=float(np.sum(c.Z0("fd", a))), C3=float(np.sum(c.Z0("ff", a))))


def coupon_convexity_mc(model, curves, coupon, a, b, n_paths=2**16, seed=1, dt=0.25):
    """Monte Carlo K = E[D^{f|d}_b C] / (Z^{f|d}_{0,b} Z^{f|f}_{0,a} / Z^{f|f}_{0,b}) of the coupon on [a, b]
    (exact scheme, CouponPaths valuation and controls). Returns (K, standard error)."""
    sim = Simulator(model, scheme=ExactScheme(dt))
    cp = CouponPaths(sim, curves, coupon)
    cp.add("K", [(a, b)])
    sim.simulate_dates([], n_paths, seed, stream=cp.dates(), on_date=cp)
    s, m = cp.sums("K"), cp.means("K")
    mu, se = cv_mean_and_stderr(s["Y"], [s["C1"], s["C2"], s["C3"]], [m["C1"], m["C2"], m["C3"]])
    norm_ = curves.Z0("fd", b) * curves.Z0("ff", a) / curves.Z0("ff", b)
    return mu / norm_, se / norm_


def with_compounding_step(cal, step):
    """The calibration `cal` with the foreign OIS index compounded on a grid of step `step` (curves and
    model unchanged): the model coupon (pricer K, CB-vol inputs) and the MC paths then use that step."""
    s = cal.curves.setup
    if s.foreign_index.kind != "OIS":
        raise ValueError(f"{s.foreign_index.name}: the compounding step applies to an OIS foreign index only")
    setup = replace(s, foreign_index=replace(s.foreign_index, on_step=step))
    curves = CurveSet(setup, cal.curves.x0, {k: cal.curves[k] for k in cal.curves.keys()})
    return replace(cal, curves=curves)


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
    Z^{d|d}_{0,T} (x^{df|d}_{0,T} - F), on the model outright F (x in units of d per f). The quote is
    converted to F (inst.to_outright), F_hat = F_mkt + mean / Z and s_F = se / Z are mapped back to the
    quoted convention (inst.quoted). For an inverted pair the quote is g(F) = 1/F (points: pip (1/F - 1/x_0)),
    so by the delta method s.e.(g) = |g'(F)| s_F = pip s_F / F^2 (pip = 1 for outrights); the second
    order bias is O(s_F^2 / F^3), far below the s.e."""
    inst, T = q.instrument, q.instrument.maturity
    res, c = st(T), cal.curves
    F_mkt = inst.to_outright(q.value, c.x0)
    mu, se = mean_and_stderr(sim.deflator(res, c, "d", "d") * (sim.fx_spot(res, c) - F_mkt))
    F, sF = F_mkt + mu / c.Z0("dd", T), se / c.Z0("dd", T)
    pip = 1.0 if inst.quotation == "outright" else inst.pip_factor
    return inst.quoted(F, c.x0), (pip * sF / F**2 if inst.inverted else pip * sF)


def _ccs(sim, st, cal, q):
    """Resetting CCS under d collateral, in f units. The foreign coupon of period [a, b] pays
    kappa^f C - 1 at b (C: Libor or compounded overnight factor, see CouponPaths) and
    E[D^{f|d}_b kappa^f C] = P_b (1 + theta L^{f|d}_{0,a}) with E[D^{f|d}_b C] = E[Y] (CouponPaths). Hence
        theta sum_j P_j L^{f|d}_j = E[X],  X = sum_j (kappa^f Y_j - C1_j),
    and B = (sum_j (kappa^d P_{j-1} - P_j) - E[X]) / (theta sum_j P_j), with the controls C1, C2, C3
    of CouponPaths (exact means sum_j P_j, sum_j P_{j-1}, sum_j Z^{f|f}_{0,T_{j-1}})."""
    c, s = cal.curves, cal.curves.setup
    Tj = q.instrument.schedule(s)
    th = s.foreign_index.tenor
    P = c.Z0("fd", Tj)
    cp = st.coupon_paths
    S, m = cp.sums(_basket(q.instrument, s)), cp.means(_basket(q.instrument, s))
    X = s.foreign_index.kappa * S["Y"] - S["C1"]
    mu, se = cv_mean_and_stderr(X, [S["C1"], S["C2"], S["C3"]], [m["C1"], m["C2"], m["C3"]])
    A0 = th * P[1:].sum()
    resetting = (s.domestic_index.kappa * P[:-1] - P[1:]).sum()
    return (resetting - mu) / A0, se / A0


def _basket(inst, setup):
    return tuple(inst.schedule(setup))


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


def mc_report(cal, n_paths=2**16, seed=1, quotes=None, dt=0.25, timings=None, compounding_step=None):
    """Monte Carlo repricing of `quotes` (default: all) on one set of paths. `timings`, if a dict,
    receives the simulation and valuation times in seconds. compounding_step: if given, the foreign OIS
    coupon is compounded with this step on the paths AND in the model (with_compounding_step); the
    column `market` stays the quote (calibrated with the index's own step)."""
    quotes = cal.quotes.quotes if quotes is None else quotes
    if compounding_step is not None:
        cal = with_compounding_step(cal, compounding_step)
    s = cal.curves.setup
    sim = Simulator(cal.model, scheme=ExactScheme(dt))
    dates = sorted(set().union(*(_dates(q.instrument, s) for q in quotes)))
    cp = CouponPaths(sim, cal.curves)
    for q in quotes:
        if isinstance(q.instrument, ResettingCCS):
            cp.add(_basket(q.instrument, s), _periods(q.instrument, s))
    t0 = time.perf_counter()
    st = sim.simulate_dates(dates, n_paths, seed, stream=cp.dates(), on_date=cp)
    st.coupon_paths = cp
    t1 = time.perf_counter()
    rows = []
    for q in quotes:
        val, se = _MC[type(q.instrument)](sim, st, cal, q)
        rows.append(CheckRow(q.instrument.label, type(q.instrument).__name__, q.value, float(val), float(se)))
    if timings is not None:
        timings.update(simulate=t1 - t0, value=time.perf_counter() - t1, n_dates=len(set(dates) | set(cp.dates())))
    return rows
