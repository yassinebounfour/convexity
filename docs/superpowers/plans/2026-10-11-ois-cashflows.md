# Overnight-Indexed Cashflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Model compounded overnight (RFR) coupons — exact daily compounding, with continuous
compounding as an approximation — alongside Libor coupons in the LGCM notes and code, including
their collateral convexity, the RFR cross-currency market (GBPUSD, USDJPY with inverted FX
quotes) and the verification by exact Gaussian benchmarks and Monte Carlo.

**Architecture:** A `Coupon` value object (libor / daily / continuous) carries the coupon type
from the `Index` to every place that used the Libor collateral convexity: `Pricer`,
`ResettingCCS`, `gaussian_approx`, `payoffs`, `calibration_check`. The convexity of a coupon under
non-native collateral has a closed form in the LGCM (covariance integrals); an independent exact
benchmark augments the exact Gaussian transitions with the log-coupon accumulator. FX quotes gain
an inverted quotation. The notes are generalised in place.

**Tech Stack:** Python 3.12, numpy, scipy, pytest, marimo, matplotlib; LaTeX (amsart).

**Spec:** `docs/superpowers/specs/2026-10-11-ois-cashflows-design.md`

## Global Constraints

- Run tests from `python/`: `python -m pytest convexity -q`. The 111 existing tests must keep passing with unchanged tolerances.
- Existing public names keep working: `market.libor_convexity`, `market.InitialCurves`, `gaussian_approx.approx/exact/approx_native`, `payoffs.*` (InitialCurves-based tests keep Libor coupons).
- Monte Carlo tests: 4 standard error tolerance, fixed seeds, control variates where noise hides the effect; never widen a tolerance to pass.
- Exact vs asymptotic vs numerical observation must be distinguished; quoted errors are measured numbers.
- Notation: blocks d, f, c, x; Cheyette convention; the model's x is units of d per unit of f.
- LaTeX: `cd docs && pdflatex -interaction=nonstopmode CollateralDiscounting.tex` (twice): 0 errors, only the known warnings (multiply defined `eq:cb_r_swaplet`, `eq:forward_cb_non_resetting`; undefined `eq:cb_r_swaplet2`).
- Performance: `calibrate(example_quotes(m))` under 30 s for each of the three markets on this machine.
- Commit trailer: `Co-Authored-By: <model name> <noreply@anthropic.com>` naming the model that wrote the code.
- Do not commit generated artifacts (`__marimo__/`, PDFs, scratch scripts).

## Review Focus

1. A foreign OIS index under domestic collateral must never be priced with the Libor convexity (the current silent bug) — negative-control test in Task 2 and Task 4.
2. Valuation inside a partially accrued period (t strictly between a and b): fixed days excluded, no double counting — test in Task 1.
3. An accrual period that is not a whole number of steps (b − a not a multiple of `on_step`): last step shortened to land on b — test in Task 1.
4. Inverted FX quotes with points and pip factor 100 (USDJPY): same Z^{f|d} as direct outrights — test in Task 3.
5. Existing InitialCurves-based code paths unchanged (Libor coupon by default): existing tests pass unchanged — every task.

---

## File structure

| file | status | responsibility |
|---|---|---|
| `python/convexity/market.py` | modify | `Coupon`, `LIBOR`, `coupon_grid`, `compounded_convexity`; `InitialCurves.coupon(role)` |
| `python/convexity/curves.py` | modify | `Index.compounding`, `Index.on_step`, `Index.coupon`; `CurveSet.coupon(role)` |
| `python/convexity/benchmarks.py` | create | exact Gaussian expectation of a coupon under domestic collateral (augmented transitions) |
| `python/convexity/pricers.py` | modify | `Pricer.coupon_convexity`, `coupon_factor`; LGCMPricer cache per coupon |
| `python/convexity/instruments.py` | modify | `ResettingCCS` coupon-aware; `FXForward(inverted=...)`; `MarketQuotes.fx_inverted`, `x0` |
| `python/convexity/gaussian_approx.py`, `payoffs.py` | modify | coupon-aware K (default Libor) |
| `python/convexity/curve_calibration.py`, `calibration.py` | modify | `CurveSet(setup, quotes.x0)` |
| `python/convexity/market_data.py` | modify | `example_quotes(market)` for EURUSD, GBPUSD, USDJPY |
| `python/convexity/calibration_check.py` | modify | compounded coupons on paths |
| `python/convexity/tests/test_ois.py` | create | tests of Tasks 1–4 |
| `python/notebooks/lgcm_calibration.py` | modify | market selector, coupon comparison |
| `docs/CollateralDiscounting.tex`, READMEs | modify | integrated theory and verification |

---

### Task 1: Coupon model, closed-form convexity and exact benchmark

**Files:** modify `market.py`, `curves.py`; create `benchmarks.py`, `tests/test_ois.py`.

**Interfaces:**
- Produces: `market.Coupon(kind="libor", step=1/365)` (frozen dataclass; kind in {"libor", "daily", "continuous"}) with `convexity(m, t, a, b) -> float`; `market.LIBOR = Coupon("libor")`; `market.coupon_grid(a, b, step) -> np.ndarray` (a, a+step, ..., last point exactly b, last step shortened); `market.compounded_convexity(m, t, a, b, compounding, step) -> float`; `InitialCurves.coupon(role) -> Coupon` (always `LIBOR`); `Index(name, currency, kind="OIS", tenor=0.25, spread=0.0, compounding="daily", on_step=1/365)` with `.coupon` (IBOR → `LIBOR`; OIS → `Coupon(compounding, on_step)`); `CurveSet.coupon(role)` (the role's index coupon); `benchmarks.coupon_value_exact(m, curves, coupon, a, b, t=0.0) -> float` = E^{f|d,b}[C] / (Z^{f|f}_{0,a}/Z^{f|f}_{0,b}) computed from the exact joint Gaussian law (no covariance-kernel formula).

**Mathematics (to implement and to prove later in the notes, Task 7).** Foreign coupon factor C on [a, b] (native f curve), paid at b in f under d collateral. Libor: C = 1/Z^{f|f}_{a,b} (times kappa, kept outside K). Daily: C = prod_k 1/Z^{f|f}_{t_k,t_{k+1}} on `coupon_grid(a, b, step)`. Continuous: C = exp(int_a^b r^{f|f}). K_t = E^{f|d,b}_t[C] / E^{f|f,b}_t[C] and E^{f|f,b}_t[C] = Z^{f|f}_{t,a}/Z^{f|f}_{t,b} (native telescoping; for t inside the period, the realised part of C is known at t and cancels in the ratio). Expected closed forms (sign convention of `market.libor_convexity`, Σ = `m.Sig`):
 - daily: ln K = − Σ_{k: t_k > t} [ I(Σ^f_{t_{k+1}}, Σ^c_b; t, t_k) − I(Σ^f_{t_k}, Σ^c_b; t, t_k) ];
 - Libor = daily with the single step [a, b];
 - continuous: ln K = − [ I(Σ^f_b − Σ^f_a, Σ^c_b; t, a)·1{t<a} + I(Σ^f_b, Σ^c_b; max(t, a), b) ].
These must be confirmed by the independent benchmark below before anything builds on them. If the benchmark disagrees, find out which is wrong (derive on paper: the change of measure Q^{f|f,b} → Q^{f|d,b} has density driven by Σ^c_{·,b}; ln C is affine in the Gaussian state) and report — do not tune the formula to the benchmark without a derivation.

- [ ] **Step 1: failing tests** (`tests/test_ois.py`)

```python
"""Compounded overnight coupons: convexity closed forms, exact benchmark (Task 1), pricer (Task 2),
markets and inverted FX (Task 3), Monte Carlo (Task 4)."""
import numpy as np
import pytest

from convexity.lgcm import default_model
from convexity.market import InitialCurves, Coupon, LIBOR, coupon_grid, libor_convexity
from convexity.benchmarks import coupon_value_exact

M, CV = default_model(), InitialCurves()


def test_coupon_grid_lands_on_b():
    g = coupon_grid(1.0, 1.25, 1 / 52)
    assert g[0] == 1.0 and g[-1] == 1.25 and np.all(np.diff(g) > 0)
    assert np.all(np.diff(g)[:-1] == pytest.approx(1 / 52)) and np.diff(g)[-1] <= 1 / 52 + 1e-15
    np.testing.assert_array_equal(coupon_grid(1.0, 1.25, 1.0), [1.0, 1.25])


def test_libor_is_the_one_step_daily_coupon():
    for t in (0.0, 1.0):
        assert Coupon("daily", 10.0).convexity(M, t, 5.0, 5.25) == pytest.approx(libor_convexity(M, t, 5.0, 5.25), rel=1e-14)
        assert LIBOR.convexity(M, t, 5.0, 5.25) == libor_convexity(M, t, 5.0, 5.25)


@pytest.mark.parametrize("coupon", [LIBOR, Coupon("daily", 1 / 52), Coupon("daily", 1 / 12)])
@pytest.mark.parametrize("t", [0.0, 2.0])
def test_closed_form_matches_exact_gaussian_benchmark(coupon, t):
    a, b = 5.0, 5.25
    assert coupon.convexity(M, t, a, b) == pytest.approx(coupon_value_exact(M, CV, coupon, a, b, t), rel=1e-11)


def test_partially_accrued_period():
    """t strictly inside the accrual period: only the days after t carry convexity."""
    c = Coupon("daily", 1 / 52)
    a, b, t = 5.0, 5.25, 5.0 + 5 / 52 + 1e-9
    assert c.convexity(M, t, a, b) == pytest.approx(coupon_value_exact(M, CV, c, a, b, t), rel=1e-11)
    assert Coupon("daily", 1 / 52).convexity(M, b, a, b) == 1.0


def test_continuous_is_the_limit_of_daily_with_order_one():
    a, b = 10.0, 10.25
    kc = Coupon("continuous").convexity(M, 0.0, a, b)
    errs = [abs(np.log(Coupon("daily", h).convexity(M, 0.0, a, b) / kc)) for h in (1 / 26, 1 / 52, 1 / 104)]
    assert errs[0] / errs[1] == pytest.approx(2, rel=0.1) and errs[1] / errs[2] == pytest.approx(2, rel=0.1)


def test_compounded_and_libor_convexities_differ_by_the_accrual_term():
    a, b = 10.0, 10.25
    kl, kc = LIBOR.convexity(M, 0.0, a, b), Coupon("continuous").convexity(M, 0.0, a, b)
    assert kl != kc
    extra = -M.I(M.Sig("f", b), M.Sig("c", b), a, b)
    assert np.log(kc / kl) == pytest.approx(extra, rel=1e-12)
```

Also test `Index` / `CurveSet` (append):
```python
from convexity.curves import Index


def test_index_coupon():
    assert Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.001).coupon == LIBOR
    assert Index("GBP_SONIA", "GBP").coupon == Coupon("daily", 1 / 365)
    assert Index("GBP_SONIA", "GBP", compounding="continuous").coupon == Coupon("continuous", 1 / 365)
    with pytest.raises(ValueError):
        Index("X", "GBP", compounding="weekly")
    with pytest.raises(ValueError):
        Index("X", "GBP", "IBOR", 0.25, 0.0, compounding="continuous")   # compounding is for OIS only
```
- [ ] **Step 2: run, expect ImportError.**

- [ ] **Step 3: implement** in `market.py`:

```python
from dataclasses import dataclass


def coupon_grid(a, b, step):
    """Overnight grid a = t_0 < ... < t_n = b with steps `step` (the last one shortened)."""
    n = max(int(np.ceil((b - a) / step - 1e-9)), 1)
    return np.r_[a + step * np.arange(n), b]


def compounded_convexity(m, t, a, b, compounding="daily", step=1 / 365):
    """K_t = E^{f|d,b}_t[C] / E^{f|f,b}_t[C] for the foreign overnight factor C compounded on [a, b] and paid
    at b under domestic collateral, LGCM (Proposition "Collateral convexity of compounded coupons").
    compounding="daily": C = prod 1/Z^{f|f}_{t_k,t_{k+1}} on coupon_grid(a, b, step) (exact);
    compounding="continuous": C = exp(int_a^b r^{f|f}) (the step -> 0 limit)."""
    sc = m.Sig("c", b)
    if compounding == "continuous":
        x = m.I(m.Sig("f", b), sc, max(t, a), b) if b > max(t, a) else 0.0
        if t < a:
            x += m.I(m.Sig("f", b), sc, t, a) - m.I(m.Sig("f", a), sc, t, a)
        return float(np.exp(-x))
    if compounding != "daily":
        raise ValueError(compounding)
    g = coupon_grid(a, b, step)
    x = sum(m.I(m.Sig("f", t1), sc, t, t0) - m.I(m.Sig("f", t0), sc, t, t0)
            for t0, t1 in zip(g[:-1], g[1:]) if t0 > t)
    return float(np.exp(-x))


@dataclass(frozen=True)
class Coupon:
    """Foreign coupon on [a, b] paid at b: "libor" (fixed at a), "daily" (exact daily compounding with
    step), "continuous" (continuous compounding, an approximation of daily)."""
    kind: str = "libor"
    step: float = 1 / 365

    def __post_init__(self):
        if self.kind not in ("libor", "daily", "continuous"):
            raise ValueError(self.kind)

    def convexity(self, m, t, a, b):
        if self.kind == "libor":
            return libor_convexity(m, t, a, b)
        return compounded_convexity(m, t, a, b, self.kind, self.step)


LIBOR = Coupon("libor")
```
`InitialCurves.coupon(self, role)` returns `LIBOR`. In `curves.py`: `Index` gets
`compounding: str = "daily"`, `on_step: float = 1 / 365` (validate: compounding in
{"daily", "continuous"}; IBOR requires the defaults; `on_step > 0`), property
`coupon` (IBOR → `LIBOR`, OIS → `Coupon(self.compounding, self.on_step)`), import `Coupon, LIBOR`
from `.market` (check for a circular import: `market.py` must not import `curves.py`).
`CurveSet.coupon(role)` returns the coupon of `setup.domestic_index` / `setup.foreign_index`.

Performance: the daily sum calls `m.I` 2·n times (n ≈ 91 per quarter). Measure the time of one
quarterly `Coupon("daily").convexity`; if it exceeds ~5 ms, vectorise (e.g. evaluate the covint
contributions for all t_k of the grid in one pass per model piece) and keep the scalar version as
a test oracle.

- [ ] **Step 4: implement `benchmarks.py`** — exact Gaussian expectation by state augmentation:

```python
"""Exact Gaussian benchmarks that do not use the covariance-kernel formulas of market.py.

coupon_value_exact: E^{f|d,b}_t[C] / E^{f|f,b}_t[C] for a coupon factor C on [a, b], computed from the
joint Gaussian law of (state, log deflator, log C) obtained by stepping the exact transitions
(transitions.transition, measure "RN") over coupon_grid(a, b, step) and accumulating
ln C = sum_k -ln Z^{f|f}_{t_k,t_{k+1}} (each term affine in z_{t_k}, Model.lnZ("ff", ...)) as an extra
state. Then E^{d,RN}[D_b C] with D_b the f-payment / d-collateral deflator of Simulator.deflator
(exp(-int r^{d|d}) exp(int q) x_b / x_0 and the initial curves), and E[exp(linear form)] = exp(mean + var/2).
"""
```
Return E^{d,RN}[D_b C] / Z^{f|d}_{0,b} divided by the native forward Z^{f|f}_{0,a}/Z^{f|f}_{0,b}.
For t > 0: K_t is deterministic in the LGCM (independent of z_t), and the exact transitions from t
only use the model parameters after t and the deterministic y_t, v^{f,c}_t (`transitions.transition`
reads `m.Y(s)`, `m.vfc(s)`), so start the stepping at t from the zero state and use the curves seen
from 0 (the curve levels cancel in the ratio). For t inside the accrual period, the grid starts at
the first overnight date after t and the realised factor is excluded (it is known at t). State this
argument in the docstring. The Libor coupon uses the one-step grid [a, b]. The continuous coupon is
not benchmarked here (it is the step -> 0 limit, tested by convergence).

- [ ] **Step 5: run tests, expect PASS**; full suite unchanged (111 + new). Measure and report: the
  benchmark agreement (max rel. diff), the convergence ratios, K − 1 for Libor, daily, continuous at
  a = 2, 10, 20 (quarterly) on `default_model()`, the timing of one daily quarterly convexity.

- [ ] **Step 6: commit** — "Compounded overnight coupons: closed-form collateral convexity and exact Gaussian benchmark".

---

### Task 2: Coupon-aware pricers, CCS and Gaussian approximations

**Files:** modify `pricers.py`, `instruments.py`, `gaussian_approx.py`, `payoffs.py`, `calibration_check.py` (K seen from T0 in `_cb_vol`/`cb_straddle_inputs`), `tests/test_ois.py`.

**Interfaces:**
- Consumes: Task 1 (`Coupon`, `CurveSet.coupon`, `InitialCurves.coupon`).
- Produces: `Pricer.coupon_convexity(a, b)` (K of the foreign index coupon seen from 0; DeterministicPricer: 1.0) replacing `Pricer.libor_convexity` (keep `libor_convexity` as a deprecated alias that delegates only when the coupon is Libor, else raises — or remove it and update all callers; choose removal if no external caller exists, grep the repo); `Pricer.forward_rate` uses `coupon_convexity`; `gaussian_approx._ingredients` takes the coupon from `s.curves.coupon("f")` (default Libor for InitialCurves) and computes K0, KT with `coupon.convexity`; `payoffs.SwapPayoff.convexities(model, t, curves=None)` uses `curves.coupon("f")` when given (default Libor); `calibration_check` uses the coupon of the setup for K_{T0,j}.

- [ ] **Step 1: failing tests** (append to `tests/test_ois.py`)

```python
from convexity.curves import Index, LGCMSetup, CurveSet, PiecewiseFlatCurve, SpreadCurve
from convexity.instruments import ResettingCCS, CurrencyBasisSwaption
from convexity.pricers import LGCMPricer, DeterministicPricer
from convexity import gaussian_approx as ga

SOFR = Index("USD_SOFR", "USD")
SONIA = Index("GBP_SONIA", "GBP", on_step=1 / 52)
SETUP_RFR = LGCMSetup(SOFR, SONIA)                         # foreign index = SONIA (compounded)


def rfr_curves():
    cs = CurveSet(SETUP_RFR, 1.27)
    cs[SETUP_RFR.key("dd")] = PiecewiseFlatCurve([2, 10, 50], [0.042, 0.038, 0.04])
    cs[SETUP_RFR.key("ff")] = PiecewiseFlatCurve([2, 10, 50], [0.040, 0.037, 0.039])
    cs[SETUP_RFR.key("fd")] = SpreadCurve(cs[SETUP_RFR.key("ff")], PiecewiseFlatCurve([5, 50], [-0.0005, -0.0008]))
    return cs


def test_rfr_ccs_uses_the_compounded_convexity_not_libor():
    cs = rfr_curves()
    pr = LGCMPricer(M, cs)
    ccs = ResettingCCS(20.0)
    Tj = ccs.schedule(SETUP_RFR)
    P, Zf = cs.Z0("fd", Tj), cs.Z0("ff", Tj)
    K = np.array([SONIA.coupon.convexity(M, 0.0, a, b) for a, b in zip(Tj[:-1], Tj[1:])])
    B = ((P[:-1] / P[1:] - Zf[:-1] / Zf[1:] * K) * P[1:]).sum() / (0.25 * P[1:].sum())
    assert ccs.quote(pr) == pytest.approx(B, rel=1e-13)
    KL = np.array([libor_convexity(M, 0.0, a, b) for a, b in zip(Tj[:-1], Tj[1:])])
    BL = ((P[:-1] / P[1:] - Zf[:-1] / Zf[1:] * KL) * P[1:]).sum() / (0.25 * P[1:].sum())
    assert abs(ccs.quote(pr) - BL) > 1e-6                   # the Libor K is not used (negative control)


def test_gaussian_approx_and_exact_use_the_coupon():
    cs = rfr_curves()
    s = ga.SwapSetup(T0=5.0, theta=0.25, n=8, curves=cs)
    a = ga.approx(M, s)
    cbs = CurrencyBasisSwaption.make(5.0, 2.0)
    assert cbs.underlying.quote(LGCMPricer(M, cs, exact=False)) == pytest.approx(a["B0"], rel=1e-13)
    # the exact quadrature reprices the forward under the annuity measure (martingale check)
    ex = ga.exact(M, s)
    assert ex["B"]["mean"] == pytest.approx(a["B0"], abs=1e-14)
    assert ex["Y"]["mean"] == pytest.approx(a["Y0"], abs=1e-14)


def test_initial_curves_paths_keep_libor():
    s = ga.SwapSetup(T0=5.0, theta=0.25, n=8, curves=CV)
    _, _, _, _, K0, _ = ga._ingredients(M, s)
    np.testing.assert_allclose(K0[1:], [libor_convexity(M, 0.0, a, b) for a, b in zip(s.Tj[:-1], s.Tj[1:])], rtol=1e-15)
```
Note `default_model()` has the factor layout the pricer expects (one f, one c factor).

- [ ] **Step 2: run, expect failures** (the CCS currently uses Libor K — the RFR test must fail on the old code: record this as the RED evidence of the bug fix).
- [ ] **Step 3: implement** the interface changes above. In `ga.exact` the state functions at T0 use K seen from T0 (`KT`) — with compounded coupons the period starting at T0 has its whole accrual after T0, so `coupon.convexity(m, T0, T_{j-1}, T_j)` is the right quantity (no partially accrued period because T0 is a period start); state it in a comment.
- [ ] **Step 4: run tests, expect PASS**; full suite (existing Libor tests unchanged).
- [ ] **Step 5: commit** — "Coupon-aware pricers, resetting CCS and Gaussian approximations (fixes Libor K applied to RFR coupons)".

---

### Task 3: Inverted FX quotation, RFR example markets, calibration on three markets

**Files:** modify `instruments.py`, `curve_calibration.py`, `calibration.py`, `market_data.py`, `calibration_check.py` (`_fx_forward` for inverted quotes), `tests/test_ois.py`.

**Interfaces:**
- Produces: `FXForward(maturity, quotation="outright", pip_factor=1e4, inverted=False)`: inverted quotes 1/F (outright) or (1/F − 1/x0)·pip_factor (points); label shows the quoted pair; `MarketQuotes(setup, fx_spot, quotes, fx_inverted=False)` where `fx_spot` is the QUOTED spot and `MarketQuotes.x0` the model spot (1/fx_spot if inverted); `calibrate`/`calibrate_curves` build `CurveSet(setup, quotes.x0)`; `market_data.example_quotes(market="EURUSD")` with markets "EURUSD" (unchanged numbers and indices), "GBPUSD" (SOFR/SONIA, RFR swaps annual fixed vs compounded, RFR resetting CCS quarterly), "USDJPY" (SOFR/TONA, f = JPY, quotes inverted: spot ~150 JPY per USD, forwards in points with pip_factor 100); `market_data.MARKETS` tuple.
- FX option vols are unchanged by inversion: in the LGCM ln x_T is Gaussian, so 1/x_T is lognormal with the same total variance and the ATM Black vol of 1/x equals that of x. State this in the `FXOption` docstring; the notes prove it in Task 7.

- [ ] **Step 1: failing tests** (append)

```python
from convexity.instruments import FXForward, MarketQuotes, Quote, Swap
from convexity.curve_calibration import calibrate_curves
from convexity.calibration import calibrate
from convexity.market_data import example_quotes, MARKETS
from convexity.calibration_check import analytic_report


def test_inverted_fx_quotes_give_the_same_curve():
    mq = example_quotes("USDJPY")
    assert mq.fx_inverted and mq.x0 == pytest.approx(1 / mq.fx_spot)
    cs, _ = calibrate_curves(mq)
    # direct outrights generated from the inverted-quote curve rebuild the same Z^{f|d}
    pr = DeterministicPricer(cs)
    direct = [Quote(FXForward(q.instrument.maturity), FXForward(q.instrument.maturity).quote(pr))
              if isinstance(q.instrument, FXForward) else q for q in mq.quotes]
    cs2, _ = calibrate_curves(MarketQuotes(mq.setup, mq.x0, direct))
    T = np.linspace(0.05, 30, 40)
    np.testing.assert_allclose(cs2.Z0("fd", T), cs.Z0("fd", T), rtol=1e-12)


@pytest.mark.parametrize("market", MARKETS)
def test_calibration_reprices_every_market(market):
    cal = calibrate(example_quotes(market))
    assert cal.diagnostics["seconds"] < 30
    rows = analytic_report(cal)
    for r in rows:
        tol = 1e-8 if r.kind == "FXForward" else 1e-10
        assert abs(r.error) < tol, (market, r.label, r.error)


def test_rfr_markets_use_compounded_coupons():
    for market in ("GBPUSD", "USDJPY"):
        mq = example_quotes(market)
        assert mq.setup.foreign_index.kind == "OIS" and mq.setup.foreign_index.coupon.kind == "daily"
```
- [ ] **Step 2: run, expect failures.**
- [ ] **Step 3: implement.** Market numbers: rounded, plausible, illustrative (say so in the module docstring); choose them so that calibration has no variance squeeze and calibrated levels are plausible (report them). GBPUSD: spot 1.27, SONIA swaps ~4.0% (1Y) to ~3.8% (30Y), SOFR as in EURUSD, small negative SONIA/SOFR basis, GBP swaption normal vols ~90–110 bp, GBPUSD ATM vols ~8–10%, CB vols ~5–10 bp. USDJPY: spot ~150 JPY per USD, TONA swaps ~0.5% (1Y) to ~1.8% (30Y), JPY/USD basis ~−20 to −40 bp, JPY swaption vols ~40–70 bp, USDJPY ATM vols ~9–12%, CB vols ~8–15 bp. FX forwards (points) up to 1Y, CCS beyond. Use `Index("GBP_SONIA", "GBP")`, `Index("JPY_TONA", "JPY")` with the default daily compounding and `on_step = 1/365`; if the calibration time exceeds 30 s with daily steps, report the profile (do not silently switch to a coarser step; a coarser `on_step` is a modelling choice to be stated in the docstring and notes, with the measured effect on the calibrated quotes).
- [ ] **Step 4: run tests, expect PASS**; full suite.
- [ ] **Step 5: commit** — "Inverted FX quotation; GBPUSD and USDJPY RFR example markets".

---

### Task 4: Monte Carlo of compounded coupons

**Files:** modify `calibration_check.py`; append to `tests/test_ois.py`.

**Interfaces:**
- Consumes: Tasks 1–3, `Simulator.simulate_dates`, `PathStates`.
- Produces: `mc_report` values RFR CCS floating legs with the compounded coupon on paths: for a daily coupon, the dates of `coupon_grid(T_{j-1}, T_j, step)` are simulated and C = prod 1/Z^{f|f}_{t_k,t_{k+1}} on each path; the value is E[D_{T_j} C] with D the f/d deflator at T_j (or, equivalently and with less variance, D_{t_{n-1}} C_{<n-1} Z^{f|d}_{t_{n-1},b}/Z^{f|f}_{t_{n-1},b} — state which and why it is exact). `mc_report(..., compounding_step=None)` lets a test override the coupon step on paths AND in the model (via a modified setup index) so that the MC cost stays small (e.g. weekly), since the closed form is exact for any step.

- [ ] **Step 1: failing tests** (append, `@pytest.mark.slow`):
 - MC of `Coupon("daily", 1/52)` convexity on `default_model()` for one quarterly period at a = 5 vs the closed form within 4 s.e., with the Libor K at > 4 s.e. on the same run if resolvable (otherwise report the resolution and use a longer maturity where the in-arrears term is larger; do not fake power);
 - GBPUSD calibrated with weekly `on_step` (stated): MC repricing of the RFR CCS quotes with maturity ≤ 5Y and of the CB vols ≤ 5Y within 4 s.e., negative control: the CCS spread recomputed with the Libor K is > 4 s.e. from MC at the longest maturity in the subset if resolvable (report the measured z).
- [ ] **Step 2–4:** implement, run, pass; suite time with slow tests reported (keep the new slow tests under ~60 s total).
- [ ] **Step 5: measure for the notes** (scratchpad script, not in the repo): for each market, analytic max errors per type, MC max |z| (2^17 paths, seed 7; weekly step on paths if daily is too slow — state it), K − 1 for Libor/daily/continuous at 2Y/10Y/30Y on the calibrated GBPUSD model, the shift of the calibrated Z^{f|d} (bp of zero rate at 10Y, 30Y) between Libor-K and RFR-K calibrations on the same GBPUSD quotes, timings. Write all numbers with commands and seeds into the report.
- [ ] **Step 6: commit** — "Monte Carlo of compounded RFR coupons in the calibration checks".

---

### Task 5: Notebook

**Files:** modify `python/notebooks/lgcm_calibration.py`.

- [ ] Add a market selector (EURUSD, GBPUSD, USDJPY) driving the cached calibration; quote tables in the quoted convention (inverted for USDJPY); a coupon-comparison panel on the selected market: K − 1 per period to 30Y for Libor, daily, continuous; and the Z^{f|d} shift (bp) between calibrating with Libor-K and RFR-K on the same quotes (for RFR markets). Keep the existing caching rules (cached functions in slider-independent cells, run buttons for expensive tables, CalibrationError callouts). Verify with `marimo check` and a headless `marimo export html` (0 MarimoException/Traceback), the suite green, `__marimo__/` untracked. Report the headline numbers shown.
- [ ] Commit — "Notebook: market selector and Libor vs compounded coupon convexity".

---

### Task 6: README and module docs

- [ ] Update `python/convexity/README.md` (module table: `benchmarks.py`, coupon types; usage for the three markets; inverted FX; what is checked: new tests), `python/convexity/__init__.py` docstring, root `README.md` if it lists capabilities. Run every snippet you add. Commit — "README: compounded coupons, RFR markets, inverted FX".

---

### Task 7: Notes, integrated throughout

**Files:** modify `docs/CollateralDiscounting.tex`.

- [ ] Generalise in place (keep existing labels; add new ones for new statements):
 1. Collateral convexity of forward rates (sections "Convexity adjustment to forward rates" in the HJM framework and "Collateral convexity adjustments" in deterministic-volatility models): define the three coupon factors; Proposition (collateral convexity of a coupon factor) with the general HJM statement and the LGCM closed forms (Libor, daily — exact, continuous — the step → 0 limit); proof by change of numéraire Q^{f|f,b} → Q^{f|d,b} and Gaussian covariance; Remark: native collateral, all coupons are model-free (telescoping); Remark: the in-arrears accrual term, its size vs the Libor term (measured numbers from Task 1/4 reports), and the error of continuous vs daily (order O(step), measured ratios, a numerical observation unless you prove it — if you prove it, state the proof).
 2. Currency basis swaplets and spreads (resetting CCS with RFR legs): replace Libor-specific statements by coupon-generic ones; the domestic resetting leg remains model-free for both coupon types.
 3. Gaussian par spread approximations: loadings unchanged in form, K_j coupon-dependent (state precisely), exact quadrature with K seen from T0.
 4. Calibration section: instruments (RFR swaps, RFR CCS), FX inverted quotation and Proposition (ATM Black vol of 1/x equals that of x in the LGCM, with proof), the three example markets, verification remark with measured numbers (Task 4 Step 5), Libor-K vs RFR-K calibrated curve shift remark.
 Every number with a source (reports/ledger or a recorded script); exact vs asymptotic vs observation labelled.
- [ ] Compile twice: 0 errors, only the known warnings. Commit — "Notes: compounded overnight coupons alongside Libor".

---

### Task 8: Final verification

- [ ] Full suite; LaTeX twice; notebook export; `git status` clean; then the whole-branch review and `superpowers:finishing-a-development-branch`.
