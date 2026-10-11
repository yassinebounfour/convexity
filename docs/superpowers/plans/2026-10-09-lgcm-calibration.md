# LGCM Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the full LGCM (curves, sigma^d, sigma^f, sigma^c, sigma^x) to swap, FX forward,
resetting CCS, swaption, FX option and CB vol quotes, verify the delivered quotes analytically and by
Monte Carlo, document it in the notes and illustrate it in a marimo notebook.

**Architecture:** Curves are identified by `CurveKey`s and held in a `CurveSet` that also exposes the
role-based interface of `market.InitialCurves`, so the simulator and payoffs run unchanged on
calibrated curves. Instruments are pure data that price themselves through a `Pricer`
(`DeterministicPricer` = DCB, `LGCMPricer` = LGCM closed forms). A generic curve bootstrapper solves
one node per pillar; the LGCM calibrator wraps it in a fixed point with the collateral basis vol.

**Tech Stack:** Python 3.12, numpy, scipy, sympy, pytest, matplotlib, marimo; LaTeX (amsart).

**Spec:** `docs/superpowers/specs/2026-10-09-lgcm-calibration-design.md`

## Global Constraints

- Run tests from `python/`: `python -m pytest convexity -q`. The 57 existing tests must keep passing.
- Monte Carlo tests: 4 standard error tolerance, fixed seeds, control variates where noise hides the effect.
- Exact vs asymptotic vs conjectural must be distinguished; quoted errors are measured numbers.
- Notation: blocks `d`, `f`, `c`, `x`; Cheyette state convention.
- `market.InitialCurves` and all existing public names stay unchanged.
- LaTeX: `cd docs && pdflatex -interaction=nonstopmode CollateralDiscounting.tex` (twice) with 0 errors and no new warnings beyond the known ones (`eq:cb_r_swaplet`, `eq:forward_cb_non_resetting` multiply defined; `eq:cb_r_swaplet2` undefined).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Quotes given in arbitrary order → calibration sorts them; result independent of input order (test in Task 5).
2. Two option quotes with the same expiry in one block → `ValueError` naming the instrument (test in Task 5).
3. Instrument whose period does not divide its length (e.g. CCS 2.1Y with quarterly periods) → `ValueError` from `schedule` (test in Task 2).
4. FX forwards in points with a non-default pip factor (e.g. 100 for JPY pairs) → same curve as outrights (test in Task 3).
5. Swaption whose underlying ends beyond the last swap pillar → flat forward extrapolation, calibration succeeds (covered by the 20Y×10Y example quote, test in Task 5).

---

## File structure

| file | status | responsibility |
|---|---|---|
| `requirements.txt`, `environment.yml` (repo root) | create | reproducible environment |
| `python/convexity/curves.py` | create | `Index`, `CurveKey`, `DiscountCurveKey`, `ForwardCurveKey`, `Curve`, `PiecewiseFlatCurve`, `SpreadCurve`, `FunctionCurve`, `LGCMSetup`, `CurveSet` |
| `python/convexity/instruments.py` | create | `schedule`, `Instrument`, `Swap`, `FXForward`, `ResettingCCS`, `Swaption`, `FXOption`, `CurrencyBasisSwaption`, `Quote`, `MarketQuotes` |
| `python/convexity/pricers.py` | create | `Pricer`, `DeterministicPricer`, `LGCMPricer`, `native_swaption_variance`, `native_swaption_exact_vol` |
| `python/convexity/curve_calibration.py` | create | `CalibrationError`, `BootstrapReport`, `bootstrap`, `calibrate_curves` |
| `python/convexity/calibration.py` | create | `BlockSpec`, `ModelSpec`, `default_spec`, `Calibration`, `calibrate` |
| `python/convexity/market_data.py` | create | `example_setup`, `example_quotes` (USD/EUR) |
| `python/convexity/calibration_check.py` | create | `CheckRow`, `analytic_report`, `mc_report` |
| `python/convexity/schemes.py` | modify | `MCScheme.simulate_dates`, `ExactScheme.simulate_dates`, `PathStates` |
| `python/convexity/montecarlo.py` | modify | `Simulator.simulate_dates`, `cv_mean_and_stderr` |
| `python/convexity/tests/test_curves.py` | create | Tasks 1–3 tests |
| `python/convexity/tests/test_calibration.py` | create | Tasks 4–6 tests |
| `python/notebooks/lgcm_calibration.py` | create | marimo notebook |
| `docs/CollateralDiscounting.tex` | modify | calibration section rewrite |
| `python/convexity/README.md`, `__init__.py` | modify | document new modules |

Note: `payoffs.py` already defines classes `Swap`, `Swaption`, `FXForward`, `FXOption` (Monte Carlo
payoffs). The calibration instruments live in `instruments.py` and are never imported together
with the payoffs. The MC valuation of instruments lives in `calibration_check.py` (spec deviation:
no `CurrencyBasisSwaption` payoff is added to `payoffs.py`; the spec is updated in Task 8).

---

### Task 1: Environment files and curves

**Files:**
- Create: `requirements.txt`, `environment.yml`, `python/convexity/curves.py`
- Test: `python/convexity/tests/test_curves.py`

**Interfaces:**
- Produces: `Index(name, currency, kind="OIS", tenor=0.25, spread=0.0)` with `.kappa`;
  `DiscountCurveKey(currency, collateral)`, `ForwardCurveKey(index, collateral)`;
  `PiecewiseFlatCurve(pillars=(), forwards=())` with `Z(T)`, `add_node(t, f)`, `set_last(f)`,
  `pillars`, `forwards`, `zero_rate(T)`, `forward(T)`; `SpreadCurve(base, spread=None)` (same node API);
  `FunctionCurve(fn)`; `LGCMSetup(domestic_collateral, foreign_collateral, domestic_index=None,
  foreign_index=None)` with `domestic`, `foreign`, `pair`, `role(ccy)`, `collateral_of(ccy)`,
  `index_by_name(name)`, `key(kind)`; `CurveSet(setup, x0, curves=None)` with `[key]`,
  `Z(key, T)`, `Z0(kind, T)`, `kappa_d`, `kappa_f`, `kappa(role)`, `fx_forward0(T)`, `x0`.

- [ ] **Step 1: environment files**

`requirements.txt`:
```
numpy>=2.0
scipy>=1.13
sympy>=1.13
pytest>=8.0
matplotlib>=3.8
marimo>=0.9
```
`environment.yml`:
```yaml
name: convexity
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - -r requirements.txt
```
Install marimo in the working interpreter: `python -m pip install "marimo>=0.9"`. After the
notebook runs (Task 7), pin the tested versions (`==`) in `requirements.txt`.

- [ ] **Step 2: failing tests** (`tests/test_curves.py`)

```python
"""Curves, curve keys and the curve set (Task 1), instruments (Task 2), curve bootstrap (Task 3)."""
import numpy as np
import pytest

from convexity.curves import (Index, DiscountCurveKey, ForwardCurveKey, PiecewiseFlatCurve,
                                          SpreadCurve, FunctionCurve, LGCMSetup, CurveSet)

SOFR = Index("USD_SOFR", "USD")
ESTR = Index("EUR_ESTR", "EUR")
E3M = Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.0010)
SETUP = LGCMSetup(SOFR, ESTR, foreign_index=E3M)


def test_index_kappa_and_validation():
    assert SOFR.kappa == 1.0 and E3M.kappa == pytest.approx(1.00025)
    with pytest.raises(ValueError):
        Index("X", "USD", "OIS", 0.25, 0.001)
    with pytest.raises(ValueError):
        Index("X", "USD", "FOO")


def test_curve_keys_are_values():
    assert DiscountCurveKey("EUR", "USD_SOFR") == DiscountCurveKey("EUR", "USD_SOFR")
    assert len({DiscountCurveKey("EUR", "USD_SOFR"), DiscountCurveKey("EUR", "USD_SOFR")}) == 1
    assert DiscountCurveKey("EUR", "USD_SOFR") != ForwardCurveKey("EUR", "USD_SOFR")
    assert str(DiscountCurveKey("EUR", "USD_SOFR")) == "Z[EUR|USD_SOFR]"


def test_piecewise_flat_curve():
    c = PiecewiseFlatCurve([1.0, 3.0], [0.02, 0.04])
    assert c.Z(0.0) == 1.0
    assert c.Z(0.5) == pytest.approx(np.exp(-0.01))
    assert c.Z(2.0) == pytest.approx(np.exp(-0.02 - 0.04))
    assert c.Z(5.0) == pytest.approx(np.exp(-0.02 - 0.08 - 0.08))        # flat extrapolation
    np.testing.assert_allclose(c.Z(np.array([0.5, 2.0])), [np.exp(-0.01), np.exp(-0.06)])
    assert c.forward(2.0) == pytest.approx(0.04, abs=1e-8)
    assert PiecewiseFlatCurve().Z(3.0) == 1.0
    with pytest.raises(ValueError):
        c.add_node(2.0, 0.01)


def test_spread_curve_multiplies_base():
    base = PiecewiseFlatCurve([10.0], [0.03])
    s = SpreadCurve(base)
    s.add_node(2.0, -0.001)
    s.set_last(-0.002)
    assert s.pillars == [2.0] and s.forwards == [-0.002]
    assert s.Z(4.0) == pytest.approx(np.exp(-0.12 + 0.008))


def test_setup_roles_and_keys():
    assert (SETUP.domestic, SETUP.foreign, SETUP.pair) == ("USD", "EUR", "EURUSD")
    assert SETUP.role("EUR") == "f" and SETUP.collateral_of("EUR") is ESTR
    assert SETUP.domestic_index is SOFR and SETUP.index_by_name("EUR_EURIBOR3M") is E3M
    assert SETUP.key("fd") == DiscountCurveKey("EUR", "USD_SOFR")
    assert SETUP.key("df") == DiscountCurveKey("USD", "EUR_ESTR")
    with pytest.raises(ValueError):
        LGCMSetup(SOFR, ESTR, foreign_index=Index("USD_LIBOR", "USD", "IBOR", 0.25, 0.001))


def test_curve_set_initial_curves_interface():
    cs = CurveSet(SETUP, 1.1)
    cs[SETUP.key("dd")] = PiecewiseFlatCurve([30.0], [0.04])
    cs[SETUP.key("ff")] = PiecewiseFlatCurve([30.0], [0.02])
    cs[SETUP.key("fd")] = SpreadCurve(cs[SETUP.key("ff")], PiecewiseFlatCurve([30.0], [-0.001]))
    assert cs.Z0("qz", 5.0) == pytest.approx(np.exp(0.005))
    assert cs.fx_forward0(5.0) == pytest.approx(1.1 * np.exp(-0.105 + 0.2))
    assert (cs.kappa_d, cs.kappa_f, cs.kappa("f")) == (1.0, E3M.kappa, E3M.kappa)
    with pytest.raises(ValueError, match="model output"):
        cs.Z0("df", 1.0)
    cs[SETUP.key("df")] = FunctionCurve(lambda T: np.exp(-0.03 * T))
    np.testing.assert_allclose(cs.Z0("df", np.array([1.0, 2.0])), np.exp([-0.03, -0.06]))
```

- [ ] **Step 3: run, expect ImportError**

Run: `cd python && python -m pytest convexity/tests/test_curves.py -q`
Expected: FAIL (`ModuleNotFoundError: convexity.curves`).

- [ ] **Step 4: implement `curves.py`**

```python
"""Indices, curve keys, discount curves and the curve set of the LGCM calibration.

A curve is identified by a CurveKey. DiscountCurveKey(currency, collateral) is Z^{currency|collateral},
the collateral being named by its overnight index: with d = USD, f = EUR,
    Z^{d|d} = Z[USD|USD_SOFR], Z^{f|f} = Z[EUR|EUR_ESTR], Z^{f|d} = Z[EUR|USD_SOFR], Z^{d|f} = Z[USD|EUR_ESTR].
ForwardCurveKey(index, collateral) identifies the projection of an IBOR index under a collateral.

CurveSet maps keys to curves. It also exposes the role-based interface of market.InitialCurves
(Z0(kind, T) for kind in dd/ff/fd/df/qz, kappa_d, kappa_f, kappa(role), fx_forward0, x0), so that the
simulator, the payoffs and gaussian_approx run unchanged on calibrated curves.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def _out(x, T):
    return float(x) if np.ndim(T) == 0 else x


@dataclass(frozen=True)
class Index:
    """Rate index. OIS: kappa = 1. IBOR: constant multiplicative funding spread
    kappa = 1 + tenor * spread, i.e. 1 + tenor L_j = kappa Z_{j-1}/Z_j under native collateral
    (kappa = 1 - theta S^{div} in the notes)."""
    name: str
    currency: str
    kind: str = "OIS"
    tenor: float = 0.25
    spread: float = 0.0

    def __post_init__(self):
        if self.kind not in ("OIS", "IBOR"):
            raise ValueError(f"{self.name}: kind must be OIS or IBOR, got {self.kind!r}")
        if self.kind == "OIS" and self.spread != 0.0:
            raise ValueError(f"{self.name}: an OIS index has no funding spread")
        if self.tenor <= 0:
            raise ValueError(f"{self.name}: tenor must be positive")

    @property
    def kappa(self):
        return 1.0 + self.tenor * self.spread


class CurveKey:
    """Hashable identifier of a curve (subclasses are frozen dataclasses)."""


@dataclass(frozen=True)
class DiscountCurveKey(CurveKey):
    currency: str
    collateral: str          # name of the collateral overnight index

    def __str__(self):
        return f"Z[{self.currency}|{self.collateral}]"


@dataclass(frozen=True)
class ForwardCurveKey(CurveKey):
    index: str
    collateral: str

    def __str__(self):
        return f"L[{self.index}|{self.collateral}]"


# ---------------------------------------------------------------------- curves
class Curve(ABC):
    @abstractmethod
    def Z(self, T):
        """Discount factor at T >= 0 (float for a scalar T, array for an array)."""

    def zero_rate(self, T):
        T = np.asarray(T, float)
        return _out(-np.log(self.Z(T)) / T, T)

    def forward(self, T, h=1e-6):
        """Instantaneous forward rate (central difference of ln Z)."""
        T = np.asarray(T, float)
        return _out(-(np.log(self.Z(T + h)) - np.log(self.Z(np.maximum(T - h, 0.0)))) / (T + h - np.maximum(T - h, 0.0)), T)


class PiecewiseFlatCurve(Curve):
    """Flat instantaneous forwards f_k on (t_{k-1}, t_k], t_0 = 0; the last forward is extrapolated."""

    def __init__(self, pillars=(), forwards=()):
        self.pillars = [float(t) for t in pillars]
        self.forwards = [float(f) for f in forwards]
        if len(self.pillars) != len(self.forwards) or np.any(np.diff([0.0] + self.pillars) <= 0):
            raise ValueError("pillars must be increasing and positive, one forward per pillar")

    def add_node(self, t, f):
        if t <= (self.pillars[-1] if self.pillars else 0.0) + 1e-12:
            raise ValueError(f"node {t} is not beyond the last pillar")
        self.pillars.append(float(t))
        self.forwards.append(float(f))

    def set_last(self, f):
        self.forwards[-1] = float(f)

    def integral(self, T):
        T = np.asarray(T, float)
        if not self.pillars:
            return np.zeros_like(T)
        t, f = np.array(self.pillars), np.array(self.forwards)
        left = np.concatenate([[0.0], t[:-1]])
        cum = np.concatenate([[0.0], np.cumsum(f * (t - left))])
        k = np.minimum(np.searchsorted(t, T, side="left"), len(t) - 1)
        return cum[k] + f[k] * (T - left[k])

    def Z(self, T):
        return _out(np.exp(-self.integral(T)), T)


class SpreadCurve(Curve):
    """Z = Z^{base} Q with Q piecewise flat: Z^{f|d} = Z^{f|f} Q^{f|f->d}. The node API acts on Q."""

    def __init__(self, base, spread=None):
        self.base, self.spread = base, spread if spread is not None else PiecewiseFlatCurve()

    @property
    def pillars(self):
        return self.spread.pillars

    @property
    def forwards(self):
        return self.spread.forwards

    def add_node(self, t, f):
        self.spread.add_node(t, f)

    def set_last(self, f):
        self.spread.set_last(f)

    def Z(self, T):
        return _out(np.asarray(self.base.Z(T)) * np.asarray(self.spread.Z(T)), T)


class FunctionCurve(Curve):
    """Curve given by a scalar function T -> Z (e.g. the model-implied off-market curve Z^{d|f})."""

    def __init__(self, fn):
        self.fn = fn

    def Z(self, T):
        T = np.asarray(T, float)
        return float(self.fn(float(T))) if T.ndim == 0 else np.array([self.fn(float(t)) for t in T.ravel()]).reshape(T.shape)


# ---------------------------------------------------------------------- setup and curve set
@dataclass(frozen=True)
class LGCMSetup:
    """Roles of the LGCM: d = domestic (the collateral currency of the cross-currency market), f = foreign.
    The collaterals are overnight indices; the indices are those of the floating legs (default: the
    collateral index). pair = foreign + domestic, x = units of domestic per unit of foreign."""
    domestic_collateral: Index
    foreign_collateral: Index
    domestic_index: Index = None
    foreign_index: Index = None

    def __post_init__(self):
        for role in ("domestic", "foreign"):
            if getattr(self, f"{role}_index") is None:
                object.__setattr__(self, f"{role}_index", getattr(self, f"{role}_collateral"))
            col, idx = getattr(self, f"{role}_collateral"), getattr(self, f"{role}_index")
            if col.kind != "OIS":
                raise ValueError(f"collateral {col.name} must be an OIS index")
            if idx.currency != col.currency:
                raise ValueError(f"{idx.name} is not in the {role} currency {col.currency}")
        if self.domestic == self.foreign:
            raise ValueError("domestic and foreign currencies must differ")
        if self.domestic_index.kind == "IBOR" and self.domestic_index.tenor != self.foreign_index.tenor:
            raise ValueError("resetting CCS: the domestic IBOR tenor must equal the foreign index tenor")

    @property
    def domestic(self):
        return self.domestic_collateral.currency

    @property
    def foreign(self):
        return self.foreign_collateral.currency

    @property
    def pair(self):
        return self.foreign + self.domestic

    def role(self, currency):
        if currency == self.domestic:
            return "d"
        if currency == self.foreign:
            return "f"
        raise ValueError(f"{currency} is neither {self.domestic} nor {self.foreign}")

    def collateral_of(self, currency):
        return self.domestic_collateral if self.role(currency) == "d" else self.foreign_collateral

    def index_by_name(self, name):
        for idx in (self.domestic_index, self.foreign_index, self.domestic_collateral, self.foreign_collateral):
            if idx.name == name:
                return idx
        raise KeyError(name)

    def key(self, kind):
        """DiscountCurveKey of kind = (currency role)(collateral role), e.g. "fd" = Z^{f|d}."""
        ccy = {"d": self.domestic, "f": self.foreign}
        col = {"d": self.domestic_collateral.name, "f": self.foreign_collateral.name}
        return DiscountCurveKey(ccy[kind[0]], col[kind[1]])


class CurveSet:
    """Curves by CurveKey, with the role-based interface of market.InitialCurves."""

    def __init__(self, setup, x0, curves=None):
        self.setup, self.x0 = setup, float(x0)
        self._curves = dict(curves or {})

    def __getitem__(self, key):
        return self._curves[key]

    def __setitem__(self, key, curve):
        self._curves[key] = curve

    def __contains__(self, key):
        return key in self._curves

    def keys(self):
        return self._curves.keys()

    def Z(self, key, T):
        return self._curves[key].Z(T)

    # ------------------------------------------------ InitialCurves interface
    @property
    def kappa_d(self):
        return self.setup.domestic_index.kappa

    @property
    def kappa_f(self):
        return self.setup.foreign_index.kappa

    def kappa(self, ccy):
        return {"d": self.kappa_d, "f": self.kappa_f}[ccy]

    def Z0(self, kind, T):
        if kind == "qz":
            return self.Z0("fd", T) / self.Z0("ff", T)
        key = self.setup.key(kind)
        if key not in self:
            raise ValueError(f"{key} is not in the curve set" + (" (model output, set after calibration)" if kind == "df" else ""))
        return self.Z(key, T)

    def fx_forward0(self, T):
        """x^{df|d}_{0,T} (d collateral)."""
        return self.x0 * self.Z0("fd", T) / self.Z0("dd", T)
```

- [ ] **Step 5: run tests, expect PASS**

Run: `cd python && python -m pytest convexity/tests/test_curves.py -q` → 6 passed.

- [ ] **Step 6: commit**

```bash
git add requirements.txt environment.yml python/convexity/curves.py python/convexity/tests/test_curves.py
git commit -m "Curves: indices, curve keys, piecewise flat and spread curves, curve set; environment files"
```

---

### Task 2: Instruments, quotes and the deterministic pricer

**Files:**
- Create: `python/convexity/instruments.py`, `python/convexity/pricers.py` (base + deterministic only)
- Test: `python/convexity/tests/test_curves.py` (append)

**Interfaces:**
- Consumes: Task 1 names.
- Produces: `schedule(start, end, tenor) -> np.ndarray`; `Instrument` (`pillar`, `priority`,
  `target(setup)`, `dependencies(setup)`, `quote(pricer)`, `label`);
  `Swap(index, end, fixed_tenor=1.0, start=0.0)` with `legs(Z) -> (float_leg, annuity)` and
  `cashflows() -> (tf, cf, tx, wx)`; `FXForward(maturity, quotation="outright", pip_factor=1e4)` with
  `outright(pricer)`; `ResettingCCS(end, start=0.0)` with `schedule(setup)`, `components(pricer) -> (A, B)`;
  `Swaption(underlying)` / `Swaption.make(index, expiry, tenor, fixed_tenor=1.0)` with `expiry`;
  `FXOption(expiry)`; `CurrencyBasisSwaption(underlying)` / `.make(expiry, tenor)`;
  `Quote(instrument, value)`; `MarketQuotes(setup, fx_spot, quotes)` with `of_type(*types)`,
  `for_target(target)`, `with_values(values)`, `without(*types)`.
  `Pricer(curves)` with `setup`, `libor_convexity(a, b)`, `forward_rate(key, a, b)`,
  `swaption_normal_vol`, `fx_black_vol`, `cb_normal_vol`; `DeterministicPricer(curves)`.

- [ ] **Step 1: failing tests** (append to `tests/test_curves.py`)

```python
from convexity.instruments import (schedule, Swap, FXForward, ResettingCCS, Swaption, FXOption,
                                               CurrencyBasisSwaption, Quote, MarketQuotes)
from convexity.pricers import DeterministicPricer


def flat_curves(rd=0.04, rf=0.02, q=-0.001, x0=1.1):
    cs = CurveSet(SETUP, x0)
    cs[SETUP.key("dd")] = PiecewiseFlatCurve([50.0], [rd])
    cs[SETUP.key("ff")] = PiecewiseFlatCurve([50.0], [rf])
    cs[SETUP.key("fd")] = SpreadCurve(cs[SETUP.key("ff")], PiecewiseFlatCurve([50.0], [q]))
    return cs


def test_schedule():
    np.testing.assert_allclose(schedule(5.0, 6.0, 0.25), [5, 5.25, 5.5, 5.75, 6])
    with pytest.raises(ValueError):
        schedule(0.0, 2.1, 0.25)


def test_swap_rates_on_flat_curves():
    pr = DeterministicPricer(flat_curves())
    ois = Swap(SOFR, 10.0)                                      # annual fixed, OIS float: telescopes
    assert ois.quote(pr) == pytest.approx(np.expm1(0.04))
    e3m = Swap(E3M, 10.0)                                       # IBOR: kappa on every float period
    Z = lambda T: np.exp(-0.02 * np.asarray(T))
    tf = schedule(0, 10, 0.25)
    expected = (E3M.kappa * Z(tf[:-1]) - Z(tf[1:])).sum() / Z(np.arange(1, 11)).sum()
    assert e3m.quote(pr) == pytest.approx(expected, rel=1e-14)
    assert ois.target(SETUP) == SETUP.key("dd") and e3m.target(SETUP) == SETUP.key("ff")


def test_fx_forward_outright_and_points():
    cs = flat_curves()
    pr = DeterministicPricer(cs)
    F = 1.1 * np.exp((0.04 - 0.02 + 0.001) * 2.0)
    assert FXForward(2.0).quote(pr) == pytest.approx(F)
    assert FXForward(2.0, "points").quote(pr) == pytest.approx((F - 1.1) * 1e4)
    assert FXForward(2.0, "points", 100.0).quote(pr) == pytest.approx((F - 1.1) * 1e2)
    assert FXForward(2.0).target(SETUP) == SETUP.key("fd")
    assert FXForward(2.0).dependencies(SETUP) == {SETUP.key("dd")}


def test_resetting_ccs_without_convexity():
    """DCB: B = sum P_j (P_{j-1}/P_j - kappa_f Zff_{j-1}/Zff_j) / (theta sum P_j)."""
    cs = flat_curves()
    pr = DeterministicPricer(cs)
    ccs = ResettingCCS(5.0)
    t = schedule(0, 5, 0.25)
    P, Zf = cs.Z0("fd", t), cs.Z0("ff", t)
    B = ((P[:-1] / P[1:] - E3M.kappa * Zf[:-1] / Zf[1:]) * P[1:]).sum() / (0.25 * P[1:].sum())
    assert ccs.quote(pr) == pytest.approx(B, rel=1e-14)
    assert ccs.priority > FXForward(1.0).priority
    assert ccs.dependencies(SETUP) == {SETUP.key("ff")}


def test_options_need_a_stochastic_model():
    pr = DeterministicPricer(flat_curves())
    for inst in (Swaption.make(SOFR, 5, 10), FXOption(5.0), CurrencyBasisSwaption.make(5, 1)):
        with pytest.raises(NotImplementedError):
            inst.quote(pr)
    assert Swaption.make(E3M, 5, 10).target(SETUP) == "f"
    assert (FXOption(5.0).target(SETUP), CurrencyBasisSwaption.make(5, 1).target(SETUP)) == ("x", "c")


def test_market_quotes_selectors():
    mq = MarketQuotes(SETUP, 1.1, [Quote(Swap(SOFR, 5.0), 0.04), Quote(FXForward(1.0), 1.12),
                                   Quote(FXOption(1.0), 0.08)])
    assert len(mq.of_type(Swap, FXForward)) == 2
    assert [q.value for q in mq.for_target(SETUP.key("fd"))] == [1.12]
    assert [q.value for q in mq.with_values([1, 2, 3]).quotes] == [1, 2, 3]
    assert len(mq.without(FXOption).quotes) == 2
```

- [ ] **Step 2: run, expect ImportError** — `python -m pytest convexity/tests/test_curves.py -q`.

- [ ] **Step 3: implement `instruments.py`**

```python
"""Calibration instruments and market quotes.

An Instrument is pure data: it declares what it calibrates (`target`: a CurveKey for linear
instruments, a model block "d", "f", "c" or "x" for options), its `pillar` and the curves it reads
besides its target (`dependencies`), and computes its model quote from a Pricer (`quote`). Linear
instruments are priced from the curves and the Libor collateral convexities given by the pricer;
options delegate to the pricer. Values are in the instrument's quotation: par rate, FX outright or
points, CB spread, normal vol (swaptions, CB spreads) or Black vol (FX).

Conventions: spot date 0; swaps are collateralised in their own currency; FX forwards and resetting
CCS are collateralised in the domestic currency; CCS periods are the foreign index tenor.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from .curves import DiscountCurveKey, ForwardCurveKey


def schedule(start, end, tenor):
    """start, start + tenor, ..., end (a whole number of periods)."""
    n = int(round((end - start) / tenor))
    if n < 1 or abs(start + n * tenor - end) > 1e-9:
        raise ValueError(f"[{start}, {end}] is not a whole number of periods of {tenor}")
    return start + tenor * np.arange(n + 1)


def _fmt(t):
    return f"{t * 12:.0f}M" if t < 1 else f"{t:g}Y"


class Instrument(ABC):
    priority = 0          # bootstrapping order between instruments of the same curve (lower first)

    @property
    @abstractmethod
    def pillar(self):
        """Date determined by the instrument (maturity or expiry)."""

    @abstractmethod
    def target(self, setup):
        """CurveKey (linear instruments) or model block in {"d", "f", "c", "x"} (options)."""

    def dependencies(self, setup):
        """Curves read by the instrument besides its target."""
        return set()

    @abstractmethod
    def quote(self, pricer):
        """Model quote, in the instrument's quotation."""

    @property
    @abstractmethod
    def label(self):
        """Short name for reports."""


# ---------------------------------------------------------------------- linear
@dataclass(frozen=True)
class Swap(Instrument):
    """Fixed vs floating swap on [start, end] under native collateral: fixed leg every `fixed_tenor`,
    floating leg on the index tenor. Par rate = sum_i (kappa Z_{t_{i-1}} - Z_{t_i}) / sum_k tau Z_{t_k}."""
    index: object
    end: float
    fixed_tenor: float = 1.0
    start: float = 0.0

    @property
    def pillar(self):
        return self.end

    @property
    def label(self):
        lead = f"{_fmt(self.start)}x" if self.start > 0 else ""
        return f"{self.index.name} swap {lead}{_fmt(self.end - self.start)}"

    def target(self, setup):
        return DiscountCurveKey(self.index.currency, setup.collateral_of(self.index.currency).name)

    def cashflows(self):
        """Float dates tf with coefficients cf (float leg = sum cf Z(tf)), fixed dates tx with weights wx."""
        tf = schedule(self.start, self.end, self.index.tenor)
        cf = np.zeros(len(tf))
        cf[:-1] += self.index.kappa
        cf[1:] -= 1.0
        tx = schedule(self.start, self.end, self.fixed_tenor)[1:]
        return tf, cf, tx, np.full(len(tx), self.fixed_tenor)

    def legs(self, Z):
        tf, cf, tx, wx = self.cashflows()
        return float((cf * Z(tf)).sum()), float((wx * Z(tx)).sum())

    def quote(self, pricer):
        key = self.target(pricer.setup)
        fl, A = self.legs(lambda T: pricer.curves.Z(key, T))
        return fl / A


@dataclass(frozen=True)
class FXForward(Instrument):
    """FX forward on the model pair (units of domestic per foreign) under domestic collateral:
    x^{df|d}_{0,T} = x_0 Z^{f|d}_{0,T} / Z^{d|d}_{0,T}. quotation "outright" or "points" = (F - x_0) pip_factor."""
    maturity: float
    quotation: str = "outright"
    pip_factor: float = 1e4

    def __post_init__(self):
        if self.quotation not in ("outright", "points"):
            raise ValueError(self.quotation)

    @property
    def pillar(self):
        return self.maturity

    @property
    def label(self):
        return f"FX forward {_fmt(self.maturity)}" + (" (points)" if self.quotation == "points" else "")

    def target(self, setup):
        return setup.key("fd")

    def dependencies(self, setup):
        return {setup.key("dd")}

    def outright(self, pricer):
        s, c = pricer.setup, pricer.curves
        return c.x0 * c.Z(s.key("fd"), self.maturity) / c.Z(s.key("dd"), self.maturity)

    def quote(self, pricer):
        F = self.outright(pricer)
        return F if self.quotation == "outright" else (F - pricer.curves.x0) * self.pip_factor


@dataclass(frozen=True)
class ResettingCCS(Instrument):
    """Resetting cross-currency basis swap on [start, end] under domestic collateral, periods theta =
    foreign index tenor: foreign index + B against the domestic index on a notional reset at each fixing.
        B = sum_j P_j (1 + theta L^r_j - (1 + theta L^{f|d}_j)) / (theta sum_j P_j),  P_j = Z^{f|d}_{0,T_j},
        1 + theta L^r_j = kappa^d P_{j-1}/P_j,  1 + theta L^{f|d}_j = kappa^f Z^{f|f}_{j-1}/Z^{f|f}_j K_j,
    K_j the Libor collateral convexity given by the pricer."""
    end: float
    start: float = 0.0
    priority = 1

    @property
    def pillar(self):
        return self.end

    @property
    def label(self):
        lead = f"{_fmt(self.start)}x" if self.start > 0 else ""
        return f"resetting CCS {lead}{_fmt(self.end - self.start)}"

    def target(self, setup):
        return setup.key("fd")

    def dependencies(self, setup):
        return {setup.key("ff")}

    def schedule(self, setup):
        return schedule(self.start, self.end, setup.foreign_index.tenor)

    def components(self, pricer):
        """Annuity A = theta sum P_j and spread B."""
        s = pricer.setup
        Tj, th = self.schedule(s), s.foreign_index.tenor
        P = pricer.curves.Z(s.key("fd"), Tj)
        one_Lr = s.domestic_index.kappa * P[:-1] / P[1:]
        fkey = ForwardCurveKey(s.foreign_index.name, s.domestic_collateral.name)
        one_L = 1.0 + th * np.array([pricer.forward_rate(fkey, a, b) for a, b in zip(Tj[:-1], Tj[1:])])
        A = th * P[1:].sum()
        return float(A), float(((one_Lr - one_L) * P[1:]).sum() / A)

    def quote(self, pricer):
        return self.components(pricer)[1]


# ---------------------------------------------------------------------- options
@dataclass(frozen=True)
class Swaption(Instrument):
    """ATM swaption on a native forward-starting swap, quoted as a normal vol."""
    underlying: Swap

    @classmethod
    def make(cls, index, expiry, tenor, fixed_tenor=1.0):
        return cls(Swap(index, expiry + tenor, fixed_tenor, expiry))

    @property
    def expiry(self):
        return self.underlying.start

    @property
    def pillar(self):
        return self.expiry

    @property
    def label(self):
        u = self.underlying
        return f"{u.index.name} swaption {_fmt(self.expiry)}x{_fmt(u.end - u.start)}"

    def target(self, setup):
        return setup.role(self.underlying.index.currency)

    def quote(self, pricer):
        return pricer.swaption_normal_vol(self)


@dataclass(frozen=True)
class FXOption(Instrument):
    """ATM FX option at `expiry`, quoted as a Black vol (collateral free in the LGCM)."""
    expiry: float

    @property
    def pillar(self):
        return self.expiry

    @property
    def label(self):
        return f"FX option {_fmt(self.expiry)}"

    def target(self, setup):
        return "x"

    def quote(self, pricer):
        return pricer.fx_black_vol(self)


@dataclass(frozen=True)
class CurrencyBasisSwaption(Instrument):
    """ATM option on the forward resetting CB spread, quoted as a normal vol under the annuity measure."""
    underlying: ResettingCCS

    @classmethod
    def make(cls, expiry, tenor):
        return cls(ResettingCCS(expiry + tenor, expiry))

    @property
    def expiry(self):
        return self.underlying.start

    @property
    def pillar(self):
        return self.expiry

    @property
    def label(self):
        u = self.underlying
        return f"CB vol {_fmt(self.expiry)}x{_fmt(u.end - u.start)}"

    def target(self, setup):
        return "c"

    def quote(self, pricer):
        return pricer.cb_normal_vol(self)


# ---------------------------------------------------------------------- quotes
@dataclass(frozen=True)
class Quote:
    instrument: Instrument
    value: float


@dataclass
class MarketQuotes:
    setup: object
    fx_spot: float
    quotes: list

    def of_type(self, *types):
        return [q for q in self.quotes if isinstance(q.instrument, types)]

    def for_target(self, target):
        return [q for q in self.quotes if q.instrument.target(self.setup) == target]

    def with_values(self, values):
        return replace(self, quotes=[Quote(q.instrument, float(v)) for q, v in zip(self.quotes, values, strict=True)])

    def without(self, *types):
        return replace(self, quotes=[q for q in self.quotes if not isinstance(q.instrument, types)])
```

- [ ] **Step 4: implement `pricers.py` (base and deterministic)**

```python
"""Pricers: the model view of the calibration instruments.

Pricer               curves, forward rates and the interface for option quotes
DeterministicPricer  deterministic collateral basis (DCB) model: Libor collateral convexity K = 1,
                     no volatility (option quotes raise NotImplementedError)
LGCMPricer           LGCM closed forms (added in Task 4)
"""
from .curves import DiscountCurveKey


class Pricer:
    def __init__(self, curves):
        self.curves = curves

    @property
    def setup(self):
        return self.curves.setup

    def libor_convexity(self, a, b):
        """K = (1 + theta L^{f|d}_{0,a}) / (1 + theta L^{f|f}_{0,a}) for the foreign index on [a, b]."""
        raise NotImplementedError

    def forward_rate(self, key, a, b):
        """L_{0,a;b-a} of the index key.index under key.collateral:
        1 + (b-a) L = kappa Z^{e|e}_{0,a} / Z^{e|e}_{0,b} * K, K = 1 under native collateral."""
        s = self.setup
        idx = s.index_by_name(key.index)
        native = DiscountCurveKey(idx.currency, s.collateral_of(idx.currency).name)
        ratio = idx.kappa * self.curves.Z(native, a) / self.curves.Z(native, b)
        if key.collateral != native.collateral:
            if (idx.currency, key.collateral) != (s.foreign, s.domestic_collateral.name):
                raise NotImplementedError(f"{key}: only the foreign index under domestic collateral")
            ratio *= self.libor_convexity(a, b)
        return (ratio - 1.0) / (b - a)

    def _no_vol(self, inst):
        raise NotImplementedError(f"{type(self).__name__} has no volatility: cannot quote {inst.label}")

    def swaption_normal_vol(self, swaption):
        self._no_vol(swaption)

    def fx_black_vol(self, option):
        self._no_vol(option)

    def cb_normal_vol(self, option):
        self._no_vol(option)


class DeterministicPricer(Pricer):
    """Deterministic collateral basis model: no convexity adjustment, no option quotes."""

    def libor_convexity(self, a, b):
        return 1.0
```

- [ ] **Step 5: run tests, expect PASS** — `python -m pytest convexity/tests/test_curves.py -q` → 12 passed.

- [ ] **Step 6: commit** — `git add` both modules and the test; message "Instruments, quotes and the deterministic pricer".

---

### Task 3: Generic curve bootstrap

**Files:**
- Create: `python/convexity/curve_calibration.py`
- Test: `python/convexity/tests/test_curves.py` (append)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `CalibrationError(Exception)`; `BootstrapReport(used, checks)`;
  `bootstrap(quotes, curves, pricer, targets=None, report=None) -> BootstrapReport`;
  `calibrate_curves(quotes, pricer_factory=DeterministicPricer) -> (CurveSet, BootstrapReport)`;
  `root_in_bracket(f, guess, lower=None, step=None) -> float`.

- [ ] **Step 1: failing tests** (append)

```python
from convexity.curve_calibration import calibrate_curves, bootstrap, BootstrapReport, CalibrationError


def linear_quotes(points=False):
    """Quotes generated from flat_curves(): the bootstrap must give back these curves."""
    pr = DeterministicPricer(flat_curves())
    insts = ([Swap(SOFR, T) for T in (1, 2, 5, 10, 30)] + [Swap(E3M, T) for T in (1, 2, 5, 10, 30)]
             + [FXForward(T, "points" if points else "outright") for T in (0.25, 0.5, 1.0)]
             + [ResettingCCS(T) for T in (1.0, 2.0, 5.0, 10.0, 30.0)])
    return MarketQuotes(SETUP, 1.1, [Quote(i, i.quote(pr)) for i in reversed(insts)])   # unsorted on purpose


def test_deterministic_bootstrap_recovers_flat_curves():
    cs, rep = calibrate_curves(linear_quotes())
    ref = flat_curves()
    T = np.linspace(0.1, 40, 50)
    for kind in ("dd", "ff", "fd"):
        np.testing.assert_allclose(cs.Z0(kind, T), ref.Z0(kind, T), rtol=1e-12)
    # the 1Y CCS falls inside the FX forward section: reported as a check, not used
    assert [q.instrument.label for q in rep.checks] == ["resetting CCS 1Y"]
    pr = DeterministicPricer(cs)
    for q in rep.used + rep.checks:
        assert q.instrument.quote(pr) == pytest.approx(q.value, abs=1e-12)


def test_points_and_outrights_give_the_same_curve():
    a, _ = calibrate_curves(linear_quotes(points=False))
    b, _ = calibrate_curves(linear_quotes(points=True))
    np.testing.assert_allclose(a.Z0("fd", [0.3, 0.9, 7.0]), b.Z0("fd", [0.3, 0.9, 7.0]), rtol=1e-13)


def test_fx_forward_curve_is_model_free():
    """Z^{f|d}(T) = F(T) Z^{d|d}(T) / x0 at every FX forward pillar."""
    mq = linear_quotes()
    cs, _ = calibrate_curves(mq)
    for q in mq.of_type(FXForward):
        T = q.instrument.maturity
        assert cs.Z0("fd", T) == pytest.approx(q.value * cs.Z0("dd", T) / 1.1, rel=1e-13)


def test_missing_dependency_is_reported():
    mq = MarketQuotes(SETUP, 1.1, [Quote(FXForward(1.0), 1.12)])
    with pytest.raises(ValueError, match="Z\\[USD\\|USD_SOFR\\]"):
        calibrate_curves(mq)


def test_partial_bootstrap_targets_only():
    mq = linear_quotes()
    cs = CurveSet(SETUP, 1.1)
    rep = bootstrap(mq, cs, DeterministicPricer(cs), targets=[SETUP.key("dd"), SETUP.key("ff")])
    assert set(cs.keys()) == {SETUP.key("dd"), SETUP.key("ff")} and len(rep.used) == 10
```

- [ ] **Step 2: run, expect ImportError.**

- [ ] **Step 3: implement `curve_calibration.py`**

```python
"""Generic sequential bootstrap of discount curves from linear instruments.

Each linear instrument (Swap, FXForward, ResettingCCS) declares the CurveKey it calibrates, its pillar
and the curves it depends on. Target curves are built in dependency order. On each curve, the quotes
are sorted by (priority, pillar); each quote adds a node at its pillar, solved by a 1D root search of
instrument.quote(pricer) - market value. A quote whose pillar is not beyond the last node of its
curve is not used, and is reported as a check. Native discount curves have flat forwards; a curve
off native collateral is Z^{e|e} times a flat-forward collateral basis curve (Remark "Which curve to
interpolate" in the notes).

The model only enters through the pricer (Libor collateral convexities of the resetting CCS):
DeterministicPricer is the DCB market practice, LGCMPricer the LGCM.
"""
from dataclasses import dataclass, field

from scipy.optimize import brentq

from .curves import CurveSet, DiscountCurveKey, PiecewiseFlatCurve, SpreadCurve
from .instruments import FXForward, ResettingCCS, Swap
from .pricers import DeterministicPricer

LINEAR = (Swap, FXForward, ResettingCCS)


class CalibrationError(Exception):
    """A calibration target cannot be matched (e.g. variance squeeze, no root)."""


@dataclass
class BootstrapReport:
    used: list = field(default_factory=list)     # quotes that set a node
    checks: list = field(default_factory=list)   # quotes inside an already built section


def root_in_bracket(f, guess, lower=None, step=None, max_expand=60):
    """Root of a monotone f, bracketed by expanding [guess - step, guess + step] (bounded below by lower)."""
    step = step if step is not None else max(abs(guess), 1e-3)
    lo, hi = guess - step, guess + step
    if lower is not None:
        lo = max(lo, lower)
    flo, fhi = f(lo), f(hi)
    for _ in range(max_expand):
        if flo * fhi <= 0:
            return brentq(f, lo, hi, xtol=1e-15, rtol=4e-16, maxiter=500)
        step *= 2
        if lower is None or lo > lower:
            lo = guess - step if lower is None else max(guess - step, lower)
            flo = f(lo)
        hi = guess + step
        fhi = f(hi)
    raise CalibrationError(f"no root found around {guess} (f({lo:.3g}) = {flo:.3g}, f({hi:.3g}) = {fhi:.3g})")


def _new_curve(key, curves):
    native = DiscountCurveKey(key.currency, curves.setup.collateral_of(key.currency).name)
    return PiecewiseFlatCurve() if key == native else SpreadCurve(curves[native])


def _target_order(quotes, setup):
    groups = {}
    for q in quotes:
        groups.setdefault(q.instrument.target(setup), []).append(q)
    deps = {t: set().union(*(q.instrument.dependencies(setup) for q in qs)) - {t} for t, qs in groups.items()}
    for t, ds in deps.items():
        for d in ds - set(groups):
            raise ValueError(f"no quotes for {d}, needed by the quotes of {t}")
    order, done = [], set()
    while len(order) < len(groups):
        ready = [t for t in groups if t not in done and deps[t] <= done]
        if not ready:
            raise ValueError(f"cyclic curve dependencies: {deps}")
        for t in sorted(ready, key=str):
            order.append(t)
            done.add(t)
    return order, groups


def _bootstrap_curve(key, quotes, curves, pricer, report):
    curve = _new_curve(key, curves)
    curves[key] = curve
    for q in sorted(quotes, key=lambda q: (q.instrument.priority, q.instrument.pillar)):
        T = q.instrument.pillar
        if curve.pillars and T <= curve.pillars[-1] + 1e-10:
            report.checks.append(q)
            continue
        guess = curve.forwards[-1] if curve.forwards else 0.0
        curve.add_node(T, guess)

        def residual(x):
            curve.set_last(x)
            return q.instrument.quote(pricer) - q.value
        curve.set_last(root_in_bracket(residual, guess, step=0.01))
        report.used.append(q)


def bootstrap(quotes, curves, pricer, targets=None, report=None):
    """Bootstrap the target curves (default: all) of the linear quotes into `curves`, in place."""
    report = report if report is not None else BootstrapReport()
    order, groups = _target_order(quotes.of_type(*LINEAR), quotes.setup)
    for key in order:
        if targets is None or key in targets:
            _bootstrap_curve(key, groups[key], curves, pricer, report)
    return report


def calibrate_curves(quotes, pricer_factory=DeterministicPricer):
    """Discount curves of all targets of the linear quotes. Default: DCB model (no convexity, no vols).
    pricer_factory(curves) -> Pricer, e.g. lambda c: LGCMPricer(model, c)."""
    curves = CurveSet(quotes.setup, quotes.fx_spot)
    report = bootstrap(quotes, curves, pricer_factory(curves))
    return curves, report
```

- [ ] **Step 4: run tests, expect PASS** (17 passed). If `test_missing_dependency_is_reported` fails because the dependency check runs before the target check, adjust the message match only — the behaviour (ValueError naming the missing curve) is the requirement.

- [ ] **Step 5: commit** — "Generic curve bootstrap with a pricer (DCB by default)".

---

### Task 4: LGCM pricer

**Files:**
- Modify: `python/convexity/pricers.py` (append `LGCMPricer` and swaption functions)
- Test: `python/convexity/tests/test_calibration.py` (create)

**Interfaces:**
- Consumes: `lgcm.Model` (`idx`, `G`, `covint`, `Y`, `sig`, `beta`, `R`), `market.libor_convexity`,
  `market.fx_total_variance`, `gaussian_approx.SwapSetup/approx/exact`.
- Produces: `native_swaption_variance(model, factors, swap, Z) -> float`,
  `native_swaption_exact_vol(model, i, swap, Z) -> float`,
  `LGCMPricer(model, curves, exact=True)` implementing the `Pricer` interface.

- [ ] **Step 1: failing tests** (`tests/test_calibration.py`)

```python
"""LGCM pricer (Task 4), calibration (Task 5), repricing checks (Task 6)."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from convexity import gaussian_approx as ga
from convexity.lgcm import Model
from convexity.curves import Index, LGCMSetup, CurveSet, PiecewiseFlatCurve, SpreadCurve
from convexity.instruments import (Swap, ResettingCCS, Swaption, FXOption, CurrencyBasisSwaption,
                                               schedule)
from convexity.pricers import (LGCMPricer, DeterministicPricer, native_swaption_variance,
                                           native_swaption_exact_vol)
from convexity.market import libor_convexity, fx_total_variance

SOFR = Index("USD_SOFR", "USD")
ESTR = Index("EUR_ESTR", "EUR")
E3M = Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.0010)
SETUP = LGCMSetup(SOFR, ESTR, foreign_index=E3M)


def small_model(scale=1.0):
    knots = [0, 1, 3, 7, 15, 60]
    n = len(knots) - 1
    ramp = np.linspace(1.0, 0.8, n)
    R = np.array([[1, .6, -.2, .25], [.6, 1, -.4, .3], [-.2, -.4, 1, .35], [.25, .3, .35, 1]])
    return Model(knots, [("d", scale * 0.009 * ramp, np.full(n, 0.03)), ("f", scale * 0.008 * ramp, np.full(n, 0.02)),
                         ("c", scale * 0.003 * ramp, np.full(n, 0.10))], R, scale * 0.08 * ramp)


def curves():
    cs = CurveSet(SETUP, 1.1)
    cs[SETUP.key("dd")] = PiecewiseFlatCurve([2, 10, 50], [0.042, 0.038, 0.04])
    cs[SETUP.key("ff")] = PiecewiseFlatCurve([2, 10, 50], [0.02, 0.026, 0.027])
    cs[SETUP.key("fd")] = SpreadCurve(cs[SETUP.key("ff")], PiecewiseFlatCurve([5, 50], [-0.001, -0.0015]))
    return cs


def test_native_swaption_exact_closed_form_matches_quadrature():
    m, cs = small_model(), curves()
    swap = Swap(E3M, 15.0, 1.0, 5.0)
    Z = lambda T: cs.Z0("ff", T)
    (i,) = m.idx["f"]
    T0 = swap.start
    tf, cf, tx, wx = swap.cashflows()
    K = swap.legs(Z)[0] / swap.legs(Z)[1]
    y = m.covint((i, "M", T0), (i, "M", T0), 0.0, T0)
    t = np.r_[tf, tx]
    c = np.r_[cf, -K * wx]
    P = Z(t) / Z(T0)
    G = np.array([m.G(i, T0, T) for T in t])
    V = lambda z: (c * P * np.exp(-G * z - 0.5 * G * G * y)).sum()
    s = np.sqrt(y)
    price = Z(T0) * quad(lambda z: max(V(z), 0.0) * norm.pdf(z / s) / s, -12 * s, 12 * s, limit=400,
                          epsabs=1e-14, epsrel=1e-12)[0]
    A0 = (wx * Z(tx)).sum()
    vol_quad = price / (A0 * np.sqrt(T0 / (2 * np.pi)))
    assert native_swaption_exact_vol(m, i, swap, Z) == pytest.approx(vol_quad, rel=1e-9)


def test_gaussian_swaption_variance_converges_to_exact():
    """Relative error of the Gaussian vol is O(scale^2) (asymptotic in the volatility)."""
    cs = curves()
    swap = Swap(E3M, 15.0, 1.0, 5.0)
    Z = lambda T: cs.Z0("ff", T)
    errs = []
    for scale in (0.5, 1.0, 2.0):
        m = small_model(scale)
        (i,) = m.idx["f"]
        exact = native_swaption_exact_vol(m, i, swap, Z)
        approx = np.sqrt(native_swaption_variance(m, [i], swap, Z) / 5.0)
        errs.append(abs(approx / exact - 1))
    assert errs[1] / errs[0] == pytest.approx(4, rel=0.15) and errs[2] / errs[1] == pytest.approx(4, rel=0.15)


def test_gaussian_variance_matches_gaussian_approx_native():
    """Same schedule on both legs: identical to gaussian_approx.approx_native (independent code)."""
    m, cs = small_model(), curves()
    swap = Swap(E3M, 15.0, 0.25, 5.0)          # kappa_f on both sides (SwapSetup reads curves.kappa_f)
    s = ga.SwapSetup(T0=5.0, theta=0.25, n=40, curves=cs)
    ref = ga.approx_native(m, s, ccy="f")
    v = native_swaption_variance(m, m.idx["f"], swap, lambda T: cs.Z0("ff", T))
    assert v == pytest.approx(ref["V"], rel=1e-12)


def test_lgcm_pricer_ccs_and_cb_vol_match_gaussian_approx():
    m, cs = small_model(), curves()
    pr = LGCMPricer(m, cs, exact=False)
    cbs = CurrencyBasisSwaption.make(5.0, 2.0)
    a = ga.approx(m, ga.SwapSetup(T0=5.0, theta=0.25, n=8, curves=cs))
    assert cbs.underlying.quote(pr) == pytest.approx(a["B0"], rel=1e-13)
    assert cbs.quote(pr) == pytest.approx(np.sqrt(a["VB"] / 5.0), rel=1e-13)
    ex = ga.exact(m, ga.SwapSetup(T0=5.0, theta=0.25, n=8, curves=cs))
    assert cbs.quote(LGCMPricer(m, cs)) == pytest.approx(ex["B"]["atm_normal_vol"], rel=1e-13)


def test_lgcm_pricer_convexity_and_fx_vol():
    m, cs = small_model(), curves()
    pr = LGCMPricer(m, cs)
    assert pr.libor_convexity(10.0, 10.25) == libor_convexity(m, 0.0, 10.0, 10.25)
    assert pr.libor_convexity(10.0, 10.25) > 1.0
    m.sig[m.idx["c"][0]][:] *= 2                       # the cache must see model changes
    assert pr.libor_convexity(10.0, 10.25) == libor_convexity(m, 0.0, 10.0, 10.25)
    assert FXOption(5.0).quote(pr) == pytest.approx(np.sqrt(fx_total_variance(m, 5.0) / 5.0), rel=1e-14)
    # LGCM CCS spread differs from the DCB one by the convexity only
    ccs = ResettingCCS(20.0)
    assert ccs.quote(pr) < ccs.quote(DeterministicPricer(cs))
```

- [ ] **Step 2: run, expect ImportError** — `python -m pytest convexity/tests/test_calibration.py -q`.

- [ ] **Step 3: implement** (append to `pricers.py`, and extend its docstring/imports)

```python
import numpy as np
from math import sqrt, pi
from scipy.optimize import brentq
from scipy.stats import norm

from . import gaussian_approx as ga
from .market import libor_convexity, fx_total_variance


def native_swaption_variance(model, factors, swap, Z):
    """Normal variance of the forward swap rate at T0 under its annuity measure, Gaussian approximation
    (Remark "Native swaps"; separate fixed and float schedules): q^T y_{T0} q with
    q = (-sum_m cf_m G_m P_m + Y sum_k w_k G_k P_k) / A on the given factors of the block."""
    T0 = swap.start
    tf, cf, tx, wx = swap.cashflows()
    Pf, Px = Z(tf) / Z(T0), Z(tx) / Z(T0)
    A = (wx * Px).sum()
    Y = (cf * Pf).sum() / A
    q = np.array([(-(cf * Pf * [model.G(i, T0, T) for T in tf]).sum()
                   + Y * (wx * Px * [model.G(i, T0, T) for T in tx]).sum()) / A for i in factors])
    y = np.array([[model.covint((i, "M", T0), (j, "M", T0), 0.0, T0) for j in factors] for i in factors])
    return float(q @ y @ q)


def native_swaption_exact_vol(model, i, swap, Z):
    """ATM normal vol from the exact one-factor price (Proposition "Exact one factor swaption").
    Under Q^{T0}, z = z^{a,i}_{T0} ~ N(0, y) and Z_{T0,T} = P(T) exp(-G z - G^2 y/2). The swap value at
    the ATM strike K, V(z) = sum_m c_m Z_{T0,t_m}, is increasing in z; with V(z*) = 0,
    E^{T0}[V^+] = sum_m c_m P_m N(-(z* + G_m y)/sqrt(y))."""
    T0 = swap.start
    tf, cf, tx, wx = swap.cashflows()
    fl, A0 = swap.legs(Z)
    K = fl / A0
    t, c = np.r_[tf, tx], np.r_[cf, -K * wx]
    P = Z(t) / Z(T0)
    G = np.array([model.G(i, T0, T) for T in t])
    y = model.covint((i, "M", T0), (i, "M", T0), 0.0, T0)
    s = sqrt(y)
    V = lambda z: float((c * P * np.exp(-G * z - 0.5 * G * G * y)).sum())
    lo, hi = -12 * s, 12 * s
    if not V(lo) < 0 < V(hi):
        raise ValueError(f"{swap.label}: swap value not increasing in the state, no exact closed form")
    zs = brentq(V, lo, hi, xtol=1e-14 * s, rtol=4e-16)
    price = Z(T0) * float((c * P * norm.cdf(-(zs + G * y) / s)).sum())
    return price / (A0 * sqrt(T0 / (2 * pi)))


class LGCMPricer(Pricer):
    """LGCM closed forms. exact=True: one-factor native swaptions by the exact closed form and CB vols
    by exact 2D quadrature (gaussian_approx.exact); otherwise, or with several factors in the block,
    the Gaussian approximations. CB vols need one f and one c factor (gaussian_approx)."""

    def __init__(self, model, curves, exact=True):
        super().__init__(curves)
        self.model, self.exact = model, exact
        self._K = {}

    def _version(self):
        m = self.model
        return hash(tuple(a.tobytes() for a in m.sig + m.beta) + (m.R.tobytes(),))

    def libor_convexity(self, a, b):
        key = (self._version(), a, b)
        if key not in self._K:
            self._K[key] = libor_convexity(self.model, 0.0, a, b)
        return self._K[key]

    def swaption_normal_vol(self, swaption):
        u = swaption.underlying
        role = self.setup.role(u.index.currency)
        key = self.setup.key(role * 2)
        Z = lambda T: self.curves.Z(key, T)
        factors = self.model.idx[role]
        if self.exact and len(factors) == 1:
            return native_swaption_exact_vol(self.model, factors[0], u, Z)
        return sqrt(native_swaption_variance(self.model, factors, u, Z) / u.start)

    def fx_black_vol(self, option):
        return sqrt(fx_total_variance(self.model, option.expiry) / option.expiry)

    def _cb_setup(self, option):
        u = option.underlying
        th = self.setup.foreign_index.tenor
        return ga.SwapSetup(T0=u.start, theta=th, n=len(u.schedule(self.setup)) - 1, curves=self.curves)

    def cb_normal_vol(self, option):
        m = self.model
        if len(m.idx["f"]) != 1 or len(m.idx["c"]) != 1:
            raise NotImplementedError("CB vols need exactly one f and one c factor")
        s = self._cb_setup(option)
        if self.exact:
            return float(ga.exact(m, s)["B"]["atm_normal_vol"])
        return sqrt(ga.approx(m, s)["VB"] / s.T0)
```

- [ ] **Step 4: run tests, expect PASS** (5 passed). Measure and note the ratios printed by
  `test_gaussian_swaption_variance_converges_to_exact` (add `print(errs)` temporarily, run with `-s`).

- [ ] **Step 5: commit** — "LGCM pricer: exact one-factor swaption, Gaussian multi-factor variance, CB and FX vols".

---

### Task 5: Model calibration

**Files:**
- Create: `python/convexity/calibration.py`, `python/convexity/market_data.py`
- Test: `python/convexity/tests/test_calibration.py` (append)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `BlockSpec(betas=(0.03,), weights=None, level=0.01)`;
  `ModelSpec(d, f, c, corr, fx_level=0.1, horizon=60.0, exact=True, tol=1e-12, max_iter=30)`;
  `default_spec()`; `Calibration(model, curves, quotes, spec, report, diagnostics)` with
  `pricer(exact=None)`, `levels(block) -> (knots, values)`; `calibrate(quotes, spec=None) -> Calibration`;
  `market_data.example_setup()`, `market_data.example_quotes() -> MarketQuotes`.

- [ ] **Step 1: failing tests** (append)

```python
import dataclasses
from convexity.calibration import calibrate, default_spec, BlockSpec, ModelSpec, CalibrationError
from convexity.curve_calibration import calibrate_curves
from convexity.market_data import example_quotes
from convexity.instruments import Quote, MarketQuotes, FXForward

MQ = example_quotes()


@pytest.fixture(scope="module")
def cal():
    return calibrate(MQ)


def test_calibration_reprices_all_quotes(cal):
    pr = cal.pricer()
    for q in MQ.quotes:
        tol = 1e-12 if isinstance(q.instrument, (Swap, FXForward, ResettingCCS)) else 1e-10
        if isinstance(q.instrument, FXForward) and q.instrument.quotation == "points":
            tol = 1e-8
        assert q.instrument.quote(pr) == pytest.approx(q.value, abs=tol), q.instrument.label
    assert cal.diagnostics["fixed_point_iterations"] <= 10


def test_round_trip_recovers_parameters(cal):
    """Quotes generated by the calibrated model are calibrated back to the same model."""
    pr = cal.pricer()
    synth = MQ.with_values([q.instrument.quote(pr) for q in MQ.quotes])
    spec2 = dataclasses.replace(cal.spec, c=dataclasses.replace(cal.spec.c, level=0.02))   # other start
    cal2 = calibrate(synth, spec2)
    for i in range(cal.model.nr + 1):
        np.testing.assert_allclose(cal2.model.sig[i], cal.model.sig[i], rtol=1e-8)
    T = np.linspace(0.1, 40, 30)
    np.testing.assert_allclose(cal2.curves.Z0("fd", T), cal.curves.Z0("fd", T), rtol=1e-12)


def test_input_order_does_not_matter(cal):
    shuffled = dataclasses.replace(MQ, quotes=list(np.random.default_rng(3).permutation(np.array(MQ.quotes, dtype=object))))
    cal2 = calibrate(shuffled)
    for i in range(cal.model.nr + 1):
        np.testing.assert_allclose(cal2.model.sig[i], cal.model.sig[i], rtol=1e-12)


def test_dcb_limit():
    """Without CB vols and with sigma^c = 0, K = 1 and the LGCM curve is the DCB curve."""
    mq = MQ.without(CurrencyBasisSwaption)
    spec = dataclasses.replace(default_spec(), c=BlockSpec((0.1,), level=0.0))
    cal0 = calibrate(mq, spec)
    dcb, _ = calibrate_curves(mq)
    T = np.linspace(0.1, 40, 30)
    np.testing.assert_allclose(cal0.curves.Z0("fd", T), dcb.Z0("fd", T), rtol=1e-14)
    assert cal0.diagnostics["fixed_point_iterations"] == 1


def test_collateral_convexity_moves_the_long_end(cal):
    """Same quotes: the LGCM Z^{f|d} differs from the DCB one beyond the FX forwards, not before."""
    dcb, _ = calibrate_curves(MQ)
    assert cal.curves.Z0("fd", 0.9) == pytest.approx(dcb.Z0("fd", 0.9), rel=1e-14)
    assert abs(cal.curves.Z0("fd", 30.0) / dcb.Z0("fd", 30.0) - 1) > 1e-4


def test_variance_squeeze_raises():
    low_fx = [Quote(q.instrument, 0.01) if isinstance(q.instrument, FXOption) and q.instrument.expiry == 10 else q
              for q in MQ.quotes]
    with pytest.raises(CalibrationError, match="FX option 10Y"):
        calibrate(dataclasses.replace(MQ, quotes=low_fx))


def test_duplicate_expiry_raises():
    dup = MQ.quotes + [q for q in MQ.quotes if isinstance(q.instrument, FXOption)][:1]
    with pytest.raises(ValueError, match="FX option"):
        calibrate(dataclasses.replace(MQ, quotes=dup))


def test_two_factor_domestic_block():
    spec = default_spec()
    corr = np.eye(5)
    corr[2:, 2:] = spec.corr[1:, 1:]
    corr[0, 1] = corr[1, 0] = 0.3
    corr[0, 2:] = corr[2:, 0] = spec.corr[0, 1:]
    corr[1, 2:] = corr[2:, 1] = 0.5 * spec.corr[0, 1:]
    spec2 = dataclasses.replace(spec, d=BlockSpec((0.03, 0.5), (1.0, 0.8), 0.006), corr=corr)
    cal2 = calibrate(MQ, spec2)
    pr = cal2.pricer()
    for q in MQ.of_type(Swaption):
        assert q.instrument.quote(pr) == pytest.approx(q.value, abs=1e-12)
    assert "d swaptions: Gaussian approximation (2 factors)" in cal2.diagnostics["approximations"]
    k, lam = cal2.levels("d")
    np.testing.assert_allclose(cal2.model.sig[1], 0.8 * cal2.model.sig[0])
```

- [ ] **Step 2: run, expect ImportError.**

- [ ] **Step 3: implement `calibration.py`**

```python
"""Calibration of the Linear Gaussian Collateral Model (section "The Linear Gaussian Collateral Model:
Calibration" of the notes).

Model: blocks d, f, c with n_d, n_f, n_c factors and the FX driver. Mean reversions and correlations
are marked. In each block sigma^{a,i}_t = lambda^a_t w^{a,i} with fixed weights; lambda^a and sigma^x
are piecewise constant on the block's calibration expiries (flat extrapolation). The model knots are
the union of the expiries.

Algorithm "LGCM calibration":
1. native curves Z^{d|d}, Z^{f|f} from swaps (model-free bootstrap);
2. lambda^d, lambda^f from ATM swaption normal vols, expiry by expiry: Gaussian approximation guess
   (the segment variance is v0 + lambda^2 v1), then the exact one-factor price;
3. fixed point: Z^{f|d} from FX forwards and resetting CCS with the LGCM Libor convexities, then
   lambda^c from CB spread vols (Gaussian guess, exact 2D quadrature), until lambda^c and Z^{f|d}
   no longer move;
4. sigma^x from ATM FX vols: the total variance is quadratic in sigma^x on the segment (exact).
Finally the off-market curve Z^{d|f} is added as a model output.
"""
from dataclasses import dataclass, field
from math import sqrt
import time

import numpy as np

from .curve_calibration import BootstrapReport, CalibrationError, bootstrap, root_in_bracket
from .curves import CurveSet, FunctionCurve
from .instruments import CurrencyBasisSwaption, FXOption, Swaption
from .lgcm import Model
from .market import fx_total_variance, lgcm_Zdf0
from .pricers import DeterministicPricer, LGCMPricer, native_swaption_variance

__all__ = ["BlockSpec", "ModelSpec", "default_spec", "Calibration", "calibrate", "CalibrationError"]


@dataclass(frozen=True)
class BlockSpec:
    """Factors of one block: constant mean reversions, fixed vol weights, initial / marked level lambda."""
    betas: tuple = (0.03,)
    weights: tuple = None
    level: float = 0.01

    def __post_init__(self):
        if self.weights is None:
            object.__setattr__(self, "weights", (1.0,) * len(self.betas))
        if len(self.weights) != len(self.betas):
            raise ValueError("one weight per factor")

    @property
    def n(self):
        return len(self.betas)


@dataclass(frozen=True)
class ModelSpec:
    d: BlockSpec
    f: BlockSpec
    c: BlockSpec
    corr: np.ndarray            # order d..., f..., c..., x
    fx_level: float = 0.10
    horizon: float = 60.0
    exact: bool = True
    tol: float = 1e-12
    max_iter: int = 30

    def __post_init__(self):
        n = self.d.n + self.f.n + self.c.n + 1
        R = np.asarray(self.corr, float)
        if R.shape != (n, n) or not np.allclose(R, R.T) or not np.allclose(np.diag(R), 1):
            raise ValueError(f"corr must be a {n}x{n} correlation matrix")
        if np.linalg.eigvalsh(R).min() <= 0:
            raise ValueError("corr must be positive definite")


def default_spec():
    """One factor per block. Correlations (d, f, c, x) as in the test model of the notes."""
    corr = np.array([[1.0, 0.6, -0.2, 0.25],
                     [0.6, 1.0, -0.4, 0.3],
                     [-0.2, -0.4, 1.0, 0.35],
                     [0.25, 0.3, 0.35, 1.0]])
    return ModelSpec(BlockSpec((0.03,), level=0.009), BlockSpec((0.02,), level=0.008),
                     BlockSpec((0.10,), level=0.003), corr)


@dataclass
class Calibration:
    model: Model
    curves: CurveSet
    quotes: object
    spec: ModelSpec
    report: BootstrapReport
    diagnostics: dict = field(default_factory=dict)

    def pricer(self, exact=None):
        return LGCMPricer(self.model, self.curves, self.spec.exact if exact is None else exact)

    def levels(self, block):
        """Knots and lambda^block (sigma^x for "x") on each piece."""
        m = self.model
        if block == "x":
            return m.knots, m.sig[m.ix].copy()
        i = m.idx[block][0]
        return m.knots, m.sig[i] / getattr(self.spec, block).weights[0]


# ---------------------------------------------------------------------- helpers
def _set_level(model, spec, block, t0, lam):
    """lambda^block = lam on every piece starting at or after t0 (flat extrapolation)."""
    k0 = int(np.searchsorted(model.knots, t0 - 1e-12))
    if block == "x":
        model.sig[model.ix][k0:] = lam
        return
    for i, w in zip(model.idx[block], getattr(spec, block).weights):
        model.sig[i][k0:] = lam * w


def _by_expiry(quotes):
    qs = sorted(quotes, key=lambda q: q.instrument.pillar)
    for a, b in zip(qs[:-1], qs[1:]):
        if abs(a.instrument.pillar - b.instrument.pillar) < 1e-10:
            raise ValueError(f"two quotes with the same expiry: {a.instrument.label}, {b.instrument.label}")
    return qs


def _build_model(spec, knots):
    n = len(knots) - 1
    factors = [(blk, np.full(n, b.level * w), np.full(n, beta))
               for blk, b in (("d", spec.d), ("f", spec.f), ("c", spec.c))
               for beta, w in zip(b.betas, b.weights)]
    return Model(knots, factors, spec.corr, np.full(n, spec.fx_level))


def _squeeze(q, detail):
    return CalibrationError(f"variance squeeze on {q.instrument.label} (market {q.value:.6g}): {detail}")


# ---------------------------------------------------------------------- steps
def _bootstrap_swaptions(model, spec, curves, quotes, block, diag):
    nf = getattr(spec, block).n
    exact = spec.exact and nf == 1
    if not exact and quotes:
        diag["approximations"].append(f"{block} swaptions: Gaussian approximation ({nf} factors)")
    key = curves.setup.key(block * 2)
    Z = lambda T: curves.Z(key, T)
    pricer = LGCMPricer(model, curves, exact=True)
    t0 = 0.0
    for q in quotes:
        u, T0 = q.instrument.underlying, q.instrument.expiry

        def var(lam):
            _set_level(model, spec, block, t0, lam)
            return native_swaption_variance(model, model.idx[block], u, Z)
        v0, v1 = var(0.0), var(1.0) - var(0.0)
        target = q.value ** 2 * T0
        if v0 >= target:
            raise _squeeze(q, f"variance {v0:.3g} already delivered before {t0}")
        lam = sqrt((target - v0) / v1)
        if exact:
            def residual(x):
                _set_level(model, spec, block, t0, x)
                return pricer.swaption_normal_vol(q.instrument) - q.value
            lam = root_in_bracket(residual, lam, lower=0.0, step=0.1 * lam)
        _set_level(model, spec, block, t0, lam)
        t0 = T0


def _bootstrap_cb_vols(model, spec, curves, quotes):
    approx, exact = LGCMPricer(model, curves, exact=False), LGCMPricer(model, curves, exact=True)
    t0 = 0.0
    for q in quotes:
        def residual(pricer):
            def f(x):
                _set_level(model, spec, "c", t0, x)
                return pricer.cb_normal_vol(q.instrument) - q.value
            return f
        if residual(approx)(0.0) >= 0:
            raise _squeeze(q, f"CB vol already delivered before {t0}")
        guess = max(model.sig[model.idx["c"][0]][int(np.searchsorted(model.knots, t0 - 1e-12))], 1e-4)
        lam = root_in_bracket(residual(approx), guess, lower=0.0, step=0.5 * guess)
        if spec.exact:
            lam = root_in_bracket(residual(exact), lam, lower=0.0, step=0.05 * lam)
        _set_level(model, spec, "c", t0, lam)
        t0 = q.instrument.expiry


def _bootstrap_fx(model, quotes):
    t0 = 0.0
    for q in quotes:
        T = q.instrument.expiry

        def w(s):
            _set_level(model, None, "x", t0, s)
            return fx_total_variance(model, T)
        w0, wp, wm = w(0.0), w(1.0), w(-1.0)
        a, b, c = 0.5 * (wp + wm) - w0, 0.5 * (wp - wm), w0 - q.value ** 2 * T
        disc = b * b - 4 * a * c
        root = (-b + sqrt(disc)) / (2 * a) if disc >= 0 else -1.0
        if root < 0:
            raise _squeeze(q, f"no sigma^x >= 0 on [{t0}, {T}] (rates and basis deliver {w0 / T:.4g} of variance per year)")
        _set_level(model, None, "x", t0, root)
        t0 = T


def _snapshot(model, curves):
    fd = curves[curves.setup.key("fd")]
    return np.r_[model.sig[model.idx["c"][0]], fd.forwards]


# ---------------------------------------------------------------------- driver
def calibrate(quotes, spec=None):
    spec = spec or default_spec()
    setup = quotes.setup
    t_start = time.perf_counter()
    swaptions = {b: _by_expiry([q for q in quotes.of_type(Swaption) if q.instrument.target(setup) == b]) for b in "df"}
    cb_vols = _by_expiry(quotes.of_type(CurrencyBasisSwaption))
    fx_vols = _by_expiry(quotes.of_type(FXOption))
    expiries = [q.instrument.pillar for q in swaptions["d"] + swaptions["f"] + cb_vols + fx_vols]
    if expiries and max(expiries) >= spec.horizon:
        raise ValueError(f"expiries must be before the horizon {spec.horizon}")
    model = _build_model(spec, np.unique(np.r_[0.0, expiries, spec.horizon]))
    diag = {"approximations": []}

    # 1. native curves
    curves = CurveSet(setup, quotes.fx_spot)
    report = bootstrap(quotes, curves, DeterministicPricer(curves), targets=[setup.key("dd"), setup.key("ff")])
    # 2. rates vols
    for b in "df":
        _bootstrap_swaptions(model, spec, curves, swaptions[b], b, diag)
    # 3. Z^{f|d} and lambda^c
    fd_report, history, prev = None, [], None
    for it in range(1, spec.max_iter + 1):
        fd_report = bootstrap(quotes, curves, LGCMPricer(model, curves, spec.exact), targets=[setup.key("fd")])
        if cb_vols:
            _bootstrap_cb_vols(model, spec, curves, cb_vols)
        snap = _snapshot(model, curves)
        change = np.inf if prev is None else float(np.abs(snap - prev).max())
        history.append(change)
        prev = snap
        if not cb_vols or change < spec.tol:
            break
    else:
        raise CalibrationError(f"Z^{{f|d}} / sigma^c fixed point did not converge: changes {history}")
    report.used += fd_report.used
    report.checks += fd_report.checks
    # 4. FX vol
    _bootstrap_fx(model, fx_vols)
    # model output: Z^{d|f}
    curves[setup.key("df")] = FunctionCurve(lambda T: lgcm_Zdf0(model, curves, T) if T > 0 else 1.0)
    diag.update(fixed_point_iterations=it, fixed_point_changes=history,
                seconds=time.perf_counter() - t_start)
    return Calibration(model, curves, quotes, spec, report, diag)
```

Note: the fixed point counts the first iteration (change = inf) — with CB vols it needs at least 2
iterations; `test_calibration_reprices_all_quotes` bounds it by 10. Record the measured count.

- [ ] **Step 4: implement `market_data.py`**

```python
"""Example market: USD (domestic, collateral SOFR) / EUR (foreign, collateral ESTR, index EURIBOR 3M).

Rounded quotes of the kind found on a broker screen (not live data). Units: rates and spreads in
decimals, FX forward points in pips (1e-4), vols in decimals (normal vols: 0.0100 = 100 bp/year).
"""
from .curves import Index, LGCMSetup
from .instruments import (CurrencyBasisSwaption, FXForward, FXOption, MarketQuotes, Quote, ResettingCCS, Swap,
                          Swaption)

SOFR = Index("USD_SOFR", "USD")
ESTR = Index("EUR_ESTR", "EUR")
EURIBOR3M = Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.0010)


def example_setup():
    return LGCMSetup(SOFR, ESTR, foreign_index=EURIBOR3M)


USD_SWAPS = {1: 4.30, 2: 3.95, 3: 3.80, 5: 3.70, 7: 3.72, 10: 3.80, 15: 3.90, 20: 3.92, 30: 3.80}
EUR_SWAPS = {1: 2.20, 2: 2.15, 3: 2.20, 5: 2.35, 7: 2.50, 10: 2.65, 15: 2.80, 20: 2.80, 30: 2.65}
FX_POINTS = {1 / 12: 19.5, 0.25: 58.0, 0.5: 114.0, 0.75: 168.0, 1.0: 222.0}
CCS_BP = {2: -8.0, 3: -10.0, 5: -12.0, 7: -13.0, 10: -14.0, 15: -13.0, 20: -12.0, 30: -10.0}
USD_SWAPTION_BP = {1: 105, 2: 108, 3: 110, 5: 108, 7: 104, 10: 98, 15: 90, 20: 84}
EUR_SWAPTION_BP = {1: 85, 2: 88, 3: 90, 5: 90, 7: 88, 10: 84, 15: 78, 20: 72}
FX_VOL = {1 / 12: 7.0, 0.25: 7.2, 0.5: 7.4, 1: 7.6, 2: 7.9, 3: 8.1, 5: 8.5, 7: 8.8, 10: 9.2, 15: 9.8}
CB_VOL_BP = {1: 9.0, 2: 9.5, 3: 10.0, 5: 10.0, 7: 10.0, 10: 9.5, 15: 9.0, 20: 8.5}
SWAPTION_TENOR, CB_VOL_TENOR = 10.0, 1.0


def example_quotes():
    q = [Quote(Swap(SOFR, T), r / 100) for T, r in USD_SWAPS.items()]
    q += [Quote(Swap(EURIBOR3M, T), r / 100) for T, r in EUR_SWAPS.items()]
    q += [Quote(FXForward(T, "points"), p) for T, p in FX_POINTS.items()]
    q += [Quote(ResettingCCS(T), b / 1e4) for T, b in CCS_BP.items()]
    q += [Quote(Swaption.make(SOFR, T, SWAPTION_TENOR), v / 1e4) for T, v in USD_SWAPTION_BP.items()]
    q += [Quote(Swaption.make(EURIBOR3M, T, SWAPTION_TENOR), v / 1e4) for T, v in EUR_SWAPTION_BP.items()]
    q += [Quote(FXOption(T), v / 100) for T, v in FX_VOL.items()]
    q += [Quote(CurrencyBasisSwaption.make(T, CB_VOL_TENOR), v / 1e4) for T, v in CB_VOL_BP.items()]
    return MarketQuotes(example_setup(), 1.10, q)
```

The swaption fixed leg is annual and the EURIBOR float leg quarterly; `Swaption.make` uses
`fixed_tenor=1.0`. If the calibration hits a variance squeeze or a negative implied level with these
numbers, adjust the offending market numbers (keep them realistic) and record why in the commit
message; do not change the algorithm to accommodate them.

- [ ] **Step 5: run tests, expect PASS** — `python -m pytest convexity/tests/test_calibration.py -q`.
  Then print `cal.diagnostics` and `cal.levels(b)` for each block (scratch script) and check the levels
  are positive and plausible (lambda^d ~ 0.6–1.2%, lambda^c ~ 0.1–0.5%, sigma^x ~ 5–10%).

- [ ] **Step 6: commit** — "LGCM calibration: swaptions, joint Z^{f|d}/sigma^c fixed point, FX vols; example market".

---

### Task 6: Repricing checks — analytic and Monte Carlo

**Files:**
- Modify: `python/convexity/schemes.py`, `python/convexity/montecarlo.py`
- Create: `python/convexity/calibration_check.py`
- Test: `python/convexity/tests/test_calibration.py` (append)

**Interfaces:**
- Consumes: Tasks 1–5, `Simulator.discount/deflator/fx_spot`, `SimResult`, `transitions.transition`.
- Produces: `PathStates` (dict of `SimResult` by date, `states(t)` lookup with rounding);
  `ExactScheme.simulate_dates(sim, dates, P, rng, antithetic) -> PathStates`;
  `Simulator.simulate_dates(dates, n_paths, seed=0, antithetic=True) -> PathStates`;
  `montecarlo.cv_mean_and_stderr(x, controls, means, antithetic=True) -> (mean, stderr)`;
  `CheckRow(label, kind, market, model, stderr=nan, approx=nan)` with `error`, `z`;
  `analytic_report(cal) -> list[CheckRow]`; `mc_report(cal, n_paths=2**16, seed=1, quotes=None, dt=0.25) -> list[CheckRow]`.

- [ ] **Step 1: failing tests** (append)

```python
from convexity.montecarlo import Simulator, cv_mean_and_stderr
from convexity.schemes import ExactScheme
from convexity.calibration_check import analytic_report, mc_report


def test_simulate_dates_martingales(cal):
    sim = Simulator(cal.model, scheme=ExactScheme(0.25))
    st = sim.simulate_dates([0.0, 2.0, 5.0], 2**14, seed=5)
    assert st(0.0).z.shape == (2**14, cal.model.nr) and np.all(st(0.0).z == 0)
    for T in (2.0, 5.0):
        for ccy, col, kind in (("d", "d", "dd"), ("f", "d", "fd"), ("f", "f", "ff")):
            D = sim.deflator(st(T), cal.curves, ccy, col)
            from convexity.montecarlo import mean_and_stderr
            mu, se = mean_and_stderr(D)
            assert abs(mu - cal.curves.Z0(kind, T)) < 4 * se


def test_cv_mean_and_stderr_reduces_variance():
    rng = np.random.default_rng(0)
    c = rng.standard_normal(2**12)
    x = 1.0 + 2.0 * c + 0.1 * rng.standard_normal(2**12)
    mu, se = cv_mean_and_stderr(x, [c], [0.0], antithetic=False)
    assert abs(mu - 1.0) < 4 * se and se < 0.01


def test_analytic_report(cal):
    rows = analytic_report(cal)
    assert len(rows) == len(MQ.quotes)
    assert max(abs(r.error) for r in rows if r.kind not in ("FXForward",)) < 1e-10
    cb = [r for r in rows if r.kind == "CurrencyBasisSwaption"]
    assert all(abs(r.approx / r.model - 1) < 1e-2 for r in cb)          # Gaussian approx gap, measured


@pytest.mark.slow
def test_mc_report_within_4_stderr(cal):
    qs = [q for q in MQ.quotes if q.instrument.pillar <= 10.0 and not (
        isinstance(q.instrument, Swaption) and q.instrument.expiry > 5)]
    rows = mc_report(cal, n_paths=2**15, seed=11, quotes=qs)
    bad = [(r.label, r.market, r.model, r.stderr) for r in rows if not abs(r.z) < 4]
    assert not bad, bad
```

Register the `slow` marker: create `python/pytest.ini` with
```
[pytest]
markers =
    slow: Monte Carlo repricing (run by default; deselect with -m "not slow")
```

- [ ] **Step 2: run, expect ImportError.**

- [ ] **Step 3: `schemes.py` additions**

```python
class PathStates(dict):
    """SimResult by date (same paths at every date). states(t) looks the date up to 1e-10."""

    def __call__(self, t):
        return self[round(float(t), 10)]
```

In `MCScheme`:
```python
    def simulate_dates(self, sim, dates, n_paths, rng, antithetic):
        """Joint state at every date of `dates` along the same paths."""
        raise NotImplementedError(f"{type(self).__name__} does not record intermediate dates")
```

In `ExactScheme`:
```python
    def simulate_dates(self, sim, dates, P, rng, antithetic):
        if not sim.is_lgcm:
            raise ValueError("the exact scheme only applies to the LGCM (kappa = 0)")
        m, nr = sim.m, sim.m.nr
        dates = np.unique(np.round(np.asarray(dates, float), 10))
        want = set(dates.tolist())
        g = np.union1d(self.grid(m, dates[-1]), dates)
        g = g[np.concatenate([[True], np.diff(g) > 1e-10])]
        n = nr + 4
        X = np.zeros((P, n))
        out = PathStates()

        def record(t):
            Yd = np.broadcast_to(m.Y(t), (P, nr, nr))
            vd = np.broadcast_to(m.vfc(t), (P, nr))
            out[round(float(t), 10)] = SimResult(float(t), X[:, :nr].copy(), X[:, nr].copy(), Yd, vd,
                                                 X[:, nr + 1].copy(), X[:, nr + 2].copy(), antithetic)
        if 0.0 in want:
            record(0.0)
        for t0, t1 in zip(g[:-1], g[1:]):
            F, mu, C, ixs = transitions.transition(m, t0, t1, "RN")
            w, U = np.linalg.eigh(C)
            X = X @ F.T + mu + self.normals(rng, P, n, antithetic) @ (U * np.sqrt(np.clip(w, 0, None))).T
            if round(float(t1), 10) in want:
                record(t1)
        return out
```
(`ixs` are `iZX = nr, iID = nr + 1, iIQ = nr + 2`, as in `transitions.transition`.)

`montecarlo.py`:
```python
    def simulate_dates(self, dates, n_paths, seed=0, antithetic=True):
        """State at every date of `dates` along the same paths (PathStates)."""
        if antithetic and n_paths % 2:
            raise ValueError("n_paths must be even with antithetic sampling")
        return self.scheme.simulate_dates(self, dates, n_paths, np.random.default_rng(seed), antithetic)
```
and
```python
def cv_mean_and_stderr(x, controls, means, antithetic=True):
    """E[x] with regression control variates of known means (antithetic pairs averaged first)."""
    pair = (lambda a: 0.5 * (a[0::2] + a[1::2])) if antithetic else (lambda a: a)
    x = pair(np.asarray(x, float))
    C = np.stack([pair(np.asarray(c, float)) for c in controls], axis=1) - np.asarray(means, float)
    beta = np.linalg.lstsq(C - C.mean(0), x - x.mean(), rcond=None)[0]
    y = x - C @ beta
    return y.mean(), y.std(ddof=1) / np.sqrt(len(y))
```
Re-export `PathStates` from `montecarlo.py` like the other scheme names.

- [ ] **Step 4: `calibration_check.py`**

```python
"""Do the calibrated model's delivered quotes match the market? Closed forms and Monte Carlo.

analytic_report  every quote repriced with the LGCM pricer (exact where available), and the Gaussian
                 approximation for swaptions and CB vols (column `approx`).
mc_report        every quote repriced by Monte Carlo with the exact LGCM scheme (one pass over the
                 union of the needed dates, deflators of Simulator.deflator). Option vols are implied
                 from straddle prices with the forward value as a control variate; the CCS floating leg
                 uses the known discount factors as control variates.
"""
from dataclasses import dataclass
from math import nan, pi, sqrt

import numpy as np
from scipy.stats import norm

from .instruments import CurrencyBasisSwaption, FXForward, FXOption, ResettingCCS, Swap, Swaption
from .market import libor_convexity
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
        return self.error / self.stderr


def analytic_report(cal):
    exact, approx = cal.pricer(exact=True), cal.pricer(exact=False)
    rows = []
    for q in cal.quotes.quotes:
        inst = q.instrument
        a = inst.quote(approx) if isinstance(inst, (Swaption, CurrencyBasisSwaption)) else nan
        rows.append(CheckRow(inst.label, type(inst).__name__, q.value, inst.quote(exact), approx=a))
    return rows


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
    u, role = q.instrument, cal.curves.setup.role(q.instrument.index.currency)
    tf, cf, tx, wx = u.cashflows()
    D = lambda t: sim.deflator(st(t), cal.curves, role, role)
    pv = sum(c * D(t) for t, c in zip(tf, cf)) - q.value * sum(w * D(t) for t, w in zip(tx, wx))
    A0 = (wx * cal.curves.Z0(role * 2, tx)).sum()
    mu, se = mean_and_stderr(pv)
    return q.value + mu / A0, se / A0


def _fx_forward(sim, st, cal, q):
    inst, T = q.instrument, q.instrument.maturity
    res, c = st(T), cal.curves
    F_mkt = q.value if inst.quotation == "outright" else c.x0 + q.value / inst.pip_factor
    mu, se = mean_and_stderr(sim.deflator(res, c, "d", "d") * (sim.fx_spot(res, c) - F_mkt))
    F, sF = F_mkt + mu / c.Z0("dd", T), se / c.Z0("dd", T)
    return (F, sF) if inst.quotation == "outright" else ((F - c.x0) * inst.pip_factor, sF * inst.pip_factor)


def _ccs(sim, st, cal, q):
    c, s = cal.curves, cal.curves.setup
    Tj = q.instrument.schedule(s)
    th = s.foreign_index.tenor
    P = c.Z0("fd", Tj)
    X = C1 = C2 = 0.0                      # path arrays after the first period
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
    """Normal vol from 0.5 E[D |V|] with the control E[D V] = 0: price = A0 vol sqrt(T0 / 2 pi)."""
    mu, se = cv_mean_and_stderr(0.5 * D * np.abs(V), [D * V], [0.0])
    k = A0 * sqrt(T0 / (2 * pi))
    return mu / k, se / k


def _swaption(sim, st, cal, q):
    u = q.instrument.underlying
    role = cal.curves.setup.role(u.index.currency)
    res = st(u.start)
    tf, cf, tx, wx = u.cashflows()
    Z0 = lambda T: cal.curves.Z0(role * 2, T)
    K = u.legs(Z0)[0] / u.legs(Z0)[1]
    disc = lambda t: sim.discount(res, cal.curves, role * 2, t) if t > u.start else np.ones(res.n_paths)
    V = sum(c * disc(t) for t, c in zip(tf, cf)) - K * sum(w * disc(t) for t, w in zip(tx, wx))
    return _straddle_vol(sim.deflator(res, cal.curves, role, role), V, (wx * Z0(tx)).sum(), u.start)


def _fx_option(sim, st, cal, q):
    T, c = q.instrument.expiry, cal.curves
    res = st(T)
    F, Z = c.fx_forward0(T), c.Z0("dd", T)
    D, x = sim.deflator(res, c, "d", "d"), sim.fx_spot(res, c)
    mu, se = cv_mean_and_stderr(0.5 * D * np.abs(x - F), [D * (x - F)], [0.0])
    # ATM Black: price = Z F (2 N(vol sqrt(T)/2) - 1)
    vol = 2 * norm.ppf(0.5 * (mu / (Z * F) + 1)) / sqrt(T)
    dprice = Z * F * sqrt(T) * norm.pdf(0.5 * vol * sqrt(T))
    return vol, se / dprice


def _cb_vol(sim, st, cal, q):
    c, s, m = cal.curves, cal.curves.setup, cal.model
    u = q.instrument.underlying
    T0, Tj, th = u.start, u.schedule(s), s.foreign_index.tenor
    res = st(T0)
    one = np.ones(res.n_paths)
    P = np.stack([one] + [sim.discount(res, c, "fd", t) for t in Tj[1:]], axis=1)
    Zf = np.stack([one] + [sim.discount(res, c, "ff", t) for t in Tj[1:]], axis=1)
    K = np.array([libor_convexity(m, T0, a, b) for a, b in zip(Tj[:-1], Tj[1:])])
    one_L = s.foreign_index.kappa * Zf[:, :-1] / Zf[:, 1:] * K
    one_Lr = s.domestic_index.kappa * P[:, :-1] / P[:, 1:]
    A = th * P[:, 1:].sum(1)
    B = ((one_Lr - one_L) * P[:, 1:]).sum(1) / A
    B0 = u.quote(cal.pricer())
    A0 = th * c.Z0("fd", Tj[1:]).sum()
    return _straddle_vol(sim.deflator(res, c, "f", "d"), A * (B - B0), A0, T0)


_MC = {Swap: _swap, FXForward: _fx_forward, ResettingCCS: _ccs, Swaption: _swaption, FXOption: _fx_option,
       CurrencyBasisSwaption: _cb_vol}


def mc_report(cal, n_paths=2**16, seed=1, quotes=None, dt=0.25):
    quotes = cal.quotes.quotes if quotes is None else quotes
    sim = Simulator(cal.model, scheme=ExactScheme(dt))
    dates = sorted(set().union(*(_dates(q.instrument, cal.curves.setup) for q in quotes)))
    st = sim.simulate_dates(dates, n_paths, seed)
    rows = []
    for q in quotes:
        val, se = _MC[type(q.instrument)](sim, st, cal, q)
        rows.append(CheckRow(q.instrument.label, type(q.instrument).__name__, q.value, float(val), float(se)))
    return rows
```
- [ ] **Step 5: run tests** — `python -m pytest convexity/tests/test_calibration.py -q`. Expected PASS.
  If a Monte Carlo row fails, do not widen the tolerance: use superpowers:systematic-debugging
  (check the deflator, the measure, the fixing dates) and fix the cause.

- [ ] **Step 6: measure for the notes** — scratch script (scratchpad dir) running
  `analytic_report(cal)` and `mc_report(cal, n_paths=2**17, seed=7)` on all quotes; save the printed
  table (max |error| per instrument type, max |z|, timings) — Task 8 quotes these numbers.

- [ ] **Step 7: commit** — "Calibration checks: analytic and Monte Carlo repricing; simulate_dates, control variates".

---

### Task 7: marimo notebook

**Files:**
- Create: `python/notebooks/lgcm_calibration.py`

**Interfaces:**
- Consumes: `example_quotes`, `calibrate`, `default_spec`, `BlockSpec`, `calibrate_curves`,
  `analytic_report`, `mc_report`, `market.fx_collateral_convexity`, `market.libor_convexity`.

- [ ] **Step 1: write the notebook** with these cells (each cell returns the names it defines,
  marimo style; first cell inserts `python/` into `sys.path`):

1. Title and explanation (markdown): what is calibrated to what, order of the algorithm.
2. Market quotes table (`mo.ui.table` of label / type / value in display units).
3. Controls: `mo.ui.slider` for rho^{fc} (-0.9..0.9, default -0.4), CB vol scale (0..3, default 1),
   beta^c (0.01..0.5, default 0.10), beta^f (−0.02..0.2, default 0.02); a `mo.ui.run_button` for the MC check
   and a `mo.ui.dropdown` for the number of paths (2^14, 2^15, 2^16).
4. Calibration cell: build `ModelSpec` from the sliders (replace the corr entry and betas; scale the CB vol
   quotes with `MQ.with_values`), `cal = calibrate(...)`, `dcb, _ = calibrate_curves(MQ)`; display timing
   and fixed point iterations.
5. Plots (matplotlib, 2×2): zero rates of Z^{d|d}, Z^{f|f}, Z^{f|d}, Z^{d|f}; instantaneous collateral basis
   forward q (LGCM vs DCB); lambda^d, lambda^f, lambda^c (step plots); sigma^x.
6. Analytic report table (`analytic_report`), errors in bp / vol points, approx column.
7. MC report (only when the run button is pressed): table with z-scores and a bar chart of z-scores.
8. Collateral convexity panel: K_j − 1 for quarterly periods to 30Y; (LGCM − DCB) Z^{f|d} zero rate in bp;
   FX forward under f collateral vs d collateral (`fx_collateral_convexity` − 1, bp); Z^{d|f} vs the DCB
   off-market curve x0 Z^{f|f}/F (bp).
9. Parameter impact table: for a few values of rho^{fc} (−0.8, −0.4, 0, 0.4) recalibrate and report the 30Y
   Z^{f|d} shift vs DCB, lambda^c at 10Y and the 10Y FX forward collateral convexity (cached with
   `functools.lru_cache` on the parameter tuple).

- [ ] **Step 2: run headless** — `cd python && marimo export html notebooks/lgcm_calibration.py -o /tmp-scratchpad/nb.html`
  (scratchpad directory). Expected: exit 0, no cell errors in the output. Also `marimo check notebooks/lgcm_calibration.py` if available.

- [ ] **Step 3: pin versions** — write the installed versions of numpy, scipy, sympy, pytest,
  matplotlib, marimo into `requirements.txt` as `==` pins.

- [ ] **Step 4: commit** — "marimo notebook: calibration results, parameter impact, collateral convexity".

---

### Task 8: Notes and documentation

**Files:**
- Modify: `docs/CollateralDiscounting.tex` (section "The Linear Gaussian Collateral Model: Calibration", lines ~3171–3514; keep the subsection "Interpolation and consistency with the model dynamic" unchanged)
- Modify: `python/convexity/README.md`, `python/convexity/__init__.py`, spec (MC valuation location)

- [ ] **Step 1: rewrite the section** with this content (LaTeX, using the notes' macros `\QZ`, `\f`, `\BS`, `\DCB`, `\RN`):
  - *Introduction*: keep the single-currency and cross-currency paragraphs; list the calibration targets
    (swaps, FX spot/forwards, resetting CCS, swaptions, FX options, CB vols) and the marked inputs (mean
    reversions, correlations, factor weights).
  - *Definition (Calibration instruments)*: par swap rate with separate fixed/float schedules and
    kappa = 1 + theta s; FX outright and points; resetting CB spread with K_j; ATM normal vol of swaptions and
    CB spreads under the annuity measure; ATM Black vol of FX options.
  - *Remark (Curve keys)*: Z^{e|c} identified by (currency, collateral index); off-market Z^{d|f} is a model output.
  - *Proposition (Model-free curves)*: native curves from native swaps; Z^{f|d}_{0,T} = x^{df|d}_{0,T} Z^{d|d}_{0,T}/x_0 from
    d-collateralised FX forwards. Proof: replication, one line each.
  - *Algorithm (Curve bootstrap with a convexity provider)*: dependency order, (priority, pillar) sort,
    one flat-forward node per pillar, the role of K_j; the DCB model is K_j = 1. Remark: quotes inside a
    built section are checks.
  - *Subsection: Calibration to single currency markets* (fill the empty subsubsections):
    *Proposition (Exact one factor swaption)* with statement and proof (exponential tilt: under Q^{T0}
    z ~ N(0,y), E[e^{-Gz - G^2y/2} 1_{z>z*}] = N(-(z*+Gy)/sqrt y)); *Remark (Gaussian guess)*: segment variance
    v0 + lambda^2 v1, closed-form initial guess, multi-factor blocks use the Gaussian approximation (asymptotic,
    error O(lambda^2) measured in Task 4); variance squeeze condition v0 < sigma_N^2 T0. Mean reversion: marked
    (keep the existing subsubsection titles, mark "Calibration of mean-reversion" as not implemented).
  - *Cross-currency*: keep "Necessity for joint calibration" and the bootstrapping intuition; replace the
    blank theorem items by *Proposition (Calibration constraints)*: (1) CB spread equation in terms of the
    segment collateral basis level and K_j (sigma^c); (2) CB vol = Gaussian approximation of
    Prop. gauss_cb_spread / exact quadrature; (3) FX vol: w(T) = a s^2 + b s + c on the last segment,
    s = sigma^x, with a = int_seg du > 0, b and c from covariance integrals; solution
    s = (-b + sqrt(b^2 - 4a(c - sigma_BS^2 T)))/(2a); variance squeeze iff no non-negative root.
  - *Algorithm (LGCM calibration)*: steps 1–4 of `calibration.py`, the fixed point, its stopping rule.
  - *Remark (Verification)*: measured numbers from Task 6 Step 6 (analytic: max errors per type; MC: max |z|
    over all quotes, paths, seed); measured fixed point iteration count; Gaussian approximation gap of CB vols.
  - *Remark (Collateral convexity in the calibrated curves)*: measured LGCM − DCB differences on the example
    market (30Y Z^{f|d} zero rate in bp, 30Y K − 1, 10Y FX forward collateral convexity), from the notebook
    or a scratch script.
  Remove the "[TODO: Remove this] Calibration to currency basis swaps" subsection only if it duplicates the new
  content; otherwise leave it.

- [ ] **Step 2: compile twice** — `cd docs && pdflatex -interaction=nonstopmode CollateralDiscounting.tex >/dev/null; pdflatex -interaction=nonstopmode CollateralDiscounting.tex | grep -E "Error|Warning.*(multiply|undefined)"`.
  Expected: 0 errors; only the three known warnings.

- [ ] **Step 3: README / `__init__`** — add the new modules to the module table and docstring, a
  "Calibration" section (usage snippet: `cal = calibrate(example_quotes()); analytic_report(cal)`), the
  "What is checked" rows for the new tests, the notebook command (`marimo edit python/notebooks/lgcm_calibration.py`),
  and the environment commands (`conda env create -f environment.yml`).

- [ ] **Step 4: commit** — "Notes: LGCM calibration section; README".

---

### Task 9: Final verification

- [ ] **Step 1:** full test suite: `cd python && python -m pytest convexity -q` → all pass (57 + new). Record the time.
- [ ] **Step 2:** fresh environment: `conda env create -f environment.yml -n convexity-check` then
  `conda run -n convexity-check python -m pytest convexity -q` (from `python/`) and the
  notebook export; remove the env afterwards (`conda env remove -n convexity-check`). If there is no
  network, say so in the report instead of claiming the environment was verified.
- [ ] **Step 3:** LaTeX compile check as in Task 8 Step 2.
- [ ] **Step 4:** use superpowers:requesting-code-review on the branch, then superpowers:finishing-a-development-branch.
