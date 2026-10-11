# Bonds and Repo Convexity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add McCloud's repo convexity (arXiv:1905.03316) to the notes and the numerical companion. It is embedded in the LGCM as a forward collateral convexity (bond curve = native curve, OIS = collateral, basis b = collateral basis) and covers zero-coupon and coupon bonds. A marimo notebook shows the impacts.

**Architecture:** There is one new module, `python/convexity/repo.py`. It holds a `RepoModel` adapter that builds an `lgcm.Model` with block f = bond curve and block c = basis, with no d factor and zero FX vol. On top of it sit the general formulas (via `Model.covint`), McCloud's Hull–White closed forms coded independently, a coupon-bond Gauss–Hermite benchmark and its frozen-weights approximation, and a Monte Carlo benchmark on the existing exact scheme. The notes get a new section before "Remarks", and there is a new notebook `python/notebooks/repo_convexity.py`.

**Tech Stack:** Python 3.12, numpy, scipy, sympy (tests), pytest, marimo, matplotlib; LaTeX (`amsart`, pdflatex).

**Spec:** `docs/superpowers/specs/2026-10-11-repo-convexity-design.md`

## Global Constraints

- Work in the worktree `/Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity` on branch `repo-convexity`. Never `cd` to the main checkout. Run git with `git -C <worktree>` or from the worktree root.
- Tests: `cd python && python -m pytest convexity -q`. Baseline: 106 passed, 5 slow deselected with `-m "not slow"`.
- Derive in LaTeX, verify in Python. Do not present a numerical observation as a proved statement. Gaussian (frozen-weights) approximations are asymptotic: state the measured order, never assert it.
- Monte Carlo tests use a 4 standard error tolerance with fixed seeds.
- Model time is in ACT/365 years. The accrual δ = `accrual(s, e, day_count)` only quotes the rate. The default is `"ACT/360"` (δ = (e − s)·365/360); `"ACT/365"` is also allowed.
- Out of scope: haircuts, default and recovery, special vs GC repo dynamics (z̄ = 0 except in a Remark), calibration to repo quotes, calendars, business-day adjustment, settlement lags.
- LaTeX build: `cd docs && pdflatex -interaction=nonstopmode CollateralDiscounting.tex` (twice). It must end with 0 errors and only the three known warnings: multiply defined `eq:cb_r_swaplet`, `eq:forward_cb_non_resetting`, and undefined `eq:cb_r_swaplet2`.
- Commit messages have no `Co-Authored-By` or other Claude attribution lines.
- Match the surrounding code: module docstrings that cite the notes' statements, terse comments, no type-hint noise beyond what `lgcm.py` and `curves.py` use.

## Review Focus

1. **Coupon exactly on s or e.** A coupon at T_k = s belongs to the seller (the price at s is ex-coupon). A coupon at T_k = e is passed through and is not part of the returned bonds. Expected: the closed form and Monte Carlo agree. Pinned in Task 3 (`test_coupon_on_window_boundaries_exact_vs_mc`).
2. **Maturities beyond the last knot** (T > horizon of `RepoModel.hull_white`, or T past the last knot of a piecewise model). `Model.piece` clamps to the last piece, so constant parameters extend flat. Expected: the Hull–White match still holds at T = 300 > horizon 200. Pinned in Task 2 (`test_hull_white_beyond_horizon`).
3. **Zero mean reversion.** The LGCM kernels are stable at β = 0, but McCloud's closed forms divide by θκ(θ+κ). Expected: `hw_*` raises a `ValueError` that points to `repo_convexity`, and `repo_convexity` at θ = κ = 0 equals the polynomial limit. Pinned in Task 2.
4. **Bad inputs.** These must raise `ValueError` with a message naming the problem: an unknown day count; t > s, s ≥ e or e > T; a coupon bond without a cashflow after e; mismatched or unsorted bond arrays; a non-positive-definite correlation. Pinned in Tasks 1 and 3.
5. **Spot vs forward overnight repo.** The spot overnight repo rate converges to the bond rate, with convexity O(h²). The forward-starting overnight rate has a nonzero O(1) adjustment. Expected: the code and the notes keep the two apart. Pinned in Task 4.

## Deviations from the spec (decided while prototyping)

- The Monte Carlo is a function, `mc_repo_rate`, built on `Simulator.simulate_dates`. It is not a `RepoPayoff` class, because a `Payoff` has a single expiry and a repo needs the state at s, at e and at the coupon dates.
- `repo_rate` and the coupon rates are agreed at 0, since the curves are time-0 curves. `repo_convexity(rm, t, s, e, T)` keeps the general t.
- McCloud's interpolations are exposed as their weights (`hw_maturity_weight`, `hw_forwardness_weight`) rather than as `hw_interpolate_*`. The tests apply them to the model's discount factors and adjustments.
- Hypothesis 6 of the spec is confirmed in its precise form. The *spot* overnight repo rate is the bond rate, with convexity O(h²), and the *forward* overnight repo rate has an O(1) adjustment. So the bond-collateralised discount curve is p̄ under rolling overnight financing.

---

## File Structure

| file | responsibility |
|---|---|
| `python/convexity/repo.py` (create) | everything repo: day count, `RepoModel`, `BondCurves`, zero-coupon repo convexity and rate, McCloud closed forms, coupon bonds (approximation and exact), Monte Carlo |
| `python/convexity/tests/test_repo.py` (create) | the checks quoted in Remark "Verification" of the new section |
| `python/convexity/README.md` (modify) | file table rows, "What is checked" rows, a short "Repo convexity" section |
| `docs/CollateralDiscounting.tex` (modify) | new section `\section{Bonds, repo and repo convexity}` before `\section{Remarks}` (line ~3849); label on Proposition "Collateral convexity" (line 1844); bibliography entry |
| `python/notebooks/repo_convexity.py` (create) | marimo notebook of the impacts |

`repo.py` is one file of about 350 lines, in line with `market.py` and `gaussian_approx.py`. Its sections are separated by the codebase's `# ----` banners.

---

### Task 1: Core module — model, curves, zero-coupon repo convexity and rate

**Files:**
- Create: `python/convexity/repo.py`
- Create: `python/convexity/tests/test_repo.py`

**Interfaces:**
- Consumes: `convexity.lgcm.Model(knots, factors, R, sigx)`, `Model.Sig(block, T, coef) -> list[(coef, i, "G", T)]`, `Model.I(A, B, s, t) -> float`, `Model.covint_brute`, `convexity.market.libor_convexity(m, t, a, b) -> float`.
- Produces:
  - `ACCRUAL: dict[str, float]`, `accrual(s, e, day_count="ACT/360") -> float`
  - `RepoModel(knots, bond_factors, basis_factors, R)`, with `.lgcm: lgcm.Model`, `RepoModel.hull_white(theta, sigma, kappa, eps, rho, horizon=200.0)`, `.bond_vol(T, coef=1.0)`, `.basis_vol(T, coef=1.0)` (loading lists)
  - `BondCurves(int_rbar0, int_b0, x0=1.0)`, `BondCurves.flat(rbar=0.035, b=-0.002)`, `.Zbond(T)`, `.Zois(T)`, `.Z0(kind, T)` for kind in {"ff", "fd", "dd", "qz"}, `.fx_forward0(T)`
  - `repo_convexity(rm, t, s, e, T) -> float` (ln K^{repo})
  - `maturity_adjustment(rm, t, e, T) -> float`, `forwardness_adjustment(rm, t, s, e, T) -> float`
  - `repo_rate(rm, curves, s, e, T, day_count="ACT/360") -> float` (agreed at 0)
  - `repo_discount_factor(rm, curves, e, T) -> float`

- [ ] **Step 1: Write the failing tests**

Create `python/convexity/tests/test_repo.py`:

```python
"""Checks of section "Bonds, repo and repo convexity" of docs/CollateralDiscounting.tex.

Run from python/ with ``python -m pytest convexity -q``.

The benchmarks are exact (nested quadrature, the Libor collateral convexity, McCloud's Hull-White
closed forms, Gauss-Hermite quadrature) except the Monte Carlo checks (4 standard errors, fixed
seeds) and the frozen-weights coupon approximation, which is asymptotic (measured order).
"""
import math

import numpy as np
import pytest

from convexity import repo
from convexity.market import libor_convexity

HW = dict(theta=0.05, sigma=0.01, kappa=0.4, eps=0.008, rho=-0.5)
CURVES = repo.BondCurves.flat(rbar=0.035, b=-0.002)
DATES = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.5, 10.0), (0.0, 2.0, 2.25, 7.0), (0.5, 1.3, 2.1, 7.5),
         (0.0, 5.0, 6.0, 30.0)]


@pytest.fixture(scope="module")
def hw():
    return repo.RepoModel.hull_white(**HW)


@pytest.fixture(scope="module")
def pw():
    """Two bond factors and one basis factor, piecewise constant, with zero, tiny and negative mean reversions."""
    knots = [0, 1, 3, 7, 15, 40]
    bond = [([0.010, 0.012, 0.009, 0.011, 0.010], [0.05, 0.0, 0.03, -0.01, 0.02]),
            ([0.006, 0.005, 0.007, 0.006, 0.005], [0.8, 0.6, 1e-7, 0.5, 0.7])]
    basis = [([0.004, 0.006, 0.005, 0.004, 0.003], [0.3, 0.5, 0.2, 0.4, 0.3])]
    R = [[1.0, 0.3, -0.4], [0.3, 1.0, 0.2], [-0.4, 0.2, 1.0]]
    return repo.RepoModel(knots, bond, basis, R)


def brute_ln_K(rm, t, s, e, T):
    """Proposition "LGCM repo convexity" by nested adaptive quadrature (lgcm.Model.covint_brute)."""
    m = rm.lgcm

    def I(A, B, a, b):
        return sum(ca * cb * m.covint_brute((i, ta, Ta), (j, tb, Tb), a, b)
                   for ca, i, ta, Ta in A for cb, j, tb, Tb in B)
    sc = rm.basis_vol(e)
    return -(I(rm.bond_vol(e) + rm.bond_vol(s, -1.0), sc, t, s) + I(rm.bond_vol(e) + rm.bond_vol(T, -1.0), sc, s, e))


# --------------------------------------------------------------------------- zero-coupon repo convexity
@pytest.mark.parametrize("t,s,e,T", DATES)
def test_repo_convexity_matches_quadrature(pw, t, s, e, T):
    assert repo.repo_convexity(pw, t, s, e, T) == pytest.approx(brute_ln_K(pw, t, s, e, T), rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("a,b", [(1.0, 1.5), (2.0, 3.0), (5.0, 10.0)])
def test_repo_to_maturity_is_libor_convexity(pw, a, b):
    """T = e: the repo convexity is the Libor collateral convexity (this fixes the sign of Sigma^c)."""
    assert repo.repo_convexity(pw, 0.0, a, b, b) == pytest.approx(math.log(libor_convexity(pw.lgcm, 0.0, a, b)), rel=1e-12)


def test_boundary_conditions(pw):
    assert repo.maturity_adjustment(pw, 0.0, 3.0, 3.0) == 0.0                     # M^{ee} = 0
    assert repo.maturity_adjustment(pw, 1.0, 1.0, 7.0) == 0.0                     # M^{tT}_t = 0
    assert abs(repo.forwardness_adjustment(pw, 1.0, 1.0, 2.0, 7.0)) < 1e-18       # F^{eT}_{tt} = 0
    C = repo.repo_convexity(pw, 0.5, 1.3, 2.1, 7.5)
    M = repo.maturity_adjustment(pw, 0.5, 2.1, 7.5) - repo.maturity_adjustment(pw, 0.5, 1.3, 7.5)
    assert C == pytest.approx(M + repo.forwardness_adjustment(pw, 0.5, 1.3, 2.1, 7.5), rel=1e-13)


def test_no_basis_correlation_gives_bond_forward():
    rm = repo.RepoModel.hull_white(**{**HW, "rho": 0.0})
    s, e = 1.3, 2.1
    fwd = (CURVES.Zbond(s) / CURVES.Zbond(e) - 1.0) / repo.accrual(s, e)
    assert repo.repo_rate(rm, CURVES, s, e, 7.5) == pytest.approx(fwd, rel=1e-15)


def test_repo_discount_factor_to_maturity_is_bond(pw):
    assert repo.repo_discount_factor(pw, CURVES, 4.0, 4.0) == pytest.approx(float(CURVES.Zbond(4.0)), rel=1e-15)


def test_day_count_only_quotes_the_rate(hw):
    s, e, T = 1.3, 2.1, 7.5
    assert repo.accrual(s, e, "ACT/360") == pytest.approx((e - s) * 365.0 / 360.0, rel=1e-15)
    assert repo.accrual(s, e, "ACT/365") == pytest.approx(e - s, rel=1e-15)
    r360 = repo.repo_rate(hw, CURVES, s, e, T, "ACT/360")
    r365 = repo.repo_rate(hw, CURVES, s, e, T, "ACT/365")
    assert r360 * 365.0 / 360.0 == pytest.approx(r365, rel=1e-14)
    with pytest.raises(ValueError, match="day count"):
        repo.accrual(s, e, "30/360")


@pytest.mark.parametrize("t,s,e,T", [(0.0, 2.0, 1.0, 5.0), (0.0, 1.0, 2.0, 1.5), (1.5, 1.0, 2.0, 5.0),
                                     (0.0, 1.0, 1.0, 5.0), (-1.0, 0.0, 1.0, 2.0)])
def test_dates_are_validated(hw, t, s, e, T):
    with pytest.raises(ValueError, match="t <= s < e <= T"):
        repo.repo_convexity(hw, t, s, e, T)


def test_model_inputs_are_validated():
    with pytest.raises(ValueError, match="R must be"):
        repo.RepoModel([0, 10], [([0.01], [0.1])], [([0.01], [0.1])], [[1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        repo.RepoModel([0, 10], [([0.01], [0.1])], [([0.01], [0.1])], [[1.0, 1.2], [1.2, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        repo.RepoModel([0, 10], [], [([0.01], [0.1])], np.eye(1))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd python && python -m pytest convexity/tests/test_repo.py -q`
Expected: collection error `ModuleNotFoundError` / `ImportError: cannot import name 'repo'`.

- [ ] **Step 3: Write the implementation**

Create `python/convexity/repo.py`:

```python
"""Bonds, repo and repo convexity (McCloud, "Repo convexity", arXiv:1905.03316, 2019).

Numerical companion of section "Bonds, repo and repo convexity" of docs/CollateralDiscounting.tex.

Two curves under one risk-neutral measure: the bond curve pbar (discount rate rbar) and the
derivative (OIS) curve p (rate r = rbar + b, b the discounting basis). They are the LGCM blocks
f (bond curve, Z^{f|f} = pbar) and c (collateral basis f->d = b, Z^{f|d} = p), with no d factor
and zero FX vol. Only RepoModel and BondCurves know this mapping.

Repo over [s, e] on the bond maturing at T, agreed at t <= s < e <= T: one unit of cash at s for
1/pbar^T_s bonds, returned at e against 1 + delta f, discounted at r. Then
    1 + delta f = (pbar_{t,s} / pbar_{t,e}) K^{repo}_t(s, e, T)      (Proposition "Repo rate")
Model time is in ACT/365 years; the accrual delta = accrual(s, e, day_count) only quotes the rate.

Main entry points
-----------------
RepoModel, BondCurves                 model (bond and basis factors) and initial curves
repo_convexity                        ln K^{repo}, Proposition "LGCM repo convexity"
maturity_adjustment, forwardness_adjustment   McCloud's M and F (Corollary "Maturity and forwardness")
repo_rate, repo_discount_factor       quoted repo rate, repo discount factor hat p^{eT}_0
hw_*                                  McCloud's Hull-White closed forms, written independently of lgcm
CouponBond, coupon_repo_rate, coupon_repo_rate_exact   repo on a coupon bond: frozen weights, Gauss-Hermite
mc_repo_rate                          Monte Carlo benchmark (exact LGCM scheme)
"""
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .lgcm import Model

ACCRUAL = {"ACT/360": 365.0 / 360.0, "ACT/365": 1.0}


def accrual(s, e, day_count="ACT/360"):
    """Accrual fraction of [s, e] (model time in ACT/365 years)."""
    if day_count not in ACCRUAL:
        raise ValueError(f"day count {day_count!r}: use one of {sorted(ACCRUAL)}")
    return (e - s) * ACCRUAL[day_count]


def _check_dates(t, s, e, T):
    if not (0.0 <= t <= s < e <= T):
        raise ValueError(f"need 0 <= t <= s < e <= T, got t={t}, s={s}, e={e}, T={T}")


# ---------------------------------------------------------------------- model and curves
class RepoModel:
    """LGCM with bond factors (block f) and basis factors (block c), no d factor, zero FX vol."""

    def __init__(self, knots, bond_factors, basis_factors, R):
        """bond_factors, basis_factors: lists of (sig[k], beta[k]) on the pieces of `knots`;
        R: correlation of the bond drivers then the basis drivers."""
        nf, nc = len(bond_factors), len(basis_factors)
        if nf == 0 or nc == 0:
            raise ValueError("need at least one bond factor and one basis factor")
        R = np.asarray(R, float)
        if R.shape != (nf + nc, nf + nc):
            raise ValueError(f"R must be {nf + nc}x{nf + nc}, got {R.shape}")
        RR = np.eye(nf + nc + 1)
        RR[:-1, :-1] = R
        if np.any(np.linalg.eigvalsh(RR) <= 0):
            raise ValueError("correlation matrix is not positive definite")
        factors = [("f", s, b) for s, b in bond_factors] + [("c", s, b) for s, b in basis_factors]
        self.lgcm = Model(knots, factors, RR, np.zeros(len(knots) - 1))

    @classmethod
    def hull_white(cls, theta, sigma, kappa, eps, rho, horizon=200.0):
        """McCloud (27): d rbar = theta(.) dt + sigma dx, d b = kappa(.) dt + eps dy, dx dy = rho dt."""
        return cls([0.0, horizon], [([sigma], [theta])], [([eps], [kappa])], [[1.0, rho], [rho, 1.0]])

    def bond_vol(self, T, coef=1.0):
        """Loading of the bond curve vol Sigma-bar_{u,T}."""
        return self.lgcm.Sig("f", T, coef)

    def basis_vol(self, T, coef=1.0):
        """Loading of the basis vol Sigma^b_{u,T} = Sigma^p_{u,T} - Sigma-bar_{u,T} (block c)."""
        return self.lgcm.Sig("c", T, coef)


@dataclass
class BondCurves:
    """Initial bond curve pbar_{0,T} = exp(-int_rbar0(T)) and OIS curve p_{0,T} = pbar_{0,T} exp(-int_b0(T)).

    Duck-type compatible with market.InitialCurves for the Monte Carlo: "ff" = bond, "fd" = OIS,
    "qz" = OIS / bond, "dd" = OIS (unused: no d factor), x0 = 1."""
    int_rbar0: Callable
    int_b0: Callable
    x0: float = 1.0

    @classmethod
    def flat(cls, rbar=0.035, b=-0.002):
        return cls(lambda T: rbar * np.asarray(T, float), lambda T: b * np.asarray(T, float))

    def Zbond(self, T):
        return np.exp(-self.int_rbar0(T))

    def Zois(self, T):
        return np.exp(-self.int_rbar0(T) - self.int_b0(T))

    def Z0(self, kind, T):
        if kind == "ff":
            return self.Zbond(T)
        if kind in ("fd", "dd"):
            return self.Zois(T)
        if kind == "qz":
            return np.exp(-self.int_b0(T))
        raise ValueError(kind)

    def fx_forward0(self, T):
        return self.x0 * self.Z0("fd", T) / self.Z0("dd", T)


# ---------------------------------------------------------------------- repo convexity (zero-coupon)
def repo_convexity(rm, t, s, e, T):
    """ln K^{repo}_t(s, e, T) = -int_t^s (Sb_{u,e} - Sb_{u,s}).S^b_{u,e} du - int_s^e (Sb_{u,e} - Sb_{u,T}).S^b_{u,e} du."""
    _check_dates(t, s, e, T)
    m, sb = rm.lgcm, rm.basis_vol(e)
    fwd = m.I(rm.bond_vol(e) + rm.bond_vol(s, -1.0), sb, t, s)
    win = m.I(rm.bond_vol(e) + rm.bond_vol(T, -1.0), sb, s, e)
    return -(fwd + win)


def maturity_adjustment(rm, t, e, T):
    """M^{eT}_t = C^{eT}_{tt} (McCloud (17)); M^{tT}_t = 0."""
    return 0.0 if e == t else repo_convexity(rm, t, t, e, T)


def forwardness_adjustment(rm, t, s, e, T):
    """F^{eT}_{ts} = C^{eT}_{ts} - (M^{eT}_t - M^{sT}_t) (McCloud (17))."""
    return repo_convexity(rm, t, s, e, T) - (maturity_adjustment(rm, t, e, T) - maturity_adjustment(rm, t, s, T))


def repo_rate(rm, curves, s, e, T, day_count="ACT/360"):
    """Repo rate f^{eT}_{0s} agreed at 0: 1 + delta f = (pbar_{0,s}/pbar_{0,e}) K^{repo}_0(s, e, T)."""
    gross = curves.Zbond(s) / curves.Zbond(e) * math.exp(repo_convexity(rm, 0.0, s, e, T))
    return float((gross - 1.0) / accrual(s, e, day_count))


def repo_discount_factor(rm, curves, e, T):
    """hat p^{eT}_0 = pbar_{0,e} exp(-M^{eT}_0) (McCloud (20) with L = 0)."""
    return float(curves.Zbond(e) * math.exp(-maturity_adjustment(rm, 0.0, e, T)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd python && python -m pytest convexity/tests/test_repo.py -q`
Expected: all pass (18 tests). The prototype measured these on the same models: quadrature agreement ≤ 6e-15 relative, the Libor identity to 1e-12, and `covint_brute` ≤ 0.2 s per case.

- [ ] **Step 5: Commit**

```bash
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity add python/convexity/repo.py python/convexity/tests/test_repo.py
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity commit -m "repo: LGCM repo convexity, maturity and forwardness adjustments, repo rate"
```

---

### Task 2: McCloud's Hull–White closed forms and the repo curve identities

**Files:**
- Modify: `python/convexity/repo.py` (append a section after `repo_discount_factor`)
- Modify: `python/convexity/tests/test_repo.py` (append)

**Interfaces:**
- Consumes: Task 1 (`RepoModel.hull_white`, `repo_convexity`, `maturity_adjustment`, `forwardness_adjustment`, `repo_discount_factor`, `BondCurves`).
- Produces (all floats; θ, κ ≠ 0):
  - `hw_B(tau, delta, mu, theta, kappa)`
  - `hw_convexity(rho, sigma, eps, theta, kappa, t, s, e, T)`
  - `hw_M(rho, sigma, eps, theta, kappa, t, e, T)`
  - `hw_F(rho, sigma, eps, theta, kappa, t, s, e, T)`
  - `hw_forward_spread(rho, sigma, eps, theta, kappa, t, e)`
  - `hw_extrapolate(spread_E, t, E, e, theta, kappa)`
  - `hw_maturity_weight(theta, mu)`
  - `hw_forwardness_weight(theta, kappa, tau)`

- [ ] **Step 1: Write the failing tests**

Append to `python/convexity/tests/test_repo.py`:

```python
# --------------------------------------------------------------------------- McCloud's Hull-White closed forms
P = (HW["rho"], HW["sigma"], HW["eps"], HW["theta"], HW["kappa"])


@pytest.mark.parametrize("t,s,e,T", DATES)
def test_hull_white_matches_mccloud(hw, t, s, e, T):
    """Proposition "Hull-White repo convexity": the LGCM formula is McCloud's (30), (33), (34)."""
    assert repo.repo_convexity(hw, t, s, e, T) == pytest.approx(repo.hw_convexity(*P, t, s, e, T), rel=1e-11, abs=1e-18)
    assert repo.maturity_adjustment(hw, t, e, T) == pytest.approx(repo.hw_M(*P, t, e, T), rel=1e-11, abs=1e-18)
    if s > t:
        assert repo.forwardness_adjustment(hw, t, s, e, T) == pytest.approx(repo.hw_F(*P, t, s, e, T), rel=1e-11)


def test_hull_white_beyond_horizon(hw):
    """Model.piece clamps to the last piece: constant parameters extend past the horizon (200)."""
    assert repo.repo_convexity(hw, 0.0, 2.0, 3.0, 300.0) == pytest.approx(repo.hw_convexity(*P, 0.0, 2.0, 3.0, 300.0), rel=1e-11)


@pytest.mark.parametrize("theta,kappa", [(1e-3, 1e-3), (1e-3, 0.5), (0.3, 1e-3), (-0.02, 0.4)])
def test_hull_white_small_and_negative_mean_reversion(theta, kappa):
    rm = repo.RepoModel.hull_white(theta, 0.01, kappa, 0.008, -0.5)
    for t, s, e, T in DATES[1:]:
        assert repo.repo_convexity(rm, t, s, e, T) == pytest.approx(
            repo.hw_convexity(-0.5, 0.01, 0.008, theta, kappa, t, s, e, T), rel=1e-7)


def test_zero_mean_reversion_limit():
    """theta = kappa = 0: ln K = -rho sigma eps [(e-s)((e-t)^2 - (e-s)^2)/2 - (T-e)(e-s)^2/2]."""
    rho, sig, eps = -0.5, 0.01, 0.008
    rm = repo.RepoModel.hull_white(0.0, sig, 0.0, eps, rho)
    for t, s, e, T in DATES:
        poly = -rho * sig * eps * ((e - s) * ((e - t) ** 2 - (e - s) ** 2) / 2 - (T - e) * (e - s) ** 2 / 2)
        assert repo.repo_convexity(rm, t, s, e, T) == pytest.approx(poly, rel=1e-12, abs=1e-18)
    with pytest.raises(ValueError, match="repo_convexity"):
        repo.hw_B(1.0, 1.0, 1.0, 0.0, 0.4)


def test_hull_white_maturity_interpolation(hw):
    """McCloud (41): hat p^{eT} = (pbar^e)^w (hat p^{e,inf})^{1-w}, w = exp(-theta (T-e)): exact in Hull-White."""
    for e, mu in [(1.0, 2.0), (2.0, 10.0), (5.0, 25.0)]:
        w = repo.hw_maturity_weight(HW["theta"], mu)
        p_inf = repo.repo_discount_factor(hw, CURVES, e, e + 1e3)
        p_eT = repo.repo_discount_factor(hw, CURVES, e, e + mu)
        assert p_eT == pytest.approx(float(CURVES.Zbond(e)) ** w * p_inf ** (1 - w), rel=1e-13)


def test_hull_white_forwardness_interpolation(hw):
    """McCloud (43): F(tau) = (1 - exp(-(theta+kappa) tau)) F(inf) at fixed e-s, T-e: exact in Hull-White."""
    d, mu = 1.0, 5.0
    F_inf = repo.forwardness_adjustment(hw, 0.0, 150.0, 150.0 + d, 150.0 + d + mu)
    for tau in [0.5, 1.0, 3.0, 6.0]:
        w = repo.hw_forwardness_weight(HW["theta"], HW["kappa"], tau)
        assert repo.forwardness_adjustment(hw, 0.0, tau, tau + d, tau + d + mu) == pytest.approx((1 - w) * F_inf, rel=1e-12)


def test_hull_white_forward_spread_and_extrapolation(hw):
    """McCloud (38): hat f^e - bar f^e = d/de M^{e,inf}_0; (39) extrapolates it from E."""
    h = 1e-5
    for e in [0.5, 2.0, 7.0]:
        dM = (repo.maturity_adjustment(hw, 0.0, e + h, 1e3) - repo.maturity_adjustment(hw, 0.0, e - h, 1e3)) / (2 * h)
        assert dM == pytest.approx(repo.hw_forward_spread(*P, 0.0, e), rel=1e-7)
    E = 2.0
    sE = repo.hw_forward_spread(*P, 0.0, E)
    for e in [2.0, 5.0, 20.0]:
        assert repo.hw_extrapolate(sE, 0.0, E, e, HW["theta"], HW["kappa"]) == pytest.approx(
            repo.hw_forward_spread(*P, 0.0, e), rel=1e-13)


def test_interpolation_fails_beyond_hull_white(pw):
    """Remark "Beyond Hull-White": no single geometric weight in the two-factor piecewise model.
    Measured: implied theta 0.0232 (mu = 2) vs 0.0296 (mu = 20); F(tau)/F(30) = 0.18, 0.31, 1.02, 1.53, 1.38
    at tau = 0.5, 1, 3, 6, 10 (not monotone, above 1)."""
    e, Minf = 2.0, repo.maturity_adjustment(pw, 0.0, 2.0, 60.0)
    th = [-math.log(1 - repo.maturity_adjustment(pw, 0.0, e, e + mu) / Minf) / mu for mu in (2.0, 20.0)]
    assert th[1] / th[0] - 1 > 0.2
    Finf = repo.forwardness_adjustment(pw, 0.0, 30.0, 31.0, 36.0)
    ratios = [repo.forwardness_adjustment(pw, 0.0, tau, tau + 1, tau + 6) / Finf for tau in (0.5, 1, 3, 6, 10)]
    assert max(ratios) > 1.2 and np.any(np.diff(ratios) < 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd python && python -m pytest convexity/tests/test_repo.py -q -k "hull_white or mean_reversion or interpolation"`
Expected: FAIL with `AttributeError: module 'convexity.repo' has no attribute 'hw_convexity'` (and similar for the other `hw_*` functions).

- [ ] **Step 3: Write the implementation**

Append to `python/convexity/repo.py`:

```python
# ---------------------------------------------------------------------- McCloud's Hull-White closed forms
def _ex(x):
    """1 - e^{-x}."""
    return -math.expm1(-x)


def _check_mr(theta, kappa):
    if theta * kappa * (theta + kappa) == 0:
        raise ValueError("McCloud's closed forms need theta, kappa, theta + kappa != 0: use repo_convexity "
                         "(stable at zero mean reversion)")


def hw_B(tau, delta, mu, theta, kappa):
    """McCloud (33): C^{eT}_{ts} = rho sigma eps B[s-t, e-s, T-e; theta, kappa]."""
    _check_mr(theta, kappa)
    th, ka = theta, kappa
    a = _ex(th * mu) * (_ex(th * (tau + delta)) / th - _ex((th + ka) * (tau + delta)) / (th + ka))
    b = _ex(th * (delta + mu)) * (_ex(th * tau) / th - math.exp(-ka * delta) * _ex((th + ka) * tau) / (th + ka))
    return (a - b) / (th * ka)


def hw_convexity(rho, sigma, eps, theta, kappa, t, s, e, T):
    _check_dates(t, s, e, T)
    return rho * sigma * eps * hw_B(s - t, e - s, T - e, theta, kappa)


def hw_M(rho, sigma, eps, theta, kappa, t, e, T):
    """McCloud (34), maturity adjustment."""
    _check_mr(theta, kappa)
    th, ka = theta, kappa
    return rho * sigma * eps / (th * ka) * _ex(th * (T - e)) * (_ex(th * (e - t)) / th - _ex((th + ka) * (e - t)) / (th + ka))


def hw_F(rho, sigma, eps, theta, kappa, t, s, e, T):
    """McCloud (34), forwardness adjustment."""
    _check_mr(theta, kappa)
    th, ka = theta, kappa
    return -rho * sigma * eps / (th * ka * (th + ka)) * _ex(th * (T - s)) * _ex(ka * (e - s)) * _ex((th + ka) * (s - t))


def hw_forward_spread(rho, sigma, eps, theta, kappa, t, e):
    """McCloud (38): hat f^e_t - bar f^e_t for the infinite-maturity repo curve."""
    _check_mr(theta, kappa)
    return rho * sigma * eps / (theta * kappa) * math.exp(-theta * (e - t)) * _ex(kappa * (e - t))


def hw_extrapolate(spread_E, t, E, e, theta, kappa):
    """McCloud (39): hat f^e - bar f^e from the spread observed at E (e >= E)."""
    return spread_E * math.exp(-theta * (e - E)) * _ex(kappa * (e - t)) / _ex(kappa * (E - t))


def hw_maturity_weight(theta, mu):
    """McCloud (41): hat p^{eT} = (bar p^e)^w (hat p^{e,inf})^{1-w}, w = exp(-theta (T - e))."""
    return math.exp(-theta * mu)


def hw_forwardness_weight(theta, kappa, tau):
    """McCloud (43): weight exp(-(theta + kappa)(s - t)) of the spot-starting gross rate."""
    return math.exp(-(theta + kappa) * tau)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd python && python -m pytest convexity/tests/test_repo.py -q`
Expected: all pass. In the prototype, the LGCM and McCloud's formulas agreed to about 4e-21 absolute on ln K ≈ −1.05e-5. The θ_implied spread in the piecewise model was 27%, and the largest F ratio was 1.53.

If `test_hull_white_matches_mccloud` fails, McCloud's paper or the transcription has a sign or term discrepancy. Do not "fix" the closed form to match. Use superpowers:systematic-debugging, then record the discrepancy for the Remark of Task 5.

- [ ] **Step 5: Commit**

```bash
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity add python/convexity/repo.py python/convexity/tests/test_repo.py
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity commit -m "repo: McCloud's Hull-White closed forms and repo curve identities"
```

---

### Task 3: Coupon bonds — exact Gauss–Hermite benchmark and frozen-weights approximation

**Files:**
- Modify: `python/convexity/repo.py` (imports and a new section)
- Modify: `python/convexity/tests/test_repo.py` (append)

**Interfaces:**
- Consumes: Task 1 (`repo_convexity`, `accrual`, `_check_dates`, `RepoModel`, `BondCurves`); `convexity.transitions.propagate(m, grid, "RN") -> (mX, P, ix)` with `ix` keys `iZX`, `iID`, `n`; `Model.lnZ(kind, t, T, Y, v) -> (a, c)`, `Model.Y(t)`, `Model.vfc(t)`.
- Produces:
  - `CouponBond(times: tuple, cashflows: tuple)` (frozen dataclass), `CouponBond.fixed(maturity, coupon, freq=1, notional=1.0)`, `.price0(curves) -> float`
  - `coupon_repo_rate(rm, curves, bond, s, e, day_count="ACT/360") -> float` (approximation)
  - `coupon_repo_rate_exact(rm, curves, bond, s, e, n_gh=40, day_count="ACT/360") -> float`
  - private: `_ois_forward_law(rm, s) -> (mean, cov)`, `_coupon_terms(rm, curves, bond, s, e) -> (num, den)`

- [ ] **Step 1: Write the failing tests**

Append to `python/convexity/tests/test_repo.py`:

```python
# --------------------------------------------------------------------------- coupon bonds
BOND = repo.CouponBond.fixed(10.0, 0.05, 2)      # 10Y semi-annual 5%: coupons at 1.5 and 2.0 fall in (1.3, 2.1]


def test_coupon_bond_construction():
    assert BOND.times[0] == pytest.approx(0.5) and BOND.times[-1] == pytest.approx(10.0) and len(BOND.times) == 20
    assert BOND.cashflows[-1] == pytest.approx(1.025) and BOND.cashflows[0] == pytest.approx(0.025)
    with pytest.raises(ValueError, match="same length"):
        repo.CouponBond((1.0, 2.0), (1.0,))
    with pytest.raises(ValueError, match="increasing"):
        repo.CouponBond((2.0, 1.0), (0.05, 1.05))
    with pytest.raises(ValueError, match="after the repo end"):
        repo.coupon_repo_rate(repo.RepoModel.hull_white(**HW), CURVES, repo.CouponBond((1.0, 2.0), (0.05, 1.05)), 1.3, 2.1)


def test_ois_forward_law_is_normalised(pw):
    """E^{RN}[D_{0,s}] = p_{0,s}: the tilt a = e_zx - e_Id has a.m + a'Pa/2 = 0."""
    from convexity import transitions
    m = pw.lgcm
    mX, P, ix = transitions.propagate(m, [0.0, 1.0, 2.5], "RN")
    a = np.zeros(ix["n"]); a[ix["iZX"]], a[ix["iID"]] = 1.0, -1.0
    assert abs(a @ mX + 0.5 * a @ P @ a) < 1e-15


@pytest.mark.parametrize("s,e,T", [(1.3, 2.1, 7.5), (0.0, 0.5, 3.0), (2.0, 3.0, 3.0)])
def test_single_cashflow_is_zero_coupon(hw, s, e, T):
    """A one-cashflow bond: both the exact and the frozen-weights rates are Proposition "Repo rate"."""
    b = repo.CouponBond((T,), (1.0,))
    zc = repo.repo_rate(hw, CURVES, s, e, T)
    assert repo.coupon_repo_rate_exact(hw, CURVES, b, s, e) == pytest.approx(zc, rel=1e-13)
    assert repo.coupon_repo_rate(hw, CURVES, b, s, e) == pytest.approx(zc, rel=1e-13)


def test_zero_basis_gives_bond_forward():
    """b = 0 (eps = 0, flat zero basis): Y_s = 1, so 1 + delta f = pbar_{0,s}/pbar_{0,e} for any coupons."""
    rm = repo.RepoModel.hull_white(0.05, 0.01, 0.4, 0.0, -0.5)
    cv = repo.BondCurves.flat(rbar=0.035, b=0.0)
    fwd = (cv.Zbond(1.3) / cv.Zbond(2.1) - 1.0) / repo.accrual(1.3, 2.1)
    assert repo.coupon_repo_rate_exact(rm, cv, BOND, 1.3, 2.1) == pytest.approx(fwd, rel=1e-13)


def test_gauss_hermite_converged(hw):
    a = repo.coupon_repo_rate_exact(hw, CURVES, BOND, 1.3, 2.1, n_gh=40)
    b = repo.coupon_repo_rate_exact(hw, CURVES, BOND, 1.3, 2.1, n_gh=60)
    assert abs(a - b) < 1e-14


def test_frozen_weights_error_order():
    """Asymptotic, not proved. Measured: errors 3.73e-4, 2.31e-5, 1.44e-6 at vol scale 4, 2, 1
    (ratios 16.2, 16.0): O(lambda^4) absolute in the rate."""
    errs = []
    for sc in [4.0, 2.0, 1.0]:
        rm = repo.RepoModel.hull_white(0.05, 0.01 * sc, 0.4, 0.008 * sc, -0.5)
        errs.append(repo.coupon_repo_rate(rm, CURVES, BOND, 1.3, 2.1) - repo.coupon_repo_rate_exact(rm, CURVES, BOND, 1.3, 2.1))
    r = [errs[0] / errs[1], errs[1] / errs[2]]
    assert all(14.0 < x < 18.0 for x in r), (errs, r)
```

(`test_coupon_on_window_boundaries_exact_vs_mc` is added in Task 4, which provides `mc_repo_rate`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd python && python -m pytest convexity/tests/test_repo.py -q -k "coupon or cashflow or basis or hermite or frozen or law"`
Expected: FAIL with `AttributeError: module 'convexity.repo' has no attribute 'CouponBond'`. `test_ois_forward_law_is_normalised` passes already, because it only uses `transitions`.

- [ ] **Step 3: Write the implementation**

In `python/convexity/repo.py`, add `from . import transitions` above `from .lgcm import Model`. Then append:

```python
# ---------------------------------------------------------------------- coupon bonds
@dataclass(frozen=True)
class CouponBond:
    """Fixed cashflows c_k at T_k (the last one includes the principal)."""
    times: tuple
    cashflows: tuple

    def __post_init__(self):
        T, c = np.asarray(self.times, float), np.asarray(self.cashflows, float)
        if T.ndim != 1 or T.size == 0 or T.shape != c.shape:
            raise ValueError("times and cashflows must be non-empty and of the same length")
        if np.any(np.diff(T) <= 0) or T[0] <= 0 or np.any(c <= 0):
            raise ValueError("times must be increasing and positive, cashflows positive")

    @classmethod
    def fixed(cls, maturity, coupon, freq=1, notional=1.0):
        n = int(round(maturity * freq))
        times = tuple(t for t in (maturity - k / freq for k in range(n - 1, -1, -1)) if t > 1e-12)
        flows = tuple(notional * coupon / freq + (notional if i == len(times) - 1 else 0.0) for i in range(len(times)))
        return cls(times, flows)

    def price0(self, curves):
        return float(sum(c * curves.Zbond(T) for T, c in zip(self.times, self.cashflows)))


def _ois_forward_law(rm, s):
    """Mean and covariance of the rate state z_s under the OIS s-forward measure Q^{p,s}.

    The deflator of a payment at s is p_{0,s} exp(z^x_s - int r^d) (Simulator.deflator, ccy f under d
    collateral), so Q^{p,s} tilts the risk-neutral law of the state by a = e_{zx} - e_{Id}."""
    m = rm.lgcm
    if s <= 0:
        return np.zeros(m.nr), np.zeros((m.nr, m.nr))
    grid = np.unique(np.r_[0.0, [k for k in m.knots if 0 < k < s], s])
    mX, P, ix = transitions.propagate(m, grid, "RN")
    a = np.zeros(ix["n"])
    a[ix["iZX"]], a[ix["iID"]] = 1.0, -1.0
    return (mX + P @ a)[: m.nr], P[: m.nr, : m.nr]


def _coupon_terms(rm, curves, bond, s, e):
    """Y(z) = sum_k num_k(z) / sum_j den_j(z) with PV_0 = p_{0,s} E^{p,s}[Y(z_s)] = p_{0,e} (1 + delta f).

    Proposition "Coupon repo rate": conditional on F_s, the bonds bought at s are worth P^ex_e at e
    (flows after e), whose OIS-discounted value at s is sum_k c_k p^e_s (pbar^{T_k}_s / pbar^e_s) K^{repo}_s(s, e, T_k);
    coupons in (s, e] are passed through at T_k, worth c_k p^{T_k}_s; a coupon at s belongs to the seller.
    Each term is (coef, alpha, const) for coef exp(alpha.z + const)."""
    if not any(T > e for T in bond.times):
        raise ValueError("the bond must have a cashflow after the repo end e")
    m = rm.lgcm
    Y, v = m.Y(s), m.vfc(s)

    def ln(kind, T):        # ln Z^{kind}_{s,T} = a.z + c
        a, c = m.lnZ(kind, s, T, Y, v)
        Z0 = curves.Zbond if kind == "ff" else curves.Zois
        return a, c + math.log(Z0(T) / Z0(s))

    num, den = [], []
    a_pe, c_pe = ln("fd", e)
    a_be, c_be = ln("ff", e)
    for T, c in zip(bond.times, bond.cashflows):
        if T <= s:
            continue
        a_b, c_b = ln("ff", T)
        den.append((c, a_b, c_b))
        if T > e:
            lk = repo_convexity(rm, s, s, e, T)
            num.append((c, a_pe + a_b - a_be, c_pe + c_b - c_be + lk))
        else:
            a_p, c_p = ln("fd", T)
            num.append((c, a_p, c_p))
    return num, den


def _gross(curves, s, e, EY):
    return float(curves.Zois(s) * EY / curves.Zois(e))


def coupon_repo_rate(rm, curves, bond, s, e, day_count="ACT/360"):
    """Frozen-weights approximation (asymptotic, Proposition "Frozen weights"): the basket sum_j den_j is
    replaced by the lognormal with the same mean and the loading sum_j w_j beta_j, w_j its weights at the mean."""
    _check_dates(0.0, s, e, max(bond.times))
    num, den = _coupon_terms(rm, curves, bond, s, e)
    mu, P = _ois_forward_law(rm, s)
    dm = np.array([c * math.exp(b @ mu + d + 0.5 * b @ P @ b) for c, b, d in den])
    Dbar = dm.sum()
    bbar = sum(w * b for w, (_, b, _) in zip(dm / Dbar, den))
    EY = sum(c * math.exp(a @ mu + d + 0.5 * bbar @ P @ bbar + 0.5 * (a - bbar) @ P @ (a - bbar))
             for c, a, d in num) / Dbar
    return (_gross(curves, s, e, EY) - 1.0) / accrual(s, e, day_count)


def coupon_repo_rate_exact(rm, curves, bond, s, e, n_gh=40, day_count="ACT/360"):
    """Exact up to Gauss-Hermite quadrature (tensor grid of n_gh^nr nodes) in the rate state z_s under Q^{p,s}."""
    _check_dates(0.0, s, e, max(bond.times))
    num, den = _coupon_terms(rm, curves, bond, s, e)
    mu, P = _ois_forward_law(rm, s)
    x, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    nr = len(mu)
    X = np.stack(np.meshgrid(*([x] * nr), indexing="ij"), -1).reshape(-1, nr)
    W = np.prod(np.stack(np.meshgrid(*([w] * nr), indexing="ij"), -1).reshape(-1, nr), axis=1)
    lam, U = np.linalg.eigh(P)
    Z = mu + X @ (U * np.sqrt(np.clip(lam, 0, None))).T
    N = sum(c * np.exp(Z @ a + d) for c, a, d in num)
    D = sum(c * np.exp(Z @ b + d) for c, b, d in den)
    return (_gross(curves, s, e, float(W @ (N / D))) - 1.0) / accrual(s, e, day_count)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd python && python -m pytest convexity/tests/test_repo.py -q`
Expected: all pass. If `test_frozen_weights_error_order` measures ratios outside (14, 18), do not widen the bounds blindly. Print `errs`, re-measure, and report the measured order in Task 5's Remark.

- [ ] **Step 5: Commit**

```bash
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity add python/convexity/repo.py python/convexity/tests/test_repo.py
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity commit -m "repo: coupon bond repo, exact Gauss-Hermite benchmark and frozen-weights approximation"
```

---

### Task 4: Monte Carlo benchmark, overnight repo limits, README

**Files:**
- Modify: `python/convexity/repo.py` (imports and a new section)
- Modify: `python/convexity/tests/test_repo.py` (append)
- Modify: `python/convexity/README.md`

**Interfaces:**
- Consumes: Tasks 1–3. Also `convexity.montecarlo.Simulator(model, scheme=...)`, `.simulate_dates(dates, n_paths, seed) -> PathStates` (call with a date to get a `SimResult`), `.discount(res, curves, kind, T)`, `.deflator(res, curves, ccy, collateral)`, `convexity.montecarlo.cv_mean_and_stderr(x, controls, means) -> (mean, stderr)`, and `convexity.schemes.ExactScheme()`.
- Produces: `mc_repo_rate(rm, curves, bond, s, e, n_paths=2**16, seed=7, day_count="ACT/360") -> (rate, stderr)`, where `bond` is a `CouponBond` or a float maturity T.

- [ ] **Step 1: Write the failing tests**

Append to `python/convexity/tests/test_repo.py`:

```python
# --------------------------------------------------------------------------- Monte Carlo
MC_HW = repo.RepoModel.hull_white(0.05, 0.03, 0.4, 0.024, -0.5)   # vol scale 3: the convexity is ~1.2bp


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_mc_zero_coupon_repo_rate(seed):
    """Measured (2^18 paths): z = -1.39, +0.57, +0.45, -0.23 against Proposition "Repo rate";
    the bond forward (K = 1) is rejected at -9.81, -7.75, -7.94, -8.65 standard errors."""
    s, e, T = 1.3, 2.1, 7.5
    r, se = repo.mc_repo_rate(MC_HW, CURVES, T, s, e, n_paths=2**18, seed=seed)
    assert abs(r - repo.repo_rate(MC_HW, CURVES, s, e, T)) < 4 * se
    no_conv = (CURVES.Zbond(s) / CURVES.Zbond(e) - 1.0) / repo.accrual(s, e)
    assert abs(r - no_conv) > 4 * se


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_mc_coupon_repo_rate(seed):
    """Measured (2^18 paths): z = -0.88, +1.42, +0.72, -0.57 against the exact quadrature; the frozen-weights
    approximation (-1.17bp off at this vol scale) is rejected at -8.47, -6.09, -6.82, -8.14 standard errors."""
    r, se = repo.mc_repo_rate(MC_HW, CURVES, BOND, 1.3, 2.1, n_paths=2**18, seed=seed)
    assert abs(r - repo.coupon_repo_rate_exact(MC_HW, CURVES, BOND, 1.3, 2.1)) < 4 * se
    assert abs(r - repo.coupon_repo_rate(MC_HW, CURVES, BOND, 1.3, 2.1)) > 4 * se


def test_coupon_on_window_boundaries_exact_vs_mc():
    """Coupons exactly at s (to the seller) and at e (passed through, not returned)."""
    b = repo.CouponBond((1.0, 2.0, 3.0, 4.0), (0.06, 0.06, 0.06, 1.06))
    r, se = repo.mc_repo_rate(MC_HW, CURVES, b, 1.0, 2.0, n_paths=2**18, seed=5)
    assert abs(r - repo.coupon_repo_rate_exact(MC_HW, CURVES, b, 1.0, 2.0)) < 4 * se


# --------------------------------------------------------------------------- overnight repo (bond-collateralised derivatives)
def test_spot_overnight_repo_is_bond_rate(hw):
    """Proposition "Overnight repo": ln K_0(0, h, T) = O(h^2), so the spot overnight repo rate is rbar_0."""
    k = [repo.repo_convexity(hw, 0.0, 0.0, h, 10.0) for h in (1 / 365, 0.5 / 365)]
    assert k[0] / k[1] == pytest.approx(4.0, rel=1e-2)
    assert abs(k[0]) * 365 < 1e-6


def test_forward_overnight_repo_limit(hw):
    """ln K_0(s, s+h, T)/h -> -rho sigma eps (E(theta, s) - E(theta + kappa, s))/kappa != 0 (Hull-White)."""
    from convexity.lgcm import E
    s = 3.0
    lim = -HW["rho"] * HW["sigma"] * HW["eps"] * (E(HW["theta"], s) - E(HW["theta"] + HW["kappa"], s)) / HW["kappa"]
    for h, tol in [(1 / 365, 2e-2), (0.1 / 365, 2e-3)]:
        assert repo.repo_convexity(hw, 0.0, s, s + h, 10.0) / h == pytest.approx(lim, rel=tol)
    assert lim > 1e-4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd python && python -m pytest convexity/tests/test_repo.py -q -k "mc or overnight or boundaries"`
Expected: the MC tests FAIL with `AttributeError: module 'convexity.repo' has no attribute 'mc_repo_rate'`. The overnight tests pass already: they exercise Task 1 code against a new closed-form limit, and that is the check.

- [ ] **Step 3: Write the implementation**

In `python/convexity/repo.py`, add these imports below `from .lgcm import Model`:

```python
from .montecarlo import Simulator, cv_mean_and_stderr
from .schemes import ExactScheme
```

Append:

```python
# ---------------------------------------------------------------------- Monte Carlo
def mc_repo_rate(rm, curves, bond, s, e, n_paths=2**16, seed=7, day_count="ACT/360"):
    """Repo rate by Monte Carlo (exact LGCM scheme, antithetic). `bond` is a CouponBond or a maturity T.

    Control variates with exact means: E[D_{0,e}] = p_{0,e}, and the same repo discounted at rbar
    (the bond-collateralised repo) is worth pbar_{0,s}. Returns (rate, stderr)."""
    if not isinstance(bond, CouponBond):
        bond = CouponBond((float(bond),), (1.0,))
    _check_dates(0.0, s, e, max(bond.times))
    if not any(T > e for T in bond.times):
        raise ValueError("the bond must have a cashflow after the repo end e")
    sim = Simulator(rm.lgcm, scheme=ExactScheme())
    flows = list(zip(bond.times, bond.cashflows))
    paths = sim.simulate_dates([s, e] + [T for T, _ in flows if s < T <= e], n_paths, seed=seed)
    rs, re_ = paths(s), paths(e)
    Pbar_s = sum(c * sim.discount(rs, curves, "ff", T) for T, c in flows if T > s)
    Pex_e = sum(c * sim.discount(re_, curves, "ff", T) for T, c in flows if T > e)
    De, Dbe = sim.deflator(re_, curves, "f", "d"), sim.deflator(re_, curves, "f", "f")
    x, xbar = De * Pex_e / Pbar_s, Dbe * Pex_e / Pbar_s
    for T, c in flows:
        if s < T <= e:
            rT = paths(T)
            x = x + sim.deflator(rT, curves, "f", "d") * c / Pbar_s
            xbar = xbar + sim.deflator(rT, curves, "f", "f") * c / Pbar_s
    pe = float(curves.Zois(e))
    g, se = cv_mean_and_stderr(x / pe, [De / pe, xbar / pe], [1.0, float(curves.Zbond(s)) / pe])
    d = accrual(s, e, day_count)
    return (g - 1.0) / d, se / d
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd python && python -m pytest convexity/tests/test_repo.py -q`
Expected: all pass. The MC tests take about 0.1 s each, so they are not marked `slow`.

- [ ] **Step 5: Update the README**

In `python/convexity/README.md`, add rows to the file table after the `tests/test_calibration.py` row:

```markdown
| `repo.py` | bonds and repo (McCloud, *Repo convexity*): `RepoModel` (LGCM with block f = bond curve, c = discounting basis, no d factor, zero FX vol), `BondCurves`; `repo_convexity` (ln K^{repo}), `maturity_adjustment`, `forwardness_adjustment`, `repo_rate` (ACT/360 or ACT/365 accrual), `repo_discount_factor`; McCloud's Hull-White closed forms `hw_*`; `CouponBond`, `coupon_repo_rate` (frozen weights) and `coupon_repo_rate_exact` (Gauss–Hermite); `mc_repo_rate` |
| `tests/test_repo.py` | the checks of Remark "Verification" of section "Bonds, repo and repo convexity" |
```

Add rows to the "What is checked" table:

```markdown
| Proposition "LGCM repo convexity" | `test_repo_convexity_matches_quadrature`, `test_repo_to_maturity_is_libor_convexity`, `test_boundary_conditions` | nested quadrature (6e-15 rel.); T = e is the Libor collateral convexity (1e-12) |
| Proposition "Hull–White repo convexity" (McCloud (30), (33), (34), (38), (39), (41), (43)) | `test_hull_white_matches_mccloud`, `test_zero_mean_reversion_limit`, `test_hull_white_*_interpolation`, `test_hull_white_forward_spread_and_extrapolation`, `test_interpolation_fails_beyond_hull_white` | McCloud's closed forms (1e-11); polynomial limit at theta = kappa = 0; the interpolations fail in a two-factor piecewise model (implied theta 0.023 vs 0.030) |
| Proposition "Coupon repo rate", Proposition "Frozen weights" | `test_single_cashflow_is_zero_coupon`, `test_zero_basis_gives_bond_forward`, `test_frozen_weights_error_order`, `test_mc_coupon_repo_rate`, `test_coupon_on_window_boundaries_exact_vs_mc` | Gauss–Hermite (converged to 1e-14); error O(lambda^4) (measured ratios 16.2, 16.0, not proved); MC within 4 s.e., frozen weights rejected at 6 to 8 s.e. at vol scale 3 |
| Proposition "Overnight repo" | `test_spot_overnight_repo_is_bond_rate`, `test_forward_overnight_repo_limit` | spot: O(h^2); forward: Hull–White closed-form limit |
| Monte Carlo of the zero-coupon repo rate | `test_mc_zero_coupon_repo_rate` | 4 s.e. (seeds 1–4, 2^18 paths); bond forward (K = 1) rejected at 7.8 to 9.8 s.e. |
```

Append a section at the end:

````markdown
## Repo convexity

`repo.py` implements section "Bonds, repo and repo convexity" of the notes (McCloud, *Repo convexity*,
arXiv:1905.03316). The repo rate on the bond maturing at T over [s, e] is
1 + delta f = (pbar_{0,s}/pbar_{0,e}) K^{repo}_0(s, e, T): a forward collateral convexity, with the bond curve as the
native curve and the OIS curve as the collateral. Run from `python/`:

```python
from convexity import repo
rm = repo.RepoModel.hull_white(theta=0.05, sigma=0.01, kappa=0.4, eps=0.008, rho=-0.5)
cv = repo.BondCurves.flat(rbar=0.035, b=-0.002)
repo.repo_rate(rm, cv, 1.0, 1.25, 10.0)                                  # 3M repo in 1Y on a 10Y bond, ACT/360
repo.coupon_repo_rate_exact(rm, cv, repo.CouponBond.fixed(10.0, 0.05, 2), 1.0, 1.25)
repo.mc_repo_rate(rm, cv, 10.0, 1.0, 1.25, n_paths=2**18)               # (rate, stderr)
```

The notebook `python/notebooks/repo_convexity.py` shows the impacts: `marimo edit python/notebooks/repo_convexity.py`.
````

- [ ] **Step 6: Run the full suite**

Run: `cd python && python -m pytest convexity -q -m "not slow"`
Expected: 106 + the new tests pass, 0 failures.

- [ ] **Step 7: Commit**

```bash
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity add python/convexity/repo.py python/convexity/tests/test_repo.py python/convexity/README.md
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity commit -m "repo: Monte Carlo benchmark, overnight repo limits, README"
```

---

### Task 5: The LaTeX section

**Files:**
- Modify: `docs/CollateralDiscounting.tex`:
  - line 1844, add a label to `\begin{prop}[Collateral convexity]`;
  - insert a new section immediately before the `\newpage%` that precedes `\section{Remarks}` (around line 3847);
  - add a bibliography entry after `McCloud:CollateralVolatility` (around line 5700).

**Interfaces:**
- Consumes: the measured numbers printed by the tests of Tasks 1–4. Re-run them, and wherever a number below differs, quote the re-measured one.
- Produces: labels `sec:repo`, `prop:repo_rate`, `prop:repo_lgcm`, `coro:repo_m_f`, `prop:repo_hw`, `prop:repo_hw_curve`, `prop:repo_coupon`, `prop:repo_frozen`, `prop:repo_overnight`, `prop:repo_bond_collateral`, `rema:repo_verification`, `prop:collateral_convexity`.

- [ ] **Step 1: Add two labels to existing statements**

At line 1844, replace `\begin{prop}[Collateral convexity]` with `\begin{prop}[Collateral convexity]\label{prop:collateral_convexity}`.
At line 275, replace `\subsection{Risk-neutral pricing of collateralized trades}` with
`\subsection{Risk-neutral pricing of collateralized trades}\label{ssec:rn_pricing}` (it holds the footnote on non-cash collateral).

- [ ] **Step 2: Insert the section**

Insert this before the `\newpage%` line that precedes `\section{Remarks}`:

```latex
\newpage%
\section{Bonds, repo and repo convexity}\label{sec:repo}%

McCloud \cite{McCloud:RepoConvexity} explains the observed basis between repo discount factors, implied from repo rates, and bond discount factors, stripped from bond prices, as a convexity effect: the repo is valued with the derivative discount rate, while the bond delivered in the repo is priced with the bond discount rate. We show that this \emph{repo convexity} is a forward collateral convexity adjustment in the sense of \autoref{prop:collateral_convexity}, with the bond curve playing the role of the native curve and the derivative curve that of the collateral. The LGCM gives it in closed form for any number of factors, and McCloud's Hull--White formulas are the one factor case.

\subsection{Setting}

\begin{hypo}
Single currency, no default, no haircut. Under one risk-neutral measure, the \emph{bond curve} and the \emph{derivative curve} are
\begin{eqnarray}
	\bar p^T_t \;=\; \EE_t\left[ e^{-\int_t^T \bar r_u du} \right], \qquad
	p^T_t \;=\; \EE_t\left[ e^{-\int_t^T r_u du} \right], \qquad r_t \;=\; \bar r_t + b_t ,
\end{eqnarray}
where $\bar r$ is the bond discount rate, $r$ the derivative (OIS) discount rate and $b$ the \emph{discounting basis} (McCloud (11)). The volatilities are $d\bar p^T_t/\bar p^T_t = \bar r_t dt - \bar\Sigma_{t,T}\cdot dW_t$, $dp^T_t/p^T_t = r_t dt - \Sigma^{p}_{t,T}\cdot dW_t$, and $\Sigma^b_{t,T} := \Sigma^{p}_{t,T} - \bar\Sigma_{t,T}$. The forward measures of the two curves are $\bar\QQ^{T}$ and $\QQ^{p,T}$.
\end{hypo}

\begin{rema}[Bonds as collateral]
This is the structure of native and non-native collateral of \autoref{sec:market_observables_list}: $\bar p = Z^{f|f}$, $p = Z^{f|d}$, $b = q^{f\to d}$ and $\Sigma^b = \Sigma^{f\to d}$, with the convention $\Sigma^{d|j} = \Sigma^{d|i} + \Sigma^{i\to j}$. As noted in the footnote on non-cash collateral in \autoref{ssec:rn_pricing}, a rehypothecable security posted as collateral earns its repo rate: the bond curve is the curve of the bond as collateral. In the code, \texttt{repo.RepoModel} is an LGCM with the block $f$ for the bond curve, the block $c$ for the basis, no $d$ factor and zero FX volatility.
\end{rema}

\begin{rema}[Liquidity basis]
McCloud splits the discount rate of the bond maturing at $T$ as $\bar r^T_t = \bar r_t + \bar z^T_t$ (curve and liquidity contributions, (9)) and defines the liquidity adjustment $L^{eT}_t = \log \bar\EE^{eT}_t[\exp\int_t^e \bar s^{eT}_u du]$, $\bar s^{eT} = \bar z^T - \bar z^e$, which is $\mu_S + \frac12\sigma_S^2$ for a Gaussian $S = \int_t^e \bar s^{eT}_u du$ ((13), (16), (21)). It multiplies the repo discount factors by $e^{-L^{eT}_t}$. We set $\bar z = 0$ in the rest of this section.
\end{rema}

\subsection{Repo contracts}

\begin{defi}[Repo]
A forward-starting repo agreed at $t$ over $[s,e]$ on the bond maturing at $T$, $t\le s<e\le T$, exchanges at $s$ one unit of cash for $1/\bar p^T_s$ bonds, which are returned at $e$ against $1+\delta f^{eT}_{ts}$. The \emph{repo rate} $f^{eT}_{ts}$ makes its value zero at $t$; the repo is discounted at $r$ (McCloud (10)).
\end{defi}

\begin{rema}[Time and day count]
Model time is in ACT/365 years: the interval $e - s$ drives the convexity. The accrual fraction $\delta$ only converts the gross forward price into a quoted rate. It is a parameter of the contract: $\delta = (e-s)\cdot 365/360$ for ACT/360 (USD, EUR repo) and $\delta = e - s$ for ACT/365 (GBP). The convexity $K^{\mathrm{repo}}$ below does not depend on $\delta$.
\end{rema}

\subsection{Repo rate as a collateral convexity adjustment}

\begin{prop}[Repo rate]\label{prop:repo_rate}
\begin{eqnarray}
	1 + \delta f^{eT}_{ts} &=& \frac{\bar p^s_t}{\bar p^e_t} \, K^{\mathrm{repo}}_t(s,e,T), \qquad
	K^{\mathrm{repo}}_t(s,e,T) \;:=\; \frac{\EE^{p,e}_t\left[\bar p^T_e/\bar p^T_s\right]}{\bar\EE^{e}_t\left[\bar p^T_e/\bar p^T_s\right]}
	\;=\; \Q^{\bar p\to p}_{t,e}\left[\bar p^T_e/\bar p^T_s\right].
\end{eqnarray}
This is model-free.
\end{prop}

\begin{proof}
The zero value of the repo at $t$ is $\EE_t[e^{-\int_t^e r_u du}(\bar p^T_e/\bar p^T_s - (1+\delta f))] = 0$, so $1 + \delta f = \EE^{p,e}_t[\bar p^T_e/\bar p^T_s]$. Under $\bar\QQ^e$, $\bar p^T/\bar p^e$ and $\bar p^s/\bar p^e$ are martingales, so $\bar\EE^e_s[\bar p^T_e/\bar p^T_s] = 1/\bar p^e_s$ and $\bar\EE^e_t[1/\bar p^e_s] = \bar p^s_t/\bar p^e_t$. Divide.
\end{proof}

\subsection{Repo convexity in the LGCM}

\begin{prop}[LGCM repo convexity]\label{prop:repo_lgcm}
With deterministic volatilities,
\begin{eqnarray}
	\ln K^{\mathrm{repo}}_t(s,e,T) &=& -\int_t^s \left(\bar\Sigma_{u,e} - \bar\Sigma_{u,s}\right)\cdot\Sigma^b_{u,e}\,du
	\;-\; \int_s^e \left(\bar\Sigma_{u,e} - \bar\Sigma_{u,T}\right)\cdot\Sigma^b_{u,e}\,du .
\end{eqnarray}
\end{prop}

\begin{proof}
$v_u := \bar\EE^e_u[\bar p^T_e/\bar p^T_s]$ is a $\bar\QQ^e$-martingale with $v_u = \bar p^s_u/\bar p^e_u$ for $u\le s$ and $v_u = \bar p^T_u/(\bar p^e_u\,\bar p^T_s)$ for $s\le u\le e$. It is lognormal with volatility $\sigma^v_u = \bar\Sigma_{u,e} - \bar\Sigma_{u,s}$ on $[t,s]$ and $\bar\Sigma_{u,e} - \bar\Sigma_{u,T}$ on $[s,e]$. Apply \autoref{prop:collateral_convexity} with $i\to j = \bar p\to p$ and maturity $e$.
\end{proof}

The first term is the Libor collateral convexity \eqref{eq:libor_collateral_convexity_lgcm} of the forward rate on $[s,e]$; it is the whole convexity for a repo to maturity ($T = e$). The second term is new: during the repo window the bond still has $T-e$ to run, and its price at $e$ is correlated with the basis.

\begin{coro}[Maturity and forwardness]\label{coro:repo_m_f}
Let $C^{eT}_{ts} = \ln K^{\mathrm{repo}}_t(s,e,T)$, $M^{eT}_t = C^{eT}_{tt} = \int_t^e (\bar\Sigma_{u,T} - \bar\Sigma_{u,e})\cdot\Sigma^b_{u,e}\,du$ and $F^{eT}_{ts} = C^{eT}_{ts} - (M^{eT}_t - M^{sT}_t)$ (McCloud (17)). Then $M^{ee}_t = 0$, $F^{eT}_{tt} = 0$ and, with the repo discount factor $\hat p^{eT}_t = \bar p^e_t e^{-M^{eT}_t}$,
\begin{eqnarray}
	1 + \delta f^{eT}_{ts} &=& \frac{\hat p^{sT}_t}{\hat p^{eT}_t} e^{F^{eT}_{ts}} .
\end{eqnarray}
\end{coro}

\begin{proof}
The definitions and \autoref{prop:repo_rate}.
\end{proof}

\subsection{The Hull--White case}

\begin{prop}[Hull--White repo convexity]\label{prop:repo_hw}
Let $d\bar r_t = \theta(\bar r^*_t - \bar r_t)dt + \sigma dx_t$ and $db_t = \kappa(b^*_t - b_t)dt + \varepsilon dy_t$ with $dx_t\,dy_t = \rho\,dt$ (McCloud (27)). With $E(\beta,\tau) = (1-e^{-\beta\tau})/\beta$, $\bar\Sigma_{u,T} = \sigma E(\theta, T-u)$ and $\Sigma^b_{u,T} = \varepsilon E(\kappa, T-u)$, and
\begin{eqnarray}
	C^{eT}_{ts} &=& \frac{\rho\sigma\varepsilon}{\theta\kappa}\left( (1-e^{-\theta(T-e)})\int_t^e e^{-\theta(e-u)}(1-e^{-\kappa(e-u)})du - (1-e^{-\theta(T-s)})\int_t^s e^{-\theta(s-u)}(1-e^{-\kappa(e-u)})du \right),
\end{eqnarray}
which is McCloud (30); integrating gives his (33) and (34).
\end{prop}

\begin{proof}
$\bar\Sigma_{u,e} - \bar\Sigma_{u,s} = \sigma e^{-\theta(s-u)}E(\theta,e-s)$ and $\bar\Sigma_{u,T} - \bar\Sigma_{u,e} = \sigma e^{-\theta(e-u)}E(\theta,T-e)$. Substitute in \autoref{prop:repo_lgcm} and split $\int_t^e = \int_t^s + \int_s^e$ in (30).
\end{proof}

\begin{rema}[Check of McCloud's formulas]
McCloud's (30), (33) and (34) agree with \autoref{prop:repo_lgcm}: measured agreement $10^{-11}$ relative or better (\texttt{test\_hull\_white\_matches\_mccloud}), including the signs, with $\rho$ the correlation of $\bar r$ and $b$. The closed forms divide by $\theta\kappa(\theta+\kappa)$; the LGCM formula is stable at zero mean reversion, where $C^{eT}_{ts} = -\rho\sigma\varepsilon\left[\tfrac12(e-s)((e-t)^2-(e-s)^2) - \tfrac12(T-e)(e-s)^2\right]$ (\texttt{test\_zero\_mean\_reversion\_limit}).
\end{rema}

\subsection{Repo discount curves}

\begin{prop}[Hull--White repo curve]\label{prop:repo_hw_curve}
In the Hull--White case of \autoref{prop:repo_hw}:
\begin{enumerate}
\item The infinite maturity repo curve $\hat p^{e}_t := \hat p^{e\infty}_t$ has instantaneous forwards $\hat f^e_t = \bar f^e_t + \frac{\rho\sigma\varepsilon}{\theta\kappa}e^{-\theta(e-t)}(1-e^{-\kappa(e-t)})$ (McCloud (38)). For $e\ge E$, $\hat f^e_t - \bar f^e_t = (\hat f^E_t - \bar f^E_t)\,e^{-\theta(e-E)}\frac{1-e^{-\kappa(e-t)}}{1-e^{-\kappa(E-t)}}$ (McCloud (39)).
\item $\hat p^{eT}_t = (\bar p^e_t)^{w}(\hat p^{e}_t)^{1-w}$ with $w = e^{-\theta(T-e)}$ (McCloud (41)).
\item $F^{eT}_{ts} = (1 - e^{-(\theta+\kappa)(s-t)})\,F^{eT}_{-\infty,s}$, so $1+\delta f^{eT}_{ts} = (1+\delta f^{eT}_{ss})^{w'}(1+\delta f^{eT}_{-\infty,s})^{1-w'}$ with $w' = e^{-(\theta+\kappa)(s-t)}$ at fixed $\hat p^{sT}_t/\hat p^{eT}_t$ (McCloud (43)).
\end{enumerate}
\end{prop}

\begin{proof}
By (34), $M^{eT}_t = (1-e^{-\theta(T-e)})M^{e\infty}_t$, which gives 2.; differentiate $M^{e\infty}_t$ in $e$ for 1.; $F^{eT}_{ts}$ in (34) has the factor $(1-e^{-(\theta+\kappa)(s-t)})$, and its other factors depend on $s$, $e$, $T$ only, which gives 3.
\end{proof}

\begin{rema}[Beyond Hull--White]
The interpolations of \autoref{prop:repo_hw_curve} use a single mean reversion and do not hold with several factors or piecewise constant parameters. In a two-factor bond, one-factor basis piecewise constant model (\texttt{test\_interpolation\_fails\_beyond\_hull\_white}), the mean reversion implied by $M^{eT}/M^{e\infty} = 1-e^{-\theta(T-e)}$ is $0.0232$ at $T-e = 2$ and $0.0296$ at $T-e = 20$ (measured), and the ratio $F(\tau)/F(\infty)$ is $0.18$, $0.31$, $1.02$, $1.53$, $1.38$ at $\tau = 0.5, 1, 3, 6, 10$: not monotone and above one. Use \autoref{prop:repo_lgcm} directly.
\end{rema}

\subsection{Coupon bonds}

\begin{defi}[Repo on a coupon bond]
A coupon bond pays $c_k$ at $T_k$ and is worth $\bar P_t = \sum_{T_k>t} c_k\bar p^{T_k}_t$ (ex-coupon at the payment dates). In the repo over $[s,e]$, one unit of cash buys $1/\bar P_s$ bonds at $s$; the coupons paid in $(s,e]$ are passed through to the bond's economic owner at $T_k$ (manufactured payments, discounted at $r$); a coupon at $s$ belongs to the seller; at $e$ the ex-coupon bonds are returned against $1+\delta f$.
\end{defi}

\begin{prop}[Coupon repo rate]\label{prop:repo_coupon}
\begin{eqnarray}
	p^e_t\,(1+\delta f) &=& p^s_t\,\EE^{p,s}_t[Y_s], \qquad
	Y_s \;=\; \frac{1}{\bar P_s}\left( \sum_{T_k>e} c_k\, p^e_s\,\frac{\bar p^{T_k}_s}{\bar p^e_s}\,K^{\mathrm{repo}}_s(s,e,T_k) + \sum_{s<T_k\le e} c_k\, p^{T_k}_s \right).
\end{eqnarray}
In the LGCM every term of $Y_s$ is the exponential of an affine function of the Gaussian state $z_s$, and $\EE^{p,s}[Y_s]$ is computed exactly by Gauss--Hermite quadrature. With a single cash flow it is \autoref{prop:repo_rate}. If $b = 0$, then $Y_s = 1$ and $1+\delta f = \bar p^s_t/\bar p^e_t$ for any coupons.
\end{prop}

\begin{proof}
Condition on $\mathcal F_s$. By \autoref{prop:repo_rate} at $t = s$, $\EE^{p,e}_s[\bar p^{T_k}_e] = (\bar p^{T_k}_s/\bar p^e_s)K^{\mathrm{repo}}_s(s,e,T_k)$, and a coupon at $T_k\in(s,e]$ is worth $c_k p^{T_k}_s$ at $s$. If $b = 0$ then $p = \bar p$, $K = 1$ and the bracket is $\bar P_s$.
\end{proof}

\begin{prop}[Frozen weights]\label{prop:repo_frozen}
Write $Y_s = \sum_k a_k e^{\alpha_k\cdot z}/\sum_j c_j e^{\beta_j\cdot z}$ with $z\sim\mathcal N(\mu, P)$ under $\QQ^{p,s}$. Replacing the denominator by the lognormal with the same mean $\bar D = \sum_j c_j e^{\beta_j\cdot\mu + \frac12\beta_j^\top P\beta_j}$ and loading $\bar\beta = \sum_j w_j\beta_j$, $w_j = c_j e^{\beta_j\cdot\mu + \frac12\beta_j^\top P\beta_j}/\bar D$, gives
\begin{eqnarray}
	\EE[Y_s] &\approx& \frac{1}{\bar D}\sum_k a_k \exp\left(\alpha_k\cdot\mu + \tfrac12\bar\beta^\top P\bar\beta + \tfrac12(\alpha_k-\bar\beta)^\top P(\alpha_k-\bar\beta)\right).
\end{eqnarray}
It is exact for a single cash flow. Measured: the error of the repo rate scales as $\lambda^4$ when all volatilities are scaled by $\lambda$ (error ratios $16.2$ and $16.0$ for $\lambda = 4\to2\to1$, \texttt{test\_frozen\_weights\_error\_order}). This is an observation, not a proof.
\end{prop}

\subsection{Bond-collateralised derivatives}

\begin{prop}[Overnight repo]\label{prop:repo_overnight}
In the LGCM, as $h\to0$: $\ln K^{\mathrm{repo}}_t(t,t+h,T) = O(h^2)$, so the spot overnight repo rate on any bond of the curve converges to $\bar r_t$. For a forward start $s>t$,
\begin{eqnarray}
	\frac1h \ln K^{\mathrm{repo}}_t(s,s+h,T) &\to& -\int_t^s \partial_s\bar\Sigma_{u,s}\cdot\Sigma^b_{u,s}\,du ,
\end{eqnarray}
which is $-\rho\sigma\varepsilon\,(E(\theta,s-t)-E(\theta+\kappa,s-t))/\kappa$ in the Hull--White case: the forward overnight repo rate is not the bond forward rate.
\end{prop}

\begin{proof}
For $s = t$ the first integral of \autoref{prop:repo_lgcm} vanishes; in the second, $\bar\Sigma_{u,t+h} - \bar\Sigma_{u,T}$ is bounded and $\Sigma^b_{u,t+h} = O(h)$ on an interval of length $h$. For $s > t$, $\bar\Sigma_{u,s+h} - \bar\Sigma_{u,s} = h\,\partial_s\bar\Sigma_{u,s} + O(h^2)$ and the second integral is $O(h^2)$.
\end{proof}

\begin{prop}[Discount curve of bond-collateralised derivatives]\label{prop:repo_bond_collateral}
A derivative collateralised by a rehypothecable bond, with the collateral financed by rolling spot overnight repo (no haircut, $\bar z = 0$), has effective discount rate $\bar r$: its discount curve is the bond curve $\bar p$.
\end{prop}

\begin{proof}
By \autoref{prop:repo_overnight}, the collateral earns $\bar r_u$ at each $u$. The effective discount rate is the rate earned on the collateral (\autoref{ssec:rn_pricing} and its footnote on non-cash collateral), so $V_t = \EE_t[e^{-\int_t^T\bar r_u du}V_T]$, and $V_T = 1$ gives $\bar p^T_t$.
\end{proof}

\begin{rema}[Term repo curve or bond curve]
McCloud (\S4) proposes the extrapolated term repo curve $\hat p^e$ for discounting bond-collateralised derivatives. That corresponds to financing the collateral with term repos locked at inception. Under rolling overnight financing, \autoref{prop:repo_bond_collateral} gives $\bar p$, and the two differ by the maturity adjustment $M^{e\infty}$. The size of the difference is shown in the notebook \texttt{python/notebooks/repo\_convexity.py}.
\end{rema}

\subsection{Verification}

\begin{rema}[Verification]\label{rema:repo_verification}
The checks are in \texttt{python/convexity/tests/test\_repo.py}. Hull--White parameters are $\theta = 5\%$, $\sigma = 1\%$, $\kappa = 40\%$, $\varepsilon = 0.8\%$, $\rho = -0.5$, with flat curves $\bar r = 3.5\%$, $b = -20$bp.
\begin{enumerate}
\item \autoref{prop:repo_lgcm} against nested adaptive quadrature in a two-factor bond, one-factor basis piecewise model with zero, tiny and negative mean reversions: $6\cdot10^{-15}$ relative. For $T = e$ it equals the Libor collateral convexity to $10^{-12}$.
\item \autoref{prop:repo_hw}: McCloud's closed forms to $10^{-11}$ relative, also past the last knot and for small and negative mean reversions; the polynomial limit at $\theta = \kappa = 0$ to $10^{-12}$.
\item \autoref{prop:repo_hw_curve}: exact to $10^{-12}$ in Hull--White; the failure beyond Hull--White is quoted in the Remark above.
\item \autoref{prop:repo_coupon}: a one-cash-flow bond reproduces \autoref{prop:repo_rate} to $10^{-13}$; $b = 0$ gives the bond forward to $10^{-13}$; Gauss--Hermite with $40$ and $60$ nodes agree to $10^{-14}$.
\item Monte Carlo (exact LGCM scheme, $2^{18}$ antithetic paths, seeds $1$ to $4$, volatilities scaled by $3$, control variates $\EE[D_{0,e}] = p^e_0$ and the bond-collateralised repo of value $\bar p^s_0$): the zero-coupon repo rate ($s=1.3$, $e=2.1$, $T=7.5$, convexity $-1.20$bp) is at $z = -1.39, +0.57, +0.45, -0.23$; the bond forward ($K = 1$) is rejected at $7.8$ to $9.8$ standard errors. The coupon repo rate (10Y semi-annual 5\%, two coupons in the window) is at $z = -0.88, +1.42, +0.72, -0.57$ against the quadrature; the frozen weights approximation, $1.17$bp off at this volatility, is rejected at $6.1$ to $8.5$ standard errors.
\item \autoref{prop:repo_overnight}: $\ln K_0(0,h,10)$ scales as $h^2$; the forward overnight limit matches the Hull--White closed form ($1.14$bp at $s = 3$).
\end{enumerate}
\end{rema}
```

- [ ] **Step 3: Add the bibliography entry**

After the `\bibitem{McCloud:CollateralVolatility}` block, add:

```latex
%
\bibitem{McCloud:RepoConvexity}
\textsc{P. McCloud,}
\textit{Repo convexity},
arXiv:1905.03316, 2019.
```

- [ ] **Step 4: Check the cross-references used**

`coro` has no `\autorefname`, so the section never `\autoref`s the corollary. Check that every other `\autoref` target in the new section exists: `grep -o 'autoref{[^}]*}' docs/CollateralDiscounting.tex | sort -u`, then grep for each new one's `\label`.

- [ ] **Step 5: Build twice and check for errors and new warnings**

Run:
```bash
cd /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity/docs && pdflatex -interaction=nonstopmode CollateralDiscounting.tex > /dev/null; pdflatex -interaction=nonstopmode CollateralDiscounting.tex > /dev/null; grep -c "^!" CollateralDiscounting.log; grep -E "multiply defined|undefined" CollateralDiscounting.log
```
Expected: `0` errors. The only warnings are the three known ones: `eq:cb_r_swaplet`, `eq:forward_cb_non_resetting`, `eq:cb_r_swaplet2`.

- [ ] **Step 6: Re-measure the quoted numbers**

Run: `cd python && python -m pytest convexity/tests/test_repo.py -q -rA 2>&1 | tail -5`. Compare the numbers quoted in the Remarks against the test docstrings, and against any re-measured values recorded in Tasks 1–4. Fix any number in the LaTeX that differs.

- [ ] **Step 7: Commit**

```bash
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity add docs/CollateralDiscounting.tex
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity commit -m "notes: section Bonds, repo and repo convexity"
```

---

### Task 6: The notebook

**Files:**
- Create: `python/notebooks/repo_convexity.py`

**Interfaces:**
- Consumes: `convexity.repo` (Tasks 1–4).
- Produces: a marimo app that runs top to bottom with `marimo export html python/notebooks/repo_convexity.py -o /tmp/x.html` (or `python python/notebooks/repo_convexity.py`, without error).

- [ ] **Step 1: Load the dataviz skill**

Invoke the `dataviz` skill before writing chart code. Keep the Okabe–Ito palette and `rcParams` of `lgcm_calibration.py` so the two notebooks match. Use one colour per object across all panels, direct labels where possible, and units in the axis labels (bp).

- [ ] **Step 2: Write the notebook**

Create `python/notebooks/repo_convexity.py`:

```python
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    _py = str(Path(__file__).resolve().parent.parent)
    if _py not in sys.path:
        sys.path.insert(0, _py)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from convexity import repo

    # colour-blind-safe palette (Okabe-Ito), one colour per object across all panels
    C = {"bond": "#0072B2", "repo": "#D55E00", "ois": "#009E73", "M": "#CC79A7", "F": "#E69F00",
         "exact": "#000000", "approx": "#56B4E9", "grey": "#666666"}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})
    return C, mo, np, plt, repo


@app.cell
def _(mo):
    mo.md(r"""
    # Repo convexity

    McCloud (*Repo convexity*, 2019) explains the basis between **repo** discount factors (implied from repo
    rates) and **bond** discount factors (stripped from bond prices) as a convexity: the repo is discounted at
    the derivative (OIS) rate $r = \bar r + b$, while the bond is priced at its own rate $\bar r$.
    In the notes (section "Bonds, repo and repo convexity"), the repo rate is a forward collateral convexity:

    $$1 + \delta f^{eT}_{ts} = \frac{\bar p^s_t}{\bar p^e_t}\,K^{\mathrm{repo}}_t(s,e,T),\qquad
    \ln K = -\int_t^s(\bar\Sigma_{u,e}-\bar\Sigma_{u,s})\cdot\Sigma^b_{u,e}du-\int_s^e(\bar\Sigma_{u,e}-\bar\Sigma_{u,T})\cdot\Sigma^b_{u,e}du .$$

    Model: Hull–White bond rate $(\theta,\sigma)$ and basis $(\kappa,\varepsilon)$ with correlation $\rho$; flat
    bond curve 3.5%, basis −20bp. Parameters are illustrative, not calibrated. Rates are ACT/360.
    "Convexity" below is the repo rate minus the bond forward rate, in bp.
    """)
    return


@app.cell
def _(mo):
    rho = mo.ui.slider(-0.9, 0.9, step=0.1, value=-0.5, label="ρ (bond rate, basis)")
    sigma = mo.ui.slider(0.002, 0.03, step=0.001, value=0.01, label="σ bond rate vol")
    eps = mo.ui.slider(0.001, 0.02, step=0.001, value=0.008, label="ε basis vol")
    theta = mo.ui.slider(0.01, 0.5, step=0.01, value=0.05, label="θ bond mean reversion")
    kappa = mo.ui.slider(0.05, 2.0, step=0.05, value=0.4, label="κ basis mean reversion")
    mo.vstack([mo.md("**Parameters**"), mo.hstack([rho, sigma, eps]), mo.hstack([theta, kappa])])
    return eps, kappa, rho, sigma, theta


@app.cell
def _(eps, kappa, repo, rho, sigma, theta):
    rm = repo.RepoModel.hull_white(theta.value, sigma.value, kappa.value, eps.value, rho.value)
    cv = repo.BondCurves.flat(rbar=0.035, b=-0.002)

    def conv_bp(model, s, e, T):
        fwd = (cv.Zbond(s) / cv.Zbond(e) - 1.0) / repo.accrual(s, e)
        return 1e4 * (repo.repo_rate(model, cv, s, e, T) - fwd)
    return conv_bp, cv, rm


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Where the convexity comes from: forwardness, term, residual maturity

    McCloud's three intervals: forwardness $\tau = s - t$, repo term $e - s$, residual maturity $\mu = T - e$
    of the bond. The convexity splits into the **maturity adjustment** $M$ (absorbed in the repo discount factors
    $\hat p^{eT} = \bar p^e e^{-M^{eT}}$) and the **forwardness adjustment** $F$ (on the rate only).
    """)
    return


@app.cell
def _(C, conv_bp, np, plt, repo, rm):
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 3.6))
    _tau = np.linspace(0.0, 10.0, 41)
    for _d, _ls in [(1 / 12, ":"), (0.25, "--"), (1.0, "-")]:
        _ax[0].plot(_tau, [conv_bp(rm, t, t + _d, t + _d + 10.0) for t in _tau], color=C["repo"], ls=_ls,
                    label=f"term {12 * _d:.0f}M")
    _ax[0].set(title="vs forwardness (bond 10Y after e)", xlabel="forward start s (years)", ylabel="convexity (bp)")
    _ax[0].legend(frameon=False)
    _mu = np.linspace(0.0, 30.0, 61)
    for _s, _ls in [(0.0, "-"), (2.0, "--"), (5.0, ":")]:
        _ax[1].plot(_mu, [conv_bp(rm, _s, _s + 0.25, _s + 0.25 + m) for m in _mu], color=C["repo"], ls=_ls,
                    label=f"start {_s:.0f}Y")
    _ax[1].set(title="vs residual maturity (3M repo)", xlabel="T − e (years)", ylabel="convexity (bp)")
    _ax[1].legend(frameon=False)
    _starts = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    _e, _T = 0.25, 10.0
    _Mp = [1e4 * (repo.maturity_adjustment(rm, 0, s + _e, s + _T) - repo.maturity_adjustment(rm, 0, s, s + _T)) / repo.accrual(s, s + _e)
           for s in _starts]
    _Fp = [1e4 * repo.forwardness_adjustment(rm, 0, s, s + _e, s + _T) / repo.accrual(s, s + _e) if s > 0 else 0.0
           for s in _starts]
    _x = np.arange(len(_starts))
    _ax[2].bar(_x - 0.2, _Mp, 0.4, color=C["M"], label="maturity M")
    _ax[2].bar(_x + 0.2, _Fp, 0.4, color=C["F"], label="forwardness F")
    _ax[2].set_xticks(_x, [f"{s:.0f}Y" for s in _starts])
    _ax[2].set(title="split, 3M repo on a bond 10Y after s", xlabel="forward start", ylabel="contribution (bp, ≈ ln K / δ)")
    _ax[2].axhline(0, color=C["grey"], lw=0.8)
    _ax[2].legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Sensitivity to the parameters

    The convexity is linear in $\rho\sigma\varepsilon$ and vanishes at $\rho = 0$. The mean reversions set its
    shape: $\kappa$ how quickly a basis shock decays, $\theta$ how much of a bond rate shock reaches long bonds.
    """)
    return


@app.cell
def _(C, conv_bp, eps, np, plt, repo, sigma, theta):
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.6))
    _rhos = np.linspace(-0.9, 0.9, 19)
    for _k, _ls in [(0.1, ":"), (0.4, "--"), (1.5, "-")]:
        _ax[0].plot(_rhos, [conv_bp(repo.RepoModel.hull_white(theta.value, sigma.value, _k, eps.value, r), 2.0, 2.25, 12.25)
                            for r in _rhos], color=C["repo"], ls=_ls, label=f"κ = {_k}")
    _ax[0].set(title="3M repo in 2Y on a 12Y bond", xlabel="ρ", ylabel="convexity (bp)")
    _ax[0].axhline(0, color=C["grey"], lw=0.8)
    _ax[0].legend(frameon=False)
    _ths = np.linspace(0.01, 0.5, 25)
    for _mu, _ls in [(2.0, ":"), (10.0, "--"), (30.0, "-")]:
        _ax[1].plot(_ths, [conv_bp(repo.RepoModel.hull_white(th, sigma.value, 0.4, eps.value, -0.5), 2.0, 2.25, 2.25 + _mu)
                           for th in _ths], color=C["repo"], ls=_ls, label=f"T − e = {_mu:.0f}Y")
    _ax[1].set(title="vs θ (ρ = −0.5, κ = 0.4)", xlabel="θ", ylabel="convexity (bp)")
    _ax[1].legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Bond curve, repo curve, and extrapolating the repo curve

    The infinite-maturity repo curve $\hat p^e = \bar p^e e^{-M^{e\infty}}$ differs from the bond curve by the
    maturity adjustment. If repo quotes stop at $E$, McCloud's (39) extrapolates the forward spread
    $\hat f - \bar f$ using the bond forwards and $(\theta, \kappa)$ only. In Hull–White it is exact (dashed line on
    the solid line).
    """)
    return


@app.cell
def _(C, cv, eps, kappa, np, plt, repo, rho, rm, sigma, theta):
    _e = np.linspace(0.05, 30.0, 120)
    _pb = np.array([float(cv.Zbond(x)) for x in _e])
    _ph = np.array([repo.repo_discount_factor(rm, cv, x, x + 1e3) for x in _e])
    _po = np.array([float(cv.Zois(x)) for x in _e])
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for _p, _k, _lab in [(_pb, "bond", "bond curve p̄"), (_ph, "repo", "repo curve p̂ (T = ∞)"), (_po, "ois", "OIS curve p")]:
        _ax[0].plot(_e, -1e2 * np.log(_p) / _e, color=C[_k], label=_lab)
    _ax[0].set(title="zero rates", xlabel="maturity (years)", ylabel="zero rate (%)")
    _ax[0].legend(frameon=False)
    _P = (rho.value, sigma.value, eps.value, theta.value, kappa.value)
    _sp = [1e4 * repo.hw_forward_spread(*_P, 0.0, x) for x in _e]
    _E = 2.0
    _xe = _e[_e >= _E]
    _ext = [1e4 * repo.hw_extrapolate(repo.hw_forward_spread(*_P, 0.0, _E), 0.0, _E, x, theta.value, kappa.value) for x in _xe]
    _ax[1].plot(_e, _sp, color=C["repo"], label="model spread")
    _ax[1].plot(_xe, _ext, color=C["exact"], ls="--", label=f"extrapolated from E = {_E:.0f}Y")
    _ax[1].axvline(_E, color=C["grey"], lw=0.8)
    _ax[1].set(title="instantaneous forward spread f̂ − f̄", xlabel="maturity (years)", ylabel="spread (bp)")
    _ax[1].legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Coupon bonds

    Repo on a 10Y bond with semi-annual coupons over [1.2Y, 1.6Y] (the coupon at 1.5Y falls in the window and is
    passed through to the bond's owner). Exact: Gauss–Hermite quadrature (Proposition "Coupon repo rate"). Approximation: frozen
    weights, which is asymptotic: its error grows like vol⁴ (measured).
    """)
    return


@app.cell
def _(C, cv, np, plt, repo, rm):
    _cps = np.linspace(0.0, 0.08, 9)
    _s, _e = 1.2, 1.6
    _fwd = (cv.Zbond(_s) / cv.Zbond(_e) - 1.0) / repo.accrual(_s, _e)
    _ex, _ap = [], []
    for _c in _cps:
        _b = repo.CouponBond.fixed(10.0, _c, 2) if _c > 0 else repo.CouponBond((10.0,), (1.0,))
        _ex.append(1e4 * (repo.coupon_repo_rate_exact(rm, cv, _b, _s, _e) - _fwd))
        _ap.append(1e4 * (repo.coupon_repo_rate(rm, cv, _b, _s, _e) - _fwd))
    _fig, _ax = plt.subplots(figsize=(5.5, 3.6))
    _ax.plot(1e2 * _cps, _ex, color=C["exact"], marker="o", label="exact (quadrature)")
    _ax.plot(1e2 * _cps, _ap, color=C["approx"], ls="--", label="frozen weights")
    _ax.set(title="repo convexity of a 10Y coupon bond", xlabel="coupon (%)", ylabel="convexity (bp)")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Monte Carlo check

    Exact LGCM simulation, $2^{18}$ antithetic paths. The control variates have exact means: $\mathbb E[D_{0,e}] = p_{0,e}$,
    and the same repo discounted at $\bar r$ (bond collateral) is worth $\bar p_{0,s}$.
    """)
    return


@app.cell
def _(cv, mo, repo, rm):
    _rows = []
    for _lab, _bond, _s, _e in [("ZCB 10Y, 3M in 1Y", 10.0, 1.0, 1.25), ("ZCB 30Y, 1Y in 5Y", 30.0, 5.0, 6.0),
                                ("10Y 5% s.a., [1.2Y, 1.6Y]", repo.CouponBond.fixed(10.0, 0.05, 2), 1.2, 1.6)]:
        _r, _se = repo.mc_repo_rate(rm, cv, _bond, _s, _e, n_paths=2**18, seed=11)
        _cf = (repo.repo_rate(rm, cv, _s, _e, _bond) if not isinstance(_bond, repo.CouponBond)
               else repo.coupon_repo_rate_exact(rm, cv, _bond, _s, _e))
        _rows.append({"repo": _lab, "formula (%)": f"{1e2 * _cf:.5f}", "MC (%)": f"{1e2 * _r:.5f}",
                      "s.e. (bp)": f"{1e4 * _se:.4f}", "z": f"{(_r - _cf) / _se:+.2f}"})
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Bond-collateralised derivatives: which curve?

    With the collateral financed by rolling **overnight** repo, the collateral earns the bond rate $\bar r$
    (Proposition "Overnight repo") and the discount curve is the **bond curve** $\bar p$ (Proposition "Discount curve
    of bond-collateralised derivatives"). Discounting on the extrapolated **term repo** curve $\hat p$, as McCloud §4
    suggests, assumes term financing locked at inception. Below: the PV of a 10Y annual fixed leg of 1% on 100m notional
    under the three curves (fixed flows only, so no index convexity enters).
    """)
    return


@app.cell
def _(cv, mo, np, repo, rm):
    _T = np.arange(1.0, 11.0)
    _pv = {"bond curve p̄": sum(1e6 * float(cv.Zbond(t)) for t in _T),
           "term repo curve p̂": sum(1e6 * repo.repo_discount_factor(rm, cv, t, t + 1e3) for t in _T),
           "OIS p": sum(1e6 * float(cv.Zois(t)) for t in _T)}
    _ref = _pv["bond curve p̄"]
    mo.ui.table([{"discount curve": k, "PV": f"{v:,.0f}", "vs bond curve": f"{v - _ref:+,.0f}"} for k, v in _pv.items()],
                selection=None)
    return


if __name__ == "__main__":
    app.run()
```

- [ ] **Step 3: Run the notebook headless**

Run: `cd /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity && python python/notebooks/repo_convexity.py`
Expected: exits 0 with no traceback. Also run `marimo check python/notebooks/repo_convexity.py` if the installed marimo has `check`; expected: no errors.

- [ ] **Step 4: Inspect the charts**

Export to HTML (`marimo export html python/notebooks/repo_convexity.py -o <scratchpad>/repo_convexity.html`) and look at the figures, or render each figure with `fig.savefig` to the scratchpad and read the PNGs. Check four things: no overlapping labels, units on every axis, the extrapolation line on top of the model line, and MC z-values within ±4. Fix anything off.

- [ ] **Step 5: Commit**

```bash
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity add python/notebooks/repo_convexity.py
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity commit -m "notebook: repo convexity impacts"
```

---

### Task 7: Final verification

**Files:** none new.

- [ ] **Step 1: Full test suite including slow**

Run: `cd python && python -m pytest convexity -q`
Expected: all pass. Report the counts.

- [ ] **Step 2: LaTeX build**

Run Task 5 Step 5 again. Expected: 0 errors, only the three known warnings.

- [ ] **Step 3: Quoted numbers vs measured**

For each number in `rema:repo_verification`, in the Remarks of the new section, and in the README rows, find the test or docstring that measures it and confirm it matches. List any that don't, and fix them.

- [ ] **Step 4: Commit any fixes**

```bash
git -C /Users/leyack/Documents/QuantFinance/ClaudeProjects/convexity/.claude/worktrees/repo-convexity status --short
```
If anything changed, commit it with the message `repo: align quoted numbers with measurements`.
