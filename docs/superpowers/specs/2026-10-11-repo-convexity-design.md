# Bonds and repo convexity — design

Date: 2026-10-11. Branch: `repo-convexity` (off `main` after the LGCM calibration merge, #1),
worktree `.claude/worktrees/repo-convexity`.

## Goal

Extend the notes and the numerical companion to bonds and repos, following McCloud, *Repo
convexity* (arXiv:1905.03316, 2019). McCloud explains the basis between repo discount factors
(implied from repo rates) and bond discount factors (stripped from bond prices) as a convexity
effect. It comes from the decorrelation between the derivative discount rate r and the bond
discount rate r̄. He works it out in a two-factor Hull–White model.

Here the result is **embedded in the collateral framework of the notes**:

- the repo rate is a forward collateral convexity adjustment;
- the general formula is the LGCM one, with multi-factor, piecewise-constant parameters, computed
  by `lgcm.Model.covint`;
- McCloud's closed forms are a special case of it, used as an independent benchmark.

The scope covers zero-coupon and fixed coupon bonds. Haircuts and default are out of scope.

Deliverables:

1. a new section in `docs/CollateralDiscounting.tex`;
2. `python/convexity/repo.py` with tests;
3. a marimo notebook `python/notebooks/repo_convexity.py` illustrating the impacts;
4. README rows.

## Setting and mapping

Single currency. There are two curves:

- **bond curve**: discount bonds p̄^T_t with discount rate r̄;
- **derivative (OIS) curve**: p^T_t with rate r = r̄ + b. The process b is McCloud's discounting
  basis (11).

Both are expectations under one risk-neutral measure:

- p̄^T_t = E_t[exp(-∫ r̄)];
- p^T_t = E_t[exp(-∫ r)].

This is the structure of "native vs non-native collateral" in the notes:

- r̄ plays the native rate;
- r plays the collateral rate;
- b plays the collateral basis q.

The footnote of the notes at line 313 (a rehypothecable security posted as collateral earns its
repo rate) is the economic link. A Remark in the new section makes it explicit.

**Implementation device.** Use the existing LGCM blocks f (bond curve: Z^{f|f} = p̄) and c (basis:
Z^{f|d} = p). The block d gets zero volatility and FX gets zero volatility. A thin adapter
`RepoModel` owns this mapping, so no other code relies on the block names meaning "bond". The sign
convention of Σ^c against b is checked once, in a test, against `market.libor_convexity`.

**Liquidity.** McCloud splits the bond rate as r̄^T = r̄ + z̄^T and defines a liquidity adjustment
L (13), (16), (21). These go in a Remark, with the Gaussian formula. Everywhere else z̄ = 0, as in
McCloud §3.

## Repo contract

Forward-starting repo on the bond maturing at T, agreed at t, over [s, e] with t ≤ s < e ≤ T and
accrual δ:

- at s: one unit of cash is exchanged for 1/p̄^T_s bonds;
- at e: the bonds are returned against 1 + δ f^{eT}_{ts}.

The repo is discounted at r (McCloud (10)). Its zero value gives

    1 + δ f = E^{p,e}_t[p̄^T_e / p̄^T_s] = (p̄^s_t / p̄^e_t) K^{repo}_t(s, e, T)

where E^{p,e} is the e-forward measure of the derivative curve and K^{repo} is the forward
collateral convexity Q^{bond→OIS}_{t,e}[p̄^T_e / p̄^T_s].

## Theory (new section "Bonds, repo and repo convexity", before "Remarks")

Every statement is labelled. Exact, asymptotic and conjectural results are kept apart.

1. **Proposition (repo rate as collateral convexity, model-free).** Statement above. Proof: under
   the bond e-forward measure, E_s[p̄^T_e / p̄^e_e] = p̄^T_s / p̄^e_s. By the tower property this gives
   E^{p̄,e}_t[p̄^T_e / p̄^T_s] = p̄^s_t / p̄^e_t. Then apply the definition of Q.

2. **Proposition (LGCM repo convexity, exact).** In the LGCM (or any Gaussian HJM), the vol of the
   Q^{p̄,e}-martingale E_u[p̄^T_e / p̄^T_s] is:
   - Σ̄_{u,s} - Σ̄_{u,e} for u < s;
   - Σ̄_{u,T} - Σ̄_{u,e} for s ≤ u < e.

   The Proposition "Collateral convexity" then gives

       ln K^{repo}_t(s,e,T) = -∫_t^s (Σ̄_{u,s} - Σ̄_{u,e})·Σ^{c}_{u,e} du - ∫_s^e (Σ̄_{u,T} - Σ̄_{u,e})·Σ^{c}_{u,e} du

   up to the sign convention of Σ^c. That convention is fixed against `market.libor_convexity`.
   - The first term is the Libor collateral convexity. It is the whole result when T = e.
   - The second term is the convexity over the repo window, which only exists when T > e.

   **Corollary:**
   - McCloud's maturity adjustment M^{eT}_t = C^{eT}_{tt} is the s = t case;
   - his forwardness adjustment is F^{eT}_{ts} = C^{eT}_{ts} - (M^{eT}_t - M^{sT}_t);
   - the boundary conditions are M^{ee} = 0 and F^{eT}_{tt} = 0;
   - the repo rate is (19) with the repo discount factor p̂^{eT}_t = p̄^e_t exp(-M^{eT}_t) (20).

3. **Proposition (Hull–White special case).** The bond rate is one factor (θ, σ) and the basis is
   one factor (κ, ε), with correlation ρ. Then Proposition 2 reduces to McCloud's (30), (33), (34).
   The reduction is checked in sympy. Any discrepancy in sign or terms with the paper is recorded
   in a Remark, quoting the check.

4. **Repo discount curve.** These three results hold exactly in the Hull–White case:
   - McCloud's infinite-maturity repo curve p̂^e (37);
   - the forward relation (38) and the extrapolation (39);
   - the geometric interpolations in maturity (41) and in forwardness (43).

   They are stated as Propositions for that case. With several factors or piecewise parameters they
   are approximations. The gap is measured and quoted in a Remark, not asserted.

5. **Coupon bonds.**
   - The bond P̄_t = Σ_k c_k p̄^{T_k}_t pays fixed cashflows c_k at T_k (the last one includes the
     principal).
   - In the repo on a coupon bond, coupons paid in (s, e] are passed through as manufactured
     payments to the bond's economic owner (the cash borrower). They are discounted at r.
   - The zero-value condition is
         0 = E[D_{t,e}(P̄_e/P̄_s) + Σ_{s<T_k≤e} D_{t,T_k} c_k/P̄_s] - p^e_t (1 + δ f),
     where D_{t,u} = exp(-∫_t^u r) and P̄_e is the ex-coupon price.
   - 1/P̄_s is not lognormal, so there is no exact closed form.
   - Approximation (asymptotic): frozen weights w_k = c_k p̄^{T_k}_t / P̄_t, a lognormal P̄ with
     loadings Σ_k w_k Σ̄_{·,T_k}, and the Proposition 2 formula for each term.
   - Benchmark: exact Gauss–Hermite quadrature over the Gaussian state in the 1+1 factor case.
   - The error is measured under vol scaling λ, and the order is quoted as measured (expected
     O(λ²), not proved).

6. **Bond-collateralised derivatives.**
   - Hypothesis, to be derived and then checked: a derivative is collateralised with a
     rehypothecable bond, and the collateral is financed by rolling overnight repo. Then its
     effective discount rate is the overnight repo rate.
   - By Proposition 2 with e = s + dt, the convexity term is O(dt²), so the overnight repo rate
     is r̄^T (the bond discount rate, with z̄ = 0). That makes the discount curve the bond curve
     p̄, not McCloud's term repo curve p̂.
   - The gap between discounting on p̂ and on p̄ is the impact quoted in the notebook.
   - If the derivation disagrees with the hypothesis, the notes say so and state what holds.
   - McCloud §4 proposes extrapolating p̂ to discount bond-collateralised derivatives. A Remark
     compares the two and states the funding assumption behind each.

7. **Remark "Verification"**, quoting the measured numbers from the tests.

Bibliography: add `McCloud:RepoConvexity` (P. McCloud, *Repo convexity*, arXiv:1905.03316, 2019).

The section must keep the build at 0 errors. It adds no new warnings beyond the three known ones
(CLAUDE.md).

## Python: `python/convexity/repo.py`

| name | contents |
|---|---|
| `RepoModel` | adapter: builds an `lgcm.Model` with blocks f = bond curve and c = basis; d and FX have zero vol; general factors and piecewise-constant parameters; `from_hull_white(theta, sigma, kappa, eps, rho)` |
| `BondCurves` | initial curves p̄_0 (bond) and p_0 (OIS), e.g. flat or `curves.PiecewiseFlatCurve` |
| `repo_convexity(m, t, s, e, T)` | ln K^{repo}, Proposition 2, via `covint` |
| `maturity_adjustment`, `forwardness_adjustment` | M and F (Corollary) |
| `repo_rate(m, curves, t, s, e, T)` | zero-coupon repo rate (simple compounding, δ = e - s) |
| `repo_discount_factor(m, curves, e, T)` | p̂^{eT}_0 |
| `hw_B`, `hw_M`, `hw_F`, `hw_repo_rate`, `hw_extrapolate`, `hw_interpolate_*` | McCloud's closed forms, written independently of `lgcm` |
| `CouponBond(times, cashflows)` | price on the bond curve |
| `coupon_repo_rate(m, curves, bond, t, s, e)` | frozen-weights approximation |
| `coupon_repo_rate_exact(...)` | Gauss–Hermite benchmark (1+1 factors) |
| `RepoPayoff` | Monte Carlo payoff on `Simulator.simulate_dates` (exact scheme): (p̄^T_e/p̄^T_s) paid at e under OIS discounting, plus pass-through coupons |

## Tests: `python/convexity/tests/test_repo.py`

- Proposition 2 against `covint_brute` (nested quadrature) on the multi-factor test model.
- The T = e case equals `market.libor_convexity` (fixes the Σ^c sign).
- The Hull–White case of Proposition 2 equals McCloud's closed forms (rel. 1e-12), including the
  θ, κ → 0 limits.
- Boundary conditions M^{ee} = 0, F^{eT}_{tt} = 0, and C = (M^{eT} - M^{sT}) + F.
- The interpolation and extrapolation identities (39), (41), (43): exact in Hull–White; the gap in
  a two-factor, piecewise model is measured and nonzero.
- Coupon bond: with a single cashflow it equals the zero-coupon repo rate (exact). The
  frozen-weights approximation is checked against quadrature, with the error ratio under λ → λ/2
  measured.
- Monte Carlo: the repo rate is within 4 s.e. of Proposition 2 (fixed seed, a control variate if
  needed). Negative control: K = 1 is rejected (> 4 s.e.) at a material vol.
- Bond-collateralised discounting (item 6): the overnight-repo limit of `repo_rate` (e → s)
  converges to the bond forward rate, with the rate measured.

Monte Carlo tests that take longer than a few seconds are marked `slow`.

## Notebook: `python/notebooks/repo_convexity.py` (marimo)

The parameters are illustrative, not live data: flat curves, and Hull–White parameters of
McCloud's order of magnitude. Sections:

1. Setting and parameters (sliders for ρ, σ, ε, θ, κ).
2. Repo convexity in bp against forwardness τ = s - t, repo term δ = e - s and residual maturity
   μ = T - e, with the split into M and F.
3. Sensitivity to ρ, κ, θ.
4. Bond and repo discount curves, the basis in bp, and the extrapolation beyond the last repo quote
   (39).
5. Coupon and zero-coupon bonds: repo rates and convexity.
6. Monte Carlo against the formula (a small number of paths, with the standard error).
7. PV of a bond-collateralised swap discounted on p̄, on p̂ and on OIS.

The charts follow the dataviz skill.

## Out of scope

Haircuts, default and recovery, special-vs-GC repo dynamics (z̄ is set to zero except in the
Remark), calibration of the basis vol and correlation to repo quotes, and day counts and calendars
(δ = e - s).
