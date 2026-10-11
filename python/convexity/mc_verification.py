"""Monte Carlo verification of the LGCM / QGCM approximations.

Run as a script for the full report (about two minutes):

    cd python && python -m convexity.mc_verification

The pytest suite runs smaller versions of the same checks.

Checks
------
1. No-arbitrage identities of Theorem QGCM under stochastic (local) volatility, and their failure
   when the stochastic y_t, v_t are replaced by their LGCM values.
2. LGCM: Gaussian approximations of Y^{f|d} and B^{f/d|d,r} against Monte Carlo (exact scheme) and
   the exact quadrature.
3. QGCM: Gaussian approximation of a native swap rate. The LGCM formula with the mean vols sigmabar
   is compared with the frozen-gradient formula using E^A[y_{T0}].
4. QGCM: the deterministic Libor collateral convexity formula of the LGCM, measured with a paired
   (common random numbers) control variate.
"""
from math import sqrt, pi

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .lgcm import default_model
from .montecarlo import Simulator, mean_and_stderr, ratio_and_stderr
from .schemes import EulerScheme, ExactScheme
from .market import InitialCurves
from .payoffs import Swaption
from . import gaussian_approx as ga


# ---------------------------------------------------------------------- helpers
def bachelier(F, K, vol, T):
    s = vol * sqrt(T)
    d = (F - K) / s
    return (F - K) * norm.cdf(d) + s * norm.pdf(d)


def implied_normal_vol(price, F, K, T):
    return brentq(lambda v: bachelier(F, K, v, T) - price, 1e-8, 1.0)


def annuity_stats(w, X, T0, strikes=(), antithetic=True):
    """Moments of X under the measure with density proportional to w; implied normal vols at strikes
    (relative to the mean). Standard errors by batch means."""
    mean, _ = ratio_and_stderr(w * X, w, antithetic)
    var, var_se = ratio_and_stderr(w * (X - mean) ** 2, w, antithetic)
    out = dict(mean=mean, var=var, var_se=var_se, nvol=sqrt(var / T0), nvol_se=0.5 * var_se / sqrt(var * T0))
    out["smile"] = {}
    for k in strikes:
        price, _ = ratio_and_stderr(w * np.maximum(X - (mean + k), 0.0), w, antithetic)
        out["smile"][k] = implied_normal_vol(price, mean, mean + k, T0)
    return out


def _fd_swap_paths(sim, res, s):
    """Y^{f|d}, B^{f/d|d,r} and the annuity-measure weights on every path (LGCM convexities K)."""
    m = sim.m
    lnP = np.stack([sim.lnZ(res, "fd", T) for T in s.Tj], axis=1)
    lnF = np.stack([sim.lnZ(res, "ff", T) for T in s.Tj], axis=1)
    P = s.Zfd0(s.Tj) / s.Zfd0(s.T0) * np.exp(lnP)
    Zff = s.Zff0(s.Tj) / s.Zff0(s.T0) * np.exp(lnF)      # normalised at T0: constants cancel in ratios
    _, _, _, _, _, KT = ga._ingredients(m, s)
    A, Y, B, _, _ = ga._rates(s, P, Zff, KT)
    w = np.exp(-res.Id + res.zx) * A          # d-value of the f annuity, deflated (constants dropped)
    return w, Y, B


# ---------------------------------------------------------------------- 1. identities
def identities(sim, T=5.0, Tp=15.0, n_paths=2**18, seed=1):
    res = sim.simulate(T, n_paths, seed=seed)
    out = {
        "Zdd": np.exp(-res.Id + sim.lnZ(res, "dd", Tp)),
        "xZfd": np.exp(-res.Id + res.zx + sim.lnZ(res, "fd", Tp)),
        "xZff": np.exp(-res.Id + res.Iq + res.zx + sim.lnZ(res, "ff", Tp)),
    }
    return {k: mean_and_stderr(v) for k, v in out.items()}   # each should be 1


# ---------------------------------------------------------------------- 2. LGCM Gaussian approximations
def lgcm_gaussian_approx(scale=1.0, n_paths=2**18, seed=2, strikes=(-0.01, 0.0, 0.01)):
    m = default_model(scale)
    s = ga.SwapSetup()
    sim = Simulator(m, scheme=ExactScheme(dt=0.5))
    res = sim.simulate(s.T0, n_paths, seed=seed)
    w, Y, B = _fd_swap_paths(sim, res, s)
    ap, ex = ga.approx(m, s), ga.exact(m, s)
    return dict(
        Y=dict(mc=annuity_stats(w, Y, s.T0, strikes), exact=ex["Y"], approx_nvol=sqrt(ap["VY"] / s.T0), fwd=ap["Y0"]),
        B=dict(mc=annuity_stats(w, B, s.T0, [k / 10 for k in strikes]), exact=ex["B"], approx_nvol=sqrt(ap["VB"] / s.T0), fwd=ap["B0"]),
    )


# ---------------------------------------------------------------------- 3. QGCM native swap rate
def qgcm_native_swap(kappa, n_paths=2**18, dt=0.02, seed=3, strikes=(-0.01, 0.0, 0.01)):
    m = default_model()
    s = ga.SwapSetup()
    sim = Simulator(m, kappa=kappa, scheme=EulerScheme(dt=dt))
    res = sim.simulate(s.T0, n_paths, seed=seed)
    swaption = Swaption(s.T0, s.n, s.theta, strike=0.0, ccy="d", collateral="d")
    A, Y = swaption.underlying(sim, res, s.curves)
    w = sim.deflator(res, s.curves, "d", "d") * A
    d = m.idx["d"]
    yd = res.Y[:, d][:, :, d]
    EAy = np.einsum("p,pij->ij", w, yd) / w.sum()               # E^A[y^{d,d}_{T0}]
    mc = annuity_stats(w, Y, s.T0, strikes)
    lg = ga.approx_native(m, s)
    fr = ga.approx_native(m, s, y=EAy)
    return dict(mc=mc, fwd=lg["Y0"], lgcm_nvol=sqrt(lg["V"] / s.T0), frozen_grad_nvol=sqrt(fr["V"] / s.T0))


# ---------------------------------------------------------------------- 4. QGCM Libor collateral convexity
def qgcm_libor_convexity(kappa, S=5.0, theta=0.25, n_paths=2**18, dt=0.02, seed=4):
    """E^{f|d,S+theta}[1+theta L^f_S] / (1+theta L^{f|f}_{0,S}) - 1, QGCM vs the LGCM formula.

    With w = x Z^{f|d} / deflator (E[w] = 1 by the no-arbitrage identity) and g the Libor ratio,
    the adjustment is E[w g] - 1 = E[w (g - 1)]: dropping the zero-mean term w - 1 is a control
    variate that removes most of the noise. The QGCM value is then estimated as the exact LGCM value
    plus the paired (common random numbers) difference of w (g - 1) between QGCM and LGCM.
    """
    m = default_model()
    Tp = S + theta

    def payoff(sim):
        res = sim.simulate(S, n_paths, seed=seed)
        w = np.exp(-res.Id + res.zx + sim.lnZ(res, "fd", Tp))
        g = np.exp(-sim.lnZ(res, "ff", Tp))                       # (1+th L^f_S) / (1+th L^{f|f}_{0,S})
        return w * (g - 1.0)

    x1 = payoff(Simulator(m, kappa=kappa, scheme=EulerScheme(dt=dt)))
    x0 = payoff(Simulator(m, scheme=EulerScheme(dt=dt)))
    diff, diff_se = mean_and_stderr(x1 - x0)
    lgcm = np.exp(-(m.I(m.Sig("f", Tp), m.Sig("c", Tp), 0, S) - m.I(m.Sig("f", S), m.Sig("c", Tp), 0, S))) - 1
    lgcm_mc, lgcm_mc_se = mean_and_stderr(x0)
    return dict(lgcm_formula=lgcm, lgcm_mc=lgcm_mc, lgcm_mc_se=lgcm_mc_se, qgcm=lgcm + diff, qgcm_se=diff_se)


# ---------------------------------------------------------------------- report
def main():
    m = default_model()
    print("1. No-arbitrage identities at T=5, T'=15 (value - 1, in standard errors); 2^18 paths")
    for label, kappa, scheme in [("LGCM exact", 0, ExactScheme(0.25)), ("LGCM euler", 0, EulerScheme(0.02)),
                                 ("QGCM k=15", 15, EulerScheme(0.02)), ("QGCM k=15, y frozen", 15, EulerScheme(0.02, "frozen"))]:
        ids = identities(Simulator(m, kappa=kappa, scheme=scheme))
        print(f"   {label:20s}" + "".join(f"  {k}: {mu - 1:+.1e} ({(mu - 1) / se:+5.1f} se)" for k, (mu, se) in ids.items()))
    print("\n2. LGCM Gaussian approximations, 5Yx10Y, normal vols in bp (MC: exact scheme, 2^18 paths)")
    for scale in (1.0, 2.0):
        r = lgcm_gaussian_approx(scale)
        for name in ("Y", "B"):
            q = r[name]
            print(f"   scale {scale}: {name}  MC {q['mc']['nvol'] * 1e4:8.3f} +/- {q['mc']['nvol_se'] * 1e4:.3f}"
                  f"   exact {sqrt(q['exact']['var'] / 5) * 1e4:8.3f}   approx {q['approx_nvol'] * 1e4:8.3f}"
                  f"   MC mean - fwd {q['mc']['mean'] - q['fwd']:+.1e}"
                  f"   MC smile " + " ".join(f"{k * 1e4:+.0f}bp:{v * 1e4:.2f}" for k, v in q['mc']['smile'].items()))
    print("\n3. QGCM native 5Yx10Y swap rate Y^{d|d}, normal vols in bp (euler, 2^18 paths)")
    for kappa in (0.0, 10.0, 20.0):
        r = qgcm_native_swap(kappa)
        print(f"   kappa {kappa:4.0f}: MC {r['mc']['nvol'] * 1e4:7.3f} +/- {r['mc']['nvol_se'] * 1e4:.3f}"
              f"   LGCM formula {r['lgcm_nvol'] * 1e4:7.3f}   frozen gradient with E^A[y] {r['frozen_grad_nvol'] * 1e4:7.3f}"
              f"   MC smile " + " ".join(f"{k * 1e4:+.0f}bp:{v * 1e4:.2f}" for k, v in r['mc']['smile'].items()))
    print("\n4. QGCM Libor collateral convexity of a 3M Libor fixing at S (relative adjustment, paired estimator)")
    for S, kappa in [(5.0, 0.0), (5.0, 10.0), (5.0, 20.0), (10.0, 0.0), (10.0, 20.0)]:
        r = qgcm_libor_convexity(kappa, S=S)
        print(f"   S {S:4.0f}  kappa {kappa:4.0f}: QGCM {r['qgcm']:+.4e} +/- {r['qgcm_se']:.1e}   LGCM formula {r['lgcm_formula']:+.4e}"
              f"   (LGCM by MC {r['lgcm_mc']:+.4e} +/- {r['lgcm_mc_se']:.1e})")


if __name__ == "__main__":
    main()
