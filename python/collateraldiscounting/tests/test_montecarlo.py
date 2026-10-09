"""Monte Carlo checks of the LGCM / QGCM (reduced versions of mc_verification.main()).

Statistical tests use a 4-standard-error tolerance; fixed seeds make them deterministic.
Run from python/ with ``python -m pytest collateraldiscounting -q``.
"""
from math import sqrt

import numpy as np
import pytest

from collateraldiscounting.lgcm import default_model
from collateraldiscounting.montecarlo import Simulator
from collateraldiscounting.schemes import EulerScheme, ExactScheme
from collateraldiscounting import mc_verification as mv, gaussian_approx as ga, transitions

Z = 4.0


def zscore(est, se, target):
    return (est - target) / se


def test_euler_y_matches_lgcm_and_exact_scheme_matches_transition_law():
    m = default_model()
    T = 3.0
    # kappa = 0: the Euler y_t is deterministic and converges to the closed form at order 1 in dt
    err = [np.abs(Simulator(m, scheme=EulerScheme(dt)).simulate(T, 2, seed=0).Y[0] - m.Y(T)).max()
           for dt in (0.05, 0.025)]
    assert 1.8 < err[0] / err[1] < 2.2
    ex = Simulator(m, scheme=ExactScheme(0.25)).simulate(T, 2**16, seed=0)
    mX, P, ix = transitions.propagate(m, np.arange(0, T + 1e-9, 0.25), "RN")
    nr = m.nr
    sample = np.column_stack([ex.z, ex.zx, ex.Id])
    target = np.concatenate([mX[:nr], [mX[ix["iZX"]], mX[ix["iID"]]]])
    se = sample.std(0) / sqrt(len(sample) / 2)               # antithetic pairs: conservative
    assert np.all(np.abs(sample.mean(0) - target) < Z * se + 1e-15)


@pytest.mark.parametrize("kappa,scheme", [(0.0, ExactScheme(0.25)), (0.0, EulerScheme(0.05)), (15.0, EulerScheme(0.05))])
def test_no_arbitrage_identities(kappa, scheme):
    ids = mv.identities(Simulator(default_model(), kappa=kappa, scheme=scheme), n_paths=2**16)
    for name, (mu, se) in ids.items():
        assert abs(zscore(mu, se, 1.0)) < Z, name


def test_frozen_y_breaks_the_identities_in_the_qgcm():
    ids = mv.identities(Simulator(default_model(), kappa=15.0, scheme=EulerScheme(0.05, "frozen")), n_paths=2**17)
    mu, se = ids["xZff"]
    assert abs(zscore(mu, se, 1.0)) > Z


def test_lgcm_monte_carlo_matches_exact_quadrature():
    r = mv.lgcm_gaussian_approx(1.0, n_paths=2**16, seed=5, strikes=())
    for name in ("Y", "B"):
        q = r[name]
        assert abs(zscore(q["mc"]["nvol"], q["mc"]["nvol_se"], sqrt(q["exact"]["var"] / 5.0))) < Z, name


def test_qgcm_native_swap_lgcm_formula_valid_at_zero_kappa_rejected_at_large_kappa():
    r0 = mv.qgcm_native_swap(0.0, n_paths=2**16, dt=0.05, strikes=())
    assert abs(zscore(r0["mc"]["nvol"], r0["mc"]["nvol_se"], r0["lgcm_nvol"])) < Z
    r1 = mv.qgcm_native_swap(20.0, n_paths=2**17, dt=0.05, strikes=())
    assert zscore(r1["mc"]["nvol"], r1["mc"]["nvol_se"], r1["lgcm_nvol"]) > Z


def test_libor_collateral_convexity_lgcm_formula():
    r = mv.qgcm_libor_convexity(0.0, n_paths=2**16, dt=0.05)
    assert abs(zscore(r["lgcm_mc"], r["lgcm_mc_se"], r["lgcm_formula"])) < Z
