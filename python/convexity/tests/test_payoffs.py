"""Monte Carlo pricing of the payoff hierarchy against closed forms.

Statistical tests use a 4-standard-error tolerance with fixed seeds.
Run from python/ with ``python -m pytest convexity -q``.
"""
import numpy as np
import pytest

from convexity.lgcm import default_model
from convexity.montecarlo import Simulator
from convexity.schemes import EulerScheme, ExactScheme
from convexity.market import InitialCurves, fx_collateral_convexity
from convexity.payoffs import (DiscountFactor, ForwardDiscountFactor, Swap, Swaption,
                                           FXForward, FXOption, price)

Z = 4.0
M, CV = default_model(), InitialCurves()
F5 = CV.fx_forward0(5.0)


def exact_claims():
    """Payoffs with an exact LGCM price."""
    return [DiscountFactor(5, 15, "d", "d"), DiscountFactor(5, 15, "f", "f"), DiscountFactor(5, 15, "f", "d"),
            ForwardDiscountFactor(5, 6, 16, "d", "d", "T1"),
            ForwardDiscountFactor(5, 6, 16, "d", "d"), ForwardDiscountFactor(5, 6, 16, "f", "d"),
            ForwardDiscountFactor(5, 6, 16, "f", "f"),
            Swap(5, 40, 0.25, 0.04, "d", "d"), Swap(5, 40, 0.25, 0.04, "f", "f"),
            Swap(5, 40, 0.25, 0.045, "f", "d", payer=False),
            FXForward(5, 1.1, "d"), FXForward(5, 1.1, "f"),
            FXOption(5, F5, "d"), FXOption(5, F5, "f"), FXOption(5, 1.0, "f", call=False),
            FXOption(2, 1.2, "d")]


def model_free_claims():
    """Payoffs whose price only depends on the initial curves (valid in the QGCM too)."""
    return [DiscountFactor(5, 15, "d", "d"), DiscountFactor(5, 15, "f", "f"), DiscountFactor(5, 15, "f", "d"),
            ForwardDiscountFactor(5, 6, 16, "d", "d", "T1"), ForwardDiscountFactor(5, 6, 16, "f", "d", "T1"),
            Swap(5, 40, 0.25, 0.04, "d", "d"), Swap(5, 40, 0.25, 0.04, "f", "f"), FXForward(5, 1.1, "d")]


@pytest.mark.parametrize("payoff", exact_claims(), ids=lambda p: f"{type(p).__name__}-{p.kind}")
def test_lgcm_monte_carlo_matches_closed_form(payoff):
    (pv, se), = price(Simulator(M, scheme=ExactScheme(0.25)), [payoff], CV, n_paths=2**16, seed=11)
    assert abs(pv - payoff.lgcm_price(M, CV)) < Z * se


def test_qgcm_model_free_prices():
    pay = model_free_claims()
    res = price(Simulator(M, kappa=15.0, scheme=EulerScheme(0.05)), pay, CV, n_paths=2**16, seed=12)
    for p, (pv, se) in zip(pay, res):
        assert abs(pv - p.lgcm_price(M, CV)) < Z * se, f"{type(p).__name__}-{p.kind}"


def test_lgcm_swaptions_close_to_gaussian_approximation():
    """The Gaussian approximation is not exact (O(vol^2) relative error) but within MC error here."""
    pay = [Swaption(5, 40, 0.25, 0.035, "d", "d"), Swaption(5, 40, 0.25, 0.042, "f", "d"),
           Swaption(5, 40, 0.25, 0.05, "f", "f", payer=False)]
    res = price(Simulator(M, scheme=ExactScheme(0.25)), pay, CV, n_paths=2**16, seed=13)
    for p, (pv, se) in zip(pay, res):
        assert abs(pv - p.lgcm_price(M, CV)) < Z * se


def test_qgcm_swaptions_differ_from_gaussian_approximation():
    """With strong local vol the swaption price leaves the Gaussian approximation (smile)."""
    p = Swaption(5, 40, 0.25, 0.042, "f", "d")
    (pv, se), = price(Simulator(M, kappa=15.0, scheme=EulerScheme(0.05)), [p], CV, n_paths=2**16, seed=14)
    assert abs(pv - p.lgcm_price(M, CV)) > Z * se


def test_put_call_parities_hold_pathwise():
    sim = Simulator(M, kappa=10.0, scheme=EulerScheme(0.1))
    res = sim.simulate(5.0, 2**10, seed=3)
    for kind, K in (("dd", 0.035), ("fd", 0.042), ("ff", 0.05)):
        ccy, col = kind
        args = (5, 40, 0.25, K, ccy, col)
        diff = Swaption(*args).value(sim, res, CV) - Swaption(*args, payer=False).value(sim, res, CV)
        assert np.allclose(diff, Swap(*args).value(sim, res, CV), rtol=0, atol=1e-15)
    for col in ("d", "f"):
        diff = FXOption(5, 1.1, col).value(sim, res, CV) - FXOption(5, 1.1, col, call=False).value(sim, res, CV)
        assert np.allclose(diff, FXForward(5, 1.1, col).value(sim, res, CV), rtol=0, atol=1e-15)


def test_fx_forward_collateral_convexity_is_material():
    """x^{df|f}_{0,T} != x^{df|d}_{0,T}: the model's FX collateral convexity is larger than the MC error."""
    pd = FXForward(5, 0.0, "d").lgcm_price(M, CV) / FXForward(5, 0.0, "d").lgcm_forward_and_discount(M, CV)[1]
    assert abs(fx_collateral_convexity(M, 5.0) - 1) > 1e-3
    assert pd == pytest.approx(F5, rel=1e-14)


def test_unsupported_pairs_raise():
    with pytest.raises(NotImplementedError):
        DiscountFactor(5, 10, "d", "f")
    with pytest.raises(NotImplementedError):
        Swap(5, 4, 0.25, 0.03, "d", "f")


def test_pricer_handles_several_expiries():
    pay = [DiscountFactor(2, 7, "d", "d"), DiscountFactor(5, 7, "d", "d")]
    res = price(Simulator(M, scheme=ExactScheme(0.5)), pay, CV, n_paths=2**14, seed=1)
    for p, (pv, se) in zip(pay, res):
        assert abs(pv - p.lgcm_price(M, CV)) < Z * se
