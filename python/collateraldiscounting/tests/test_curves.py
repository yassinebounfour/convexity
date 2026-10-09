"""Curves, curve keys and the curve set (Task 1), instruments (Task 2), curve bootstrap (Task 3)."""
import numpy as np
import pytest

from collateraldiscounting.curves import (Index, DiscountCurveKey, ForwardCurveKey, PiecewiseFlatCurve,
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
    assert cs.fx_forward0(5.0) == pytest.approx(1.1 * np.exp(-0.095 + 0.2))
    assert (cs.kappa_d, cs.kappa_f, cs.kappa("f")) == (1.0, E3M.kappa, E3M.kappa)
    with pytest.raises(ValueError, match="model output"):
        cs.Z0("df", 1.0)
    cs[SETUP.key("df")] = FunctionCurve(lambda T: np.exp(-0.03 * T))
    np.testing.assert_allclose(cs.Z0("df", np.array([1.0, 2.0])), np.exp([-0.03, -0.06]))


from collateraldiscounting.instruments import (schedule, Swap, FXForward, ResettingCCS, Swaption, FXOption,
                                               CurrencyBasisSwaption, Quote, MarketQuotes)
from collateraldiscounting.pricers import DeterministicPricer


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


from collateraldiscounting.curve_calibration import calibrate_curves, bootstrap, BootstrapReport, CalibrationError

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
