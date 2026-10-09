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
