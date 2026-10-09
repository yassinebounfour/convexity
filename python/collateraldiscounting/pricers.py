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
