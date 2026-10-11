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
# Long-dated FX vols must exceed what the rates and basis vols already deliver (about 8.9% at 15Y
# with sigma^x = 0 after 5Y): 7Y-15Y are set so that sigma^x stays flat around 7-8%.
FX_VOL = {1 / 12: 7.0, 0.25: 7.2, 0.5: 7.4, 1: 7.6, 2: 7.9, 3: 8.1, 5: 8.5, 7: 8.9, 10: 9.5, 15: 10.7}
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
