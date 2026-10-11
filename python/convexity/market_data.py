"""Example markets, all with domestic d = USD (collateral and index SOFR):

* "EURUSD"  f = EUR, collateral ESTR, floating index EURIBOR 3M (Libor-type coupon), FX quoted as USD per EUR.
* "GBPUSD"  f = GBP, collateral and index SONIA (compounded overnight coupon, daily on_step = 1/365),
            RFR swaps annual fixed vs compounded, resetting CCS quarterly, FX quoted as USD per GBP.
* "USDJPY"  f = JPY, collateral and index TONA (compounded overnight coupon, daily on_step = 1/365),
            FX quoted INVERTED as JPY per USD (the model spot is x_0 = 1/spot, units of USD per JPY),
            forwards in points of 1/100 (pip_factor 100).

The numbers are rounded, plausible and illustrative of the kind of levels seen on a broker screen:
they are NOT live data. Units: rates and spreads in decimals, FX forward points in pips (1e-4; 1e-2 for
USDJPY), vols in decimals (normal vols: 0.0100 = 100 bp/year). In the dictionaries rates are in percent,
basis, swaption and CB vols in bp, FX vols in percent.
"""
from .curves import Index, LGCMSetup
from .instruments import (CurrencyBasisSwaption, FXForward, FXOption, MarketQuotes, Quote, ResettingCCS, Swap,
                          Swaption)

SOFR = Index("USD_SOFR", "USD")
ESTR = Index("EUR_ESTR", "EUR")
EURIBOR3M = Index("EUR_EURIBOR3M", "EUR", "IBOR", 0.25, 0.0010)
SONIA = Index("GBP_SONIA", "GBP")             # daily compounded, on_step = 1/365
TONA = Index("JPY_TONA", "JPY")

MARKETS = ("EURUSD", "GBPUSD", "USDJPY")


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




# ---------------------------------------------------------------------- GBPUSD (SOFR / SONIA)
GBP_SWAPS = {1: 4.05, 2: 3.88, 3: 3.80, 5: 3.78, 7: 3.85, 10: 3.95, 15: 4.00, 20: 3.92, 30: 3.80}
GBP_FX_POINTS = {1 / 12: 3.1, 0.25: 9.2, 0.5: 18.4, 0.75: 27.7, 1.0: 37.0}
GBP_CCS_BP = {2: -4.0, 3: -5.0, 5: -6.0, 7: -6.5, 10: -7.0, 15: -6.5, 20: -6.0, 30: -5.0}
GBP_SWAPTION_BP = {1: 98, 2: 102, 3: 104, 5: 102, 7: 98, 10: 92, 15: 85, 20: 80}
GBP_FX_VOL = {1 / 12: 7.8, 0.25: 8.0, 0.5: 8.2, 1: 8.5, 2: 8.8, 3: 9.0, 5: 9.4, 7: 9.8, 10: 10.4, 15: 11.5}
GBP_CB_VOL_BP = {1: 6.0, 2: 6.5, 3: 7.0, 5: 7.5, 7: 7.5, 10: 7.0, 15: 6.5, 20: 6.0}

# ---------------------------------------------------------------------- USDJPY (SOFR / TONA), inverted quotes
JPY_SWAPS = {1: 0.55, 2: 0.70, 3: 0.80, 5: 0.95, 7: 1.10, 10: 1.30, 15: 1.55, 20: 1.70, 30: 1.80}
JPY_SPOT = 150.0                                  # JPY per USD
JPY_FX_POINTS = {1 / 12: -49.0, 0.25: -148.0, 0.5: -299.0, 0.75: -447.0, 1.0: -593.0}   # in 1/100 JPY
JPY_CCS_BP = {2: -28.0, 3: -30.0, 5: -32.0, 7: -33.0, 10: -34.0, 15: -32.0, 20: -30.0, 30: -26.0}
JPY_SWAPTION_BP = {1: 45, 2: 52, 3: 58, 5: 64, 7: 66, 10: 62, 15: 55, 20: 48}
JPY_FX_VOL = {1 / 12: 9.5, 0.25: 9.8, 0.5: 10.0, 1: 10.2, 2: 10.4, 3: 10.6, 5: 10.9, 7: 11.2, 10: 11.6, 15: 12.2}
JPY_CB_VOL_BP = {1: 10.0, 2: 11.0, 3: 12.0, 5: 13.0, 7: 13.0, 10: 12.0, 15: 11.0, 20: 10.0}


def _market(setup, swaps, spot, points, ccs, swaptions, fx_vol, cb_vol, fidx, inverted=False, pip=1e4):
    q = [Quote(Swap(SOFR, T), r / 100) for T, r in USD_SWAPS.items()]
    q += [Quote(Swap(fidx, T), r / 100) for T, r in swaps.items()]
    q += [Quote(FXForward(T, "points", pip, inverted), p) for T, p in points.items()]
    q += [Quote(ResettingCCS(T), b / 1e4) for T, b in ccs.items()]
    q += [Quote(Swaption.make(SOFR, T, SWAPTION_TENOR), v / 1e4) for T, v in USD_SWAPTION_BP.items()]
    q += [Quote(Swaption.make(fidx, T, SWAPTION_TENOR), v / 1e4) for T, v in swaptions.items()]
    q += [Quote(FXOption(T), v / 100) for T, v in fx_vol.items()]
    q += [Quote(CurrencyBasisSwaption.make(T, CB_VOL_TENOR), v / 1e4) for T, v in cb_vol.items()]
    return MarketQuotes(setup, spot, q, inverted)


def example_quotes(market="EURUSD"):
    """Quotes of one of MARKETS (default EURUSD, the original example)."""
    if market == "EURUSD":
        return _market(example_setup(), EUR_SWAPS, 1.10, FX_POINTS, CCS_BP, EUR_SWAPTION_BP, FX_VOL, CB_VOL_BP, EURIBOR3M)
    if market == "GBPUSD":
        return _market(LGCMSetup(SOFR, SONIA), GBP_SWAPS, 1.27, GBP_FX_POINTS, GBP_CCS_BP, GBP_SWAPTION_BP,
                       GBP_FX_VOL, GBP_CB_VOL_BP, SONIA)
    if market == "USDJPY":
        return _market(LGCMSetup(SOFR, TONA), JPY_SWAPS, JPY_SPOT, JPY_FX_POINTS, JPY_CCS_BP, JPY_SWAPTION_BP,
                       JPY_FX_VOL, JPY_CB_VOL_BP, TONA, inverted=True, pip=100.0)
    raise ValueError(f"unknown market {market!r}, expected one of {MARKETS}")
