# region imports
from AlgorithmImports import *
# endregion
"""
Shared Deribit venue model for the V2 LEAN backtests.

Everything that the original project got wrong at the engine level is handled here, in code:
  * 24/7 trading in UTC (registered in the algorithm, not in database files the CLI overwrites)
  * 1 contract = 1 coin, lot 0.1 BTC / 1 ETH, perpetual lot 0.001 BTC / 0.01 ETH
  * European options, no simulated early assignment, cash settlement at 08:00 UTC on expiry day at
    Deribit's official delivery price (30-minute TWAP of the index), never at a later LEAN price
  * Deribit fees: options 0.03% of underlying per coin capped at 12.5% of the premium, delivery 0.015%
    capped at 12.5% of intrinsic, perpetual 0.05% taker; perpetual hourly funding is charged
  * market orders fill at the modelled bid/ask; limit orders (market making) fill only when a real
    Deribit trade printed through the limit price, capped at a share of the printed volume
  * contract choice is restricted to contracts that had already traded (chain files list first-trade time)
  * Greeks use each contract's own implied volatility (from its quote), never a constant
"""
import bisect
import csv
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

SECONDS_PER_YEAR = 365.0 * 86400.0
LOTS = {"BTC": (0.1, 0.001), "ETH": (1.0, 0.01)}
OPTION_FEE, DELIVERY_FEE, PERP_FEE, FEE_CAP = 0.0003, 0.00015, 0.0005, 0.125


# ------------------------------------------------------------------------------------------ pricing
def bs_price(F, K, T, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0) if is_call else max(K - F, 0.0)
    sT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sT * sT) / sT
    d2 = d1 - sT
    c = F * norm.cdf(d1) - K * norm.cdf(d2)
    return c if is_call else c - (F - K)


def bs_delta(F, K, T, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return (1.0 if F > K else 0.0) if is_call else (-1.0 if F < K else 0.0)
    sT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sT * sT) / sT
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def bs_gamma(F, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    sT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sT * sT) / sT
    return norm.pdf(d1) / (F * sT)


def bs_vega(F, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    sT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sT * sT) / sT
    return F * norm.pdf(d1) * math.sqrt(T)


def implied_vol(price, F, K, T, is_call):
    if T <= 0 or price <= 0:
        return None
    lo, hi = bs_price(F, K, T, 0.01, is_call), bs_price(F, K, T, 6.0, is_call)
    if not (lo < price < hi):
        return None
    return brentq(lambda s: bs_price(F, K, T, s, is_call) - price, 0.01, 6.0, xtol=1e-6)


# ------------------------------------------------------------------------------------------ SABR (beta = 1)
def sabr_vol(F, K, T, alpha, rho, nu):
    """Hagan et al. (2002) lognormal implied volatility for beta = 1."""
    if abs(math.log(F / K)) < 1e-9:
        zx = 1.0
    else:
        z = nu / alpha * math.log(F / K)
        x = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
        zx = z / x
    return alpha * zx * (1 + (rho * nu * alpha / 4 + (2 - 3 * rho * rho) * nu * nu / 24) * T)


def sabr_calibrate(F, T, strikes, vols):
    """Least-squares fit of (alpha, rho, nu) to the observed smile of one expiry."""
    strikes, vols = np.asarray(strikes, float), np.asarray(vols, float)
    atm = float(np.interp(F, strikes, vols)) if len(strikes) > 1 else float(vols[0])

    def resid(p):
        a, r, n = p
        return [sabr_vol(F, k, T, a, r, n) - v for k, v in zip(strikes, vols)]

    sol = least_squares(resid, x0=[atm, -0.1, 1.0], bounds=([0.01, -0.999, 0.01], [5.0, 0.999, 10.0]))
    return tuple(sol.x)


def sabr_bartlett_delta(F, K, T, params, is_call, h_rel=1e-4):
    """Bartlett (2006) smile-consistent delta: bump F and move alpha by rho*nu*dF/F (beta = 1)."""
    a, r, n = params
    h = F * h_rel

    def px(f, alpha):
        return bs_price(f, K, T, sabr_vol(f, K, T, alpha, r, n), is_call)

    return (px(F + h, a + r * n * h / F) - px(F - h, a - r * n * h / F)) / (2 * h)


# ------------------------------------------------------------------------------------------ fees & fills
class DeribitFeeModel(FeeModel):
    def __init__(self, venue):
        super().__init__()
        self.venue = venue

    def GetOrderFee(self, parameters):
        sec, order = parameters.Security, parameters.Order
        q = float(order.AbsoluteQuantity)
        tag = order.Tag or ""
        if sec.Type == SecurityType.Option:
            s = self.venue.spot()
            if tag.startswith("SETTLE@"):
                intrinsic = float(tag.split("@")[1].split()[0])
                fee = min(DELIVERY_FEE * s, FEE_CAP * intrinsic) * q
            elif tag.startswith("MMFILL@"):          # Deribit charges makers the same 0.03% on options
                fee = min(OPTION_FEE * s, FEE_CAP * float(tag.split("@")[1].split()[0])) * q
            else:
                if order.Type == OrderType.Limit:
                    px = float(order.LimitPrice)
                else:  # the price the fill model will use: ask for buys, bid for sells
                    px = float(sec.AskPrice if order.Direction == OrderDirection.Buy else sec.BidPrice) or float(sec.Price)
                fee = min(OPTION_FEE * s, FEE_CAP * max(px, 0.0)) * q
        else:
            fee = self.venue.perp_fee * q * float(sec.Price)
        return OrderFee(CashAmount(fee, "USD"))

    def get_order_fee(self, parameters):
        return self.GetOrderFee(parameters)


class DeribitFillModel(ImmediateFillModel):
    """SETTLE@<price>: cash settlement at the official delivery price.
    Limit orders: filled only if a real Deribit trade printed at/through the limit during the last bar,
    for at most `participation` of the printed volume (no queue priority is assumed beyond that)."""

    def __init__(self, venue, participation=0.25, spread_mult=1.0):
        super().__init__()
        self.venue = venue
        self.participation = participation
        self.spread_mult = spread_mult          # sensitivity runs: scale the quoted half-spread

    def MarketFill(self, asset, order):
        tag = order.Tag or ""
        if tag.startswith("SETTLE@") or tag.startswith("MMFILL@"):
            fill = OrderEvent(order, self.venue.algo.UtcTime, OrderFee.Zero)
            fill.FillPrice = float(tag.split("@")[1].split()[0])
            fill.FillQuantity = order.Quantity
            fill.Status = OrderStatus.Filled
            fill.Message = ("Deribit cash settlement at official delivery price" if tag.startswith("SETTLE@")
                            else "market-making quote hit by a real Deribit trade")
            return fill
        fill = super().MarketFill(asset, order)
        if self.spread_mult != 1.0 and fill.Status == OrderStatus.Filled:
            b, a = float(asset.BidPrice), float(asset.AskPrice)
            if a > 0 and b >= 0:
                mid = 0.5 * (a + b)
                fill.FillPrice = max(mid + self.spread_mult * (float(fill.FillPrice) - mid), 0.0)
        return fill

    def market_fill(self, asset, order):
        return self.MarketFill(asset, order)

    def LimitFill(self, asset, order):
        fill = OrderEvent(order, self.venue.algo.UtcTime, OrderFee.Zero)
        bar = asset.Cache.GetData[TradeBar]()
        # only a real print that ended at this time step counts (never a fill-forwarded copy)
        if bar is None or bar.IsFillForward or bar.EndTime != asset.LocalTime or bar.Volume <= 0:
            return fill
        lim = float(order.LimitPrice)
        crossed = (order.Direction == OrderDirection.Sell and float(bar.High) >= lim) or \
                  (order.Direction == OrderDirection.Buy and float(bar.Low) <= lim)
        if not crossed:
            return fill
        ticket = self.venue.algo.Transactions.GetOrderTicket(order.Id)
        done = float(order.AbsoluteQuantity) - abs(float(ticket.QuantityFilled) if ticket else 0.0)
        lot = asset.SymbolProperties.LotSize
        q = min(done, math.floor(float(bar.Volume) * self.participation / float(lot)) * float(lot))
        if q <= 0:
            return fill
        fill.FillPrice = lim
        fill.FillQuantity = q if order.Direction == OrderDirection.Buy else -q
        fill.Status = OrderStatus.Filled if q >= done - 1e-12 else OrderStatus.PartiallyFilled
        return fill

    def limit_fill(self, asset, order):
        return self.LimitFill(asset, order)


# ------------------------------------------------------------------------------------------ venue
class DeribitVenue:
    def __init__(self, algo, coin, underlying_resolution=Resolution.Minute, perp_leverage=10.0,
                 mm_participation=0.25, spread_mult=None):
        self.algo, self.coin = algo, coin
        self.funding_paid = 0.0
        self.settled = []
        opt_lot, perp_lot = LOTS[coin]
        hours = SecurityExchangeHours.AlwaysOpen(TimeZones.Utc)
        for st in (SecurityType.Equity, SecurityType.Option):
            algo.MarketHoursDatabase.SetEntry(Market.USA, coin, st, hours, TimeZones.Utc)
        algo.SymbolPropertiesDatabase.SetEntry(Market.USA, coin, SecurityType.Option,
                                               SymbolProperties(f"Deribit {coin} option", "USD", 1, 0.0001, opt_lot, coin))
        algo.SymbolPropertiesDatabase.SetEntry(Market.USA, coin, SecurityType.Equity,
                                               SymbolProperties(f"Deribit {coin}-PERPETUAL", "USD", 1, 0.01, perp_lot, coin))
        u = algo.AddEquity(coin, underlying_resolution, Market.USA, True, perp_leverage, False)
        u.SetDataNormalizationMode(DataNormalizationMode.Raw)
        u.SetFeeModel(DeribitFeeModel(self))
        self.u = u.Symbol
        self.opt_lot, self.perp_lot = opt_lot, perp_lot
        if spread_mult is None:
            spread_mult = float(algo.GetParameter("spread_mult") or 1.0)
        self.perp_fee = float(algo.GetParameter("perp_fee") or PERP_FEE)   # sensitivity: e.g. 0 for maker fills
        self.spread_mult = spread_mult
        self.fill_model = DeribitFillModel(self, mm_participation, spread_mult)
        root = os.path.join(Globals.DataFolder, "alternative", "deribit")
        self.chain_dir = os.path.join(root, "chain", coin.lower())
        self.flow_dir = os.path.join(root, "flow", coin.lower())
        self.delivery = {}
        with open(os.path.join(root, "delivery", f"{coin.lower()}.csv")) as fh:
            for row in csv.DictReader(fh):
                self.delivery[row["date"]] = float(row["price"])
        self.funding = {}
        fpath = os.path.join(root, "funding", f"{coin}-PERPETUAL.csv")
        if os.path.exists(fpath):
            with open(fpath) as fh:
                for row in csv.DictReader(fh):
                    if row["interest_1h"]:
                        self.funding[int(row["ts"]) // 3_600_000] = float(row["interest_1h"])
        self._chain_day, self._chain = None, []
        self._iv_cache = {}
        self._spot_at = {}            # minute -> perpetual price, to pair option quotes with the spot they were built on
        algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.At(8, 0), self.settle_expiring)
        algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.Every(timedelta(hours=1)), self.charge_funding)

    # ----------------------------------------------------------------- market state
    def now(self):
        """Naive UTC datetime (the algorithm time zone is UTC)."""
        return self.algo.Time

    def now_ms(self):
        return int(self.now().replace(tzinfo=timezone.utc).timestamp() * 1000)

    def spot(self):
        return float(self.algo.Securities[self.u].Price)

    def on_data(self):
        """Call at the top of OnData: remembers the perpetual price at each minute boundary."""
        t = self.algo.Time
        if t.second == 0:
            self._spot_at[t] = self.spot()
            if len(self._spot_at) > 240:
                for k in sorted(self._spot_at)[:-180]:
                    del self._spot_at[k]

    def quote_spot(self, sym):
        """Perpetual price at the end time of the contract's latest quote bar (what the quote was built on)."""
        bar = self.algo.Securities[sym].Cache.GetData[QuoteBar]()
        if bar is not None:
            s = self._spot_at.get(bar.EndTime)
            if s:
                return s
        return self.spot()

    def chain(self):
        """Contracts that exist *now*: listed in today's chain file with first trade before now."""
        day = self.now().strftime("%Y%m%d")
        if day != self._chain_day:
            self._chain_day, self._chain = day, []
            p = os.path.join(self.chain_dir, f"{day}.csv")
            if os.path.exists(p):
                with open(p) as fh:
                    for r in csv.DictReader(fh):
                        self._chain.append((datetime.strptime(r["expiry"], "%Y%m%d"), float(r["strike"]),
                                            r["right"], int(r["first_trade_ms"])))
        now = self.now_ms()
        return [c for c in self._chain if c[3] <= now]

    def expiries(self, min_hours=0.0, max_days=60.0):
        now = self.now()
        out = set()
        for e, _, _, _ in self.chain():
            h = (e + timedelta(hours=8) - now).total_seconds() / 3600.0
            if min_hours <= h <= max_days * 24:
                out.add(e)
        return sorted(out)

    def strikes(self, expiry, right):
        return sorted({k for e, k, r, _ in self.chain() if e == expiry and r == right})

    def symbol(self, expiry, strike, right):
        r = OptionRight.Call if right == "call" else OptionRight.Put
        return Symbol.CreateOption(self.coin, Market.USA, OptionStyle.European, r, strike, expiry)

    def add(self, sym):
        if sym in self.algo.Securities and self.algo.Securities[sym].IsTradable:
            return self.algo.Securities[sym]
        o = self.algo.AddOptionContract(sym, Resolution.Minute)
        o.SetFeeModel(DeribitFeeModel(self))
        o.SetFillModel(self.fill_model)
        o.SetOptionAssignmentModel(NullOptionAssignmentModel())
        o.ExerciseSettlement = SettlementType.Cash
        return o

    def drop_unused(self, keep=()):
        keep = set(keep)
        for sym in [s for s in self.algo.Securities.Keys if s.SecurityType == SecurityType.Option]:
            if sym not in keep and not self.algo.Portfolio[sym].Invested and \
                    not self.algo.Transactions.GetOpenOrders(sym):
                self.algo.RemoveSecurity(sym)

    def quote(self, sym):
        s = self.algo.Securities[sym]
        b, a = float(s.BidPrice), float(s.AskPrice)
        if a <= 0:
            return None
        return b, a, 0.5 * (b + a)

    def tau(self, sym):
        exp = sym.ID.Date + timedelta(hours=8)
        return max((exp - self.now()).total_seconds(), 0.0) / SECONDS_PER_YEAR

    def iv(self, sym):
        """Implied vol of the quote mid against the spot it was built on; last good value if not invertible."""
        q = self.quote(sym)
        v = None
        if q:
            v = implied_vol(q[2], self.quote_spot(sym), float(sym.ID.StrikePrice), self.tau(sym),
                            sym.ID.OptionRight == OptionRight.Call)
        if v is None:
            return self._iv_cache.get(sym)
        self._iv_cache[sym] = v
        return v

    def delta(self, sym, iv=None):
        iv = iv or self.iv(sym)
        if iv is None:
            return None
        return bs_delta(self.spot(), float(sym.ID.StrikePrice), self.tau(sym), iv, sym.ID.OptionRight == OptionRight.Call)

    def option_delta_exposure(self, ivs=None):
        """Net option delta in coins (positions x per-contract delta from each contract's own IV)."""
        total = 0.0
        for sym, h in self.algo.Portfolio.items():
            if h.Invested and sym.SecurityType == SecurityType.Option:
                d = self.delta(sym, (ivs or {}).get(sym))
                if d is not None:
                    total += float(h.Quantity) * d
        return total

    # ----------------------------------------------------------------- sizing
    def coins_for_notional(self, fraction, cap):
        """Contracts (coins) so that one leg's notional is `fraction` of equity, rounded to the lot size."""
        s = self.spot()
        if s <= 0:
            return 0.0
        q = float(self.algo.Portfolio.TotalPortfolioValue) * fraction / s
        q = min(q, cap)
        return math.floor(q / self.opt_lot) * self.opt_lot

    def round_perp(self, q):
        return round(q / self.perp_lot) * self.perp_lot

    # ----------------------------------------------------------------- settlement & funding
    def settle_expiring(self):
        today = self.now().date()
        price = self.delivery.get(today.strftime("%Y-%m-%d"))
        for sym, h in list(self.algo.Portfolio.items()):
            if not (h.Invested and sym.SecurityType == SecurityType.Option and sym.ID.Date.date() == today):
                continue
            for o in self.algo.Transactions.GetOpenOrders(sym):
                self.algo.Transactions.CancelOrder(o.Id)
            if price is None:
                self.algo.Error(f"missing Deribit delivery price for {today}; settling at perpetual price")
                price = self.spot()
            k = float(sym.ID.StrikePrice)
            intrinsic = max(price - k, 0.0) if sym.ID.OptionRight == OptionRight.Call else max(k - price, 0.0)
            self.algo.MarketOrder(sym, -h.Quantity, tag=f"SETTLE@{intrinsic:.6f} delivery={price}")
            self.settled.append((str(today), sym.Value, float(h.Quantity), price, intrinsic))

    def charge_funding(self):
        h = self.algo.Portfolio[self.u]
        if not h.Invested:
            return
        rate = self.funding.get(self.now_ms() // 3_600_000)   # record stamped at the end of the hour
        if rate is None:
            return
        if rate is None:
            return
        amount = -float(h.Quantity) * self.spot() * rate   # longs pay positive funding
        self.algo.Portfolio.CashBook["USD"].AddAmount(amount)
        self.funding_paid -= amount

    def summary(self):
        return {"fees_paid": round(float(self.algo.Portfolio.TotalFees), 2), "funding_paid": round(self.funding_paid, 2),
                "settlements": len(self.settled)}
