# region imports
from AlgorithmImports import *
# endregion
"""LEAN mechanics check for the Deribit configuration (synthetic data, see make_mechanics_test.py)."""


class DeribitFeeModel(FeeModel):
    """Deribit: options 0.03% of underlying per coin, capped at 12.5% of the option price;
    settlement (tag SETTLE@) 0.015% of underlying capped at 12.5% of intrinsic; perpetual 0.05% taker."""

    def __init__(self, algo):
        super().__init__()
        self.algo = algo

    def GetOrderFee(self, parameters):
        sec, order = parameters.Security, parameters.Order
        q = abs(order.AbsoluteQuantity)
        if sec.Type == SecurityType.Option:
            s = self.algo.Securities[sec.Symbol.Underlying].Price
            px = sec.Price
            if (order.Tag or "").startswith("SETTLE@"):
                px = float(order.Tag.split("@")[1].split()[0])
                fee = min(0.00015 * s, 0.125 * px) * q
            else:
                fee = min(0.0003 * s, 0.125 * max(px, 0.0)) * q
        else:
            fee = 0.0005 * q * sec.Price
        return OrderFee(CashAmount(fee, "USD"))

    def get_order_fee(self, parameters):
        return self.GetOrderFee(parameters)


class DeribitFillModel(ImmediateFillModel):
    """Normal orders fill at bid/ask through LEAN; orders tagged 'SETTLE@<price>' are the cash
    settlement of an expiring option at Deribit's official delivery price."""

    def __init__(self, algo):
        super().__init__()
        self.algo = algo

    def MarketFill(self, asset, order):
        tag = order.Tag or ""
        if tag.startswith("SETTLE@"):
            fill = OrderEvent(order, self.algo.UtcTime, OrderFee.Zero)
            fill.FillPrice = float(tag.split("@")[1].split()[0])
            fill.FillQuantity = order.Quantity
            fill.Status = OrderStatus.Filled
            return fill
        return super().MarketFill(asset, order)

    def market_fill(self, asset, order):
        return self.MarketFill(asset, order)


class MechanicsTest(QCAlgorithm):
    def Initialize(self):
        self.SetTimeZone(TimeZones.Utc)
        self.SetStartDate(2026, 10, 10)
        self.SetEndDate(2026, 10, 12)
        self.SetCash(1_000_000)
        # Register Deribit contract specs in code (the lean CLI refreshes the database files on disk)
        always_open = SecurityExchangeHours.AlwaysOpen(TimeZones.Utc)
        for st in (SecurityType.Equity, SecurityType.Option):
            self.MarketHoursDatabase.SetEntry(Market.USA, "BTC", st, always_open, TimeZones.Utc)
        self.SymbolPropertiesDatabase.SetEntry(Market.USA, "BTC", SecurityType.Option,
                                               SymbolProperties("Deribit BTC option", "USD", 1, 0.01, 0.1, "BTC"))
        self.SymbolPropertiesDatabase.SetEntry(Market.USA, "BTC", SecurityType.Equity,
                                               SymbolProperties("Deribit BTC-PERPETUAL", "USD", 1, 0.01, 0.001, "BTC"))
        eq = self.AddEquity("BTC", Resolution.Minute)
        eq.SetDataNormalizationMode(DataNormalizationMode.Raw)
        eq.SetFeeModel(DeribitFeeModel(self))
        self.u = eq.Symbol
        self.SetBenchmark(self.u)
        self.legs = {}
        for right, k in ((OptionRight.Call, 80000), (OptionRight.Put, 81000)):
            sym = Symbol.CreateOption("BTC", Market.USA, OptionStyle.European, right, k, datetime(2026, 10, 11))
            o = self.AddOptionContract(sym, Resolution.Minute)
            o.SetFeeModel(DeribitFeeModel(self))
            o.SetFillModel(DeribitFillModel(self))
            o.SetOptionAssignmentModel(NullOptionAssignmentModel())
            o.ExerciseSettlement = SettlementType.Cash
            self.legs[right] = o
        self.slices = {}
        self.done = set()
        import scipy, numpy
        self.Log(f"PY scipy={scipy.__version__} numpy={numpy.__version__}")
        c = self.legs[OptionRight.Call]
        self.Log(f"PROPS option multiplier={c.SymbolProperties.ContractMultiplier} lot={c.SymbolProperties.LotSize} "
                 f"style={c.Style} settle={c.ExerciseSettlement} equity lot={eq.SymbolProperties.LotSize} "
                 f"tz={c.Exchange.TimeZone} hours_sat={eq.Exchange.Hours.IsOpen(datetime(2026,10,10,3,0), False)}")

    def OnData(self, data):
        key = self.Time.strftime("%a %H")
        self.slices[key] = self.slices.get(key, 0) + 1
        if self.Time == datetime(2026, 10, 10, 3, 0):
            self.Log(f"SAT03 slice ok: BTC={self.Securities[self.u].Price}")
        if self.Time == datetime(2026, 10, 10, 12, 0) and "buy" not in self.done:
            self.done.add("buy")
            self.MarketOrder(self.legs[OptionRight.Call].Symbol, 0.5)
            self.MarketOrder(self.legs[OptionRight.Put].Symbol, 0.3)
            self.MarketOrder(self.u, -0.123)
        if self.Time == datetime(2026, 10, 11, 8, 0) and "settle" not in self.done:
            self.done.add("settle")
            c = self.legs[OptionRight.Call]
            self.Log(f"PRE-SETTLE call last price={c.Price} bid={c.BidPrice} ask={c.AskPrice} data_time={c.GetLastData().EndTime if c.GetLastData() else None}")
            self.MarketOrder(c.Symbol, -self.Portfolio[c.Symbol].Quantity, tag="SETTLE@300")

    def OnOrderEvent(self, e):
        if e.Status == OrderStatus.Filled:
            self.Log(f"FILL {e.Symbol.Value} qty={e.FillQuantity} px={e.FillPrice} fee={e.OrderFee.Value.Amount:.4f} msg='{e.Message}'")

    def OnEndOfAlgorithm(self):
        self.Log("SLICES " + ", ".join(f"{k}:{v}" for k, v in sorted(self.slices.items()) if k.endswith(("00", "03", "08", "12", "23"))))
        for s, h in self.Portfolio.items():
            if h.Invested:
                self.Log(f"HOLD {s.Value} qty={h.Quantity} px={h.Price}")
        self.Log(f"END equity={self.Portfolio.TotalPortfolioValue:.2f} cash={self.Portfolio.Cash:.2f} fees={self.Portfolio.TotalFees:.4f}")
