"""Write the job lists for run_suite.py (all parameters fixed in advance)."""
import json
from pathlib import Path

WEEKLY = ["SHORT_STRANGLE", "IRON_CONDOR", "IRON_CONDOR_HEDGED", "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD",
          "CALENDAR_SPREAD", "CALL_RATIO_SPREAD", "CALL_RATIO_SPREAD_HEDGED", "CALL_BUTTERFLY",
          "BROKEN_WING_BUTTERFLY", "LONG_STRADDLE", "HAR_VRP", "TREND_SKEWED_CONDOR"]
INTRADAY = ["GAMMA_SCALPING", "SABR_DELTA_HEDGED_STRANGLE", "SHORT_STRANGLE_STOP_HEDGE", "AS_MARKET_MAKING",
            "NBP_FLOW_STRADDLE", "POST_SPIKE_PREMIUM_SELL", "PARITY_CONVERSION_ARB", "TREND_RATIO_SPREAD_STOP",
            "TREND_BUTTERFLY", "BUY_AND_HOLD"]
END = "2026-10-15"        # same end as the original 7.8-year runs
END_5D = "2026-10-16"     # 5-day window includes the Friday 08:00 settlement


def jobs(project, coin, strategies, start, end, res="minute", **extra):
    return [{"project": project, "strategy": s, "coin": coin, "start": start, "end": end, "res": res, **extra}
            for s in strategies]


suites = {
    "weekly_btc": jobs("DeribitWeekly", "BTC", WEEKLY, "2019-01-01", END),
    "weekly_eth": jobs("DeribitWeekly", "ETH", WEEKLY, "2020-01-01", END),
    "intraday_eth_5d": jobs("DeribitIntraday", "ETH", INTRADAY, "2026-10-11", END_5D, "second",
                            last_entry="2026-10-16 08:00"),
    "intraday_btc_5d": jobs("DeribitIntraday", "BTC", INTRADAY, "2026-10-11", END_5D, "second",
                            last_entry="2026-10-16 08:00"),
    "intraday_eth_1y": jobs("DeribitIntraday", "ETH", INTRADAY, "2025-10-01", END),
    "sensitivity_btc": (jobs("DeribitWeekly", "BTC", ["SHORT_STRANGLE", "IRON_CONDOR", "CALL_BUTTERFLY"],
                             "2019-01-01", END, spread_mult=0.5)
                        + jobs("DeribitWeekly", "BTC", ["SHORT_STRANGLE", "IRON_CONDOR", "CALL_BUTTERFLY"],
                               "2019-01-01", END, spread_mult=2.0)
                        + jobs("DeribitWeekly", "BTC", ["SHORT_STRANGLE", "IRON_CONDOR", "CALL_BUTTERFLY"],
                               "2019-01-01", END, spread_mult=0.0)),
    # research machine: reproduce one workstation result first, then the intraday set over the full history
    "env_check": jobs("DeribitWeekly", "BTC", ["SHORT_STRANGLE"], "2019-01-01", END),
    "intraday_full": (jobs("DeribitIntraday", "BTC", INTRADAY, "2019-01-01", END)
                      + jobs("DeribitIntraday", "ETH", INTRADAY, "2020-01-01", END)),
    # same strategies with the 1-second perpetual (tick trades) over the full history
    "intraday_full_second": (jobs("DeribitIntraday", "BTC", INTRADAY, "2019-01-01", END, "second")
                             + jobs("DeribitIntraday", "ETH", INTRADAY, "2020-01-01", END, "second")),
}
# option-selling finalists picked by analysis/screen_report.py on 2019-2023 only; each is run on the
# in-sample period and, separately, on the untouched 2024-2026 period
FINALISTS = [("BTC", "fri", "put40", "h4", "har1.00"), ("BTC", "mon", "put30", "none", "har1.10"),
             ("BTC", "fri", "put40", "h4", "none"),
             ("ETH", "mon", "strangle25", "none", "har1.10"), ("ETH", "mon", "strangle25", "h4", "har1.10"),
             # hypothesis from the earlier full-sample check (analysis/hedged_costs.py), so not a blind test:
             ("BTC", "fri", "straddle", "daily", "none"), ("ETH", "fri", "straddle", "daily", "none")]
suites["volsell_finalists"] = [
    {"project": "DeribitVolSell", "strategy": "VOL_SELL", "coin": c, "start": s0, "end": e0, "res": "minute",
     "structure": st, "entry": en, "hedge": h, "filter": f}
    for c, en, st, h, f in FINALISTS
    for s0, e0 in ((("2019-01-01" if c == "BTC" else "2020-01-01"), "2023-12-31"), ("2024-01-01", END))]
# the three candidates that made money in both periods, at 4x the size (100 % of equity per leg, same coin caps)
suites["volsell_scaled"] = [dict(j, unit_notional=1.0) for j in suites["volsell_finalists"]
                            if (j["coin"], j["structure"], j["entry"], j["hedge"], j["filter"]) in
                            {("BTC", "straddle", "fri", "daily", "none"), ("BTC", "put30", "mon", "none", "har1.10"),
                             ("ETH", "strangle25", "mon", "none", "har1.10")}]
out = Path(__file__).parent / "suites"
out.mkdir(exist_ok=True)
for name, js in suites.items():
    (out / f"{name}.json").write_text(json.dumps(js, indent=1))
    print(name, len(js))
