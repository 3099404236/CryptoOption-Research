"""Summarise V2 results next to the original project's numbers (markdown tables).

Usage: python summarize.py results/*.json > results/SUMMARY.md
"""
import json
import re
import sys
from pathlib import Path

# original project (results_7_8y_comparison.json / results_5d_hft_eth_comparison.json on the workstation)
ORIG_WEEKLY_BTC = {   # 2019-01-01 .. 2026-10-15, $20M, LEAN as US equity options
    "BROKEN_WING_BUTTERFLY": (0.3082, 0.0351, 0.096, 0.4535),
    "RATIO_SPREAD_HEDGED": (0.2522, 0.0293, 0.102, 0.4293),
    "RATIO_SPREAD": (0.2118, 0.0250, 0.115, 0.3169),
    "BEAR_CALL_SPREAD": (0.1327, 0.0161, 0.077, 0.3119),
    "NAKED_SHORT_STRANGLE": (0.1687, 0.0202, 0.127, 0.2448),
    "CALENDAR_SPREAD": (0.1164, 0.0142, 0.095, 0.2154),
    "BULL_PUT_SPREAD": (-0.0303, -0.0039, 0.162, -0.066),
    "HAR_RV_ADAPTIVE": (0.0989, 0.0122, 0.105, 0.188),
}
V2_FOR_ORIG = {"BROKEN_WING_BUTTERFLY": "CALL_BUTTERFLY", "RATIO_SPREAD_HEDGED": "CALL_RATIO_SPREAD_HEDGED",
               "RATIO_SPREAD": "CALL_RATIO_SPREAD", "BEAR_CALL_SPREAD": "BEAR_CALL_SPREAD",
               "NAKED_SHORT_STRANGLE": "SHORT_STRANGLE", "CALENDAR_SPREAD": "CALENDAR_SPREAD",
               "BULL_PUT_SPREAD": "BULL_PUT_SPREAD", "HAR_RV_ADAPTIVE": "HAR_VRP"}
ORIG_5D_ETH = {   # 2026-10-11 .. 10-15 (really Mon-Thu 09:30-16:00 UTC), $500k
    "PUT_CALL_PARITY_BOX_ARB": ("PARITY_CONVERSION_ARB", 0.0018),
    "SABR_VANNA_GAMMA_SCALPER": ("SABR_DELTA_HEDGED_STRANGLE", -0.0169),
    "SHORT_STRANGLE_HFT": ("SHORT_STRANGLE_STOP_HEDGE", -0.0067),
    "AS_INVENTORY_MARKET_MAKING": ("AS_MARKET_MAKING", -0.0051),
    "NEGATIVE_GAMMA_SQUEEZE_HARVEST": ("POST_SPIKE_PREMIUM_SELL", -0.0024),
    "OFI_IV_PREDICT_STRADDLE": ("NBP_FLOW_STRADDLE", -0.0023),
    "GAMMA_SCALPING_HFT": ("GAMMA_SCALPING", 0.0067),
    "RATIO_SPREAD_SECOND_STOP": ("TREND_RATIO_SPREAD_STOP", -0.0244),
    "BROKEN_WING_BUTTERFLY_SECOND": ("TREND_BUTTERFLY", 0.0047),
    "BENCHMARK_BUY_HOLD": ("BUY_AND_HOLD", -0.0414),
}


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def load(paths):
    rows = []
    for p in paths:
        for r in json.loads(Path(p).read_text()):
            if r.get("ok"):
                rows.append(r)
    return rows


def table(rows, title):
    print(f"\n### {title}\n")
    print("| strategy | total return | CAGR | max DD | Calmar | Sharpe (excess) | fees | funding | trades/cycles |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["metrics"]["cagr"]):
        m, line = r["metrics"], r["summary_line"]
        fees = line.split("fees=")[1].split()[0] if "fees=" in line else "?"
        fund = line.split("funding=")[1].split()[0] if "funding=" in line else "?"
        mc = re.search(r"cycles'?[=:]\s*(\d+)", line)
        cyc = mc.group(1) if mc else "?"
        calmar = f"{m['calmar']:.2f}" if m.get("calmar") is not None else "n/a"
        print(f"| {r['strategy']} | {pct(m['total_return'])} | {pct(m['cagr'], 2)} | {pct(m['max_drawdown'])} | "
              f"{calmar} | {m['sharpe_excess']:.2f} | ${float(fees):,.0f} | ${float(fund):,.0f} | {cyc} |")


def main():
    rows = load(sys.argv[1:])
    groups = {}
    for r in rows:
        key = (r["project"], r["coin"], r["start"], r["end"], r.get("res", "minute"), str(r.get("spread_mult", 1.0)))
        groups.setdefault(key, []).append(r)
    print("# V2 results\n")
    for key, rs in sorted(groups.items()):
        table(rs, f"{key[0]} {key[1]} {key[2]}..{key[3]} res={key[4]} spread x{key[5]}")
    wk = {r["strategy"]: r for r in rows if r["project"] == "DeribitWeekly" and r["coin"] == "BTC"
          and str(r.get("spread_mult", 1.0)) == "1.0"}
    if wk:
        print("\n### Original 7.8-year claims vs V2 (BTC weekly)\n")
        print("| original strategy | original total / CAGR / DD | V2 counterpart | V2 total / CAGR / DD |")
        print("|---|---|---|---|")
        for o, (tot, cagr, dd, _) in ORIG_WEEKLY_BTC.items():
            v = wk.get(V2_FOR_ORIG[o])
            vm = v["metrics"] if v else None
            print(f"| {o} | {pct(tot)} / {pct(cagr, 2)} / {dd*100:.1f}% | {V2_FOR_ORIG[o]} | "
                  + (f"{pct(vm['total_return'])} / {pct(vm['cagr'], 2)} / {vm['max_drawdown']*100:.1f}% |" if vm else "not run |"))
    hf = {r["strategy"]: r for r in rows if r["project"] == "DeribitIntraday" and r["coin"] == "ETH"
          and r.get("res") == "second"}
    if hf:
        print("\n### Original 5-day ETH claims vs V2 (2026-10-11..10-16, 24/7, settlement included)\n")
        print("| original strategy | original net | V2 counterpart | V2 net | V2 max DD |")
        print("|---|---|---|---|---|")
        for o, (v2, net) in ORIG_5D_ETH.items():
            v = hf.get(v2)
            print(f"| {o} | {pct(net, 2)} | {v2} | " + (f"{pct(v['metrics']['total_return'], 2)} | {v['metrics']['max_drawdown']*100:.2f}% |" if v else "not run | |"))


if __name__ == "__main__":
    main()
