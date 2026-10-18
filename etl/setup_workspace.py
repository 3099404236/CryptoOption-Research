"""
Create a LEAN workspace whose market/contract databases describe Deribit instruments correctly.

The original project ran BTC/ETH as US *equities* and their options as US equity options, which
silently imposed NYSE hours (09:30-16:00 New York, weekdays only), the New York data time zone,
a 100x contract multiplier and whole-share lots. Here we register exact-symbol overrides
(LEAN looks up "<SecurityType>-<market>-<ticker>" before the "[*]" default):

  market hours      Equity-usa-BTC/ETH, Option-usa-BTC/ETH: 24/7, exchange & data time zone UTC
  symbol properties option multiplier 1 (1 Deribit contract = 1 coin), lot 0.1 BTC / 1 ETH
                    underlying (perpetual) multiplier 1, lot 0.001 BTC / 0.01 ETH

Usage: python setup_workspace.py <workspace_dir> [--template <old_workspace_data_dir>]
"""
import argparse
import json
import shutil
from pathlib import Path

LOTS = {"BTC": {"option": 0.1, "equity": 0.001}, "ETH": {"option": 1.0, "equity": 0.01}}
# Effective US policy rate by month start (approximate FRED DFF levels; used only as LEAN's
# risk-free rate for Sharpe/Sortino). 2026 values are an assumption.
RATES = [
    ("2018-01-01", 0.0142), ("2018-07-01", 0.0191), ("2019-01-01", 0.0240), ("2019-08-01", 0.0213),
    ("2019-09-01", 0.0204), ("2019-10-01", 0.0183), ("2019-11-01", 0.0155), ("2020-03-01", 0.0065),
    ("2020-04-01", 0.0005), ("2021-01-01", 0.0008), ("2022-03-01", 0.0020), ("2022-04-01", 0.0033),
    ("2022-05-01", 0.0077), ("2022-06-01", 0.0121), ("2022-07-01", 0.0168), ("2022-08-01", 0.0233),
    ("2022-09-01", 0.0256), ("2022-10-01", 0.0308), ("2022-11-01", 0.0378), ("2022-12-01", 0.0410),
    ("2023-01-01", 0.0433), ("2023-02-01", 0.0457), ("2023-03-01", 0.0465), ("2023-04-01", 0.0483),
    ("2023-05-01", 0.0506), ("2023-06-01", 0.0508), ("2023-07-01", 0.0512), ("2023-08-01", 0.0533),
    ("2024-09-01", 0.0513), ("2024-10-01", 0.0483), ("2024-11-01", 0.0464), ("2024-12-01", 0.0448),
    ("2025-01-01", 0.0433), ("2025-09-01", 0.0422), ("2025-10-01", 0.0409), ("2025-11-01", 0.0388),
    ("2025-12-01", 0.0364), ("2026-01-01", 0.0360),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace")
    ap.add_argument("--template", default="/mnt/quant_storage/quant_workspace/lean_workspace/data")
    a = ap.parse_args()
    ws, tpl = Path(a.workspace), Path(a.template)
    data = ws / "data"
    (data / "market-hours").mkdir(parents=True, exist_ok=True)
    (data / "symbol-properties").mkdir(parents=True, exist_ok=True)

    lean_json = {"data-folder": "data", "engine-image": "quantconnect/lean:latest", "default-language": "python",
                 "job-organization-id": "local", "data-provider": "QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider",
                 "id": "Local"}
    if not (ws / "lean.json").exists():
        (ws / "lean.json").write_text(json.dumps(lean_json, indent=4))

    mh = json.loads((tpl / "market-hours" / "market-hours-database.json").read_text())
    always_open = json.loads(json.dumps(mh["entries"]["Crypto-coinbase-[*]"]))  # 24/7 UTC template
    always_open["holidays"] = []
    for k in ("earlyCloses", "lateOpens", "bankHolidays"):
        always_open.pop(k, None)
    for t in ("BTC", "ETH"):
        for st in ("Equity", "Option"):
            mh["entries"][f"{st}-usa-{t}"] = always_open
    (data / "market-hours" / "market-hours-database.json").write_text(json.dumps(mh, indent=1))

    sp = (tpl / "symbol-properties" / "symbol-properties-database.csv").read_text().splitlines()
    sp = [l for l in sp if not (l.startswith("usa,BTC,") or l.startswith("usa,ETH,"))]
    for t, lots in LOTS.items():
        sp.append(f"usa,{t},option,Deribit {t} option (USD view),USD,1,0.01,{lots['option']}")
        sp.append(f"usa,{t},equity,Deribit {t}-PERPETUAL (USD),USD,1,0.01,{lots['equity']}")
    (data / "symbol-properties" / "symbol-properties-database.csv").write_text("\n".join(sp) + "\n")

    for kind in ("map_files", "factor_files"):
        d = data / "equity" / "usa" / kind
        d.mkdir(parents=True, exist_ok=True)
        for t in ("btc", "eth"):
            row = "{d},%s,1" % t if kind == "map_files" else "{d},1,1"
            (d / f"{t}.csv").write_text(row.format(d="19980102") + "\n" + row.format(d="20501231") + "\n")

    ir = data / "alternative" / "interest-rate" / "usa"
    ir.mkdir(parents=True, exist_ok=True)
    (ir / "interest-rate.csv").write_text("\n".join(f"{d},{r}" for d, r in RATES) + "\n")
    print(f"workspace ready: {ws}")


if __name__ == "__main__":
    main()
