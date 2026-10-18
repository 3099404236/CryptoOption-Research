"""
Batch runner for the V2 backtests.

Fixes relative to the original batch scripts:
  * strategy/coin/period are passed as LEAN parameters (no regex editing of main.py)
  * each result is read from the folder that *this* LEAN run reports, never "the newest folder"
  * the log must contain the SUMMARY line of the requested strategy, otherwise the run is marked failed
  * metrics are recomputed from the daily equity curve with one consistent definition:
      CAGR, annualised volatility, Sharpe (excess over the US policy rate), max drawdown,
      Calmar = CAGR / max drawdown (not total return / drawdown), plus LEAN's own PSR

Usage: python run_suite.py <suite.json> <out.json> [--workers N]
suite.json: [{"project": "DeribitWeekly", "strategy": "IRON_CONDOR", "coin": "BTC",
              "start": "2019-01-01", "end": "2026-10-15", "res": "minute"}, ...]
"""
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

WS = Path("/mnt/quant_storage/quant_workspace/lean_workspace_v2")
LEAN = os.environ.get("LEAN_BIN", "/home/czf/miniconda3/bin/lean")
RATES = [(datetime.strptime(d, "%Y-%m-%d"), r) for d, r in (
    l.split(",") for l in (WS / "data/alternative/interest-rate/usa/interest-rate.csv").read_text().split())]


def rf(day):
    r = 0.0
    for d, v in RATES:
        if d <= day:
            r = float(v)
    return r


def equity_series(result):
    vals = result["charts"]["Strategy Equity"]["series"]["Equity"]["values"]
    pts = []
    for v in vals:
        if isinstance(v, list):
            t, x = v[0], v[-1]           # [time, open, high, low, close] candles
        else:
            t, x = v["x"], v["y"]
        pts.append((datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None), float(x)))
    pts.sort()
    daily = {pts[0][0].date() - timedelta(days=1): pts[0][1]}   # starting equity
    for t, x in pts:
        daily[t.date()] = x              # last value of each UTC day
    return sorted(daily.items())


def metrics(series):
    days = [d for d, _ in series]
    eq = [x for _, x in series]
    rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq)) if eq[i - 1] > 0]
    years = max((days[-1] - days[0]).days / 365.25, 1e-9)
    total = eq[-1] / eq[0] - 1
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if eq[-1] > 0 else -1.0
    ex = [r - rf(datetime.combine(days[i + 1], datetime.min.time())) / 365.0 for i, r in enumerate(rets)]
    mu = sum(ex) / len(ex) if ex else 0.0
    sd = math.sqrt(sum((r - mu) ** 2 for r in ex) / max(len(ex) - 1, 1)) if ex else 0.0
    vol = math.sqrt(sum((r - sum(rets) / len(rets)) ** 2 for r in rets) / max(len(rets) - 1, 1)) * math.sqrt(365) if rets else 0.0
    peak, mdd = eq[0], 0.0
    for x in eq:
        peak = max(peak, x)
        mdd = max(mdd, 1 - x / peak)
    avg_rf = sum(rf(datetime.combine(d, datetime.min.time())) for d in days) / len(days)
    return {"start_equity": eq[0], "end_equity": eq[-1], "total_return": total, "years": years, "cagr": cagr,
            "ann_vol": vol, "sharpe_excess": (mu / sd * math.sqrt(365)) if sd > 0 else 0.0,
            "max_drawdown": mdd, "calmar": (cagr / mdd) if mdd > 0 else None, "avg_risk_free": avg_rf,
            "trading_days": len(days)}


def run(job):
    args = [LEAN, "backtest", job["project"]]
    for k in ("strategy", "coin", "start", "end", "res", "cash", "unit_notional", "max_coins", "spread_mult", "last_entry",
              "perp_fee", "structure", "entry", "hedge", "filter"):
        if k in job:
            args += ["--parameter", k, str(job[k])]
    t0 = time.time()
    p = subprocess.run(args, cwd=WS, capture_output=True, text=True)
    out = p.stdout + p.stderr
    m = re.search(r"output (?:is stored )?in\s+'([^']+)'", out.replace("\n", " "))
    rec = {**job, "seconds": round(time.time() - t0, 1), "ok": False}
    if not m:
        rec["error"] = out[-1500:]
        return rec
    folder = WS / m.group(1).strip()
    rec["folder"] = str(folder)
    log = (folder / "log.txt").read_text(errors="ignore")
    rt = re.search(r"Runtime Error: [^\n]*", log)
    if rt:                               # an algorithm exception is a failed run even if LEAN wrote output
        rec["error"] = rt.group(0)
        rec["log_tail"] = log[-1500:]
        return rec
    if f"SUMMARY mode={job['strategy']}" not in log:
        rec["error"] = "summary line for requested strategy not found in this run's log"
        rec["log_tail"] = log[-1500:]
        return rec
    main = [f for f in folder.glob("*.json") if not re.search(r"(order-events|summary|data-monitor)", f.name)][0]
    result = json.loads(main.read_text())
    st = result.get("statistics", {})
    rec["lean"] = {k: st.get(k) for k in ("Net Profit", "Compounding Annual Return", "Drawdown", "Sharpe Ratio",
                                          "Probabilistic Sharpe Ratio", "Total Fees", "Total Orders", "Win Rate")}
    rec["summary_line"] = re.search(r"SUMMARY [^\n]*", log).group(0)
    # LEAN notes about the custom 08:00 expiry and Python shutdown timeouts after the run are not errors
    benign = ("IsOptionContractExpired", "Program.Exit", "ExecuteWithTimeLimit", "Failed to shutdown python")
    rec["errors_in_log"] = sum(1 for l in log.splitlines() if "ERROR::" in l and not any(b in l for b in benign))
    rec["metrics"] = metrics(equity_series(result))
    rec["ok"] = True
    return rec


def main():
    import argparse
    import threading
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    jobs = json.loads(Path(a.suite).read_text())
    out = Path(a.out)
    done = json.loads(out.read_text()) if out.exists() else []
    keyf = lambda r: (r["project"], r["strategy"], r["coin"], r["start"], r["end"], r.get("res", "minute"),
                      str(r.get("spread_mult", 1.0)), str(r.get("perp_fee", "")),
                      r.get("structure", ""), r.get("entry", ""), r.get("hedge", ""), r.get("filter", ""))
    seen = {keyf(r) for r in done if r.get("ok")}
    todo = [j for j in jobs if keyf(j) not in seen]
    lock, launch = threading.Lock(), threading.Lock()

    def work(job):
        with launch:                 # LEAN names output folders by the second: stagger launches
            time.sleep(3)
        rec = run(job)
        with lock:
            nonlocal done
            done = [r for r in done if keyf(r) != keyf(job)] + [rec]
            out.write_text(json.dumps(done, indent=1, default=str))
            m = rec.get("metrics", {})
            var = "/".join(str(job[k]) for k in ("structure", "entry", "hedge", "filter") if k in job)
            print(f"{job['project']:15s} {job['coin']} {job['strategy']:28s} {var} {job['start']}..{job['end']} x{job.get('spread_mult', 1.0)} ok={rec['ok']} "
                  f"ret={m.get('total_return', float('nan')):+.2%} cagr={m.get('cagr', float('nan')):+.2%} "
                  f"mdd={m.get('max_drawdown', float('nan')):.2%} sharpe={m.get('sharpe_excess', float('nan')):.2f} "
                  f"({rec['seconds']}s)", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))


if __name__ == "__main__":
    main()
