"""
Rank the screened option-selling variants on the in-sample years only, then show the out-of-sample
result of the pre-declared choice (and, for honesty, how in-sample rank relates to out-of-sample overall).

Selection rule, fixed before looking at 2024-2026:
  per coin, among variants with realistic costs (perpetual hedges at the 0.05 % taker fee),
  require in-sample t-stat >= 2 and a positive in-sample result in at least 60 % of the in-sample years,
  then take the highest in-sample Sharpe; the runner-up and the best "simple" variant (no filter) are shown too.

Units: weekly P&L per 1-coin unit per leg as % of spot; equity returns assume one unit = 25 % of equity
per leg (as in all V2 runs), weeks without a trade count as zero.
"""
import math
import sys

import numpy as np
import pandas as pd

IS_END = pd.Timestamp("2024-01-01")
UNIT = 0.25
PKL = sys.argv[1] if len(sys.argv) > 1 else "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/screen_volsell.pkl"

df = pd.read_pickle(PKL)
df["t0"] = pd.to_datetime(df.t0)
df["net_taker"] = df.option_pnl - df.option_costs + df.hedge_pnl - df.hedge_fee_taker - df.funding
df["net_maker"] = df.option_pnl - df.option_costs + df.hedge_pnl - df.funding
FILTERS = {"none": None, "har1.00": 1.0, "har1.10": 1.1, "har1.20": 1.2}


def stats(x, weeks_index):
    """x: weekly P&L per unit (% of spot) on the traded weeks; weeks_index: all weeks of the period."""
    s = pd.Series(0.0, index=weeks_index)
    s.loc[x.index.intersection(weeks_index)] = x.loc[x.index.intersection(weeks_index)]
    r = UNIT * s
    n = len(r)
    if n < 20:
        return None
    mu, sd = r.mean(), r.std()
    eq = r.cumsum()
    dd = (eq.cummax() - eq).max()
    years = r.groupby(r.index.year).sum()
    return dict(weeks=n, traded=int((s != 0).sum()), ann_ret=52 * mu, ann_vol=math.sqrt(52) * sd,
                sharpe=(math.sqrt(52) * mu / sd) if sd > 0 else 0.0, t=(mu / (sd / math.sqrt(n))) if sd > 0 else 0.0,
                max_dd=dd, worst_week=r.min(), pos_years=float((years > 0).mean()), n_years=len(years))


rows = []
for (cur, entry), g0 in df.groupby(["cur", "entry"]):
    all_weeks = pd.DatetimeIndex(sorted(g0.t0.unique()))
    is_weeks, oos_weeks = all_weeks[all_weeks < IS_END], all_weeks[all_weeks >= IS_END]
    for (structure, hedge), g in g0.groupby(["structure", "hedge"]):
        g = g.set_index("t0")
        for fname, ratio in FILTERS.items():
            sel = g if ratio is None else g[(g.har > 0) & (g.atm_iv >= ratio * g.har)]
            for fee in ("taker", "maker"):
                x = sel["net_" + fee]
                a, b = stats(x, is_weeks), stats(x, oos_weeks)
                if a is None or b is None:
                    continue
                rows.append(dict(cur=cur, entry=entry, structure=structure, hedge=hedge, filter=fname, fee=fee,
                                 **{"is_" + k: v for k, v in a.items()}, **{"oos_" + k: v for k, v in b.items()}))
R = pd.DataFrame(rows)
R.to_csv(PKL.replace(".pkl", "_variants.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
show = ["entry", "structure", "hedge", "filter", "fee", "is_traded", "is_ann_ret", "is_sharpe", "is_t", "is_max_dd",
        "is_worst_week", "is_pos_years"]
print(f"variants screened: {len(R)} ({R.groupby('cur').size().to_dict()})")
for cur, g in R.groupby("cur"):
    t = g[g.fee == "taker"].sort_values("is_sharpe", ascending=False)
    print(f"\n===== {cur}: top 15 by IN-SAMPLE Sharpe (taker hedges), {len(t)} variants =====")
    print(t[show].head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"in-sample Sharpe distribution: median {t.is_sharpe.median():.2f}, 90th pct {t.is_sharpe.quantile(0.9):.2f}, "
          f"share > 0: {(t.is_sharpe > 0).mean():.0%}")
    ok = t[(t.is_t >= 2.0) & (t.is_pos_years >= 0.6)]
    print(f"variants passing the rule (t >= 2, >= 60% positive years): {len(ok)}")
    chosen = list(ok.head(2).itertuples())
    simple = ok[ok["filter"] == "none"].head(1)
    if len(simple):
        chosen += [r for r in simple.itertuples() if r.Index not in {c.Index for c in chosen}]
    for r in chosen:
        print(f"  CHOSEN {cur} {r.entry}/{r.structure}/{r.hedge}/{r.filter}: IS ann {r.is_ann_ret:+.2%} Sharpe {r.is_sharpe:.2f} "
              f"t {r.is_t:.2f} maxDD {r.is_max_dd:.1%} | OOS ann {r.oos_ann_ret:+.2%} Sharpe {r.oos_sharpe:.2f} "
              f"t {r.oos_t:.2f} maxDD {r.oos_max_dd:.1%} worst week {r.oos_worst_week:+.2%}")
    print(f"  across all {len(t)} variants: corr(IS Sharpe, OOS Sharpe) = {t.is_sharpe.corr(t.oos_sharpe):.2f}; "
          f"OOS Sharpe of the IS top decile: median {t.head(max(1, len(t)//10)).oos_sharpe.median():.2f} "
          f"vs all variants {t.oos_sharpe.median():.2f}")
