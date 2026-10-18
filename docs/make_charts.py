"""
Charts for the README, drawn only from the result files in results/ (run from the repository root):
    python docs/make_charts.py
Needs matplotlib, numpy, pandas and a CJK font (Microsoft YaHei / SimHei / Noto Sans CJK).
"""
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
RES, IMG = ROOT / "results", ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, ORANGE, GRAY, RED, GREEN = "#2563eb", "#ea580c", "#94a3b8", "#dc2626", "#059669"

WEEKLY_CN = {"HAR_VRP": "HAR 波动率择时", "BULL_PUT_SPREAD": "牛市看跌价差", "SHORT_STRANGLE": "裸卖宽跨",
             "BEAR_CALL_SPREAD": "熊市看涨价差", "TREND_SKEWED_CONDOR": "趋势偏斜铁鹰", "CALL_RATIO_SPREAD": "1×2 看涨比例价差",
             "CALL_RATIO_SPREAD_HEDGED": "比例价差＋对冲", "BROKEN_WING_BUTTERFLY": "破翅蝶式", "IRON_CONDOR_HEDGED": "铁鹰＋对冲",
             "CALL_BUTTERFLY": "对称看涨蝶式", "CALENDAR_SPREAD": "日历价差", "IRON_CONDOR": "铁鹰", "LONG_STRADDLE": "买入跨式"}
INTRADAY_CN = {"PARITY_CONVERSION_ARB": "转换/逆转套利", "SABR_DELTA_HEDGED_STRANGLE": "SABR 对冲卖宽跨",
               "TREND_BUTTERFLY": "趋势蝶式", "TREND_RATIO_SPREAD_STOP": "趋势比例价差＋止损",
               "SHORT_STRANGLE_STOP_HEDGE": "止损对冲卖宽跨", "GAMMA_SCALPING": "Gamma 剥头皮",
               "POST_SPIKE_PREMIUM_SELL": "异动后卖权利金", "NBP_FLOW_STRADDLE": "净买压跨式", "AS_MARKET_MAKING": "AS 做市"}


def load(name):
    return json.loads((RES / "lean" / f"{name}.json").read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", IMG / name)


# 1. representative strategies: in-sample vs out-of-sample annual return ------------------------------------------
def is_oos():
    cta = pd.read_csv(RES / "analysis" / "cta_results.csv")
    pick = lambda cur, s: cta[(cta.cur == cur) & (cta.bar == "1D") & (cta.strategy == s) & (cta.voltarget)].iloc[0]
    bh = lambda cur: cta[(cta.cur == cur) & (cta.bar == "1D") & (cta.strategy == "BUY_HOLD") & (~cta.voltarget)].iloc[0]
    xs = pd.read_csv(RES / "analysis" / "xsmom_results.csv")
    xm = xs[(xs.kind == "momentum") & (xs.L == 56)].set_index("period").cagr
    vs = {(r["coin"], r["structure"], r["start"][:4]): r["metrics"]["cagr"] for r in load("volsell_scaled")}
    carry = (RES / "analysis" / "funding_carry.txt").read_text(encoding="utf-8")
    cv = [float(x) / 100 for x in re.findall(r"always\s+(?:in|out-of)-sample[^:]*:\s*\+?(-?[\d.]+)%", carry)]
    combo = (RES / "analysis" / "combo.txt").read_text(encoding="utf-8").split("100 % notional")[1]
    cb = [float(x) / 100 for x in re.findall(r"3 opt\+carry\s+CAGR\s+\+?(-?[\d.]+)%", combo)]
    rows = [("BTC 趋势跟踪（28 天动量，日线）", pick("BTC", "TSMOM_28d_longflat").is_cagr, pick("BTC", "TSMOM_28d_longflat").oos_cagr),
            ("ETH 趋势跟踪（112 天动量，日线）", pick("ETH", "TSMOM_112d_longflat").is_cagr, pick("ETH", "TSMOM_112d_longflat").oos_cagr),
            ("多币横截面动量（56 天，多强空弱）", xm["is"], xm["oos"]),
            ("BTC 资金费率套利", cv[0], cv[1]), ("ETH 资金费率套利", cv[2], cv[3]),
            ("卖 ETH 宽跨（波动率过滤，4 倍仓位）", vs[("ETH", "strangle25", "2020")], vs[("ETH", "strangle25", "2024")]),
            ("卖 BTC 看跌（波动率过滤，4 倍仓位）", vs[("BTC", "put30", "2019")], vs[("BTC", "put30", "2024")]),
            ("卖 BTC 跨式＋每日对冲（4 倍仓位）", vs[("BTC", "straddle", "2019")], vs[("BTC", "straddle", "2024")]),
            ("组合：3 个卖期权＋BTC 资金费率套利", cb[0], cb[1]),
            ("对照：BTC 买入持有", bh("BTC").is_cagr, bh("BTC").oos_cagr),
            ("对照：ETH 买入持有", bh("ETH").is_cagr, bh("ETH").oos_cagr)]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    y = np.arange(len(rows))[::-1]
    a = np.array([r[1] for r in rows]) * 100
    b = np.array([r[2] for r in rows]) * 100
    ax.barh(y + 0.19, a, 0.38, color=GRAY, label="2019/20–2023 样本内（用来挑选）")
    ax.barh(y - 0.19, b, 0.38, color=BLUE, label="2024–2026 样本外（挑完后只检验一次）")
    for yy, v in zip(y + 0.19, a):
        ax.text(v + (1 if v >= 0 else -1), yy, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8, color="#475569")
    for yy, v in zip(y - 0.19, b):
        ax.text(v + (1 if v >= 0 else -1), yy, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=BLUE)
    ax.axvline(1.8, color=GREEN, ls=":", lw=1)
    ax.axvline(4.4, color=GREEN, ls="--", lw=1)
    ax.text(4.9, y[0] + 0.55, "国债：样本内约 1.8%（点线）、样本外约 4.4%（虚线）", fontsize=8, color=GREEN)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("年化收益（%），扣除 Deribit 手续费、价差和资金费")
    ax.set_xlim(min(a.min(), b.min()) - 12, max(a.max(), b.max()) + 16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    ax.set_title("代表策略：样本内 vs 样本外年化收益", fontsize=12, loc="left")
    save(fig, "is_vs_oos.png")


# 2. the original weekly option structures (LEAN, full costs) -----------------------------------------------------
def weekly():
    btc = {r["strategy"]: r["metrics"]["cagr"] * 100 for r in load("weekly_btc")}
    eth = {r["strategy"]: r["metrics"]["cagr"] * 100 for r in load("weekly_eth")}
    order = sorted(btc, key=lambda s: btc[s])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(order))
    ax.barh(y + 0.19, [btc[s] for s in order], 0.38, color=ORANGE, label="BTC 2019-01 至 2026-10（7.8 年）")
    ax.barh(y - 0.19, [eth[s] for s in order], 0.38, color="#7c3aed", label="ETH 2020-01 至 2026-10（6.8 年）")
    ax.axvline(0, color="#334155", lw=0.8)
    ax.axvline(2.8, color=GREEN, ls="--", lw=1)
    ax.text(2.95, len(order) - 0.6, "同期国债约 2.8%", fontsize=8, color=GREEN)
    ax.set_yticks(y, [WEEKLY_CN[s] for s in order])
    ax.set_xlabel("年化收益（%）")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("周度期权策略（QuantConnect LEAN，全成本）：26 次回测 24 次亏损\n同期买入持有：BTC 年化 +48.9%，ETH +54.3%", fontsize=12, loc="left")
    save(fig, "weekly_options.png")


# 3. intraday option strategies over the full history ------------------------------------------------------------
def intraday():
    rows = [r for r in load("intraday_full") if r["strategy"] != "BUY_AND_HOLD"]
    btc = {r["strategy"]: r["metrics"]["total_return"] * 100 for r in rows if r["coin"] == "BTC"}
    eth = {r["strategy"]: r["metrics"]["total_return"] * 100 for r in rows if r["coin"] == "ETH"}
    order = sorted(btc, key=lambda s: btc[s])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(order))
    ax.barh(y + 0.19, [btc[s] for s in order], 0.38, color=ORANGE, label="BTC 2019–2026")
    ax.barh(y - 0.19, [eth[s] for s in order], 0.38, color="#7c3aed", label="ETH 2020–2026")
    ax.set_yticks(y, [INTRADAY_CN[s] for s in order])
    ax.set_xlabel("总收益（%）")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("日内期权策略（分钟级，LEAN，全历史）：18 次回测全部亏损", fontsize=12, loc="left")
    save(fig, "intraday_options.png")


# 4. CTA: in-sample Sharpe does not carry over --------------------------------------------------------------------
def cta():
    df = pd.read_csv(RES / "analysis" / "cta_results.csv")
    df = df[df.strategy != "BUY_HOLD"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, cur in zip(axes, ("BTC", "ETH")):
        g = df[df.cur == cur]
        for bar, col in (("1h", GRAY), ("4h", ORANGE), ("1D", BLUE)):
            h = g[g.bar == bar]
            ax.scatter(h.is_sharpe, h.oos_sharpe, s=14, alpha=0.65, color=col, label={"1h": "1 小时线", "4h": "4 小时线", "1D": "日线"}[bar])
        best = g[g.bar == "1D"].sort_values("is_sharpe").iloc[-1]
        ax.scatter([best.is_sharpe], [best.oos_sharpe], s=90, facecolors="none", edgecolors=RED, linewidths=1.5)
        ax.annotate("按样本内选出的\n" + best.strategy.replace("_longflat", "（只做多）"), (best.is_sharpe, best.oos_sharpe),
                    xytext=(-118, 26), textcoords="offset points", fontsize=8, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
        lim = [min(g.is_sharpe.min(), g.oos_sharpe.min()) - 0.1, max(g.is_sharpe.max(), g.oos_sharpe.max()) + 0.1]
        ax.plot(lim, lim, color="#334155", lw=0.8, ls="--")
        ax.axhline(0, color="#334155", lw=0.6)
        ax.set_title(f"{cur}：{len(g)} 个趋势/均值回归变体", fontsize=11, loc="left")
        ax.set_xlabel("样本内超额夏普（2019/20–2023）")
    axes[0].set_ylabel("样本外超额夏普（2024–2026）")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("CTA 策略：样本内越好 ≠ 样本外越好（虚线 = 两者相等）", fontsize=12, x=0.01, ha="left")
    save(fig, "cta_is_vs_oos.png")


# 5. option-selling screen: 1,120 variants ------------------------------------------------------------------------
def screen():
    df = pd.read_csv(RES / "analysis" / "screen_volsell_variants.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for cur, col in (("BTC", ORANGE), ("ETH", "#7c3aed")):
        g = df[df.cur == cur]
        ax.scatter(g.is_sharpe, g.oos_sharpe, s=10, alpha=0.45, color=col, label=f"{cur}（{len(g)} 个变体）")
    fin = [("BTC", "mon", "put30", "none", "har1.10", "BTC 卖看跌", (-70, 22)), ("ETH", "mon", "strangle25", "none", "har1.10", "ETH 宽跨", (-60, -26)),
           ("BTC", "fri", "straddle", "daily", "none", "BTC 跨式＋每日对冲", (-40, 30))]
    for cur, e, s, h, f, lab, off in fin:
        r = df[(df.cur == cur) & (df.entry == e) & (df.structure == s) & (df.hedge == h) & (df["filter"] == f) & (df.fee == "taker")]
        if len(r):
            ax.scatter(r.is_sharpe, r.oos_sharpe, s=110, facecolors="none", edgecolors=RED, linewidths=1.6)
            ax.annotate(lab, (r.is_sharpe.iloc[0], r.oos_sharpe.iloc[0]), xytext=off, textcoords="offset points", fontsize=8, color=RED,
                        arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.axhline(0, color="#334155", lw=0.6)
    ax.axvline(0, color="#334155", lw=0.6)
    lim = [df.is_sharpe.min() - 0.1, df.is_sharpe.max() + 0.1]
    ax.plot(lim, lim, color="#334155", lw=0.8, ls="--")
    ax.set_xlabel("样本内夏普（2019/20–2023）")
    ax.set_ylabel("样本外夏普（2024–2026）")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("卖期权：1,120 个变体的快速筛选（红圈 = 进入模拟盘的 3 个）\n结构 × 对冲方式 × 入场日 × 波动率过滤 × 对冲费率 × 币种", fontsize=12, loc="left")
    save(fig, "volsell_screen.png")


# 6. millisecond lead-lag: the edge decays with latency -----------------------------------------------------------
def latency():
    txt = (RES / "analysis" / "leadlag_ms.txt").read_text(encoding="utf-8")
    cur, curves = None, {}
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"=+ (BTC|ETH):", line)
        if m:
            cur = m.group(1)
        m = re.match(r"Binance >=\s*(\d+) bp within\s+100 ms, 2024-26", line)
        if m and cur and m.group(1) in ("10", "20"):
            for l2 in lines[i + 1:i + 4]:
                if "hold  5s" in l2:
                    pts = re.findall(r"(\d+)ms\s+([+-][\d.]+)", l2)
                    curves[f"{cur}：币安 0.1 秒内动 ≥ {m.group(1)} 基点"] = ([int(a) for a, _ in pts], [float(b) for _, b in pts])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for (lab, (x, y)), col, ls in zip(curves.items(), (ORANGE, ORANGE, "#7c3aed", "#7c3aed"), ("-", "--", "-", "--")):
        ax.plot(x, y, marker="o", ms=4, color=col, ls=ls, label=lab)
    for c, lab in ((10, "标准吃单来回成本 ≈ 10 基点"), (5, "吃单进、挂单出 ≈ 5 基点"), (2, "VIP / 做市费率 ≈ 2 基点")):
        ax.axhline(c, color=GREEN, lw=0.9, ls=":")
        ax.text(1000, c + 0.3, lab, ha="right", fontsize=8, color=GREEN)
    ax.axvspan(100, 200, color="#fde68a", alpha=0.35)
    ax.text(150, -1.2, "个人能做到的\n最低延迟", ha="center", fontsize=8, color="#92400e")
    ax.set_xlabel("看到币安信号后，过多少毫秒才在 Deribit 成交")
    ax.set_ylabel("能吃到的 Deribit 追赶幅度（基点，未扣成本）")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("毫秒级：币安领先 Deribit 的价差，1 秒内基本被吃光（2024–2026，持有 5 秒）", fontsize=12, loc="left")
    save(fig, "ms_latency.png")


if __name__ == "__main__":
    is_oos()
    weekly()
    intraday()
    cta()
    screen()
    latency()
