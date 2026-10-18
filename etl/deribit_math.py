"""Vectorised Black-76 helpers shared by the ETL (numpy/scipy). Prices in USD, F and K in USD, T in years."""
import re

import numpy as np
from scipy.special import ndtr

# plain listed options only; Deribit combo books (e.g. BTC-PS-..., ETH-CDIAG-...) print their own trades
OPTION_RE = re.compile(r"^(BTC|ETH)-\d{1,2}[A-Z]{3}\d{2}-\d+(\.\d+)?-[CP]$")


def is_option(name):
    return bool(OPTION_RE.match(name))

SQRT_2PI = np.sqrt(2.0 * np.pi)
YEAR_SECONDS = 365.0 * 86400.0


def parse_instrument(names):
    """'BTC-29MAR24-65000-P' -> (expiry 08:00 UTC epoch ms, strike, is_call)."""
    months = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
    out = {}
    for n in set(names):
        _, e, k, cp = n.split("-")
        day, mon, yr = int(e[:-5]), months[e[-5:-2]], 2000 + int(e[-2:])
        expiry = np.datetime64(f"{yr:04d}-{mon:02d}-{day:02d}T08:00:00").astype("datetime64[ms]").astype(np.int64)
        out[n] = (int(expiry), float(k), cp == "C")
    return out


def black76(F, K, T, sigma, is_call):
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    live = (T > 0) & (sigma > 0)
    sT = np.where(live, sigma * np.sqrt(np.where(live, T, 1.0)), 1.0)
    d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
    d2 = d1 - sT
    call = F * ndtr(d1) - K * ndtr(d2)
    price = np.where(is_call, call, call - (F - K))
    return np.where(live, np.maximum(price, intrinsic), intrinsic)


def vega(F, K, T, sigma):
    """dPrice/dSigma (per 1.00 of vol)."""
    T = np.maximum(np.asarray(T, dtype=float), 1e-9)
    sT = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
    return F * np.exp(-0.5 * d1 * d1) / SQRT_2PI * np.sqrt(T)


def implied_vol(price, F, K, T, is_call, lo=0.01, hi=6.0, iters=60):
    """Vectorised bisection; NaN where the price is outside the no-arbitrage range."""
    price, F, K, T = (np.asarray(x, dtype=float) for x in (price, F, K, T))
    is_call = np.asarray(is_call, dtype=bool)
    lo_v = np.full(price.shape, lo)
    hi_v = np.full(price.shape, hi)
    p_lo = black76(F, K, T, lo_v, is_call)
    p_hi = black76(F, K, T, hi_v, is_call)
    ok = (price >= p_lo - 1e-12) & (price <= p_hi) & (T > 0)
    for _ in range(iters):
        mid = 0.5 * (lo_v + hi_v)
        above = black76(F, K, T, mid, is_call) > price
        hi_v = np.where(above, mid, hi_v)
        lo_v = np.where(above, lo_v, mid)
    return np.where(ok, 0.5 * (lo_v + hi_v), np.nan)


OTM_EDGES = [-np.inf, -0.03, -0.01, 0.01, 0.03, 0.06, 0.10, 0.20, np.inf]
OTM_NAMES = ["ITM>3%", "ITM1-3%", "ATM", "OTM1-3%", "OTM3-6%", "OTM6-10%", "OTM10-20%", "OTM>20%"]
DTE_EDGES = [0, 1, 3, 8, 21, 10_000]
DTE_NAMES = ["<1d", "1-3d", "3-8d", "8-21d", ">21d"]


def otm_bucket(strike, spot, is_call):
    otm = np.where(is_call, strike / spot - 1.0, 1.0 - strike / spot)
    return np.digitize(otm, OTM_EDGES[1:-1])


def dte_bucket(days):
    return np.digitize(days, DTE_EDGES[1:-1])
