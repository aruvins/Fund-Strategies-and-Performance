"""
HW7: Basic Fund Performance Measures
Data files: netreturns.csv, tbill.csv, msp500.csv
"""

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------------------
# Load and merge data
# ------------------------------------------------------------------------------

netreturns = pd.read_csv("Python data/netreturns.csv")
tbill      = pd.read_csv("Python data/tbill.csv")
msp500     = pd.read_csv("Python data/msp500.csv")

# Restrict benchmark data to fund sample period (1975-1994)
def filter_dates(df):
    return df[(df["year"] >= 1975) & (df["year"] <= 1994)].copy()

market = pd.merge(
    filter_dates(tbill)[["year", "month", "tbillretto_t30ret"]],
    filter_dates(msp500)[["year", "month", "vwretd", "ewretd"]],
    on=["year", "month"]
).rename(columns={"tbillretto_t30ret": "rf", "vwretd": "mkt_vw", "ewretd": "mkt_ew"})

data = pd.merge(
    netreturns[["year", "month", "fundindex", "ret"]],
    market,
    on=["year", "month"]
)

data["excess_fund"]   = data["ret"]    - data["rf"]
data["excess_mkt_vw"] = data["mkt_vw"] - data["rf"]
data["excess_mkt_ew"] = data["mkt_ew"] - data["rf"]

# ------------------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------------------

def rank_table(series, ascending=False):
    df = series.to_frame()
    df["Rank"] = series.rank(ascending=ascending, method="min").astype(int)
    return df.sort_values("Rank")

# ------------------------------------------------------------------------------
# Task 1: Arithmetic Average Monthly Return
# ------------------------------------------------------------------------------

arith_avg = data.groupby("fundindex")["ret"].mean().rename("Arith Avg")

print("TASK 1: Arithmetic Average Monthly Return")
print(rank_table(arith_avg).to_string(float_format="{:.6f}".format))

# ------------------------------------------------------------------------------
# Task 2: Geometric Average Monthly Return
# ------------------------------------------------------------------------------

geo_avg = (
    data.groupby("fundindex")["ret"]
    .apply(lambda r: (np.prod(1 + r) ** (1 / len(r))) - 1)
    .rename("Geo Avg")
)

print("\nTASK 2: Geometric Average Monthly Return")
print(rank_table(geo_avg).to_string(float_format="{:.6f}".format))

# How and why is this ranking different from #1?
    # This is different from the arithmetic average because of volatility: 
    # the geometric average is always less than or equal to the arithmetic 
    # average, and the difference grows with volatility. The geometric average 
    # represents the actual compound return over time, while the arithmetic 
    # average is a simple mean that does not account for compounding or volatility.

# ------------------------------------------------------------------------------
# Task 3: Jensen's Alpha — Value-Weighted S&P 500
# ------------------------------------------------------------------------------

def run_jensen(group, mkt_col):
    y = group["excess_fund"].values
    x = group[mkt_col].values
    slope, intercept, r, _, _ = stats.linregress(x, y)
    resid = y - (intercept + slope * x)
    return pd.Series({
        "alpha":     intercept,
        "beta":      slope,
        "r_squared": r ** 2,
        "resid_std": resid.std(ddof=2),
    })

jensen_vw = data.groupby("fundindex").apply(
    run_jensen, mkt_col="excess_mkt_vw", include_groups=False
)

print("\nTASK 3: Jensen's Alpha (VW S&P 500)")
print(rank_table(jensen_vw["alpha"].rename("Jensen Alpha VW")).to_string(float_format="{:.6f}".format))

# How and why is this ranking different from #1?
    # This ranking is different from the arithmetic average because Jensen's alpha measures risk-adjusted performance 
    # relative to the market. A fund with a high arithmetic average return might have a low or negative alpha if it 
    # took on a lot of risk (high beta) to achieve that return, while a fund with a lower arithmetic average could 
    # have a higher alpha if it achieved its returns with less risk. Therefore, the rankings can differ significantly
    # based on how much risk each fund took on relative to the market benchmark.

# ------------------------------------------------------------------------------
# Task 4: Jensen's Alpha — Equal-Weighted S&P 500
# ------------------------------------------------------------------------------

jensen_ew = data.groupby("fundindex").apply(
    run_jensen, mkt_col="excess_mkt_ew", include_groups=False
)

print("\nTASK 4: Jensen's Alpha (EW S&P 500)")
print(rank_table(jensen_ew["alpha"].rename("Jensen Alpha EW")).to_string(float_format="{:.6f}".format))

# How and why is this ranking different from #3?
    # This ranking is different from the value-weighted version because the equal-weighted
    # S&P 500 gives each stock in the index equal weight, while the value-weighted version 
    # gives weight based on market capitalization. This can lead to different risk exposures 
    # and performance characteristics for the funds being evaluated.

# ------------------------------------------------------------------------------
# Task 5: Tracking Error Variance — Value-Weighted S&P 500
# ------------------------------------------------------------------------------

tev = (
    data.groupby("fundindex")
    .apply(lambda g: (g["ret"] - g["mkt_vw"]).var(ddof=1), include_groups=False)
    .rename("TEV")
)

print("\nTASK 5: Tracking Error Variance (VW S&P 500, lower = better tracker)")
print(rank_table(tev, ascending=True).to_string(float_format="{:.8f}".format))

# What does this ranking tell you about the risk-taking behavior of these funds? What doesn’t it tell you (i.e., systematic vs. idiosyncratic risk-taking)?
    # This ranking tells us about the total variability of the fund's 
    # returns relative to the benchmark, which includes both systematic 
    # and idiosyncratic risk. A lower TEV indicates that the fund's 
    # returns are more closely aligned with the benchmark, suggesting 
    # less active risk-taking. However, it does not differentiate between
    # systematic risk (market-related) and idiosyncratic risk (fund-specific).
    # A fund could have a low TEV by taking on less idiosyncratic risk but 
    # still have a high beta (systematic risk), or vice versa. Therefore, 
    # while TEV gives us insight into overall tracking performance, it does
    # not provide a complete picture of the types of risks the fund is taking on.

# ------------------------------------------------------------------------------
# Task 6: Information Ratio (alpha / residual std from Task 3)
# ------------------------------------------------------------------------------

ir = (jensen_vw["alpha"] / jensen_vw["resid_std"]).rename("Info Ratio")

print("\nTASK 6: Information Ratio (Jensen VW)")
print(rank_table(ir).to_string(float_format="{:.6f}".format))

# How and why is this ranking different from #3?
    # This ranking is different from Jensen's alpha alone because the 
    # information ratio takes into account the volatility of the fund's 
    # returns relative to the benchmark (residual standard deviation). 
    # A fund with a high alpha but also high volatility (high residual 
    # std) may have a lower information ratio than a fund with a slightly 
    # lower alpha but much lower volatility. Therefore, the information 
    # ratio provides a risk-adjusted measure of performance, which can 
    # lead to different rankings compared to using alpha alone.

# ------------------------------------------------------------------------------
# Summary: All rankings side-by-side
# ------------------------------------------------------------------------------

summary = pd.DataFrame({
    "Arith Avg":     arith_avg,
    "Arith Rank":    rank_table(arith_avg)["Rank"],
    "Geo Avg":       geo_avg,
    "Geo Rank":      rank_table(geo_avg)["Rank"],
    "Alpha VW":      jensen_vw["alpha"],
    "Alpha VW Rank": rank_table(jensen_vw["alpha"].rename("a"))["Rank"],
    "Alpha EW":      jensen_ew["alpha"],
    "Alpha EW Rank": rank_table(jensen_ew["alpha"].rename("a"))["Rank"],
    "TEV":           tev,
    "TEV Rank":      rank_table(tev, ascending=True)["Rank"],
    "Info Ratio":    ir,
    "IR Rank":       rank_table(ir)["Rank"],
}).sort_values("Arith Rank")

print("\nSUMMARY: All Measures and Rankings")
print(summary.to_string(float_format="{:.5f}".format))