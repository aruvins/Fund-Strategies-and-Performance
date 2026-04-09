"""
HW8: Conditional Performance Evaluation
Data files: netreturns.csv, tbill.csv, fscc_regressors.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import os

"""
1. Compute the conditional-beta Jensen alpha for each mutual fund,
and rank the funds on this performance measure. That is, use a six-
factor model, where the first factor is the normal RMRF variable, and
the remaining five factors are the macro variables times RMRF. This is
the Ferson-Schadt conditional beta version of the Jensen model.
"""

# ------------------------------------------------------------------------------
# Load and merge data
# ------------------------------------------------------------------------------

netreturns = pd.read_csv("Python data/netreturns.csv")
tbill      = pd.read_csv("Python data/tbill.csv")
fscc_regressors = pd.read_csv("Python data/fscc_regressors.csv")

os.makedirs('HW_8/HW8_output', exist_ok=True)


# Restrict regressors data to fund sample period (1975-1994)
def filter_dates(df):
    return df[(df["year"] >= 1975) & (df["year"] <= 1994)].copy()

regressors = filter_dates(fscc_regressors)[["year", "month", "rmrf", "div_yld_rmrf", "term_rmrf", "tbill_rmrf", "qual_rmrf", "djan_rmrf", "rf"]]

data = pd.merge(
    netreturns[["year", "month", "fundindex", "ret"]],
    regressors,
    on=["year", "month"]
)

data["excess_fund"] = data["ret"] - data["rf"]

# ------------------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------------------

def rank_table(series, ascending=False):
    df = series.to_frame()
    df["Rank"] = series.rank(ascending=ascending, method="min").astype(int)
    return df.sort_values("Rank")

# ------------------------------------------------------------------------------
# Task 1: Conditional-Beta Jensen Alpha
# ------------------------------------------------------------------------------

def run_jensen(group):
    y = group["excess_fund"].values
    X = group[["rmrf", "div_yld_rmrf", "term_rmrf", "tbill_rmrf", "qual_rmrf", "djan_rmrf"]].values
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results.params[0]  # Return intercept (alpha)

conditional_alpha = (
    data.groupby("fundindex")
    .apply(run_jensen, include_groups=False)
    .rename("Conditional Alpha")
)

print("TASK 1: Conditional-Beta Jensen Alpha")
print(rank_table(conditional_alpha).to_string(float_format="{:.5f}".format))


rank_table(conditional_alpha).to_csv('HW_8/HW8_output/conditional_alphas_ranked.csv')


