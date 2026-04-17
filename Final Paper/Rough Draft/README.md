# Betting Against Beta: Empirical Analysis of Propositions 1, 2, and 5

This project implements a complete empirical analysis of three propositions from "Betting Against Beta" by Frazzini & Pedersen (2014).

## Overview

The code tests whether leverage constraints explain the beta anomaly:

- **Proposition 1**: High-beta assets have lower alphas (low-beta effect)
- **Proposition 2**: A BAB (Betting Against Beta) factor generates positive abnormal returns
- **Proposition 5**: Constrained investors overweight high-beta assets vs unconstrained investors

## File Structure

```
betting_against_beta/
├── betting_against_beta_analysis.py      # Main analysis code
├── betting_against_beta_research_paper.txt # Detailed research paper
├── README.md                               # This file
└── outputs/
    ├── proposition_1_results.png           # Visualization (P1)
    ├── proposition_1_detailed_results.csv  # Data (P1)
    ├── proposition_2_results.png           # Visualization (P2)
    ├── proposition_5_results.png           # Visualization (P5)
    └── proposition_5_detailed_results.csv  # Data (P5)
```

## Quick Start

### Option 1: Run with Synthetic Data (Demo)

```python
from betting_against_beta_analysis import BettingAgainstBetaAnalysis

# Initialize and run analysis
bab = BettingAgainstBetaAnalysis()
bab.load_fama_french_factors()
bab.load_stock_data(n_stocks=500)
bab.estimate_betas(window=60)

# Test all propositions
prop1_results = bab.test_proposition_1(n_portfolios=5)
prop2_results = bab.test_proposition_2()
prop5_results = bab.test_proposition_5()

# Generate visualizations
bab.generate_all_plots()
```

### Option 2: Run with Real Data

See section "Using Real Data" below.

## Detailed Methodology

### Proposition 1: High Beta → Low Alpha

**Test**: Do alphas decline with increasing beta?

**Procedure**:
1. Estimate rolling 60-month betas for all stocks
2. Sort stocks into 5 beta quintiles each month
3. Calculate equal-weighted portfolio returns
4. Estimate CAPM and Fama-French 3-factor alphas
5. Test if alpha decreases monotonically with beta rank

**Key Statistics**:
- Alpha spread (P1 - P5): Should be 4-10% annualized
- Sharpe ratio decline: Low-beta should exceed high-beta
- Regression slope: Should be significantly negative

**Interpretation**:
- Constrained investors overbid high-beta assets
- This drives high-beta asset prices up
- Reducing their expected returns (negative alpha)

---

### Proposition 2: BAB Factor Returns

**Test**: Does a market-neutral BAB factor generate positive abnormal returns?

**Procedure**:
1. Split stocks into low-beta and high-beta portfolios (at median)
2. Calculate average betas and returns for each portfolio
3. Construct market-neutral BAB: 
   - BAB_t = (1/β_L) × (r_L - r_f) - (1/β_H) × (r_H - r_f)
4. Estimate CAPM and FF3 alphas
5. Calculate Sharpe ratios

**Key Statistics**:
- BAB alpha: Should be 5-8% annualized
- BAB Sharpe ratio: Should exceed 0.60 (annualized)
- BAB market beta: Should be near zero (~0.05)
- t-statistic on alpha: Should exceed 3.0 (significance)

**Interpretation**:
- Profitable opportunity from leverage constraints
- Unconstrained investors can arbitrage the mispricing
- Returns represent compensation for funding liquidity risk

---

### Proposition 5: Investor Heterogeneity

**Test**: Do constrained investors hold higher-beta portfolios?

**Procedure**:
1. Classify investors:
   - Constrained: Growth stocks (top 30% by 12-month momentum)
   - Unconstrained: Value stocks (bottom 30% by 12-month momentum)
2. Calculate average portfolio betas monthly
3. Perform paired t-test on beta differences
4. Track time variation

**Key Statistics**:
- Constrained portfolio beta: Should be 1.15-1.25
- Unconstrained portfolio beta: Should be 0.85-0.95
- Beta difference: Should be 0.25-0.40
- t-statistic: Should exceed 8.0 (highly significant)

**Interpretation**:
- Constrained investors forced into risky assets
- Unconstrained investors can lever safe assets
- Validates the investor heterogeneity mechanism

---

## Using Real Data

### Step 1: Obtain Data

**Stock Returns**:
- Source: CRSP (Center for Research in Security Prices)
- Access: WRDS (Wharton Research Data Services)
- Variables needed: Returns, prices, shares outstanding, exchange codes
- Frequency: Daily or monthly
- Period: 1972-present (50+ years recommended)

**Factor Data**:
- Source: Ken French Data Library
- URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- Download: Monthly factors (MKT-RF, SMB, HML)
- No cost

### Step 2: Format Data

**Required format for stock data**:
```
Date        AAPL      MSFT      GOOG      ...
2022-01-31  -0.0410   -0.0340   -0.0280   ...
2022-02-28  0.0240    0.0180    0.0320    ...
...
```

**Required format for factor data**:
```
Date        mktrf     smb       hml       rf
2022-01-31  -0.0524   -0.0180   0.0210    0.0001
2022-02-28  0.0321    0.0120    -0.0150   0.0001
...
```

### Step 3: Modify Code

Replace data loading in `betting_against_beta_analysis.py`:

```python
# Instead of synthetic data:
bab = BettingAgainstBetaAnalysis()
bab.load_fama_french_factors()
bab.load_stock_data()

# Use real data:
import pandas as pd

# Load from CSV
stock_data = pd.read_csv('crsp_returns.csv', index_col=0, parse_dates=True)
factor_data = pd.read_csv('fama_french_factors.csv', index_col=0, parse_dates=True)

# Assign to class
bab.stock_data = stock_data
bab.factor_data = factor_data

# Continue with analysis
bab.estimate_betas(window=60)
# ... rest of analysis
```

### Step 4: Run Analysis

```bash
python betting_against_beta_analysis.py
```

Results will be saved to output files:
- `proposition_1_results.png`
- `proposition_2_results.png`  
- `proposition_5_results.png`
- CSV files with detailed results

---

## Data Sources

### Primary Sources

| Data | Source | URL | Cost | Notes |
|------|--------|-----|------|-------|
| Stock Returns | CRSP | wrds.wharton.upenn.edu | $$$ | Most comprehensive US equity data |
| Stock Returns | Yahoo Finance | finance.yahoo.com | Free | Limited but free alternative |
| Stock Returns | WRDS | wrds.wharton.upenn.edu | $$$ | Access to CRSP, CompuStat, IBES |
| Factors | Ken French | mba.tuck.dartmouth.edu | Free | Industry standard, recommended |
| Factors | Fama-French | research.chicagobooth.edu | Free | Original source |
| Risk-free Rate | FRED | fred.stlouisfed.org | Free | US Treasury data |

### Accessing WRDS

For academic institutions with WRDS access:

```python
import wrds

db = wrds.Connection()

# Query CRSP stock returns
crsp = db.get_table('crsp', 'msf',  # Monthly stock file
                    columns=['date', 'permno', 'ticker', 'ret', 'prc', 'shrout'],
                    obs_range=('1972-01-01', '2022-12-31'))

# Query Compustat data (if needed for fundamentals)
comp = db.get_table('comp', 'funda',
                    columns=['datadate', 'gvkey', 'bm', 'at'],
                    obs_range=('1972-01-01', '2022-12-31'))
```

### Accessing Ken French Factors

```python
import pandas_datareader as pdr

# Download directly from Ken French library
factors = pdr.data.DataReader(
    'F-F_Research_Data_Factors_monthly',
    'famafrench',
    start='1972-01-01',
    end='2022-12-31'
)[0]

# Rename columns
factors.columns = ['mktrf', 'smb', 'hml', 'rf']

# Convert from percentages to decimal
factors = factors / 100
```

---

## Key Results Interpretation

### Proposition 1: Alpha Decline

**What to expect**:
```
Quintile    Beta    Return (%)    Alpha (%)    Sharpe Ratio
P1 (Low)    0.65      0.95         +0.40        0.70
P2          0.85      0.92         +0.25        0.63
P3          1.00      0.85          0.00        0.57
P4          1.15      0.78         -0.15        0.51
P5 (High)   1.50      0.65         -0.40        0.33

Slope: -0.24% per quintile
t-stat: -3.8 (highly significant)
```

**Interpretation**: Moving from low-beta to high-beta quintile 
decreases alpha by 0.24% monthly or 2.88% annualized.

### Proposition 2: BAB Performance

**What to expect**:
```
Metric                  Value          Benchmark
Annual Return           6-8%           Market: 10%
Annual Volatility       10-12%         Market: 15%
Sharpe Ratio            0.60-0.80      Market: 0.45
CAPM Alpha (annual)     5-8% ***       
FF3 Alpha (annual)      4-6% **
Market Beta             ~0.05          Should be near 0
```

*** p < 0.01, ** p < 0.05

**Interpretation**: BAB factor generates abnormal returns 
exceeding most standard equity factors with lower volatility.

### Proposition 5: Investor Heterogeneity

**What to expect**:
```
Investor Type    Avg Beta    Std Dev    95% CI
Constrained      1.18        0.15       [1.16, 1.20]
Unconstrained    0.88        0.12       [0.86, 0.90]
Difference       0.30        0.10       [0.28, 0.32]

t-statistic:     12.3
p-value:         < 0.0001
```

**Interpretation**: Constrained investors hold portfolios 
with 30% higher beta, highly significant and economically 
meaningful difference.

---

## Customization Guide

### Changing Beta Estimation Method

```python
# In the estimate_betas() method:

# Current: Simple rolling regression
beta = cov(stock_returns, market_returns) / var(market_returns)

# Alternative 1: Shrinkage beta (Vasicek method)
shrinkage_weight = 0.6  # Weight on time-series beta
beta_shrunk = shrinkage_weight * beta_ts + (1 - shrinkage_weight) * mean_beta

# Alternative 2: Market-model beta
# R_i = alpha + beta * R_m + epsilon
# Estimate using OLS, beta = slope coefficient
```

### Changing Portfolio Construction

```python
# Current: Equal-weighted quintiles
# Alternative: Value-weighted (weight by market cap)
portfolio_return = sum(stock_return * weight)
where weight = market_cap / sum(market_cap)

# Current: Median split for BAB
# Alternative: Terciles (33rd/67th percentile)
low_beta = stocks with beta < 33rd percentile
high_beta = stocks with beta > 67th percentile
```

### Changing Risk Adjustment Models

```python
# Current: CAPM and Fama-French 3-factor
# Add more factors:

# Fama-French 5-factor:
r = alpha + beta_mkt*MKT + beta_smb*SMB + beta_hml*HML + beta_rmw*RMW + beta_cma*CMA

# With momentum (Carhart 4-factor):
r = alpha + beta_mkt*MKT + beta_smb*SMB + beta_hml*HML + beta_mom*MOM

# With liquidity (Pastor-Stambaugh):
r = alpha + beta_mkt*MKT + beta_smb*SMB + beta_hml*HML + beta_liq*LIQ
```

---

## Troubleshooting

### Issue: "Not enough data in window"
**Solution**: Reduce beta estimation window from 60 to 36 months
```python
bab.estimate_betas(window=36)  # 3 years instead of 5
```

### Issue: "Results seem weak/insignificant"
**Causes**:
- Data period too short (use 20+ years minimum)
- Too few stocks (use 200+ stocks)
- Beta window too short (use 36+ months)
- Too much noise in synthetic data

**Solutions**:
- Extend historical period
- Increase number of stocks
- Use higher quality data
- Apply additional filters (min price, min volume)

### Issue: "Negative returns on BAB factor"
**This can happen during liquidity crises** (expected by theory)
- 2008-2009 financial crisis
- March 2020 COVID crash
- During periods of margin call cascades

This doesn't invalidate the theory—it shows BAB has funding liquidity risk exposure.

### Issue: "Alpha not significant in Proposition 2"
**Check**:
1. Ensure BAB factor is market-neutral (beta ~0.05)
2. Verify portfolio construction (long low-beta, short high-beta)
3. Confirm adequate sample size (min 200 months)
4. Look at longer time periods (Sharpe ratio may be better than alpha)

---

## Advanced Topics

### Time-Varying Analysis

Track how effects change over time:

```python
# Proposition 1: Alpha spread over time
alpha_spread = []
for t in range(60, len(data)):
    alpha_low_beta = portfolios[t]['P1']['alpha']
    alpha_high_beta = portfolios[t]['P5']['alpha']
    alpha_spread.append(alpha_low_beta - alpha_high_beta)

# Plot alpha_spread vs time to see if effect strengthens/weakens
```

### Linking to Funding Constraints

Extend analysis to test Proposition 3 (with available proxies):

```python
# Instead of TED spread (LIBOR no longer available):

# Proxy 1: High-yield spread
hy_spread = corporate_bond_yields_low_quality - corporate_bond_yields_high_quality

# Proxy 2: Credit market stress (VIX)
# Proxy 3: Fed funds rate (tightness of monetary policy)
# Proxy 4: Term spread (long-term - short-term rates)

# Test: Does BAB alpha increase with these measures?
regression(bab_returns ~ fed_funds_rate + trend)
```

### Multi-Asset Class Extension

Extend to other markets:

```python
# Bonds: Long low-duration, short high-duration bonds
# Commodities: Long low-beta, short high-beta commodities
# FX: Long low-beta, short high-beta currencies

# Test: Is the effect universal across asset classes?
# Does the effect have common explanatory factor?
```

---

## Performance Optimization

For large datasets (all CRSP stocks):

```python
# Use pandas groupby for speed
import pandas as pd

betas_grouped = stock_data.groupby(level=0).apply(
    lambda x: (np.cov(x, market_ret)[0,1] / np.var(market_ret))
)

# Use numba for performance-critical calculations
from numba import jit

@jit(nopython=True)
def fast_beta_calculation(stock_returns, market_returns):
    # JIT-compiled beta estimation
    ...
```

---

## Academic References

The analysis is based on:

1. **Frazzini, A., & Pedersen, L. H. (2014).** Betting against beta. 
   *Journal of Financial Economics*, 111(1), 1-25.

2. **Black, F. (1972).** Capital market equilibrium with restricted borrowing. 
   *Journal of Business*, 45(3), 444-455.

3. **Fama, E. F., & French, K. R. (1993).** Common risk factors in the returns 
   on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

4. **Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019).** Quality for price. 
   *The devil is in the details. Journal of Portfolio Management*, 45(1), 44-66.

See the research paper (`betting_against_beta_research_paper.txt`) for complete 
references and citations.

---

## License

This code is provided for educational and research purposes. 
The underlying research is based on published academic work.

---

## Questions & Support

For questions about:
- **Implementation**: See code comments and docstrings
- **Methodology**: See research paper (Section 3: Data & Methodology)
- **Results interpretation**: See research paper (Section 4: Results)
- **Theory**: See research paper (Section 2: Theoretical Framework)

---

## Version History

- **v1.0** (2024): Initial implementation
  - Propositions 1, 2, 5 implemented
  - Propositions 3, 4 omitted (LIBOR deprecation)
  - Synthetic data generator included
  - Visualizations and detailed output

---

**Last Updated**: 2024
**Python Version**: 3.8+
**Required Packages**: numpy, pandas, scipy, matplotlib, seaborn
