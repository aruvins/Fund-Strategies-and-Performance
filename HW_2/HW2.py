# -*- coding: utf-8 -*-
"""
@author: Aidan Ruvins

HW2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('./HW_2/HW2_output', exist_ok=True)

# ---------------------------------------------------------
# 1. Clean data: take absolute value of price; remove price<$5
# ---------------------------------------------------------
date_format = "%Y%m%d"
msf=pd.read_csv('Python data/msf.csv', parse_dates=['DATE'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
msf=msf[['PERMNO','DATE','PRC','SHROUT','RET']]
condition=(msf['DATE'].dt.year<=2015) & (msf['DATE'].dt.year>=2000)
msf=msf[condition]

msf['ABS_PRC']=np.abs(msf['PRC'])

# Remove prices less than or equal to $5
condition=msf['ABS_PRC']<=5
msf=msf[~condition]

#reset index to a sequential number
msf=msf.reset_index(drop=True)

# ---------------------------------------------------------
# 2. Compute the size (in $1000) of the stocks (price*shrout)
# ---------------------------------------------------------
msf['SIZE']=msf['ABS_PRC']*msf['SHROUT']

# ---------------------------------------------------------
# 3. Get the year and month for each date
# ---------------------------------------------------------

#convert date column to date format
msf['YEAR']=msf['DATE'].dt.year
msf['MONTH']=pd.DatetimeIndex(msf['DATE']).month
msf['QTR']=pd.DatetimeIndex(msf['DATE']).quarter

# As shown in S2_Panel Data: For each stock, calculate its average size over the sample period
msf.sort_values(by='PERMNO', inplace=True)
summary = msf.groupby('PERMNO').agg(mean_size=('SIZE', 'mean'),num_obs=('SIZE', 'count')).reset_index()

summary.to_csv('./HW_2/HW2_output/output.csv', index=False)

# ---------------------------------------------------------
# 4. For each stock, get its month t+1 stock return
# ---------------------------------------------------------

#calculate an index for each month
msf['MNO']=(msf['YEAR'] - 1980) * 12 + msf['MONTH']
fret=msf[['PERMNO','MNO','RET']]
fret['MNO']=fret['MNO']-1.0

fret.rename(columns={'RET': 'FRET1'}, inplace=True)

fret.sort_values(by=['PERMNO','MNO'],ascending=[1, 1], inplace=True)
msf.sort_values(by=['PERMNO','MNO'],ascending=[1, 1], inplace=True)

msf1=pd.merge(msf, fret, on=['PERMNO','MNO'], how='inner')

# ---------------------------------------------------------
# 5. Each month, rank all stocks into 10 portfolios by month t return
# ---------------------------------------------------------

msf1['RET'] = pd.to_numeric(msf1['RET'], errors='coerce')
msf1 = msf1.dropna(subset=['RET','SIZE'])# Drop rows where 'RET' or 'SIZE' has missing values

msf1.sort_values(by=['MNO'],ascending=True, inplace=True)
msf1['RET_RANK'] = msf1.groupby('MNO')['RET'].transform(lambda x: pd.qcut(x, q=10, labels=False))

# ----------------------------------------------------------
# 6. Get the equal weighted and market capital weighted average returns at month t+1 for each of the 10 portfolios
# ----------------------------------------------------------
msf1.sort_values(by=['RET_RANK','MNO'],ascending=[True, True], inplace=True)
msf1['FRET1'] = pd.to_numeric(msf1['FRET1'], errors='coerce')
msf1 = msf1.dropna(subset=['FRET1'])

#Equal weighted average by month
summary_ewts = msf1.groupby(['RET_RANK','MNO'])['FRET1'].agg(['mean']).reset_index()
summary_ewts = summary_ewts.rename(columns={'mean': 'EW_FRET1'})

#Value weighted average by month
def weighted_average(group):
    return np.average(group['FRET1'], weights=group['SIZE'])

# Calculate the weighted average by group
summary_vwts = msf1.groupby(['RET_RANK','MNO']).apply(weighted_average).reset_index(name='VW_FRET1')

# Merge the two summaries
summary_ts=pd.merge(summary_ewts, summary_vwts, on=['RET_RANK','MNO'], how='inner')

# ----------------------------------------------------------
# 7. Get the average returns of the 10 portfolios over the entire time period.
# 8. Get the standard deviation of the returns of each portfolio
# ----------------------------------------------------------

summary = summary_ts.groupby('RET_RANK')[['EW_FRET1','VW_FRET1']].agg(
    ['mean','std'] # Calculate mean and standard deviation
).reset_index() 

summary.to_csv('summary.csv', index=False)

# Save the final summary to the HW2_output directory
summary.to_csv('./HW_2/HW2_output/return_summary.csv', index=False)

# ----------------------------------------------------------
# As shown in S2_Panel Data: Perform t-test for each group against a null hypothesis of zero
# ----------------------------------------------------------
from scipy.stats import ttest_1samp


# Get unique groups
groups = summary_ts['RET_RANK'].unique()

results = []

for group in groups:
    group_values = summary_ts[summary_ts['RET_RANK'] == group]['EW_FRET1']
    
    t_statistic, p_value = ttest_1samp(group_values, 0)
    
    results.append({
        'RET_RANK': group,
        't_statistic': t_statistic,
        'p_value': p_value
    })

# Create a DataFrame from the results
result_df = pd.DataFrame(results)

# Display the results
print(result_df)

#save dataframe to .csv
result_df.to_csv('./HW_2/HW2_output/return_ttest.csv', index=False)