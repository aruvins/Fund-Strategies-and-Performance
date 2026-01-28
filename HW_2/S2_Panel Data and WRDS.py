# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:41:39 2023

@author: Lin Tong

S2_Panel Data and WRDS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Clean data: take absolute value of price, sometimes remove price<$5
date_format = "%Y%m%d"
msf=pd.read_csv('msf.csv', parse_dates=['DATE'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
msf=msf[['PERMNO','DATE','PRC','SHROUT','RET']]
#condition=(msf['DATE'].dt.year<=2000) & (msf['DATE'].dt.year>=1990)
#msf=msf[condition]


msf['ABS_PRC']=np.abs(msf['PRC'])

condition=msf['ABS_PRC']<=5
msf=msf[~condition]
#reset index to a sequential number
msf=msf.reset_index(drop=True)


#Compute the size of the stocks (price*shrout)
msf['SIZE']=msf['ABS_PRC']*msf['SHROUT']


#**Get the year and month for each date

#convert date column to date format


msf['YEAR']=msf['DATE'].dt.year
msf['MONTH']=pd.DatetimeIndex(msf['DATE']).month
msf['QTR']=pd.DatetimeIndex(msf['DATE']).quarter

#**For each stock, calculate its average size over the sample period
msf.sort_values(by='PERMNO', inplace=True)
summary = msf.groupby('PERMNO').agg(mean_size=('SIZE', 'mean'),num_obs=('SIZE', 'count')).reset_index()

summary.to_csv('output.csv', index=False)



#For each stock, each month, get its future 1 month return

#calculate an index for each month
msf['MNO']=(msf['YEAR']-1980)*12+msf['MONTH']

fret=msf[['PERMNO','MNO','RET']]

fret['MNO']=fret['MNO']-1.0


fret.rename(columns={'RET': 'FRET1'}, inplace=True)

fret.sort_values(by=['PERMNO','MNO'],ascending=[1, 1], inplace=True)
msf.sort_values(by=['PERMNO','MNO'],ascending=[1, 1], inplace=True)

msf1=pd.merge(msf, fret, on=['PERMNO','MNO'], how='inner')


#Each month, rank all stocks into 10 portfolios by its current month market capital;
msf1.sort_values(by=['PERMNO','MNO'],ascending=[True, True], inplace=True)
msf1 = msf1.dropna(subset=['SIZE'])# Drop rows where 'SIZE' has missing values

msf1.sort_values(by=['MNO'],ascending=True, inplace=True)

msf1['SIZE_RANK'] = msf1.groupby('MNO')['SIZE'].transform(lambda x: pd.qcut(x, q=10, labels=False))

#Get the equal weighted and principal weighted average returns of each of the 10 portfolios every month
msf1.sort_values(by=['SIZE_RANK','MNO'],ascending=[True, True], inplace=True)
msf1['FRET1'] = pd.to_numeric(msf1['FRET1'], errors='coerce')
msf1 = msf1.dropna(subset=['FRET1'])


#Equal weighted average by month
summary_ewts = msf1.groupby(['SIZE_RANK','MNO'])['FRET1'].agg(['mean']).reset_index()
summary_ewts = summary_ewts.rename(columns={'mean': 'EW_FRET1'})

#Value weighted average by month
def weighted_average(group):
    return np.average(group['FRET1'], weights=group['SIZE'])

# Calculate the weighted average by group
summary_pwts = msf1.groupby(['SIZE_RANK','MNO']).apply(weighted_average).reset_index(name='PW_FRET1')

#merge the two
summary_ts=pd.merge(summary_ewts, summary_pwts, on=['SIZE_RANK','MNO'], how='inner')


#Get the average EW and PW average returns of the 10 portfolios over the entire time period

summary = summary_ts.groupby('SIZE_RANK')[['EW_FRET1', 'PW_FRET1']].agg(['mean', 'std']).reset_index()
# Perform t-test for each group against a null hypothesis of zero 
from scipy.stats import ttest_1samp


# Get unique groups
groups = summary_ts['SIZE_RANK'].unique()

results = []

for group in groups:
    group_values = summary_ts[summary_ts['SIZE_RANK'] == group]['EW_FRET1']
    
    t_statistic, p_value = ttest_1samp(group_values, 0)
    
    results.append({
        'SIZE_RANK': group,
        't_statistic': t_statistic,
        'p_value': p_value
    })

# Create a DataFrame from the results
result_df = pd.DataFrame(results)

# Display the results
print(result_df)


#save dataframe to .csv
result_df.to_csv('ttest.csv', index=False)
summary.to_csv('summary.csv', index=False)

