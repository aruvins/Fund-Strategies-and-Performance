# -*- coding: utf-8 -*-
"""
@author: Aidan Ruvins

HW 5
Fund Strategies and Performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

os.makedirs('HW_5/HW5_output', exist_ok=True)

#portfolios sorted by size and momentum in S4;
perf_sizemom=pd.read_csv('./HW_4/HW4_output/perf_sizemom.csv')

#raw returns for TOP BOT and DIF
perf_sizemom.sort_values(by=['OVERALL_RANK','QNO'], inplace=True)
#average fret for each portfolio during each quarter
ts_port =perf_sizemom.groupby(['OVERALL_RANK','QNO']).agg(FRET1=('FRET1', 'mean')).reset_index()

#average by porfolio
op =ts_port.groupby(['OVERALL_RANK']).agg(FRET1=('FRET1', 'mean')).reset_index()


#%%
# ------------------------------------------------------------------------------------------------------------------------------
# Quarterly raw returns of top (TOP) and bottom (BOT) decile portfolios, their return difference (DIF) and t-statistic
# ------------------------------------------------------------------------------------------------------------------------------

#Quarterly returns of top (TOP) decile portfolios
top = ts_port[(ts_port['OVERALL_RANK']==9)]
top['TOP_QRET']=top['FRET1']
top=top[['TOP_QRET','QNO']]

# Quarterly returns of bottom (BOT) decile portfolios
bot=ts_port[(ts_port['OVERALL_RANK']==0)]
bot['BOT_QRET']=bot['FRET1']
bot=bot[['BOT_QRET','QNO']]

# Return difference (DIF) between top and bottom decile portfolios
diff=pd.merge(top,bot, on=['QNO'], how='inner')
diff['TOP_BOT']=diff['TOP_QRET']-diff['BOT_QRET']

# average
diff_stat = diff.agg(MEA_DIFF=('TOP_BOT', 'mean')).reset_index()

# Perform t-test for this average against a null hypothesis of zero 
from scipy.stats import ttest_1samp
t_statistic, p_value = ttest_1samp(diff['TOP_BOT'], 0)

# Print the results
print(f"T-Statistic: {t_statistic}")
print(f"P-Value: {p_value}")



#%%
# ------------------------------------------------------------------------------------------------------------------------------
# FF alpha for the difference
# ------------------------------------------------------------------------------------------------------------------------------

# --------------------------Quarterly fama french factors --------------------------
date_format = "%Y%m%d"
factors_monthly=pd.read_csv('Python data/factors_monthly.csv', parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format=date_format))

# Convert variable names to uppercase
factors_monthly.columns = factors_monthly.columns.str.upper()

# Extract year, month, and quarter from the date
factors_monthly['YEAR']=factors_monthly['DATE'].dt.year
factors_monthly['MONTH']=pd.DatetimeIndex(factors_monthly['DATE']).month
factors_monthly['QTR']=pd.DatetimeIndex(factors_monthly['DATE']).quarter

factors_monthly['MNO']=(factors_monthly['YEAR']-1980)*12+factors_monthly['MONTH']
factors_monthly['QNO']=(factors_monthly['YEAR']-1980)*4+factors_monthly['QTR']
ff=factors_monthly[factors_monthly['YEAR']>=1975]
ff=ff[['MNO','QNO','MKTRF','SMB','HML','UMD','RF']]

ff.sort_values(by=['QNO','MNO'], inplace=True)

#convert monthly factors to quarterly
ff['LOG_MKTRF']=np.log(1+ff['MKTRF'])
ff['LOG_SMB']=np.log(1+ff['SMB'])
ff['LOG_HML']=np.log(1+ff['HML'])
ff['LOG_UMD']=np.log(1+ff['UMD'])
ff['LOG_RF']=np.log(1+ff['RF'])

#sum over each quarter
ff.sort_values(by=['QNO'], inplace=True)
ff_qtr = ff.groupby(['QNO']).agg(
        LOG_MKTRF=('LOG_MKTRF', 'sum'), 
        LOG_SMB=('LOG_SMB', 'sum'), 
        LOG_HML=('LOG_HML', 'sum'), 
        LOG_UMD=('LOG_UMD', 'sum'),
        LOG_RF=('LOG_RF','sum')
).reset_index()

ff_qtr['QMKTRF']=np.exp(ff_qtr['LOG_MKTRF'])-1.0
ff_qtr['QSMB']=np.exp(ff_qtr['LOG_SMB'])-1.0
ff_qtr['QHML']=np.exp(ff_qtr['LOG_HML'])-1.0
ff_qtr['QUMD']=np.exp(ff_qtr['LOG_UMD'])-1.0
ff_qtr['QRF']=np.exp (ff_qtr['LOG_RF'])-1.0

ff_qtr=ff_qtr[['QNO','QMKTRF','QSMB','QHML','QUMD','QRF']]

# merge with portfoilo raw returns
ts_port.sort_values(by=['QNO'],inplace=True)
ff_qtr.sort_values(by=['QNO'],inplace=True)

ts_port=pd.merge(ts_port,ff_qtr, on=['QNO'], how='inner')

ts_port['QRET_RF']=ts_port['FRET1']-ts_port['QRF']

ts_port.sort_values(by=['OVERALL_RANK','QNO'])

# -------------------------- Fama French 4 factor alpha--------------------------

result = ts_port.groupby('OVERALL_RANK').apply(lambda group: sm.OLS(group['QRET_RF'], sm.add_constant(group[['QMKTRF', 'QSMB','QHML','QUMD']])).fit())


#print results
for group, model in result.items():
    print(f"Regression results for Group {group}:\n{model.summary()}\n")

#save result in a dataframe
coefficients_data = {'OVERALL_RANK': [], 'Intercept': [], 'MKT_Coef': [], 'SMB_Coef': [],'HML_Coef':[], 'UMD_Coef':[], 'Intercept_TValue': [],'MSE':[]}

for group, model in result.items():
    coefficients_data['OVERALL_RANK'].append(group)
    coefficients_data['Intercept'].append(model.params['const'])
    coefficients_data['MKT_Coef'].append(model.params['QMKTRF'])
    coefficients_data['SMB_Coef'].append(model.params['QSMB'])
    coefficients_data['HML_Coef'].append(model.params['QHML'])
    coefficients_data['UMD_Coef'].append(model.params['QUMD'])
    coefficients_data['Intercept_TValue'].append(model.tvalues['const'])
    coefficients_data['MSE'].append(model.mse_resid)

ff4_coefficients_df = pd.DataFrame(coefficients_data)

ff4_coefficients_df.to_csv('./HW_5/HW5_output/ff4_coefficients_df.csv', index=False)

# ------------------------------------------------------------------------------------------------------------------------------
# FF4 alphas of TOP, BOT, and DIF, and t-statistic
# ------------------------------------------------------------------------------------------------------------------------------
diff=pd.merge(diff, ff_qtr, on=['QNO'], how='inner')

model = sm.OLS(diff['TOP_BOT'], sm.add_constant(diff[['QMKTRF','QSMB','QHML','QUMD']])).fit()

print(model.summary())



#%%
# ------------------------------------------------------------------------------------------------------------------------------
# CAPM for Annualized information ratios (IR) TOP, BOT, and DIF
# ------------------------------------------------------------------------------------------------------------------------------

# --------------------------CAPM Alpha--------------------------
# Run a regression by group

result = ts_port.groupby('OVERALL_RANK').apply(lambda group: sm.OLS(group['QRET_RF'], sm.add_constant(group[['QMKTRF']])).fit())


#print results
for group, model in result.items():
    print(f"Regression results for Group {group}:\n{model.summary()}\n")

#save result in a dataframe
coefficients_data = {'OVERALL_RANK': [], 'Intercept': [], 'MKT_Coef': [], 'Intercept_TValue': [], 'MSE':[]}

for group, model in result.items():
    coefficients_data['OVERALL_RANK'].append(group)
    coefficients_data['Intercept'].append(model.params['const'])
    coefficients_data['MKT_Coef'].append(model.params['QMKTRF'])
    coefficients_data['Intercept_TValue'].append(model.tvalues['const'])
    coefficients_data['MSE'].append(model.mse_resid)
    
capm_coefficients_df = pd.DataFrame(coefficients_data)

# Print the coefficients DataFrame
print(capm_coefficients_df)

# ------------------------------------------------------------------------------------------------------------------------------
# Annualized information ratios (IR) of TOP, BOT, and DIF, based on CAPM
# ------------------------------------------------------------------------------------------------------------------------------

capm_coefficients_df['IR']=capm_coefficients_df['Intercept']/np.sqrt(capm_coefficients_df['MSE'])

#save to .csv
capm_coefficients_df.to_csv('./HW_5/HW5_output/capm_coefficients_df.csv', index=False)