# -*- coding: utf-8 -*-
"""
@author: Aidan Ruvins

Fund Strategies and Performance
Homework 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#1. stock exchange 

ex=pd.read_csv('Python data/ex.csv')
ex = ex.rename(columns={'qno': 'QNO'})


#2.using monthly data get quarterend size and price 
date_format = "%Y%m%d"
size=pd.read_csv('Python data/msf.csv', parse_dates=['DATE'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
size=size[['PERMNO','DATE','PRC','SHROUT','RET','VOL']]

size['YEAR']=size['DATE'].dt.year
size['MONTH']=pd.DatetimeIndex(size['DATE']).month
size['QTR']=pd.DatetimeIndex(size['DATE']).quarter



size['MNO']=(size['YEAR']-1980)*12+size['MONTH']
size['QNO']=(size['YEAR']-1980)*4+size['QTR']



#in crsp, price is recorded as negative when the close price is not trading price but the mean of bid and ask;
size['PRC']=np.abs(size['PRC'])

#market capitalization

size['SIZE']=size['PRC']*size['SHROUT']

 
#delete observations with 0 size    
condition=size['SIZE']<=0
size=size[~condition]

size=size[['PERMNO','MNO','QNO','SIZE','PRC']]


size.sort_values(by=['PERMNO','QNO','MNO'], inplace=True)


#keep the last observation in each quarter---obtain quarter end market capitalization
size1 = size.groupby(['PERMNO','QNO']).last().reset_index()

#%%
#3. momentum, illiquidity, turnover

# obtain illiquidity and turnover from daily stock return file
date_format = "%Y%m%d"
dsf=pd.read_csv('Python data/dsf.csv', parse_dates=['DATE'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
dsf=dsf[['PERMNO','DATE','PRC','SHROUT','RET','VOL']]

#create month index and quarter index
dsf['YEAR']=dsf['DATE'].dt.year
dsf['MONTH']=pd.DatetimeIndex(dsf['DATE']).month
dsf['QTR']=pd.DatetimeIndex(dsf['DATE']).quarter
dsf['PRC']=np.abs(dsf['PRC'])
dsf['MNO']=(dsf['YEAR']-1980)*12+dsf['MONTH']
dsf['QNO']=(dsf['YEAR']-1980)*4+dsf['QTR']
dsf['DVOL']=dsf['VOL']*dsf['PRC']

#delete observations with negative shrout or negative volume 
condition1=dsf['SHROUT']<=0
dsf=dsf[~condition1]
condition2=dsf['DVOL']<=0
dsf=dsf[~condition2]
#missing return is recorded as 'C' in WRDS. Need to fix this.
dsf['RET']=pd.to_numeric(dsf['RET'], errors='coerce')


#daily turnover
dsf['TURND']=dsf['VOL']/dsf['SHROUT']

#amihud illiquidity ratio
dsf['ILQ']=np.abs(dsf['RET'])/dsf['DVOL']

# ------------------------------------------------------
# HW3: dollar daily turnover

dsf['DTD'] = dsf['VOL'] * dsf['PRC'] / dsf['SHROUT']
# ------------------------------------------------------

dsf=dsf[['PERMNO','DATE','QNO','ILQ','TURND','RET','DTD']]


#for each stock, get the average daily turnover and illiquidity ratio for over each quarter
dsf.sort_values(by=['PERMNO','QNO'], inplace=True)

illiq = dsf.groupby(['PERMNO','QNO']).agg(
    mea_ILLIQ=('ILQ', 'mean'), 
    mea_TURND=('TURND', 'mean'), 
    n_ilq=('ILQ', 'count'), 
    n_turn=('TURND', 'count'),
    mea_DTD=('DTD', 'mean'),
    VOLATILITY=('RET', 'std'),
    n_obs=('RET', 'count')
).reset_index()

#delete fund-quarter observations with less than 44 data points 

condition3 = illiq['n_ilq'] >= 44 
condition4= illiq['n_turn'] >= 44
condition5= illiq['n_obs'] >= 44
illiq=illiq[condition3]
illiq=illiq[condition4]
illiq=illiq[condition5]


illiq=illiq[['PERMNO','QNO','mea_ILLIQ','mea_TURND','mea_DTD','VOLATILITY']]
illiq = illiq.rename(columns={'mea_ILLIQ': 'ILLIQ'})
illiq = illiq.rename(columns={'mea_TURND': 'TURND'})

illiq.sort_values(by=['PERMNO','QNO'], inplace=True)

#%%
#convert monthly returns to quarterly

#import data and clean
date_format = "%Y%m%d"
crsp=pd.read_csv('Python data/msf.csv', parse_dates=['DATE'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
crsp['YEAR']=crsp['DATE'].dt.year
crsp['MONTH']=pd.DatetimeIndex(crsp['DATE']).month
crsp['QTR']=pd.DatetimeIndex(crsp['DATE']).quarter
crsp['MNO']=(crsp['YEAR']-1980)*12+crsp['MONTH']
crsp['QNO']=(crsp['YEAR']-1980)*4+crsp['QTR']
#in crsp, price is recorded as negative when the close price is not trading price but the mean of bid and ask;
crsp['PRC']=np.abs(crsp['PRC'])
#missing return is recorded as 'C' in WRDS. Need to fix this.
crsp['RET']=pd.to_numeric(crsp['RET'], errors='coerce')
#rret is used to compute cumulative past returns later
crsp['RRET']=np.log(1+crsp['RET'])

crsp=crsp[['PERMNO','QNO','MNO','RRET','PRC','SHROUT','RET']]

crsp.sort_values(by=['PERMNO','QNO'], inplace=True)

#log quarterly return =sum of monthly log returns
qmout = crsp.groupby(['PERMNO','QNO']).agg(RRET=('RRET', 'sum'), n=('RRET', 'count')).reset_index()
#set sum to be missing if less than 3 observations in a quarter
qmout['RRET'][qmout['n']<3]=np.nan
qmout['QRET']=np.exp(qmout['RRET'])-1
qmout=qmout[['PERMNO','QNO','QRET']]

qmout1=pd.merge(qmout, illiq, on=['PERMNO','QNO'], how='left')
qmout1=pd.merge(qmout1, size1, on=['PERMNO','QNO'], how='left')

qmout1=qmout1[['PERMNO','QNO','QRET',
               'TURND','ILLIQ','mea_DTD',
               'VOLATILITY','SIZE','PRC']]



#%%
#get momentum factor
#momentum -- last 12 month cumulative return

tempmom0=crsp[['PERMNO','MNO','RRET']]
tempmom0.sort_values(by=['PERMNO','MNO'], inplace=True)

#for each stock each month, get its past 11 months log returns

for i in range(1, 12):
    new_df_name = f'tempmom{i}'
    globals()[new_df_name] = tempmom0.copy()  # Create a copy to avoid modifying the original DataFrame
    globals()[new_df_name]['MNO'] = globals()[new_df_name]['MNO'] + i+0.0  # Modify column 'A' in this example
    globals()[new_df_name] = globals()[new_df_name].rename(columns={'RRET': f'RET{i}'})
    globals()[new_df_name]=globals()[new_df_name][['PERMNO','MNO',f'RET{i}']]
    globals()[new_df_name].sort_values(by=['PERMNO','MNO'], inplace=True)
    
  
#merge all dataframes

# Sample DataFrames
df_list = [tempmom0,tempmom1,tempmom2,tempmom3,tempmom4,tempmom5,tempmom6,tempmom7,tempmom8, tempmom9, tempmom10, tempmom11]

# Merge DataFrames in a loop
mom = df_list[0]  # Start with the first DataFrame

for df in df_list[1:]:
    mom = pd.merge(mom, df, on=['PERMNO','MNO'], how='left')

mom['MOM']=mom['RRET']+mom['RET1']+mom['RET2']+mom['RET3']+mom['RET4']+mom['RET5']+mom['RET6']+mom['RET7']+mom['RET8']+mom['RET9']+mom['RET10']+mom['RET11']
mom['PRRET']=np.exp(mom['MOM'])-1.0
#keep the last observation in each quarter
mom['QNO']=np.floor((mom['MNO']-1)/3)+1.0
mom = mom.groupby(['PERMNO','QNO']).last().reset_index()
mom=mom[['PERMNO','QNO','PRRET']]


#combine with other factors
mom.sort_values(by=['PERMNO','QNO'], inplace=True)
qmout1.sort_values(by=['PERMNO','QNO'], inplace=True)
demo_factors = pd.merge(mom, qmout1, on=['PERMNO','QNO'], how='outer')
demo_factors = pd.merge(demo_factors, ex, on=['PERMNO','QNO'], how='left')
demo_factors=demo_factors[['PERMNO','QNO','PRRET',
                           'QRET','TURND','ILLIQ',
                           'ex','SIZE','PRC',
                           'mea_DTD','VOLATILITY']]

demo_factors.to_csv('demo_factors.csv', index=False)



