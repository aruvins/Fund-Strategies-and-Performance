# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:31:18 2023

@author: Lin Tong
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#quarterly factors constructed in S3 
demo_factors=pd.read_csv('demo_factors.csv')


#Portfolio restriction: exclude stocks with price less than $5 at time of stock ranking (portfolio formation)
demo_factors=demo_factors[demo_factors['PRC']>=5]
demo_factors.drop_duplicates(inplace=True)

#stock future returns:quarter t+1 returns

fret=demo_factors.copy()
fret['QNO']=fret['QNO']-1.0
fret['FRET1']=fret['QRET']
fret=fret[['PERMNO', 'QNO', 'FRET1']]

#keep common stocks only
common=pd.read_csv('common.csv')
demo_factors=pd.merge(demo_factors, common, on=['PERMNO'], how='inner')

#%%
#1. Form portfolios based on momentum factor*

#Adjust the sign of the variables so that they have positive relation with stock returns.***;
demo_factors['PRRET']=1*demo_factors['PRRET'] #the momentum factor do not need to be change 


#Each quarter, form equal-weighted decile portfolios based on the each sign-adjusted signal
mom_rank=demo_factors.copy()
mom_rank = mom_rank.dropna(subset=['PRRET'])# Drop rows where 'PRRET' has missing values

mom_rank.sort_values(by=['QNO','PRRET'],ascending=[True, True], inplace=True)


mom_rank['MOM_RANK'] = mom_rank.groupby('QNO')['PRRET'].transform(lambda x: pd.qcut(x, q=10, labels=False))


mom_rank.sort_values (by=['QNO','MOM_RANK'], inplace=True)

#merge with quarter t+1 return for each decile
mom_rank=mom_rank[['PERMNO','QNO','MOM_RANK']]
mom_rank = mom_rank.dropna(subset=['MOM_RANK'])#drop observations with missing variable

mom_perf=pd.merge(mom_rank,fret, on=['PERMNO','QNO'], how='inner')
mom_perf.sort_values(by=['PERMNO','QNO'], inplace=True)
#save to csv
mom_perf.to_csv('mom_perf.csv', index=False)

#%%
#Form portfolios using two factors (e.g., size and momentum)


#Adjust the sign of the variables so that they have positive relation with stock returns***;
demo_factors['SIZE']=-1*demo_factors['SIZE'] #the size factor needs to be changed

#Rank stocks by each sign-adjusted signal into percentiles
demo_factors = demo_factors.dropna(subset=['SIZE'])# Drop rows where 'SIZE' has missing values
rank1=demo_factors.copy()
rank1['SIZE_RANK'] = rank1.groupby('QNO')['SIZE'].transform(lambda x: pd.qcut(x, q=100, labels=False))

rank2=rank1.copy()
rank2 = rank2.dropna(subset=['PRRET'])# Drop rows where 'PRRET' has missing values

rank2['MOM_RANK'] = rank2.groupby('QNO')['PRRET'].transform(lambda x: pd.qcut(x, q=100, labels=False))

#Take the average of the above percentile ranks across variables in the same category, i.e., “category combo” 
#Also take the average across the all category combos to form an “overall combo” variable;
rank2['MEAN_RANK']=(rank2['SIZE_RANK']+rank2['MOM_RANK'])/2


#form decile portfolios based on the overall combo***;
rank2 = rank2.dropna(subset=['MEAN_RANK'])# Drop rows where 'Mean_Rank' has missing values

rank2['OVERALL_RANK'] = rank2.groupby('QNO')['MEAN_RANK'].transform(lambda x: pd.qcut(x, q=10, labels=False))


#combine with future 1 quarter return
rank2.sort_values(by=['PERMNO','QNO'], inplace=True)
fret.sort_values(by=['PERMNO','QNO'], inplace=True)
perf_sizemom=pd.merge(rank2, fret, on=['PERMNO','QNO'], how='inner')

#save file
perf_sizemom.to_csv('perf_sizemom.csv', index=False)



