# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:41:39 2023

@author: Lin Tong

S1_Python Intro
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[]
#import .csv file

ceosal=pd.read_csv('./Python data/ceosal1.csv')

#Edit data (add column, delete obs, keep variables/observations)
#1. add new varible:  observation number

ceosal['obs_no']=ceosal.index

#delete observations
ceosal1=ceosal[ceosal['salary']>1000]

#alternative
condition=ceosal['salary']<=1000
ceosal1=ceosal[~condition]

#format (ceosal['salary'])
#print (ceosal['salary'])


#creat new datasets
finance=ceosal[ceosal['finance']==1]

columns_to_keep=['salary', 'obs_no', 'finance']
finance=finance[columns_to_keep]

salary=ceosal[['obs_no','salary']]
roe=ceosal[['obs_no','roe']]


#merge two dataset

#first sort two datasets by the same order
#When inplace=True, the operation modifies the original object, and no new object is created
salary.sort_values(by='obs_no', inplace=True)
roe.sort_values(by='obs_no', inplace=True)

roe_sal=pd.merge(salary, roe, on='obs_no', how='inner')


#Summary statistics
#Sort the observations by salary (low to high vs. high to low)
ceosal.sort_values(by='salary', ascending=1, inplace=True)

#Rank observations into 10 groups by salary

salary['sal_rank'] = pd.qcut(salary['salary'], q=10, labels=False)


#Compute the mean, standard deviation, P25, median, P75, Max, Min Salary
salary.sort_values(by='sal_rank', ascending=1, inplace=True)
summary_sal = salary.groupby('sal_rank')['salary'].agg(['mean', 'sum', 'count','std']).reset_index()

#save dataframe to .csv
summary_sal.to_csv('output.csv', index=False)

#Linear Regression

#Regress salary on ROE

import statsmodels.api as sm

Y=ceosal['salary']
X=sm.add_constant(ceosal[['roe', 'finance']])

# Fit the OLS regression model
model = sm.OLS(Y, X).fit()


print(model.summary())

# Extract the coefficients (estimates) and t-values
coefficients = model.params
t_values = model.tvalues
p_values = model.pvalues
residual=model.resid

independent_variable_names=['const','roe','finance']
# Create a DataFrame to store the estimates and t-values
results_df = pd.DataFrame({'Variables': independent_variable_names, 'Coefficients': coefficients, 'T-Values': t_values, 'p-Values': p_values})


#save dataframe to .csv
results_df.to_csv('reg.csv', index=False)




