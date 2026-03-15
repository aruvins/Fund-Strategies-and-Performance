#!/usr/bin/env python
# coding: utf-8

# # Random Stock Selection

# Fall 2021 | Instructor: Lin Tong

# In this example, we will randomly select 100 stocks from the database "rpsdata_rfs_cleaned_2000.csv" using the random seed 2021. We will do some data preparations for the subsequent procesures. 

# ## Data Preparation

# In[2]:


import pandas as pd
df = pd.read_csv("Python data/rpsdata_rfs_cleaned_2000.csv")
df["datadate"]=pd.to_datetime(df["datadate"],format="%Y-%m-%d")
df["DATE"]=pd.to_datetime(df["DATE"],format="%Y-%m-%d")
df.head()


# Randomly sample stocks. 

# In[530]:


import random
num_stock=100   #The number of stocks.
random.seed(2021)
stocklist =  random.sample(list(df.permno.unique()),num_stock) 
df = df[df.permno.isin(stocklist)].copy()
del df["datadate"]
del df["fyear"]
df.sort_values(by=["permno","DATE"],inplace=True)
df.reset_index(drop=True,inplace=True)


# When the data table contains many NaNs, we must be very careful in dropping NaNs in order to keep as many rows and columns as possible.
# 
# 1. Remove all columns except the ones with less than 20% NaNs
# 2. Apply imputation to the remaining columns.
# 
# We can change 20% to other threshold to optimize our performance. The following code applies this procedure to a copy of **df**.

# In[ ]:

# from scikit-learn import SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
nanpercent = 0.2
dftemp = df.copy()
#Remove all columns except the ones with less than nanpercent NaNs
dftemp = dftemp[dftemp.columns[dftemp.isna().mean(axis=0)<nanpercent]]  
#Impute the remaining columns. 
dftemp.loc[:,dftemp.columns[dftemp.isna().sum(axis=0)>0]] = imputer.fit_transform(dftemp.loc[:,dftemp.columns[dftemp.isna().sum(axis=0)>0]]) 
print("Before")
print(df.shape)
print("After")
print(dftemp.shape)


# %%

