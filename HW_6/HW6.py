#!/usr/bin/env python
# coding: utf-8

# # Random Stock Selection

# Aidan Ruvins
# Spring 2026 | Instructor: Lin Tong

import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer

# ------------------------------------------------
# Data Preparation
# ------------------------------------------------
df = pd.read_csv("Python data/rpsdata_rfs_cleaned_2000.csv")
df["datadate"]=pd.to_datetime(df["datadate"],format="%Y-%m-%d")
df["DATE"]=pd.to_datetime(df["DATE"],format="%Y-%m-%d")

# Randomly sample stocks
num_stock=100   #The number of stocks.
random.seed(2021)
stocklist =  random.sample(list(df.permno.unique()),num_stock) 
df = df[df.permno.isin(stocklist)].copy()

# Remove unnecessary columns and sort the data
if "datadate" in df.columns:
    del df["datadate"]
if "fyear" in df.columns:
    del df["fyear"]

df.sort_values(by=["permno","DATE"],inplace=True)
df.reset_index(drop=True,inplace=True)

# Data cleaning: Remove columns with more than 20% NaNs and impute the remaining NaNs
imputer = SimpleImputer(strategy='mean')
nanpercent = 0.2
dftemp = df.copy()

# Remove all columns except the ones with less than nanpercent NaNs
dftemp = dftemp[dftemp.columns[dftemp.isna().mean(axis=0)<nanpercent]]

# Define the features and target variable
target = 'RET'
features = [col for col in dftemp.columns if col not in ["permno", "DATE", target]]

# Impute the remaining columns
dftemp.loc[:,dftemp.columns[dftemp.isna().sum(axis=0)>0]] = imputer.fit_transform(dftemp.loc[:,dftemp.columns[dftemp.isna().sum(axis=0)>0]]) 


# ------------------------------------------------
# Dynamic Fully Connected Neural Network
# ------------------------------------------------

def build_model(input_dim):
    random.seed(2021)
    np.random.seed(2021)
    tf.random.set_seed(2021)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='sigmoid'),
        Dense(16, activation='sigmoid'),
        Dense(1, activation='linear') 
    ])

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Prepare rolling window data
dates = sorted(dftemp["DATE"].unique())
train_size = 120
test_size = 12
predictions = []

for i in range(train_size, len(dates), test_size):
    train_dates = dates[:i]
    test_dates = dates[i:i+test_size]

    if not test_dates:
        break

    train_data = dftemp[dftemp["DATE"].isin(train_dates)]
    test_data = dftemp[dftemp["DATE"].isin(test_dates)]

    X_train = train_data[features].values
    y_train = train_data[target].values
    X_test = test_data[features].values

    model = build_model(len(features))
    model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=1000, 
        verbose=0
    )

    pred = model.predict(X_test).flatten()
    output_chunk = test_data[['DATE', 'permno', target]].copy()
    output_chunk['predicted_ret'] = pred
    predictions.append(output_chunk)

# Combine all predictions into a single DataFrame
res = pd.concat(predictions, ignore_index=True)

# ------------------------------------------------
# 3. Equally Weighted Long-Short Portfolio for Each Month.
# ------------------------------------------------
portfolio_returns = []

for date, month_data in res.groupby('DATE'):
    month_data = month_data.sort_values(by='predicted_ret', ascending=False)
    
    # Select the top 10 and bottom 10 stocks based on predicted returns
    top_10 = month_data.head(10)
    bottom_10 = month_data.tail(10)

    # Calculate the average return for the long and short positions
    long_return = top_10[target].mean()
    short_return = bottom_10[target].mean()

    # Monthly Long-Short Portfolio Return
    ls_return = long_return - short_return

    portfolio_returns.append({
        'DATE': date, 
        'portfolio_return': ls_return
    })

portfolio_df = pd.DataFrame(portfolio_returns)

# -------------------- Reporting Metrics -----------------

# 1. Average monthly return and standard deviation of the portfolio returns.
mean_return = portfolio_df['portfolio_return'].mean()
std_return = portfolio_df['portfolio_return'].std()

# 2.Cumulative return of the portfolio return up to Dec 2019.
port_2019 = portfolio_df[portfolio_df['DATE'] <= pd.to_datetime('2019-12-31')]
cumulative_return = (1 + port_2019['portfolio_return']).prod() - 1

print("--- Portfolio Performance Metrics ---")
print(f"Average monthly return: {mean_return:.4f}")
print(f"Standard deviation of monthly returns: {std_return:.4f}")
print(f"Cumulative return up to Dec 2019: {cumulative_return:.4f}")