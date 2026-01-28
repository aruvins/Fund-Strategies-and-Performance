"""
@author: Aidan Ruvins

HW1.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Read the excel data (wage1.csv) file into Python
# ---------------------------------------------------------

wage = pd.read_csv('./Python data/wage1.csv')

# Build a local data library to save the Python format data
# Create output directory
os.makedirs('HW1_output', exist_ok=True)

# ---------------------------------------------------------
# Edit data
# ---------------------------------------------------------

# 1. Add new variable: Individual ID
wage['id'] = wage.index

# 2. Create a new dataset new dataset that has female only
female = wage[wage['female'] == 1]

# 3. Create a new dataset that removes married individuals
not_married = wage[wage['married'] == 0]

# 4. Create a new dataset with only wage and another dataset with only education and experience
wage_only = wage[['id', 'wage']]
edu_exp = wage[['id', 'educ', 'exper']]

#  5. Merge the two datasets
wage_only.sort_values(by = 'id', inplace = True)
edu_exp.sort_values(by = 'id', inplace = True)

wage_edu_exp = pd.merge(wage_only, edu_exp, on='id', how='inner')

# ---------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------

# 1. Sort the observations by wage(low to high vs. high to low)
wage.sort_values(by = 'wage', ascending = True, inplace = True) # Low to High
wage.sort_values(by = 'wage', ascending = False, inplace = True) # High to Low

# 2. Rank observations into 5 groups by wage
wage['wage_rank'] = pd.qcut(wage['wage'], q = 5, labels = False)

# 3. Compute the mean, standard deviation of salary (for female and male separately)
wage.sort_values(by='wage_rank', ascending=True, inplace=True)

# Female
wage_female = wage[wage['female'] == 1]
summary_female = wage_female.groupby('wage_rank')['wage'].agg(['mean','std']).reset_index()
print("Female Statistics:")
print(summary_female)

# Male
wage_male = wage[wage['female'] == 0]
summary_male = wage_male.groupby('wage_rank')['wage'].agg(['mean','std']).reset_index()
print("Male Statistics:")
print(summary_male)

#save dataframe to .csv
summary_female.to_csv('./HW1_output/female_wage_statistics.csv', index=False)
summary_male.to_csv('./HW1_output/male_wage_statistics.csv', index=False)

# ---------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------

import statsmodels.api as sm

# Define the dependent and independent variables
Y = wage['wage']
X = sm.add_constant(wage[['educ']])

# Fit the regression model
model = sm.OLS(Y, X).fit()
print(model.summary())

# Extract the coefficients (estimates) and t-values
coefficients = model.params
t_values = model.tvalues
p_values = model.pvalues
residual=model.resid

independent_variable_names=['const','educ']
# Create a DataFrame to store the estimates and t-values
results_df = pd.DataFrame({'Variables': independent_variable_names, 'Coefficients': coefficients, 'T-Values': t_values, 'p-Values': p_values})


#save dataframe to .csv
results_df.to_csv('./HW1_output/wage_regression_results.csv', index=False)

