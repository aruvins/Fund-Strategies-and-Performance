#!/usr/bin/env python
# coding: utf-8

# # Demo: Fully-Connect Neural Network Model for Intraday Price Prediction 
# 

# Fall 2021 | Instructor: Lin Tong

# In this example, we will use intraday data of AMC to predict its stock price in every five-minute interval during a trading day. The main purpose of this demo is to show you the procedure of training a fully-connected neural network.

# ## Data Preparation
# In[18]:


import pandas as pd
df=pd.read_csv('AMC_5min.txt',header=None)
df.columns=["DateTime", "Open", "High", "Low", "Close","Volume"]
df["DateTime"]=pd.to_datetime(df["DateTime"],format="%Y-%m-%d %H:%M:%S")
df["Date"]=df['DateTime'].dt.date   #Extract Date so that we can group data
#Column Minute below represents how many minutes from 9:30am, which will be a feature in the model
df["Minute"]=df['DateTime'].dt.hour * 60 + df['DateTime'].dt.minute -570
df.sort_values(by="DateTime",inplace=True)
df["Return"]=df.groupby(by="Date").Close.pct_change() #Calculate return for each five minute
df.dropna(how="any",inplace=True)
df=df[df.DateTime<="2020-12-31"]
df=df[(df.Minute>=0) & (df.Minute<=390)] #Include only the data during trading time
df.reset_index(drop=True,inplace=True)
df.head()


# Create lagged sequences for ["Open", "High", "Low", "Close","Volume"] and use them as input features.

# In[19]:


num_lag = 6  
for i in range(num_lag):
    columnID=[s+str(i+1) for s in ["Open", "High", "Low", "Close","Volume"]]
    df[columnID]=df.groupby(by="Date")[["Open", "High", "Low", "Close","Volume"]].shift(i+1)

df.dropna(how="any",inplace=True)
df.reset_index(drop=True,inplace=True)
df.head()


# Create the names of features and target.

# In[20]:


featurename=list(df.columns)
featurename.remove("DateTime")
featurename.remove("Date")
featurename.remove("Close")
featurename.remove("Open")
featurename.remove("High")
featurename.remove("Low")
featurename.remove("Volume")
featurename.remove("Return")
targetname="Close"
featurename


# We split data into training and testing sets using **test_start_date** as the boundary. We also standardize the feature variables because they are in very different scales.

# In[21]:


from sklearn import preprocessing
import datetime

test_start_date=datetime.date(2019, 1, 1)
train_x = df[df.Date<test_start_date].loc[:,featurename]
train_y = df[df.Date<test_start_date].loc[:,targetname]
test_x = df[df.Date>=test_start_date].loc[:,featurename]
test_y = df[df.Date>=test_start_date].loc[:,targetname]

min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
test_x = min_max_scaler.transform(test_x)


# Next, we define and compile the network, train the model, and evaluate its performance

# In[22]:


#pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.backend.clear_session()  #Clean the session to reset the model/layer ID

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(32, activation='sigmoid',input_shape=(train_x.shape[1],)),
  tf.keras.layers.Dense(8, activation='sigmoid'),
  tf.keras.layers.Dense(1, activation='linear')
])

model.summary()


# In[23]:


import os
import numpy as np
import random as python_random
#Set up the random seeds to reproduce results. 
import os
os.environ['PYTHONHASHSEED']=str(0)
np.random.seed(2021)
python_random.seed(2021)
tf.random.set_seed(2021)

#Define an optimizer, for example, Adam
opt = keras.optimizers.Adam(learning_rate=0.01)


#Compile the model
model.compile(optimizer=opt,   #Set optimizer='adam' if you want to use default learning rate.
              loss='mean_squared_error',
              metrics=[keras.metrics.MeanSquaredError()])

#Train and record how the performance metrics changes during training. 
history = model.fit(train_x, train_y, 
                    verbose=0,            #1: print training details #0: silent
                    batch_size=10000, 
                    epochs=200, 
                    validation_split=0.2  #20% of the training data to be used as validation data.
                   )

score=model.evaluate(train_x, train_y, verbose=0)  #return metric, which is MSE
print("Training Error:")
print(score[0]**0.5)
score=model.evaluate(test_x, test_y, verbose=0)  #return metric, which is MSE
print("Testing Error:")
print(score[0]**0.5)


# The following code shows how MSE changed with epochs during training. 

# In[24]:


import matplotlib.pyplot as plt
# summarize history for MSE
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# We generated the predicted price for both training and testing periods and save in the original df in a column called "Signal".

# In[25]:


df.loc[df.Date<test_start_date,"Signal"]=model.predict(train_x)
df.loc[df.Date>=test_start_date,"Signal"]=model.predict(test_x)


# In[26]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[27]:


dftemp=df.loc[df.Date==datetime.date(2019, 1, 2)]
plt.plot(dftemp.DateTime, dftemp.Signal)
plt.plot(dftemp.DateTime, dftemp.Close)

plt.legend(["Predicted","True"],loc="upper left")
plt.title("Intraday Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(axis='both')
plt.show()


# We also build a linear model for comparison.

# In[28]:


import numpy as np
alphaList = np.logspace(start=-3,stop=-1,num=10)
alphaList
from sklearn.linear_model import LassoCV
lasso = LassoCV(alphas=alphaList, #Candidates for alpha  
                cv=5,             #Number of folds, decrease it to speed up.
                max_iter=2000,    #Number of iters, increase it to speed up.
                tol=0.1,       #Tolerance, increase it to speed up. 
                n_jobs=-1,        #-1 means using all processors.
                random_state=2021 #random seed, to reproduce resulsklearnts.
               )

lasso.fit(train_x, train_y)
from sklearn.metrics import mean_squared_error
print("Training Error:")
print(mean_squared_error(train_y,lasso.predict(train_x),squared=False))
print("Testing Error:")
print(mean_squared_error(test_y,lasso.predict(test_x),squared=False))


# In[17]:


df.loc[df.Date<test_start_date,"Signal"]=lasso.predict(train_x)
df.loc[df.Date>=test_start_date,"Signal"]=lasso.predict(test_x)

dftemp=df.loc[df.Date==datetime.date(2019, 1, 2)]
plt.plot(dftemp.DateTime, dftemp.Signal, label="Predicted")
plt.plot(dftemp.DateTime, dftemp.Close,  label="True")

plt.legend(["Predicted","True"],loc="upper left")
plt.title("Intraday Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(axis='both')
plt.show()


# ## Tuning Parameter

# * Learning rate/Batch size: Larger learning rate for a  large batch size; Smaller learning rate for a small batch size
# * Epoch: Wathc the plot of model's MSE. No need to increase epoch if MSE stay stablized a few epochs.
# * Number of neurons: Overfit if too many; Underfit if too few. 
