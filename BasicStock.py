#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:49:18 2020

@author: Abdullah
"""

#Importing Necessary Packages for data processing
import pandas as pd
import numpy as np

#Importing Necssary Packages for graphing
import matplotlib.pyplot as plt

#Setting the size of the graph
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#For norminlizing Data - ? don't know what normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#For data retrevial 
import requests as req
import json
def getting_Data(link):
    r = req.get(link)
    data = json.loads(r.text)
    return data

#Date
from datetime import datetime, date
def findTimeDifference(time1, time2):
    return time2 - time1
    


#Creating a dataframe with Pandas from retrived data
df = pd.json_normalize(getting_Data(
    'https://www.tadawul.com.sa/Charts/ChartGenerator?chart-type=SQL_MI_MSPV&chart-parameter=tasi&methodType=parsingMethod'))


df['dateTime'] = pd.to_datetime(df['dateTime'])


#Time
x = df.iloc[:, 0].astype(int).values

#IndexPrice
y = df.iloc[:, 1].values




for i in range(0,len(x), 1):
    if len(x) <= i+1:
        i = i - 1
    x[i] = findTimeDifference(x[i].astype(datetime), x[i+1].astype(datetime))
    if x[i] < 0:
        print(i , " is Negative " , x[i])
    

# plt.plot(x,y)

#Creating a training and Testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle = False)

print(x_train.shape[0]) 
print(y_train.shape)


##Implementing LSTM
#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, x_train.shape[0])))
# model.add(LSTM(units=50))
model.add(Dense(1))


model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(x_train, y_train, epochs=1, validation_data=(x_train,y_test))
# closing_price = model.predict(x_test)
# closing_price = scaler.inverse_transform(closing_price)


print("Done")
