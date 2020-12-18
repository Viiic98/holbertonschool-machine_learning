#!/usr/bin/env python3
""" Data preprocessing """
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


file_name = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
days = 1460
split = 0.7

df = pd.read_csv('data/' + file_name)
df = df.interpolate()
df = df.drop(['Open', 'High', 'Low',
              'Close', 'Volume_(BTC)', 'Volume_(Currency)'], axis=1)
day = (60 * 24)
df = df.iloc[day * -days:]
day += 60
df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')
df = df.resample('H', on='Timestamp').mean()
# Normalize data
sc = MinMaxScaler(feature_range=(0, 1))
scaled = sc.fit_transform(df)
del df
data = []
for i in range(len(scaled) - 24):
    data.append(scaled[i: i + 25])
data = np.array(data)
row = round(split * data.shape[0])
train = data[:int(row), :]
np.random.shuffle(train)
X_train = train[:, :-1]
y_train = train[:, -1]
X_test = data[int(row):, :-1]
y_test = data[int(row):, -1]
