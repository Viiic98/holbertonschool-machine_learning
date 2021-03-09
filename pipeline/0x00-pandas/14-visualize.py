#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.loc[df["Timestamp"] >= 1483228800]
df = df.rename(columns={'Timestamp': 'Date'})
df.Date = pd.to_datetime(df.Date, unit='s')
df = df.drop(columns='Weighted_Price')
df = df.set_index('Date')

# Fill NaN
df.Close.fillna(method='ffill', inplace=True)
df.High.fillna(value=df.Close.shift(1, axis=0), inplace=True)
df.Low.fillna(value=df.Close.shift(1, axis=0), inplace=True)
df.Open.fillna(value=df.Close.shift(1, axis=0), inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# new data frame
new_df = pd.DataFrame()
new_df['High'] = df.High.resample('D').max()
new_df['Low'] = df.Low.resample('D').min()
new_df['Open'] = df.Open.resample('D').min()
new_df['Close'] = df.Close.resample('D').max()
new_df['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
new_df['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()

new_df.plot()
plt.show()
