import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# dataset information
stock_name = 'AMGN'
period = 'Minute'
start_date = '2025-10-01'
# end_date = '2025-10-08'
end_date = None
timezone = 'UTC'
if end_date:
    bid_file_name = f'data/{stock_name}.US-USD_{period}_{start_date}_to_{end_date}_{timezone}_Bid.csv'
    ask_file_name = f'data/{stock_name}.US-USD_{period}_{start_date}_to_{end_date}_{timezone}_Ask.csv'
else:
    bid_file_name = f'data/{stock_name}.US-USD_{period}_{start_date}_{timezone}_Bid.csv'
    ask_file_name = f'data/{stock_name}.US-USD_{period}_{start_date}_{timezone}_Ask.csv'

# read csv file
bid_data = pd.read_csv(bid_file_name)
ask_data = pd.read_csv(ask_file_name)

bid_data.rename(columns={timezone: "timestamp", "Open": "bid_open", "Close": "bid_close", "High": "bid_high", "Low": "bid_low", "Volume": "bid_volume"}, inplace=True)
ask_data.rename(columns={timezone: "timestamp", "Open": "ask_open", "Close": "ask_close", "High": "ask_high", "Low": "ask_low", "Volume": "ask_volume"}, inplace=True)

# merge bid and ask data
data = pd.merge(bid_data, ask_data, on="timestamp", how='left')

# calculate volume = min(ask volume, bid volume)
# data['volume'] = data[['ask_volume','bid_volume']].min(axis=1)

# calculate excess demand volume = bid volume - ask volume
data['volume'] = data['bid_volume'] - data['ask_volume']

# calculate price = (closing ask price + closing bid price) / 2
data['price'] = (data['ask_close'] + data['bid_close']) / 2

# calculate price movement = price(t+1) - price(t)
data['price_change'] = -1 * data['price'].diff(periods=-1)

# calculate spread = closing ask price - closing bid price
data['spread'] = data['ask_close'] - data['bid_close']

# print(data.columns)
# print(data.head())

# plot volume and price movement trends over time side by side
# plt.subplot(2, 1, 1)
# plt.plot(data['timestamp'], data['volume'], label='Volume', color='blue')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(data['timestamp'], data['price_change'], label='Change in Price', color='red')
# plt.xlabel('Timestamp')
# plt.legend()
# plt.show()

# regression analysis on volumes and price movements
volume_arr = data['volume'][:-1].values.reshape(-1, 1)
price_arr = data['price_change'][:-1].values

m = LinearRegression().fit(volume_arr, price_arr)
r_sq = m.score(volume_arr, price_arr)
print(f"R-squared: {r_sq}")

# print the slope and intercept
print(f"Slope: {m.coef_}")
print(f"Intercept: {m.intercept_}")

# get linear regression line
predicted_price = m.predict(volume_arr)

# plot
plt.scatter(volume_arr, price_arr, color='blue', label='Actual Data')
plt.plot(volume_arr, predicted_price, color='red', label='Best Fit Line')
plt.xlabel('Volume')
plt.ylabel('Change in Price')
plt.legend()
plt.show()