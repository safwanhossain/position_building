import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

# Enable LaTeX text rendering globally
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Or other serif fonts

PANDAS_FOLDER_NAME = "real_data"

def convert_bid_ask_data_to_pd(bid_files, ask_files, save_file_name):
    """ Given the bid and ask data files, (1) create a pandas data frame and save it
    to the folder if it doesn't exist already
    """
    save_file_path = os.path.join(PANDAS_FOLDER_NAME, save_file_name)
    if os.path.exists(os.path.join(PANDAS_FOLDER_NAME, save_file_name)):
        data = pd.read_pickle(save_file_path)
        return data

    # read csv file
    data = pd.DataFrame()
    for bid_file, ask_file in zip(bid_files, ask_files):
        bid_data = pd.read_csv(bid_file)
        ask_data = pd.read_csv(ask_file)

        bid_data.rename(columns={"UTC": "timestamp", "Open": "bid_open", "Close": "bid_close", "High": "bid_high", "Low": "bid_low", "Volume": "bid_volume"}, inplace=True)
        ask_data.rename(columns={"UTC": "timestamp", "Open": "ask_open", "Close": "ask_close", "High": "ask_high", "Low": "ask_low", "Volume": "ask_volume"}, inplace=True)

        # merge bid and ask data
        curr_data = pd.merge(bid_data, ask_data, on="timestamp", how='left')
        data = pd.concat([data, curr_data])

    data.to_pickle(save_file_path)
    return data


def plot_volume_price_chart(data, start=None, end=None):
    """
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe containing bid-ask data
    title : str
        Title for the plot
    start : int, optional
        Starting index for the data slice (default: None, which means 0)
    end : int, optional
        Ending index for the data slice (default: None, which means len(data))
    """
    # Slice the data based on start and end indices
    if start is None:
        start = 0
    if end is None:
        end = len(data)

    data_slice = data.iloc[start:end]

    fig, axes = plt.subplots(1, 1, figsize=(12, 8), sharex=True)

    # Top: Price chart with ranges
    # Plot bid range (low to high)
    axes.fill_between(data_slice['timestamp'], data_slice['bid_low'], data_slice['bid_high'],
                          alpha=0.2, color='blue', label='Bid Range (Low-High)')
    axes.plot(data_slice['timestamp'], data_slice['bid_close'], label='Bid Close', color='darkblue', linewidth=1.5)

    # Plot ask range (low to high)
    axes.fill_between(data_slice['timestamp'], data_slice['ask_low'], data_slice['ask_high'],
                          alpha=0.2, color='red', label='Ask Range (Low-High)')
    axes.plot(data_slice['timestamp'], data_slice['ask_close'], label='Ask Close', color='darkred', linewidth=1.5)

    axes.set_ylabel('Price', fontsize=12)
    axes.legend(loc='best')
    axes.set_title('Bid-Ask Prices and Ranges', fontsize=14)
    axes.grid(True, alpha=0.3)

    ax_volume = axes.twinx()
    x_indices = np.arange(len(data_slice))
    width = 0.4
    # Plot excess demand volume on the right y-axis
    excess_demand = data_slice['bid_volume'] - data_slice['ask_volume']
    ax_volume.bar(x_indices - width/2, data_slice['bid_volume'] - data_slice['ask_volume'], width=width,
                label='Excess Demand', color='green', alpha=0.6)
    ax_volume.set_ylabel('Excess Demand Volume', fontsize=12, color='green')
    ax_volume.tick_params(axis='y', labelcolor='green')
    ax_volume.legend(loc='upper right')

    plt.suptitle("Volume and Price Plot", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()
    return fig


def regress_alpha_from_excess_volume(data, use_open_close=True, pred_window=1, plot=True):
    # compute mid-price. There two ways to do this. 
    # (0) We can compute the avg of the bid open and close (same for ask) and then take the average of these.
    # (1) We can compute the avg of the bid high and low (same for ask) and then take the average of these.
    if use_open_close == 0:
        bid_mid_price = (data["bid_open"] + data["bid_close"]) / 2
        ask_mid_price = (data["ask_open"] + data["ask_close"]) / 2
    else:
        bid_mid_price = (data["bid_high"] + data["bid_low"]) / 2
        ask_mid_price = (data["ask_high"] + data["ask_low"]) / 2
    mid_price = (bid_mid_price + ask_mid_price) / 2

    # We want to predict mid_price change from the excess demand: mid_{t+1} = mid_{t} + m (excess_volume)
    mid_price_change = -1 * mid_price.diff(periods=-1)
    mid_price_change = mid_price_change.iloc[0:-1]      # the last value is NaN due to above

    # If we want to smooth out thigns and compute this over a window
    mid_price_change = mid_price_change.rolling(window=pred_window).mean().shift(-pred_window+1)
    mid_price_change = mid_price_change.iloc[0:-pred_window]

    # Our independent variable is the excess demand. The Walrassian model predicts the change in price is due to 
    # the mismatch in supply and demand. excess demand leads to increased prices. 
    excess_demand = data["bid_volume"] - data["ask_volume"]
    excess_demand = excess_demand.iloc[0:-1]            # the element can be used to predict
    excess_demand = excess_demand.rolling(window=pred_window).mean().shift(-pred_window+1)
    excess_demand = excess_demand.iloc[0:-pred_window]

    m = LinearRegression().fit(excess_demand.values.reshape(-1,1), mid_price_change.values)
    predicted_price_change = m.predict(excess_demand.values.reshape(-1,1))
    
    n = len(excess_demand)
    p = 1  # number of predictors (excluding intercept)
    residuals = mid_price_change.values - predicted_price_change
    residual_std_error = np.sqrt(np.sum(residuals**2) / (n - p - 1))

    # Standard error of the coefficient
    X = excess_demand.values.reshape(-1, 1)
    X_with_intercept = np.column_stack([np.ones(n), X])
    var_coef = residual_std_error**2 * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    std_error_slope = np.sqrt(var_coef[1, 1])

    # t-statistic and p-value
    t_stat = m.coef_[0] / std_error_slope
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - p - 1))  # Two-sided p-value

    if plot:
        plt.figure()
        plt.scatter(excess_demand, mid_price_change, alpha=0.4, s=7)
        plt.plot(excess_demand, predicted_price_change, color="red")
        plt.title(f"Mid-price change vs Excess Demand ($p = $ {p_value:.1e}, $a={m.coef_[0]:.2})$")
        plt.xlabel("Excess Demand")
        plt.ylabel("Change in Price")
        plt.show()

    return m.coef_[0], p_value


def regress_beta_from_excess_volume(data, use_open_close=True, pred_window=1, plot=True):
    # One can imagine the execution price as being the mid_price (predicted/determined by alpha) plus half the
    # current ask-bid spread. This is the temporary impact as modulated by beta. So we want to compute the spread
    # as a function of the excess volume, as our model suggests.

    # compute the ask-bid spreak. There two ways to do this. 
    # (0) We can compute the avg of the bid open and close (same for ask) and then take the spread of these.
    # (1) We can compute the avg of the bid high and low (same for ask) and then take the spread of these.
    if use_open_close:
        bid_mid_price = (data["bid_open"] + data["bid_close"]) / 2
        ask_mid_price = (data["ask_open"] + data["ask_close"]) / 2
    else:
        bid_mid_price = (data["bid_high"] + data["bid_low"]) / 2
        ask_mid_price = (data["ask_high"] + data["ask_low"]) / 2
    ask_bid_spread = (ask_mid_price - bid_mid_price)
    ask_bid_spread = ask_bid_spread.iloc[1:]            # can't predict the first item
    ask_bid_spread = ask_bid_spread.rolling(window=pred_window).mean().shift(-pred_window+1)
    ask_bid_spread = ask_bid_spread.iloc[0:-pred_window]

    # We want to predict ask-bid spread from the excess demand: spread = beta*(excess_volume)
    # Our independent variable is the excess demand.
    excess_demand = data["bid_volume"] - data["ask_volume"]
    excess_demand = excess_demand.iloc[0:-1]            # the element can be used to predict
    excess_demand = excess_demand.rolling(window=pred_window).mean().shift(-pred_window+1)
    excess_demand = excess_demand.iloc[0:-pred_window]

    m = LinearRegression().fit(excess_demand.values.reshape(-1,1), ask_bid_spread.values)
    predicted_spread = m.predict(excess_demand.values.reshape(-1,1))
    
    n = len(excess_demand)
    p = 1  # number of predictors (excluding intercept)
    residuals = ask_bid_spread.values - predicted_spread
    residual_std_error = np.sqrt(np.sum(residuals**2) / (n - p - 1))

    # Standard error of the coefficient
    X = excess_demand.values.reshape(-1, 1)
    X_with_intercept = np.column_stack([np.ones(n), X])
    var_coef = residual_std_error**2 * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    std_error_slope = np.sqrt(var_coef[1, 1])

    # t-statistic and p-value
    t_stat = m.coef_[0] / std_error_slope
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - p - 1))  # Two-sided p-value

    if plot:
        plt.figure()
        plt.scatter(excess_demand, ask_bid_spread, alpha=0.4, s=7)
        plt.plot(excess_demand, predicted_spread, color="red")
        plt.title(f"Ask-Bid Spread vs Excess Demand ($p = $ {p_value:.1e}, $b={m.coef_[0]:.2})$")
        plt.xlabel("Excess Demand")
        plt.ylabel("Ask-Bid Spread")
        plt.show()

    return m.coef_[0], p_value


if __name__ == "__main__":
    # Experiment 1: Second level data over two days for EUR-to-USD
    EXPERIMENT = "USD_CAD_Minute"

    if EXPERIMENT == "EUR_USD_Second":
        data_folder = "data_second/EUR_USD"
        data_dir = os.path.join(PANDAS_FOLDER_NAME, data_folder)

        bid_files = ["EUR-USD_Second_2025-10-07_12h-18h_UTC_bid.csv", "EUR-USD_Second_2025-10-08_12h-18h_UTC_bid.csv"]
        bid_files = [os.path.join(data_dir, bid_file) for bid_file in bid_files]
        ask_files = ["EUR-USD_Second_2025-10-07_12h-18h_UTC_ask.csv", "EUR-USD_Second_2025-10-08_12h-18h_UTC_ask.csv"]
        ask_files = [os.path.join(data_dir, ask_file) for ask_file in ask_files]
        data = convert_bid_ask_data_to_pd(bid_files, ask_files, "EUR_USD_second_2_days")
    elif EXPERIMENT == "USD_CAD_Minute":
        data_folder = "data_minute/USD_CAD"
        data_dir = os.path.join(PANDAS_FOLDER_NAME, data_folder)

        bid_files = ["USD-CAD_Minute_2025-10-07_UTC_bid.csv", "USD-CAD_Minute_2025-10-08_UTC_bid.csv"]
        bid_files = [os.path.join(data_dir, bid_file) for bid_file in bid_files]
        ask_files = ["USD-CAD_Minute_2025-10-07_UTC_ask.csv", "USD-CAD_Minute_2025-10-08_UTC_ask.csv"]
        ask_files = [os.path.join(data_dir, ask_file) for ask_file in ask_files]
        data = convert_bid_ask_data_to_pd(bid_files, ask_files, "USD_CAD_minute_2_days")

    for window in range(2, 50):
        alpha, alpha_p_val = regress_alpha_from_excess_volume(data, use_open_close=False, pred_window=window, plot=False)
        beta, beta_p_val = regress_beta_from_excess_volume(data, use_open_close=False, pred_window=window, plot=False)
        print(f"Alpha: {alpha}, alpha_p_val: {alpha_p_val}, Beta: {beta}, beta_p_val: {beta_p_val}")
    
    alpha, p_val = regress_alpha_from_excess_volume(data, use_open_close=False, pred_window=50, plot=True)
