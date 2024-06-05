import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from datetime import timedelta

# Machine learning tools
from sklearn.linear_model import LinearRegression, Ridge, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import mutual_info_regression

# Deep Learning tools
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

# Additional tools
from scipy import signal
from scipy.optimize import minimize
from itertools import combinations, product

# External data and APIs
import yfinance as yf
from dune_client.client import DuneClient
import requests
import streamlit as st

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class VaultSimulator_baseline:
    def __init__(self, data, initial_data, features, targets, temporals, start_date, alpha=300):
        self.data = data[data.index <= pd.to_datetime(start_date).tz_localize('UTC')]
        self.features = features
        self.targets = targets
        self.alpha = alpha
        self.model = None
        self.temporals = temporals
        self.results = pd.DataFrame()
        self.initial_data = initial_data
        self.start_date = start_date

    def train_model(self):
        X = self.initial_data[self.features]
        y = self.initial_data[self.targets]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
        self.model = MultiOutputRegressor(Ridge(alpha=self.alpha))
        self.model.fit(X_train, y_train)
        print("Model trained.")

    def run_simulation(self, actions=None):
        start_date = pd.to_datetime(self.start_date).tz_localize('UTC')
        periods = 1  # Forecast period in days
        cycles = 24  # Use number of actions or default to 24 cycles

        for cycle in range(cycles):
            start = start_date + timedelta(days=periods * cycle)
            if start <= self.data.index.max():
                X_test = self.data.loc[[self.data.index.max()], self.features]
            else:
                X_test = self.data.tail(1)[self.features]

            # Apply actions to the data before prediction
            if actions:
                self.apply_action(actions[cycle])

            predictions = self.forecast(X_test)
            future_index = pd.DatetimeIndex([start])
            self.update_state(future_index, predictions)
            self.recalculate_temporal_features(future_index)
            print(f"Cycle {cycle + 1} completed. Data updated for {start}.")

    def apply_action(self, action):
        for vault, percentage_change in action.items():
            dai_ceiling_key = f'{vault}_dai_ceiling'
            if dai_ceiling_key in self.data.columns:
                print('current_ceiling', self.data[dai_ceiling_key])
                self.data[dai_ceiling_key] *= (1 + percentage_change / 100)
                print(f"Adjusted {dai_ceiling_key} by {percentage_change}%")
                print('adjusted ceiling', self.data[dai_ceiling_key])

    def forecast(self, X):
        return self.model.predict(X)

    def update_state(self, indices, predictions):
        print('Current state', self.data[self.targets].iloc[-1])
        print('Current temporal', self.data[self.temporals].iloc[-1])
        # Create a new DataFrame for the predictions
        new_data = pd.DataFrame(predictions, index=indices, columns=self.targets)
        # Append new data to results
        self.results = pd.concat([self.results, new_data])
        # Update the main data with predictions
        self.data.update(new_data)
        # Append new data if the index does not already exist
        if not self.data.index.isin(indices).any():
            self.data = self.data.reindex(self.data.index.union(new_data.index), method='nearest')
            for column in new_data.columns:
                self.data.loc[new_data.index, column] = new_data[column]
            self.data.sort_index(inplace=True)  # Ensure the index is sorted
        else:
            self.data.update(new_data)
        print('new state update:', self.data[self.targets].iloc[-1], self.data[self.temporals].iloc[-1])
    
        # Recalculate temporal features right after updating the state
        self.recalculate_temporal_features(indices)

    def recalculate_temporal_features(self, start_index):
        # Ensure start_index is a DatetimeIndex
        if not isinstance(start_index, pd.DatetimeIndex):
            start_index = pd.to_datetime(start_index)
        
        vault_names = ['ETH Vault', 'stETH Vault', 'BTC Vault', 'Altcoin Vault', 'Stablecoin Vault', 'LP Vault', 'PSM Vault', 'RWA Vault']
        total_usd_col = 'Vaults Total USD Value'
        self.data[total_usd_col] = self.data[[f'{vault}_collateral_usd' for vault in vault_names]].sum(axis=1)

        for vault in vault_names:
            usd_col = f'{vault}_collateral_usd'
            pct_col = f'{vault}_collateral_usd % of Total'
            self.data[pct_col] = self.data[usd_col] / self.data[total_usd_col]  # Update the percentage column
            print('new pct:', self.data[pct_col])
            # Calculate the 7-day moving average for the USD collateral
            ma_col_usd_7d = f'{usd_col}_7d_ma'
            self.data[ma_col_usd_7d] = self.data[usd_col].rolling(window=7, min_periods=1).mean()
            for window in [30]:
                ma_col_pct = f'{pct_col}_{window}d_ma'
                self.data[ma_col_pct] = self.data[pct_col].rolling(window=window, min_periods=1).mean()

            dai_ceiling_col = f'{vault}_dai_ceiling'
            if dai_ceiling_col in self.data.columns:
                prev_dai_ceiling_col = f'{vault}_prev_dai_ceiling'
                self.data[prev_dai_ceiling_col] = self.data[dai_ceiling_col].shift(1)

    def plot_vault_data(self, column):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data[column], label=column)
        plt.title(f"Time Series for {column}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_simulation_results(self):
        plt.figure(figsize=(14, 7))
        for target in self.targets:
            plt.plot(self.results.index, self.results[target], label=target)
        plt.title("Simulation Results")
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_summary_statistics(self, pre_data):
        for column in self.data.columns:
            if column not in pre_data.columns:
                pre_data[column] = pd.NA  # Handle missing column in pre_data
            pre_stats = pre_data.describe()
            post_stats = self.data.describe()
            print(f"--- {column} ---")
            print("Pre-Simulation:\n", pre_stats)
            print("Post-Simulation:\n", post_stats, "\n")

    def plot_dai_ceilings_and_usd_balances(self, start_simulation_date, vault_names):
        if isinstance(start_simulation_date, str):
            start_simulation_date = pd.to_datetime(start_simulation_date)
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
        for vault in vault_names:
            axes[0].plot(self.data.index, self.data[f'{vault} Vault_dai_ceiling'], label=f'{vault} Dai Ceiling')
        axes[0].axvline(x=start_simulation_date, color='r', linestyle='--', label='Start of Simulation')
        axes[0].set_title('Dai Ceilings Over Time')
        axes[0].set_ylabel('Dai Ceiling')
        axes[0].legend()
        for vault in vault_names:
            axes[1].plot(self.data.index, self.data[f'{vault} Vault_collateral_usd'], label=f'{vault} USD Balance')
        axes[1].axvline(x=start_simulation_date, color='r', linestyle='--', label='Start of Simulation')
        axes[1].set_title('USD Balances Per Vault Over Time')
        axes[1].set_ylabel('USD Balance')
        axes[1].set_xlabel('Date')
        axes[1].legend()
        plt.show()

    def calculate_error_metrics(self, actual_data):
        vault_names = ['ETH', 'stETH', 'BTC', 'Altcoin', 'Stablecoin', 'LP', 'PSM']
        for vault in vault_names:
            column = f'{vault} Vault_collateral_usd'
            try:
                mse = mean_squared_error(actual_data[column], self.data[column])
                mae = mean_absolute_error(actual_data[column], self.data[column])
                rmse = sqrt(mse)
                print(f"--- Metrics for {vault} Vault ---")
                print(f"MSE: {mse}")
                print(f"MAE: {mae}")
                print(f"RMSE: {rmse}\n")
            except KeyError:
                print(f"Data for {vault} Vault not available in the dataset.")


# Run the simulation
simulation_data = test_data_copy
start_date = '2022-05-20'
simulation = VaultSimulator_baseline(simulation_data, test_data_copy, features, targets, temporals, start_date)
simulation.train_model()
simulation.run_simulation()
simulation.plot_simulation_results()


# In[842]:


historical = test_data[targets]
result = simulation.results



evaluate_predictions(result, historical)


# In[843]:


filter = test_data[(test_data.index <= '2022-05-26') & (test_data.index >='2022-05-20')]

filter[targets]
filter['stETH Vault_market_price'].plot()


# In[844]:


filter


# In[845]:


filtered_columns = filter.columns[~filter.columns.isin(temporals)]
filtered_data = filter[filtered_columns]
filtered_data.columns
filtered_data


# In[846]:

filter['mcap_total_volume']


# In[847]:


list(filter[[f for f in filter.columns if not 'ma' in f and not 'ceiling' in f]].columns)


# In[848]:


historical.plot()


# In[849]:


result = simulation.results
result
