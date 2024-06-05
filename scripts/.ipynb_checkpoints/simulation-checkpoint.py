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

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl

# Disable scientific notation globally
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.use_locale'] = False
mpl.rcParams['axes.formatter.limits'] = (-5, 6)

def apply_scalar_formatter(ax):
    for axis in [ax.xaxis, ax.yaxis]:
        if isinstance(axis.get_major_formatter(), ScalarFormatter):
            axis.set_major_formatter(ScalarFormatter())
            axis.get_major_formatter().set_scientific(False)
            axis.get_major_formatter().set_useOffset(False)

"""
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
"""
class RL_VaultSimulator:
    def __init__(self, data, initial_data, features, targets, temporals, start_date, end_date, scale_factor=300000000,minimum_value_factor=0.05,volatility_window=250, alpha=100):
        self.data = data[data.index <= pd.to_datetime(start_date).tz_localize(None)]
        self.features = features
        self.targets = targets
        self.alpha = alpha
        self.model = None
        self.temporals = temporals
        self.results = pd.DataFrame()
        self.initial_data = initial_data
        self.start_date = pd.to_datetime(start_date).tz_localize(None)
        self.end_date = pd.to_datetime(end_date).tz_localize(None)
        self.current_date = self.start_date
        self.dai_ceilings_history = pd.DataFrame()
        self.volatility_window = volatility_window
        self.scale_factor = scale_factor
        self.minimum_value_factor = minimum_value_factor


    def get_latest_data(self):
        # Return the latest data that the environment will use
        return self.data
        
    def reset(self):
        self.data = self.initial_data[self.initial_data.index <= self.start_date]
        self.results = pd.DataFrame()
        self.train_model()
        print('sim reset current date', self.current_date)
        print("Simulation reset and model retrained.")

    def train_model(self):
        X = self.initial_data[self.features]
        y = self.initial_data[self.targets]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
        self.model = MultiOutputRegressor(Ridge(alpha=self.alpha))
        self.model.fit(X_train, y_train)
        print("Model trained.")
        
    def update_dai_ceilings(self):
        # Extract the current DAI ceilings
        current_ceilings = self.data.loc[self.current_date, ['ETH Vault_dai_ceiling', 'BTC Vault_dai_ceiling', 'stETH Vault_dai_ceiling', 'Altcoin Vault_dai_ceiling', 'Stablecoin Vault_dai_ceiling', 'LP Vault_dai_ceiling','RWA Vault_dai_ceiling','PSM Vault_dai_ceiling']]
        current_ceilings['timestamp'] = self.current_date  # Adding a timestamp for reference

        # Append to the historical DataFrame
        self.dai_ceilings_history = pd.concat([self.dai_ceilings_history, current_ceilings.to_frame().T.set_index('timestamp')])



    def run_simulation(self, simulation_date, action=None):
        # Ensure the date is timezone-aware. Localize if it is naive.
        #if pd.to_datetime(simulation_date).tzinfo is None:
           # self.current_date = pd.to_datetime(simulation_date).tz_localize('UTC')
        #else:
            #self.current_date = pd.to_datetime(simulation_date).tz_convert('UTC')
    
        cycle_start_date = self.current_date
        end_date = min(cycle_start_date + timedelta(days=24), self.end_date)
    
        while self.current_date <= end_date and self.current_date <= self.initial_data.index.max():
            if self.current_date in self.data.index:
                X_test = self.data.loc[[self.current_date], self.features]
            else:
                X_test = self.data.tail(1)[self.features]
    
            volatilities = self.calculate_historical_volatility()
            predictions = self.forecast(X_test, volatilities)
            future_index = pd.DatetimeIndex([self.current_date])
            self.update_state(future_index, predictions)
    
            # Update the DAI ceilings history right after updating the state
            self.update_dai_ceilings()
    
            print('current state', self.data.iloc[-1])
            if action:
                self.apply_action(action)
                print(f"Action applied on: {self.current_date}")
    
            print(f"Day completed: {self.current_date}")
            self.current_date += timedelta(days=1)
    
        print(f"Cycle completed up to: {self.current_date - timedelta(days=1)}")
        if self.current_date > self.end_date:
            print(f"Simulation completed up to: {self.end_date}")
        else:
            print(f"Simulation completed up to: {self.current_date - timedelta(days=1)}")




    def apply_action(self, action):
        base_value_if_zero = 10000  # Base value to set if the initial DAI ceiling is zero
        if action:
            for vault, percentage_change in action.items():
                # Append the suffix '_dai_ceiling' to the vault name to match the DataFrame columns
                dai_ceiling_key = vault.replace('_collateral_usd', '_dai_ceiling') 
                print('vault', vault)
                print('Applying action to:', dai_ceiling_key)
                if dai_ceiling_key in self.data.columns:
                    original_value = self.data[dai_ceiling_key].iloc[-1]
                    if original_value == 0:
                        # If the original value is 0, initialize it with the base value
                        new_value = base_value_if_zero * (1 + percentage_change / 100)
                        print(f"Initialized and adjusted {dai_ceiling_key} from 0 to {new_value}")
                    else:
                        new_value = original_value * (1 + percentage_change / 100)
                        print(f"Adjusted {dai_ceiling_key} by {percentage_change}% from {original_value} to {new_value}")

                    self.data.at[self.data.index[-1], dai_ceiling_key] = new_value
                else:
                    print(f"No 'dai_ceiling' column found for {dai_ceiling_key}, no action applied.")
        else:
            print("No action provided; no adjustments made.")


    def forecast(self, X, volatilities):
        predictions = self.model.predict(X)
        predictions = np.maximum(predictions, 0)  # Ensure predictions are non-negative before adjustment
        
        # Scale factor for volatility should be set based on historical volatility analysis
        #try different scale factor; optimal @ 500000000 or 4, 453000000, 467000000
        scale_factor = self.scale_factor  # This should be calibrated based on your data
        noise = np.random.normal(0, volatilities * scale_factor, predictions.shape)
        
        # Apply noise and ensure predictions do not fall below a realistic minimum
        # Example: Set a floor at 1% of the mean historical asset value or a fixed value known to be a plausible minimum
        #0.2, 0.15 optimal, 0.12
        minimum_value =  self.minimum_value_factor * self.initial_data[self.targets].mean()  # This is an example and should be adjusted
        adjusted_predictions = np.maximum(predictions + noise, minimum_value)
        
        return adjusted_predictions
        # try scale 453000000, min val .12 or .1, window 25 or 15


    def calculate_historical_volatility(self): #try 365 for straightish lines, maybe try longer?
        # try different window; optimal @ 280, 25, 15, 380, 360
        # 180, 150 also good, 60 for smaller scale and min val
        window=self.volatility_window
    
        # Assuming daily data, calculate percentage change
        daily_returns = self.data[self.targets].pct_change()
    
        # Handling possible NaN values in daily returns
        daily_returns = daily_returns.dropna()
    
        # Calculate volatility as the standard deviation of returns
        volatility = daily_returns.rolling(window=window, min_periods=1).std()
    
        # Return the average volatility over the window
        return volatility.mean(axis=0)  # Use axis=0 to average volatilities across columns if needed



    
    def update_state(self, indices, predictions):
        #print('Current state', self.data[self.targets].iloc[-1])
        #print('Current temporal', self.data[self.temporals].iloc[-1])
        # Create a new DataFrame for the predictions
        new_data = pd.DataFrame(predictions, index=indices, columns=self.targets)
        new_data = new_data.clip(lower=0)
        #print('New data', new_data)
        self.results = pd.concat([self.results, new_data])  # Append new data to results
        self.data.update(new_data)
        # Append new data if the index does not already exist
        if not self.data.index.isin(indices).any():
            # Directly assign the new data to the respective index positions
            self.data = self.data.reindex(self.data.index.union(new_data.index), method='nearest')
            for column in new_data.columns:
                self.data.loc[new_data.index, column] = new_data[column]
            self.data.sort_index(inplace=True)  # Ensure the index is sorted
        else:
            # If indices overlap, directly update the values
            self.data.update(new_data)
        #print('new state update:',self.data[self.targets].iloc[-1],self.data[self.temporals].iloc[-1])
    
        # Recalculate temporal features right after updating the state
        self.recalculate_temporal_features(indices)


    

    def recalculate_temporal_features(self, start_index):
        vault_names = ['ETH Vault', 'stETH Vault', 'BTC Vault', 'Altcoin Vault', 'Stablecoin Vault', 'LP Vault', 'PSM Vault', 'RWA Vault']
        total_usd_col = 'Vaults Total USD Value'
        self.data[total_usd_col] = self.data[[f'{vault}_collateral_usd' for vault in vault_names]].sum(axis=1)
    
        for vault in vault_names:
            usd_col = f'{vault}_collateral_usd'
            pct_col = f'{vault}_collateral_usd % of Total'
            self.data[pct_col] = self.data[usd_col] / self.data[total_usd_col]  # Update the percentage column
            # Calculate the 7-day moving average for the USD collateral
            #ma_col_usd_7d = f'{usd_col}_7d_ma'
            #self.data[ma_col_usd_7d] = self.data[usd_col].rolling(window=7, min_periods=1).mean()
            ma_col_usd_30d = f'{usd_col}_30d_ma'
            self.data[ma_col_usd_30d] = self.data[usd_col].rolling(window=30, min_periods=1).mean()
            # Calculate the 7-day and 30-day moving averages for the percentage of total
            for window in [7,30]:
                ma_col_pct = f'{pct_col}_{window}d_ma'
                self.data[ma_col_pct] = self.data[pct_col].rolling(window=window, min_periods=1).mean()
             
            dai_ceiling_col = f'{vault}_dai_ceiling'
            if dai_ceiling_col in self.data.columns:
                prev_dai_ceiling_col = f'{vault}_prev_dai_ceiling'
                self.data[prev_dai_ceiling_col] = self.data[dai_ceiling_col].shift(1)
    def print_summary_statistics(self, pre_data):
        for column in self.data.columns:
            if column not in pre_data.columns:
                pre_data[column] = pd.NA  # Handle missing column in pre_data
            pre_stats = pre_data.describe()
            post_stats = self.data.describe()
            print(f"--- {column} ---")
            print("Pre-Simulation:\n", pre_stats)
            print("Post-Simulation:\n", post_stats, "\n")
            
    def plot_vault_data(self, column):
        vault_usd_col = f'{column}_collateral_usd'
        vault_dai_col = f'{column}_dai_ceiling'
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot USD balance on the primary y-axis
        ax1.plot(self.data.index, self.data[vault_usd_col], label=f'{column} USD Balance', color='blue')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("USD Balance", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Create a secondary y-axis for the DAI ceiling
        ax2 = ax1.twinx()
        ax2.plot(self.data.index, self.data[vault_dai_col], label=f'{column} DAI Ceiling', color='orange')
        ax2.set_ylabel("DAI Ceiling", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper right')
        
        fig.suptitle(f"Time Series for {column}")
        
        apply_scalar_formatter(ax1)
        apply_scalar_formatter(ax2)
        plt.show()
        
        
    def plot_simulation_results(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        for target in self.targets:
            ax.plot(self.results.index, self.results[target], label=target)
        ax.set_title("Simulation Results")
        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        ax.legend()
        ax.grid(True)
        #ax.ticklabel_format(useOffset=False, style='plain')
        apply_scalar_formatter(ax)
        plt.show()

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
        #axes[0].ticklabel_format(useOffset=False, style='plain')
        apply_scalar_formatter(axes[0])
        
        for vault in vault_names:
            axes[1].plot(self.data.index, self.data[f'{vault} Vault_collateral_usd'], label=f'{vault} USD Balance')
        axes[1].axvline(x=start_simulation_date, color='r', linestyle='--', label='Start of Simulation')
        axes[1].set_title('USD Balances Per Vault Over Time')
        axes[1].set_ylabel('USD Balance')
        axes[1].set_xlabel('Date')
        axes[1].legend()
        #axes[1].ticklabel_format(useOffset=False, style='plain')
        apply_scalar_formatter(axes[1])
        
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

"""
simulation_data = test_data_copy  # Assuming this is defined with your actual data
simulation_data.index = simulation_data.index.tz_localize(None)  # Remove timezone information
start_date = '2022-05-20'
end_date = '2024-03-20'

simulation = RL_VaultSimulator(simulation_data, simulation_data, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)
simulation.train_model()
simulation.run_simulation(start_date)
simulation.plot_simulation_results()


# Plot Dai ceilings and USD balances
vault_names = ['ETH', 'stETH', 'BTC', 'Altcoin', 'Stablecoin', 'LP', 'PSM', 'RWA']
simulation.plot_dai_ceilings_and_usd_balances(start_date, vault_names)

# Calculate error metrics against actual data (if available)
# simulation.calculate_error_metrics(actual_data)


# result = simulation.results
# evaluate_predictions(result, historical)

# ### Filter for MVO

# In[852]:


historical_data = historical[historical.index <= '2022-05-19']


# In[853]:


historical_data_mvo = historical_data.copy()

historical_data_mvo.index= historical_data_mvo.index.tz_localize(None)


# In[854]:


result.index


# In[855]:


combined_data = pd.concat([historical_data_mvo, result])

# Optional: Sort the DataFrame by index if it's not already sorted
combined_data.sort_index(inplace=True)

# Now 'combined_data' contains both historical and simulation data in one DataFrame
print(combined_data)


# In[856]:


historical_comparison = historical[historical.index <= '2022-06-12']
historical_comparison


# In[857]:


result
"""