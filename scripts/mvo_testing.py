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

portfolio = aggregated_vault_data[['BTC Vault_collateral_usd','ETH Vault_collateral_usd','stETH Vault_collateral_usd','Stablecoin Vault_collateral_usd'
                  ,'Altcoin Vault_collateral_usd','LP Vault_collateral_usd','RWA Vault_collateral_usd','PSM Vault_collateral_usd']]


# In[744]:


portfolio = portfolio[portfolio.index > '2021-11-18']
portfolio


# From 2020-12-29 through 2021-01-02 there is one stablecoin in the psm vault, on 2021-01-03 the balance jumps to 2964388.545068.  This skews returns for this vault, so decided to replace 1 with 0 for more accurate returns.  The 1 was likely a test.
# 
# portfolio['PSM Vault_collateral_usd'].replace(1.0, 0.0)

# In[745]:


portfolio 


# In[746]:


portfolio.columns


# In[747]:


returns = portfolio.pct_change()
returns.replace([np.inf, -np.inf], np.nan, inplace=True)

returns


# In[748]:


returns.fillna(0, inplace=True)


# In[749]:


returns.isna().sum().sum()


# In[750]:


returns


# In[751]:


returns['PSM Vault_collateral_usd'].describe()


# In[752]:


# Calculate covariance matrix and mean returns
covariance_matrix = returns.cov()
mean_returns = returns.mean()


# In[753]:


covariance_matrix


# In[754]:


mean_returns 


# In[755]:


#This gets sharpe ratio for each individual vault, not considering correlations between assets


# Assume risk-free rate is close to 0 for simplification in the crypto context
risk_free_rate = 0.00

# Calculate daily returns


# Calculate mean returns and standard deviation of returns
mean_daily_returns = returns.mean()
std_daily_returns = returns.std()

# Annualize the mean returns and standard deviation
annual_mean_returns = mean_daily_returns * 365  # 365 days, assuming daily data
annual_std_returns = std_daily_returns * np.sqrt(365)  # Square root of days to annualize

# Calculate Sharpe Ratio for each asset
non_cov_sharpe_ratios = (annual_mean_returns - risk_free_rate) / annual_std_returns

# Display the Sharpe ratios
print("Sharpe Ratios for each asset:")
print(non_cov_sharpe_ratios)


# In[756]:


#We can see, not considering covariance, stETH vault has generated highest return


# In[757]:


aggregated_vaults.columns


# In[758]:


#Now we look at returns and variance based on solely the market price, not usd value of vaults 


# In[759]:


portfolio_market_prices = aggregated_vaults[['day','BTC Vault_market_price','ETH Vault_market_price','stETH Vault_market_price','Stablecoin Vault_market_price'
                  ,'Altcoin Vault_market_price','LP Vault_market_price','RWA Vault_market_price','PSM Vault_market_price']]
portfolio_market_prices.set_index('day',inplace=True)


# In[760]:


portfolio_market_prices = portfolio_market_prices[portfolio_market_prices.index > '2021-11-18']


# In[761]:


portfolio_market_prices['LP Vault_market_price'].describe()


# In[762]:


market_price_returns = portfolio_market_prices.pct_change()
market_price_returns['LP Vault_market_price'].describe()


# In[763]:


market_price_returns['LP Vault_market_price'].fillna(0).describe()


# In[764]:


market_price_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
market_price_returns['LP Vault_market_price'].describe()


# In[765]:


market_price_returns.fillna(0, inplace=True)
market_price_returns


# In[766]:


# Calculate covariance matrix and mean returns for each weighted average market price 
mp_covariance_matrix = market_price_returns.cov()
mp_mean_returns = market_price_returns.mean()
mp_mean_returns


# In[767]:


mp_covariance_matrix


# In[768]:


# Sharpe ratios based on market price, accounting for covariance 
individual_volatilities = np.sqrt(np.diag(mp_covariance_matrix))

# Individual Sharpe ratios
asset_sharpe_ratios = (mp_mean_returns - risk_free_rate) / individual_volatilities
print("Individual Sharpe Ratios:\n", asset_sharpe_ratios)


# In[769]:


#calculate avaerage composition for each asset historically, use that as bound for psm/rwa/stablecoin weightings 

composition = aggregated_vaults[['day','BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total',
       'stETH Vault_collateral_usd % of Total',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total']]


# In[770]:


composition.set_index('day', inplace=True)
composition


# In[771]:


composition = composition[composition.index > '2021-11-18']
composition = composition / 100
composition


# In[772]:


historical_bounds = composition.mean()
historical_bounds

latest_composition = composition.iloc[-1]
latest_composition


# In[773]:


returns


# In[774]:


composition.columns = returns.columns


# In[775]:


print("Returns columns:", returns.columns)
print("Covariance matrix columns:", covariance_matrix.columns)
print("Composition columns:", composition.columns)


# In[776]:


print("Composition shape:", composition.shape)
print("Covariance matrix shape:", covariance_matrix.shape)


# In[777]:


# Assume returns and composition are already loaded DataFrames
# Calculate daily portfolio returns
daily_portfolio_returns = (returns * composition).sum(axis=1)

# Calculate mean return and standard deviation of portfolio returns
mean_portfolio_return = daily_portfolio_returns.mean()
std_dev_portfolio = daily_portfolio_returns.std()

# Define the risk-free rate (annualized if your returns are daily)
# Example: 0.5% annualized risk-free rate
risk_free_rate = 0

# Calculate the Sharpe Ratio
sharpe_ratio = (mean_portfolio_return - risk_free_rate) / std_dev_portfolio

print("Mean Portfolio Return:",mean_portfolio_return)
print("Standard Deviation of Portfolio:", std_dev_portfolio)
print("Portfolio Sharpe Ratio:", sharpe_ratio)


# In[778]:


daily_portfolio_returns.plot()


# In[779]:


#Now we calculate weights targeting best sharpe ratio



# Number of assets
num_assets = len(portfolio.columns)

# Calculate returns and covariance
mean_returns = returns.mean()
covariance_matrix = returns.cov()

# Objective function (Minimize risk for a given return)
def objective(weights): 
    return - (weights.T @ mean_returns) / (np.sqrt(weights.T @ covariance_matrix @ weights))  # Negative Sharpe Ratio for minimization

# Constraints
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights must be 1

# Bounds with custom limits
bounds = [(0.1, 1), (0.1, 1), (0.1, 1), (0, 0.2), (0.05, 0.25), (0.1, 1), (0, 0.02), (0.05, 0.16)]
# Adjusted bounds to prioritize certain assets:
# BTC, ETH, stETH, LP: Minimum 10%, no maximum (up to 1 or 100%)
# Stablecoin, RWA: Maximum 20%
# Altcoin, PSM: Moderate flexibility with a minimum of 5% and a maximum of 25%

# Initial guess
initial_weights = np.array(num_assets * [1. / num_assets,])  # Equal weight for start

# Minimize the objective function
opt_result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

# Output the results
if opt_result.success:
    sharpe_optimal_weights = opt_result.x
    print("Adjusted Optimal Weights:", sharpe_optimal_weights)
else:
    print("Optimization failed:", opt_result.message)


# In[780]:


# Calculate portfolio standard deviation (volatility)
optimized_portfolio_std_dev = np.sqrt(sharpe_optimal_weights.T @ covariance_matrix @ sharpe_optimal_weights)

# Calculate portfolio return
optimized_portfolio_return = sharpe_optimal_weights.T @ mean_returns

# Risk-free rate assumption (can be set to a fixed value, e.g., 0 if not provided)
risk_free_rate = 0

# Calculate portfolio Sharpe ratio
optimized_portfolio_sharpe_ratio = (optimized_portfolio_return) / optimized_portfolio_std_dev
print("Optimized Portfolio Return:", optimized_portfolio_return)
print("Optimized Portfolio Standard Deviation:", optimized_portfolio_std_dev)
print("Optimized Portfolio Sharpe Ratio:", optimized_portfolio_sharpe_ratio)


# In[781]:


# Individual vault sharpe ratios
individual_volatilities = np.sqrt(np.diag(covariance_matrix))

# Individual Sharpe ratios
vault_sharpe_ratios = (mean_returns - risk_free_rate) / individual_volatilities
print("Individual Sharpe Ratios:\n", vault_sharpe_ratios)


# In[782]:


# individual asset sharpe ratios (based on market price)

asset_sharpe_ratios.fillna(0, inplace=True)
asset_sharpe_ratios


# In[ ]:





# ## Sortino Ratio

# In[783]:


returns


# In[784]:


composition 


# ### Portfolio Sortino Ratio

# In[785]:


# Annual risk-free rate
annual_risk_free_rate = 0.05

# Convert to daily risk-free rate
daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/365) - 1

# Now use this daily rate in your calculations
# Assuming you use it as the Minimum Acceptable Return (MAR)
MAR = daily_risk_free_rate  # This could also be set to zero if you're only measuring against downside deviation

# Ensure that 'returns' and 'composition' are aligned and sorted by 'day'
returns.sort_index(inplace=True)
composition.sort_index(inplace=True)

# Calculate daily portfolio returns by weighting the returns by their composition percentages
portfolio_daily_returns = (returns * composition).sum(axis=1)

# Excess returns and downside deviation
excess_returns = portfolio_daily_returns - MAR
downside_returns = excess_returns[excess_returns < 0]  # Only consider negative excess returns for downside deviation

# Calculate downside deviation and average excess returns
downside_deviation = np.sqrt((downside_returns**2).mean())
average_excess_return = excess_returns.mean()

# Calculate Sortino Ratio
portfolio_sortino_ratio = average_excess_return / downside_deviation if downside_deviation != 0 else np.inf

print('Daily Risk-Free Rate:', daily_risk_free_rate)
print('Minimum Acceptable Return (MAR):', MAR)
print('Average Excess Return:', average_excess_return)
print('Downside Deviation:', downside_deviation)
print("Portfolio Sortino Ratio:", portfolio_sortino_ratio)


# In[786]:


# Ensure your data is in a pandas DataFrame for easier plotting
# Assuming 'portfolio_daily_returns' is already a pandas Series with datetime index

# 1. Time Series Plot of Portfolio Returns
plt.figure(figsize=(10, 6))
portfolio_daily_returns.plot(title='Daily Portfolio Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.grid(True)
plt.show()

# 2. Histogram of Portfolio Returns
plt.figure(figsize=(10, 6))
sns.histplot(portfolio_daily_returns, kde=True, bins=30)
plt.title('Histogram of Portfolio Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.show()

# 3. Cumulative Returns Plot
cumulative_returns = (1 + portfolio_daily_returns).cumprod()
plt.figure(figsize=(10, 6))
cumulative_returns.plot(title='Historical Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.show()

# 4. Downside Returns Plot
plt.figure(figsize=(10, 6))
downside_returns.plot(style='ro', title='Downside Returns Below MAR')
plt.axhline(0, color='k', linestyle='--')  # Add a line at 0 for reference
plt.xlabel('Date')
plt.ylabel('Downside Returns')
plt.grid(True)
plt.show()

# 5. Excess Returns Over MAR
excess_returns.plot(style='go', title='Excess Returns Over MAR')
plt.axhline(0, color='k', linestyle='--')  # Add a line at MAR for reference
plt.xlabel('Date')
plt.ylabel('Excess Returns')
plt.grid(True)
plt.show()


# ### MVO with Sortino Ratio Objective

# In[787]:


historical_bounds


# In[788]:


MAR


# In[789]:


returns


# In[790]:


# Calculate excess returns below MAR
excess_returns = returns - MAR

# Downside deviation
downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
downside_deviation = np.sqrt(downside_returns.mean())

# Average excess returns
average_excess_returns = excess_returns.mean()

# Calculate Sortino Ratio for each asset
sortino_ratios = average_excess_returns / downside_deviation
print("Sortino Ratios for each asset:\n", sortino_ratios)

# Implementing MVO based on Sortino Ratio

def sortino_ratio_objective(weights):
    portfolio_returns = np.dot(returns, weights)
    excess_portfolio_returns = portfolio_returns - MAR
    downside_portfolio_returns = np.where(excess_portfolio_returns < 0, excess_portfolio_returns**2, 0)
    portfolio_downside_deviation = np.sqrt(np.mean(downside_portfolio_returns) + 1e-6)  # Adding epsilon to avoid division by zero
    portfolio_return = np.mean(excess_portfolio_returns)
    return -portfolio_return / portfolio_downside_deviation

# Relax bounds to give more flexibility
bounds = [(0.01, .4),(0.2, .4),(0.01, .2),(0.01, 0.02),(0.01, .4),(0.01, 0.25),(0.01, 0.02),(0.01,0.25)]  # Minimum bound slightly above zero

# Initial weights more evenly distributed
initial_weights = np.full(num_assets, 1/num_assets)

# Optimization settings
options = {
    'maxiter': 1000,  # Increase the maximum number of iterations
    'disp': True      # Display progress
}

# Redo the optimization with updated settings
result = minimize(sortino_ratio_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints, options=options)

# Output the results
if result.success:
    sortino_optimized_weights = result.x
    print("Optimized Weights for Maximum Sortino Ratio:", sortino_optimized_weights)
else:
    print("Optimization failed:", result.message)


# In[791]:


# Assuming `returns` is a DataFrame with each column representing an asset's returns
# and `optimized_weights` is a NumPy array with the optimized weights for each asset.

# Step 1: Calculate portfolio returns
optimized_portfolio_daily_returns = np.dot(returns, sortino_optimized_weights)

# Step 2: Calculate excess returns relative to MAR

optimized_excess_returns = optimized_portfolio_daily_returns - MAR

# Step 3: Identify downside returns
optimized_downside_returns = np.where(optimized_excess_returns < 0, optimized_excess_returns**2, 0)

# Step 4: Calculate downside deviation
optimized_downside_deviation = np.sqrt(np.mean(optimized_downside_returns))

# Step 5: Calculate average excess return above MAR
optimized_average_excess_return = np.mean(excess_returns[excess_returns > 0])

# Step 6: Compute the Sortino Ratio
optimized_portfolio_sortino_ratio = optimized_average_excess_return / optimized_downside_deviation

print('Minimum Acceptable Return (MAR):', MAR)

print("Average Excess Return:",optimized_average_excess_return)
print("Downside Deviation:",optimized_downside_deviation)
print("Portfolio Sortino Ratio:", optimized_portfolio_sortino_ratio)


# In[792]:


# Ensure your data is in a pandas DataFrame for easier plotting
# Assuming 'portfolio_daily_returns' is already a pandas Series with datetime index
optimized_portfolio_daily_returns_series = pd.Series(optimized_portfolio_daily_returns, index=returns.index)


# 1. Time Series Plot of Portfolio Returns
plt.figure(figsize=(10, 6))
optimized_portfolio_daily_returns_series.plot(title='Daily Portfolio Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.grid(True)
plt.show()

# 2. Histogram of Portfolio Returns
plt.figure(figsize=(10, 6))
sns.histplot(optimized_portfolio_daily_returns, kde=True, bins=30)
plt.title('Histogram of Portfolio Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.show()

# 3. Cumulative Returns Plot
optimized_cumulative_returns = (1 + optimized_portfolio_daily_returns_series).cumprod()
plt.figure(figsize=(10, 6))
optimized_cumulative_returns.plot(title='Optimized Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.show()

optimized_downside_returns_series = pd.Series(optimized_downside_returns, index=returns.index)

# 4. Downside Returns Plot
plt.figure(figsize=(10, 6))
optimized_downside_returns_series.plot(style='ro', title='Downside Returns Below MAR')
plt.axhline(0, color='k', linestyle='--')  # Add a line at 0 for reference
plt.xlabel('Date')
plt.ylabel('Downside Returns')
plt.grid(True)
plt.show()

optimized_excess_returns_series = pd.Series(optimized_excess_returns, index=returns.index)

# 5. Excess Returns Over MAR
optimized_excess_returns_series.plot(style='go', title='Excess Returns Over MAR')
plt.axhline(0, color='k', linestyle='--')  # Add a line at MAR for reference
plt.xlabel('Date')
plt.ylabel('Excess Returns')
plt.grid(True)
plt.show()


# In[793]:


cumulative_returns


# In[794]:


optimized_cumulative_returns


# In[795]:


optimized_portfolio_daily_returns_series


# In[796]:


portfolio_daily_returns


# In[797]:


returns