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

def to_sequences(X, y, time_steps=1):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# In[701]:


st_aggregated_vault_data = aggregated_vault_data[aggregated_vault_data.index > '2021-11-18']
st_aggregated_vault_data.isna().sum().sum()


# In[702]:


significant_eth_aggregated_all_vaults_spearman[['ETH Vault_collateral_usd','ETH Vault_dai_ceiling']].plot()


# In[703]:


# Ensure x_st and y_st are pandas Series with appropriate indices
x_st = st_aggregated_vault_data['ETH Vault_dai_ceiling']
y_st = st_aggregated_vault_data['ETH Vault_collateral_usd']

# Split the data, ensuring it retains its DataFrame structure
X_train, X_test, y_train, y_test = train_test_split(x_st.to_frame(), y_st, test_size=0.2, random_state=42)

# Fit the model
model_st = LinearRegression()
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)



# Call the plotting function
plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")


# In[704]:


# Assuming 'st_aggregated_vault_data' contains your dataset
x_st = st_aggregated_vault_data[['ETH Vault_dai_ceiling','ETH Vault_dai_ceiling_7d_ma','ETH Vault_collateral_usd_7d_ma', 'BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total','ETH Vault_collateral_usd % of Total_7d_ma','ETH Vault_collateral_usd % of Total_30d_ma',
       'stETH Vault_collateral_usd % of Total',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total']]

y_st = st_aggregated_vault_data['ETH Vault_collateral_usd']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)

# Initialize the linear regression model
model_st = LinearRegression()

# Fit the model to the training data
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")

model_st = Ridge(alpha=550.0)

tscv = TimeSeriesSplit(n_splits=5)
mae_scores, r2_scores = [], []

for train_index, test_index in tscv.split(x_st):
    X_train, X_test = x_st.iloc[train_index], x_st.iloc[test_index]
    y_train, y_test = y_st.iloc[train_index], y_st.iloc[test_index]

    model_st.fit(X_train, y_train)
    y_pred = model_st.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated MAE scores:", mae_scores)
print("Cross-validated R-squared scores:", r2_scores)


# In[705]:




# Load and prepare the dataset
#data = significant_spearman_df_no_collat.drop(columns=['eth_a_vault_cumulative_collateral'])

data_multi = significant_eth_aggregated_all_vaults_spearman.drop(columns=['ETH Vault_collateral_usd','ETH Vault_hypothetical_dai_ceiling'])

target = significant_eth_aggregated_all_vaults_spearman['ETH Vault_collateral_usd']

# performs .74 '3_m_tbill_yield','eth_a_vault_liquidation_ratio','debt_ratio_Lag_2m','dsr_rate','vc_revenue','dai_abs_deviation_30d_ma','b_s_DAI_30d_rolling_avg_pct_chg','M2V','psm_lifetime_fees'

# Ensure data is in the correct shape for scaling
if data_multi.ndim == 1:
    data_multi = data_multi.values.reshape(-1, 1)  # Reshape data to 2D if it's 1D
else:
    data_multi = data_multi.values

# Scale features and target
scaler_feature = MinMaxScaler(feature_range=(0, 1))
data_scaled_multi = scaler_feature.fit_transform(data_multi)
scaler_target = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Create sequences for LSTM
time_step = 15
X, y = to_sequences(data_scaled_multi, target_scaled, time_step)

# Split into train and test sets
train_size = int(len(X) * 0.67)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Define the LSTM model with dropout and L2 regularization
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(45, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.08)),
    LSTM(20, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.02)),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with early stopping
history = model.fit(
    X_train, y_train, epochs=100, batch_size=40,
    validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping]
)

# Predict on training and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and targets
train_predict_multi_inv_eth = scaler_target.inverse_transform(train_predict)
test_predict_multi_inv_eth = scaler_target.inverse_transform(test_predict)
y_train_multi_inv_eth = scaler_target.inverse_transform(y_train.reshape(-1, 1))
y_test_multi_inv_eth = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
train_rmse_eth = np.sqrt(mean_squared_error(y_train_multi_inv_eth, train_predict_multi_inv_eth))
test_rmse_eth = np.sqrt(mean_squared_error(y_test_multi_inv_eth, test_predict_multi_inv_eth))

# Calculate R-squared
train_r2_eth = r2_score(y_train_multi_inv_eth, train_predict_multi_inv_eth)
test_r2_eth = r2_score(y_test_multi_inv_eth, test_predict_multi_inv_eth)

# Calculate MAE
train_mae_eth = mean_absolute_error(y_train_multi_inv_eth, train_predict_multi_inv_eth)
test_mae_eth = mean_absolute_error(y_test_multi_inv_eth, test_predict_multi_inv_eth)

# Calculate MAPE
train_mape_eth = mean_absolute_percentage_error(y_train_multi_inv_eth, train_predict_multi_inv_eth)
test_mape_eth = mean_absolute_percentage_error(y_test_multi_inv_eth, test_predict_multi_inv_eth)

# Output the metrics
print(f"Train RMSE: {train_rmse_eth}, Test RMSE: {test_rmse_eth}")
print(f"Train R²: {train_r2_eth}, Test R²: {test_r2_eth}")
print(f"Train MAE: {train_mae_eth}, Test MAE: {test_mae_eth}")
print(f"Train MAPE: {train_mape_eth}%, Test MAPE: {test_mape_eth}%")


# ## stETH vault

# In[706]:


significant_steth_aggregated_all_vaults_spearman[['stETH Vault_collateral_usd','stETH Vault_dai_ceiling']].plot()


# In[707]:


# Ensure x_st and y_st are pandas Series with appropriate indices
x_st = st_aggregated_vault_data['stETH Vault_dai_ceiling']
y_st = st_aggregated_vault_data['stETH Vault_collateral_usd']

# Split the data, ensuring it retains its DataFrame structure
X_train, X_test, y_train, y_test = train_test_split(x_st.to_frame(), y_st, test_size=0.2, random_state=42)

# Fit the model
model_st = LinearRegression()
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)



# Call the plotting function
plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")


# In[708]:


# Assuming 'st_aggregated_vault_data' contains your dataset

x_st = st_aggregated_vault_data[['stETH Vault_dai_ceiling','stETH Vault_dai_ceiling_7d_ma','stETH Vault_collateral_usd_7d_ma', 'BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total', 
       'stETH Vault_collateral_usd % of Total','stETH Vault_collateral_usd % of Total_30d_ma','stETH Vault_collateral_usd % of Total_30d_ma',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total']]

y_st = st_aggregated_vault_data['stETH Vault_collateral_usd']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)

# Initialize the linear regression model
model_st = LinearRegression()

# Fit the model to the training data
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")

model_st = Ridge(alpha=300.0)

tscv = TimeSeriesSplit(n_splits=5)
mae_scores, r2_scores = [], []

for train_index, test_index in tscv.split(x_st):
    X_train, X_test = x_st.iloc[train_index], x_st.iloc[test_index]
    y_train, y_test = y_st.iloc[train_index], y_st.iloc[test_index]

    model_st.fit(X_train, y_train)
    y_pred = model_st.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated MAE scores:", mae_scores)
print("Cross-validated R-squared scores:", r2_scores)


significant_wbtc_aggregated_all_vaults_spearman[['BTC Vault_collateral_usd', 'BTC Vault_dai_ceiling']].plot()


# In[710]:


for index, value in correlated_aggregated_wbtc_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[711]:


# Make sure x_st and y_st are pandas Series with the right indices (assuming they are columns in a DataFrame)
x_st = st_aggregated_vault_data['BTC Vault_dai_ceiling'].to_frame()  # Convert to DataFrame
y_st = st_aggregated_vault_data['BTC Vault_collateral_usd']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)

# Model training
model_st = LinearRegression()
model_st.fit(X_train, y_train)

# Predictions
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Performance metrics
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")


# In[712]:


# Assuming 'st_aggregated_vault_data' contains your dataset

x_st = st_aggregated_vault_data[['BTC Vault_dai_ceiling', 'BTC Vault_dai_ceiling_7d_ma','BTC Vault_collateral_usd_7d_ma','BTC Vault_collateral_usd % of Total',
                                 'BTC Vault_collateral_usd % of Total_7d_ma','BTC Vault_collateral_usd % of Total_30d_ma',
       'ETH Vault_collateral_usd % of Total','BTC Vault_dart','BTC Vault_dart_7d_ma','BTC Vault_prev_dai_ceiling',
       'stETH Vault_collateral_usd % of Total',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total']]

y_st = st_aggregated_vault_data['BTC Vault_collateral_usd']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)

# Initialize the linear regression model
model_st = LinearRegression()

# Fit the model to the training data
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)


plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")

model_st = Ridge(alpha=3000.0)

tscv = TimeSeriesSplit(n_splits=5)
mae_scores, r2_scores = [], []

for train_index, test_index in tscv.split(x_st):
    X_train, X_test = x_st.iloc[train_index], x_st.iloc[test_index]
    y_train, y_test = y_st.iloc[train_index], y_st.iloc[test_index]

    model_st.fit(X_train, y_train)
    y_pred = model_st.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated MAE scores:", mae_scores)
print("Cross-validated R-squared scores:", r2_scores)


# In[713]:






# Load and prepare the dataset
#data = significant_spearman_df_no_collat.drop(columns=['eth_a_vault_cumulative_collateral'])

data_multi = significant_wbtc_aggregated_all_vaults_spearman.drop(columns=['BTC Vault_collateral_usd'])

target = significant_wbtc_aggregated_all_vaults_spearman['BTC Vault_collateral_usd']

# performs .74 '3_m_tbill_yield','eth_a_vault_liquidation_ratio','debt_ratio_Lag_2m','dsr_rate','vc_revenue','dai_abs_deviation_30d_ma','b_s_DAI_30d_rolling_avg_pct_chg','M2V','psm_lifetime_fees'

# Ensure data is in the correct shape for scaling
if data_multi.ndim == 1:
    data_multi = data_multi.values.reshape(-1, 1)  # Reshape data to 2D if it's 1D
else:
    data_multi = data_multi.values

# Scale features and target
scaler_feature = MinMaxScaler(feature_range=(0, 1))
data_scaled_multi = scaler_feature.fit_transform(data_multi)
scaler_target = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Create sequences for LSTM
time_step = 15
X, y = to_sequences(data_scaled_multi, target_scaled, time_step)

# Split into train and test sets
train_size = int(len(X) * 0.67)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Define the LSTM model with dropout and L2 regularization
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(45, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(20, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.02)),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.002), loss='mean_squared_error')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with early stopping
history = model.fit(
    X_train, y_train, epochs=100, batch_size=40,
    validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping]
)

# Predict on training and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and targets
train_predict_multi_inv_btc = scaler_target.inverse_transform(train_predict)
test_predict_multi_inv_btc = scaler_target.inverse_transform(test_predict)
y_train_multi_inv_btc = scaler_target.inverse_transform(y_train.reshape(-1, 1))
y_test_multi_inv_btc = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
train_rmse_btc = np.sqrt(mean_squared_error(y_train_multi_inv_btc, train_predict_multi_inv_btc))
test_rmse_btc = np.sqrt(mean_squared_error(y_test_multi_inv_btc, test_predict_multi_inv_btc))

# Calculate R-squared
train_r2_btc = r2_score(y_train_multi_inv_btc, train_predict_multi_inv_btc)
test_r2_btc = r2_score(y_test_multi_inv_btc, test_predict_multi_inv_btc)

# Calculate MAE
train_mae_btc = mean_absolute_error(y_train_multi_inv_btc, train_predict_multi_inv_btc)
test_mae_btc = mean_absolute_error(y_test_multi_inv_btc, test_predict_multi_inv_btc)

# Calculate MAPE
train_mape_btc = mean_absolute_percentage_error(y_train_multi_inv_btc, train_predict_multi_inv_btc)
test_mape_btc = mean_absolute_percentage_error(y_test_multi_inv_btc, test_predict_multi_inv_btc)

# Output the metrics
print(f"Train RMSE: {train_rmse_btc}, Test RMSE: {test_rmse_btc}")
print(f"Train R²: {train_r2_btc}, Test R²: {test_r2_btc}")
print(f"Train MAE: {train_mae_btc}, Test MAE: {test_mae_btc}")
print(f"Train MAPE: {train_mape_btc}%, Test MAPE: {test_mape_btc}%")


# ## Altcoin Vault

# In[714]:


st_aggregated_vault_data[['Altcoin Vault_collateral_usd','Altcoin Vault_dai_ceiling']].plot()


# In[715]:


for index, value in correlated_aggregated_alt_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[716]:


x_st = st_aggregated_vault_data['Altcoin Vault_dai_ceiling'].to_frame()
y_st = st_aggregated_vault_data['Altcoin Vault_collateral_usd']

X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)  # Assuming 'y' is your target variable




model_st = LinearRegression()
model_st.fit(X_train, y_train)

y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")


# In[717]:


#need to see if these vaults are closed 

st_aggregated_vault_data[['Altcoin Vault_dai_ceiling','Altcoin Vault_collateral_usd','Altcoin Vault_market_price']]


# In[718]:


# Assuming 'st_aggregated_vault_data' contains your dataset

x_st = st_aggregated_vault_data[['Altcoin Vault_dai_ceiling','Altcoin Vault_dai_ceiling_7d_ma','Altcoin Vault_collateral_usd_7d_ma', 'BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total','Altcoin Vault_dart','stETH Vault_collateral_usd',
       'stETH Vault_collateral_usd % of Total',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total','Altcoin Vault_collateral_usd % of Total_30d_ma','Altcoin Vault_collateral_usd % of Total_30d_ma',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total']]

y_st = st_aggregated_vault_data['Altcoin Vault_collateral_usd']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)

# Initialize the linear regression model
model_st = LinearRegression()

# Fit the model to the training data
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")

model_st = Ridge(alpha=20000.0)

tscv = TimeSeriesSplit(n_splits=5)
mae_scores, r2_scores = [], []

for train_index, test_index in tscv.split(x_st):
    X_train, X_test = x_st.iloc[train_index], x_st.iloc[test_index]
    y_train, y_test = y_st.iloc[train_index], y_st.iloc[test_index]

    model_st.fit(X_train, y_train)
    y_pred = model_st.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated MAE scores:", mae_scores)
print("Cross-validated R-squared scores:", r2_scores)


# In[719]:




# Load and prepare the dataset
#data = significant_spearman_df_no_collat.drop(columns=['eth_a_vault_cumulative_collateral'])

data_multi = significant_alt_aggregated_all_vaults_spearman.drop(columns=['Altcoin Vault_collateral_usd'])

target = significant_alt_aggregated_all_vaults_spearman['Altcoin Vault_collateral_usd']

# performs .74 '3_m_tbill_yield','eth_a_vault_liquidation_ratio','debt_ratio_Lag_2m','dsr_rate','vc_revenue','dai_abs_deviation_30d_ma','b_s_DAI_30d_rolling_avg_pct_chg','M2V','psm_lifetime_fees'

# Ensure data is in the correct shape for scaling
if data_multi.ndim == 1:
    data_multi = data_multi.values.reshape(-1, 1)  # Reshape data to 2D if it's 1D
else:
    data_multi = data_multi.values

# Scale features and target
scaler_feature = MinMaxScaler(feature_range=(0, 1))
data_scaled_multi = scaler_feature.fit_transform(data_multi)
scaler_target = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Create sequences for LSTM
time_step = 15
X, y = to_sequences(data_scaled_multi, target_scaled, time_step)

# Split into train and test sets
train_size = int(len(X) * 0.67)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Define the LSTM model with dropout and L2 regularization
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(45, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.08)),
    LSTM(20, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.02)),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with early stopping
history = model.fit(
    X_train, y_train, epochs=100, batch_size=40,
    validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping]
)

# Predict on training and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and targets
train_predict_multi_inv = scaler_target.inverse_transform(train_predict)
test_predict_multi_inv = scaler_target.inverse_transform(test_predict)
y_train_multi_inv = scaler_target.inverse_transform(y_train.reshape(-1, 1))
y_test_multi_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_multi_inv, train_predict_multi_inv))
test_rmse = np.sqrt(mean_squared_error(y_test_multi_inv, test_predict_multi_inv))

# Calculate R-squared
train_r2 = r2_score(y_train_multi_inv, train_predict_multi_inv)
test_r2 = r2_score(y_test_multi_inv, test_predict_multi_inv)

# Calculate MAE
train_mae = mean_absolute_error(y_train_multi_inv, train_predict_multi_inv)
test_mae = mean_absolute_error(y_test_multi_inv, test_predict_multi_inv)

# Calculate MAPE
train_mape = mean_absolute_percentage_error(y_train_multi_inv, train_predict_multi_inv)
test_mape = mean_absolute_percentage_error(y_test_multi_inv, test_predict_multi_inv)

# Output the metrics
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
print(f"Train R²: {train_r2}, Test R²: {test_r2}")
print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
print(f"Train MAPE: {train_mape}%, Test MAPE: {test_mape}%")


# ## Stablecoin Vault

# In[720]:


st_aggregated_vault_data[['Stablecoin Vault_collateral_usd','Stablecoin Vault_dai_ceiling']].plot()


# In[721]:


for index, value in correlated_aggregated_stb_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[722]:


x_st = st_aggregated_vault_data['Stablecoin Vault_dai_ceiling'].to_frame()
y_st = st_aggregated_vault_data['Stablecoin Vault_collateral_usd']

X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)  # Assuming 'y' is your target variable




model_st = LinearRegression()
model_st.fit(X_train, y_train)

y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")


# In[723]:


#need to add status for aggregated vaults; dai ceiling is 0 so vault is closed.  model has trouble

st_aggregated_vault_data['Stablecoin Vault_dai_ceiling']


# In[724]:


# Assuming 'st_aggregated_vault_data' contains your dataset

x_st = st_aggregated_vault_data[['Stablecoin Vault_dai_ceiling', 'Stablecoin Vault_dai_ceiling_7d_ma','Stablecoin Vault_collateral_usd_7d_ma', 'BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total','Stablecoin Vault_dart',
       'stETH Vault_collateral_usd % of Total',
       'Stablecoin Vault_collateral_usd % of Total','Stablecoin Vault_collateral_usd % of Total_7d_ma','Stablecoin Vault_collateral_usd % of Total_30d_ma',
       'Altcoin Vault_collateral_usd % of Total','Vaults Total USD Value',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total']]

y_st = st_aggregated_vault_data['Stablecoin Vault_collateral_usd']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)

# Initialize the linear regression model
model_st = LinearRegression()

# Fit the model to the training data
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")

model_st = Ridge(alpha=5000.0)

tscv = TimeSeriesSplit(n_splits=5)
mae_scores, r2_scores = [], []

for train_index, test_index in tscv.split(x_st):
    X_train, X_test = x_st.iloc[train_index], x_st.iloc[test_index]
    y_train, y_test = y_st.iloc[train_index], y_st.iloc[test_index]

    model_st.fit(X_train, y_train)
    y_pred = model_st.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated MAE scores:", mae_scores)
print("Cross-validated R-squared scores:", r2_scores)


# In[725]:






# Load and prepare the dataset
#data = significant_spearman_df_no_collat.drop(columns=['eth_a_vault_cumulative_collateral'])

data_multi = significant_stb_aggregated_all_vaults_spearman.drop(columns=['Stablecoin Vault_collateral_usd'])

target = significant_stb_aggregated_all_vaults_spearman['Stablecoin Vault_collateral_usd']

# performs .74 '3_m_tbill_yield','eth_a_vault_liquidation_ratio','debt_ratio_Lag_2m','dsr_rate','vc_revenue','dai_abs_deviation_30d_ma','b_s_DAI_30d_rolling_avg_pct_chg','M2V','psm_lifetime_fees'

# Ensure data is in the correct shape for scaling
if data_multi.ndim == 1:
    data_multi = data_multi.values.reshape(-1, 1)  # Reshape data to 2D if it's 1D
else:
    data_multi = data_multi.values

# Scale features and target
scaler_feature = MinMaxScaler(feature_range=(0, 1))
data_scaled_multi = scaler_feature.fit_transform(data_multi)
scaler_target = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Create sequences for LSTM
time_step = 15
X, y = to_sequences(data_scaled_multi, target_scaled, time_step)

# Split into train and test sets
train_size = int(len(X) * 0.67)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Define the LSTM model with dropout and L2 regularization
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(45, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.08)),
    LSTM(20, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.02)),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with early stopping
history = model.fit(
    X_train, y_train, epochs=100, batch_size=40,
    validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping]
)

# Predict on training and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and targets
train_predict_multi_inv = scaler_target.inverse_transform(train_predict)
test_predict_multi_inv = scaler_target.inverse_transform(test_predict)
y_train_multi_inv = scaler_target.inverse_transform(y_train.reshape(-1, 1))
y_test_multi_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_multi_inv, train_predict_multi_inv))
test_rmse = np.sqrt(mean_squared_error(y_test_multi_inv, test_predict_multi_inv))

# Calculate R-squared
train_r2 = r2_score(y_train_multi_inv, train_predict_multi_inv)
test_r2 = r2_score(y_test_multi_inv, test_predict_multi_inv)

# Calculate MAE
train_mae = mean_absolute_error(y_train_multi_inv, train_predict_multi_inv)
test_mae = mean_absolute_error(y_test_multi_inv, test_predict_multi_inv)

# Calculate MAPE
train_mape = mean_absolute_percentage_error(y_train_multi_inv, train_predict_multi_inv)
test_mape = mean_absolute_percentage_error(y_test_multi_inv, test_predict_multi_inv)

# Output the metrics
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
print(f"Train R²: {train_r2}, Test R²: {test_r2}")
print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
print(f"Train MAPE: {train_mape}%, Test MAPE: {test_mape}%")


# ## LP Vault

# In[726]:


st_aggregated_vault_data[['LP Vault_collateral_usd','LP Vault_dai_ceiling']].plot()


# In[727]:


for index, value in correlated_aggregated_lp_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[728]:


x_st = st_aggregated_vault_data['LP Vault_dai_ceiling'].to_frame()
y_st = st_aggregated_vault_data['LP Vault_collateral_usd']

X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)  # Assuming 'y' is your target variable




model_st = LinearRegression()
model_st.fit(X_train, y_train)

y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")


# In[729]:


st_aggregated_vault_data[['LP Vault_collateral_usd','LP Vault_dai_ceiling']]


# In[730]:


# Assuming 'st_aggregated_vault_data' contains your dataset

x_st = st_aggregated_vault_data[['LP Vault_dai_ceiling','LP Vault_dai_ceiling_7d_ma','LP Vault_collateral_usd_7d_ma','LP Vault_collateral_usd_30d_ma', 'BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total',
       'stETH Vault_collateral_usd % of Total',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total',
       'LP Vault_collateral_usd % of Total','LP Vault_collateral_usd % of Total_7d_ma','LP Vault_collateral_usd % of Total_30d_ma',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total']]

y_st = st_aggregated_vault_data['LP Vault_collateral_usd']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)

# Initialize the linear regression model
model_st = LinearRegression()

# Fit the model to the training data
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")

model_st = Ridge(alpha=10000.0)

tscv = TimeSeriesSplit(n_splits=5)
mae_scores, r2_scores = [], []

for train_index, test_index in tscv.split(x_st):
    X_train, X_test = x_st.iloc[train_index], x_st.iloc[test_index]
    y_train, y_test = y_st.iloc[train_index], y_st.iloc[test_index]

    model_st.fit(X_train, y_train)
    y_pred = model_st.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated MAE scores:", mae_scores)
print("Cross-validated R-squared scores:", r2_scores)


st_aggregated_vault_data[['RWA Vault_collateral_usd','RWA Vault_dai_ceiling']].plot()


# In[732]:


for index, value in correlated_aggregated_rwa_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[733]:


x_st = st_aggregated_vault_data['RWA Vault_dai_ceiling'].to_frame()
y_st = st_aggregated_vault_data['RWA Vault_collateral_usd']

X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)  # Assuming 'y' is your target variable

model_st = LinearRegression()
model_st.fit(X_train, y_train)

y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")


# In[734]:


# Assuming 'st_aggregated_vault_data' contains your dataset

x_st = st_aggregated_vault_data[['RWA Vault_dai_ceiling','RWA Vault_dai_ceiling_7d_ma', 'RWA Vault_collateral_usd_7d_ma', 'BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total','RWA Vault_dart',
       'stETH Vault_collateral_usd % of Total',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total','RWA Vault_collateral_usd % of Total_7d_ma','RWA Vault_collateral_usd % of Total_30d_ma',
       'PSM Vault_collateral_usd % of Total']]

y_st = st_aggregated_vault_data['RWA Vault_collateral_usd']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)

# Initialize the linear regression model
model_st = LinearRegression()

# Fit the model to the training data
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")

model_st = Ridge(alpha=4000.0)

tscv = TimeSeriesSplit(n_splits=5)
mae_scores, r2_scores = [], []

for train_index, test_index in tscv.split(x_st):
    X_train, X_test = x_st.iloc[train_index], x_st.iloc[test_index]
    y_train, y_test = y_st.iloc[train_index], y_st.iloc[test_index]

    model_st.fit(X_train, y_train)
    y_pred = model_st.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated MAE scores:", mae_scores)
print("Cross-validated R-squared scores:", r2_scores)


# In[735]:






# Load and prepare the dataset
#data = significant_spearman_df_no_collat.drop(columns=['eth_a_vault_cumulative_collateral'])

data_multi = st_aggregated_vault_data[['RWA Vault_dai_ceiling', 'BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total',
       'stETH Vault_collateral_usd % of Total',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total']]

target = st_aggregated_vault_data['RWA Vault_collateral_usd']

# performs .74 '3_m_tbill_yield','eth_a_vault_liquidation_ratio','debt_ratio_Lag_2m','dsr_rate','vc_revenue','dai_abs_deviation_30d_ma','b_s_DAI_30d_rolling_avg_pct_chg','M2V','psm_lifetime_fees'

# Ensure data is in the correct shape for scaling
if data_multi.ndim == 1:
    data_multi = data_multi.values.reshape(-1, 1)  # Reshape data to 2D if it's 1D
else:
    data_multi = data_multi.values

# Scale features and target
scaler_feature = MinMaxScaler(feature_range=(0, 1))
data_scaled_multi = scaler_feature.fit_transform(data_multi)
scaler_target = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Create sequences for LSTM
time_step = 15
X, y = to_sequences(data_scaled_multi, target_scaled, time_step)

# Split into train and test sets
train_size = int(len(X) * 0.67)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Define the LSTM model with dropout and L2 regularization
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(45, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.08)),
    LSTM(20, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.02)),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with early stopping
history = model.fit(
    X_train, y_train, epochs=100, batch_size=40,
    validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping]
)

# Predict on training and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and targets
train_predict_multi_inv = scaler_target.inverse_transform(train_predict)
test_predict_multi_inv = scaler_target.inverse_transform(test_predict)
y_train_multi_inv = scaler_target.inverse_transform(y_train.reshape(-1, 1))
y_test_multi_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_multi_inv, train_predict_multi_inv))
test_rmse = np.sqrt(mean_squared_error(y_test_multi_inv, test_predict_multi_inv))

# Calculate R-squared
train_r2 = r2_score(y_train_multi_inv, train_predict_multi_inv)
test_r2 = r2_score(y_test_multi_inv, test_predict_multi_inv)

# Calculate MAE
train_mae = mean_absolute_error(y_train_multi_inv, train_predict_multi_inv)
test_mae = mean_absolute_error(y_test_multi_inv, test_predict_multi_inv)

# Calculate MAPE
train_mape = mean_absolute_percentage_error(y_train_multi_inv, train_predict_multi_inv)
test_mape = mean_absolute_percentage_error(y_test_multi_inv, test_predict_multi_inv)

# Output the metrics
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
print(f"Train R²: {train_r2}, Test R²: {test_r2}")
print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
print(f"Train MAPE: {train_mape}%, Test MAPE: {test_mape}%")


# ## PSM Vault

# In[736]:


st_aggregated_vault_data[['PSM Vault_collateral_usd','PSM Vault_dai_ceiling']].plot()


# In[737]:


for index, value in correlated_aggregated_psm_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[738]:


x_st = st_aggregated_vault_data['PSM Vault_dai_ceiling'].to_frame()
y_st = st_aggregated_vault_data['PSM Vault_collateral_usd']

X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)  # Assuming 'y' is your target variable




model_st = LinearRegression()
model_st.fit(X_train, y_train)

y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")


# In[739]:


st_aggregated_vault_data[['stETH Vault_collateral_usd % of Total','PSM Vault_collateral_usd % of Total']].plot()


# In[740]:


# Assuming 'st_aggregated_vault_data' contains your dataset

x_st = st_aggregated_vault_data[['PSM Vault_dai_ceiling','PSM Vault_dai_ceiling_7d_ma',
                                 
                                 'PSM Vault_collateral_usd_7d_ma',
                            
                                 'BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total',
       'stETH Vault_collateral_usd % of Total','where_is_dai_Bridge',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total','PSM Vault_collateral_usd % of Total_7d_ma','PSM Vault_collateral_usd % of Total_30d_ma']]

y_st = st_aggregated_vault_data['PSM Vault_collateral_usd']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_st, y_st, test_size=0.2, random_state=42)

# Initialize the linear regression model
model_st = LinearRegression()

# Fit the model to the training data
model_st.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model_st.predict(X_train)
y_test_pred = model_st.predict(X_test)

# Calculate MAE and R-squared for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate MAE and R-squared for test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train MAE:", mae_train)
print("Train R-squared:", r2_train)
print("Test MAE:", mae_test)
print("Test R-squared:", r2_test)

plot_single_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")
# Initialize the linear regression model
model_st = Ridge(alpha=200.0)

tscv = TimeSeriesSplit(n_splits=5)
mae_scores, r2_scores = [], []

for train_index, test_index in tscv.split(x_st):
    X_train, X_test = x_st.iloc[train_index], x_st.iloc[test_index]
    y_train, y_test = y_st.iloc[train_index], y_st.iloc[test_index]

    model_st.fit(X_train, y_train)
    y_pred = model_st.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated MAE scores:", mae_scores)
print("Cross-validated R-squared scores:", r2_scores)


# # LSTM Multivariate Cross Validation

# In[741]:


list(dataset_no_nan.columns)


# In[742]:


dataset_no_nan['eth_a_vault_cumulative_collateral'].plot()
