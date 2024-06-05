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

X = st_aggregated_vault_data[['ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','PSM Vault_dai_ceiling',
'PSM Vault_collateral_usd_7d_ma','BTC Vault_collateral_usd_7d_ma','ETH Vault_collateral_usd_7d_ma','stETH Vault_collateral_usd_7d_ma','Altcoin Vault_collateral_usd_7d_ma','LP Vault_collateral_usd_7d_ma','Stablecoin Vault_collateral_usd_7d_ma','RWA Vault_collateral_usd_7d_ma',
'BTC Vault_market_price', 'ETH Vault_market_price', 'stETH Vault_market_price', 'Stablecoin Vault_market_price', 'Altcoin Vault_market_price', 'LP Vault_market_price', 'PSM Vault_market_price', 'effective_funds_rate',
'M1V', 'WM2NS', 'fed_reverse_repo',
'ETH Vault_liquidation_ratio', 'BTC Vault_liquidation_ratio', 'stETH Vault_liquidation_ratio', 'Altcoin Vault_liquidation_ratio', 'Stablecoin Vault_liquidation_ratio', 'LP Vault_liquidation_ratio', 'PSM Vault_liquidation_ratio',
'BTC Vault_collateral_usd % of Total', 'ETH Vault_collateral_usd % of Total', 'stETH Vault_collateral_usd % of Total', 'where_is_dai_Bridge',
'Stablecoin Vault_collateral_usd % of Total', 'Altcoin Vault_collateral_usd % of Total', 'LP Vault_collateral_usd % of Total',
'RWA Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'ETH Vault_collateral_usd % of Total_7d_ma', 'ETH Vault_collateral_usd % of Total_30d_ma',
'stETH Vault_collateral_usd % of Total_7d_ma', 'stETH Vault_collateral_usd % of Total_30d_ma',
'BTC Vault_collateral_usd % of Total_7d_ma', 'BTC Vault_collateral_usd % of Total_30d_ma',
'Altcoin Vault_collateral_usd % of Total_7d_ma', 'Altcoin Vault_collateral_usd % of Total_30d_ma',
'Stablecoin Vault_collateral_usd % of Total_7d_ma', 'Stablecoin Vault_collateral_usd % of Total_30d_ma',
'LP Vault_collateral_usd % of Total_7d_ma', 'LP Vault_collateral_usd % of Total_30d_ma',
'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma', 'Vaults Total USD Value',
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling',
'RWA Vault_collateral_usd', 'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma',
'where_is_dai_Bridge', 'dai_market_Volume_30d_ma', 'dai_market_Volume_7d_ma','eth_market_Close_7d_ma','eth_market_Volume_30d_ma','btc_market_Close_7d_ma','btc_market_Volume_30d_ma',
'btc_market_Volume_30d_ma','eth_market_Volume_30d_ma']]

y = st_aggregated_vault_data[['ETH Vault_collateral_usd','stETH Vault_collateral_usd','BTC Vault_collateral_usd','Altcoin Vault_collateral_usd','Stablecoin Vault_collateral_usd','LP Vault_collateral_usd','PSM Vault_collateral_usd']] #maybe don't forecast psm since maybe cant capture usd vault 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MultiOutput Regressor with a RandomForest
model = MultiOutputRegressor(Ridge(alpha=300))

# Train the model
model.fit(X_train, y_train)




# Assuming 'model', 'X_train', 'X_test', 'y_train', 'y_test' are already defined and model is trained

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute R² score for each target on the training set
r2_scores_train = r2_score(y_train, y_train_pred, multioutput='raw_values')
print("R² on the training set:", r2_scores_train)

# Compute R² score for each target on the testing set
r2_scores_test = r2_score(y_test, y_test_pred, multioutput='raw_values')
print("R² on the testing set:", r2_scores_test)

# Evaluate the model
error = mean_absolute_error(y_test, y_test_pred, multioutput='raw_values')
print("MAE on test set:", error)

error = mean_absolute_error(y_train, y_train_pred, multioutput='raw_values')
print("MAE on test set:", error)

plot_regression_time_series(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Regression Time Series Comparison")
# Initialize the linear regression model


# ### Regressor Feature Ranking

# 
# 
# X = st_aggregated_vault_data.drop(columns=['ETH Vault_collateral_usd','stETH Vault_collateral_usd','BTC Vault_collateral_usd','Altcoin Vault_collateral_usd','Stablecoin Vault_collateral_usd','LP Vault_collateral_usd','PSM Vault_collateral_usd'])
# 
# y = st_aggregated_vault_data[['ETH Vault_collateral_usd','stETH Vault_collateral_usd','BTC Vault_collateral_usd','Altcoin Vault_collateral_usd','Stablecoin Vault_collateral_usd','LP Vault_collateral_usd','PSM Vault_collateral_usd']] #maybe don't forecast psm since maybe cant capture usd vault 
# 
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Assuming X_train and y_train are already defined
# forest = RandomForestRegressor(n_estimators=100)
# forest.fit(X_train, y_train)
# 
# # Get feature importances
# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]
# 
# # Print the feature ranking
# print("Feature ranking:")
# for f in range(X_train.shape[1]):
#     print(f"{f + 1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]})")
# 

# # Absolute threshold
# threshold = 0  # Keep features with importance greater than 0.01
# important_features = X_train.columns[importances > threshold]
# print("Important features:", important_features)
# 

# ### PCA

# 
# 
# # Create a pipeline that includes PCA, scaling, and regression
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('pca', PCA(n_components=0.95)),  # Retain 95% of variance
#     ('regressor', MultiTaskLassoCV(cv=5, random_state=0))
# ])
# 
# # Fit the pipeline
# pipeline.fit(X_train, y_train)
# 
# # You can access individual steps if you need, for example:
# print("PCA Components Shape:", pipeline.named_steps['pca'].components_.shape)
# print("Regression Coefficients:", pipeline.named_steps['regressor'].coef_)
# 

# ### variance_inflation_factor

# ### Multiregressor CV

# In[827]:


# Assuming 'x_st' has a datetime index or similar for plotting

    
model_st = Ridge(alpha=100000.0)

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



model_st = Ridge(alpha=100000.0)
tscv = TimeSeriesSplit(n_splits=5)

# Call the function to plot CV results
plot_cv_results(x_st, y_st, model_st, tscv)


# In[828]:






# ### Nan checker

# In[829]:


X = st_aggregated_vault_data[['ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','PSM Vault_dai_ceiling',
'PSM Vault_collateral_usd_7d_ma','BTC Vault_collateral_usd_7d_ma','ETH Vault_collateral_usd_7d_ma','stETH Vault_collateral_usd_7d_ma','Altcoin Vault_collateral_usd_7d_ma','LP Vault_collateral_usd_7d_ma','Stablecoin Vault_collateral_usd_7d_ma','RWA Vault_collateral_usd_7d_ma',
'BTC Vault_market_price', 'ETH Vault_market_price', 'stETH Vault_market_price', 'Stablecoin Vault_market_price', 'Altcoin Vault_market_price', 'LP Vault_market_price', 'PSM Vault_market_price', 'effective_funds_rate',
'M1V', 'WM2NS', 'fed_reverse_repo',
'ETH Vault_liquidation_ratio', 'BTC Vault_liquidation_ratio', 'stETH Vault_liquidation_ratio', 'Altcoin Vault_liquidation_ratio', 'Stablecoin Vault_liquidation_ratio', 'LP Vault_liquidation_ratio', 'PSM Vault_liquidation_ratio',
'BTC Vault_collateral_usd % of Total', 'ETH Vault_collateral_usd % of Total', 'stETH Vault_collateral_usd % of Total', 'where_is_dai_Bridge',
'Stablecoin Vault_collateral_usd % of Total', 'Altcoin Vault_collateral_usd % of Total', 'LP Vault_collateral_usd % of Total',
'RWA Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'ETH Vault_collateral_usd % of Total_7d_ma', 'ETH Vault_collateral_usd % of Total_30d_ma',
'stETH Vault_collateral_usd % of Total_7d_ma', 'stETH Vault_collateral_usd % of Total_30d_ma',
'BTC Vault_collateral_usd % of Total_7d_ma', 'BTC Vault_collateral_usd % of Total_30d_ma',
'Altcoin Vault_collateral_usd % of Total_7d_ma', 'Altcoin Vault_collateral_usd % of Total_30d_ma',
'Stablecoin Vault_collateral_usd % of Total_7d_ma', 'Stablecoin Vault_collateral_usd % of Total_30d_ma',
'LP Vault_collateral_usd % of Total_7d_ma', 'LP Vault_collateral_usd % of Total_30d_ma',
'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma', 'Vaults Total USD Value',
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling',
'RWA Vault_collateral_usd', 'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma',
'where_is_dai_Bridge', 'dai_market_Volume_30d_ma', 'dai_market_Volume_7d_ma','eth_market_Close_7d_ma','eth_market_Volume_30d_ma','btc_market_Close_7d_ma','btc_market_Volume_30d_ma',
'btc_market_Volume_30d_ma','eth_market_Volume_30d_ma','Altcoin Vault_prev_dai_ceiling_7d_ma']]
# Check for NaNs across the DataFrame
na_columns = X.isna().any()

# Columns that contain at least one NaN value
na_cols = na_columns[na_columns].index.tolist()

# Display columns with NaNs and the rows in those columns that contain NaNs
for col in na_cols:
    # Display rows where NaN values are found in the specific column
    print(f"NaNs in column '{col}':")
    display(X[X[col].isna()])


# ### Best Regressor on 20% Test

# In[830]:


X = state_space[['ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','PSM Vault_dai_ceiling',
'PSM Vault_collateral_usd_7d_ma','BTC Vault_collateral_usd_7d_ma','ETH Vault_collateral_usd_7d_ma','stETH Vault_collateral_usd_7d_ma','Altcoin Vault_collateral_usd_7d_ma','LP Vault_collateral_usd_7d_ma','Stablecoin Vault_collateral_usd_7d_ma','RWA Vault_collateral_usd_7d_ma',
'BTC Vault_market_price', 'ETH Vault_market_price', 'stETH Vault_market_price', 'Stablecoin Vault_market_price', 'Altcoin Vault_market_price', 'LP Vault_market_price', 'effective_funds_rate',
'M1V', 'WM2NS', 'fed_reverse_repo',
'ETH Vault_liquidation_ratio', 'BTC Vault_liquidation_ratio', 'stETH Vault_liquidation_ratio', 'Altcoin Vault_liquidation_ratio', 'Stablecoin Vault_liquidation_ratio', 'LP Vault_liquidation_ratio', 'PSM Vault_liquidation_ratio',
'BTC Vault_collateral_usd % of Total', 'ETH Vault_collateral_usd % of Total', 'stETH Vault_collateral_usd % of Total', 'where_is_dai_Bridge',
'Stablecoin Vault_collateral_usd % of Total', 'Altcoin Vault_collateral_usd % of Total', 'LP Vault_collateral_usd % of Total',
'RWA Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'ETH Vault_collateral_usd % of Total_7d_ma', 'ETH Vault_collateral_usd % of Total_30d_ma',
'stETH Vault_collateral_usd % of Total_7d_ma', 'stETH Vault_collateral_usd % of Total_30d_ma',
'BTC Vault_collateral_usd % of Total_7d_ma', 'BTC Vault_collateral_usd % of Total_30d_ma',
'Altcoin Vault_collateral_usd % of Total_7d_ma', 'Altcoin Vault_collateral_usd % of Total_30d_ma',
'Stablecoin Vault_collateral_usd % of Total_7d_ma', 'Stablecoin Vault_collateral_usd % of Total_30d_ma',
'LP Vault_collateral_usd % of Total_7d_ma', 'LP Vault_collateral_usd % of Total_30d_ma',
'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma', 'Vaults Total USD Value',
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling',
'RWA Vault_collateral_usd', 'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma',
'where_is_dai_Bridge', 'dai_market_Volume_30d_ma', 'dai_market_Volume_7d_ma','eth_market_Close_7d_ma','eth_market_Volume_30d_ma','btc_market_Close_7d_ma','btc_market_Volume_30d_ma',
'btc_market_Volume_30d_ma','eth_market_Volume_30d_ma']]

y = state_space[['ETH Vault_collateral_usd','stETH Vault_collateral_usd','BTC Vault_collateral_usd','Altcoin Vault_collateral_usd','Stablecoin Vault_collateral_usd','LP Vault_collateral_usd','PSM Vault_collateral_usd']] #maybe don't forecast psm since maybe cant capture usd vault 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

final_model = MultiOutputRegressor(Ridge(alpha=300))
final_model.fit(X_train, y_train)

# Predict on test data
y_pred_test = final_model.predict(X_test)

# Evaluate the final model on test data
final_test_mae = mean_absolute_error(y_test, y_pred_test, multioutput='raw_values')
final_test_r2 = r2_score(y_test, y_pred_test, multioutput='raw_values')

print("Final Test MAE:", final_test_mae)
print("Final Test R²:", final_test_r2)

plot_multioutput_cv_results(X, y, n_splits=5, alpha=300)


# ### Best Regressor on 80% Test

# In[831]:


st_aggregated_vault_data[['mcap_market_cap', 'mcap_total_volume']]


# In[832]:


X = st_aggregated_vault_data[['ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','PSM Vault_dai_ceiling',
'mcap_total_volume', 'defi_apy_medianAPY', 'defi_apy_avg7day', 'dpi_market_Volume', 
'BTC Vault_market_price', 'ETH Vault_market_price', 'stETH Vault_market_price', 'Stablecoin Vault_market_price', 'Altcoin Vault_market_price', 'LP Vault_market_price', 'effective_funds_rate',
'M1V', 'WM2NS', 'fed_reverse_repo',
'ETH Vault_liquidation_ratio', 'BTC Vault_liquidation_ratio', 'stETH Vault_liquidation_ratio', 'Altcoin Vault_liquidation_ratio', 'Stablecoin Vault_liquidation_ratio', 'LP Vault_liquidation_ratio', 
'BTC Vault_collateral_usd % of Total', 'ETH Vault_collateral_usd % of Total', 'stETH Vault_collateral_usd % of Total', 'where_is_dai_Bridge',
'Stablecoin Vault_collateral_usd % of Total', 'Altcoin Vault_collateral_usd % of Total', 'LP Vault_collateral_usd % of Total',
'RWA Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'ETH Vault_collateral_usd % of Total_7d_ma', 'ETH Vault_collateral_usd % of Total_30d_ma',
'stETH Vault_collateral_usd % of Total_7d_ma', 'stETH Vault_collateral_usd % of Total_30d_ma',
'BTC Vault_collateral_usd % of Total_7d_ma', 'BTC Vault_collateral_usd % of Total_30d_ma',
'Altcoin Vault_collateral_usd % of Total_7d_ma', 'Altcoin Vault_collateral_usd % of Total_30d_ma',
'Stablecoin Vault_collateral_usd % of Total_7d_ma', 'Stablecoin Vault_collateral_usd % of Total_30d_ma',
'LP Vault_collateral_usd % of Total_7d_ma', 'LP Vault_collateral_usd % of Total_30d_ma',
'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma', 'Vaults Total USD Value',
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling',
'RWA Vault_collateral_usd', 'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma',
'where_is_dai_Bridge', 'dai_market_Volume_30d_ma', 'dai_market_Volume_7d_ma','eth_market_Close_7d_ma','eth_market_Volume_30d_ma','btc_market_Close_7d_ma','btc_market_Volume_30d_ma',
'btc_market_Volume_30d_ma','LP Vault_dai_floor_90d_ma_pct_change']]

y = st_aggregated_vault_data[['ETH Vault_collateral_usd','stETH Vault_collateral_usd','BTC Vault_collateral_usd','Altcoin Vault_collateral_usd','Stablecoin Vault_collateral_usd','LP Vault_collateral_usd','PSM Vault_collateral_usd']] #maybe don't forecast psm since maybe cant capture usd vault 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

final_model = MultiOutputRegressor(Ridge(alpha=300))
final_model.fit(X_train, y_train)

# Predict on test data
y_pred_test = final_model.predict(X_test)

# Evaluate the final model on test data
final_test_mae = mean_absolute_error(y_test, y_pred_test, multioutput='raw_values')
final_test_r2 = r2_score(y_test, y_pred_test, multioutput='raw_values')

print("Final Test MAE:", final_test_mae)
print("Final Test R²:", final_test_r2)

plot_multioutput_cv_results(X, y, n_splits=5, alpha=400)