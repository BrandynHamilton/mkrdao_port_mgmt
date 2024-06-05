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

correlations = numeric_dataset.corr()
aggregated_correlations = aggregated_vault_data.corr()


# In[627]:


agg_params = [
    'BTC Vault_dai_ceiling', 'BTC Vault_dai_floor', 'BTC Vault_liquidation_penalty', 'BTC Vault_liquidation_ratio', 'BTC Vault_annualized stability fee',
    'ETH Vault_dai_ceiling', 'ETH Vault_dai_floor', 'ETH Vault_liquidation_penalty', 'ETH Vault_liquidation_ratio', 'ETH Vault_annualized stability fee',
    'stETH Vault_dai_ceiling', 'stETH Vault_dai_floor', 'stETH Vault_liquidation_penalty', 'stETH Vault_liquidation_ratio', 'stETH Vault_annualized stability fee',
    'Stablecoin Vault_dai_ceiling', 'Stablecoin Vault_dai_floor', 'Stablecoin Vault_liquidation_penalty', 'Stablecoin Vault_liquidation_ratio', 'Stablecoin Vault_annualized stability fee',
    'Altcoin Vault_dai_ceiling', 'Altcoin Vault_dai_floor', 'Altcoin Vault_liquidation_penalty', 'Altcoin Vault_liquidation_ratio', 'Altcoin Vault_annualized stability fee',
    'LP Vault_dai_ceiling', 'LP Vault_dai_floor', 'LP Vault_liquidation_penalty', 'LP Vault_liquidation_ratio', 'LP Vault_annualized stability fee',
    'RWA Vault_dai_ceiling', 'RWA Vault_dai_floor', 'RWA Vault_liquidation_penalty', 'RWA Vault_liquidation_ratio', 'RWA Vault_annualized stability fee',
    'PSM Vault_dai_ceiling', 'PSM Vault_dai_floor', 'PSM Vault_liquidation_penalty', 'PSM Vault_liquidation_ratio', 'PSM Vault_annualized stability fee'
]

eth_params = ['ETH Vault_dai_ceiling', 'ETH Vault_dai_floor', 'ETH Vault_liquidation_penalty', 'ETH Vault_liquidation_ratio', 'ETH Vault_annualized stability fee']
btc_params = ['BTC Vault_dai_ceiling', 'BTC Vault_dai_floor', 'BTC Vault_liquidation_penalty', 'BTC Vault_liquidation_ratio', 'BTC Vault_annualized stability fee']
steth_params = ['stETH Vault_dai_ceiling', 'stETH Vault_dai_floor', 'stETH Vault_liquidation_penalty', 'stETH Vault_liquidation_ratio', 'stETH Vault_annualized stability fee']
alt_params = ['Altcoin Vault_dai_ceiling', 'Altcoin Vault_dai_floor', 'Altcoin Vault_liquidation_penalty', 'Altcoin Vault_liquidation_ratio', 'Altcoin Vault_annualized stability fee']
stablecoin_params = ['Stablecoin Vault_dai_ceiling', 'Stablecoin Vault_dai_floor', 'Stablecoin Vault_liquidation_penalty', 'Stablecoin Vault_liquidation_ratio', 'Stablecoin Vault_annualized stability fee']
lp_params = ['LP Vault_dai_ceiling', 'LP Vault_dai_floor', 'LP Vault_liquidation_penalty', 'LP Vault_liquidation_ratio', 'LP Vault_annualized stability fee']
rwa_params = ['RWA Vault_dai_ceiling', 'RWA Vault_dai_floor', 'RWA Vault_liquidation_penalty', 'RWA Vault_liquidation_ratio', 'RWA Vault_annualized stability fee']
psm_params = ['PSM Vault_dai_ceiling', 'PSM Vault_dai_floor', 'PSM Vault_liquidation_penalty', 'PSM Vault_liquidation_ratio', 'PSM Vault_annualized stability fee']


# In[628]:


# Assuming 'df' is your DataFrame
non_numeric_columns = dataset_no_nan.select_dtypes(exclude=[np.number])

# This will show you the columns that do not contain numeric data
print(non_numeric_columns.columns)


# BTC Vault_collateral_usd          
# ETH Vault_collateral_usd          
# stETH Vault_collateral_usd        
# Stablecoin Vault_collateral_usd   
# Altcoin Vault_collateral_usd      
# LP Vault_collateral_usd           
# RWA Vault_collateral_usd          
# PSM Vault_collateral_usd          

# ## Aggregated Vaults

# In[629]:


correlated_aggregated_eth_collateral_usd = aggregated_correlations['ETH Vault_collateral_usd']
correlated_aggregated_eth_collateral_usd_sorted = correlated_aggregated_eth_collateral_usd.sort_values(ascending=False)
for index, value in correlated_aggregated_eth_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[630]:


for index, value in correlated_aggregated_eth_collateral_usd_sorted.items():
     if index in eth_params:
        print(f"{index:50} {value}")


# In[631]:


correlated_aggregated_steth_collateral_usd = aggregated_correlations['stETH Vault_collateral_usd']
correlated_aggregated_steth_collateral_usd_sorted = correlated_aggregated_steth_collateral_usd.sort_values(ascending=False)
for index, value in correlated_aggregated_steth_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[632]:


for index, value in correlated_aggregated_steth_collateral_usd_sorted.items():
    if index in steth_params:
        print(f"{index:50} {value}")


# In[633]:


correlated_aggregated_wbtc_collateral_usd = aggregated_correlations['BTC Vault_collateral_usd']
correlated_aggregated_wbtc_collateral_usd_sorted = correlated_aggregated_wbtc_collateral_usd.sort_values(ascending=False)
for index, value in correlated_aggregated_wbtc_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[634]:


for index, value in correlated_aggregated_wbtc_collateral_usd_sorted.items():
    if index in btc_params:
        print(f"{index:50} {value}")


# In[635]:


correlated_aggregated_stb_collateral_usd = aggregated_correlations['Stablecoin Vault_collateral_usd']
correlated_aggregated_stb_collateral_usd_sorted = correlated_aggregated_stb_collateral_usd.sort_values(ascending=False)
for index, value in correlated_aggregated_stb_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[636]:


for index, value in correlated_aggregated_stb_collateral_usd_sorted.items():
    if index in stablecoin_params:
        print(f"{index:50} {value}")


# In[637]:


correlated_aggregated_alt_collateral_usd = aggregated_correlations['Altcoin Vault_collateral_usd']
correlated_aggregated_alt_collateral_usd_sorted = correlated_aggregated_alt_collateral_usd.sort_values(ascending=False)
for index, value in correlated_aggregated_alt_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[638]:


for index, value in correlated_aggregated_alt_collateral_usd_sorted.items():
    if index in alt_params:
        print(f"{index:50} {value}")


# In[639]:


correlated_aggregated_lp_collateral_usd = aggregated_correlations['LP Vault_collateral_usd']
correlated_aggregated_lp_collateral_usd_sorted = correlated_aggregated_lp_collateral_usd.sort_values(ascending=False)
for index, value in correlated_aggregated_lp_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[640]:


for index, value in correlated_aggregated_lp_collateral_usd_sorted.items():
    if index in lp_params:
        print(f"{index:50} {value}")


# In[641]:


correlated_aggregated_rwa_collateral_usd = aggregated_correlations['RWA Vault_collateral_usd']
correlated_aggregated_rwa_collateral_usd_sorted = correlated_aggregated_rwa_collateral_usd.sort_values(ascending=False)
for index, value in correlated_aggregated_rwa_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[642]:


for index, value in correlated_aggregated_rwa_collateral_usd_sorted.items():
    if index in rwa_params:
        print(f"{index:50} {value}")


# In[643]:


correlated_aggregated_psm_collateral_usd = aggregated_correlations['PSM Vault_collateral_usd']
correlated_aggregated_psm_collateral_usd_sorted = correlated_aggregated_psm_collateral_usd.sort_values(ascending=False)
for index, value in correlated_aggregated_psm_collateral_usd_sorted.items():
    print(f"{index:50} {value}")


# In[644]:


for index, value in correlated_aggregated_psm_collateral_usd_sorted.items():
    if index in psm_params:
        print(f"{index:50} {value}")


# ## Individual Vaults

# In[645]:


correlations = numeric_dataset.corr()

collateral_target_correlations = correlations['eth_a_vault_cumulative_collateral']
collateral_sorted_correlations = collateral_target_correlations.sort_values(ascending=False)

# Display the sorted correlations
print(collateral_sorted_correlations)


# In[646]:


for index, value in collateral_sorted_correlations.items():
    print(f"{index:50} {value}")


# In[647]:


# Select correlations of all features with the target variable, excluding the target variable itself
etha_target_correlations = correlations['eth_a_vault_liquidation_ratio'].drop('eth_a_vault_liquidation_ratio')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
etha_sorted_correlations = etha_target_correlations.sort_values(ascending=False)

for index, value in etha_sorted_correlations.items():
    print(f"{index:50} {value}")


# In[648]:


# Select correlations of all features with the target variable, excluding the target variable itself
dsr_target_correlations = correlations['dsr_rate'].drop('dsr_rate')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
dsr_sorted_correlations = dsr_target_correlations.sort_values(ascending=False)

for index, value in dsr_sorted_correlations.items():
    print(f"{index:50} {value}")


# In[649]:


# Select correlations of all features with the target variable, excluding the target variable itself
stability_fee_target_correlations = correlations['eth_a_vault_annualized stability fee'].drop('eth_a_vault_annualized stability fee')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
stability_fee_target_correlations = stability_fee_target_correlations.sort_values(ascending=False)

for index, value in stability_fee_target_correlations.items():
    print(f"{index:50} {value}")


# In[650]:


# Select correlations of all features with the target variable, excluding the target variable itself
safety_ratio_target_correlations = correlations['eth_a_vault_safety_collateral_ratio'].drop('eth_a_vault_safety_collateral_ratio')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
safety_ratio_target_correlations = safety_ratio_target_correlations.sort_values(ascending=False)

for index, value in safety_ratio_target_correlations.items():
    print(f"{index:50} {value}")


# In[651]:


# Select correlations of all features with the target variable, excluding the target variable itself
dai_target_correlations = correlations['dai_total_balance'].drop('dai_total_balance')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
dai_sorted_correlations = dai_target_correlations.sort_values(ascending=False)

for index, value in dai_sorted_correlations.items():
    print(f"{index:50} {value}")


# In[652]:


# Select correlations of all features with the target variable, excluding the target variable itself
sb_target_correlations = correlations['daily_surplus_buffer'].drop('daily_surplus_buffer')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
sb_sorted_correlations = sb_target_correlations.sort_values(ascending=False)

for index, value in sb_sorted_correlations.items():
    print(f"{index:50} {value}")


# In[653]:


# Select correlations of all features with the target variable, excluding the target variable itself
ni_target_correlations = correlations['1.9 - Net Income'].drop('1.9 - Net Income')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
ni_sorted_correlations = ni_target_correlations.sort_values(ascending=False)

for index, value in ni_sorted_correlations.items():
    print(f"{index:50} {value}")


# In[654]:


# Select correlations of all features with the target variable, excluding the target variable itself
dai_market_Volume_target_correlations = correlations['dai_market_Volume'].drop('dai_market_Volume')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
dai_market_Volume_sorted_correlations = dai_market_Volume_target_correlations.sort_values(ascending=False)

for index, value in dai_market_Volume_sorted_correlations.items():
    print(f"{index:50} {value}")


# In[655]:


# Select correlations of all features with the target variable, excluding the target variable itself
dai_treasury_target_correlations = correlations['where_is_dai_Treasury'].drop('where_is_dai_Treasury')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
dai_t_sorted_correlations = dai_treasury_target_correlations.sort_values(ascending=False)

for index, value in dai_t_sorted_correlations.items():
    print(f"{index:50} {value}")


# In[656]:


# Select correlations of all features with the target variable, excluding the target variable itself
dai_p_target_correlations = correlations['dai_market_Close'].drop('dai_market_Close')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
dai_p_sorted_correlations = dai_p_target_correlations.sort_values(ascending=False)

for index, value in dai_p_sorted_correlations.items():
    print(f"{index:50} {value}")


# In[657]:


# Select correlations of all features with the target variable, excluding the target variable itself
mkr_target_correlations = correlations['mkr_market_Close'].drop('mkr_market_Close')

# Sort the correlations to see the most positive and negative correlations at the top and bottom, respectively
mkr_target_correlations = mkr_target_correlations.sort_values(ascending=False)

for index, value in mkr_target_correlations.items():
    print(f"{index:50} {value}")


# Since the relationship appears to be nonlienar, need to try the spearman and mi features with high correlations instead of pearson

# ## Aggregated Vaults

# BTC Vault_collateral_usd
# ETH Vault_collateral_usd
# stETH Vault_collateral_usd
# Stablecoin Vault_collateral_usd
# Altcoin Vault_collateral_usd
# LP Vault_collateral_usd
# RWA Vault_collateral_usd
# PSM Vault_collateral_usd

# In[658]:


aggregated_spearman_corr = aggregated_vault_data.corr(method='spearman')


# In[659]:


eth_spearman_aggregated_usd = aggregated_spearman_corr['ETH Vault_collateral_usd']
eth_spearman_aggregated_usd_sorted = eth_spearman_aggregated_usd.sort_values(ascending=False)
for index, value in eth_spearman_aggregated_usd_sorted.items():
    print(f"{index:50} {value}")


# In[660]:


steth_spearman_aggregated_usd = aggregated_spearman_corr['stETH Vault_collateral_usd']
steth_spearman_aggregated_usd_sorted = steth_spearman_aggregated_usd.sort_values(ascending=False)
for index, value in steth_spearman_aggregated_usd_sorted.items():
    print(f"{index:50} {value}")


# In[661]:


wbtc_spearman_aggregated_usd = aggregated_spearman_corr['BTC Vault_collateral_usd']
wbtc_spearman_aggregated_usd_sorted = wbtc_spearman_aggregated_usd.sort_values(ascending=False)
for index, value in wbtc_spearman_aggregated_usd_sorted.items():
    print(f"{index:50} {value}")


# In[662]:


alt_spearman_aggregated_usd = aggregated_spearman_corr['Altcoin Vault_collateral_usd']
alt_spearman_aggregated_usd_sorted = alt_spearman_aggregated_usd.sort_values(ascending=False)
for index, value in alt_spearman_aggregated_usd_sorted.items():
    print(f"{index:50} {value}")


# In[663]:


stb_spearman_aggregated_usd = aggregated_spearman_corr['Stablecoin Vault_collateral_usd']
stb_spearman_aggregated_usd_sorted = stb_spearman_aggregated_usd.sort_values(ascending=False)
for index, value in stb_spearman_aggregated_usd_sorted.items():
    print(f"{index:50} {value}")


# In[664]:


lp_spearman_aggregated_usd = aggregated_spearman_corr['LP Vault_collateral_usd']
lp_spearman_aggregated_usd_sorted = lp_spearman_aggregated_usd.sort_values(ascending=False)
for index, value in lp_spearman_aggregated_usd_sorted.items():
    print(f"{index:50} {value}")


# In[665]:


rwa_spearman_aggregated_usd = aggregated_spearman_corr['RWA Vault_collateral_usd']
rwa_spearman_aggregated_usd_sorted = rwa_spearman_aggregated_usd.sort_values(ascending=False)
for index, value in rwa_spearman_aggregated_usd_sorted.items():
    print(f"{index:50} {value}")


# In[666]:


psm_spearman_aggregated_usd = aggregated_spearman_corr['PSM Vault_collateral_usd']
psm_spearman_aggregated_usd_sorted = psm_spearman_aggregated_usd.sort_values(ascending=False)
for index, value in psm_spearman_aggregated_usd_sorted.items():
    print(f"{index:50} {value}")


# ## Individual Vaults

# In[667]:


# Assuming 'df' is your DataFrame
# Calculate Spearman's correlation
spearman_corr = numeric_dataset.corr(method='spearman')

# Prepare data for mutual information calculation
X = numeric_dataset.drop('eth_a_vault_cumulative_collateral', axis=1)  # features
y = numeric_dataset['eth_a_vault_cumulative_collateral']               # target variable

# Calculate mutual information
mi = mutual_info_regression(X, y)
mi /= np.max(mi)  # Normalize results to scale from 0 to 1

# Convert to DataFrame for better visualization
mi_df = pd.DataFrame(mi, index=X.columns, columns=['Mutual Information'])

# Display results
print(spearman_corr)


# In[668]:


print(mi_df.sort_values(by='Mutual Information', ascending=False))


# In[669]:


spear_corr = spearman_corr['eth_a_vault_cumulative_collateral']
spear_corr_sorted = spear_corr.sort_values(ascending=False)

# Print each feature with its sorted mutual information
for index, value in spear_corr_sorted.items():
    print(f"{index:50} {value:.5f}")


# In[670]:


# Convert DataFrame to a dictionary
mi_dict = mi_df['Mutual Information'].to_dict()

# Sort the dictionary by values (mutual information scores) in descending order
sorted_mi_dict = dict(sorted(mi_dict.items(), key=lambda item: item[1], reverse=True))

# Print each feature with its sorted mutual information
for index, value in sorted_mi_dict.items():
    print(f"{index:50} {value:.5f}")



# # Features for Forecasting/Simulating Balances 

# ## Individual Vault
# 
# Now, we need to forecast the balances.  Having accurate forecasting on backtesting will infer that it will accurately simulate vault balance changes as result of key parameter changes, namely Liquidation Ratio and Dai Ceiling.
# 
# Correlation analysis with collateral balance shows that parameters with most correlation are Liquidation Ratio, Dai Ceiling
# Therefore, we will see which data science methods will work best for the forecasting portion

# In[671]:


#linear correlated features
significant_correlations = collateral_sorted_correlations[(collateral_sorted_correlations > 0.6) | (collateral_sorted_correlations < -0.6)]
for index, value in significant_correlations.items():
    print(f"{index:50} {value}")


# filtered_spear_corr = spear_corr_sorted[(spear_corr_sorted.abs() > 0.6)]
# for index, value in filtered_spear_corr.items():
#     print(f"{index:50} {value:.5f}")

# In[672]:


filtered_aggregated_eth_usd = eth_spearman_aggregated_usd_sorted[(eth_spearman_aggregated_usd_sorted.abs() > 0.8)]
for index, value in filtered_aggregated_eth_usd.items():
    print(f"{index:50} {value:.5f}")


# In[673]:


aggregated_vault_data


# In[674]:


significant_eth_aggregated_columns = list(filtered_aggregated_eth_usd.index)
significant_eth_aggregated_df = aggregated_vault_data[significant_eth_aggregated_columns]


# In[675]:


filtered_mi_dict = {key: val for key, val in sorted_mi_dict.items() if val > 0.6}

# Now print each feature with its mutual information that meets the condition
for index, value in filtered_mi_dict.items():
    print(f"{index:50} {value:.5f}")


# In[676]:


significant_columns = list(significant_correlations.index)  # List of significant columns
significant_dataset = dataset_no_nan[significant_columns]
print(significant_dataset.head())


# ## Aggregated Vaults

# In[677]:


aggregated_vault_data['Vaults Total USD Value'].plot()


# ### Spearman

# In[678]:


st_high_corr = steth_spearman_aggregated_usd_sorted[steth_spearman_aggregated_usd_sorted.abs() > .95]
st_high_corr_no_target = st_high_corr.drop('stETH Vault_collateral_usd', axis=0)
st_high_corr_no_target



# In[679]:


filtered_aggregated_eth_usd_allvaults_spearman = eth_spearman_aggregated_usd_sorted[(eth_spearman_aggregated_usd_sorted.abs() > 0.8)]
filtered_aggregated_steth_usd_allvaults_spearman = steth_spearman_aggregated_usd_sorted[(steth_spearman_aggregated_usd_sorted.abs() > 0.8)]
filtered_aggregated_wbtc_usd_allvaults_spearman = wbtc_spearman_aggregated_usd_sorted[(wbtc_spearman_aggregated_usd_sorted.abs() > 0.8)]
filtered_aggregated_stb_usd_allvaults_spearman = stb_spearman_aggregated_usd_sorted[(stb_spearman_aggregated_usd_sorted.abs() > 0.8)]
filtered_aggregated_alt_usd_allvaults_spearman = alt_spearman_aggregated_usd_sorted[(alt_spearman_aggregated_usd_sorted.abs() > 0.8)]
filtered_aggregated_lp_usd_allvaults_spearman = lp_spearman_aggregated_usd_sorted[(lp_spearman_aggregated_usd_sorted.abs() > 0.8)]
filtered_aggregated_rwa_usd_allvaults_spearman = rwa_spearman_aggregated_usd_sorted[(rwa_spearman_aggregated_usd_sorted.abs() > 0.8)]
filtered_aggregated_psm_usd_allvaults_spearman = psm_spearman_aggregated_usd_sorted[(psm_spearman_aggregated_usd_sorted.abs() > 0.8)]

print('eth spearman features:',filtered_aggregated_eth_usd_allvaults_spearman.shape[0])
print('stETH spearman features:',filtered_aggregated_steth_usd_allvaults_spearman.shape[0])
print('BTC spearman features:',filtered_aggregated_wbtc_usd_allvaults_spearman.shape[0])
print('stablecoin spearman features:',filtered_aggregated_stb_usd_allvaults_spearman.shape[0])
print('altcoin spearman features:',filtered_aggregated_alt_usd_allvaults_spearman.shape[0])
print('LP spearman features:',filtered_aggregated_lp_usd_allvaults_spearman.shape[0])
print('RWA spearman features:',filtered_aggregated_rwa_usd_allvaults_spearman.shape[0])
print('PSM spearman features:',filtered_aggregated_psm_usd_allvaults_spearman.shape[0])


# filtered_aggregated_steth_usd_allvaults_spearman.drop(index = st_high_corr_no_target.index, inplace=True)

# In[680]:


significant_eth_aggregated_columns_av_s = list(filtered_aggregated_eth_usd_allvaults_spearman.index)
significant_eth_aggregated_df_av_s = aggregated_vault_data[significant_eth_aggregated_columns_av_s]


# In[681]:


significant_steth_aggregated_columns_av_s = list(filtered_aggregated_steth_usd_allvaults_spearman.index)
significant_steth_aggregated_df_av_s = aggregated_vault_data[significant_steth_aggregated_columns_av_s]


# In[682]:


significant_wbtc_aggregated_columns_av_s = list(filtered_aggregated_wbtc_usd_allvaults_spearman.index)
significant_wbtc_aggregated_df_av_s = aggregated_vault_data[significant_wbtc_aggregated_columns_av_s]


# In[683]:


significant_stb_aggregated_columns_av_s = list(filtered_aggregated_stb_usd_allvaults_spearman.index)
significant_stb_aggregated_df_av_s = aggregated_vault_data[significant_stb_aggregated_columns_av_s]


# In[684]:


significant_alt_aggregated_columns_av_s = list(filtered_aggregated_alt_usd_allvaults_spearman.index)
significant_alt_aggregated_df_av_s = aggregated_vault_data[significant_alt_aggregated_columns_av_s]


# In[685]:


significant_lp_aggregated_columns_av_s = list(filtered_aggregated_lp_usd_allvaults_spearman.index)
significant_lp_aggregated_df_av_s = aggregated_vault_data[significant_lp_aggregated_columns_av_s]


# In[686]:


significant_rwa_aggregated_columns_av_s = list(filtered_aggregated_rwa_usd_allvaults_spearman.index)
significant_rwa_aggregated_df_av_s = aggregated_vault_data[significant_rwa_aggregated_columns_av_s]


# In[687]:


significant_psm_aggregated_columns_av_s = list(filtered_aggregated_psm_usd_allvaults_spearman.drop(columns=[['PSM Vault_hypothetical_dai_ceiling','psm_balance']]).index)
significant_psm_aggregated_df_av_s = aggregated_vault_data[significant_psm_aggregated_columns_av_s]


# In[688]:


significant_eth_aggregated_all_vaults_spearman = significant_eth_aggregated_df_av_s[significant_eth_aggregated_df_av_s.index > '2021-11-18']
significant_steth_aggregated_all_vaults_spearman = significant_steth_aggregated_df_av_s[significant_steth_aggregated_df_av_s.index > '2021-11-18']
significant_wbtc_aggregated_all_vaults_spearman = significant_wbtc_aggregated_df_av_s[significant_wbtc_aggregated_df_av_s.index > '2021-11-18']
significant_stb_aggregated_all_vaults_spearman = significant_stb_aggregated_df_av_s[significant_stb_aggregated_df_av_s.index > '2021-11-18']
significant_alt_aggregated_all_vaults_spearman = significant_alt_aggregated_df_av_s[significant_alt_aggregated_df_av_s.index > '2021-11-18']
significant_lp_aggregated_all_vaults_spearman = significant_lp_aggregated_df_av_s[significant_lp_aggregated_df_av_s.index > '2021-11-18']
significant_psm_aggregated_all_vaults_spearman = significant_psm_aggregated_df_av_s[significant_psm_aggregated_df_av_s.index > '2021-11-18']
significant_rwa_aggregated_all_vaults_spearman = significant_rwa_aggregated_df_av_s[significant_rwa_aggregated_df_av_s.index > '2021-11-18']




# ### Pearson

# In[689]:


filtered_aggregated_eth_usd_allvaults_pearson = correlated_aggregated_eth_collateral_usd_sorted[(correlated_aggregated_eth_collateral_usd_sorted.abs() > 0.8)]
filtered_aggregated_steth_usd_allvaults_pearson = correlated_aggregated_steth_collateral_usd[(correlated_aggregated_steth_collateral_usd.abs() > 0.8)]
filtered_aggregated_wbtc_usd_allvaults_pearson = correlated_aggregated_wbtc_collateral_usd[(correlated_aggregated_wbtc_collateral_usd.abs() > 0.8)]
filtered_aggregated_stb_usd_allvaults_pearson = correlated_aggregated_stb_collateral_usd[(correlated_aggregated_stb_collateral_usd.abs() > 0.8)]
filtered_aggregated_alt_usd_allvaults_pearson = correlated_aggregated_alt_collateral_usd[(correlated_aggregated_alt_collateral_usd.abs() > 0.8)]
filtered_aggregated_lp_usd_allvaults_pearson = correlated_aggregated_lp_collateral_usd[(correlated_aggregated_lp_collateral_usd.abs() > 0.8)]
filtered_aggregated_rwa_usd_allvaults_pearson = correlated_aggregated_rwa_collateral_usd[(correlated_aggregated_rwa_collateral_usd.abs() > 0.8)]
filtered_aggregated_psm_usd_allvaults_pearson = correlated_aggregated_psm_collateral_usd[(correlated_aggregated_psm_collateral_usd.abs() > 0.8)]


print('eth pearson features:',filtered_aggregated_eth_usd_allvaults_pearson.shape[0])
print('stETH pearson features:',filtered_aggregated_steth_usd_allvaults_pearson.shape[0])
print('BTC pearson features:',filtered_aggregated_wbtc_usd_allvaults_pearson.shape[0])
print('stablecoin pearson features:',filtered_aggregated_stb_usd_allvaults_pearson.shape[0])
print('altcoin pearson features:',filtered_aggregated_alt_usd_allvaults_pearson.shape[0])
print('LP pearson features:',filtered_aggregated_lp_usd_allvaults_pearson.shape[0])
print('RWA pearson features:',filtered_aggregated_rwa_usd_allvaults_pearson.shape[0])
print('PSM pearson features:',filtered_aggregated_psm_usd_allvaults_pearson.shape[0])


# In[690]:


significant_eth_aggregated_columns_av = list(filtered_aggregated_eth_usd_allvaults_pearson.index)
significant_eth_aggregated_df_av = aggregated_vault_data[significant_eth_aggregated_columns_av]


# In[691]:


significant_steth_aggregated_columns_av = list(filtered_aggregated_steth_usd_allvaults_pearson.index)
significant_steth_aggregated_df_av = aggregated_vault_data[significant_steth_aggregated_columns_av]


# In[692]:


significant_wbtc_aggregated_columns_av = list(filtered_aggregated_wbtc_usd_allvaults_pearson.index)
significant_wbtc_aggregated_df_av = aggregated_vault_data[significant_wbtc_aggregated_columns_av]


# In[693]:


significant_stb_aggregated_columns_av = list(filtered_aggregated_stb_usd_allvaults_pearson.index)
significant_stb_aggregated_df_av = aggregated_vault_data[significant_stb_aggregated_columns_av]


# In[694]:


significant_alt_aggregated_columns_av = list(filtered_aggregated_alt_usd_allvaults_pearson.index)
significant_alt_aggregated_df_av = aggregated_vault_data[significant_alt_aggregated_columns_av]


# In[695]:


significant_lp_aggregated_columns_av = list(filtered_aggregated_lp_usd_allvaults_pearson.index)
significant_lp_aggregated_df_av = aggregated_vault_data[significant_lp_aggregated_columns_av]


# In[696]:


significant_rwa_aggregated_columns_av = list(filtered_aggregated_rwa_usd_allvaults_pearson.index)
significant_rwa_aggregated_df_av = aggregated_vault_data[significant_rwa_aggregated_columns_av]


# In[697]:


significant_psm_aggregated_columns_av = list(filtered_aggregated_psm_usd_allvaults_pearson.drop(columns=[['PSM Vault_hypothetical_dai_ceiling','psm_balance']]).index)
significant_psm_aggregated_df_av = aggregated_vault_data[significant_psm_aggregated_columns_av]


# In[698]:


significant_eth_aggregated_all_vaults_pearson = significant_eth_aggregated_df_av[significant_eth_aggregated_df_av.index > '2021-11-18']
significant_steth_aggregated_all_vaults_pearson = significant_steth_aggregated_df_av[significant_steth_aggregated_df_av.index > '2021-11-18']
significant_wbtc_aggregated_all_vaults_pearson = significant_wbtc_aggregated_df_av[significant_wbtc_aggregated_df_av.index > '2021-11-18']
significant_stb_aggregated_all_vaults_pearson = significant_stb_aggregated_df_av[significant_stb_aggregated_df_av.index > '2021-11-18']
significant_alt_aggregated_all_vaults_pearson = significant_alt_aggregated_df_av[significant_alt_aggregated_df_av.index > '2021-11-18']
significant_lp_aggregated_all_vaults_pearson = significant_lp_aggregated_df_av[significant_lp_aggregated_df_av.index > '2021-11-18']
significant_psm_aggregated_all_vaults_pearson = significant_psm_aggregated_df_av[significant_psm_aggregated_df_av.index > '2021-11-18']
significant_rwa_aggregated_all_vaults_pearson = significant_rwa_aggregated_df_av[significant_rwa_aggregated_df_av.index > '2021-11-18']