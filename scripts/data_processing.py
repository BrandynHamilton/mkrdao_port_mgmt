#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Core libraries
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
"""
# Set random seeds to ensure reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
"""

# In[2]:


@st.cache_data()
def fetch_data_from_api(api_url, params=None):
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'rows' in data['result']:
            return pd.DataFrame(data['result']['rows'])
        return data
    else:
        #print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # or an empty dict


# In[3]:


def fetch_historical_data(api_url, api_key):
    # Use the API key either as a query parameter or in the headers
    params = {'vs_currency': 'usd', 'days': 'max', 'interval': 'daily', 'x_cg_demo_api_key': api_key}
    headers = {'x-cg-demo-api-key': api_key}  # Alternatively, use this header

    response = requests.get(api_url, params=params, headers=headers)

    if response.status_code == 200:
        # Parse the JSON response
        historical_pricedata = response.json()
        # Extract the 'prices' and 'market_caps' data
        historical_price = historical_pricedata['prices']
        market_cap = pd.DataFrame(historical_pricedata['market_caps'], columns=['date', 'marketcap'])

        # Convert the 'timestamp' column from UNIX timestamps in milliseconds to datetime objects
        history = pd.DataFrame(historical_price, columns=['timestamp', 'price'])
        history['date'] = pd.to_datetime(history['timestamp'], unit='ms')
        history.set_index('date', inplace=True)
        history.drop(columns='timestamp', inplace=True)

        vol = pd.DataFrame(historical_pricedata['total_volumes'], columns=['date', 'volume'])
        vol['date'] = pd.to_datetime(vol['date'], unit='ms')
        vol.set_index('date', inplace=True)
        
        return history, market_cap, vol
    else:
        #print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# api_key_dune = st.secrets["api_key"]
# api_key_cg = st.secrets["api_key_cg"]
# api_key_FRED = st.secrets["FRED_API_KEY"]

# dune = DuneClient(api_key_dune)

# In[4]:


def fetch_dune_data(num):
    result = dune.get_latest_result(num)
    return pd.DataFrame(result.result.rows)


# In[5]:


pd.options.display.float_format = '{:,.5f}'.format


# ### First, lets get MakerDAO Financial Statements from https://dune.com/steakhouse/makerdao

# Balance Sheet

# In[6]:


# Balance Sheet
#bs_raw = dune.get_latest_result(2840463)


# In[7]:


#bs_df = pd.DataFrame(bs_raw.result.rows)
#bs_df['period'] = pd.to_datetime(bs_df['period'])
#bs_df.set_index('period', inplace=True)
#bs_df.index = bs_df.index.normalize()
#bs_df = bs_df.sort_index()


# In[8]:


#bs_df


# In[9]:


bs_path = 'data/csv/bs.csv'


# In[10]:


#bs_df.to_csv(bs_path)


# In[11]:


bs_csv = pd.read_csv(bs_path, index_col='period', parse_dates=True)


# In[12]:


#categorizing items as asset, liability, or equity
def categorize_item(item):
    if item in ['Crypto-Loans', 'Real-World Assets', 'Others assets', 'Stablecoins']:
        return 'Assets'
    elif item in ['DAI','DSR']:  # Assuming DAI represents a liability here; adjust according to your accounting rules
        return 'Liabilities'
    elif item == 'Equity':
        return 'Equity'
    else:
        return 'Other'  # For any item not explicitly categorized

# Assuming 'df' is your DataFrame
bs_csv['category'] = bs_csv['item'].apply(categorize_item)



# In[13]:


bs_csv = bs_csv.iloc[::-1]


# In[14]:


bs_csv.tail()


# In[15]:


bs_csv.describe()


# In[16]:


pivoted_balance_sheet = bs_csv.pivot(columns='item', values='balance')


# In[17]:


# Percent Changes in account balances 
pivoted_balance_sheet['Crypto-Loans_pct_chg'] = pivoted_balance_sheet['Crypto-Loans'].pct_change()
pivoted_balance_sheet['DAI_pct_chg'] = pivoted_balance_sheet['DAI'].pct_change()
pivoted_balance_sheet['DSR_pct_chg'] = pivoted_balance_sheet['DSR'].pct_change()
pivoted_balance_sheet['Equity_pct_chg'] = pivoted_balance_sheet['Equity'].pct_change()
pivoted_balance_sheet['Others_assets_pct_chg'] = pivoted_balance_sheet['Others assets'].pct_change()  # Assuming this is the correct column name
pivoted_balance_sheet['Real-World_Assets_pct_chg'] = pivoted_balance_sheet['Real-World Assets'].pct_change()
pivoted_balance_sheet['Stablecoins_pct_chg'] = pivoted_balance_sheet['Stablecoins'].pct_change()


# In[18]:


# Rolling Averages, Standard Deviation
# Define the window size for rolling calculation
window_size = 7  # for example, a 7-day rolling window

# Calculate rolling averages
pivoted_balance_sheet['Crypto-Loans_7d_rolling_avg'] = pivoted_balance_sheet['Crypto-Loans'].rolling(window=window_size).mean()
pivoted_balance_sheet['DAI_7d_rolling_avg'] = pivoted_balance_sheet['DAI'].rolling(window=window_size).mean()
pivoted_balance_sheet['DSR_7d_rolling_avg'] = pivoted_balance_sheet['DSR'].rolling(window=window_size).mean()
pivoted_balance_sheet['Equity_7d_rolling_avg'] = pivoted_balance_sheet['Equity'].rolling(window=window_size).mean()
pivoted_balance_sheet['Others_assets_7d_rolling_avg'] = pivoted_balance_sheet['Others assets'].rolling(window=window_size).mean()
pivoted_balance_sheet['Real-World_Assets_7d_rolling_avg'] = pivoted_balance_sheet['Real-World Assets'].rolling(window=window_size).mean()
pivoted_balance_sheet['Stablecoins_7d_rolling_avg'] = pivoted_balance_sheet['Stablecoins'].rolling(window=window_size).mean()

# Calculate volatility (standard deviation)
pivoted_balance_sheet['Crypto-Loans_7d_rolling_volatility'] = pivoted_balance_sheet['Crypto-Loans_pct_chg'].rolling(window=window_size).std()
pivoted_balance_sheet['DAI_7d_rolling_volatility'] = pivoted_balance_sheet['DAI_pct_chg'].rolling(window=window_size).std()
pivoted_balance_sheet['DSR_7d_rolling_volatility'] = pivoted_balance_sheet['DSR_pct_chg'].rolling(window=window_size).std()
pivoted_balance_sheet['Equity_7d_rolling_volatility'] = pivoted_balance_sheet['Equity_pct_chg'].rolling(window=window_size).std()
pivoted_balance_sheet['Others_assets_7d_rolling_volatility'] = pivoted_balance_sheet['Others_assets_pct_chg'].rolling(window=window_size).std()
pivoted_balance_sheet['Real-World_Assets_7d_rolling_volatility'] = pivoted_balance_sheet['Real-World_Assets_pct_chg'].rolling(window=window_size).std()
pivoted_balance_sheet['Stablecoins_7d_rolling_volatility'] = pivoted_balance_sheet['Stablecoins_pct_chg'].rolling(window=window_size).std()



# In[19]:


# Assuming you've already calculated percent changes (_pct_chg)
window_size = 30  # Adjust based on your analysis needs

# Calculate rolling averages and volatilities based on percent changes
for col in ['Crypto-Loans', 'DAI', 'DSR', 'Equity', 'Others_assets', 'Real-World_Assets', 'Stablecoins']:
    pct_chg_col = f'{col}_pct_chg'  # The column names for percent changes you've calculated
    pivoted_balance_sheet[f'{col}_30d_rolling_avg_pct_chg'] = pivoted_balance_sheet[pct_chg_col].rolling(window=window_size).mean()
    pivoted_balance_sheet[f'{col}_30d_volatility_pct_chg'] = pivoted_balance_sheet[pct_chg_col].rolling(window=window_size).std()


# In[20]:


pivoted_balance_sheet.columns = [f'b_s_{col}' if col != 'period' else col for col in pivoted_balance_sheet.columns]


# In[21]:


pivoted_balance_sheet.tail()


# In[22]:


pivoted_balance_sheet.shape[0]


# MONTHLY Income Statement/PnL (also includes more detailed balance sheet)

# In[23]:


#is_df = fetch_dune_data(2641549) 


# In[24]:


#is_df.head()


# In[25]:


#is_df_wide = is_df.pivot_table(index='period', columns='item', values='value', aggfunc='sum').reset_index()
#is_df_wide = is_df_wide.iloc[::-1]


# In[26]:


#is_df_wide.head()


# In[27]:


is_path = 'data/csv/is.csv'


# In[28]:


#is_df.to_csv(is_path)


# In[29]:


is_csv = pd.read_csv(is_path, index_col='period', parse_dates=True)


# In[30]:


cleaned_is = is_csv.drop(columns=['Unnamed: 0','year'])


# In[31]:


cleaned_is.describe()


# In[32]:


pivoted_income_statement = is_csv.pivot_table(index='period', 
                            columns='item', 
                            values='value', 
                            aggfunc='sum').reset_index()


# In[33]:


pivoted_income_statement.drop(columns=['1 - PnL','2 - Assets','2.8 - Operating Reserves','3 - Liabilities & Equity','3.8 - Equity (Operating Reserves)'], inplace=True)


# In[34]:


pivoted_income_statement['Total Revenues']= pivoted_income_statement[['1.1 - Lending Revenues', '1.2 - Liquidations Revenues', '1.3 - Trading Revenues']].sum(axis=1)
pivoted_income_statement['Total Expenses'] = pivoted_income_statement[['1.4 - Lending Expenses', '1.5 - Liquidations Expenses', '1.6 - Workforce Expenses']].sum(axis=1)
pivoted_income_statement['profit_margin'] = pivoted_income_statement['1.9 - Net Income'] / pivoted_income_statement['2.9 - Total Assets']
pivoted_income_statement['ROA'] = pivoted_income_statement['1.9 - Net Income'] / pivoted_income_statement['2.9 - Total Assets']
pivoted_income_statement['ROE'] = pivoted_income_statement['1.9 - Net Income'] / pivoted_income_statement['3.7 - Equity (Surplus Buffer)']
pivoted_income_statement['debt_to_equity'] = pivoted_income_statement['3.1 - Liabilities (DAI)'] / pivoted_income_statement['3.7 - Equity (Surplus Buffer)']
pivoted_income_statement['debt_ratio'] = pivoted_income_statement['3.1 - Liabilities (DAI)'] / pivoted_income_statement['2.9 - Total Assets'] 
pivoted_income_statement['cumulative_revenues'] = pivoted_income_statement['Total Revenues'].cumsum()
pivoted_income_statement['cumulative_expenses'] = pivoted_income_statement['Total Expenses'].cumsum()
pivoted_income_statement['cumulative_net_income'] = pivoted_income_statement['1.9 - Net Income'].cumsum()


# In[35]:


pivoted_income_statement[['1.9 - Net Income','2.9 - Total Assets']]


# In[ ]:





# In[36]:


pivoted_income_statement.tail()


# In[37]:


# Percent Changes in account balances
window_size = 3  # Three months
# Calculate rolling averages and volatilities based on percent changes
for col in ['Total Revenues', 'Total Expenses', '1.9 - Net Income']:
    pct_chg_col = f'{col}_pct_chg'  # Define the percent change column name
    pivoted_income_statement[pct_chg_col] = pivoted_income_statement[col].pct_change()
    # Use the pct_chg_col variable correctly now
    pivoted_income_statement[f'{col}_3m_rolling_avg_pct_chg'] = pivoted_income_statement[pct_chg_col].rolling(window=window_size).mean()
    pivoted_income_statement[f'{col}_3m_volatility_pct_chg'] = pivoted_income_statement[pct_chg_col].rolling(window=window_size).std()
    for lag in range(1,13):
        pivoted_income_statement[f'{col}_3m_rolling_avg_pct_chg_lag_{lag}'] = pivoted_income_statement[f'{col}_3m_rolling_avg_pct_chg'].shift(lag)
        pivoted_income_statement[f'{col}_3m_volatility_pct_chg_lag_{lag}'] = pivoted_income_statement[f'{col}_3m_volatility_pct_chg'].shift(lag)


# In[38]:


# Generate lagged features
for lag in range(1, 13):  # From 1 to 12 months
    pivoted_income_statement[f'Total_Revenues_Lag_{lag}m'] = pivoted_income_statement['Total Revenues'].shift(lag)
    pivoted_income_statement[f'Total_Expenses_Lag_{lag}m'] = pivoted_income_statement['Total Expenses'].shift(lag)
    pivoted_income_statement[f'Net_Income_Lag_{lag}m'] = pivoted_income_statement['1.9 - Net Income'].shift(lag)
    pivoted_income_statement[f'profit_margin_Lag_{lag}m'] = pivoted_income_statement['profit_margin'].shift(lag)
    pivoted_income_statement[f'ROA_Lag_{lag}m'] = pivoted_income_statement['ROA'].shift(lag)
    pivoted_income_statement[f'ROE_Lag_{lag}m'] = pivoted_income_statement['ROE'].shift(lag)
    pivoted_income_statement[f'debt_to_equity_Lag_{lag}m'] = pivoted_income_statement['debt_to_equity'].shift(lag)
    pivoted_income_statement[f'debt_ratio_Lag_{lag}m'] = pivoted_income_statement['debt_ratio'].shift(lag)


# In[39]:


pivoted_income_statement.shape[0]


# Assets/Revenue Per Type
# Coinbase asset type: http://forum.makerdao.com/t/mip81-coinbase-usdc-institutional-rewards/17703/254?u=sebventures
# 

# In[40]:


#assets_raw = dune.get_latest_result(58495)


# In[41]:


#assets_p_t_ts = pd.DataFrame(assets_raw.result.rows)
#assets_p_t_ts['dt'] = pd.to_datetime(assets_p_t_ts['dt'])
#assets_p_t_ts.set_index('dt', inplace=True)


# In[42]:


#assets_p_t_ts.head()


# In[43]:


#assets_p_t_ts.to_csv(as_path)


# In[44]:


as_path = 'data/csv/as.csv'


# In[45]:


as_csv = pd.read_csv(as_path, index_col='dt', parse_dates=True)


# In[46]:


as_csv = as_csv.drop(columns=['total_asset'])


# Daily Interest Revenues By Vault

# In[47]:


#ir_v = fetch_dune_data(3567939) 


# In[48]:


#ir_v['period'] = pd.to_datetime(ir_v['period'])
#ir_v.set_index('period', inplace=True)


# In[49]:


#ir_v.head()


# In[50]:


daily_int_path = 'data/csv/d_int.csv'


# In[51]:


#ir_v.to_csv(daily_int_path)


# In[52]:


ir_csv = pd.read_csv(daily_int_path, index_col='period', parse_dates=True)


# In[53]:


ir_csv.shape[0]


# In[54]:


ir_csv = ir_csv.rename_axis('day')

 


# In[55]:


ir_csv = ir_csv.rename(columns={'collateral':'ilk'})


# In[56]:


ir_csv['ilk'].unique()


# In[57]:


top_vaults = ir_csv.groupby('ilk').sum().sort_values('daily_revenues', ascending=False)


# In[58]:


#10 Most Revenue Generating Vaults
top_10_vaults = top_vaults.head(10)


# In[59]:


top_10_vaults


# Dai Maturity Profile
# 
# Step-by-Step Process:
# Tracking DAI Movements:
# 
# First, all transactions involving DAI are tracked to understand how DAI moves in and out of wallets. This includes both inflows (adding DAI to a wallet) and outflows (removing DAI from a wallet).
# Defining Maturity Buckets:
# 
# Maturity buckets are predefined categories based on time durations, such as "1-day", "1-week", "1-month", "1-year", etc. Each bucket represents a hypothesis about how long DAI tends to stay put before being moved again.
# Assigning Weights to Buckets:
# 
# Weights are assigned to each maturity bucket to reflect assumptions or historical observations about the distribution of DAI across these buckets. For example, if historically 30% of DAI is moved or used within a day, then the "1-day" bucket might get a weight of 0.30 (or 30%).
# Applying Weights Based on Wallet Types:
# 
# DAI can be held in different types of wallets or contracts, each with its own expected behavior. For example, DAI in a savings contract (like DSR) might be considered more long-term ("1-year"), while DAI in a regular wallet might be more liquid ("1-day" or "1-week"). The weights applied to the DAI in these wallets reflect these expectations.
# Calculating DAI Amounts per Bucket:
# 
# For each wallet (or DAI holding), the total amount of DAI is distributed across the maturity buckets based on the assigned weights. This means if a wallet has 100 DAI and the "1-day" bucket weight is 30%, then 30 DAI is considered to have a 1-day maturity.
# The process is repeated for each wallet and each maturity bucket, based on the specific weights for that wallet type and the total DAI it holds.
# Aggregating Across the Ecosystem:
# 
# Finally, to get the ecosystem-wide view, the amounts of DAI in each maturity bucket from all wallets are aggregated. This provides a snapshot of how much DAI is considered to be in each maturity bucket across the entire MakerDAO system at any given time.

# In[60]:


#d_m = fetch_dune_data(907852)


# In[61]:


#d_m['dt'] = pd.to_datetime(d_m['dt'])


# In[62]:


#d_m


# In[63]:


#d_m.to_csv(d_m_path)


# In[64]:


d_m_path = 'data/csv/d_m.csv'


# In[65]:


d_m_csv = pd.read_csv(d_m_path, index_col='dt', parse_dates=True)


# In[66]:


#print(d_m_csv.describe())


# In[67]:


clean_dm = d_m_csv.drop(columns=['Unnamed: 0'])


# In[68]:


clean_dm.describe()


# In[69]:


dai_maturity_df = d_m_csv.pivot_table(
    index='dt',  # or 'period' if your DataFrame's time column is named 'period'
    columns='maturity',
    values=['outflow', 'outflow_dai_only', 'outflow_surplus_buffer'],
    aggfunc='sum'  # or any other aggregation function that fits your needs
)

# Flatten the MultiIndex columns (optional, for cleaner column names)
dai_maturity_df.columns = ['_'.join(col).strip() for col in dai_maturity_df.columns.values]

# Reset the index if you want 'dt' back as a regular column
dai_maturity_df.reset_index(inplace=True)



# In[70]:


dai_maturity_df.tail()


# In[71]:


dai_maturity_df['dt'] = pd.to_datetime(dai_maturity_df['dt'])
dai_maturity_df.set_index('dt', inplace=True)


start_date = dai_maturity_df.index.min()
end_date = dai_maturity_df.index.max()
date_range = pd.date_range(start=start_date, end=end_date, freq='D')  # 'D' for daily frequency




# In[72]:


dai_maturity_df_reindexed = dai_maturity_df.reindex(date_range)

# Forward-fill missing values
dai_maturity_df_reindexed.ffill(inplace=True)

# Optionally, reset the index if you want 'dt' back as a column
dai_maturity_df_reindexed.reset_index(inplace=True)
dai_maturity_df_reindexed.rename(columns={'index': 'day'}, inplace=True)


# In[73]:


dai_maturity_df_reindexed.columns = [f'dai_maturity_{col}' if col != 'day' and not col.startswith('dai_maturity_') else col for col in dai_maturity_df_reindexed.columns]


# In[74]:


dai_maturity_df_reindexed.tail()


# MakerDAO Stablecoin Ratio
# This can give insights into the proportion of assets held in stablecoins (including DAI) relative to other assets. A higher stablecoin ratio might suggest a preference for stability within the MakerDAO system, which can have implications for DAI's stability.

# In[75]:


#stablecoin_ratio_df = fetch_dune_data(58136)


# In[76]:


#stablecoin_ratio_df['dt'] = pd.to_datetime(stablecoin_ratio_df['dt'])
#stablecoin_ratio_df.set_index('dt', inplace=True)


# In[77]:


#stablecoin_ratio_df.to_csv(stablecoin_ratio_path)


# In[78]:


stablecoin_ratio_path = 'data/csv/srp.csv'


# In[79]:


stablecoin_ratio_csv = pd.read_csv(stablecoin_ratio_path, index_col='dt', parse_dates=True)


# In[80]:


stablecoin_ratio_csv.head()


# In[81]:


stable_coin_ratios = stablecoin_ratio_csv[['stablecoins_ratio','usdc_ratio']]


# In[82]:


stable_coin_ratios = stable_coin_ratios.rename_axis('day')


# In[83]:


stable_coin_ratios.index


# Maker Peg Stability Module Stats
# Data on the Peg Stability Module, which helps maintain DAI's peg to the USD, can be vital. Insights into the inflows, outflows, and balances within the PSM can directly indicate efforts to stabilize DAI.

# In[84]:


#psm_stats_df = fetch_dune_data(17216)


# In[85]:


#psm_stats_df['date'] = pd.to_datetime(psm_stats_df['date'])


# In[86]:


psm_stats_path = 'data/csv/psm.csv'


# In[87]:


#psm_stats_df.to_csv(psm_stats_path)


# In[88]:


psm_stats_csv = pd.read_csv(psm_stats_path, index_col='date', parse_dates=True)


# In[89]:


psm_stats_csv.columns = [f'psm_{col}' if col != 'dt' and not col.startswith('psm_') else col for col in psm_stats_csv.columns]


# In[90]:


psm_stats_csv = psm_stats_csv.drop(columns=['psm_Unnamed: 0'])


# In[91]:


psm_stats_csv = psm_stats_csv.rename_axis('day')


# In[92]:


psm_stats_csv.columns


# In[93]:


psm_stats_csv[['psm_balance','psm_fees','psm_inflow','psm_outflow','psm_turnover']].describe()


# Where is dai lets us know how dai is being used; dai in lending could be considered to add to stability 

# In[94]:


#where_is_dai_df = fetch_dune_data(54599)


# In[95]:


#where_is_dai_df['dt'] = pd.to_datetime(where_is_dai_df['dt'])


# In[96]:


wid_path = 'data/csv/wid.csv'


# In[97]:


#where_is_dai_df.to_csv(wid_path)


# In[98]:


where_is_dai_csv = pd.read_csv(wid_path, index_col='dt', parse_dates=True)


# In[99]:


where_is_dai_csv = where_is_dai_csv.rename_axis('day')


# In[100]:


#print(where_is_dai_csv.describe())


# In[101]:


where_is_dai_csv.columns = [f'where_is_dai_{col}' if col != 'day' and not col.startswith('where_is_dai_') else col for col in where_is_dai_csv.columns]


# In[102]:


where_is_dai_csv = where_is_dai_csv.drop(columns=['where_is_dai_Unnamed: 0'])


# In[103]:


where_is_dai_csv.shape[0]


# In[104]:


where_is_dai_csv_table = where_is_dai_csv.pivot_table(values='where_is_dai_balance', index='day', columns='where_is_dai_wallet', aggfunc='sum')


# In[105]:


where_is_dai_csv_table.describe()


# In[106]:


where_is_dai_csv_table.columns = [f'where_is_dai_{col}' if col != 'day' and not col.startswith('where_is_dai_') else col for col in where_is_dai_csv_table.columns]


# In[107]:


where_is_dai_csv_table.shape[0]


# In[108]:


where_is_dai_csv_table


# Daily surplus buffer
# Provides information on the surplus buffer in MakerDAO, which is a key financial metric. The surplus buffer acts as a reserve to cover potential system shortfalls and ensures the stability and solvency of the system. This data could be valuable for understanding the financial health and risk management strategies of MakerDAO over time

# In[109]:


#daily_surplus_buffer = fetch_dune_data(3567837)


# In[110]:


#daily_surplus_buffer['period'] = pd.to_datetime(daily_surplus_buffer['period'])


# In[111]:


dsb_path = 'data/csv/dsb.csv'
#daily_surplus_buffer.to_csv(dsb_path)
daily_surplus_buffer_csv = pd.read_csv(dsb_path, index_col='period', parse_dates=True)


# In[112]:


daily_surplus_buffer_csv = daily_surplus_buffer_csv.drop(columns=['Unnamed: 0'])


# In[113]:


daily_surplus_buffer_csv.describe()


# In[114]:


daily_surplus_buffer_csv.columns = [f'daily_surplus_buffer_{col}' if col != 'period' and not col.startswith('surplus_buffer') else col for col in daily_surplus_buffer_csv.columns]


# In[115]:


daily_surplus_buffer_csv = daily_surplus_buffer_csv.rename(columns={'surplus_buffer':'daily_surplus_buffer'})


# In[116]:


daily_surplus_buffer_csv = daily_surplus_buffer_csv.rename_axis('day')


# In[117]:


daily_surplus_buffer_csv.head()


# Stability Fee history and Vault History - Rates Set by DAO
# dart = debt balance

# In[118]:


#sf_df = fetch_dune_data(3551110)


# In[119]:


#sf_df['period'] = pd.to_datetime(sf_df['period'])


# In[120]:


sf_path = 'data/csv/sf.csv'
#sf_df.to_csv(sf_path)
sf_history_csv = pd.read_csv(sf_path, index_col='period', parse_dates=True)


# In[121]:


sf_history_csv


# In[122]:


sf_history_csv_reset = sf_history_csv.reset_index()


# In[123]:


sf_history_csv_clean = sf_history_csv_reset.drop_duplicates(subset=['period', 'ilk'], keep='last')


# In[124]:


sf_history_csv_clean = sf_history_csv_clean.drop(columns='Unnamed: 0')


# In[125]:


sf_history_csv_clean['period'] = sf_history_csv_clean['period'].dt.date


# In[126]:


sf_history_csv_clean = sf_history_csv_clean.rename(columns={'period':'day'})


# DAI Savings Rate Historical - Set by DAO

# In[127]:


#dsr_rate = fetch_dune_data(3581248)


# In[128]:


dsr_rate_path = 'data/csv/dsr.csv'
#dsr_rate.to_csv(dsr_rate_path)
dsr_rate_csv = pd.read_csv(dsr_rate_path, index_col='dt', parse_dates=True)


# In[129]:


dsr_rate_csv['dsr_rate'].describe()


# In[130]:


dsr_rate_csv = dsr_rate_csv.drop(columns=['Unnamed: 0'])


# In[131]:


dsr_rate_csv['dai_percent_in_dsr'] = dsr_rate_csv['dsr_balance'] / dsr_rate_csv['total_balance']
dsr_rate_csv['dai_percent_out_dsr'] = dsr_rate_csv['non_dsr_balance'] / dsr_rate_csv['total_balance']


# In[132]:


dsr_rate_csv = dsr_rate_csv.rename_axis('day')


# In[133]:


#print(dsr_rate_csv.describe())


# dsr flows

# In[134]:


#dsr_flows = fetch_dune_data(1753750)


# In[135]:


dsr_flows_path='data/csv/dsr_flows.csv'
#dsr_flows.to_csv(dsr_flows_path)
dsr_flows_csv = pd.read_csv(dsr_flows_path, index_col='period', parse_dates=True)


# In[136]:


dsr_flows_csv.shape[0]


# In[137]:


dsr_flows_csv = dsr_flows_csv.drop(columns=['Unnamed: 0','balance'])
dsr_flows_csv = dsr_flows_csv.rename_axis('day')


# In[138]:


dsr_flows_csv.columns = [f'dsr_{col}' if col != 'day' and not col.startswith('surplus_buffer') else col for col in dsr_flows_csv.columns]


# In[139]:


dsr_df = dsr_flows_csv.merge(dsr_rate_csv, on=['day'], how='inner')


# In[140]:


dsr_df = dsr_df.rename(columns={'total_balance':'dai_total_balance'})
dsr_df = dsr_df.rename(columns={'non_dsr_balance':'dai_circulating'})


# In[141]:


dsr_df.describe()


# ## Daily comprehensive vault stats
# 
# These next api calls retrive daily metrics for each vault in makerdao

# In[142]:


#cum_bal_and_safetyprice_and_safetyvalue = vault_stats_6_20_through_6_21 


# In[143]:


cumbal_stats_path ='data/csv/cumbal.csv'
#cum_bal_and_safetyprice_and_safetyvalue.to_csv(cumbal_stats_path)
cumbal_csv = pd.read_csv(cumbal_stats_path, index_col = 'day', parse_dates=True)


# In[144]:


cumbal_csv[cumbal_csv['ilk']=='PSM-USDC-A']


# In[145]:


#debtbal_lpenalty_lratio = fetch_dune_data(3568425)


# In[146]:


debtbal_lpenalty_lratio_path = 'data/csv/debtbal_lpenalty_lratio.csv'
#debtbal_lpenalty_lratio.to_csv(debtbal_lpenalty_lratio_path)
debtbal_lpenalty_lratio_path_csv = pd.read_csv(debtbal_lpenalty_lratio_path, index_col = 'day', parse_dates=True)


# In[147]:


debtbal_lpenalty_lratio_path_csv.head()


# In[148]:


debtbal_lpenalty_lratio_path_csv_reset = debtbal_lpenalty_lratio_path_csv.reset_index()
debtbal_lpenalty_lratio_path_csv_clean = debtbal_lpenalty_lratio_path_csv_reset.drop_duplicates(subset=['day', 'ilk'], keep='last')


# In[149]:


debtbal_lpenalty_lratio_path_csv_clean = debtbal_lpenalty_lratio_path_csv_clean.drop(columns='Unnamed: 0')


# In[150]:


#dceiling_dfloor_scratio = fetch_dune_data(3568438)


# In[151]:


dceiling_dfloor_scratio_path = 'data/csv/dceiling_dfloor_scratio.csv'
#dceiling_dfloor_scratio.to_csv(dceiling_dfloor_scratio_path)
dceiling_dfloor_scratio_csv = pd.read_csv(dceiling_dfloor_scratio_path, index_col = 'day', parse_dates=True)


# In[152]:


#vault_market_price = fetch_dune_data(3568453)


# In[153]:


vault_market_price_path = 'data/csv/vault_market_price.csv'
#vault_market_price.to_csv(vault_market_price_path)
vault_market_price_csv = pd.read_csv(vault_market_price_path, index_col = 'day', parse_dates=True)


# In[154]:


vault_market_price_csv.head()


# In[155]:


vault_market_price_csv_reset = vault_market_price_csv.reset_index()
vault_market_price_csv_clean = vault_market_price_csv_reset.drop_duplicates(subset=['day', 'ilk'], keep='last')


# In[156]:


vault_market_price_csv_clean = vault_market_price_csv_clean.drop(columns=['Unnamed: 0'])


# In[157]:


cumbal_csv_reset = cumbal_csv.reset_index()
dceiling_dfloor_scratio_csv_reset = dceiling_dfloor_scratio_csv.reset_index()


# In[158]:


cumbal_csv_reset[cumbal_csv_reset['ilk'] == 'PSM-USDC-A']


# In[159]:


cumbal_csv_clean = cumbal_csv_reset.drop_duplicates(subset=['day', 'ilk'], keep='last')
dceiling_dfloor_scratio_csv_clean = dceiling_dfloor_scratio_csv_reset.drop_duplicates(subset=['day', 'ilk'], keep='last')


# In[160]:


cumbal_csv_clean[cumbal_csv_clean['ilk'] == 'PSM-USDC-A']


# In[161]:


cumbal_csv_clean = cumbal_csv_clean.drop(columns=['Unnamed: 0'])


# In[162]:


dceiling_dfloor_scratio_csv_clean = dceiling_dfloor_scratio_csv_clean.drop(columns=['Unnamed: 0'])


# In[163]:


comprehensive_vault_stats = pd.merge(cumbal_csv_clean, dceiling_dfloor_scratio_csv_clean, on=['day', 'ilk'], how='left')


# In[164]:


comprehensive_vault_stats = comprehensive_vault_stats[comprehensive_vault_stats['day'] < '2024-03-22']


# In[165]:


comprehensive_vault_stats = comprehensive_vault_stats.merge(vault_market_price_csv_clean, on=['day', 'ilk'], how='left' )


# In[166]:


comprehensive_vault_stats = comprehensive_vault_stats.merge(debtbal_lpenalty_lratio_path_csv_clean, on=['day', 'ilk'], how='left')


# In[167]:


comprehensive_vault_stats[comprehensive_vault_stats['ilk'] == 'PSM-USDC-A']


# In[168]:


comprehensive_vault_stats['day'] = pd.to_datetime(comprehensive_vault_stats['day'])
sf_history_csv_clean['day'] = pd.to_datetime(sf_history_csv_clean['day'])


# In[169]:


sf_history_csv_clean[sf_history_csv_clean['ilk']=='WBTC-A']


# In[170]:


ir_csv[ir_csv['ilk'] == 'WBTC-A']


# In[171]:


comprehensive_vault_stats.columns


# In[172]:


comprehensive_vault_stats = comprehensive_vault_stats.merge(ir_csv, on=['day', 'ilk'], how='left')


# In[173]:


wtbca1 = comprehensive_vault_stats[comprehensive_vault_stats['ilk'] == 'WBTC-A']


# In[174]:


wtbca1


# In[175]:


wtbca1[['day','daily_revenues']].tail(10)


# In[176]:


comprehensive_vault_stats = comprehensive_vault_stats.merge(sf_history_csv_clean, on=['day', 'ilk'], how='left')


# In[177]:


comprehensive_vault_stats


# In[178]:


comprehensive_vault_stats[comprehensive_vault_stats['ilk'] == 'WBTC-A']


# # Define the columns to be backward filled
# bfill_columns = ['annualized', 'annualized_revenues', 'dart','total_ann_revenues']
# 
# # Apply bfill within each 'ilk' group for the specified columns
# for column in bfill_columns:
#     comprehensive_vault_stats[column] = comprehensive_vault_stats.groupby('ilk')[column].bfill()
# 
# # Optionally, apply ffill if you want to ensure there are no remaining NAs at the start of the dataset
# for column in bfill_columns:
#     comprehensive_vault_stats[column] = comprehensive_vault_stats.groupby('ilk')[column].ffill()
# 
# # Check results
# #print(comprehensive_vault_stats[comprehensive_vault_stats['ilk'] == 'WBTC-A'][['day', 'ilk'] + bfill_columns])
# 

# ## More Maker Protocol Engingeering 

# In[179]:


comprehensive_vault_stats[comprehensive_vault_stats['ilk'] == 'WBTC-A']


# In[180]:


def localize_or_convert(df, column_name, timezone='UTC'):
    # Check if the first datetime object in the column is naive or aware
    if df[column_name].dt.tz is None:
        # If naive, use tz_localize
        df[column_name] = df[column_name].dt.tz_localize(timezone)
    else:
        # If aware, use tz_convert
        df[column_name] = df[column_name].dt.tz_convert(timezone)
    return df

# Apply the function to your DataFrames



# In[181]:


comprehensive_vault_stats = localize_or_convert(comprehensive_vault_stats, 'day')
dai_maturity_df_reindexed = localize_or_convert(dai_maturity_df_reindexed, 'day')


# In[182]:


stable_coin_ratios.index


# In[183]:


comprehensive_vault_stats.tail(1000)


# In[184]:


comprehensive_vault_stats.fillna(0,inplace=True)


# In[185]:


comprehensive_vault_stats = comprehensive_vault_stats.rename(columns={'annualized':'annualized stability fee'})


# In[186]:


def determine_status(row):
    # The vault is considered closed if 'dai_ceiling' is 0
    # This includes the first day if 'dai_ceiling' starts at 0 or if it drops to 0 from a nonzero value
    if pd.isnull(row['prev_dai_ceiling']) and row['dai_ceiling'] == 0:
        return 'Closed'
    elif row['prev_dai_ceiling'] >= 0 and row['dai_ceiling'] == 0:
        return 'Closed'
    # Check if 'safety_price' is 0
    elif row['safety_price'] == 0:
        return 'Closed'
    else:
        return 'Open'


# In[187]:


comprehensive_vault_stats['market_price'] = np.where(comprehensive_vault_stats['liquidation_ratio'].isnull(), comprehensive_vault_stats['safety_price'], comprehensive_vault_stats['market_price'])


# In[188]:


comprehensive_vault_stats = comprehensive_vault_stats.sort_values('day')
comprehensive_vault_stats['prev_dai_ceiling'] = comprehensive_vault_stats.groupby('ilk')['dai_ceiling'].shift(1)


# In[189]:


comprehensive_vault_stats['status'] = comprehensive_vault_stats.apply(determine_status, axis=1)


# In[190]:


comprehensive_vault_stats.tail(1000)


# In[191]:


comprehensive_vault_stats.columns


# In[192]:


comprehensive_vault_stats['market_collateral_ratio'] = np.where(comprehensive_vault_stats['status']=='Open',((comprehensive_vault_stats['usd_safety_value'] *comprehensive_vault_stats['liquidation_ratio']) / comprehensive_vault_stats['debt_balance']), np.nan)
comprehensive_vault_stats['market_collateral_ratio'] = np.where(comprehensive_vault_stats['debt_balance']==0,0, comprehensive_vault_stats['market_collateral_ratio'])

comprehensive_vault_stats['safety_collateral_ratio'] = np.where(comprehensive_vault_stats['status']=='Open',comprehensive_vault_stats['safety_collateral_ratio'], np.nan)
comprehensive_vault_stats['safety_collateral_ratio'] = np.where(comprehensive_vault_stats['status'] == 'Closed', 0, comprehensive_vault_stats['safety_collateral_ratio'])
comprehensive_vault_stats['safety_collateral_ratio'] = np.where(comprehensive_vault_stats['debt_balance'] <= 0, 0, comprehensive_vault_stats['safety_collateral_ratio'])
comprehensive_vault_stats[['annualized_revenues', 'dart']] = np.where(comprehensive_vault_stats[['annualized_revenues', 'dart']] <= 0, 0, comprehensive_vault_stats[['annualized_revenues', 'dart']])

comprehensive_vault_stats['collateral_usd'] = comprehensive_vault_stats['usd_safety_value'] * comprehensive_vault_stats['liquidation_ratio']
comprehensive_vault_stats['hypothetical_dai_ceiling'] = comprehensive_vault_stats['collateral_usd']* (comprehensive_vault_stats['liquidation_ratio'] / 2)


# In[193]:


#replace values when vault is closed to 0?


# In[194]:


def clean_small_values(value, threshold=1e-8):
    """
    Set small negative and positive values within a specified threshold to zero.

    Parameters:
    - value: The value to be cleaned.
    - threshold: Values within +/- this threshold will be set to zero.

    Returns:
    - The cleaned value.
    """
    if -threshold < value < threshold:
        return 0.0
    else:
        return value

# Apply this function to the entire DataFrame
comprehensive_vault_stats[['cumulative_collateral','usd_safety_value','collateral_usd','hypothetical_dai_ceiling','debt_balance']] = comprehensive_vault_stats[['cumulative_collateral','usd_safety_value','collateral_usd','hypothetical_dai_ceiling','debt_balance']].applymap(lambda x: clean_small_values(x))


# In[195]:


comprehensive_vault_stats = comprehensive_vault_stats[comprehensive_vault_stats['ilk'] != 'SAI']


# UNIV2ETHUSDT-A is an outlier and will be removed from dataset.  

# In[196]:


comprehensive_vault_stats[comprehensive_vault_stats['safety_collateral_ratio'] < 0 ]


# In[197]:


comprehensive_vault_stats = comprehensive_vault_stats[comprehensive_vault_stats['ilk'] != 'UNIV2ETHUSDT-A']


# In[198]:


comprehensive_vault_stats['debt_balance'].describe()


# In[199]:


#pd.set_option('display.max_columns', None)  # Show all columns
#pd.set_option('display.max_rows', None)  # Show all rows
#pd.set_option('display.max_colwidth', None)  # Show full content of each column


# In[200]:


comprehensive_vault_stats['market_collateral_ratio'].describe()


# In[201]:


comprehensive_vault_stats['safety_collateral_ratio'].describe()


# In[202]:


type(comprehensive_vault_stats['debt_balance'].iloc[-1])


# In[203]:


# Check entries where `debt_balance` is zero but `market_collateral_ratio` is not
anomalies = comprehensive_vault_stats[
    (comprehensive_vault_stats['debt_balance'] == 0) & 
    (comprehensive_vault_stats['market_collateral_ratio'] != 0)
]
#print(anomalies[['day', 'ilk', 'market_collateral_ratio', 'debt_balance']])


# In[204]:


comprehensive_vault_stats[['day','ilk','market_collateral_ratio','debt_balance']][comprehensive_vault_stats['market_collateral_ratio'] > 3.31 ].sort_values('market_collateral_ratio')


# In[205]:


comprehensive_vault_stats[['day','ilk','safety_collateral_ratio']][comprehensive_vault_stats['safety_collateral_ratio'] > 100 ]


# In[206]:


comprehensive_vault_stats[['day','ilk','debt_balance']][comprehensive_vault_stats['debt_balance'] < 0 ]


# In[207]:


comprehensive_vault_stats = comprehensive_vault_stats[comprehensive_vault_stats['ilk'] != 'DIRECT-AAVEV2-DAI']


# In[208]:


comprehensive_vault_stats[['day','ilk','debt_balance']][comprehensive_vault_stats['debt_balance'] < 0 ]


# In[209]:


comprehensive_vault_stats


# In[210]:


open_vaults = comprehensive_vault_stats[comprehensive_vault_stats['status'] == 'Open']

# Group by 'day' and count the entries for each day
open_ilk_count_per_day = open_vaults.groupby('day').size().reset_index(name='total_open_ilk_count')

# Merge this count back into the original DataFrame on the 'day' column
comprehensive_vault_stats = comprehensive_vault_stats.merge(open_ilk_count_per_day, on='day', how='left')

# Fill NaN values with 0 for days with no open ilks
comprehensive_vault_stats['total_open_ilk_count'] = comprehensive_vault_stats['total_open_ilk_count'].fillna(0).astype(int)


# In[211]:


comprehensive_vault_stats['total_open_ilk_count'].describe()


# In[212]:


comprehensive_vault_stats['total_open_ilk_count'].plot()


# In[213]:


comprehensive_vault_stats['status'].describe()


# In[214]:


comprehensive_vault_stats[['liquidation_ratio','liquidation_penalty','annualized stability fee',
       'annualized_revenues', 'dart', 'total_ann_revenues',
       'prev_dai_ceiling']].describe()


# In[215]:


closed_vaults = comprehensive_vault_stats[comprehensive_vault_stats['status'] == 'Closed']


# In[216]:


zero_balances = comprehensive_vault_stats[comprehensive_vault_stats['cumulative_collateral'] == 0]


# In[217]:


usdt_a = comprehensive_vault_stats[comprehensive_vault_stats['ilk']=='USDT-A']


# In[218]:


usdt_a.tail()


# In[219]:


zero_balances['ilk'].unique()


# In[220]:


closed_vaults['ilk'].unique()


# In[221]:


wbtc = comprehensive_vault_stats[comprehensive_vault_stats['ilk']=='WBTC-A']
eth_a = comprehensive_vault_stats[comprehensive_vault_stats['ilk']=='ETH-A']


# In[222]:


wbtc['annualized_revenues']


# In[223]:


eth_a['status']


# In[224]:


eth_a_df = eth_a['dai_ceiling'].to_frame('eth_a_dai_ceiling')


# In[225]:


eth_a_zero_dai_ceiling = eth_a[eth_a['dai_ceiling'] == 0]


# In[226]:


eth_a_zero_dai_ceiling.head()


# In[227]:


eth_a_df.head()


# In[228]:


wbtc = wbtc.set_index('day')


# In[229]:


comprehensive_vault_stats[comprehensive_vault_stats['ilk']=='PSM-USDC-A']


# In[230]:


comprehensive_vault_stats.columns


# In[231]:


comprehensive_vault_stats['ilk'].sort_values().unique()


# In[232]:


comprehensive_vault_stats['status'].head()


# In[233]:


comprehensive_vault_stats[comprehensive_vault_stats['ilk']=='PSM-USDC-A']


# In[234]:


comprehensive_vault_stats[comprehensive_vault_stats['ilk'] =='PSM-USDC-A']


# In[235]:


comprehensive_vault_stats['BTC Vault'] = comprehensive_vault_stats['ilk'].isin(['WBTC-A','WBTC-B','WBTC-C','RENBTC-A'])
comprehensive_vault_stats['ETH Vault'] = comprehensive_vault_stats['ilk'].isin(['ETH-A','ETH-B','ETH-C'])
comprehensive_vault_stats['stETH Vault'] = comprehensive_vault_stats['ilk'].isin(['WSTETH-A', 'WSTETH-B','RETH-A'])
comprehensive_vault_stats['Stablecoin Vault'] = comprehensive_vault_stats['ilk'].isin(['USDC-A', 'USDC-B', 'USDT-A','GUSD-A','PAXUSD-A','TUSD-A'])
comprehensive_vault_stats['Altcoin Vault'] = comprehensive_vault_stats['ilk'].isin(['AAVE-A', 'BAL-A', 'BAT-A', 'COMP-A','GNO-A','KNC-A', 'LINK-A', 'LRC-A', 'MANA-A', 'MATIC-A','UNI-A','YFI-A', 'ZRX-A'])
comprehensive_vault_stats['LP Vault'] = comprehensive_vault_stats['ilk'].isin(['CRVV1ETHSTETH-A','GUNIV3DAIUSDC1-A', 'GUNIV3DAIUSDC2-A','UNIV2AAVEETH-A', 'UNIV2DAIETH-A','UNIV2DAIUSDC-A', 'UNIV2DAIUSDT-A', 'UNIV2LINKETH-A','UNIV2UNIETH-A', 'UNIV2USDCETH-A', 'UNIV2WBTCDAI-A','UNIV2WBTCETH-A'])
comprehensive_vault_stats['RWA Vault'] = comprehensive_vault_stats['ilk'].isin(['RWA001-A', 'RWA002-A', 'RWA003-A', 'RWA004-A','RWA005-A', 'RWA006-A', 'RWA008-A', 'RWA009-A', 'RWA012-A','RWA013-A'])
comprehensive_vault_stats['PSM Vault'] = comprehensive_vault_stats['ilk'].isin(['PSM-GUSD-A', 'PSM-PAX-A', 'PSM-USDC-A'])


# In[236]:


lps = comprehensive_vault_stats[comprehensive_vault_stats['ilk'].isin(['CRVV1ETHSTETH-A','GUNIV3DAIUSDC1-A', 'GUNIV3DAIUSDC2-A','PSM-GUSD-A', 'PSM-PAX-A', 'PSM-USDC-A','UNIV2AAVEETH-A', 'UNIV2DAIETH-A','UNIV2DAIUSDC-A', 'UNIV2DAIUSDT-A', 'UNIV2LINKETH-A','UNIV2UNIETH-A', 'UNIV2USDCETH-A', 'UNIV2WBTCDAI-A','UNIV2WBTCETH-A'])]
lps.columns
lps[['day','ilk','annualized stability fee','collateral_usd']].tail(20)


# In[237]:


psms = comprehensive_vault_stats[comprehensive_vault_stats['ilk'].isin(['PSM-GUSD-A', 'PSM-PAX-A', 'PSM-USDC-A'])]
psms.tail(50)


# In[238]:


comprehensive_vault_stats.duplicated().sum()


# In[239]:


comprehensive_vault_stats[['day','ilk','PSM Vault']].tail(50)


# In[240]:


#then we join this, and divide category usd by total daily usd to get percentage of portfolio 
total_daily_usd_bal = comprehensive_vault_stats.groupby('day')['collateral_usd'].sum()
total_daily_usd_bal


# In[241]:


vaultsbtc = comprehensive_vault_stats[comprehensive_vault_stats['ilk'].isin(['WBTC-A','WBTC-B','WBTC-C','RENBTC-A'])]
vaultspsm = comprehensive_vault_stats[comprehensive_vault_stats['ilk'].isin(['PSM-GUSD-A', 'PSM-PAX-A', 'PSM-USDC-A'])]
vaultslp = comprehensive_vault_stats[comprehensive_vault_stats['ilk'].isin(['CRVV1ETHSTETH-A','GUNIV3DAIUSDC1-A', 'GUNIV3DAIUSDC2-A','UNIV2AAVEETH-A', 'UNIV2DAIETH-A','UNIV2DAIUSDC-A', 'UNIV2DAIUSDT-A', 'UNIV2LINKETH-A','UNIV2UNIETH-A', 'UNIV2USDCETH-A', 'UNIV2WBTCDAI-A','UNIV2WBTCETH-A'])]
vaultsalt = comprehensive_vault_stats[comprehensive_vault_stats['ilk'].isin(['AAVE-A', 'BAL-A', 'BAT-A', 'COMP-A','GNO-A','KNC-A', 'LINK-A', 'LRC-A', 'MANA-A', 'MATIC-A','UNI-A','YFI-A', 'ZRX-A'])]


# In[242]:


psm_usdc = vaultspsm[vaultspsm['ilk']=='PSM-USDC-A']
psm_usdc['dai_ceiling'].plot()


# In[243]:


#print(vaultspsm['ilk'].unique())
#print(vaultslp['ilk'].unique())


# In[244]:


vaultslp['dai_ceiling'].plot()


# In[245]:


vaultsbtc.columns


# In[246]:


#print(vaultsbtc.columns)
#print(vaultsbtc.index)


# In[247]:


btc_vault_daily_collateral_sum = vaultsbtc.groupby('day')['collateral_usd'].sum().reset_index()
btc_vault_daily_collateral_sum


# In[248]:


vaultsbtc['liquidation_penalty'].describe()


# In[249]:


btc_vault_data = comprehensive_vault_stats[comprehensive_vault_stats['BTC Vault']]
btc_vault_daily_sum = btc_vault_data.groupby('day')['collateral_usd'].sum().reset_index(name='BTC Vault_Collateral_usd')
btc_vault_daily_sum


# In[250]:


sum_columns = [
    'dai_ceiling', 'dai_floor', 'debt_balance', 'daily_revenues',
    'annualized_revenues', 'dart', 
    'prev_dai_ceiling', 'collateral_usd', 'hypothetical_dai_ceiling'
]

btc_vault_data = comprehensive_vault_stats[comprehensive_vault_stats['BTC Vault']]

# Dictionary to hold the results with new column names
sum_results = {}

# Group by 'day' and sum each column, assigning new column names with f-string
for column in sum_columns:
    sum_results[f'BTC Vault_{column}'] = btc_vault_data.groupby('day')[column].sum()

# Convert the dictionary to a DataFrame
btc_vault_daily_sum = pd.DataFrame(sum_results)

# Resetting index to get 'day' as a column if needed
btc_vault_daily_sum.reset_index(inplace=True)

# Display the resulting DataFrame
btc_vault_daily_sum


# In[251]:


# Assuming 'comprehensive_vault_stats' is already loaded and available

# Columns to sum
sum_columns = [
    'dai_ceiling', 'dai_floor', 'debt_balance', 'daily_revenues',
    'annualized_revenues', 'dart', 
    'prev_dai_ceiling', 'collateral_usd', 'hypothetical_dai_ceiling'
]

categories = [
    'BTC Vault', 'ETH Vault', 'stETH Vault', 'Stablecoin Vault',
    'Altcoin Vault', 'LP Vault', 'RWA Vault','PSM Vault'
]

# Dictionary to hold DataFrames for each category
category_dfs = {}

# Loop through each category to filter data, sum by day, and create a DataFrame
for category in categories:
    # Filter data for the current category
    category_data = comprehensive_vault_stats[comprehensive_vault_stats[category]]

    # Dictionary to hold the results with new column names
    sum_results = {}

    # Group by 'day' and sum each column, assigning new column names with f-string
    for column in sum_columns:
        sum_results[f'{category}_{column}'] = category_data.groupby('day')[column].sum()

    # Convert the dictionary to a DataFrame and reset index to make 'day' a column
    category_df = pd.DataFrame(sum_results).reset_index()

    # Store the DataFrame in the dictionary using the category as the key
    category_dfs[category] = category_df




# In[252]:


category_dfs.keys()


# In[253]:


category_dfs['BTC Vault']


# In[254]:


# Example to access the DataFrame for 'BTC Vault'
btc_vault_daily_sum = category_dfs['BTC Vault']
eth_vault_daily_sum = category_dfs['ETH Vault']
stETH_vault_daily_sum = category_dfs['stETH Vault']
Stablecoin_Vault = category_dfs['Stablecoin Vault']
Altcoin_Vault = category_dfs['Altcoin Vault']
LP_Vault = category_dfs['LP Vault']
RWA_Vault = category_dfs['RWA Vault']
PSM_vault = category_dfs['PSM Vault']


# In[255]:


btc_vault_daily_sum.head(20)


# In[256]:


aggregated_vaults = pd.merge(btc_vault_daily_sum, eth_vault_daily_sum, on=['day'],how='inner')
aggregated_vaults = aggregated_vaults.merge(stETH_vault_daily_sum, on=['day'],how='inner')
aggregated_vaults = aggregated_vaults.merge(Stablecoin_Vault, on=['day'],how='inner')
aggregated_vaults = aggregated_vaults.merge(Altcoin_Vault, on=['day'],how='inner')
aggregated_vaults = aggregated_vaults.merge(LP_Vault, on=['day'],how='inner')
aggregated_vaults = aggregated_vaults.merge(RWA_Vault, on=['day'],how='inner')
aggregated_vaults = aggregated_vaults.merge(PSM_vault, on=['day'],how='inner')


# In[257]:


aggregated_vaults.isna().sum().sum()


# In[258]:


aggregated_vaults.columns


# In[259]:


aggregated_vaults['Vaults Total USD Value'] = aggregated_vaults[['BTC Vault_collateral_usd','ETH Vault_collateral_usd','stETH Vault_collateral_usd',
    'Stablecoin Vault_collateral_usd','Altcoin Vault_collateral_usd', 'LP Vault_collateral_usd','RWA Vault_collateral_usd','PSM Vault_collateral_usd'
    ]].sum(axis=1)


# aggregated_vaults['BTC Debt Ratio'] = aggregated_vaults['BTC Vault_debt_balance'] / aggregated_vaults['BTC Vault_collateral_usd']
# aggregated_vaults['ETH Debt Ratio']
# aggregated_vaults['stETH Debt Ratio']
# aggregated_vaults['Stablecoin Debt Ratio']
# aggregated_vaults['Altcoin Debt Ratio']
# aggregated_vaults['LP Debt Ratio']
# aggregated_vaults['RWA Debt Ratio']
# aggregated_vaults['PSM Debt Ratio']

# aggregated_vaults['BTC Debt Ratio'] = aggregated_vaults['BTC Vault_debt_balance'] / aggregated_vaults['BTC Vault_collateral_usd']
# aggregated_vaults['BTC Debt Ratio'].describe()

# In[260]:


# Calculate the percentage of total for each vault type
vault_types = ['BTC Vault_collateral_usd', 'ETH Vault_collateral_usd', 'stETH Vault_collateral_usd',
               'Stablecoin Vault_collateral_usd', 'Altcoin Vault_collateral_usd', 'LP Vault_collateral_usd',
               'RWA Vault_collateral_usd', 'PSM Vault_collateral_usd']

for vault in vault_types:
    aggregated_vaults[f'{vault} % of Total'] = (aggregated_vaults[vault] / aggregated_vaults['Vaults Total USD Value']) * 100

# This will add each vault's percentage of the total USD value as new columns in your DataFrame


# In[261]:


aggregated_vaults.columns


# In[262]:


# Columns of interest
columns = ['BTC Vault_collateral_usd', 'ETH Vault_collateral_usd', 'stETH Vault_collateral_usd',
           'Stablecoin Vault_collateral_usd', 'Altcoin Vault_collateral_usd', 'LP Vault_collateral_usd',
           'RWA Vault_collateral_usd', 'PSM Vault_collateral_usd']

# Check for negative values in these columns
negative_values = (aggregated_vaults[columns] < 0).any()

#print(negative_values)


# In[263]:


aggregated_vaults


# In[264]:


aggregated_vaults.columns


# In[265]:


aggregated_vaults[['BTC Vault_collateral_usd','ETH Vault_collateral_usd','stETH Vault_collateral_usd','Stablecoin Vault_collateral_usd'
                  ,'Altcoin Vault_collateral_usd','LP Vault_collateral_usd','RWA Vault_collateral_usd','PSM Vault_collateral_usd']].plot()


# In[266]:


aggregated_vaults[['BTC Vault_collateral_usd % of Total',
       'ETH Vault_collateral_usd % of Total',
       'stETH Vault_collateral_usd % of Total',
       'Stablecoin Vault_collateral_usd % of Total',
       'Altcoin Vault_collateral_usd % of Total',
       'LP Vault_collateral_usd % of Total',
       'RWA Vault_collateral_usd % of Total',
       'PSM Vault_collateral_usd % of Total']].plot()


# In[267]:


# Assuming comprehensive_vault_stats is your DataFrame

# Columns to sum
sum_columns = [
    'dai_ceiling', 'dai_floor', 'debt_balance', 'daily_revenues',
    'annualized_revenues', 'dart', 'total_ann_revenues', 
    'prev_dai_ceiling', 'collateral_usd', 'hypothetical_dai_ceiling'
]

# List of vault categories
categories = [
    'BTC Vault', 'ETH Vault', 'stETH Vault', 'Stablecoin Vault',
    'Altcoin Vault', 'LP Vault', 'RWA Vault'
]

# Melt the DataFrame to turn category columns into values
melted_data = comprehensive_vault_stats.melt(
    id_vars=['day'] + sum_columns + ['status'],  # Include 'status' to filter by open status in the next steps
    value_vars=categories,
    var_name='Vault Category',
    value_name='Is In Category'
)

# Filter only the entries marked as True for being in a category
filtered_data = melted_data[melted_data['Is In Category']]

# Calculate the sums for each category and each day
pivot_table = pd.pivot_table(
    filtered_data,
    index='day',
    columns='Vault Category',
    values=sum_columns,
    aggfunc='sum',  # Change to 'mean' or any other function as needed
    fill_value=0
)

# Optionally, handle open status here if necessary
# This can involve additional filtering or adjustments based on the 'status' column
# For example, you might want to exclude certain records or adjust weights before summing

# Reset index if needed to flatten the DataFrame after pivoting
pivot_table.reset_index(inplace=True)

# Display the results
#print(pivot_table)


# In[268]:


# Assuming 'pivot_table' is your current DataFrame after pivoting
pivot_table.columns = ['_'.join(col).strip() for col in pivot_table.columns.values]

# Reset the index to flatten the DataFrame if necessary
pivot_table.reset_index(inplace=True)

# Show the result
pivot_table


# In[269]:


categories = [
    'BTC Vault', 'ETH Vault', 'stETH Vault', 'Stablecoin Vault',
    'Altcoin Vault', 'LP Vault', 'RWA Vault','PSM Vault'
]

sum_columns = [
    'dai_ceiling', 'dai_floor', 'debt_balance', 'daily_revenues',
    'annualized_revenues', 'dart', 'total_ann_revenues', 
    'prev_dai_ceiling', 'collateral_usd', 'hypothetical_dai_ceiling'
]

sums_by_category = comprehensive_vault_stats.groupby(['day'] + categories)[sum_columns].sum().reset_index()


# In[270]:


sums_by_category


# In[271]:


#print(comprehensive_vault_stats.head())
#print(comprehensive_vault_stats['status'].unique())  # Check unique statuses
#print(comprehensive_vault_stats['day'].describe())   # Get an overview of the 'day' column


# In[272]:


comprehensive_vault_stats.columns


# In[273]:


comprehensive_vault_stats[comprehensive_vault_stats['ilk']=='GUNIV3DAIUSDC2-A']['annualized stability fee']


# In[274]:


wbtc['ilk'].unique()


# In[275]:


def weighted_average(data, value_column, weight_column):
    d = data.dropna(subset=[value_column, weight_column])
    if d.empty:
        return np.nan
    numer = (d[value_column] * d[weight_column]).sum()
    denom = d[weight_column].sum()
    return numer / denom if denom != 0 else np.nan


# In[276]:


# Test filtering for a single category

parameters = [
    'safety_collateral_ratio', 'liquidation_penalty',
    'liquidation_ratio', 'annualized stability fee',
    'market_collateral_ratio','market_price'
]
test_data = comprehensive_vault_stats[(comprehensive_vault_stats['BTC Vault']) & (comprehensive_vault_stats['status'] == 'Open')]
#print(test_data.head())  # Check the output

# Test grouping and weighted calculation for one day
if not test_data.empty:
    example_group = test_data.groupby('day').get_group(list(test_data['day'])[0])
    example_result = {param: weighted_average(example_group, param, 'collateral_usd') for param in parameters}
    #print(example_result)


# In[277]:


comprehensive_vault_stats['ilk'].unique()


# In[278]:


comprehensive_vault_stats['ilk'][comprehensive_vault_stats['status'] == 'Open'].unique()


# In[279]:


parameters = [
    'safety_collateral_ratio', 'liquidation_penalty',
    'liquidation_ratio', 'annualized stability fee',
    'market_collateral_ratio', 'market_price'
]

categories = [
    'BTC Vault', 'ETH Vault', 'stETH Vault', 'Stablecoin Vault',
    'Altcoin Vault', 'LP Vault', 'RWA Vault', 'PSM Vault'
]

# Dictionary to store results
results = {category: [] for category in categories}

# Loop over each day and category
for day, day_group in comprehensive_vault_stats.groupby('day'):
    for category in categories:
        # Filter the day's data for open vaults in this category
        category_data = day_group[(day_group[category] == True) & (day_group['status'] == 'Open')]
        
        if not category_data.empty:
            # Calculate weighted averages for each parameter using the function
            daily_results = {}
            for param in parameters:
                if 'collateral_usd' in category_data.columns and param in category_data.columns:
                    weighted_avg = weighted_average(category_data, param, 'collateral_usd')
                    daily_results[f'{category}_{param}'] = weighted_avg
            
            daily_results['day'] = day
            results[category].append(daily_results)

# Convert list of dictionaries to DataFrames
for category in results:
    if results[category]:
        results[category] = pd.DataFrame(results[category])
        results[category]['day'] = pd.to_datetime(results[category]['day'])  # Convert 'day' to datetime if necessary

# Example: Output results for 'BTC Vault'
if 'BTC Vault' in results and not results['BTC Vault'].empty:
    print(results['BTC Vault'].tail())


# In[280]:


results['BTC Vault'].tail()





results['BTC Vault']['BTC Vault_market_price']


# In[282]:


btc_wa = results['BTC Vault']
eth_wa = results['ETH Vault']
steth_wa = results['stETH Vault']
stablecoin_wa = results['Stablecoin Vault']
altcoin_wa = results['Altcoin Vault']
lp_wa = results['LP Vault']
rwa_wa = results['RWA Vault']
psm_wa = results['PSM Vault']


# In[283]:


lp_wa['LP Vault_market_price'].describe()


# In[284]:


lp_wa['LP Vault_market_price'].pct_change().describe()


# In[285]:


lp_wa['LP Vault_annualized stability fee'].plot()


# In[286]:


eth_wa


# In[287]:


psm_wa['PSM Vault_annualized stability fee'].plot()


# In[288]:


vault_wa = pd.merge(btc_wa, eth_wa, on=['day'], how='right')
vault_wa = vault_wa.merge(steth_wa, on=['day'], how='left')
vault_wa = vault_wa.merge(stablecoin_wa, on=['day'], how='left')
vault_wa = vault_wa.merge(altcoin_wa, on=['day'], how='left')
vault_wa = vault_wa.merge(lp_wa, on=['day'], how='left')
vault_wa = vault_wa.merge(rwa_wa, on=['day'], how='left')
vault_wa = vault_wa.merge(psm_wa, on=['day'], how='left')

vault_wa


# In[289]:


vault_wa.fillna(0,inplace=True)


# In[290]:


vault_wa.columns


# In[291]:


vault_wa[[
    'BTC Vault_safety_collateral_ratio', 'ETH Vault_safety_collateral_ratio','stETH Vault_safety_collateral_ratio','Stablecoin Vault_safety_collateral_ratio',
    'Altcoin Vault_safety_collateral_ratio','LP Vault_safety_collateral_ratio','RWA Vault_safety_collateral_ratio','PSM Vault_safety_collateral_ratio'
    
]]


# In[292]:


stablecoin_wa.set_index('day', inplace=True)


# In[293]:


lp_wa


# In[294]:


stablecoin_wa.ffill()


# In[295]:


vault_wa['day']


# In[296]:


vault_wa[['day','BTC Vault_market_price','ETH Vault_market_price','stETH Vault_market_price','Stablecoin Vault_market_price'
                  ,'Altcoin Vault_market_price','LP Vault_market_price','RWA Vault_market_price','PSM Vault_market_price']]


# In[297]:


aggregated_vaults['day']


# In[298]:


aggregated_vaults = aggregated_vaults.merge(vault_wa, on=['day'],how='inner')
list(aggregated_vaults.columns)


# In[299]:


aggregated_vaults


# In[300]:


# Calculate the number of null values in each column
null_counts = aggregated_vaults.isna().sum()

# Display the number of nulls for each column
#print(null_counts)


# In[301]:


# Display only those columns that have null values
null_columns = null_counts[null_counts > 0]
#print(null_columns)


# In[302]:


aggregated_vaults.isna().sum().sum()


# In[303]:


no_nan_vaults = comprehensive_vault_stats.copy()
no_nan_vaults = no_nan_vaults.fillna(0)


# In[304]:


no_nan_vaults.columns


# In[305]:


comprehensive_vault_stats.shape[0]


# In[306]:


comprehensive_vault_stats['safety_collateral_ratio'].describe()


# In[307]:


comprehensive_vault_stats.columns


# In[308]:


comprehensive_vault_stats[comprehensive_vault_stats['ilk']=='ETH-A']


# In[309]:


no_nan_vaults[no_nan_vaults['ilk']=='WSTETH-A'].head()


# In[310]:


top_10_ilks = top_10_vaults.index.tolist()


# In[311]:


top_10_ilks


# In[312]:


no_nan_vaults.columns


# In[313]:


topvaults = no_nan_vaults[no_nan_vaults['ilk'].isin(top_10_ilks)]


# In[314]:


topvaults


# In[315]:


wbtc_a_vault = no_nan_vaults[no_nan_vaults['ilk']=='WBTC-A']
eth_a_vault = no_nan_vaults[no_nan_vaults['ilk']=='ETH-A']
wsteth_a_vault = no_nan_vaults[no_nan_vaults['ilk']=='WSTETH-A']
eth_c_vault = no_nan_vaults[no_nan_vaults['ilk']=='ETH-C']
eth_b_vault = no_nan_vaults[no_nan_vaults['ilk']=='ETH-B']


# In[316]:


eth_a_vault['cumulative_collateral'].plot()


# In[317]:


eth_a_vault[['collateral_usd','dai_ceiling','debt_balance']].plot()


# In[318]:


eth_c_vault['cumulative_collateral'].plot()


# In[319]:


eth_b_vault['cumulative_collateral'].plot()


# In[320]:


#We start with eth-a vault, which has long history and has generated most revenues 
eth_a_vault.columns


# In[321]:


eth_a_vault.describe()


# In[322]:


eth_a_vault.columns


# In[323]:


def calculate_moving_averages(df, columns, windows=[7, 30, 90]):
    """
    Applies moving averages to specified columns of a DataFrame.

    Parameters:
    - df: pandas.DataFrame, the DataFrame containing the data.
    - columns: list, column names to apply the moving averages.
    - windows: list of integers, the window sizes for the moving averages.

    Returns:
    - df: pandas.DataFrame, the DataFrame with new columns for moving averages.
    """
    for window in windows:
        for column in columns:
            ma_column_name = f'{column}_{window}d_ma'
            df[ma_column_name] = df[column].rolling(window=window).mean()
    return df


# In[324]:


eth_a_vault = calculate_moving_averages(eth_a_vault,['market_price','cumulative_collateral','collateral_usd','debt_balance','safety_collateral_ratio','market_collateral_ratio','daily_revenues','annualized stability fee'])
eth_b_vault = calculate_moving_averages(eth_b_vault,['market_price','cumulative_collateral','collateral_usd','debt_balance','safety_collateral_ratio','market_collateral_ratio','daily_revenues','annualized stability fee'])
wbtc_a_vault = calculate_moving_averages(wbtc_a_vault,['market_price','cumulative_collateral','collateral_usd','debt_balance','safety_collateral_ratio','market_collateral_ratio','daily_revenues','annualized stability fee'])
wsteth_a_vault = calculate_moving_averages(wsteth_a_vault,['market_price','cumulative_collateral','collateral_usd','debt_balance','safety_collateral_ratio','market_collateral_ratio','daily_revenues','annualized stability fee'])
eth_c_vault = calculate_moving_averages(eth_c_vault,['market_price','cumulative_collateral','collateral_usd','debt_balance','safety_collateral_ratio','market_collateral_ratio','daily_revenues','annualized stability fee'])


# In[325]:


aggregated_vaults = calculate_moving_averages(aggregated_vaults, aggregated_vaults.drop(columns=['day']).columns)


# In[326]:


aggregated_vaults.columns


# In[327]:


eth_a_vault.columns


# #7 day MA
# eth_a_vault['market_price_7d_ma'] = eth_a_vault['market_price'].rolling(window=7).mean()
# eth_a_vault['collateral_usd_7d_ma'] = eth_a_vault['collateral_usd'].rolling(window=7).mean()
# eth_a_vault['debt_balance_7d_ma'] = eth_a_vault['debt_balance'].rolling(window=7).mean()
# eth_a_vault['safety_collateral_ratio_7d_ma'] = eth_a_vault['safety_collateral_ratio'].rolling(window=7).mean()
# eth_a_vault['market_collateral_ratio_7d_ma'] = eth_a_vault['market_collateral_ratio'].rolling(window=7).mean()
# eth_a_vault['daily_revenues_7d_ma'] = eth_a_vault['daily_revenues'].rolling(window=7).mean()
# 
# # Calculate 30-day moving averages
# eth_a_vault['market_price_30d_ma'] = eth_a_vault['market_price'].rolling(window=30).mean()
# eth_a_vault['collateral_usd_30d_ma'] = eth_a_vault['collateral_usd'].rolling(window=30).mean()
# eth_a_vault['debt_balance_30d_ma'] = eth_a_vault['debt_balance'].rolling(window=30).mean()
# eth_a_vault['cumulative_collateral_30d_ma'] = eth_a_vault['cumulative_collateral'].rolling(window=30).mean()
# eth_a_vault['safety_collateral_ratio_30d_ma'] = eth_a_vault['safety_collateral_ratio'].rolling(window=30).mean()
# eth_a_vault['market_collateral_ratio_30d_ma'] = eth_a_vault['market_collateral_ratio'].rolling(window=30).mean()
# eth_a_vault['daily_revenues_30d_ma'] = eth_a_vault['daily_revenues'].rolling(window=30).mean()
# 
# #calculate 90 day ma stability fee
# eth_a_vault['annualized stability fee_90d_ma'] = eth_a_vault['annualized stability fee'].rolling(window=90).mean()
# 
# # Display the head of the DataFrame to verify the new columns
# #print(eth_a_vault[['market_price', 'market_price_7d_ma', 'market_price_30d_ma', 
#                    'collateral_usd', 'collateral_usd_7d_ma', 'collateral_usd_30d_ma',
#                    'debt_balance', 'debt_balance_7d_ma', 'debt_balance_30d_ma']].tail())

# ## Individual vault engingeering

# In[328]:


eth_a_vault['status']


# In[329]:


def enhance_features(df, columns):
    """
    Enhances a DataFrame by calculating percent changes, rolling volatility, and lags for specified columns.

    Parameters:
    - df: pandas.DataFrame, the DataFrame containing the data.
    - columns: list, column names to apply the transformations.

    Returns:
    - df: pandas.DataFrame, the DataFrame with new columns for percent changes, volatility, and lags.
    """
    # Calculate percent change for selected columns
    for column in columns:
        pct_change_col = f'{column}_pct_change'
        df[pct_change_col] = df[column].pct_change()

    # Calculate volatility (standard deviation) of the percent changes over a 7-day rolling window
    for column in columns:
        pct_change_col = f'{column}_pct_change'
        volatility_col = f'{pct_change_col}_volatility_7d'
        df[volatility_col] = df[pct_change_col].rolling(window=7).std()

    # Calculate lag for selected columns (30-day lag as an example)
    for column in columns:
        lag_col = f'{column}_lag30'
        df[lag_col] = df[column].shift(30)

    return df


# In[330]:


columns_to_enhance = ['debt_balance', 'cumulative_collateral', 
               'safety_price', 'safety_collateral_ratio', 
               'market_collateral_ratio','annualized stability fee','daily_revenues']

eth_a_vault = enhance_features(eth_a_vault, columns_to_enhance)
eth_b_vault = enhance_features(eth_b_vault, columns_to_enhance)
wbtc_a_vault = enhance_features(wbtc_a_vault, columns_to_enhance)
wsteth_a_vault = enhance_features(wsteth_a_vault, columns_to_enhance)
eth_c_vault = enhance_features(eth_c_vault, columns_to_enhance)


# In[331]:


aggregated_vaults = enhance_features(aggregated_vaults, aggregated_vaults.drop(columns=['day']).columns)


# In[332]:


eth_a_vault.columns


# In[333]:


aggregated_vaults.columns


# In[334]:


eth_b_vault.columns


# In[335]:


# Calculate percent change for selected columns
for column in ['debt_balance', 'cumulative_collateral', 
               'safety_price', 'safety_collateral_ratio', 
               'market_collateral_ratio','annualized stability fee','daily_revenues']:
    eth_a_vault[f'{column}_pct_change'] = eth_a_vault[column].pct_change()

# Calculate volatility (standard deviation) of the percent changes over a 7-day rolling window
for column in ['debt_balance_pct_change', 'cumulative_collateral_pct_change', 
               'safety_price_pct_change', 'safety_collateral_ratio_pct_change', 
               'market_collateral_ratio_pct_change','annualized stability fee_pct_change','daily_revenues_pct_change']:
    eth_a_vault[f'{column}_volatility_7d'] = eth_a_vault[column].rolling(window=7).std()

# Calculate lag for selected columns (1-day lag as an example)
for column in ['debt_balance', 'cumulative_collateral', 
               'safety_price', 'safety_collateral_ratio', 
               'market_collateral_ratio','annualized stability fee','daily_revenues']:
    eth_a_vault[f'{column}_lag30'] = eth_a_vault[column].shift(30)


# In[336]:


eth_a_vault['status']


# In[337]:


eth_a_vault.columns = [f'eth_a_vault_{col}' if col != 'period' and not col.startswith('day') else col for col in eth_a_vault.columns]
eth_b_vault.columns = [f'eth_b_vault_{col}' if col != 'period' and not col.startswith('day') else col for col in eth_b_vault.columns]
wbtc_a_vault.columns = [f'wbtc_a_vault_{col}' if col != 'period' and not col.startswith('day') else col for col in wbtc_a_vault.columns]
wsteth_a_vault.columns = [f'wsteth_a_vault_{col}' if col != 'period' and not col.startswith('day') else col for col in wsteth_a_vault.columns]
eth_c_vault.columns = [f'eth_c_vault_{col}' if col != 'period' and not col.startswith('day') else col for col in eth_c_vault.columns]


# In[338]:


eth_a_vault['eth_a_vault_status']


# In[339]:


eth_a_vault.set_index('day', inplace=True)
eth_b_vault.set_index('day', inplace=True)
wbtc_a_vault.set_index('day', inplace=True)
wsteth_a_vault.set_index('day', inplace=True)
eth_c_vault.set_index('day', inplace=True)
eth_a_vault = eth_a_vault.drop(columns=['eth_a_vault_ilk'])
eth_b_vault = eth_b_vault.drop(columns=['eth_b_vault_ilk'])
wbtc_a_vault = wbtc_a_vault.drop(columns=['wbtc_a_vault_ilk'])
wsteth_a_vault = wsteth_a_vault.drop(columns=['wsteth_a_vault_ilk'])
eth_c_vault = eth_c_vault.drop(columns=['eth_c_vault_ilk'])


# In[340]:


aggregated_vaults = localize_or_convert(aggregated_vaults, 'day')
aggregated_vaults.set_index('day', inplace=True)


# In[341]:


eth_a_vault.columns


# In[342]:


eth_a_vault = eth_a_vault.merge(eth_b_vault, left_index=True, right_index=True, how='left')
eth_a_vault = eth_a_vault.merge(wbtc_a_vault, left_index=True, right_index=True, how='left')
eth_a_vault = eth_a_vault.merge(wsteth_a_vault, left_index=True, right_index=True, how='left')
eth_a_vault = eth_a_vault.merge(eth_c_vault, left_index=True, right_index=True, how='left')


# In[343]:


ethb_col = eth_b_vault.columns
wbtca_col = wbtc_a_vault.columns
wstetha_col = wsteth_a_vault.columns
ethc_col = eth_c_vault.columns


# In[344]:


eth_a_vault[ethb_col] = eth_a_vault[ethb_col].fillna(0)
eth_a_vault[wbtca_col] = eth_a_vault[wbtca_col].fillna(0)
eth_a_vault[wstetha_col] = eth_a_vault[wstetha_col].fillna(0)
eth_a_vault[ethc_col] = eth_a_vault[ethc_col].fillna(0)


# In[345]:


list(eth_a_vault.columns)


# In[346]:


#print(list(dai_maturity_df_reindexed['dai_maturity_outflow_surplus_buffer_1-block']))


# In[347]:


##print(list(dai_maturity_df_reindexed['dai_maturity_outflow_surplus_buffer_1-day']))


# In[348]:


dai_maturity_df_reindexed


# dai_maturity_outflow_surplus_buffer_1-block        nan
# dai_maturity_outflow_surplus_buffer_1-day          nan
# dai_maturity_outflow_surplus_buffer_1-month        nan
# dai_maturity_outflow_surplus_buffer_1-week         nan
# dai_maturity_outflow_surplus_buffer_3-months       nan
# 1 - PnL                                            nan
# 2 - Assets                                         nan
# 2.8 - Operating Reserves                           nan
# 3 - Liabilities & Equity                           nan
# 3.8 - Equity (Operating Reserves)                  nan

# In[349]:


eth_a_vault = eth_a_vault.merge(dai_maturity_df_reindexed, on=['day'], how='inner')


# In[350]:


aggregated_vaults = aggregated_vaults.merge(dai_maturity_df_reindexed, on=['day'], how='inner')


# In[351]:


stable_coin_ratios.head()


# In[352]:


#eth_a_vault = eth_a_vault.merge(stable_coin_ratios, on=['day'], how='inner')


# In[353]:


eth_a_vault.head()


# In[354]:


start_date = eth_a_vault['day'].min()
end_date = eth_a_vault['day'].max()
date_range = pd.date_range(start=start_date, end=end_date)

psm_full_range_df = pd.DataFrame(index=date_range)



# In[355]:


psm_full_range_df.head()


# In[356]:


psm_columns = ['psm_change', 'psm_change_excl_rwa', 'psm_change_excl_rwa_30d_avg', 'psm_change_excl_rwa_7d_avg', 'psm_fees', 'psm_inflow', 'psm_inflow_exl_rwa', 'psm_lifetime_fees', 'psm_lifetime_turnover', 'psm_outflow', 'psm_balance', 'psm_turnover']
for column in psm_columns:
    psm_full_range_df[column] = psm_stats_csv[column]

# Fill missing values with zeros
psm_full_range_df.fillna(0, inplace=True)


# In[357]:


psm_stats_csv


# In[358]:


eth_a_vault.describe()


# In[359]:


eth_a_vault.columns


# In[360]:


psm_full_range_df = psm_full_range_df.rename_axis('day')


# In[361]:


psm_full_range_df.duplicated().sum()


# In[362]:


aggregated_vaults = aggregated_vaults.merge(psm_stats_csv, on=['day'], how='left')


# In[363]:


aggregated_vaults.fillna(0, inplace=True)


# In[364]:


aggregated_vaults


# In[365]:


eth_a_vault= eth_a_vault.merge(psm_full_range_df, on=['day'], how='inner')


# In[366]:


eth_a_vault[['day','psm_change']].tail()


# In[367]:


where_is_dai_csv_table_full = pd.DataFrame(index=date_range)


where_is_dai_csv_table_columns = where_is_dai_csv_table.columns
for column in where_is_dai_csv_table_columns:
    where_is_dai_csv_table_full[column] = where_is_dai_csv_table[column]

# Fill missing values with zeros
where_is_dai_csv_table_full.fillna(0, inplace=True)


# In[368]:


where_is_dai_csv_table_full = where_is_dai_csv_table_full.rename_axis('day')


# In[369]:


where_is_dai_csv_table_full.head()


# In[370]:


eth_a_vault= eth_a_vault.merge(where_is_dai_csv_table_full, on=['day'], how='inner')


# In[371]:


aggregated_vaults = aggregated_vaults.merge(where_is_dai_csv_table_full, on=['day'], how='left')
aggregated_vaults.fillna(0, inplace=True)


# In[372]:


aggregated_vaults


# In[373]:


eth_a_vault.head()


# In[374]:


daily_surplus_buffer_csv = daily_surplus_buffer_csv.fillna(0)


# In[375]:


eth_a_vault= eth_a_vault.merge(daily_surplus_buffer_csv, on=['day'], how='inner')


# In[376]:


aggregated_vaults = aggregated_vaults.merge(daily_surplus_buffer_csv, on=['day'], how='left')


# In[377]:


list(aggregated_vaults.columns)


# In[378]:


eth_a_vault['eth_a_vault_status']


# In[379]:


dsr_df.fillna(0, inplace=True)


# In[380]:


dsr_df.head()


# In[381]:


dsr_df.reset_index(inplace=True)
eth_a_vault.reset_index(inplace=True)
# Remove timezone information from both 'day' columns
eth_a_vault['day'] = eth_a_vault['day'].dt.tz_localize(None)
dsr_df['day'] = dsr_df['day'].dt.tz_localize(None)


# In[382]:


dsr_df_full = pd.DataFrame(index=date_range)
dsr_df_full = dsr_df_full.rename_axis('day')

dsr_df_full.reset_index(inplace=True)
dsr_df_full['day'] = dsr_df_full['day'].dt.tz_localize(None)


# In[383]:


dsr_df_full = dsr_df_full.merge(dsr_df, on=['day'], how='outer')


# In[384]:


dsr_df_full.fillna(0, inplace=True)


# In[385]:


dsr_df_full


# In[386]:


dsr_df_full = dsr_df_full.set_index('day')


# In[387]:


#dsr_df_full = dsr_df_full.drop(columns=['level_0','index'])


# In[388]:


dsr_df_full.head()


# In[389]:


eth_a_vault= eth_a_vault.merge(dsr_df_full, on=['day'], how='inner')


# In[390]:


aggregated_vaults


# In[391]:


dsr_df_full.reset_index(inplace=True)
dsr_df_full = localize_or_convert(dsr_df_full, 'day')


# In[392]:


aggregated_vaults = aggregated_vaults.merge(dsr_df_full, on=['day'], how='left')


# In[393]:


list(aggregated_vaults.columns)


# In[394]:


eth_a_vault.columns


# In[395]:


eth_a_vault = eth_a_vault.drop(columns=['index'])


# In[396]:


eth_a_vault.head()


# In[397]:


pivoted_balance_sheet = pivoted_balance_sheet.rename_axis('day')


# In[398]:


pivoted_income_statement.rename(columns={'period':'day'}, inplace=True)


# In[399]:


pivoted_balance_sheet.reset_index(inplace=True)
pivoted_balance_sheet['day'] = pivoted_balance_sheet['day'].dt.tz_localize(None)


# In[400]:


pivoted_balance_sheet = pivoted_balance_sheet.set_index('day')


# In[401]:


pivoted_balance_sheet = pivoted_balance_sheet.fillna(0)


# In[402]:


##print(pivoted_balance_sheet.describe())


# In[403]:


eth_a_vault= eth_a_vault.merge(pivoted_balance_sheet, on=['day'], how='left') 


# In[404]:


pivoted_balance_sheet.reset_index(inplace=True)
pivoted_balance_sheet = localize_or_convert(pivoted_balance_sheet, 'day')


# In[405]:


aggregated_vaults = aggregated_vaults.merge(pivoted_balance_sheet, on=['day'], how='left') 


# In[406]:


aggregated_vaults.fillna(0, inplace=True)


# In[407]:


aggregated_vaults


# In[408]:


eth_a_vault = eth_a_vault.fillna(0)


# In[409]:


eth_a_vault.shape[0]


# In[410]:


##print(list(eth_a_vault.columns))


# In[411]:


eth_a_vault = eth_a_vault.sort_values(by='day')
pivoted_income_statement = pivoted_income_statement.sort_values(by='day')


# In[412]:


pivoted_income_statement.tail()


# In[413]:


total_vault_data = pd.merge_asof(eth_a_vault, pivoted_income_statement, on='day')


# In[414]:


aggregated_vaults


# In[415]:


pivoted_income_statement


# In[416]:


pivoted_income_statement = localize_or_convert(pivoted_income_statement, 'day')


# In[417]:


aggregated_vault_data = pd.merge_asof(aggregated_vaults, pivoted_income_statement, on='day')


# In[418]:


#print(list(total_vault_data.columns))


# In[419]:


aggregated_vault_data.fillna(0, inplace=True)


# ### Now for CoinGecko Crypto Market Data

# In[420]:


#lets get price feeds for accepted collateral types

ir_csv['ilk'].unique()


# In[421]:


#need to use yfinance instead, coingecko clocked to 1 year historical


# In[422]:


#sp = yf.Ticker("^GSPC")


# In[423]:


#sp_from_nov_raw = sp.history(period="54mo")


# In[424]:


sp_path = 'data/csv/sp500.csv'
#sp_from_nov_raw.to_csv(sp_path)
sp_from_nov = pd.read_csv(sp_path)


# In[425]:


sp_from_nov.head()


# In[426]:


sp_from_nov = sp_from_nov.drop(columns=['Dividends','Stock Splits','Open','Low','High'])


# In[427]:


sp_from_nov.columns = [f's&p_500_market_{col}' if col != 'Date' else col for col in sp_from_nov.columns]


# In[428]:


sp_from_nov


# In[429]:


#btc = yf.Ticker('BTC-USD')


# In[430]:


#btc_from_nov = btc.history(period='54mo')
btc_path = 'data/csv/btc.csv'
#btc_from_nov.to_csv(btc_path)
btc_from_nov = pd.read_csv(btc_path)


# In[431]:


btc_from_nov.head()


# In[432]:


#btc_from_nov = btc_from_nov.drop(columns=['Dividends','Stock Splits','Open','Low','High'])


# In[433]:


#btc_from_nov.columns = [f'btc_market_{col}' if col != 'Date' else col for col in btc_from_nov.columns]


# In[434]:


btc_from_nov.head()


# eth = yf.Ticker('ETH-USD')
# eth_from_nov = eth.history(period='54mo')

# In[435]:


eth_from_nov_path = 'data/csv/eth.csv'
#eth_from_nov.to_csv(eth_from_nov_path)
eth_from_nov = pd.read_csv(eth_from_nov_path)


# In[436]:


eth_from_nov.head()


# In[437]:


eth_from_nov = eth_from_nov.drop(columns=['Dividends','Stock Splits','Open','Low','High'])


# In[438]:


eth_from_nov.columns = [f'eth_market_{col}' if col != 'Date' else col for col in eth_from_nov.columns]


# In[439]:


eth_from_nov.head()


# mkr = yf.Ticker('MKR-USD')
# mkr_from_nov = mkr.history(period='54mo')

# In[440]:


mkr_from_nov_path = 'data/csv/mkr.csv'
#mkr_from_nov.to_csv(mkr_from_nov_path)
mkr_from_nov = pd.read_csv(mkr_from_nov_path)
mkr_from_nov.head()


# In[441]:


mkr_from_nov = mkr_from_nov.drop(columns=['Unnamed: 0'])


# In[442]:


#mkr_from_nov.columns = [f'mkr_market_{col}' if col != 'Date' else col for col in mkr_from_nov.columns]


# In[443]:


mkr_from_nov.head()


# vix = yf.Ticker('^VIX')
# vix_from_nov = vix.history(period='54mo')

# In[444]:


vix_from_nov_path = 'data/csv/vix.csv'
#vix_from_nov.to_csv(vix_from_nov_path)
vix_from_nov = pd.read_csv(vix_from_nov_path)
vix_from_nov = vix_from_nov.drop(columns=['Dividends','Stock Splits','Open','Low','High'])
vix_from_nov.head()


# In[445]:


vix_from_nov.columns = [f'vix_market_{col}' if col != 'Date' else col for col in vix_from_nov.columns]


# dai = yf.Ticker('DAI-USD')
# dai_from_nov = dai.history(period='54mo')

# In[446]:


dai_from_nov_path = 'data/csv/dai.csv'
#dai_from_nov.to_csv(dai_from_nov_path)
dai_from_nov = pd.read_csv(dai_from_nov_path)
dai_from_nov.head()


# In[447]:


dai_from_nov = dai_from_nov.drop(columns=['Dividends','Stock Splits','Open','Low','High'])


# In[448]:


dai_from_nov.columns = [f'dai_market_{col}' if col != 'Date' else col for col in dai_from_nov.columns]


# In[449]:


dai_from_nov.head()


# In[450]:


dai_from_nov['dai_deviation'] = dai_from_nov['dai_market_Close'] - 1
dai_from_nov['dai_abs_deviation'] = dai_from_nov['dai_deviation'].abs()

average_deviation = dai_from_nov['dai_abs_deviation'].mean()
standard_deviation = dai_from_nov['dai_market_Close'].std()

#print(f"DAI Average Deviation from $1: {average_deviation}")
#print(f"Standard Deviation of DAI Price: {standard_deviation}")


# ### Market Cap Stats

# In[451]:


mcap_path = 'data/csv/cryptomarketcap.csv'
mcap = pd.read_csv(mcap_path)
mcap['Date'] = pd.to_datetime(mcap['snapped_at'], unit='ms')
mcap['Date'] = mcap['Date'].dt.tz_localize('UTC')
mcap.drop(columns=['snapped_at'], inplace=True)
#mcap.set_index('day', inplace=True)
mcap['Date'] = mcap['Date'].dt.normalize()  # This will set the time to 00:00:00
# Format the 'Date' to match 'dai_from_nov' which includes the colon in the timezone
mcap['Date'] = mcap['Date'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')


# In[452]:


mcap.columns = [f'mcap_{col}' if col != 'Date' else col for col in mcap.columns]
mcap.columns


# ### Defi APY

# In[453]:


apy_path = 'data/csv/medianAPY.csv'
defi_apy = pd.read_csv(apy_path)
defi_apy['timestamp'] = pd.to_datetime(defi_apy['timestamp'])
defi_apy['timestamp'] = defi_apy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
defi_apy.rename(columns={'timestamp': 'Date'}, inplace=True)


# In[454]:


defi_apy.columns = [f'defi_apy_{col}' if col != 'Date' else col for col in defi_apy.columns]
defi_apy


# ### Stablecoin Mcap

# In[455]:


sbl_path = 'data/csv/stablecoins.csv'
stablecoin_tvl = pd.read_csv(sbl_path)
stablecoin_tvl['Date'] = pd.to_datetime(stablecoin_tvl['Date'])
stablecoin_tvl['Date'] = stablecoin_tvl['Date'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
stablecoin_tvl = stablecoin_tvl[['Date', 'Total']]
stablecoin_tvl


# In[456]:


stablecoin_tvl.columns = [f'stablecoin_tvl_{col}' if col != 'Date' else col for col in stablecoin_tvl.columns]
stablecoin_tvl.columns


# ### DPI Index

# In[457]:


dpi = yf.Ticker('DPI-USD') 
dpi_from_nov = dpi.history(period='5y')
dpi_from_nov = dpi_from_nov[[ 'Close','Volume']]
dpi_from_nov


# In[458]:


dpi_from_nov.columns = [f'dpi_market_{col}' if col != 'Date' else col for col in dpi_from_nov.columns]
dpi_from_nov.reset_index(inplace=True)
dpi_from_nov


# In[459]:


mcap['Date'] 


# In[460]:


dai_from_nov['Date']


# In[461]:


crypto_market_data = pd.merge(dai_from_nov, eth_from_nov, on=['Date'], how='inner')


# In[462]:


crypto_market_data = crypto_market_data.merge(mcap, on=['Date'], how='left')


# In[463]:


crypto_market_data = crypto_market_data.merge(stablecoin_tvl, on=['Date'], how='left')


# In[464]:


crypto_market_data[ 'stablecoin_tvl_Total'
].fillna(0, inplace=True)


# In[465]:


crypto_market_data = crypto_market_data.merge(defi_apy, on=['Date'], how='left')


# In[466]:


crypto_market_data[['defi_apy_medianAPY','defi_apy_avg7day']] = crypto_market_data[['defi_apy_medianAPY','defi_apy_avg7day']].fillna(0)


# In[467]:


dpi_from_nov['Date'] = dpi_from_nov['Date'].astype(str)


# In[468]:


crypto_market_data = crypto_market_data.merge(dpi_from_nov, on=['Date'], how='left')


# In[469]:


crypto_market_data[['dpi_market_Close',	'dpi_market_Volume']] = crypto_market_data[['dpi_market_Close',	'dpi_market_Volume']].fillna(0)


# In[470]:


crypto_market_data


# In[471]:


crypto_market_data = crypto_market_data.merge(btc_from_nov, on=['Date'], how='inner')


# In[472]:


crypto_market_data = crypto_market_data.merge(mkr_from_nov, on=['Date'], how='inner')


# In[473]:


crypto_market_data


# In[474]:


sp_from_nov['Date'] = pd.to_datetime(sp_from_nov['Date'],utc=True)
vix_from_nov['Date']= pd.to_datetime(vix_from_nov['Date'],utc=True)


# In[475]:


sp_from_nov = localize_or_convert(sp_from_nov, 'Date')
vix_from_nov= localize_or_convert(vix_from_nov, 'Date')


# In[476]:


sp_from_nov.describe()


# In[477]:


crypto_market_data.set_index('Date', inplace=True)


# In[478]:


crypto_market_data.index = pd.to_datetime(crypto_market_data.index)


# In[479]:


sp_from_nov.set_index('Date', inplace=True)
vix_from_nov.set_index('Date', inplace=True)


# In[480]:


#sp_from_nov.index = pd.to_datetime(sp_from_nov.index)


# In[481]:


sp_from_nov_normalized = sp_from_nov.index.normalize()
crypto_market_data_normalized = crypto_market_data.index.normalize()
vix_from_nov_normalized = vix_from_nov.index.normalize()


# In[482]:


sp_from_nov.index = sp_from_nov_normalized
crypto_market_data.index = crypto_market_data_normalized
vix_from_nov.index = vix_from_nov_normalized


# In[483]:


sp_from_nov.head()


# In[484]:


vix_from_nov.head()


# In[485]:


crypto_market_data.head()


# In[486]:


crypto_market_data = crypto_market_data.merge(sp_from_nov, on=['Date'], how='left')


# In[487]:


crypto_market_data[['s&p_500_market_Close','s&p_500_market_Volume']] = crypto_market_data[['s&p_500_market_Close','s&p_500_market_Volume']].ffill()


# In[488]:


crypto_market_data.isna().sum()


# In[489]:


crypto_market_data = crypto_market_data.merge(vix_from_nov, on=['Date'], how='left')


# In[490]:


crypto_market_data['vix_market_Close'] = crypto_market_data['vix_market_Close'].ffill()
crypto_market_data.drop(columns=['vix_market_Volume'], inplace=True)


# In[491]:


crypto_market_data


# In[492]:


for col in crypto_market_data:
    crypto_market_data[f'{col}_daily_returns'] = crypto_market_data[col].pct_change()
    crypto_market_data[f'{col}_7d_ma']= crypto_market_data[col].rolling(window=7).mean()
    crypto_market_data[f'{col}_30d_ma'] = crypto_market_data[col].rolling(window=30).mean()



# In[493]:


crypto_market_data.rename_axis('day',inplace=True)


# In[494]:


crypto_market_data = crypto_market_data.fillna(0)


# In[495]:


crypto_market_data.head(20)


# In[496]:


nan_rows = crypto_market_data[crypto_market_data.isna().any(axis=1)]
#print(nan_rows)


# In[497]:


nan_rows.columns


# for column in crypto_market_data[f'{col}_daily_returns']:
#     crypto_market_data[f'{column}_volatility_7d'] = eth_a_vault[column].rolling(window=7).std()

# for column in [ 'cumulative_collateral', 
#                'safety_price', 'safety_collateral_ratio', 
#                'market_collateral_ratio','annualized stability fee','daily_revenues']:
#     eth_a_vault[f'{column}_lag30'] = eth_a_vault[column].shift(30)

# ## Now for Macro Economic Data from FRED Api

# def fetch_and_process_tbill_data(api_url, data_key, date_column, value_column, date_format='datetime'):
#     # Retrieve the API key from Streamlit secrets
#     api_key = st.secrets["FRED_API_KEY"]
# 
#     # Append the API key to the URL
#     api_url_with_key = f"{api_url}&api_key={api_key}"
# 
#     response = requests.get(api_url_with_key)
#     if response.status_code == 200:
#         data = response.json()
#         df = pd.DataFrame(data[data_key])
#         
#         if date_format == 'datetime':
#             df[date_column] = pd.to_datetime(df[date_column])
#         
#         df.set_index(date_column, inplace=True)
#         df[value_column] = df[value_column].astype(float)
#         return df
#     else:
#         #print(f"Failed to retrieve data: {response.status_code}")
#         return pd.DataFrame()  # Return an empty DataFrame in case of failure

# three_month_tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&file_type=json"
# three_month_tbill = fetch_and_process_tbill_data(three_month_tbill_historical_api, "observations", "date", "value")
# 

# three_month_tbill = three_month_tbill[three_month_tbill.index >= '2019-11-01']

# three_month_tbill['3m_tbill'] = three_month_tbill['value'] / 100

# three_month_tbill

# In[498]:


tmo_path = 'data/csv/3mo_tbill.csv'
three_month_tbill = pd.read_csv(tmo_path)
three_month_tbill_csv = pd.read_csv(tmo_path, index_col='date', parse_dates=True)


# In[499]:


three_month_tbill_csv.describe()


# https://fred.stlouisfed.org/series/FEDTARMDLR

# In[500]:


forecast_ffr_path = "data/csv/FEDTARMDLR.csv"
forecast_ffr = pd.read_csv(forecast_ffr_path)
forecast_ffr.head()


# In[501]:


forecast_ffr=forecast_ffr.rename(columns={'DATE':'date'})
forecast_ffr=forecast_ffr.rename(columns={'FEDTARMDLR':'forecast_fed_funds'})
forecast_ffr['date'] = pd.to_datetime(forecast_ffr['date'])
forecast_ffr['date'] = forecast_ffr['date'].dt.tz_localize('UTC')


# In[502]:


forecast_ffr.set_index('date',inplace=True)


# In[503]:


forecast_ffr = forecast_ffr.resample('M').last().ffill()
forecast_ffr.reset_index(inplace=True)
forecast_ffr.head()


# In[504]:


#sticky_price_consumer_index

spi_path = "data/csv/sticky_price_consumer_price_index.csv"
sticky_index = pd.read_csv(spi_path)


# In[505]:


#print(sticky_index.describe())


# https://fred.stlouisfed.org/series/PCECTPIMDLR

# In[506]:


pce_path = "data/csv/PCECTPIMDLR.csv"
forecast_pce_inflation = pd.read_csv(pce_path)
forecast_pce_inflation.head()


# In[507]:


forecast_pce_inflation=forecast_pce_inflation.rename(columns={'DATE':'date'})
forecast_pce_inflation=forecast_pce_inflation.rename(columns={'PCECTPIMDLR':'forecast_pce'})
forecast_pce_inflation['date'] = pd.to_datetime(forecast_pce_inflation['date'])
forecast_pce_inflation['date'] = forecast_pce_inflation['date'].dt.tz_localize('UTC')


# In[508]:


forecast_pce_inflation.set_index('date',inplace=True)


# In[509]:


forecast_pce_inflation = forecast_pce_inflation.resample('M').last().ffill()
forecast_pce_inflation.reset_index(inplace=True)
forecast_pce_inflation.head()


# https://fred.stlouisfed.org/series/M1SL

# In[510]:


m1_path =  "data/csv/M1SL.csv"
m1 = pd.read_csv(m1_path)


# In[511]:


m1.tail()


# In[512]:


m1=m1.rename(columns={'DATE':'date'})
m1['date'] = pd.to_datetime(m1['date'])
m1['date'] = m1['date'].dt.tz_localize('UTC')


# https://fred.stlouisfed.org/series/M1V

# In[513]:


m1_v_path =  "data/csv/M1V.csv"
m1_v = pd.read_csv(m1_v_path)


# In[514]:


m1_v.tail()


# In[515]:


m1_v=m1_v.rename(columns={'DATE':'date'})
m1_v['date'] = pd.to_datetime(m1_v['date'])
m1_v['date'] = m1_v['date'].dt.tz_localize('UTC')


# In[516]:


m1_v.set_index('date',inplace=True)
m1_v = m1_v.resample('M').last().ffill()
m1_v.reset_index(inplace=True)
m1_v.tail()


# https://fred.stlouisfed.org/series/WM2NS

# In[517]:


m2_path =  "data/csv/WM2NS.csv"
m2 = pd.read_csv(m2_path)


# In[518]:


m2.tail()


# In[519]:


m2=m2.rename(columns={'DATE':'date'})
m2['date'] = pd.to_datetime(m2['date'])
m2['date'] = m2['date'].dt.tz_localize('UTC')


# In[520]:


m2.set_index('date',inplace=True)
m2 = m2.resample('M').mean()
m2.reset_index(inplace=True)
m2.tail()


# https://fred.stlouisfed.org/series/M2V

# In[521]:


m2_v_path =  "data/csv/M2V.csv"
m2_v = pd.read_csv(m2_v_path)


# In[522]:


m2_v.tail()


# In[523]:


m2_v=m2_v.rename(columns={'DATE':'date'})
m2_v['date'] = pd.to_datetime(m2_v['date'])
m2_v['date'] = m2_v['date'].dt.tz_localize('UTC')


# In[524]:


m2_v.set_index('date',inplace=True)
m2_v = m2_v.resample('M').last().ffill()
m2_v.reset_index(inplace=True)
m2_v.tail()

#https://fred.stlouisfed.org/series/GDPC1
# In[525]:


rgdp_path = "data/csv/GDPC1.csv"
rgdp = pd.read_csv(rgdp_path)
rgdp.tail()


# In[526]:


rgdp=rgdp.rename(columns={'DATE':'date'})
rgdp=rgdp.rename(columns={'GDPC1':'real_gdp'})
rgdp['date'] = pd.to_datetime(rgdp['date'])
rgdp['date'] = rgdp['date'].dt.tz_localize('UTC')


# In[527]:


rgdp.set_index('date',inplace=True)
rgdp = rgdp.resample('M').last().ffill()
rgdp.reset_index(inplace=True)
rgdp.tail()


# https://fred.stlouisfed.org/series/GDPC1MDLR

# In[528]:


forecast_median_real_gdp_path = "data/csv/GDPC1MDLR.csv"
forecast_median_real_gdp = pd.read_csv(forecast_median_real_gdp_path)
forecast_median_real_gdp.head()


# In[529]:


forecast_median_real_gdp=forecast_median_real_gdp.rename(columns={'DATE':'date'})
forecast_median_real_gdp=forecast_median_real_gdp.rename(columns={'GDPC1MDLR':'forecast_real_gdp'})
forecast_median_real_gdp['date'] = pd.to_datetime(forecast_median_real_gdp['date'])
forecast_median_real_gdp['date'] = forecast_median_real_gdp['date'].dt.tz_localize('UTC')


# In[530]:


forecast_median_real_gdp.set_index('date',inplace=True)
forecast_median_real_gdp = forecast_median_real_gdp.resample('M').last().ffill()
forecast_median_real_gdp.reset_index(inplace=True)
forecast_median_real_gdp.tail()


# https://fred.stlouisfed.org/series/GDP

# In[531]:


gdp_path = "data/csv/GDP.csv"
gdp = pd.read_csv(gdp_path)
gdp.tail()


# In[532]:


gdp=gdp.rename(columns={'DATE':'date'})
gdp['date'] = pd.to_datetime(gdp['date'])
gdp['date'] = gdp['date'].dt.tz_localize('UTC')


# In[533]:


gdp.set_index('date',inplace=True)
gdp = gdp.resample('M').last().ffill()
gdp.reset_index(inplace=True)
gdp.tail()


# https://fred.stlouisfed.org/series/QBPQYTNIYVENTKREV

# In[534]:


#in millions

vcr_path = "data/csv/QBPQYTNIYVENTKREV.csv"
vc_revenue = pd.read_csv(vcr_path)
vc_revenue.tail()


# In[535]:


vc_revenue=vc_revenue.rename(columns={'DATE':'date'})
vc_revenue=vc_revenue.rename(columns={'QBPQYTNIYVENTKREV':'vc_revenue'})
vc_revenue['date'] = pd.to_datetime(vc_revenue['date'])
vc_revenue['date'] = vc_revenue['date'].dt.tz_localize('UTC')


# In[536]:


vc_revenue.set_index('date',inplace=True)
vc_revenue = vc_revenue.resample('M').last().ffill()
vc_revenue.reset_index(inplace=True)
vc_revenue.tail()


# https://fred.stlouisfed.org/series/RRPTTLD

# In[537]:


#in billions

reverse_repo_path = "data/csv/RRPTTLD.csv"
reverse_repo = pd.read_csv(reverse_repo_path)
reverse_repo.tail(20)


# In[538]:


reverse_repo['RRPTTLD'] = reverse_repo['RRPTTLD'].replace('.', np.nan).astype(float)


# In[539]:


reverse_repo=reverse_repo.rename(columns={'DATE':'date'})
reverse_repo=reverse_repo.rename(columns={'RRPTTLD':'fed_reverse_repo'})
reverse_repo['date'] = pd.to_datetime(reverse_repo['date'])
reverse_repo['date'] = reverse_repo['date'].dt.tz_localize('UTC')


# In[540]:


reverse_repo.set_index('date',inplace=True)
reverse_repo = reverse_repo.resample('M').mean()
reverse_repo.reset_index(inplace=True)
reverse_repo.tail(20)


# https://fred.stlouisfed.org/series/RPTTLD

# In[541]:


#multiply by 1 billion for real figure

repo_path = "data/csv/RPTTLD.csv"
repo = pd.read_csv(repo_path)
repo.head()


# In[542]:


repo['RPTTLD'] = repo['RPTTLD'].replace('.', np.nan).astype(float)


# In[543]:


repo=repo.rename(columns={'DATE':'date'})
repo=repo.rename(columns={'RPTTLD':'fed_repo'})
repo['date'] = pd.to_datetime(repo['date'])
repo['date'] = repo['date'].dt.tz_localize('UTC')


# In[544]:


repo.head(20)


# In[545]:


repo.set_index('date',inplace=True)
repo = repo.resample('M').mean()
repo.reset_index(inplace=True)
repo.head(20)


# https://fred.stlouisfed.org/series/QBPBSTAS

# In[546]:


#in millions, multiply by 1 milion for real figure

fdic_assets_path = "data/csv/QBPBSTAS.csv"
fdic_assets = pd.read_csv(fdic_assets_path)
fdic_assets.tail()


# In[547]:


fdic_assets=fdic_assets.rename(columns={'DATE':'date'})
fdic_assets=fdic_assets.rename(columns={'QBPBSTAS':'FDIC_Assets'})
fdic_assets['date'] = pd.to_datetime(fdic_assets['date'])
fdic_assets['date'] = fdic_assets['date'].dt.tz_localize('UTC')


# In[548]:


fdic_assets.set_index('date',inplace=True)
fdic_assets = fdic_assets.resample('M').last().ffill()
fdic_assets.reset_index(inplace=True)
fdic_assets.head(20)


# https://fred.stlouisfed.org/series/QBPBSTLKTL

# In[549]:


#in millions, multiply by 1 milion for real figure

fdic_liabilities_path = "data/csv/QBPBSTLKTL.csv"
fdic_liabilities = pd.read_csv(fdic_liabilities_path)
fdic_liabilities.tail()


# In[550]:


fdic_liabilities=fdic_liabilities.rename(columns={'DATE':'date'})
fdic_liabilities=fdic_liabilities.rename(columns={'QBPBSTLKTL':'FDIC_Liabilities'})
fdic_liabilities['date'] = pd.to_datetime(fdic_liabilities['date'])
fdic_liabilities['date'] = fdic_liabilities['date'].dt.tz_localize('UTC')


# In[551]:


fdic_liabilities.set_index('date',inplace=True)
fdic_liabilities = fdic_liabilities.resample('M').last().ffill()
fdic_liabilities.reset_index(inplace=True)
fdic_liabilities.head(20)


# https://fred.stlouisfed.org/series/QBPBSTLKTEQK

# In[552]:


#in millions, multiply by 1 milion for real figure

fdic_equity_path = "data/csv/QBPBSTLKTEQK.csv"
fdic_equity = pd.read_csv(fdic_equity_path)
fdic_equity.tail()


# In[553]:


fdic_equity=fdic_equity.rename(columns={'DATE':'date'})
fdic_equity=fdic_equity.rename(columns={'QBPBSTLKTEQK':'FDIC_Equity'})
fdic_equity['date'] = pd.to_datetime(fdic_equity['date'])
fdic_equity['date'] = fdic_equity['date'].dt.tz_localize('UTC')


# In[554]:


fdic_equity.set_index('date',inplace=True)
fdic_equity = fdic_equity.resample('M').last().ffill()
fdic_equity.reset_index(inplace=True)
fdic_equity.head(20)


# In[555]:


eff_rate_path =  "data/csv/effective_federal_funds_rate.csv"
fed_funds_rate = pd.read_csv(eff_rate_path)


# In[556]:


fed_funds_rate.describe()


# In[557]:


fed_funds_rate=fed_funds_rate.rename(columns={'DATE':'date'})
fed_funds_rate['date'] = pd.to_datetime(fed_funds_rate['date'])


# In[558]:


fed_funds_rate['date'] = fed_funds_rate['date'].dt.tz_localize('UTC')


# In[559]:


fed_funds_rate.describe()


# In[560]:


three_month_tbill_csv.index = three_month_tbill_csv.index.tz_localize('UTC')


# In[561]:


three_month_tbill_csv.tail()


# In[562]:


three_month_tbill_csv = three_month_tbill_csv.rename(columns={'value':'3_m_tbill_yield'})


# In[563]:


three_month_tbill_csv.head()


# In[564]:


#print(three_month_tbill_csv.describe())


# In[565]:


fed_funds_rate['EFFR'] = fed_funds_rate['EFFR'].replace('.', np.nan).astype(float)


# In[566]:


fed_funds_rate = fed_funds_rate.rename(columns={'EFFR':'effective_funds_rate'})
fed_funds_rate.set_index('date',inplace=True)
fed_funds_rate = fed_funds_rate.resample('M').mean().ffill()
fed_funds_rate.reset_index(inplace=True)
fed_funds_rate.head(20)


# In[567]:


macro_data = pd.merge_asof(fed_funds_rate.sort_values('date'), three_month_tbill_csv.sort_values('date'), on='date', direction='nearest')


# In[568]:


macro_data.head()


# macro_data['effective_funds_rate'] = macro_data['effective_funds_rate'].replace('.', np.nan)
# 
# macro_data['effective_funds_rate'] = pd.to_numeric(macro_data['effective_funds_rate'], errors='coerce')
# 
# macro_data['effective_funds_rate'].ffill(inplace=True)
# 
# 

# In[569]:


#print(macro_data['effective_funds_rate'].describe())


# In[570]:


sticky_index.rename(columns={'DATE': 'date', 'CORESTICKM159SFRBATL': 'sticky_cpi'}, inplace=True)


# In[571]:


sticky_index.tail()


# In[572]:


sticky_index['date']= pd.to_datetime(sticky_index['date'])


# In[573]:


sticky_index = localize_or_convert(sticky_index, 'date')
sticky_index.set_index('date',inplace=True)
sticky_index = sticky_index.resample('M').last().ffill()
sticky_index.reset_index(inplace=True)
sticky_index.head(20)


# In[574]:


macro_data = macro_data.merge(sticky_index, on=['date'],how='inner')


# In[575]:


#print(macro_data['date'].dtype)
#print(forecast_ffr['date'].dtype)


# In[576]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), forecast_ffr.sort_values('date'), on='date', direction='nearest')


# In[577]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), forecast_pce_inflation.sort_values('date'), on='date', direction='nearest')


# In[578]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), m1.sort_values('date'), on='date', direction='nearest')


# In[579]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), m1_v.sort_values('date'), on='date', direction='nearest')


# In[580]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), m2.sort_values('date'), on='date', direction='nearest')


# In[581]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), m2_v.sort_values('date'), on='date', direction='nearest')


# In[582]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), rgdp.sort_values('date'), on='date', direction='nearest')


# In[583]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), forecast_median_real_gdp.sort_values('date'), on='date', direction='nearest')


# In[584]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), gdp.sort_values('date'), on='date', direction='nearest')


# In[585]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), vc_revenue.sort_values('date'), on='date', direction='nearest')


# In[586]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), reverse_repo.sort_values('date'), on='date', direction='nearest')


# In[587]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), repo.sort_values('date'), on='date', direction='nearest')


# In[588]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), fdic_assets.sort_values('date'), on='date', direction='nearest')


# In[589]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), fdic_liabilities.sort_values('date'), on='date', direction='nearest')


# In[590]:


macro_data = pd.merge_asof(macro_data.sort_values('date'), fdic_equity.sort_values('date'), on='date', direction='nearest')


# In[591]:


macro_data


# In[592]:


macro_data = macro_data.drop(columns=['3m_tbill','realtime_start','realtime_end'])


# In[593]:


macro_data.set_index('date',inplace=True)


# In[594]:


macro_data.rename_axis('day',inplace=True)


# In[ ]:





# In[595]:


crypto_market_data = crypto_market_data.sort_values(by='day')
macro_data = macro_data.sort_values(by='day')



# In[596]:


crypto_market_data.head()


# In[597]:


macro_data.head()


# In[598]:


macro_and_crypto = pd.merge_asof(crypto_market_data, macro_data, on='day')


# In[599]:


columns_to_backfill = ['effective_funds_rate', '3_m_tbill_yield', 'sticky_cpi', 'forecast_fed_funds', 'forecast_pce', 'M1SL', 'M1V', 'WM2NS', 'M2V', 'real_gdp', 'forecast_real_gdp', 'GDP', 'vc_revenue', 'fed_reverse_repo', 'fed_repo', 'FDIC_Assets', 'FDIC_Liabilities', 'FDIC_Equity']
macro_and_crypto[columns_to_backfill] = macro_and_crypto[columns_to_backfill].fillna(method='bfill')


# In[600]:


macro_and_crypto = localize_or_convert(macro_and_crypto, 'day')


# In[601]:


total_vault_data = localize_or_convert(total_vault_data, 'day')


# In[602]:


total_vault_data[['dai_maturity_outflow_surplus_buffer_1-block','dai_maturity_outflow_surplus_buffer_1-day','dai_maturity_outflow_surplus_buffer_1-month','dai_maturity_outflow_surplus_buffer_1-week','dai_maturity_outflow_surplus_buffer_3-months']]








# 1 - PnL                                            nan
# 2 - Assets                                         nan
# 2.8 - Operating Reserves                           nan
# 3 - Liabilities & Equity                           nan
# 3.8 - Equity (Operating Reserves)                  nan

# In[603]:


dataset = total_vault_data.merge(macro_and_crypto, on=['day'],how='inner')


# In[604]:


aggregated_vault_data = aggregated_vault_data.merge(macro_and_crypto, on=['day'],how='left')


# In[605]:


aggregated_vault_data.fillna(0, inplace=True)


# In[606]:


aggregated_vault_data.isna().sum().sum()


# In[607]:


list(aggregated_vault_data.columns)


# In[608]:


aggregated_vault_data.set_index('day', inplace=True)


# In[609]:


dataset['real_gdp'].isna().sum()


# ## Preparing for Correlations

# In[610]:


nan_rows = dataset[dataset.isna().any(axis=1)]
#print(nan_rows)


# In[611]:


# Checking for columns with NaN values and their count of NaNs
nan_columns = dataset.isna().sum()
#print(nan_columns[nan_columns > 0])


# In[612]:


# Temporarily adjust the display settings to show more rows
with pd.option_context('display.max_rows', None):  # None means show all rows
    # Checking for columns with NaN values and their count of NaNs
    nan_columns = dataset.isna().sum()
    #print(nan_columns[nan_columns > 0])


# In[613]:


dataset_no_nan = dataset.fillna(0)


# In[614]:


dataset_no_nan.isna().any().sum()


# In[615]:


#print(list(dataset_no_nan.columns))


# In[616]:


dataset_no_nan.set_index('day',inplace=True)


# In[617]:


dataset_no_nan.head(20)


# In[618]:


numeric_dataset = dataset_no_nan.select_dtypes(include=[np.number])


# In[619]:


inf_mask = np.isinf(aggregated_vault_data)
rows_with_inf = numeric_dataset[inf_mask.any(axis=1)]
#print("Rows with Infinite Values:")
#print(rows_with_inf)


# In[620]:


inf_mask = np.isinf(numeric_dataset)

# Check which rows contain any infinite values
rows_with_inf = numeric_dataset[inf_mask.any(axis=1)]

# Display these rows
#print("Rows with Infinite Values:")
#print(rows_with_inf)


# In[621]:


# Iterate through rows that contain inf and display which columns have inf
for index, row in rows_with_inf.iterrows():
    columns_with_inf = row.index[row.apply(np.isinf)]
    #print(f"Row {index} has infinite values in columns: {list(columns_with_inf)}")


# In[622]:


# Check if any row has at least one NaN value
rows_with_nan = numeric_dataset[numeric_dataset.isna().any(axis=1)]

# #print rows with NaN values if any
#print("Rows with NaN values:")
#print(rows_with_nan)




# In[623]:


numeric_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_dataset.fillna(method='ffill', inplace=True)


# In[624]:


aggregated_vault_data.replace([np.inf, -np.inf], np.nan, inplace=True)
aggregated_vault_data.fillna(method='ffill', inplace=True)


# In[625]:


#print(dataset_no_nan.isnull().sum())


# ## Running Correlations for Feature Engineering

# Check the pearson correlation, spearman correlation, and mutual information values.  Take the most correlated features (>0.5) and test each feature set with each ml model.  Feature set with highest score will be chosen.   
# 
# If the non-linear ml models perform better, assume non-linear relationship and focus on features from spearman/mutual information

# ## Pearson correlation

# In[626]:





# # Individual Regressor Models

# ## ETH Vault

# In[699]:




# # MVO

# We will compare MVO with varying targets, including sharpe ratio, sortino ratio, and pure returns

# ## Sharpe ratios

# In[743]:



st_aggregated_vault_data = aggregated_vault_data[aggregated_vault_data.index > '2021-11-18']
st_aggregated_vault_data.isna().sum().sum()

# # RDL 

# ## State Space Features

# In[798]:





# In[824]:


#print(list(st_aggregated_vault_data.columns))


# In[825]:





# In[826]:


## Multiple Regression


# Example data
# X represents features including one Dai ceiling and possibly other market conditions
# y represents multiple vault balances (multi-output)



# ## Forecast Set Up

# In[833]:


targets = ['ETH Vault_collateral_usd','stETH Vault_collateral_usd','BTC Vault_collateral_usd','Altcoin Vault_collateral_usd','Stablecoin Vault_collateral_usd','LP Vault_collateral_usd','PSM Vault_collateral_usd']
features = ['ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','PSM Vault_dai_ceiling',

'BTC Vault_market_price', 'ETH Vault_market_price', 'stETH Vault_market_price', 'Stablecoin Vault_market_price', 'Altcoin Vault_market_price', 'LP Vault_market_price', 'effective_funds_rate',
'M1V', 'WM2NS', 'fed_reverse_repo','mcap_total_volume','defi_apy_medianAPY', 'defi_apy_avg7day', 'dpi_market_Volume',
'ETH Vault_liquidation_ratio', 'BTC Vault_liquidation_ratio', 'stETH Vault_liquidation_ratio', 'Altcoin Vault_liquidation_ratio', 'Stablecoin Vault_liquidation_ratio', 'LP Vault_liquidation_ratio', 
'RWA Vault_dai_ceiling',
'BTC Vault_collateral_usd % of Total', 'ETH Vault_collateral_usd % of Total', 'stETH Vault_collateral_usd % of Total',
'Stablecoin Vault_collateral_usd % of Total', 'Altcoin Vault_collateral_usd % of Total', 'LP Vault_collateral_usd % of Total',
'RWA Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'ETH Vault_collateral_usd % of Total_7d_ma', 'ETH Vault_collateral_usd % of Total_30d_ma',
'stETH Vault_collateral_usd % of Total_7d_ma', 'stETH Vault_collateral_usd % of Total_30d_ma',
'BTC Vault_collateral_usd % of Total_7d_ma', 'BTC Vault_collateral_usd % of Total_30d_ma',
'Altcoin Vault_collateral_usd % of Total_7d_ma', 'Altcoin Vault_collateral_usd % of Total_30d_ma',
'Stablecoin Vault_collateral_usd % of Total_7d_ma', 'Stablecoin Vault_collateral_usd % of Total_30d_ma',
'LP Vault_collateral_usd % of Total_7d_ma', 'LP Vault_collateral_usd % of Total_30d_ma',
 'Vaults Total USD Value',
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling',
'RWA Vault_collateral_usd', 'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma',
'where_is_dai_Bridge', 'dai_market_Volume_30d_ma', 'dai_market_Volume_7d_ma','eth_market_Close_7d_ma','eth_market_Volume_30d_ma','btc_market_Close_7d_ma',
'btc_market_Volume_30d_ma','LP Vault_dai_floor_90d_ma_pct_change'


       ]



temporal_features = ['ETH Vault_collateral_usd % of Total_7d_ma', 'ETH Vault_collateral_usd % of Total_30d_ma',
'stETH Vault_collateral_usd % of Total_7d_ma', 'stETH Vault_collateral_usd % of Total_30d_ma',
'BTC Vault_collateral_usd % of Total_7d_ma', 'BTC Vault_collateral_usd % of Total_30d_ma',
'Altcoin Vault_collateral_usd % of Total_7d_ma', 'Altcoin Vault_collateral_usd % of Total_30d_ma',
'Stablecoin Vault_collateral_usd % of Total_7d_ma', 'Stablecoin Vault_collateral_usd % of Total_30d_ma',
'LP Vault_collateral_usd % of Total_7d_ma', 'LP Vault_collateral_usd % of Total_30d_ma',
'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma', 'Vaults Total USD Value',
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling',
'RWA Vault_collateral_usd', 'RWA Vault_collateral_usd % of Total_7d_ma', 'RWA Vault_collateral_usd % of Total_30d_ma']
descripts = [


'ETH Vault_collateral_usd','stETH Vault_collateral_usd','BTC Vault_collateral_usd','Altcoin Vault_collateral_usd','Stablecoin Vault_collateral_usd','LP Vault_collateral_usd','PSM Vault_collateral_usd',
'ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','PSM Vault_dai_ceiling'



]
temporals = [
 'BTC Vault_collateral_usd % of Total', 'ETH Vault_collateral_usd % of Total', 'stETH Vault_collateral_usd % of Total', 'where_is_dai_Bridge',
'Stablecoin Vault_collateral_usd % of Total', 'Altcoin Vault_collateral_usd % of Total', 'LP Vault_collateral_usd % of Total',
'RWA Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total', 'PSM Vault_collateral_usd % of Total_7d_ma', 'PSM Vault_collateral_usd % of Total_30d_ma',
'ETH Vault_collateral_usd % of Total_7d_ma', 
'stETH Vault_collateral_usd % of Total_7d_ma', 
'BTC Vault_collateral_usd % of Total_7d_ma',
'Altcoin Vault_collateral_usd % of Total_7d_ma', 
'Stablecoin Vault_collateral_usd % of Total_7d_ma', 
'LP Vault_collateral_usd % of Total_7d_ma', 
'PSM Vault_collateral_usd % of Total_7d_ma', 
'RWA Vault_collateral_usd % of Total_7d_ma', 'Vaults Total USD Value',
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling',
'RWA Vault_collateral_usd', 'RWA Vault_collateral_usd % of Total_7d_ma', 
'BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling'
]

prev_ceiling = ['BTC Vault_prev_dai_ceiling', 'ETH Vault_prev_dai_ceiling', 'stETH Vault_prev_dai_ceiling', 'Altcoin Vault_prev_dai_ceiling',
'LP Vault_prev_dai_ceiling', 'Stablecoin Vault_prev_dai_ceiling', 'PSM Vault_prev_dai_ceiling']

dai_ceilings = [
            'ETH Vault_dai_ceiling','stETH Vault_dai_ceiling','BTC Vault_dai_ceiling', 
    'Altcoin Vault_dai_ceiling','Stablecoin Vault_dai_ceiling','LP Vault_dai_ceiling','RWA Vault_dai_ceiling','PSM Vault_dai_ceiling'
        ]


# In[834]:


test_data = st_aggregated_vault_data[targets + features]


# In[835]:


test_data.index = pd.to_datetime(test_data.index)


# In[836]:


test_data.isna().sum().sum()


# In[837]:


test_data_copy = test_data.copy()


# In[838]:


# Check for all duplicated columns (including the first occurrence)
all_duplicated_columns = test_data_copy.columns.duplicated(keep=False)

# #print out the names of all duplicated columns
#print("All duplicate columns:")
for idx, is_duplicate in enumerate(all_duplicated_columns):
    if is_duplicate:
        print(test_data_copy.columns[idx])


# In[839]:


temporals


# In[840]:


test_data_copy.columns

test_data_copy.to_csv('data/csv/test_data.csv')

# import test_data_copy, targets, features
