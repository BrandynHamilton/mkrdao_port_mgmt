import plotly.graph_objs as go
import plotly.offline as pyo
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

from scripts.data_processing import test_data_copy, test_data, targets, features, temporals, dai_ceilings
from scripts.mvo_agent import MVOAgent
from scripts.utils import mvo, historical_sortino, visualize_mvo_results, evaluate_predictions, generate_action_space 
from scripts.simulation import RL_VaultSimulator
from scripts.environment import SimulationEnvironment


random.seed(20)
np.random.seed(20)
tf.random.set_seed(20)

simulation_data = test_data_copy

portfolio_mvo_weights, portfolio_returns, portfolio_composition, total_portfolio_value = mvo(simulation_data)


# In[882]:


# Assuming you have these vault names as keys in optimized_weight_dict
vault_names = [
    'BTC Vault_collateral_usd', 'ETH Vault_collateral_usd', 'stETH Vault_collateral_usd', 'Stablecoin Vault_collateral_usd', 
    'Altcoin Vault_collateral_usd', 'LP Vault_collateral_usd', 'RWA Vault_collateral_usd', 'PSM Vault_collateral_usd'
]

# Map from the detailed keys to simplified keys used in optimized_weight_dict
key_mapping = {
    'BTC Vault_collateral_usd',
    'ETH Vault_collateral_usd',
    'stETH Vault_collateral_usd',
    'Stablecoin Vault_collateral_usd'
    'Altcoin Vault_collateral_usd',
    'LP Vault_collateral_usd',
    'RWA Vault_collateral_usd',
    'PSM Vault_collateral_usd'
}

# Apply this mapping to the initial_weights to align with optimized_weight_dict
initial_weights = dict(portfolio_composition.loc['2022-05-20'].to_dict())

optimized_weight_dict = dict(zip(vault_names, portfolio_mvo_weights))

# Now both dictionaries use the same keys:
print("Initial Weights on 2022-05-20:", initial_weights)
print("Optimized Weights:", optimized_weight_dict)

# Your RlAgent can now be initialized and used with these dictionaries


# In[883]:





# ### Sets Up Action Space: for Some Reason, need to run this first for decent predictions on first try: think related to how some variables are initialized

# In[884]:


str(simulation_data.index.max())


# test_ceilings = test_data[dai_ceilings]
# 
# 
# start_date_dt = pd.to_datetime(start_date)  # Convert string to datetime
# historical_cutoff = start_date_dt - pd.DateOffset(days=1)
# test_historical_data = test_ceilings[test_ceilings.index <= historical_cutoff]
# 
# test_historical_data
# last_dai_ceiling = test_historical_data.iloc[-1]
# last_dai_ceiling

# In[885]:


#test_data_copy[['mcap_total_volume']].plot()


# ### Non Q Sim

# In[886]:






# Define action space as a list of dictionaries
vault_action_ranges = {
        'stETH Vault_dai_ceiling': [-0.5, 0.5],
        'ETH Vault_dai_ceiling': [-0.5, 0.5],
        'BTC Vault_dai_ceiling': [-0.5, 0.5], # can try -1 for BTC
        'Altcoin Vault_dai_ceiling': [-0.25, 0.25],
        'Stablecoin Vault_dai_ceiling': [-0.25, 0.25],
        'LP Vault_dai_ceiling': [0 , 0], # The volatility causes abnormal daily returns
        'RWA Vault_dai_ceiling': [0, 0],  # If no changes are allowed
        'PSM Vault_dai_ceiling': [-0.5, 0.5]
    }

action_space = generate_action_space(vault_action_ranges)

# Assuming `optimized_weight_dict` and `initial_weights` are defined elsewhere in your script
agent = MVOAgent(action_space, optimized_weight_dict, vault_action_ranges, initial_strategy_period=1)
initial_action = agent.initial_strategy(initial_weights)

print("Initial Action Computed:", initial_action)


simulation_data = test_data_copy  
simulation_data.index = simulation_data.index.tz_localize(None)  # Remove timezone information
#05-20-2022
start_date = '2022-05-20'
end_date = '2024-03-20'
#end_date = '2022-07-20'


simulator = RL_VaultSimulator(simulation_data, test_data_copy, features, targets, temporals, start_date, end_date, scale_factor=300000000, minimum_value_factor=0.05, volatility_window=250)
simulator.train_model()
#simulator.set_parameters()

environment = SimulationEnvironment(simulator, start_date, end_date, agent)
environment.reset()

state, reward, done, info = environment.run()

#simulator.plot_simulation_results()
action_df = environment.generate_action_dataframe()


# ### DAI Ceilings
# 

# In[888]:


sim_dai_ceilings = simulator.dai_ceilings_history
#sim_dai_ceilings.plot()


# In[889]:


sim_dai_ceilings


# ### Actions Log

# In[890]:
historical = test_data_copy[targets]

start_date_dt = pd.to_datetime(start_date)  # Convert string to datetime
historical_cutoff = start_date_dt - pd.DateOffset(days=1)
historical_data = historical[historical.index <= historical_cutoff]

test_data.index = pd.to_datetime(test_data.index).tz_localize(None)


historical_data = historical_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
historical_data

historical_optimized_weights, historical_portfolio_returns, historical_portfolio_composition, historical_total_portfolio_value = mvo(historical_data)
historical_portfolio_daily_returns,  historical_downside_returns, historical_excess_returns, historical_sortino_ratio = historical_sortino(historical_portfolio_returns,historical_portfolio_composition)


# In[891]:


action_df.set_index('date', inplace=True)
action_df.columns


# In[892]:


action_df['dai ceilings'].dropna().to_dict()


# In[893]:


action_df['target weights'].dropna().to_dict()


# In[894]:


action_df['current composition'].dropna().to_dict()


# In[895]:


action_df['current sortino ratio'].dropna().plot()


# In[896]:


#action_df['current cumulative return'].dropna().plot()


# In[897]:


action_df['action'].dropna().to_dict()


# In[898]:


action_df
actions_log_path = 'results/mvoactionslog.csv'
action_df.to_csv(actions_log_path)


# In[899]:


simulations_path = 'results/mvosimulations.csv'
simulator.results.to_csv(simulations_path)


# In[900]:


mvo_sim_results = simulator.results
evaluate_predictions(mvo_sim_results, historical)


# ### Data Prep for MVO

# In[901]:


start_date_dt = pd.to_datetime(start_date)  # Convert string to datetime
historical_cutoff = start_date_dt - pd.DateOffset(days=1)
historical_cutoff


# In[902]:


historical_ceilings = test_data_copy[dai_ceilings]
ceiling_historical_cutoff = historical_cutoff.tz_localize(None)

historical_ceilings_for_sim =  historical_ceilings[historical_ceilings.index <= ceiling_historical_cutoff]
historical_ceilings_for_sim
combined_ceiling_data = pd.concat([historical_ceilings_for_sim, sim_dai_ceilings])
combined_ceiling_data


# In[903]:


test_data.columns


# In[904]:


#combined_ceiling_data.plot()


# In[905]:


#historical_ceilings.plot()


# In[906]:


historical_data = historical[historical.index <= historical_cutoff]
mvo_combined_data = pd.concat([historical_data, mvo_sim_results])
#mvo_combined_data = pd.concat([historical_data, mvo_sim_results])
#print(mvo_combined_data)
print(mvo_combined_data)


# In[907]:


mvo_combined_data.index = mvo_combined_data.index.tz_localize(None)
#mvo_combined_data.index = mvo_combined_data.index.tz_localize(None)

test_data['RWA Vault_collateral_usd'].index = test_data['RWA Vault_collateral_usd'].index.tz_localize(None)


#Since RWA is not a target, we need to add back in for MVO calculations
mvo_sim_w_RWA = mvo_combined_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
#mvo_sim_w_RWA = mvo_combined_data.merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')

# Optional: Sort the DataFrame by index if it's not already sorted
mvo_sim_w_RWA.sort_index(inplace=True)
#mvo_sim_w_RWA.sort_index(inplace=True)


# Now 'combined_data' contains both historical and simulation data in one DataFrame
#mvo_sim_w_RWA.plot()
#mvo_sim_w_RWA.plot()


# In[908]:


mvo_sim_w_RWA.iloc[-1].sum()


# In[909]:


sim_cutoff = mvo_sim_w_RWA.index.max()
sim_cutoff


# In[910]:


# Assuming 'test_data' is the DataFrame with the timezone-aware index
#test_data.index = pd.to_datetime(test_data.index).tz_localize(None)

# Now perform the merge
historical_sim = test_data[test_data.index <= sim_cutoff]
historical_sim = historical_sim[targets].merge(test_data['RWA Vault_collateral_usd'], left_index=True, right_index=True, how='left')
#historical_sim.plot()
historical_sim


# ### Historical Stats

# In[911]:


historical_optimized_weights, historical_portfolio_returns, historical_portfolio_composition, historical_total_portfolio_value = mvo(historical_sim)
print('average daily return per vault:', historical_portfolio_returns.mean())
print('current composition:', historical_portfolio_composition.iloc[-1])
print(f'current portfolio value: ${historical_total_portfolio_value.iloc[-1]:,.2f}')


# In[912]:


historical_portfolio_daily_returns,  historical_downside_returns, historical_excess_returns, historical_sortino_ratio = historical_sortino(historical_portfolio_returns,historical_portfolio_composition)


# In[913]:


historical_returns = visualize_mvo_results(historical_portfolio_daily_returns, historical_downside_returns, historical_excess_returns)


# ### Sim Stats - Final Simulation w/ Agent Scores

# In[914]:


mvo_sim_w_RWA.iloc[-1].sum()


# In[915]:


#mvo_sim_w_RWA.iloc[-1].sum()


# In[916]:


# Optimized Weights for Simulation

mvo_portfolio_mvo_weights, mvo_portfolio_returns, mvo_portfolio_composition, mvo_total_portfolio_value = mvo(mvo_sim_w_RWA)
print('optimized weights:', mvo_portfolio_mvo_weights)
print('current composition:', mvo_portfolio_composition.iloc[-1])
print(f'current portfolio value: ${mvo_total_portfolio_value.iloc[-1]:,.2f}')


# mvo_portfolio_mvo_weights, mvo_portfolio_returns, mvo_portfolio_composition, mvo_total_portfolio_value = mvo(mvo_sim_w_RWA)
# print('optimized weights:', mvo_portfolio_mvo_weights)
# print('current composition:', mvo_portfolio_composition.iloc[-1])
# print(f'current portfolio value: ${mvo_total_portfolio_value.iloc[-1]:,.2f}')

# mvo_sim_portfolio_daily_returns,  mvo_sim_downside_returns, mvo_sim_excess_returns, mvo_sim_sortino_ratio = historical_sortino(mvo_portfolio_returns,mvo_portfolio_composition)

# In[917]:


mvo_sim_portfolio_daily_returns,  mvo_sim_downside_returns, mvo_sim_excess_returns, mvo_sim_sortino_ratio = historical_sortino(mvo_portfolio_returns,mvo_portfolio_composition)


# In[918]:


mvo_optimized_returns = visualize_mvo_results(mvo_sim_portfolio_daily_returns, mvo_sim_downside_returns, mvo_sim_excess_returns)


# In[919]:


#mvo_optimized_returns = visualize_mvo_results(mvo_sim_portfolio_daily_returns, mvo_sim_downside_returns, mvo_sim_excess_returns)


# ### Comparisons on charts

# In[920]:


#print(mvo_optimized_returns.describe())
print(mvo_optimized_returns.describe())


# In[921]:


# Find the duplicate indices
#mvo_duplicate_indices = mvo_optimized_returns.index[mvo_optimized_returns.index.duplicated()]
mvo_duplicate_indices = mvo_optimized_returns.index[mvo_optimized_returns.index.duplicated()]

# Print the duplicate indices
#print("MVO Duplicate indices:", mvo_duplicate_indices)
print("RL Duplicate indices:", mvo_duplicate_indices)

# Drop duplicates, keeping the first occurrence
mvo_optimized_returns = mvo_optimized_returns[~mvo_optimized_returns.index.duplicated(keep='first')]
#mvo_optimized_returns = mvo_optimized_returns[~mvo_optimized_returns.index.duplicated(keep='first')]

# Verify if duplicates are removed
print("After removing duplicates")
#print("MVO indices:", mvo_optimized_returns.index)
print("RL indices:", mvo_optimized_returns.index)


# In[922]:


historical_returns.index


# In[923]:


all_dates = mvo_optimized_returns.index.union(historical_returns.index)
#optimized_returns = optimized_returns.reindex(all_dates, fill_value=0)
#historical_returns = historical_returns.reindex(all_dates, fill_value=0)

plt.figure(figsize=(10, 5))
plt.plot(mvo_optimized_returns.index, mvo_optimized_returns.values, label='RL Robo Advisor Optimized Returns', color='blue')
#plt.plot(mvo_optimized_returns.index, mvo_optimized_returns.values, label='MVO Robo Advisor Optimized Returns', color='green')
plt.plot(historical_returns.index, historical_returns.values, label='Historical Returns', color='red')
plt.title('Optimized vs Historical Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()


# ### Sim Composition

# In[924]:


mvo_total_portfolio_value


# In[925]:


plt.pie(mvo_portfolio_composition.iloc[-1], labels = mvo_portfolio_composition.columns, autopct='%1.1f%%')
plt.show() 
print(f'current portfolio value: ${mvo_total_portfolio_value.iloc[-1]:,.2f}')


# plt.pie(mvo_portfolio_composition.iloc[-1], labels = mvo_portfolio_composition.columns, autopct='%1.1f%%')
# plt.show() 
# print(f'current portfolio value: ${mvo_total_portfolio_value.iloc[-1]:,.2f}')

# ### Historical Composition 

# In[926]:


plt.pie(historical_portfolio_composition.iloc[-1], labels = historical_portfolio_composition.columns, autopct='%1.1f%%' )
plt.show() 
print(f'current portfolio value: ${historical_total_portfolio_value.iloc[-1]:,.2f}')


# In[927]:




# Create traces for each line
trace1 = go.Scatter(
    x=mvo_total_portfolio_value.index,
    y=mvo_total_portfolio_value.values,
    mode='lines',
    name='RL Robo Advisor',
    line=dict(color='blue')
)
"""
trace2 = go.Scatter(
    x=mvo_total_portfolio_value.index,
    y=mvo_total_portfolio_value.values,
    mode='lines',
    name='MVO Robo Advisor Portfolio',
    line=dict(color='green')
)
"""
trace3 = go.Scatter(
    x=historical_total_portfolio_value.index,
    y=historical_total_portfolio_value.values,
    mode='lines',
    name='Historical Portfolio',
    line=dict(color='red')
)

# Create the layout
layout = go.Layout(
    title='Robo Advisor vs Historical TVL',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Value'),
    legend=dict(x=0.1, y=0.9)
)

# Combine the data and layout into a figure
fig = go.Figure(data=[trace1, trace3], layout=layout)

# Render the figure
pyo.iplot(fig)


# In[928]:


## Add historical MAR from annual risk_free rate to daily risk free  


# In[929]:


ax = mvo_sim_excess_returns.plot(style='go', label='MVO Simulated Excess Returns', title='Sim vs Historical Excess Returns Over MAR')
# Plot the second set of excess returns on the same axis
#mvo_sim_excess_returns.plot(ax=ax, style='bo', alpha=0.7, label='MVO Simulated Excess Returns')  # 'ro' for red dots, you can change the color and marker style
historical_excess_returns.plot(ax=ax, style='ro', alpha=0.7, label='Historical Excess Returns')  # 'ro' for red dots, you can change the color and marker style

# Add a line at the minimum acceptable return (MAR) for reference
plt.axhline(0, color='k', linestyle='--')

# Set the labels and grid
plt.xlabel('Date')
plt.ylabel('Excess Returns')
plt.grid(True)

# Add the legend
plt.legend()

# Show the plot
plt.show()


# sim_portfolio_daily_returns.describe()

# historical_portfolio_daily_returns.describe()

# (1 + sim_portfolio_daily_returns).cumprod()

# # Set the display option to show all rows
# pd.set_option('display.max_rows', None)
# 
# # Assuming 'sim_portfolio_daily_returns' is your daily returns Series
# cumulative_returns = (1 + sim_portfolio_daily_returns).cumprod()
# 
# # Display the cumulative returns
# cumulative_returns
# 
# # Reset display option if you want to limit the display in other parts of your code
# 

# pd.reset_option('display.max_rows')
# (1 + historical_portfolio_daily_returns).cumprod()

# In[ ]:



