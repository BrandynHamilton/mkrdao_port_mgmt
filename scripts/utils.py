
def mvo(data, annual_risk_free_rate=0.05, penalty_factor=10000):  # Increased penalty factor
    portfolio = data[['BTC Vault_collateral_usd', 'ETH Vault_collateral_usd', 'stETH Vault_collateral_usd', 
                      'Stablecoin Vault_collateral_usd', 'Altcoin Vault_collateral_usd', 'LP Vault_collateral_usd', 
                      'RWA Vault_collateral_usd', 'PSM Vault_collateral_usd']]
    log_returns = np.log(portfolio / portfolio.shift(1))
    log_returns.fillna(0, inplace=True)
    
    total_portfolio_value = portfolio.sum(axis=1)
    composition = portfolio.divide(total_portfolio_value, axis=0)
    
    daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/365) - 1
    excess_returns = log_returns.subtract(daily_risk_free_rate, axis=0)
    
    downside_returns = np.where(log_returns < daily_risk_free_rate, log_returns - daily_risk_free_rate, 0)
    daily_downside_deviation = np.sqrt((downside_returns ** 2).mean())
    annual_downside_deviation = daily_downside_deviation * np.sqrt(365)
    
    annualized_returns = excess_returns.mean() * 365
    sortino_ratios = annualized_returns / annual_downside_deviation if annual_downside_deviation != 0 else np.inf
    print("Individual Sortino Ratios:", sortino_ratios)

    def sortino_ratio(weights):
        portfolio_returns = np.sum(weights * excess_returns, axis=1)
        annualized_returns = np.mean(portfolio_returns) * 365
        downside_deviation = np.sqrt(np.mean(np.minimum(0, portfolio_returns - daily_risk_free_rate) ** 2)) * np.sqrt(365)
        
        lower_bounds, upper_bounds = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        penalties = penalty_factor * (np.sum(np.maximum(0, lower_bounds - weights) ** 2) +
                                     np.sum(np.maximum(0, weights - upper_bounds) ** 2))
        
        return -(annualized_returns / downside_deviation) + penalties if downside_deviation != 0 else -np.inf

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0.1, 0.5), (0.05, 0.3), (0.05, 0.5), (0.1, 0.1), (0.01, 0.2), (0.01, 0.3), (0.01, 0.02), (0.01, 0.25)]
    initial_weights = np.clip(composition.iloc[-1].values, a_min=[b[0] for b in bounds], a_max=[b[1] for b in bounds])

    result = minimize(sortino_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        optimized_weights = result.x
        print('Optimized weights:', optimized_weights)
        return optimized_weights, log_returns, composition, total_portfolio_value
    else:
        raise Exception('Optimization did not converge')

# Example usage with your data variable
# all_data_mvo, all_data_returns, all_data_composition, all_data_portfolio_value = mvo(your_data_variable)


# In[802]:


## w/o percent columsn



# In[803]:



# In[804]:


def optimized_sortino(returns_df, weights, annual_risk_free_rate=0.05):
    """
    Calculate the Sortino Ratio for a given set of returns and weights.
    
    Parameters:
        returns_df (DataFrame): DataFrame of returns.
        weights (array): Asset weights in the portfolio.
        annual_risk_free_rate (float): Annual risk-free rate, default is 5%.
        
    Returns:
        tuple: Contains daily returns, downside returns, excess returns, and the Sortino Ratio.
    """
    # Calculate daily risk-free rate
    daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/365) - 1
    
    # Portfolio daily returns
    daily_returns = returns_df.dot(weights)
    
    # Excess returns over the risk-free rate
    excess_returns = daily_returns - daily_risk_free_rate
    
    # Calculate downside returns
    downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
    
    # Daily and annualized downside deviation
    daily_downside_deviation = np.sqrt(downside_returns.mean())
    annualized_downside_deviation = daily_downside_deviation * np.sqrt(365)
    
    # Annualized excess return calculation
    if len(excess_returns) > 0:
        annualized_excess_return = ((1 + excess_returns).prod())**(365 / len(excess_returns)) - 1
    else:
        annualized_excess_return = -1  # Handle case with no returns
    
    # Calculate Sortino Ratio
    target_sortino_ratio = (annualized_excess_return / annualized_downside_deviation if annualized_downside_deviation != 0 else np.inf)
    
    # Debugging prints can be commented out in production
    print('Daily downside deviation:', daily_downside_deviation)
    print('Annualized downside deviation:', annualized_downside_deviation)
    print('Annualized excess return:', annualized_excess_return)
    print("Sortino Ratio:", target_sortino_ratio)

    return daily_returns, downside_returns, excess_returns, target_sortino_ratio


# In[805]:


def historical_sortino(returns, composition, annual_risk_free_rate=0.05):
    """
    Calculate the Sortino Ratio for a portfolio based on historical returns and asset composition.
    
    Parameters:
        returns (DataFrame): DataFrame of returns.
        composition (DataFrame): DataFrame of asset weights in the portfolio.
        annual_risk_free_rate (float): Annual risk-free rate, default is 5%.
        
    Returns:
        tuple: Contains portfolio daily returns, downside returns, excess returns, and the Sortino Ratio.
    """
    # Calculate daily risk-free rate
    daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/365) - 1
    
    # Portfolio daily returns
    portfolio_daily_returns = (returns * composition).sum(axis=1)
    
    # Excess returns over the risk-free rate
    excess_returns = portfolio_daily_returns - daily_risk_free_rate
    
    # Calculate downside returns
    downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
    
    # Daily and annualized downside deviation
    daily_downside_deviation = np.sqrt(downside_returns.mean())
    annualized_downside_deviation = daily_downside_deviation * np.sqrt(365)
    
    # Annualized excess return calculation
    if len(excess_returns) > 0:
        annualized_excess_return = ((1 + excess_returns).prod())**(365 / len(excess_returns)) - 1
    else:
        annualized_excess_return = -1  # Handle case with no returns
    
    # Calculate Sortino Ratio
    sortino_ratio = annualized_excess_return / annualized_downside_deviation if annualized_downside_deviation != 0 else np.inf
    
    # Debugging prints can be commented out in production
    print('Daily downside deviation:', daily_downside_deviation)
    print('Annualized downside deviation:', annualized_downside_deviation)
    print('Annualized excess return:', annualized_excess_return)
    print("Portfolio Sortino Ratio:", sortino_ratio)

    return portfolio_daily_returns, downside_returns, excess_returns, sortino_ratio


# In[806]:


def visualize_mvo_results(daily_returns, downside_returns, excess_returns):

    # 1. Time Series Plot of Portfolio Returns
    plt.figure(figsize=(10, 6))
    daily_returns.plot(title='Daily Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.grid(True)
    plt.show()
    
    # 2. Histogram of Portfolio Returns
    plt.figure(figsize=(10, 6))
    sns.histplot(daily_returns, kde=True, bins=30)
    plt.title('Histogram of Portfolio Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()
    
    # 3. Cumulative Returns Plot
    optimized_cumulative_returns = (1 + daily_returns).cumprod()
    plt.figure(figsize=(10, 6))
    optimized_cumulative_returns.plot(title='Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()

    downside_returns = pd.Series(downside_returns, index=daily_returns.index)
    
    # 4. Downside Returns Plot
    plt.figure(figsize=(10, 6))
    downside_returns.plot(style='ro', title='Downside Returns Below MAR')
    plt.axhline(0, color='k', linestyle='--')  # Add a line at 0 for reference
    plt.xlabel('Date')
    plt.ylabel('Downside Returns')
    plt.grid(True)
    plt.show()
    
    excess_returns = pd.Series(excess_returns, index=daily_returns.index)

    # 5. Excess Returns Over MAR
    excess_returns.plot(style='go', title='Excess Returns Over MAR')
    plt.axhline(0, color='k', linestyle='--')  # Add a line at MAR for reference
    plt.xlabel('Date')
    plt.ylabel('Excess Returns')
    plt.grid(True)
    plt.show()

    return optimized_cumulative_returns
