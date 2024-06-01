class SimulationEnvironment:
    def __init__(self, simulator, start_date, end_date, agent=None):
        self.simulator = simulator
        self.start_date = pd.to_datetime(start_date).tz_localize(None)
        self.end_date = pd.to_datetime(end_date).tz_localize(None)
        self.current_date = self.start_date 
        self.agent = agent
        self.action_log = []
        self.dai_ceiling = None
        self.previous_total_portfolio_value = None  # Initialize previous total portfolio value
        self.value_change_factor = 1
        
    def run(self, predefined_actions=None):
        state = self.reset()  # Ensures starting from a clean state every time
        done = False
        reward = 0
        while self.current_date <= self.end_date:
            next_cycle_date = min(self.current_date + pd.DateOffset(days=24), self.end_date)
            print(f"Running simulation from {self.current_date} to {next_cycle_date}")
            
            if predefined_actions:
                action = predefined_actions
            else:
                action = self.agent.agent_policy(state, self.dai_ceiling) if self.agent else None

            if action:
                print(f"Action at {self.current_date}: {action}")
                self.action_log.append({'date': self.current_date, 'action': action})
            else:
                print(f"No action taken at {self.current_date}")

            self.simulator.run_simulation(self.current_date.strftime('%Y-%m-%d'), action)
            state = self.get_state(self.current_date)
            print('state in run as of self.current_date', state)
            reward, current_weights = self.reward_function()  # Assumes this function is defined to handle rewards
            print(f"Reward at the end of the cycle on {next_cycle_date}: {reward}")

            self.current_date = next_cycle_date + pd.DateOffset(days=1)
            if self.current_date > self.end_date:
                done = True
                print(f"Reached or passed end date: stopping simulation at {self.current_date}")
                break
            
            print(f"Completed cycle up to {next_cycle_date}")
        return state, reward, done, {}

    def reset(self):
        self.simulator.reset()
        self.current_date = self.start_date
        print("Environment and simulator reset.")
        self.action_log.append({'date': self.current_date, 'env reset start date': self.start_date})
        self.simulator.current_date = self.current_date
        return self.get_state(self.current_date)

    def get_state(self, simulation_date):
        print('self date', self.current_date)
        print('self.simulator date', self.simulator.current_date)
        self.current_date = self.simulator.current_date
        print('new self date', self.current_date)
        simulation_date = self.simulator.current_date

        if simulation_date in self.simulator.data.index:
            state_data = self.simulator.data.loc[simulation_date, self.simulator.targets]
        else:
            previous_dates = self.simulator.data.index[self.simulator.data.index < simulation_date]
            if not previous_dates.empty:
                last_available_date = previous_dates[-1]
                state_data = self.simulator.data.loc[last_available_date, self.simulator.targets]
            else:
                state_data = self.simulator.data[self.simulator.targets].iloc[-1]

        total_value = state_data.sum()
        self.action_log.append({'date': self.current_date, 'get state calculated portfolio value': total_value})
        relative_weights = state_data / total_value if total_value != 0 else state_data
        print('relative weights:', relative_weights)
        print(f"Current state fetched for date {simulation_date}: {relative_weights}")
        return relative_weights.to_dict()

    def run_daily_simulation(self, simulation_date):
        volatilities = self.simulator.calculate_historical_volatility()
        if simulation_date in self.simulator.data.index:
            X_test = self.simulator.data.loc[[simulation_date], self.simulator.features]
        else:
            X_test = self.simulator.data.loc[self.simulator.data.index < simulation_date, self.simulator.features].iloc[-1].to_frame().T

        predictions = self.simulator.forecast(X_test, volatilities)
        self.simulator.update_state(pd.DatetimeIndex([simulation_date]), predictions)
   
    def step(self, action=None):
        reward = 0
        if action:
            print('action', action)
            self.simulator.apply_action(action)
            self.action_log.append({'date': self.current_date, 'action': action})

        if not hasattr(self, 'current_simulation_date') or self.current_simulation_date < self.start_date:
            self.current_simulation_date = self.start_date
        
        self.current_simulation_date += pd.DateOffset(days=1)
        
        if self.current_simulation_date <= self.end_date:
            self.run_daily_simulation(self.current_simulation_date)
            done = False
        else:
            done = True
            reward, current_weights = self.reward_function()

        current_state = self.get_state(self.current_simulation_date)
        return current_state, reward, done, {}

    def apply_action(self, action):
        for vault_name, adjustment in action.items():
            ...

        self.action_log.append({
            'date': self.current_date,
            'action': action,
            'additional_info': {}
        })

    def generate_action_dataframe(self):
        return pd.DataFrame(self.action_log)

    def reward_function(self):
        print('Simulator data for MVO:', self.simulator.data[self.simulator.targets])
        optimized_weights, returns, composition, total_portfolio_value = mvo(self.simulator.data)
        self.action_log.append({'date': self.current_date, 'mvo calculated current portfolio value': total_portfolio_value.iloc[-1]})
        self.action_log.append({'date': self.current_date, 'calculated optimized weights': optimized_weights})
        self.action_log.append({'date': self.current_date, 'current composition': composition.iloc[-1]})

        if self.agent is None:
            current_weights = composition.iloc[-1].to_dict()
            target_weights = {k: v for k, v in zip(composition.columns, optimized_weights)}
            print('Target weights (no agent):', target_weights)
        else:
            current_weights = composition.iloc[-1].to_dict()
            self.agent.target_weights = {k: v for k, v in zip(composition.columns, optimized_weights)}
            target_weights = self.agent.target_weights
            self.action_log.append({'date': self.current_date, 'agent weights updated to': target_weights})
            print('Agent weights updated:', target_weights)

        current_daily_returns, current_downside_returns, current_excess_returns, current_sortino_ratio = historical_sortino(returns, composition)
        self.action_log.append({'date': self.current_date, 'current sortino ratio': current_sortino_ratio})
        target_daily_returns, target_downside_returns, target_excess_returns, target_sortino_ratio = optimized_sortino(returns, optimized_weights)
        self.action_log.append({'date': self.current_date, 'target sortino ratio': target_sortino_ratio})
        self.action_log.append({'date': self.current_date, 'current weights': current_weights})
        self.action_log.append({'date': self.current_date, 'target weights': target_weights})
        print('Current Financials:')
        cumulative_return = visualize_mvo_results(current_daily_returns, current_downside_returns, current_excess_returns)
        self.action_log.append({'date': self.current_date, 'current cumulative return': cumulative_return.iloc[-1]})

        max_distance = sum(abs(1 - value) for value in target_weights.values())
        distance_penalty = sum(abs(current_weights.get(key, 0) - value) for key, value in target_weights.items()) / max_distance if max_distance != 0 else 0
        self.action_log.append({'date': self.current_date, 'Distance penalty': distance_penalty})

        sortino_scale = 100000
        scaled_sortino_diff = (target_sortino_ratio - current_sortino_ratio) / sortino_scale
        print('Sortino ratio:', current_sortino_ratio)
        print('Scaled Sortino diff:', scaled_sortino_diff)
        self.action_log.append({'date': self.current_date, 'Scaled Sortino diff': scaled_sortino_diff})

        # Calculate the percentage change in total portfolio value
        if self.previous_total_portfolio_value is not None:
            portfolio_value_change_pct = (total_portfolio_value.iloc[-1] - self.previous_total_portfolio_value) / self.previous_total_portfolio_value
        else:
            portfolio_value_change_pct = 0  # No change if there is no previous value
        
        # Scale the portfolio value change percentage
        scaled_portfolio_value_change = portfolio_value_change_pct * self.value_change_factor
        print('port change pct', portfolio_value_change_pct)
        print('scaled port change', scaled_portfolio_value_change)
        
        # Update the previous total portfolio value
        self.previous_total_portfolio_value = total_portfolio_value.iloc[-1]
        
        # Incorporate the scaled percentage change in total portfolio value into the reward
        reward = scaled_sortino_diff - distance_penalty + scaled_portfolio_value_change
        reward_no_scale = current_sortino_ratio - distance_penalty + scaled_portfolio_value_change
        
        self.action_log.append({'date': self.current_date, 'Reward': reward})
        self.action_log.append({'date': self.current_date, 'Reward no scale': reward_no_scale})

        self.dai_ceiling = self.simulator.data[dai_ceilings]
        self.action_log.append({'date': self.current_date, 'dai ceilings': self.dai_ceiling})
        print('reward', reward)
        
        return reward_no_scale, current_weights