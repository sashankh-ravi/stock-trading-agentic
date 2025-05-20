import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from data_collector import DataCollector
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Nifty500TradingEnv(gym.Env):
    def __init__(self, 
                 api_key: str,
                 initial_balance: float = 1000000,
                 window_size: int = 60,
                 monthly_return_target: float = 0.15,
                 stop_loss_pct: float = 0.05,
                 universe_size: int = 500):
        super().__init__()
        
        # Initialize parameters
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.monthly_return_target = monthly_return_target
        self.stop_loss_pct = stop_loss_pct
        self.universe_size = universe_size
        
        # Initialize data collector
        self.data_collector = DataCollector(api_key)
        
        # Action space: [stock_index (500), action_type (3), position_size (4)]
        # action_type: 0=Hold, 1=Buy, 2=Sell
        # position_size: 0=25%, 1=50%, 2=75%, 3=100% of allowed risk
        self.action_space = spaces.MultiDiscrete([500, 3, 4])
        
        # Observation space includes:
        # - Technical indicators for each stock (10 features)
        # - Fundamental scores (4 features)
        # - Sentiment scores (1 feature)
        # - Portfolio state (5 features)
        feature_count = 20  # Total features per stock
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(universe_size, feature_count),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset_env_state()
        
    def reset_env_state(self):
        """Reset environment state variables"""
        self.balance = self.initial_balance
        self.positions = {}  # {symbol: {'shares': int, 'entry_price': float}}
        self.data = None
        self.current_step = 0
        self.monthly_returns = []
        self.current_month = None
        self.monthly_high = self.initial_balance
        
    def fetch_data(self):
        """Fetch and prepare all necessary data"""
        logger.info("Fetching market data and analysis...")
        self.data = self.data_collector.prepare_training_data()
        
        # Initialize universe of stocks
        self.tradable_symbols = list(self.data.keys())[:self.universe_size]
        
    def calculate_monthly_return(self) -> float:
        """Calculate current month's return"""
        portfolio_value = self.get_portfolio_value()
        if not self.monthly_high:
            self.monthly_high = portfolio_value
        return (portfolio_value - self.monthly_high) / self.monthly_high
        
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        value = self.balance
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            value += position['shares'] * current_price
        return value
        
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        return self.data[symbol]['market_data'].iloc[self.current_step + self.window_size]['Close']
        
    def get_observation(self) -> np.ndarray:
        """Get current observation state"""
        obs = np.zeros((self.universe_size, 20))
        
        for idx, symbol in enumerate(self.tradable_symbols):
            if symbol in self.data:
                stock_data = self.data[symbol]
                market_data = stock_data['market_data']
                analysis = stock_data['analysis']
                
                # Get current window of data
                window = market_data.iloc[self.current_step:self.current_step + self.window_size]
                
                # Technical features (10)
                tech_features = [
                    window['Close'].mean(),
                    window['Close'].std(),
                    window['Volume'].mean(),
                    window['High'].max(),
                    window['Low'].min(),
                    window['Close'].pct_change().mean(),
                    window['Close'].pct_change().std(),
                    window['Volume'].pct_change().mean(),
                    (window['Close'] - window['Close'].mean()) / window['Close'].std(),
                    window['Volume'].rolling(5).mean().iloc[-1]
                ]
                
                # Fundamental features (4)
                fundamental_features = [
                    analysis['health_score'] / 100,
                    analysis['growth_score'] / 100,
                    analysis['risk_score'] / 100,
                    analysis['recommendation_score'] / 100
                ]
                
                # Sentiment feature (1)
                sentiment_feature = [analysis['news_sentiment']['sentiment_score']]
                
                # Position features (5)
                position = self.positions.get(symbol, {'shares': 0, 'entry_price': 0})
                current_price = self.get_current_price(symbol)
                position_features = [
                    1 if position['shares'] > 0 else 0,  # Position flag
                    position['shares'] * current_price / self.get_portfolio_value(),  # Position size
                    (current_price - position['entry_price']) / position['entry_price'] if position['shares'] > 0 else 0,  # Current P/L
                    1 if current_price <= position['entry_price'] * (1 - self.stop_loss_pct) else 0,  # Stop loss flag
                    1 if self.current_step > 0 and current_price >= position['entry_price'] * (1 + self.monthly_return_target) else 0  # Take profit flag
                ]
                
                # Combine all features
                obs[idx] = np.concatenate([
                    tech_features,
                    fundamental_features,
                    sentiment_feature,
                    position_features
                ])
                
        return obs
        
    def step(self, action):
        """Execute one trading step"""
        stock_idx, action_type, position_size = action
        symbol = self.tradable_symbols[stock_idx]
        current_price = self.get_current_price(symbol)
        
        # Initialize step variables
        info = {}
        done = False
        
        # Store initial portfolio value
        initial_portfolio_value = self.get_portfolio_value()
        
        # Check if new month started
        current_date = self.data[symbol]['market_data'].index[self.current_step + self.window_size]
        if self.current_month != current_date.month:
            if self.current_month is not None:
                monthly_return = self.calculate_monthly_return()
                self.monthly_returns.append(monthly_return)
                info['monthly_return'] = monthly_return
                
                # Check if we hit monthly target
                if monthly_return >= self.monthly_return_target:
                    info['monthly_target_achieved'] = True
                
            self.current_month = current_date.month
            self.monthly_high = self.get_portfolio_value()
        
        # Execute trading action
        if action_type == 1:  # Buy
            if symbol not in self.positions or self.positions[symbol]['shares'] == 0:
                # Calculate position size based on risk
                portfolio_value = self.get_portfolio_value()
                max_loss = portfolio_value * self.stop_loss_pct
                price_to_stop = current_price * (1 - self.stop_loss_pct)
                max_shares = (max_loss / (current_price - price_to_stop))
                
                # Adjust for selected position size
                position_sizes = [0.25, 0.50, 0.75, 1.0]
                shares_to_buy = int(max_shares * position_sizes[position_size])
                cost = shares_to_buy * current_price
                
                if cost <= self.balance:
                    self.positions[symbol] = {
                        'shares': shares_to_buy,
                        'entry_price': current_price
                    }
                    self.balance -= cost
                    info['action'] = 'buy'
                    
        elif action_type == 2:  # Sell
            if symbol in self.positions and self.positions[symbol]['shares'] > 0:
                shares = self.positions[symbol]['shares']
                self.balance += shares * current_price
                profit_pct = (current_price - self.positions[symbol]['entry_price']) / self.positions[symbol]['entry_price']
                self.positions[symbol] = {'shares': 0, 'entry_price': 0}
                info['action'] = 'sell'
                info['profit_pct'] = profit_pct
        
        # Check stop losses and take profits for all positions
        for sym, pos in list(self.positions.items()):
            if pos['shares'] > 0:
                curr_price = self.get_current_price(sym)
                entry_price = pos['entry_price']
                
                # Check stop loss
                if curr_price <= entry_price * (1 - self.stop_loss_pct):
                    self.balance += pos['shares'] * curr_price
                    profit_pct = (curr_price - entry_price) / entry_price
                    self.positions[sym] = {'shares': 0, 'entry_price': 0}
                    info[f'{sym}_stop_loss'] = profit_pct
                
                # Check take profit (monthly target)
                elif curr_price >= entry_price * (1 + self.monthly_return_target):
                    self.balance += pos['shares'] * curr_price
                    profit_pct = (curr_price - entry_price) / entry_price
                    self.positions[sym] = {'shares': 0, 'entry_price': 0}
                    info[f'{sym}_take_profit'] = profit_pct
        
        # Calculate reward
        final_portfolio_value = self.get_portfolio_value()
        step_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
        
        # Reward engineering
        reward = 0
        
        # Base reward component: Step return
        reward += step_return
        
        # Monthly target component
        monthly_return = self.calculate_monthly_return()
        if monthly_return >= self.monthly_return_target:
            reward += 1.0  # Bonus for hitting monthly target
        
        # Risk management component
        if any(f'{sym}_stop_loss' in info for sym in self.tradable_symbols):
            reward -= 0.5  # Penalty for hitting stop loss
        
        # Progress the environment
        self.current_step += 1
        
        # Check if episode is done
        min_steps = len(self.data[self.tradable_symbols[0]]['market_data']) - self.window_size
        done = self.current_step >= min_steps - 1
        
        # Add performance metrics to info
        info['portfolio_value'] = final_portfolio_value
        info['monthly_return'] = monthly_return
        info['total_return'] = (final_portfolio_value - self.initial_balance) / self.initial_balance
        
        return self.get_observation(), reward, done, False, info
        
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if self.data is None:
            self.fetch_data()
            
        self.reset_env_state()
        return self.get_observation(), {}