"""Base Strategy Module

This module defines the base strategy interface and common utilities for all strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

@dataclass
class Signal:
    """Trading signal with confidence and risk metrics"""
    timestamp: pd.Timestamp
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0 to 1
    price: float
    stop_loss: float
    take_profit: float
    position_size: float  # 0 to 1 (percentage of capital)
    metrics: Dict  # Strategy-specific metrics that led to this signal

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, 
                lookback: int = 20,
                stop_loss_pct: float = 0.05,
                take_profit_pct: float = 0.15):
        self.lookback = lookback
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.required_data = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal for the given data"""
        pass
    
    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate standard risk metrics"""
        returns = data['Close'].pct_change().dropna()
        if len(returns) < 2:
            return {
                'volatility': 0,
                'sharpe': 0,
                'max_drawdown': 0,
                'var_95': 0
            }
            
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        var_95 = np.percentile(returns, 5)
        
        return {
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'var_95': var_95
        }
    
    def calculate_position_size(self, 
                              data: pd.DataFrame,
                              risk_per_trade: float = 0.01  # 1% risk per trade
                              ) -> float:
        """Calculate position size based on volatility and risk parameters"""
        
        atr = self.calculate_atr(data)
        if atr == 0:
            return 0
            
        # Use ATR for dynamic position sizing
        price = data['Close'].iloc[-1]
        risk_amount = risk_per_trade * price
        position_size = risk_amount / atr
        
        # Normalize to 0-1 range and apply maximum position constraint
        return min(position_size / price, 0.25)  # Max 25% of capital per trade
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(data) < 2:
            return 0
            
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else 0
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that all required data is present and valid"""
        if data is None or data.empty:
            return False
            
        for col in self.required_data:
            if col not in data.columns:
                return False
                
        if data.isnull().any().any():
            return False
            
        return True
