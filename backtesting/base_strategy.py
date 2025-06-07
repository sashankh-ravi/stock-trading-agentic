"""Base strategy interface for backtesting"""

from dataclasses import dataclass
from typing import Dict, Union, List
from datetime import datetime
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class Signal:
    """Trading signal representation"""
    timestamp: datetime
    symbol: str
    direction: int  # 1 for buy, -1 for sell, 0 for hold
    size: float
    type: str  # 'ENTRY', 'EXIT', 'STOP_LOSS', 'TAKE_PROFIT'
    reason: str
    price: float
    params: Dict = None

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self):
        self.params = {}
        self.indicators = {}
        self.current_position = 0
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals for given data
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame mappings
        
        Returns:
            List of trading signals
        """
        raise NotImplementedError
    
    def set_parameters(self, params: Dict):
        """Set strategy parameters"""
        self.params = params
    
    def get_parameters(self) -> Dict:
        """Get current strategy parameters"""
        return self.params
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate technical indicators needed by the strategy
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            Dictionary of calculated indicators
        """
        return {}
    
    def position_size(self, data: pd.DataFrame, signal_type: str) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            data: OHLCV DataFrame
            signal_type: Type of signal
            
        Returns:
            Position size
        """
        portfolio_value = 1000000  # Example fixed value
        risk_per_trade = self.params.get('risk_per_trade', 0.02)
        atr = data['ATR'].iloc[-1] if 'ATR' in data else data['Close'].std()
        
        # Calculate position size based on ATR
        risk_amount = portfolio_value * risk_per_trade
        size = risk_amount / atr
        
        return round(size, 2)
    
    def update_position(self, size: float):
        """Update current position after trade"""
        self.current_position += size
    
    def get_position(self) -> float:
        """Get current position size"""
        return self.current_position
