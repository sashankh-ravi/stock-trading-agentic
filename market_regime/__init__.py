"""
Market Regime Analysis Module - Core Implementation

This module provides the core components for market regime analysis, 
extracted from the original market_regime.py file.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any

@dataclass
class RegimeState:
    """Container for market regime states"""
    trend: str  # 'strong_up', 'moderate_up', 'weak_up', 'neutral', 'weak_down', 'moderate_down', 'strong_down'
    volatility: str  # 'low', 'normal', 'high'
    volume: str  # 'low', 'normal', 'high'
    momentum: Optional[str] = None  # 'positive', 'neutral', 'negative'
    
    def __str__(self) -> str:
        return f"Trend: {self.trend}, Volatility: {self.volatility}, Volume: {self.volume}, Momentum: {self.momentum}"
    
    def as_dict(self) -> Dict[str, str]:
        """Convert to dictionary for easy serialization"""
        return {
            'trend': self.trend,
            'volatility': self.volatility,
            'volume': self.volume,
            'momentum': self.momentum
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'RegimeState':
        """Create RegimeState from dictionary"""
        return cls(
            trend=data.get('trend', 'neutral'),
            volatility=data.get('volatility', 'normal'),
            volume=data.get('volume', 'normal'),
            momentum=data.get('momentum', None)
        )

class MarketRegimeAnalyzer:
    """
    Comprehensive market regime analysis
    
    This class identifies market regimes, transitions, and provides
    regime-specific parameter adjustments.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer with custom configuration
        
        Args:
            config: Optional configuration parameters
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
            
        # Track regime transitions
        self.previous_regime: Optional[RegimeState] = None
        self.current_regime: Optional[RegimeState] = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration parameters"""
        return {
            # Trend detection parameters
            'trend': {
                'sma_short': 20,
                'sma_medium': 50,
                'sma_long': 200,
                'strong_threshold': 0.10,  # 10% for strong trend
                'moderate_threshold': 0.05,  # 5% for moderate trend
                'weak_threshold': 0.02,  # 2% for weak trend
            },
            
            # Volatility detection parameters
            'volatility': {
                'atr_period': 14,
                'bb_period': 20,
                'high_vol_percentile': 80,
                'low_vol_percentile': 20,
            },
            
            # Volume detection parameters
            'volume': {
                'ma_period': 20,
                'high_vol_ratio': 1.5,
                'low_vol_ratio': 0.5,
            },
            
            # Momentum detection parameters
            'momentum': {
                'rsi_period': 14,
                'rsi_high': 70,
                'rsi_low': 30,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
            }
        }
        
    # Import other necessary methods from market_regime.py
