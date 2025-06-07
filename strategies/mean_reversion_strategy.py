"""
Mean Reversion Strategy

This module implements mean reversion trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_strategy import BaseStrategy, Signal

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback_period: int = 20, z_threshold: float = 2.0):
        """
        Initialize mean reversion strategy
        
        Args:
            lookback_period: Period for calculating mean and standard deviation
            z_threshold: Z-score threshold for entry signals
        """
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.name = "Mean Reversion"
        
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        Generate mean reversion signal based on price deviations from mean
        
        Args:
            data: OHLCV price data
            
        Returns:
            Trading signal
        """
        if len(data) < self.lookback_period:
            return None
            
        # Calculate rolling mean and standard deviation
        rolling_mean = data['Close'].rolling(window=self.lookback_period).mean()
        rolling_std = data['Close'].rolling(window=self.lookback_period).std()
        
        # Calculate z-score (deviation from mean in standard deviation units)
        z_score = (data['Close'] - rolling_mean) / rolling_std
        
        # Current values
        current_price = data['Close'].iloc[-1]
        current_z = z_score.iloc[-1]
        
        # Default to hold
        signal = Signal(
            timestamp=data.index[-1],
            symbol=data.index.name if data.index.name else "UNKNOWN",
            action="hold",
            confidence=0.0,
            price=current_price,
            stop_loss=current_price * 0.95,
            take_profit=current_price * 1.05,
            position_size=0.0,
            metrics={"z_score": current_z}
        )
        
        # Oversold condition (potential buy)
        if current_z < -self.z_threshold:
            # Calculate metrics
            reversion_potential = abs(current_z) / self.z_threshold
            distance_from_mean = (rolling_mean.iloc[-1] - current_price) / current_price
            
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 - 0.5 * abs(distance_from_mean))
            take_profit = rolling_mean.iloc[-1] * 1.01  # Slightly above mean
            
            # Confidence based on z-score magnitude
            confidence = min(0.8, 0.4 + 0.1 * abs(current_z))
            
            signal = Signal(
                timestamp=data.index[-1],
                symbol=data.index.name if data.index.name else "UNKNOWN",
                action="buy",
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0.1 * confidence,
                metrics={
                    "z_score": current_z,
                    "reversion_potential": reversion_potential,
                    "distance_from_mean": distance_from_mean
                }
            )
            
        # Overbought condition (potential sell)
        elif current_z > self.z_threshold:
            # Calculate metrics
            reversion_potential = abs(current_z) / self.z_threshold
            distance_from_mean = (current_price - rolling_mean.iloc[-1]) / current_price
            
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 + 0.5 * abs(distance_from_mean))
            take_profit = rolling_mean.iloc[-1] * 0.99  # Slightly below mean
            
            # Confidence based on z-score magnitude
            confidence = min(0.8, 0.4 + 0.1 * abs(current_z))
            
            signal = Signal(
                timestamp=data.index[-1],
                symbol=data.index.name if data.index.name else "UNKNOWN",
                action="sell",
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0.1 * confidence,
                metrics={
                    "z_score": current_z,
                    "reversion_potential": reversion_potential,
                    "distance_from_mean": distance_from_mean
                }
            )
            
        return signal
