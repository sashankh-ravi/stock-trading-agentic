"""
Volatility-based trading strategy for capturing price movements during high volatility periods.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

class VolatilityStrategy(BaseStrategy):
    def __init__(self, 
                 vix_threshold: float = 20.0,
                 atr_periods: int = 14,
                 atr_multiplier: float = 2.0,
                 rsi_period: int = 14,
                 bollinger_period: int = 20,
                 bollinger_std: float = 2.0):
        """
        Initialize the Volatility Strategy
        
        Args:
            vix_threshold: VIX level to consider high volatility
            atr_periods: Period for ATR calculation
            atr_multiplier: Multiplier to apply to ATR for signals
            rsi_period: Period for RSI calculation
            bollinger_period: Period for Bollinger Bands
            bollinger_std: Standard deviation for Bollinger Bands
        """
        super().__init__()
        self.vix_threshold = vix_threshold
        self.atr_periods = atr_periods
        self.atr_multiplier = atr_multiplier
        self.rsi_period = rsi_period
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        Generate trading signal based on volatility conditions
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            Signal object
        """
        try:
            if data.empty or len(data) < self.bollinger_period:
                return None
                
            # Get the latest data point
            current = data.iloc[-1]
            symbol = current.name.split(' ')[0] if hasattr(current.name, 'split') else 'UNKNOWN'
                
            # Calculate ATR (Average True Range)
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(self.atr_periods).mean().iloc[-1]
            
            # Calculate Bollinger Bands
            rolling_mean = close.rolling(self.bollinger_period).mean()
            rolling_std = close.rolling(self.bollinger_period).std()
            
            upper_band = rolling_mean + (rolling_std * self.bollinger_std)
            lower_band = rolling_mean - (rolling_std * self.bollinger_std)
            
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_close = close.iloc[-1]
            
            # Calculate volatility breakout levels
            breakout_up = current_close + (atr * self.atr_multiplier)
            breakout_down = current_close - (atr * self.atr_multiplier)
            
            # Calculate RSI
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
            loss = -delta.clip(upper=0).rolling(self.rsi_period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Volatility signal conditions
            bb_width = (current_upper - current_lower) / rolling_mean.iloc[-1]
            is_high_volatility = bb_width > 0.05  # 5% width threshold
            
            # Determine signal based on volatility conditions
            action = 'hold'
            confidence = 0.0
            price = current_close
            stop_loss = price * 0.95
            take_profit = price * 1.05
            position_size = 0.0
            
            if is_high_volatility:
                # Bollinger Band breakout with high volatility
                if current_close > current_upper and rsi < 70:
                    action = 'buy'
                    confidence = min(0.7, bb_width * 10)  # Scale by band width
                    stop_loss = breakout_down
                    take_profit = price + (price - stop_loss) * 2  # 2:1 reward/risk
                    position_size = 0.1
                elif current_close < current_lower and rsi > 30:
                    action = 'sell'
                    confidence = min(0.7, bb_width * 10)  # Scale by band width
                    stop_loss = breakout_up
                    take_profit = price - (stop_loss - price) * 2  # 2:1 reward/risk
                    position_size = 0.1
            
            # Add metrics for analysis
            metrics = {
                'atr': atr,
                'bb_width': bb_width,
                'rsi': rsi,
                'upper_band': current_upper,
                'lower_band': current_lower
            }
            
            # Create signal
            signal = Signal(
                timestamp=pd.Timestamp.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                metrics=metrics
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating volatility signal: {e}")
            return None
