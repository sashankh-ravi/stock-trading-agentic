"""Momentum Strategy

Implements a comprehensive momentum strategy using:
1. Price momentum
2. Volume momentum
3. Relative strength
4. Cross-asset momentum
"""

import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy, Signal
import logging

logger = logging.getLogger(__name__)

class MomentumStrategy(BaseStrategy):
    def __init__(self,
                lookback: int = 20,
                momentum_period: int = 252,  # 1 year
                vol_period: int = 21,        # 1 month
                stop_loss_pct: float = 0.05,
                take_profit_pct: float = 0.15,
                min_momentum_percentile: float = 75,
                min_volume_momentum: float = 1.2):
        """Initialize momentum strategy
        
        Args:
            lookback: General lookback period
            momentum_period: Period for momentum calculations
            vol_period: Period for volatility calculations
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            min_momentum_percentile: Minimum momentum rank percentile
            min_volume_momentum: Minimum volume momentum ratio
        """
        super().__init__(lookback, stop_loss_pct, take_profit_pct)
        self.momentum_period = momentum_period
        self.vol_period = vol_period
        self.min_momentum_percentile = min_momentum_percentile
        self.min_volume_momentum = min_volume_momentum

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on momentum indicators"""
        try:
            if len(data) < max(self.lookback, self.momentum_period):
                return self._create_signal(data)

            # Calculate momentum metrics
            close = data['Close']
            volume = data['Volume']

            # 1. Price Momentum
            returns = close.pct_change()
            
            # Calculate different momentum periods
            momentum_dict = {}
            for period in [21, 63, 126, 252]:  # 1M, 3M, 6M, 1Y
                if len(close) > period:
                    momentum_dict[f'm{period}'] = (close.iloc[-1] / close.iloc[-period]) - 1

            # 2. Risk-adjusted Momentum (MAR ratio)
            vol = returns.rolling(self.vol_period).std() * np.sqrt(252)
            risk_adj_momentum = momentum_dict['m252'] / (vol.iloc[-1] if not np.isnan(vol.iloc[-1]) else 1)

            # 3. Volume Momentum
            volume_sma = volume.rolling(self.vol_period).mean()
            volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1]

            # 4. Trend Strength
            ema_short = close.ewm(span=20, adjust=False).mean()
            ema_long = close.ewm(span=50, adjust=False).mean()
            trend_strength = (ema_short.iloc[-1] / ema_long.iloc[-1]) - 1

            # Create base signal
            signal = self._create_signal(data)
            current_price = close.iloc[-1]

            # Calculate momentum percentile rank
            momentum_rank = pd.Series(momentum_dict['m252']).rank(pct=True).iloc[-1] * 100

            # Generate signal based on momentum conditions
            if momentum_rank > self.min_momentum_percentile and volume_ratio > self.min_volume_momentum:
                # Long signal conditions
                if all(v > 0 for v in momentum_dict.values()) and trend_strength > 0:
                    signal.action = 'buy'
                    # Calculate confidence based on momentum metrics
                    confidence_factors = [
                        min(1, momentum_rank / 100),
                        min(1, risk_adj_momentum / 2),
                        min(1, volume_ratio / self.min_volume_momentum),
                        min(1, trend_strength / 0.05)
                    ]
                    signal.confidence = np.mean(confidence_factors)
                
                # Short signal conditions (if implemented)
                elif all(v < 0 for v in momentum_dict.values()) and trend_strength < 0:
                    signal.action = 'sell'
                    confidence_factors = [
                        min(1, momentum_rank / 100),
                        min(1, abs(risk_adj_momentum) / 2),
                        min(1, volume_ratio / self.min_volume_momentum),
                        min(1, abs(trend_strength) / 0.05)
                    ]
                    signal.confidence = np.mean(confidence_factors)

                if signal.action != 'hold':
                    # Calculate ATR for position sizing
                    atr = self._calculate_atr(data)

                    # Set stop loss and take profit
                    if signal.action == 'buy':
                        signal.stop_loss = current_price * (1 - self.stop_loss_pct)
                        signal.take_profit = current_price * (1 + self.take_profit_pct)
                    else:
                        signal.stop_loss = current_price * (1 + self.stop_loss_pct)
                        signal.take_profit = current_price * (1 - self.take_profit_pct)

                    # Calculate position size based on ATR
                    signal.position_size = self._calculate_position_size(current_price, atr)
                    
                    # Add analysis metrics
                    signal.metrics.update({
                        'momentum_1m': momentum_dict['m21'],
                        'momentum_3m': momentum_dict['m63'],
                        'momentum_6m': momentum_dict['m126'],
                        'momentum_1y': momentum_dict['m252'],
                        'risk_adj_momentum': risk_adj_momentum,
                        'volume_ratio': volume_ratio,
                        'trend_strength': trend_strength,
                        'momentum_rank': momentum_rank
                    })

            return signal

        except Exception as e:
            logger.error(f"Error generating momentum signal: {e}")
            return self._create_signal(data)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]

    def _calculate_position_size(self, price: float, atr: float) -> float:
        """Calculate position size based on ATR"""
        try:
            # Use ATR-based position sizing
            risk_per_share = 2 * atr  # Risk 2 ATR per trade
            position_size = 1.0  # This will be adjusted by the backtester
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1.0
