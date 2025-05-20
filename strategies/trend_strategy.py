"""Trend Following Strategy

Implements trend detection using multiple timeframes and confirmation signals.
Core idea: Strong trends with volume confirmation and clear support/resistance levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .base_strategy import BaseStrategy, Signal

class TrendStrategy(BaseStrategy):
    def __init__(self,
                lookback: int = 20,
                stop_loss_pct: float = 0.05,
                take_profit_pct: float = 0.15,
                ema_short: int = 9,
                ema_medium: int = 21,
                ema_long: int = 50,
                volume_ma: int = 20,
                rsi_period: int = 14,
                rsi_overbought: float = 70,
                rsi_oversold: float = 30):
        """Initialize trend strategy parameters"""
        super().__init__(lookback, stop_loss_pct, take_profit_pct)
        self.ema_short = ema_short
        self.ema_medium = ema_medium
        self.ema_long = ema_long
        self.volume_ma = volume_ma
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on trend analysis"""
        if not self.validate_data(data):
            return self._create_neutral_signal(data)
            
        # Calculate core indicators
        ema_signals = self._calculate_ema_signals(data)
        volume_signal = self._analyze_volume(data)
        momentum = self._calculate_momentum(data)
        risk_metrics = self.calculate_risk_metrics(data)
        
        # Combine signals
        signal_strength = self._combine_signals(ema_signals, volume_signal, momentum)
        
        # Generate final signal
        return self._create_signal(data, signal_strength, risk_metrics)
    
    def _calculate_ema_signals(self, data: pd.DataFrame) -> Dict:
        """Calculate EMA-based signals"""
        close = data['Close']
        ema_short = close.ewm(span=self.ema_short, adjust=False).mean()
        ema_medium = close.ewm(span=self.ema_medium, adjust=False).mean()
        ema_long = close.ewm(span=self.ema_long, adjust=False).mean()
        
        # Trend strength and direction
        trend_strength = (ema_short - ema_long) / ema_long
        is_uptrend = (ema_short > ema_medium) & (ema_medium > ema_long)
        is_downtrend = (ema_short < ema_medium) & (ema_medium < ema_long)
        
        return {
            'trend_strength': trend_strength.iloc[-1],
            'is_uptrend': is_uptrend.iloc[-1],
            'is_downtrend': is_downtrend.iloc[-1]
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze volume trends and breakouts"""
        volume = data['Volume']
        volume_ma = volume.rolling(window=self.volume_ma).mean()
        volume_ratio = volume / volume_ma
        
        return {
            'volume_trend': volume_ratio.iloc[-1],
            'volume_breakout': volume_ratio.iloc[-1] > 2.0,
            'volume_confirming': volume_ratio.iloc[-1] > 1.5
        }
    
    def _calculate_momentum(self, data: pd.DataFrame) -> Dict:
        """Calculate momentum indicators"""
        close = data['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'rsi': rsi.iloc[-1],
            'is_overbought': rsi.iloc[-1] > self.rsi_overbought,
            'is_oversold': rsi.iloc[-1] < self.rsi_oversold
        }
    
    def _combine_signals(self, ema_signals: Dict, volume_signals: Dict, momentum: Dict) -> float:
        """Combine different signals into a single strength indicator"""
        trend_confidence = 0.0
        
        # Strong trend conditions (50% weight)
        if ema_signals['is_uptrend']:
            trend_confidence += 0.5 * min(1.0, ema_signals['trend_strength'])
        elif ema_signals['is_downtrend']:
            trend_confidence += 0.5 * min(1.0, -ema_signals['trend_strength'])
            
        # Volume confirmation (30% weight)
        if volume_signals['volume_confirming']:
            trend_confidence += 0.3
            
        # Momentum alignment (20% weight)
        if (ema_signals['is_uptrend'] and momentum['is_oversold']) or \
           (ema_signals['is_downtrend'] and momentum['is_overbought']):
            trend_confidence += 0.2
            
        return min(1.0, trend_confidence)
    
    def _create_signal(self, data: pd.DataFrame, signal_strength: float, risk_metrics: Dict) -> Signal:
        """Create a trading signal based on analysis"""
        current_price = data['Close'].iloc[-1]
        action = 'hold'
        
        # Determine action based on signal strength and conditions
        if signal_strength > 0.7:  # Strong signal threshold
            ema_signals = self._calculate_ema_signals(data)
            action = 'buy' if ema_signals['is_uptrend'] else 'sell'
        
        position_size = self.calculate_position_size(data) if action != 'hold' else 0.0
        stop_loss = current_price * (1 - self.stop_loss_pct) if action == 'buy' else \
                    current_price * (1 + self.stop_loss_pct)
        take_profit = current_price * (1 + self.take_profit_pct) if action == 'buy' else \
                     current_price * (1 - self.take_profit_pct)
        
        return Signal(
            timestamp=data.index[-1],
            symbol=data.get('symbol', 'Unknown'),
            action=action,
            confidence=signal_strength,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            metrics={
                'risk_metrics': risk_metrics,
                'ema_signals': self._calculate_ema_signals(data),
                'volume_signals': self._analyze_volume(data),
                'momentum': self._calculate_momentum(data)
            }
        )
    
    def _create_neutral_signal(self, data: pd.DataFrame) -> Signal:
        """Create a neutral signal for invalid data"""
        return Signal(
            timestamp=data.index[-1] if not data.empty else pd.Timestamp.now(),
            symbol=data.get('symbol', 'Unknown'),
            action='hold',
            confidence=0.0,
            price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            position_size=0.0,
            metrics={}
        )
