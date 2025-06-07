"""Strategy Manager

This module implements dynamic strategy combination and risk management:
1. Strategy weighting based on performance
2. Dynamic risk allocation
3. Correlation-based portfolio adjustment
4. Regime-based strategy selection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from .base_strategy import BaseStrategy, Signal
from .trend_strategy import TrendStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .volatility_strategy import VolatilityStrategy
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class StrategyCombiner:
    def __init__(self,
                lookback: int = 60,
                min_weight: float = 0.1,
                max_weight: float = 0.4,
                rebalance_frequency: int = 20,
                min_confidence: float = 0.5):
        """Initialize strategy combiner
        
        Args:
            lookback: Days for performance evaluation
            min_weight: Minimum strategy weight
            max_weight: Maximum strategy weight
            rebalance_frequency: Days between rebalancing
            min_confidence: Minimum confidence threshold for signals
        """
        self.lookback = lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.rebalance_frequency = rebalance_frequency
        self.min_confidence = min_confidence
        self.strategy_performance = {}
        self.last_rebalance = None
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> List[BaseStrategy]:
        """Initialize available trading strategies"""
        strategies = []
        
        # Add trend strategy
        trend = TrendStrategy()
        strategies.append(trend)
        
        # Add mean reversion strategy
        mean_rev = MeanReversionStrategy()
        strategies.append(mean_rev)
        
        # Add volatility strategy
        vol = VolatilityStrategy()
        strategies.append(vol)
        
        return strategies
    
    def analyze(self, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze price data and generate trading signals
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            Dictionary with trading signal information or None if no signal
        """
        if data.empty or len(data) < 20:
            return None
        
        all_signals = []
        
        # Get signals from each strategy
        for strategy in self.strategies:
            signal = strategy.generate_signal(data)
            if signal:
                all_signals.append(signal)
        
        if not all_signals:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'stop_loss': data['Close'].iloc[-1] * 0.95,
                'take_profit': data['Close'].iloc[-1] * 1.05,
                'signals': []
            }
        
        # Combine signals from all strategies
        combined_signals = self.combine_signals(all_signals)
        
        if not combined_signals:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'stop_loss': data['Close'].iloc[-1] * 0.95,
                'take_profit': data['Close'].iloc[-1] * 1.05,
                'signals': []
            }
        
        # Use the first signal (should only be one per symbol)
        signal = combined_signals[0]
        
        # Only consider signals with confidence above threshold
        if signal.confidence < self.min_confidence:
            action = 'hold'
        else:
            action = signal.action
        
        # Format signals from each strategy for output
        strategy_signals = []
        for i, strategy in enumerate(self.strategies):
            if i < len(all_signals) and all_signals[i]:
                strategy_signal = all_signals[i][0]
                strategy_signals.append({
                    'strategy': strategy.__class__.__name__,
                    'action': strategy_signal.action,
                    'confidence': strategy_signal.confidence
                })
        
        return {
            'action': action,
            'confidence': signal.confidence,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'signals': strategy_signals
        }
    
    def combine_signals(self,
                       signals: List[Signal],
                       weights: List[float] = None) -> List[Signal]:
        """Combine signals from multiple strategies with dynamic weighting"""
        if not signals:
            return []
            
        # Use equal weights if none provided
        if weights is None:
            weights = [1.0 / len(signals)] * len(signals)
            
        combined = {}
        
        # Combine signals by symbol
        for i, signal in enumerate(signals):
            weight = weights[i]
            symbol = signal.symbol
            
            if symbol not in combined:
                combined[symbol] = signal
                combined[symbol].confidence *= weight
            else:
                # Weight signals by confidence
                total_conf = combined[symbol].confidence + signal.confidence * weight
                if total_conf > 0:
                    # Weighted average of signal attributes
                    for attr in ['price', 'stop_loss', 'take_profit', 'position_size']:
                        current = getattr(combined[symbol], attr)
                        new = getattr(signal, attr)
                        weighted = (current * combined[symbol].confidence + 
                                  new * signal.confidence * weight) / total_conf
                        setattr(combined[symbol], attr, weighted)
                            
                        # Merge metrics
                        combined[symbol].metrics.update(signal.metrics)
                        
                    combined[symbol].confidence = total_conf
                    
        return list(combined.values())
    
    def update_strategy_weights(self,
                              strategies: List[BaseStrategy],
                              performance_metrics: List[Dict]) -> List[float]:
        """Update strategy weights based on recent performance"""
        if not strategies or not performance_metrics:
            return [1.0 / len(strategies)] * len(strategies) if strategies else []
            
        # Calculate strategy scores
        scores = []
        for metrics in performance_metrics:
            score = self._calculate_strategy_score(metrics)
            scores.append(score)
            
        # Convert to numpy array
        scores = np.array(scores)
        
        # Handle all zero scores
        if np.sum(scores) == 0:
            return [1.0 / len(strategies)] * len(strategies)
            
        # Calculate base weights
        base_weights = scores / np.sum(scores)
        
        # Apply min/max constraints
        weights = self._constrain_weights(base_weights)
        
        return weights.tolist()
    
    def _calculate_strategy_score(self, metrics: Dict) -> float:
        """Calculate a performance score for a strategy based on metrics"""
        if not metrics:
            return 0.0
            
        # Components of the score
        profit_factor = metrics.get('profit_factor', 1.0)
        win_rate = metrics.get('win_rate', 0.5)
        sharpe = metrics.get('sharpe_ratio', 0.0)
        
        # Adjust components to avoid extreme values
        profit_factor = min(profit_factor, 5.0)
        sharpe = max(min(sharpe, 3.0), -3.0)
        
        # Calculate score
        score = (profit_factor * 0.4) + (win_rate * 0.3) + ((sharpe + 3) / 6 * 0.3)
        
        return max(score, 0.1)  # Ensure minimum score
    
    def _constrain_weights(self, weights: np.ndarray) -> np.ndarray:
        """Apply min/max constraints to strategy weights"""
        n = len(weights)
        
        # Apply minimum weight constraint
        below_min = weights < self.min_weight
        min_deficit = np.sum(self.min_weight * below_min - weights * below_min)
        
        if min_deficit > 0:
            # Distribute deficit among other strategies
            above_min = ~below_min
            if np.any(above_min):
                adjustment = min_deficit / np.sum(above_min)
                weights[above_min] -= adjustment
                weights[below_min] = self.min_weight
        
        # Apply maximum weight constraint
        above_max = weights > self.max_weight
        max_excess = np.sum(weights * above_max - self.max_weight * above_max)
        
        if max_excess > 0:
            # Distribute excess among other strategies
            below_max = ~above_max
            if np.any(below_max):
                adjustment = max_excess / np.sum(below_max)
                weights[below_max] += adjustment
                weights[above_max] = self.max_weight
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime based on price action
        
        Returns:
            'trend', 'mean_reversion', 'volatility', or 'neutral'
        """
        if len(data) < 60:
            return 'neutral'
            
        # Calculate metrics
        returns = data['Close'].pct_change().dropna()
        
        # Trend strength indicators
        adr = data['High'].iloc[-20:] / data['Low'].iloc[-20:] - 1
        adx = self._calculate_adx(data)
        
        # Volatility indicators
        volatility = returns.std() * np.sqrt(252)
        recent_volatility = returns.iloc[-20:].std() * np.sqrt(252)
        volatility_change = recent_volatility / volatility if volatility > 0 else 1.0
        
        # Mean reversion indicators
        # Calculate z-score of price relative to moving average
        ma50 = data['Close'].rolling(50).mean().iloc[-1]
        price = data['Close'].iloc[-1]
        z_score = (price - ma50) / (data['Close'].rolling(50).std().iloc[-1])
        
        # Determine regime
        if adx > 25 and np.mean(adr) > 0.015:
            return 'trend'
        elif abs(z_score) > 2.0 and volatility_change < 0.8:
            return 'mean_reversion'
        elif volatility_change > 1.5 or recent_volatility > 0.4:
            return 'volatility'
        else:
            return 'neutral'
            
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        if len(data) <= period:
            return 0.0
            
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff(-1)
        
        # Condition for +DM: +DM = max(0, High(t) - High(t-1))
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        
        # Condition for -DM: -DM = max(0, Low(t-1) - Low(t))
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate smoothed values
        smoothed_plus_dm = plus_dm.rolling(period).mean()
        smoothed_minus_dm = minus_dm.rolling(period).mean()
        smoothed_tr = tr.rolling(period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
