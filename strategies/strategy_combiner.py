"""Strategy Combination Manager

This module implements dynamic strategy combination and risk management:
1. Strategy weighting based on performance
2. Dynamic risk allocation
3. Correlation-based portfolio adjustment
4. Regime-based strategy selection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
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
                rebalance_frequency: int = 20):
        """Initialize strategy combiner
        
        Args:
            lookback: Days for performance evaluation
            min_weight: Minimum strategy weight
            max_weight: Maximum strategy weight
            rebalance_frequency: Days between rebalancing
        """
        self.lookback = lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.rebalance_frequency = rebalance_frequency
        self.strategy_performance = {}
        self.last_rebalance = None
        
    def combine_signals(self,
                       signals: List[List[Signal]],
                       weights: List[float] = None) -> List[Signal]:
        """Combine signals from multiple strategies with dynamic weighting"""
        try:
            if not signals:
                return []
                
            # Use equal weights if none provided
            if weights is None:
                weights = [1.0 / len(signals)] * len(signals)
                
            combined = {}
            
            # Combine signals by symbol
            for strategy_signals, weight in zip(signals, weights):
                for signal in strategy_signals:
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
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return []
            
    def update_strategy_weights(self,
                              strategies: List[BaseStrategy],
                              performance_metrics: List[Dict]) -> List[float]:
        """Update strategy weights based on recent performance"""
        try:
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
                
            # Calculate base weights from scores
            weights = scores / np.sum(scores)
            
            # Apply weight constraints
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / np.sum(weights)  # Renormalize
            
            return weights.tolist()
            
        except Exception as e:
            logger.error(f"Error updating strategy weights: {e}")
            return [1.0 / len(strategies)] * len(strategies) if strategies else []
            
    def _calculate_strategy_score(self, metrics: Dict) -> float:
        """Calculate overall strategy score from performance metrics"""
        try:
            # Define metric weights
            weights = {
                'sharpe_ratio': 0.3,
                'sortino_ratio': 0.2,
                'win_rate': 0.15,
                'profit_factor': 0.15,
                'max_drawdown': 0.2
            }
            
            score = 0.0
            
            # Sharpe Ratio contribution
            if metrics.get('sharpe_ratio', 0) > 0:
                score += weights['sharpe_ratio'] * min(3, metrics['sharpe_ratio']) / 3
                
            # Sortino Ratio contribution
            if metrics.get('sortino_ratio', 0) > 0:
                score += weights['sortino_ratio'] * min(4, metrics['sortino_ratio']) / 4
                
            # Win Rate contribution (0.5-0.7 is typical range)
            if 'win_rate' in metrics:
                win_rate_score = (metrics['win_rate'] - 0.5) / 0.2  # Normalize
                score += weights['win_rate'] * min(1, max(0, win_rate_score))
                
            # Profit Factor contribution (1-3 is typical range)
            if metrics.get('profit_factor', 0) > 1:
                pf_score = (metrics['profit_factor'] - 1) / 2  # Normalize
                score += weights['profit_factor'] * min(1, pf_score)
                
            # Max Drawdown contribution (penalize large drawdowns)
            if 'max_drawdown' in metrics:
                dd_score = 1 - min(1, abs(metrics['max_drawdown']) / 0.2)  # Normalize
                score += weights['max_drawdown'] * dd_score
                
            return max(0, score)
            
        except Exception as e:
            logger.error(f"Error calculating strategy score: {e}")
            return 0.0
            
    def adjust_for_regime(self,
                         weights: List[float],
                         regime_metrics: Dict) -> List[float]:
        """Adjust strategy weights based on market regime"""
        try:
            if not weights:
                return weights
                
            current_regime = regime_metrics.get('current_regime', 'neutral')
            current_vol = regime_metrics.get('current_vol_regime', 'normal_vol')
            
            # Define regime-based adjustments
            regime_adjustments = {
                'bull': {
                    'trend': 1.2,
                    'mean_reversion': 0.8,
                    'volatility': 1.0
                },
                'bear': {
                    'trend': 0.8,
                    'mean_reversion': 1.0,
                    'volatility': 1.2
                },
                'neutral': {
                    'trend': 1.0,
                    'mean_reversion': 1.2,
                    'volatility': 0.8
                }
            }
            
            vol_adjustments = {
                'high_vol': {
                    'trend': 0.8,
                    'mean_reversion': 0.7,
                    'volatility': 1.5
                },
                'normal_vol': {
                    'trend': 1.0,
                    'mean_reversion': 1.0,
                    'volatility': 1.0
                },
                'low_vol': {
                    'trend': 1.2,
                    'mean_reversion': 1.3,
                    'volatility': 0.7
                }
            }
            
            # Apply adjustments
            adjusted_weights = np.array(weights)
            for i, w in enumerate(weights):
                if isinstance(strategies[i], TrendStrategy):
                    factor = regime_adjustments[current_regime]['trend'] * \
                            vol_adjustments[current_vol]['trend']
                elif isinstance(strategies[i], MeanReversionStrategy):
                    factor = regime_adjustments[current_regime]['mean_reversion'] * \
                            vol_adjustments[current_vol]['mean_reversion']
                elif isinstance(strategies[i], VolatilityStrategy):
                    factor = regime_adjustments[current_regime]['volatility'] * \
                            vol_adjustments[current_vol]['volatility']
                else:
                    factor = 1.0
                    
                adjusted_weights[i] *= factor
                
            # Normalize weights
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
            
            # Apply min/max constraints
            adjusted_weights = np.clip(adjusted_weights, self.min_weight, self.max_weight)
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
            
            return adjusted_weights.tolist()
            
        except Exception as e:
            logger.error(f"Error adjusting weights for regime: {e}")
            return weights
