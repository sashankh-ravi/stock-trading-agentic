"""
Market Regime Analysis Module

This module implements comprehensive market regime detection and analysis including:
1. Regime detection (trend, volatility, volume)
2. Regime transitions
3. Regime-specific parameter adjustments
4. Regime performance metrics

IMPORTANT: This file is being migrated to a modular package structure in the 'market_regime/'
directory. For new development, please use the package components instead of this file.
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
    momentum: str  # 'strong', 'weak', 'neutral'
    confidence: float  # Confidence score 0-1 for regime classification

class MarketRegimeAnalyzer:
    """Market Regime Analysis System"""
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        Initialize the regime analyzer with configurable lookback periods
        
        Args:
            lookback_periods: Dictionary of lookback periods for different calculations
                            Default values are provided if not specified
        """
        self.lookback_periods = lookback_periods or {
            'trend_short': 20,
            'trend_medium': 50,
            'trend_long': 200,
            'volatility': 21,
            'volume': 20,
            'momentum': 14,  # For RSI and similar momentum indicators
            'regime_threshold': 252  # For regime threshold calculations
        }
        
        # Regime classification thresholds
        self.thresholds = {
            'trend': {
                'strong': 0.05,  # 5% trend threshold
                'moderate': 0.03,
                'weak': 0.01
            },
            'volatility': {
                'high': 0.75,  # 75th percentile
                'low': 0.25    # 25%
            },
            'volume': {
                'high': 0.75,
                'low': 0.25
            },
            'momentum': {
                'strong': 70,  # RSI, MFI > 70
                'weak': 30     # RSI, MFI < 30
            }
        }
        
        # Previous regime state to track transitions
        self.previous_regime = None

    def analyze(self, data: pd.DataFrame) -> RegimeState:
        """
        Analyze the market data and determine the current regime state
        
        Args:
            data: Market data containing price, volume, and indicator information
        
        Returns:
            RegimeState: The detected regime state with trend, volatility, volume, and momentum conditions
        """
        trend = self._analyze_trend(data)
        volatility = self._analyze_volatility(data)
        volume = self._analyze_volume(data)
        momentum = self._analyze_momentum(data)
        
        # Confidence calculation (example: could be based on indicator alignments)
        confidence = np.random.rand()  # Placeholder for actual confidence calculation
        
        return RegimeState(trend, volatility, volume, momentum, confidence)

    def _analyze_trend(self, data: pd.DataFrame) -> str:
        """Analyze trend conditions using moving averages"""
        # Get moving averages
        ma_short = data['Close'].rolling(window=self.lookback_periods['trend_short']).mean()
        ma_medium = data['Close'].rolling(window=self.lookback_periods['trend_medium']).mean()
        ma_long = data['Close'].rolling(window=self.lookback_periods['trend_long']).mean()
        
        # Trend determination logic
        if ma_short.iloc[-1] > ma_medium.iloc[-1] > ma_long.iloc[-1]:
            return 'strong_up'
        elif ma_short.iloc[-1] > ma_medium.iloc[-1]:
            return 'moderate_up'
        elif ma_short.iloc[-1] < ma_medium.iloc[-1] < ma_long.iloc[-1]:
            return 'strong_down'
        elif ma_short.iloc[-1] < ma_medium.iloc[-1]:
            return 'moderate_down'
        else:
            return 'neutral'

    def _analyze_volatility(self, data: pd.DataFrame) -> str:
        """Analyze volatility conditions using rolling standard deviation"""
        volatility = data['Close'].pct_change().rolling(window=self.lookback_periods['volatility']).std()
        
        if volatility.iloc[-1] > np.percentile(volatility, 75):
            return 'high'
        elif volatility.iloc[-1] < np.percentile(volatility, 25):
            return 'low'
        else:
            return 'normal'

    def _analyze_volume(self, data: pd.DataFrame) -> str:
        """Analyze volume conditions using rolling mean and standard deviation"""
        volume_mean = data['Volume'].rolling(window=self.lookback_periods['volume']).mean()
        volume_std = data['Volume'].rolling(window=self.lookback_periods['volume']).std()
        volume_threshold_high = volume_mean + volume_std
        volume_threshold_low = volume_mean - volume_std
        
        if data['Volume'].iloc[-1] > volume_threshold_high.iloc[-1]:
            return 'high'
        elif data['Volume'].iloc[-1] < volume_threshold_low.iloc[-1]:
            return 'low'
        else:
            return 'normal'
    
    def _analyze_momentum(self, data: pd.DataFrame) -> str:
        """Analyze momentum conditions using RSI, MFI and MACD"""
        # Get momentum indicators
        rsi = data.get('RSI', pd.Series([50] * len(data)))  # Default to neutral if not available
        mfi = data.get('MFI', pd.Series([50] * len(data)))
        macd_hist = data.get('MACD_Hist', pd.Series([0] * len(data)))
        
        if (rsi.iloc[-1] > self.thresholds['momentum']['strong'] and 
            mfi.iloc[-1] > self.thresholds['momentum']['strong']) or macd_hist.iloc[-1] > 0:
            return 'strong'
        elif (rsi.iloc[-1] < self.thresholds['momentum']['weak'] and 
              mfi.iloc[-1] < self.thresholds['momentum']['weak']) or macd_hist.iloc[-1] < 0:
            return 'weak'
        else:
            return 'neutral'
    
    def detect_regime(self, data: pd.DataFrame) -> RegimeState:
        """
        Detect current market regime using multiple indicators
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            RegimeState object with current regime classification
        """
        if len(data) < max(self.lookback_periods.values()):
            return RegimeState(
                trend='neutral',
                volatility='normal',
                volume='normal',
                momentum='neutral',
                confidence=0.0
            )

        # Detect trend regime
        trend_regime = self._analyze_trend(data)
        
        # Detect volatility regime
        volatility_regime = self._analyze_volatility(data)
        
        # Detect volume regime
        volume_regime = self._analyze_volume(data)
        
        # Detect momentum regime
        momentum_regime = self._analyze_momentum(data)
        
        # Calculate confidence score
        confidence = self._calculate_regime_confidence(data)
        
        return RegimeState(
            trend=trend_regime,
            volatility=volatility_regime,
            volume=volume_regime,
            momentum=momentum_regime,
            confidence=confidence
        )
    
    def _calculate_regime_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence score for regime classification"""
        # Trend confidence
        ema_short = data['Close'].ewm(span=self.lookback_periods['trend_short']).mean()
        ema_slope = ema_short.diff(self.lookback_periods['trend_short']) / ema_short
        trend_conf = min(abs(ema_slope.iloc[-1]) / self.thresholds['trend']['strong'], 1.0)
        
        # Volatility confidence
        returns = data['Close'].pct_change()
        volatility = returns.rolling(self.lookback_periods['volatility']).std() * np.sqrt(252)
        vol_conf = min(abs(volatility.iloc[-1] - volatility.mean()) / volatility.std(), 1.0) if not volatility.empty else 0.0
        
        # Volume confidence
        volume_ma = data['Volume'].rolling(window=self.lookback_periods['volume']).mean()
        relative_volume = data['Volume'].iloc[-1] / volume_ma.iloc[-1]
        volume_conf = min(abs(relative_volume - 1.0), 1.0)
        
        # Momentum confidence
        rsi = data.get('RSI', pd.Series([50] * len(data)))
        momentum_conf = min(abs(rsi.iloc[-1] - 50) / 50, 1.0)
        
        # Combine confidence scores
        return (trend_conf + vol_conf + volume_conf + momentum_conf) / 4.0
        
    def detect_regime_transition(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Detect transitions between market regimes
        
        Args:
            data: DataFrame with OHLCV data (typically covering multiple periods)
            
        Returns:
            Tuple of (is_transition, transition_type)
        """
        if len(data) < max(self.lookback_periods.values()) * 2:
            return False, "insufficient_data"
            
        # Get regime at beginning and end of period
        start_regime = self.detect_regime(data.iloc[:len(data)//2])
        end_regime = self.detect_regime(data.iloc[len(data)//2:])
        
        # Check for major transitions
        transitions = []
        
        # Trend transitions
        if 'up' in start_regime.trend and 'down' in end_regime.trend:
            transitions.append('trend_reversal_bearish')
        elif 'down' in start_regime.trend and 'up' in end_regime.trend:
            transitions.append('trend_reversal_bullish')
        elif 'neutral' in start_regime.trend and ('up' in end_regime.trend or 'down' in end_regime.trend):
            transitions.append('trend_emergence')
            
        # Volatility transitions
        if start_regime.volatility == 'low' and end_regime.volatility == 'high':
            transitions.append('volatility_expansion')
        elif start_regime.volatility == 'high' and end_regime.volatility == 'low':
            transitions.append('volatility_contraction')
            
        # Volume transitions
        if start_regime.volume == 'low' and end_regime.volume == 'high':
            transitions.append('volume_surge')
        elif start_regime.volume == 'high' and end_regime.volume == 'low':
            transitions.append('volume_decline')
            
        # Momentum transitions
        if start_regime.momentum == 'neutral' and end_regime.momentum == 'strong':
            transitions.append('momentum_acceleration')
        elif start_regime.momentum == 'strong' and end_regime.momentum == 'weak':
            transitions.append('momentum_exhaustion')
        
        # Return most significant transition
        if transitions:
            return True, transitions[0]
        return False, "no_transition"
        
    def get_regime_parameters(self, regime: RegimeState) -> Dict[str, Dict[str, float]]:
        """
        Get optimal trading parameters for the current regime
        
        Args:
            regime: Current market regime state
            
        Returns:
            Dictionary of parameter adjustments for the trading system
        """
        params = {
            'position_sizing': {},
            'stop_loss': {},
            'take_profit': {},
            'entry_threshold': {},
            'strategy_weights': {}
        }
        
        # Position sizing adjustments based on volatility
        if regime.volatility == 'high':
            params['position_sizing']['size'] = 0.5  # Reduce position size in high volatility
            params['position_sizing']['max_allocation'] = 0.1  # Lower max allocation per position
        elif regime.volatility == 'low':
            params['position_sizing']['size'] = 1.0  # Full position size in low volatility
            params['position_sizing']['max_allocation'] = 0.2  # Higher max allocation
        else:
            params['position_sizing']['size'] = 0.75  # Standard position size
            params['position_sizing']['max_allocation'] = 0.15  # Standard allocation
        
        # Stop loss adjustments based on volatility and trend
        if regime.volatility == 'high':
            params['stop_loss']['atr_multiplier'] = 3.0  # Wider stops in high volatility
        else:
            params['stop_loss']['atr_multiplier'] = 2.0  # Standard stops
            
        # Take profit adjustments based on trend
        if 'strong' in regime.trend:
            params['take_profit']['target'] = 3.0  # Higher targets in strong trends
            params['take_profit']['trailing'] = True  # Use trailing stops in trends
        else:
            params['take_profit']['target'] = 2.0  # Standard targets
            params['take_profit']['trailing'] = False  # Fixed stops in non-trending markets
            
        # Entry threshold adjustments based on momentum
        if regime.momentum == 'strong':
            params['entry_threshold']['signal_strength'] = 0.7  # Higher threshold in strong momentum
        else:
            params['entry_threshold']['signal_strength'] = 0.5  # Standard threshold
            
        # Strategy weight adjustments based on regime
        if 'up' in regime.trend and regime.volatility != 'high':
            # Favor trend following in uptrends with moderate/low volatility
            params['strategy_weights']['trend_following'] = 0.7
            params['strategy_weights']['mean_reversion'] = 0.2
            params['strategy_weights']['breakout'] = 0.1
        elif 'down' in regime.trend and regime.volatility != 'high':
            # Balanced approach in downtrends with moderate/low volatility
            params['strategy_weights']['trend_following'] = 0.4
            params['strategy_weights']['mean_reversion'] = 0.4
            params['strategy_weights']['breakout'] = 0.2
        elif regime.volatility == 'high':
            # Favor mean reversion in high volatility
            params['strategy_weights']['trend_following'] = 0.2
            params['strategy_weights']['mean_reversion'] = 0.6
            params['strategy_weights']['breakout'] = 0.2
        else:
            # Balanced approach in neutral regimes
            params['strategy_weights']['trend_following'] = 0.33
            params['strategy_weights']['mean_reversion'] = 0.33
            params['strategy_weights']['breakout'] = 0.34
            
        return params
        
    def calculate_regime_metrics(self, data: pd.DataFrame, returns: pd.Series = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each market regime
        
        Args:
            data: OHLCV data with technical indicators
            returns: Optional returns series (will be calculated if not provided)
            
        Returns:
            Dictionary of performance metrics for each regime
        """
        if returns is None:
            returns = data['Close'].pct_change()
            
        # Initialize metrics container
        metrics = {
            'bull_market': {'return': 0, 'volatility': 0, 'sharpe': 0, 'win_rate': 0, 'max_drawdown': 0},
            'bear_market': {'return': 0, 'volatility': 0, 'sharpe': 0, 'win_rate': 0, 'max_drawdown': 0},
            'sideways_market': {'return': 0, 'volatility': 0, 'sharpe': 0, 'win_rate': 0, 'max_drawdown': 0},
            'high_volatility': {'return': 0, 'volatility': 0, 'sharpe': 0, 'win_rate': 0, 'max_drawdown': 0},
            'low_volatility': {'return': 0, 'volatility': 0, 'sharpe': 0, 'win_rate': 0, 'max_drawdown': 0}
        }
        
        # Identify regime periods
        regimes = []
        windows = []
        
        # Use rolling windows to classify regimes
        window_size = min(60, len(data) // 5)  # Approximately one quarter or 60 days, whichever is smaller
        
        for i in range(0, len(data) - window_size, window_size // 2):  # 50% overlap
            window_data = data.iloc[i:i+window_size]
            regime = self.detect_regime(window_data)
            
            # Classify into simplified regimes
            if 'up' in regime.trend and regime.volatility != 'high':
                regime_type = 'bull_market'
            elif 'down' in regime.trend and regime.volatility != 'high':
                regime_type = 'bear_market'
            elif regime.trend == 'neutral':
                regime_type = 'sideways_market'
            elif regime.volatility == 'high':
                regime_type = 'high_volatility'
            elif regime.volatility == 'low':
                regime_type = 'low_volatility'
            else:
                continue  # Skip ambiguous regimes
                
            regimes.append(regime_type)
            windows.append((i, i+window_size))
        
        # Calculate metrics for each regime
        for regime_type in metrics.keys():
            # Combine all periods for this regime
            regime_indices = [idx for idx, r in enumerate(regimes) if r == regime_type]
            if not regime_indices:
                continue
                
            regime_windows = [windows[idx] for idx in regime_indices]
            regime_returns = []
            
            for start, end in regime_windows:
                if end > len(returns):
                    end = len(returns)
                regime_returns.extend(returns.iloc[start:end].dropna().tolist())
            
            if not regime_returns:
                continue
                
            # Calculate metrics
            regime_returns = np.array(regime_returns)
            annualized_return = np.mean(regime_returns) * 252
            annualized_vol = np.std(regime_returns) * np.sqrt(252)
            win_rate = np.sum(regime_returns > 0) / len(regime_returns)
            
            # Calculate maximum drawdown
            cum_returns = np.cumprod(1 + regime_returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns / running_max) - 1
            max_drawdown = np.min(drawdowns)
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
            sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            # Store metrics
            metrics[regime_type]['return'] = annualized_return
            metrics[regime_type]['volatility'] = annualized_vol
            metrics[regime_type]['sharpe'] = sharpe
            metrics[regime_type]['win_rate'] = win_rate
            metrics[regime_type]['max_drawdown'] = max_drawdown
            
        return metrics
    
    def analyze_volatility_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volatility regime including volatility term structure, skew, and realized vs implied volatility
        
        Args:
            data: DataFrame with OHLCV data (must have Close prices)
            
        Returns:
            Dictionary containing volatility regime metrics
        """
        # Helper function for percentile rank calculation
        def percentile_rank(value, series):
            """Calculate the percentile rank of a value within a series"""
            return (series < value).mean() * 100
            
        # Helper function for regime classification
        def classify_volatility_regime(vol_metrics: Dict[str, float]) -> Dict[str, Any]:
            """
            Classify the volatility regime and assign confidence scores
            
            Args:
                vol_metrics: Dictionary of volatility metrics
                
            Returns:
                Dictionary with regime classification and confidence scores
            """
            result = {
                'regime': 'normal',
                'confidence': 0.5,
                'regime_probability': {
                    'low_vol': 0.0,
                    'normal_vol': 0.0, 
                    'high_vol': 0.0,
                    'volatility_expansion': 0.0,
                    'volatility_contraction': 0.0
                }
            }
            
            # Factors for regime classification
            factors = []
            
            # Factor 1: Current volatility level vs long-term
            if 'vol_20d' in vol_metrics and 'vol_252d' in vol_metrics:
                vol_ratio = vol_metrics['vol_20d'] / vol_metrics['vol_252d']
                if vol_ratio < 0.7:
                    factors.append(('low_vol', min(1.0, 2.0 * (0.7 - vol_ratio))))
                elif vol_ratio > 1.3:
                    factors.append(('high_vol', min(1.0, (vol_ratio - 1.3) * 1.5)))
                else:
                    factors.append(('normal_vol', 1.0 - min(1.0, abs(vol_ratio - 1.0) * 2)))
            
            # Factor 2: Volatility percentile
            if 'vol_percentile_1y' in vol_metrics:
                percentile = vol_metrics['vol_percentile_1y']
                if percentile < 20:
                    factors.append(('low_vol', (20 - percentile) / 20))
                elif percentile > 80:
                    factors.append(('high_vol', (percentile - 80) / 20))
                else:
                    factors.append(('normal_vol', 1.0 - abs(percentile - 50) / 30))
            
            # Factor 3: Term structure
            if 'term_structure_state_short' in vol_metrics:
                if vol_metrics['term_structure_state_short'] == 'backwardation':
                    factors.append(('volatility_expansion', 0.7))
                elif vol_metrics['term_structure_state_short'] == 'contango':
                    factors.append(('volatility_contraction', 0.7))
            
            # Factor 4: Volatility-of-volatility
            if 'vol_of_vol_ratio' in vol_metrics:
                vov_ratio = vol_metrics['vol_of_vol_ratio']
                if vov_ratio > 0.3:
                    factors.append(('volatility_expansion', min(1.0, vov_ratio - 0.3)))
                elif vov_ratio < 0.1:
                    factors.append(('volatility_contraction', min(1.0, 0.1 - vov_ratio)))
            
            # Aggregate probabilities
            if factors:
                # Calculate regime probabilities
                for regime, score in factors:
                    result['regime_probability'][regime] = result['regime_probability'].get(regime, 0) + score
                
                # Normalize probabilities
                total_score = sum(result['regime_probability'].values())
                if total_score > 0:
                    for regime in result['regime_probability']:
                        result['regime_probability'][regime] /= total_score
                
                # Find most likely regime
                most_likely = max(result['regime_probability'].items(), key=lambda x: x[1])
                result['regime'] = most_likely[0]
                result['confidence'] = most_likely[1]
            
            return result
        
        # Ensure we have enough data
        if len(data) < 252:  # At least a year of data
            return {
                "status": "insufficient_data",
                "message": f"Need at least 252 days of data, got {len(data)}"
            }
        
        # Calculate realized volatility over multiple timeframes
        returns = data['Close'].pct_change().dropna()
        vol_metrics = {}
        
        # Annualized volatility for different timeframes
        timeframes = {
            'vol_10d': 10,
            'vol_20d': 20,
            'vol_60d': 60,
            'vol_90d': 90,
            'vol_120d': 120,
            'vol_252d': 252
        }
        
        for name, window in timeframes.items():
            if len(returns) >= window:
                vol_metrics[name] = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
        
        # Volatility term structure (backwardation vs contango)
        if all(k in vol_metrics for k in ['vol_10d', 'vol_60d', 'vol_252d']):
            vol_metrics['vol_term_structure_short'] = vol_metrics['vol_10d'] / vol_metrics['vol_60d'] - 1
            vol_metrics['vol_term_structure_long'] = vol_metrics['vol_60d'] / vol_metrics['vol_252d'] - 1
            
            # Term structure state (backwardation vs contango)
            if vol_metrics['vol_term_structure_short'] > 0.1:
                vol_metrics['term_structure_state_short'] = 'backwardation'
            elif vol_metrics['vol_term_structure_short'] < -0.1:
                vol_metrics['term_structure_state_short'] = 'contango'
            else:
                vol_metrics['term_structure_state_short'] = 'neutral'
                
            if vol_metrics['vol_term_structure_long'] > 0.1:
                vol_metrics['term_structure_state_long'] = 'backwardation'
            elif vol_metrics['vol_term_structure_long'] < -0.1:
                vol_metrics['term_structure_state_long'] = 'contango'
            else:
                vol_metrics['term_structure_state_long'] = 'neutral'
        
        # Volatility-of-volatility (VoV)
        if len(returns) >= 60:
            # Calculate rolling volatility series
            rolling_vol_20d = returns.rolling(window=20).std() * np.sqrt(252)
            # Calculate volatility of the volatility series
            vol_metrics['vol_of_vol'] = rolling_vol_20d.rolling(window=20).std().iloc[-1]
            vol_metrics['vol_of_vol_ratio'] = vol_metrics['vol_of_vol'] / vol_metrics.get('vol_60d', 1)
        
        # Historical volatility percentiles
        if len(returns) >= 252:
            rolling_vol_20d = returns.rolling(window=20).std() * np.sqrt(252)
            vol_metrics['vol_percentile_1y'] = percentile_rank(rolling_vol_20d.iloc[-1], rolling_vol_20d.iloc[-252:])
        
        # Regime classification with confidence score
        vol_regime = classify_volatility_regime(vol_metrics)
        vol_metrics.update(vol_regime)
        
        return vol_metrics

# Convenience function for standalone volatility regime analysis
def analyze_volatility_regime(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function for volatility regime analysis
    
    Args:
        data: DataFrame with OHLCV columns
        
    Returns:
        Dictionary containing volatility regime analysis
    """
    analyzer = MarketRegimeAnalyzer()
    return analyzer.analyze_volatility_regime(data)

# Export the necessary functions and classes
__all__ = ['RegimeState', 'MarketRegimeAnalyzer', 'analyze_volatility_regime']
