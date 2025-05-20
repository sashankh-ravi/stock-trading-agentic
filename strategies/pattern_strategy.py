"""Pattern Recognition Strategy

Implements advanced pattern recognition using:
1. Candlestick Patterns
2. Price Action Patterns
3. Volume Profile Analysis
4. Support/Resistance Detection
5. Multi-Day Pattern Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .base_strategy import BaseStrategy, Signal
import logging
import talib
from scipy.signal import argrelextrema
from .multi_day_patterns import MultiDayPatternAnalyzer

logger = logging.getLogger(__name__)

class PatternStrategy(BaseStrategy):
    def __init__(self,
                lookback: int = 20,
                pattern_confidence: float = 0.7,
                volume_threshold: float = 1.5,
                stop_loss_pct: float = 0.05,
                take_profit_pct: float = 0.15):
        """Initialize pattern recognition strategy
        
        Args:
            lookback: Period for pattern detection
            pattern_confidence: Minimum confidence for pattern signals
            volume_threshold: Volume surge threshold
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        super().__init__(lookback, stop_loss_pct, take_profit_pct)
        self.pattern_confidence = pattern_confidence
        self.volume_threshold = volume_threshold
        
        # Define candlestick patterns to detect
        self.bullish_patterns = {
            'CDL3WHITESOLDIERS': 100,
            'CDLMORNINGSTAR': 100,
            'CDLPIERCING': 90,
            'CDLENGULFING': 85,
            'CDLHAMMER': 80,
            'CDLINVERTEDHAMMER': 75
        }
        
        self.bearish_patterns = {
            'CDL3BLACKCROWS': 100,
            'CDLEVENINGSTAR': 100,
            'CDLENGULFING': 85,
            'CDLHANGINGMAN': 80,
            'CDLSHOOTINGSTAR': 75,
            'CDLHARAMI': 70
        }
        
        # Initialize multi-day pattern analyzer
        self.multi_day_analyzer = MultiDayPatternAnalyzer(lookback=lookback)

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on pattern recognition"""
        try:
            if len(data) < self.lookback:
                return self._create_signal(data)

            # Get OHLCV data
            open_prices = data['Open'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            volumes = data['Volume'].values

            # 1. Detect candlestick patterns
            pattern_signals = self._detect_candlestick_patterns(
                open_prices, high_prices, low_prices, close_prices
            )

            # 2. Detect multi-day patterns (new)
            multi_day_patterns = self.multi_day_analyzer.analyze(data)

            # 3. Detect price action patterns
            price_patterns = self._detect_price_patterns(
                close_prices, volumes
            )

            # 4. Check for additional technical patterns
            additional_patterns = self._check_additional_patterns(data)
            # Merge with price_patterns
            price_patterns.update(additional_patterns)

            # 5. Analyze volume profile
            volume_signal = self._analyze_volume_profile(volumes, close_prices)

            # 6. Find support/resistance levels
            levels = self._find_support_resistance(high_prices, low_prices)

            # 6. Check additional patterns
            additional_patterns = self._check_additional_patterns(data)

            # Create base signal
            signal = self._create_signal(data)
            current_price = close_prices[-1]

            # Combine all signals
            signal_strength, confidence = self._combine_signals(
                pattern_signals, multi_day_patterns, price_patterns, 
                volume_signal, levels, current_price
            )

            if abs(signal_strength) > self.pattern_confidence:
                # Set action based on signal strength
                signal.action = "buy" if signal_strength > 0 else "sell"
                signal.confidence = confidence
                
                # Calculate dynamic stop loss and take profit
                atr = self._calculate_atr(data)
                if signal.action == "buy":
                    signal.stop_loss = current_price * (1 - self.stop_loss_pct)
                    signal.take_profit = current_price * (1 + self.take_profit_pct)
                else:
                    signal.stop_loss = current_price * (1 + self.stop_loss_pct)
                    signal.take_profit = current_price * (1 - self.take_profit_pct)

                # Calculate position size based on ATR
                signal.position_size = self._calculate_position_size(current_price, atr)

            # Add analysis metrics
            signal.metrics.update({
                'pattern_signals': pattern_signals,
                'multi_day_patterns': multi_day_patterns,
                'price_patterns': price_patterns,
                'volume_signal': volume_signal,
                'support_resistance': levels,
                'additional_patterns': additional_patterns
            })

            return signal

        except Exception as e:
            logger.error(f"Error generating pattern signal: {e}")
            return self._create_signal(data)

    def _detect_candlestick_patterns(self,
                                  open_prices: np.ndarray,
                                  high_prices: np.ndarray,
                                  low_prices: np.ndarray,
                                  close_prices: np.ndarray) -> Dict:
        """Detect candlestick patterns using TA-Lib"""
        patterns = {}
        
        # Check bullish patterns
        for pattern_name, weight in self.bullish_patterns.items():
            pattern_func = getattr(talib, pattern_name)
            result = pattern_func(open_prices, high_prices, low_prices, close_prices)
            if result[-1] > 0:  # Bullish pattern
                patterns[pattern_name] = weight / 100.0
                
        # Check bearish patterns
        for pattern_name, weight in self.bearish_patterns.items():
            pattern_func = getattr(talib, pattern_name)
            result = pattern_func(open_prices, high_prices, low_prices, close_prices)
            if result[-1] < 0:  # Bearish pattern
                patterns[pattern_name] = -weight / 100.0
                
        return patterns

    def _detect_price_patterns(self,
                            prices: np.ndarray,
                            volumes: np.ndarray) -> Dict:
        """Detect price action patterns"""
        patterns = {}
        
        # Find local maxima and minima
        max_idx = argrelextrema(prices, np.greater)[0]
        min_idx = argrelextrema(prices, np.less)[0]
        
        if len(max_idx) >= 2 and len(min_idx) >= 2:
            # Check for double top
            if self._is_double_top(prices[max_idx[-2:]], volumes[max_idx[-2:]]):
                patterns['double_top'] = -0.8
                
            # Check for double bottom
            if self._is_double_bottom(prices[min_idx[-2:]], volumes[min_idx[-2:]]):
                patterns['double_bottom'] = 0.8
                
            # Check for head and shoulders
            if len(max_idx) >= 3:
                if self._is_head_and_shoulders(prices[max_idx[-3:]]):
                    patterns['head_shoulders'] = -0.9
                    
            # Check for inverse head and shoulders
            if len(min_idx) >= 3:
                if self._is_head_and_shoulders(prices[min_idx[-3:]], inverse=True):
                    patterns['inv_head_shoulders'] = 0.9
        
        return patterns

    def _analyze_volume_profile(self,
                             volumes: np.ndarray,
                             prices: np.ndarray) -> float:
        """Analyze volume profile for confirmation"""
        try:
            # Calculate volume moving average
            vol_ma = np.mean(volumes[-self.lookback:])
            current_vol = volumes[-1]
            
            # Calculate price change
            price_change = (prices[-1] / prices[-2]) - 1
            
            # Volume surge with price increase
            if current_vol > vol_ma * self.volume_threshold:
                if price_change > 0:
                    return 0.8  # Strong bullish
                elif price_change < 0:
                    return -0.8  # Strong bearish
                    
            return 0.0  # Neutral
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return 0.0

    def _find_support_resistance(self,
                              highs: np.ndarray,
                              lows: np.ndarray) -> Dict:
        """Find support and resistance levels"""
        try:
            # Find local peaks
            high_peaks = argrelextrema(highs, np.greater)[0]
            low_peaks = argrelextrema(lows, np.less)[0]
            
            if len(high_peaks) > 0 and len(low_peaks) > 0:
                # Get recent levels
                resistance = highs[high_peaks[-3:]]
                support = lows[low_peaks[-3:]]
                
                current_price = (highs[-1] + lows[-1]) / 2
                
                # Find closest levels
                closest_resistance = min(resistance[resistance > current_price], default=None)
                closest_support = max(support[support < current_price], default=None)
                
                return {
                    'resistance': closest_resistance,
                    'support': closest_support,
                    'r_strength': self._calculate_level_strength(closest_resistance, highs),
                    's_strength': self._calculate_level_strength(closest_support, lows)
                }
                
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            
        return {'resistance': None, 'support': None, 'r_strength': 0, 's_strength': 0}

    def _calculate_level_strength(self,
                               level: float,
                               prices: np.ndarray,
                               threshold: float = 0.02) -> float:
        """Calculate strength of a support/resistance level"""
        if level is None:
            return 0
            
        # Count touches of the level
        touches = np.sum(np.abs(prices - level) / level < threshold)
        
        # Normalize strength
        strength = min(touches / 5, 1.0)  # Cap at 5 touches
        return strength

    def _combine_signals(self,
                       pattern_signals: Dict,
                       multi_day_patterns: Dict,
                       price_patterns: Dict,
                       volume_signal: float,
                       levels: Dict,
                       current_price: float) -> Tuple[float, float]:
        """Combine all signals into final signal and confidence"""
        # Weight for each component
        weights = {
            'candlestick': 0.25,
            'multi_day': 0.25,
            'price_pattern': 0.2,
            'volume': 0.15,
            'support_resistance': 0.15
        }
        
        # Calculate weighted pattern signal
        if pattern_signals:
            pattern_score = np.mean(list(pattern_signals.values()))
        else:
            pattern_score = 0

        # Calculate weighted multi-day pattern signal
        if multi_day_patterns:
            multi_day_score = np.mean(list(multi_day_patterns.values()))
        else:
            multi_day_score = 0
            
        # Calculate weighted price pattern signal
        if price_patterns:
            price_score = np.mean(list(price_patterns.values()))
        else:
            price_score = 0
            
        # Calculate support/resistance signal
        sr_score = 0
        if levels['support'] is not None and levels['resistance'] is not None:
            # Distance to nearest level
            dist_to_resistance = (levels['resistance'] - current_price) / current_price
            dist_to_support = (current_price - levels['support']) / current_price
            
            if dist_to_resistance < dist_to_support:
                sr_score = -levels['r_strength']
            else:
                sr_score = levels['s_strength']
                
        # Combine signals
        signal_strength = (
            weights['candlestick'] * pattern_score +
            weights['multi_day'] * multi_day_score +
            weights['price_pattern'] * price_score +
            weights['volume'] * volume_signal +
            weights['support_resistance'] * sr_score
        )
        
        # Calculate confidence based on agreement
        signals = [
            1 if s > 0 else -1 if s < 0 else 0
            for s in [pattern_score, multi_day_score, price_score, volume_signal, sr_score]
            if s != 0
        ]
        
        if signals:
            agreement = np.mean(np.abs(signals))
            confidence = min(1.0, agreement * abs(signal_strength))
        else:
            confidence = 0
            
        return signal_strength, confidence

    def _is_double_top(self,
                     peaks: np.ndarray,
                     volumes: np.ndarray,
                     threshold: float = 0.02) -> bool:
        """Check for double top pattern"""
        if len(peaks) < 2:
            return False
            
        price_diff = abs(peaks[1] - peaks[0]) / peaks[0]
        volume_decline = volumes[1] < volumes[0]
        
        return price_diff < threshold and volume_decline

    def _is_double_bottom(self,
                        troughs: np.ndarray,
                        volumes: np.ndarray,
                        threshold: float = 0.02) -> bool:
        """Check for double bottom pattern"""
        if len(troughs) < 2:
            return False
            
        price_diff = abs(troughs[1] - troughs[0]) / troughs[0]
        volume_increase = volumes[1] > volumes[0]
        
        return price_diff < threshold and volume_increase

    def _is_head_and_shoulders(self,
                            peaks: np.ndarray,
                            inverse: bool = False,
                            threshold: float = 0.05) -> bool:
        """Check for head and shoulders pattern"""
        if len(peaks) < 3:
            return False
            
        if not inverse:
            # Regular H&S: middle peak higher than shoulders
            return (peaks[1] > peaks[0] and
                   peaks[1] > peaks[2] and
                   abs(peaks[0] - peaks[2]) / peaks[0] < threshold)
        else:
            # Inverse H&S: middle trough lower than shoulders
            return (peaks[1] < peaks[0] and
                   peaks[1] < peaks[2] and
                   abs(peaks[0] - peaks[2]) / peaks[0] < threshold)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for position sizing
        
        Args:
            data: OHLCV DataFrame
            period: ATR calculation period
            
        Returns:
            float: ATR value
        """
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            # Use TA-Lib for ATR calculation
            atr = talib.ATR(high, low, close, timeperiod=period)
            
            # Return the latest ATR value
            return atr[-1]
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            # Fallback: calculate a simple approximation of ATR
            range_values = data['High'] - data['Low']
            return range_values.mean()
            
    def _calculate_position_size(self, price: float, atr: float) -> float:
        """Calculate position size based on ATR
        
        Uses a risk-based position sizing approach where position size
        is inversely proportional to volatility (ATR)
        
        Args:
            price: Current asset price
            atr: Average True Range value
            
        Returns:
            float: Suggested position size multiplier (0-1)
        """
        # Base risk percentage (1% of account per trade)
        risk_pct = 0.01
        
        # Use ATR-based sizing: smaller position for higher volatility
        risk_multiple = 2.0  # Risk multiple for ATR
        
        # Calculate position size as percentage of maximum
        atr_ratio = atr / price  # Normalized ATR
        
        if atr_ratio > 0:
            # Position size inversely proportional to volatility
            position_size = min(risk_pct / (atr_ratio * risk_multiple), 1.0)
            return max(0.1, position_size)  # Minimum position size of 10%
        else:
            return 0.5  # Default to 50% if ATR calculation fails

    def _check_additional_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Check for additional technical patterns not covered by the basic pattern detection
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dict: Additional pattern signals
        """
        patterns = {}
        
        try:
            if len(data) < 30:  # Need sufficient data
                return patterns
                
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values
            
            # Calculate moving averages for trend context
            ma10 = talib.SMA(close, timeperiod=10)
            ma20 = talib.SMA(close, timeperiod=20)
            ma50 = talib.SMA(close, timeperiod=50)
            
            # === Price-based patterns ===
            
            # Golden Cross (MA10 crosses above MA20)
            if ma10[-2] < ma20[-2] and ma10[-1] > ma20[-1]:
                patterns['golden_cross'] = 0.85
                
            # Death Cross (MA10 crosses below MA20)
            if ma10[-2] > ma20[-2] and ma10[-1] < ma20[-1]:
                patterns['death_cross'] = -0.85
                
            # Cup and Handle pattern (more complex)
            if self._is_cup_and_handle(close[-50:], volume[-50:]):
                patterns['cup_handle'] = 0.9
                
            # === Breakout patterns ===
            
            # Check for breakout above resistance
            resistance_high = max(high[-30:-5])
            if high[-1] > resistance_high and close[-1] > resistance_high:
                patterns['resistance_breakout'] = 0.8
                
            # Check for breakdown below support
            support_low = min(low[-30:-5])
            if low[-1] < support_low and close[-1] < support_low:
                patterns['support_breakdown'] = -0.8
                
            # === Volatility-based patterns ===
            
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            # Bollinger Band squeeze and expansion
            bb_width = (upper - lower) / middle
            
            # Squeeze (narrowing bands)
            if bb_width[-1] < min(bb_width[-20:-1]):
                patterns['bollinger_squeeze'] = 0.7
                
            # === Volume-based patterns ===
            
            # Volume climax (spike in volume with price reversal)
            vol_avg = np.mean(volume[-10:])
            if volume[-1] > vol_avg * 2.5:
                # Bearish climax (high volume and price falls)
                if close[-1] < close[-2] and high[-1] > high[-2]:
                    patterns['volume_climax_bearish'] = -0.85
                # Bullish climax (high volume and price rises from low)
                elif close[-1] > close[-2] and low[-1] < low[-2]:
                    patterns['volume_climax_bullish'] = 0.85
            
        except Exception as e:
            logger.error(f"Error in additional pattern check: {e}")
            
        return patterns
    
    def _is_cup_and_handle(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """
        Detect cup and handle pattern
        
        Cup and Handle pattern characteristics:
        1. U-shaped price movement (cup)
        2. Small downward drift (handle)
        3. Breakout above resistance
        
        Args:
            prices: Close price array
            volumes: Volume array
            
        Returns:
            bool: True if pattern detected
        """
        if len(prices) < 40:
            return False
            
        try:
            # Cup portion typically spans 30-40 bars
            cup_period = min(40, len(prices) - 10)
            
            # Find lowest point in cup range
            cup_low_idx = np.argmin(prices[:cup_period])
            
            # Check if cup low point is roughly in the middle (U-shape)
            if cup_low_idx < cup_period * 0.3 or cup_low_idx > cup_period * 0.7:
                return False
                
            # Left and right cup heights should be similar
            left_height = prices[0] - prices[cup_low_idx]
            right_height = prices[cup_period] - prices[cup_low_idx]
            
            if abs(left_height - right_height) / left_height > 0.3:
                return False
                
            # Handle portion should be a small pullback (last 7-10 bars)
            handle_start = cup_period
            handle_depth = (max(prices[handle_start:]) - min(prices[handle_start:])) / prices[handle_start]
            
            # Handle shouldn't be too deep
            if handle_depth > 0.15:
                return False
                
            # Final breakout above resistance
            if prices[-1] > max(prices[5:cup_period]):
                # Check if volume is increasing on breakout
                if volumes[-1] > np.mean(volumes[-5:]):
                    return True
                    
        except Exception as e:
            logger.error(f"Error detecting cup and handle: {e}")
            
        return False
