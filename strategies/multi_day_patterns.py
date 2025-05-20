"""
Multi-Day Candlestick Pattern Analysis Module

This module extends the pattern recognition capabilities by analyzing multi-day candlestick patterns and formations
beyond what is available in TA-Lib. It implements sequence-based pattern recognition for more robust trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MultiDayPatternAnalyzer:
    """
    Advanced multi-day candlestick pattern analyzer that detects patterns spanning multiple days
    and evaluates their reliability based on contextual factors.
    """
    
    def __init__(self, lookback: int = 20, min_pattern_bars: int = 3, max_pattern_bars: int = 10):
        """
        Initialize the multi-day pattern analyzer
        
        Parameters:
        -----------
        lookback : int
            Number of bars to look back for pattern detection
        min_pattern_bars : int
            Minimum number of bars required for a valid pattern
        max_pattern_bars : int
            Maximum number of bars to consider in a pattern sequence
        """
        self.lookback = lookback
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        
        # Define multi-day pattern detection functions
        self.pattern_detectors = {
            # Trend reversal patterns
            'three_day_reversal_bottom': self._detect_three_day_reversal_bottom,
            'three_day_reversal_top': self._detect_three_day_reversal_top,
            'key_reversal_day': self._detect_key_reversal,
            
            # Trend continuation patterns
            'three_day_strength': self._detect_three_day_strength,
            'three_outside_up': self._detect_three_outside_up,
            'three_outside_down': self._detect_three_outside_down,
            'three_inside_up': self._detect_three_inside_up,
            'three_inside_down': self._detect_three_inside_down,
            
            # Gap patterns
            'gap_and_trap': self._detect_gap_and_trap,
            'exhaustion_gap': self._detect_exhaustion_gap,
            'island_reversal': self._detect_island_reversal,
            
            # Volatility patterns
            'volatility_squeeze': self._detect_volatility_squeeze,
            'volatility_expansion': self._detect_volatility_expansion,
            
            # Multi-day momentum patterns
            'three_day_momentum_shift': self._detect_momentum_shift,
        }
        
        # Pattern weights based on reliability (0-100)
        self.pattern_weights = {
            'three_day_reversal_bottom': 85,
            'three_day_reversal_top': 85,
            'key_reversal_day': 80,
            'three_day_strength': 75,
            'three_outside_up': 90,
            'three_outside_down': 90,
            'three_inside_up': 85,
            'three_inside_down': 85,
            'gap_and_trap': 80,
            'exhaustion_gap': 75,
            'island_reversal': 90,
            'volatility_squeeze': 70,
            'volatility_expansion': 75,
            'three_day_momentum_shift': 80
        }
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze price data for multi-day candlestick patterns
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of patterns and their signal strength (-1.0 to 1.0)
        """
        if len(df) < self.lookback:
            return {}  # Not enough data
            
        # Get recent price data
        recent_data = df.iloc[-self.lookback:].copy()
        
        # Calculate additional metrics
        self._add_metrics(recent_data)
        
        # Detect patterns
        patterns = {}
        market_context = self._determine_market_context(recent_data)
        
        # Run each pattern detector
        for pattern_name, detector_func in self.pattern_detectors.items():
            try:
                result = detector_func(recent_data)
                
                if result is not None:
                    is_valid, direction, strength = result
                    
                    if is_valid:
                        # Adjust strength based on market context
                        adjusted_strength = self._adjust_pattern_strength(
                            pattern_name, strength, market_context, recent_data
                        )
                        
                        # Calculate signal value
                        signal_value = adjusted_strength * (1 if direction == 'bullish' else -1)
                        
                        # Add to patterns dictionary
                        patterns[pattern_name] = signal_value
            except Exception as e:
                logger.error(f"Error detecting pattern {pattern_name}: {e}")
        
        return patterns
    
    def _add_metrics(self, df: pd.DataFrame) -> None:
        """Add technical metrics needed for pattern detection"""
        # Basic candlestick metrics
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['range'] = df['High'] - df['Low']
        df['body_percent'] = df['body_size'] / df['range']
        
        # Candle color and direction
        df['is_green'] = df['Close'] > df['Open']
        df['is_up_day'] = df['Close'] > df['Close'].shift(1)
        
        # Volume metrics
        df['volume_ma'] = df['Volume'].rolling(10).mean()
        df['rel_volume'] = df['Volume'] / df['volume_ma']
        
        # Trend metrics
        df['ma5'] = df['Close'].rolling(5).mean()
        df['ma20'] = df['Close'].rolling(20).mean()
        df['trend_direction'] = np.where(df['ma5'] > df['ma5'].shift(5), 1, 
                              np.where(df['ma5'] < df['ma5'].shift(5), -1, 0))
        
        # Gap metrics
        df['gap_up'] = df['Low'] > df['High'].shift(1)
        df['gap_down'] = df['High'] < df['Low'].shift(1)
        df['gap_size'] = np.where(df['gap_up'], df['Low'] - df['High'].shift(1),
                                 np.where(df['gap_down'], df['High'] - df['Low'].shift(1), 0))
        
        # Volatility metrics
        df['atr5'] = df['range'].rolling(5).mean()
        df['volatility'] = df['range'] / df['Close'].shift(1)
    
    def _determine_market_context(self, df: pd.DataFrame) -> Dict:
        """Determine the current market context for pattern relevance adjustment"""
        recent_close = df['Close'].iloc[-1]
        
        # Determine trend
        ma5 = df['ma5'].iloc[-1] if 'ma5' in df.columns else df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['ma20'].iloc[-1] if 'ma20' in df.columns else df['Close'].rolling(20).mean().iloc[-1]
        
        if ma5 > ma20 and ma5 > ma5.rolling(5).mean().iloc[-1]:
            trend = 'strong_uptrend'
        elif ma5 > ma20:
            trend = 'uptrend'
        elif ma5 < ma20 and ma5 < ma5.rolling(5).mean().iloc[-1]:
            trend = 'strong_downtrend'
        elif ma5 < ma20:
            trend = 'downtrend'
        else:
            trend = 'sideways'
            
        # Calculate recent volatility
        recent_volatility = df['range'].iloc[-5:].mean() / df['Close'].iloc[-5:].mean()
        if recent_volatility > 0.02:
            volatility = 'high'
        elif recent_volatility < 0.01:
            volatility = 'low'
        else:
            volatility = 'normal'
            
        # Determine volume context
        recent_rel_volume = df['Volume'].iloc[-3:].mean() / df['Volume'].iloc[-10:].mean()
        if recent_rel_volume > 1.5:
            volume = 'high'
        elif recent_rel_volume < 0.7:
            volume = 'low'
        else:
            volume = 'normal'
            
        return {
            'trend': trend,
            'volatility': volatility,
            'volume': volume
        }
    
    def _adjust_pattern_strength(self, pattern_name: str, base_strength: float, 
                              market_context: Dict, data: pd.DataFrame) -> float:
        """
        Adjust pattern strength based on market context and pattern-specific factors
        
        Parameters:
        -----------
        pattern_name : str
            Name of the detected pattern
        base_strength : float
            Base strength of the pattern (0-1)
        market_context : Dict
            Dictionary with market context information
        data : pd.DataFrame
            Price data with technical metrics
            
        Returns:
        --------
        float
            Adjusted strength value (0-1)
        """
        # Get base weight for this pattern
        base_weight = self.pattern_weights.get(pattern_name, 75) / 100.0
        
        # Apply market context adjustments
        context_multiplier = 1.0
        
        # Different patterns are more relevant in different market contexts
        if pattern_name in ['three_day_reversal_bottom', 'three_day_reversal_top', 'key_reversal_day']:
            # Reversal patterns are more reliable at extremes or after extended trends
            if 'strong' in market_context['trend']:
                context_multiplier *= 1.2
            if market_context['volume'] == 'high':
                context_multiplier *= 1.3
        
        elif pattern_name in ['three_day_strength', 'three_outside_up', 'three_outside_down']:
            # Continuation patterns are more reliable in established trends
            if market_context['trend'] in ['uptrend', 'downtrend']:
                context_multiplier *= 1.2
            if market_context['volatility'] == 'low':
                context_multiplier *= 1.1
        
        elif pattern_name in ['gap_and_trap', 'exhaustion_gap', 'island_reversal']:
            # Gap patterns are more significant with high volume
            if market_context['volume'] == 'high':
                context_multiplier *= 1.4
        
        elif pattern_name in ['volatility_squeeze', 'volatility_expansion']:
            # Volatility patterns are more significant after volatility extremes
            if market_context['volatility'] in ['high', 'low']:
                context_multiplier *= 1.3
        
        # Ensure the final strength is in the range [0, 1]
        adjusted_strength = min(base_weight * base_strength * context_multiplier, 1.0)
        
        return adjusted_strength
    
    # ==== Pattern Detection Methods ====
    
    def _detect_three_day_reversal_bottom(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect a three-day bullish reversal pattern:
        1. Downtrend
        2. Three consecutive down days
        3. Third day opens lower and closes higher than previous day
        4. High volume on third day
        """
        if len(df) < 5:
            return None
            
        last_prices = df.iloc[-5:]
        
        # Check for downtrend
        if not (last_prices['ma5'].iloc[-1] < last_prices['ma20'].iloc[-1]):
            return None
            
        # Check for three consecutive down days, then a reversal
        if not (last_prices['is_up_day'].iloc[-4:-1].sum() == 0 and last_prices['is_up_day'].iloc[-1]):
            return None
            
        # Third day opens lower and closes higher
        if not (last_prices['Open'].iloc[-1] < last_prices['Close'].iloc[-2] and 
                last_prices['Close'].iloc[-1] > last_prices['Open'].iloc[-1]):
            return None
            
        # High volume on reversal day
        volume_factor = last_prices['rel_volume'].iloc[-1]
        
        pattern_strength = min(0.6 + (0.4 * volume_factor / 2.0), 1.0)
        
        return (True, 'bullish', pattern_strength)
    
    def _detect_three_day_reversal_top(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect a three-day bearish reversal pattern:
        1. Uptrend
        2. Three consecutive up days
        3. Third day opens higher and closes lower than previous day
        4. High volume on third day
        """
        if len(df) < 5:
            return None
            
        last_prices = df.iloc[-5:]
        
        # Check for uptrend
        if not (last_prices['ma5'].iloc[-1] > last_prices['ma20'].iloc[-1]):
            return None
            
        # Check for three consecutive up days, then a reversal
        if not (last_prices['is_up_day'].iloc[-4:-1].sum() == 3 and not last_prices['is_up_day'].iloc[-1]):
            return None
            
        # Third day opens higher and closes lower
        if not (last_prices['Open'].iloc[-1] > last_prices['Close'].iloc[-2] and 
                last_prices['Close'].iloc[-1] < last_prices['Open'].iloc[-1]):
            return None
            
        # High volume on reversal day
        volume_factor = last_prices['rel_volume'].iloc[-1]
        
        pattern_strength = min(0.6 + (0.4 * volume_factor / 2.0), 1.0)
        
        return (True, 'bearish', pattern_strength)
    
    def _detect_key_reversal(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect a key reversal day:
        1. Price makes a new high/low (5-day lookback)
        2. Price reverses and closes in the opposite direction
        3. Volume is higher than previous day
        """
        if len(df) < 6:
            return None
            
        last_prices = df.iloc[-6:]
        
        # Check if today made a new high or low
        made_new_high = last_prices['High'].iloc[-1] > last_prices['High'].iloc[-6:-1].max()
        made_new_low = last_prices['Low'].iloc[-1] < last_prices['Low'].iloc[-6:-1].min()
        
        # Check if price reversed from high or low
        bullish_reversal = made_new_low and last_prices['Close'].iloc[-1] > last_prices['Open'].iloc[-1]
        bearish_reversal = made_new_high and last_prices['Close'].iloc[-1] < last_prices['Open'].iloc[-1]
        
        # Check if volume increased
        volume_increased = last_prices['Volume'].iloc[-1] > last_prices['Volume'].iloc[-2]
        
        if (bullish_reversal or bearish_reversal) and volume_increased:
            direction = 'bullish' if bullish_reversal else 'bearish'
            
            # Strength based on the size of the reversal
            if direction == 'bullish':
                range_percent = (last_prices['Close'].iloc[-1] - last_prices['Low'].iloc[-1]) / last_prices['Low'].iloc[-1]
            else:
                range_percent = (last_prices['High'].iloc[-1] - last_prices['Close'].iloc[-1]) / last_prices['High'].iloc[-1]
                
            pattern_strength = min(0.5 + range_percent * 10, 1.0)  # Normalize to [0, 1]
            
            # Adjust for volume
            volume_factor = last_prices['rel_volume'].iloc[-1]
            pattern_strength = min(pattern_strength * (1 + (volume_factor - 1) * 0.3), 1.0)
            
            return (True, direction, pattern_strength)
        
        return None
    
    def _detect_three_day_strength(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect three consecutive days in the same direction with increasing strength:
        1. Three days all green or all red
        2. Each day's range is larger than the previous
        3. Each close is more extreme than the previous
        """
        if len(df) < 3:
            return None
            
        last_prices = df.iloc[-3:]
        
        # Check if all three days are in the same direction
        all_up = last_prices['is_up_day'].all()
        all_down = (~last_prices['is_up_day']).all()
        
        if not (all_up or all_down):
            return None
            
        # Check for increasing ranges
        increasing_range = (last_prices['range'].diff() > 0).iloc[1:].all()
        
        # Check for more extreme closes
        if all_up:
            more_extreme_closes = (last_prices['Close'].diff() > 0).iloc[1:].all()
            direction = 'bullish'
        else:
            more_extreme_closes = (last_prices['Close'].diff() < 0).iloc[1:].all()
            direction = 'bearish'
            
        if increasing_range and more_extreme_closes:
            # Calculate strength based on the magnitude of the moves
            total_move = abs(last_prices['Close'].iloc[-1] - last_prices['Close'].iloc[-3]) / last_prices['Close'].iloc[-3]
            pattern_strength = min(total_move * 10, 1.0)  # Normalize to [0, 1]
            
            return (True, direction, pattern_strength)
            
        return None
    
    def _detect_three_outside_up(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect three outside up pattern:
        1. Bearish candle
        2. Bullish candle that engulfs the previous candle
        3. Third candle that closes higher than the second
        """
        if len(df) < 3:
            return None
            
        prices = df.iloc[-3:].copy()
        
        # Day 1: Bearish candle
        day1_bearish = prices['Close'].iloc[0] < prices['Open'].iloc[0]
        
        # Day 2: Bullish candle that engulfs day 1
        day2_bullish = prices['Close'].iloc[1] > prices['Open'].iloc[1]
        day2_engulfs = (prices['Open'].iloc[1] <= prices['Close'].iloc[0] and 
                       prices['Close'].iloc[1] >= prices['Open'].iloc[0])
        
        # Day 3: Closes higher than day 2
        day3_higher = prices['Close'].iloc[2] > prices['Close'].iloc[1]
        
        if day1_bearish and day2_bullish and day2_engulfs and day3_higher:
            # Calculate strength based on the size of the move and volume
            move_size = (prices['Close'].iloc[2] - prices['Close'].iloc[0]) / prices['Close'].iloc[0]
            volume_increase = prices['Volume'].iloc[1:].mean() / prices['Volume'].iloc[0]
            
            pattern_strength = min(0.6 + (move_size * 5) + (volume_increase * 0.2), 1.0)
            
            return (True, 'bullish', pattern_strength)
            
        return None
    
    def _detect_three_outside_down(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect three outside down pattern:
        1. Bullish candle
        2. Bearish candle that engulfs the previous candle
        3. Third candle that closes lower than the second
        """
        if len(df) < 3:
            return None
            
        prices = df.iloc[-3:].copy()
        
        # Day 1: Bullish candle
        day1_bullish = prices['Close'].iloc[0] > prices['Open'].iloc[0]
        
        # Day 2: Bearish candle that engulfs day 1
        day2_bearish = prices['Close'].iloc[1] < prices['Open'].iloc[1]
        day2_engulfs = (prices['Open'].iloc[1] >= prices['Close'].iloc[0] and 
                       prices['Close'].iloc[1] <= prices['Open'].iloc[0])
        
        # Day 3: Closes lower than day 2
        day3_lower = prices['Close'].iloc[2] < prices['Close'].iloc[1]
        
        if day1_bullish and day2_bearish and day2_engulfs and day3_lower:
            # Calculate strength based on the size of the move and volume
            move_size = (prices['Close'].iloc[0] - prices['Close'].iloc[2]) / prices['Close'].iloc[0]
            volume_increase = prices['Volume'].iloc[1:].mean() / prices['Volume'].iloc[0]
            
            pattern_strength = min(0.6 + (move_size * 5) + (volume_increase * 0.2), 1.0)
            
            return (True, 'bearish', pattern_strength)
            
        return None
    
    def _detect_three_inside_up(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect three inside up pattern:
        1. Bearish candle
        2. Bullish candle with range inside the previous candle
        3. Third candle that breaks above the high of the first candle
        """
        if len(df) < 3:
            return None
            
        prices = df.iloc[-3:].copy()
        
        # Day 1: Bearish candle
        day1_bearish = prices['Close'].iloc[0] < prices['Open'].iloc[0]
        
        # Day 2: Bullish candle inside day 1's range
        day2_bullish = prices['Close'].iloc[1] > prices['Open'].iloc[1]
        day2_inside = (prices['High'].iloc[1] <= prices['High'].iloc[0] and 
                      prices['Low'].iloc[1] >= prices['Low'].iloc[0])
        
        # Day 3: Breaks above the high of day 1
        day3_breakout = prices['Close'].iloc[2] > prices['High'].iloc[0]
        
        if day1_bearish and day2_bullish and day2_inside and day3_breakout:
            # Calculate strength based on breakout size and volume
            breakout_size = (prices['Close'].iloc[2] - prices['High'].iloc[0]) / prices['High'].iloc[0]
            volume_confirmation = prices['Volume'].iloc[2] > prices['Volume'].iloc[1]
            
            pattern_strength = min(0.7 + (breakout_size * 10) + (0.1 if volume_confirmation else 0), 1.0)
            
            return (True, 'bullish', pattern_strength)
            
        return None
    
    def _detect_three_inside_down(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect three inside down pattern:
        1. Bullish candle
        2. Bearish candle with range inside the previous candle
        3. Third candle that breaks below the low of the first candle
        """
        if len(df) < 3:
            return None
            
        prices = df.iloc[-3:].copy()
        
        # Day 1: Bullish candle
        day1_bullish = prices['Close'].iloc[0] > prices['Open'].iloc[0]
        
        # Day 2: Bearish candle inside day 1's range
        day2_bearish = prices['Close'].iloc[1] < prices['Open'].iloc[1]
        day2_inside = (prices['High'].iloc[1] <= prices['High'].iloc[0] and 
                      prices['Low'].iloc[1] >= prices['Low'].iloc[0])
        
        # Day 3: Breaks below the low of day 1
        day3_breakdown = prices['Close'].iloc[2] < prices['Low'].iloc[0]
        
        if day1_bullish and day2_bearish and day2_inside and day3_breakdown:
            # Calculate strength based on breakdown size and volume
            breakdown_size = (prices['Low'].iloc[0] - prices['Close'].iloc[2]) / prices['Low'].iloc[0]
            volume_confirmation = prices['Volume'].iloc[2] > prices['Volume'].iloc[1]
            
            pattern_strength = min(0.7 + (breakdown_size * 10) + (0.1 if volume_confirmation else 0), 1.0)
            
            return (True, 'bearish', pattern_strength)
            
        return None
    
    def _detect_gap_and_trap(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect gap and trap pattern:
        1. Price gaps in one direction
        2. Initial move in the direction of the gap
        3. Price reverses and closes beyond the opposite side of the gap
        """
        if len(df) < 2:
            return None
            
        prices = df.iloc[-2:].copy()
        
        # Check for gaps
        gap_up = prices['gap_up'].iloc[1]
        gap_down = prices['gap_down'].iloc[1]
        
        if not (gap_up or gap_down):
            return None
            
        if gap_up:
            # Gap up trap: Opens above previous day, but closes below previous day's close
            trap = prices['Close'].iloc[1] < prices['Close'].iloc[0]
            direction = 'bearish'
        else:
            # Gap down trap: Opens below previous day, but closes above previous day's close
            trap = prices['Close'].iloc[1] > prices['Close'].iloc[0]
            direction = 'bullish'
            
        if trap:
            # Calculate strength based on the size of the gap and the trap
            gap_size = abs(prices['gap_size'].iloc[1]) / prices['Close'].iloc[0]
            trap_size = abs(prices['Close'].iloc[1] - prices['Open'].iloc[1]) / prices['Open'].iloc[1]
            
            pattern_strength = min(0.6 + (gap_size * 5) + (trap_size * 5), 1.0)
            
            return (True, direction, pattern_strength)
            
        return None
    
    def _detect_exhaustion_gap(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect exhaustion gap pattern:
        1. Strong trend for several days
        2. Gap in the direction of the trend with high volume
        3. Price reversal after the gap
        """
        if len(df) < 6:
            return None
            
        prices = df.iloc[-6:].copy()
        
        # Check for strong trend
        if 'trend_direction' in prices.columns:
            prior_trend = prices['trend_direction'].iloc[:-1].mean()
            strong_uptrend = prior_trend > 0.6
            strong_downtrend = prior_trend < -0.6
        else:
            # Calculate trend manually
            price_direction = np.sign(prices['Close'].diff()).iloc[:-1]
            strong_uptrend = (price_direction > 0).mean() > 0.8
            strong_downtrend = (price_direction < 0).mean() > 0.8
            
        if not (strong_uptrend or strong_downtrend):
            return None
            
        # Check for gap in the trend direction
        gap_up = prices['gap_up'].iloc[-1]
        gap_down = prices['gap_down'].iloc[-1]
        
        if (strong_uptrend and not gap_up) or (strong_downtrend and not gap_down):
            return None
            
        # Check for reversal after the gap
        if strong_uptrend:
            reversal = prices['Close'].iloc[-1] < prices['Open'].iloc[-1]
            direction = 'bearish'
        else:
            reversal = prices['Close'].iloc[-1] > prices['Open'].iloc[-1]
            direction = 'bullish'
            
        if reversal:
            # Calculate strength based on gap size, volume, and reversal size
            gap_size = abs(prices['gap_size'].iloc[-1]) / prices['Close'].iloc[-2]
            volume_surge = prices['rel_volume'].iloc[-1]
            reversal_size = abs(prices['Close'].iloc[-1] - prices['Open'].iloc[-1]) / prices['Open'].iloc[-1]
            
            pattern_strength = min(0.5 + (gap_size * 3) + (min(volume_surge, 3) * 0.1) + (reversal_size * 3), 1.0)
            
            return (True, direction, pattern_strength)
            
        return None
    
    def _detect_island_reversal(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect island reversal pattern:
        1. Gap in the trend direction
        2. One or more days of sideways movement
        3. Gap in the opposite direction
        """
        if len(df) < 4:
            return None
            
        # We need to check for various island lengths
        for island_length in range(1, 4):
            if len(df) < island_length + 3:
                continue
                
            # Get relevant price segments
            pre_island = df.iloc[-(island_length+3):-(island_length+1)]
            island = df.iloc[-(island_length+1):-1]
            post_island = df.iloc[-1:]
            
            # Check for gaps on both sides of the island
            gap_into_island = (island['Low'].iloc[0] > pre_island['High'].iloc[-1] or 
                            island['High'].iloc[0] < pre_island['Low'].iloc[-1])
                            
            gap_out_of_island = (post_island['Low'].iloc[0] > island['High'].iloc[-1] or 
                               post_island['High'].iloc[0] < island['Low'].iloc[-1])
                               
            if not (gap_into_island and gap_out_of_island):
                continue
                
            # Determine direction of the pattern
            if post_island['Low'].iloc[0] > island['High'].iloc[-1]:
                direction = 'bullish'  # Gap up out of island
            else:
                direction = 'bearish'  # Gap down out of island
                
            # Calculate strength based on the size of the gaps and the island
            gap_into_size = min(abs(island['Open'].iloc[0] - pre_island['Close'].iloc[-1]) / pre_island['Close'].iloc[-1], 0.1)
            gap_out_size = min(abs(post_island['Open'].iloc[0] - island['Close'].iloc[-1]) / island['Close'].iloc[-1], 0.1)
            
            # Volume confirmation
            volume_confirmation = post_island['Volume'].iloc[0] > island['Volume'].mean()
            
            pattern_strength = min(0.7 + (gap_into_size * 10) + (gap_out_size * 10) + (0.1 if volume_confirmation else 0), 1.0)
            
            return (True, direction, pattern_strength)
            
        return None
    
    def _detect_volatility_squeeze(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect volatility squeeze pattern:
        1. Decreasing volatility for multiple days
        2. Range contraction
        3. Often precedes a volatility expansion
        """
        if len(df) < 10:
            return None
            
        prices = df.iloc[-10:].copy()
        
        # Calculate volatility measures
        if 'atr5' not in prices.columns:
            prices['atr5'] = prices['range'].rolling(5).mean()
            
        # Check for volatility contraction
        recent_volatility = prices['atr5'].iloc[-3:].mean()
        prior_volatility = prices['atr5'].iloc[-6:-3].mean()
        
        volatility_contraction = recent_volatility < prior_volatility * 0.7
        
        if not volatility_contraction:
            return None
            
        # Check for narrowing price range
        recent_range = (prices['High'] - prices['Low']).iloc[-3:].mean()
        prior_range = (prices['High'] - prices['Low']).iloc[-6:-3].mean()
        
        range_contraction = recent_range < prior_range * 0.7
        
        if range_contraction:
            # Neutral pattern - need additional confirmation for direction
            
            # Try to determine likely breakout direction
            if prices['Close'].iloc[-1] > prices['Close'].iloc[-2]:
                direction = 'bullish'
            elif prices['Close'].iloc[-1] < prices['Close'].iloc[-2]:
                direction = 'bearish'
            else:
                direction = 'bullish'  # Default to bullish if unclear
                
            # Calculate strength based on degree of contraction
            contraction_ratio = prior_volatility / recent_volatility
            
            pattern_strength = min(0.5 + (contraction_ratio - 1) * 0.2, 1.0)
            
            return (True, direction, pattern_strength)
            
        return None
    
    def _detect_volatility_expansion(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect volatility expansion pattern:
        1. Sudden increase in volatility after a period of contraction
        2. Often occurs with breakouts
        3. Direction determined by price movement
        """
        if len(df) < 6:
            return None
            
        prices = df.iloc[-6:].copy()
        
        # Calculate if today's range is significantly larger than recent average
        recent_range_avg = prices['range'].iloc[-4:-1].mean()
        today_range = prices['range'].iloc[-1]
        
        range_expansion = today_range > recent_range_avg * 1.5
        
        if not range_expansion:
            return None
            
        # Check if preceded by a volatility contraction
        prior_volatility = prices['volatility'].iloc[-4:-1].mean()
        older_volatility = prices['volatility'].iloc[-6:-4].mean()
        
        preceded_by_contraction = prior_volatility < older_volatility * 0.8
        
        if preceded_by_contraction:
            # Determine direction based on today's price movement
            if prices['Close'].iloc[-1] > prices['Open'].iloc[-1]:
                direction = 'bullish'
            else:
                direction = 'bearish'
                
            # Calculate strength based on the size of expansion and volume
            expansion_ratio = today_range / recent_range_avg
            volume_confirmation = prices['Volume'].iloc[-1] > prices['Volume'].iloc[-4:-1].mean() * 1.3
            
            pattern_strength = min(0.6 + (expansion_ratio - 1) * 0.1 + (0.2 if volume_confirmation else 0), 1.0)
            
            return (True, direction, pattern_strength)
            
        return None
    
    def _detect_momentum_shift(self, df: pd.DataFrame) -> Optional[Tuple[bool, str, float]]:
        """
        Detect significant momentum shift over three days:
        1. Consistent price movement in one direction
        2. Sudden shift with increasing momentum in the opposite direction
        3. Often indicates trend reversal
        """
        if len(df) < 6:
            return None
            
        prices = df.iloc[-6:].copy()
        
        # Calculate daily price changes
        prices['change'] = prices['Close'].pct_change()
        prices['abs_change'] = prices['change'].abs()
        
        # Check for consistent direction before the last 3 days
        prior_changes = prices['change'].iloc[-6:-3]
        consistent_prior_direction = (prior_changes > 0).all() or (prior_changes < 0).all()
        
        if not consistent_prior_direction:
            return None
            
        prior_direction = 1 if prior_changes.iloc[0] > 0 else -1
        
        # Check for opposite direction in the last 3 days
        recent_changes = prices['change'].iloc[-3:]
        opposite_recent_direction = (recent_changes * prior_direction < 0).all()
        
        if not opposite_recent_direction:
            return None
            
        # Check for increasing momentum in the recent days
        increasing_momentum = (recent_changes.abs().diff() > 0).iloc[1:].all()
        
        if increasing_momentum:
            direction = 'bullish' if recent_changes.iloc[-1] > 0 else 'bearish'
            
            # Calculate strength based on the magnitude of the shift
            momentum_magnitude = recent_changes.abs().sum() / prior_changes.abs().sum()
            volume_confirmation = prices['Volume'].iloc[-3:].mean() > prices['Volume'].iloc[-6:-3].mean()
            
            pattern_strength = min(0.6 + (momentum_magnitude * 0.2) + (0.2 if volume_confirmation else 0), 1.0)
            
            return (True, direction, pattern_strength)
            
        return None
