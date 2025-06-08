"""
Market Regime Analysis Module - Volatility Analysis

This module provides the volatility analysis component of the market regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any

def analyze_volatility(data: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Analyzes the volatility component of the market regime
    
    Args:
        data: DataFrame with OHLCV and technical indicators
        config: Configuration parameters
        
    Returns:
        Volatility classification: 'low', 'normal', 'high'
    """
    if len(data) < config['volatility']['atr_period']:
        return 'normal'  # Not enough data
    
    # Calculate ATR if not already present
    if 'atr' not in data.columns:
        data['atr'] = calculate_atr(data, config['volatility']['atr_period'])
    
    # Calculate Bollinger Band width if not already present
    if 'bb_width' not in data.columns:
        data['bb_width'] = calculate_bb_width(data, config['volatility']['bb_period'])
    
    # Get current values
    current_atr = data['atr'].iloc[-1]
    current_bb_width = data['bb_width'].iloc[-1]
    
    # Get historical percentiles
    atr_history = data['atr'].dropna()
    bb_history = data['bb_width'].dropna()
    
    if len(atr_history) < 20 or len(bb_history) < 20:
        return 'normal'  # Not enough historical data for reliable percentiles
    
    atr_percentile = percentile_rank(current_atr, atr_history)
    bb_percentile = percentile_rank(current_bb_width, bb_history)
    
    # Average the two percentiles
    avg_volatility_percentile = (atr_percentile + bb_percentile) / 2
    
    # Classify volatility
    if avg_volatility_percentile > config['volatility']['high_vol_percentile']:
        return 'high'
    elif avg_volatility_percentile < config['volatility']['low_vol_percentile']:
        return 'low'
    else:
        return 'normal'

def calculate_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        data: DataFrame with OHLCV data
        period: ATR period
        
    Returns:
        Series with ATR values
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_bb_width(data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Bollinger Band width
    
    Args:
        data: DataFrame with OHLCV data
        period: Bollinger Band period
        
    Returns:
        Series with Bollinger Band width values
    """
    # Calculate Bollinger Bands
    middle = data['close'].rolling(window=period).mean()
    std_dev = data['close'].rolling(window=period).std()
    
    upper = middle + (2 * std_dev)
    lower = middle - (2 * std_dev)
    
    # Calculate width as percentage of middle band
    bb_width = (upper - lower) / middle
    
    return bb_width

def percentile_rank(value: float, series: pd.Series) -> float:
    """
    Calculate the percentile rank of a value within a series
    
    Args:
        value: Value to find percentile for
        series: Series of historical values
        
    Returns:
        Percentile rank (0-100)
    """
    n_smaller = (series < value).sum()
    n_equal = (series == value).sum()
    n = len(series)
    
    if n == 0:
        return 50.0  # Default to middle if no data
    
    # Calculate percentile rank
    percentile = 100 * (n_smaller + 0.5 * n_equal) / n
    
    return percentile
