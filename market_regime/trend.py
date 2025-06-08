"""
Market Regime Analysis Module - Trend Analysis

This module provides the trend analysis component of the market regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any

def analyze_trend(data: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Analyzes the trend component of the market regime
    
    Args:
        data: DataFrame with OHLCV and technical indicators
        config: Configuration parameters
        
    Returns:
        Trend classification: 'strong_up', 'moderate_up', 'weak_up', 'neutral', 
                            'weak_down', 'moderate_down', 'strong_down'
    """
    if len(data) < config['trend']['sma_long']:
        return 'neutral'  # Not enough data
        
    # Get closing prices
    close = data['close'].values
    
    # Calculate moving averages if not already present
    if 'sma_short' not in data.columns:
        data['sma_short'] = data['close'].rolling(window=config['trend']['sma_short']).mean()
    if 'sma_medium' not in data.columns:
        data['sma_medium'] = data['close'].rolling(window=config['trend']['sma_medium']).mean()
    if 'sma_long' not in data.columns:
        data['sma_long'] = data['close'].rolling(window=config['trend']['sma_long']).mean()
    
    # Get latest values
    current_price = close[-1]
    sma_short = data['sma_short'].iloc[-1]
    sma_medium = data['sma_medium'].iloc[-1]
    sma_long = data['sma_long'].iloc[-1]
    
    # Calculate price changes
    price_vs_short = (current_price / sma_short) - 1
    price_vs_medium = (current_price / sma_medium) - 1
    price_vs_long = (current_price / sma_long) - 1
    
    # Average the price changes for a composite trend measure
    avg_trend_measure = (price_vs_short + price_vs_medium + price_vs_long) / 3
    
    # Classify trend
    strong_threshold = config['trend']['strong_threshold']
    moderate_threshold = config['trend']['moderate_threshold']
    weak_threshold = config['trend']['weak_threshold']
    
    if avg_trend_measure > strong_threshold:
        return 'strong_up'
    elif avg_trend_measure > moderate_threshold:
        return 'moderate_up'
    elif avg_trend_measure > weak_threshold:
        return 'weak_up'
    elif avg_trend_measure < -strong_threshold:
        return 'strong_down'
    elif avg_trend_measure < -moderate_threshold:
        return 'moderate_down'
    elif avg_trend_measure < -weak_threshold:
        return 'weak_down'
    else:
        return 'neutral'

def analyze_moving_average_slopes(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates the slopes of moving averages to determine trend direction and strength
    
    Args:
        data: DataFrame with OHLCV and technical indicators
        config: Configuration parameters
        
    Returns:
        Dictionary with slope values for different moving averages
    """
    # Ensure moving averages are calculated
    if 'sma_short' not in data.columns:
        data['sma_short'] = data['close'].rolling(window=config['trend']['sma_short']).mean()
    if 'sma_medium' not in data.columns:
        data['sma_medium'] = data['close'].rolling(window=config['trend']['sma_medium']).mean()
    if 'sma_long' not in data.columns:
        data['sma_long'] = data['close'].rolling(window=config['trend']['sma_long']).mean()
    
    # Calculate slopes (percentage change over the last 5 periods)
    slope_period = 5
    
    slopes = {}
    for ma_name in ['sma_short', 'sma_medium', 'sma_long']:
        if len(data) >= slope_period:
            current = data[ma_name].iloc[-1]
            previous = data[ma_name].iloc[-slope_period]
            slopes[ma_name] = (current / previous) - 1
        else:
            slopes[ma_name] = 0
            
    return slopes
