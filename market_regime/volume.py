"""
Market Regime Analysis Module - Volume Analysis

This module provides the volume analysis component of the market regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any

def analyze_volume(data: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Analyzes the volume component of the market regime
    
    Args:
        data: DataFrame with OHLCV and technical indicators
        config: Configuration parameters
        
    Returns:
        Volume classification: 'low', 'normal', 'high'
    """
    if len(data) < config['volume']['ma_period']:
        return 'normal'  # Not enough data
    
    # Get volume data
    volume = data['volume']
    
    # Calculate volume moving average if not already present
    if 'volume_ma' not in data.columns:
        data['volume_ma'] = volume.rolling(window=config['volume']['ma_period']).mean()
    
    # Get current values
    current_volume = volume.iloc[-1]
    volume_ma = data['volume_ma'].iloc[-1]
    
    # Calculate ratio of current volume to moving average
    volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
    
    # Classify volume
    if volume_ratio > config['volume']['high_vol_ratio']:
        return 'high'
    elif volume_ratio < config['volume']['low_vol_ratio']:
        return 'low'
    else:
        return 'normal'

def analyze_volume_trend(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the trend in volume
    
    Args:
        data: DataFrame with OHLCV and technical indicators
        config: Configuration parameters
        
    Returns:
        Dictionary with volume trend metrics
    """
    if len(data) < config['volume']['ma_period'] * 2:
        return {'trend': 'neutral', 'strength': 0.0}
    
    # Get volume data
    volume = data['volume']
    
    # Calculate short and long volume moving averages
    short_period = config['volume']['ma_period']
    long_period = short_period * 2
    
    vol_ma_short = volume.rolling(window=short_period).mean()
    vol_ma_long = volume.rolling(window=long_period).mean()
    
    # Get latest values
    current_vol_ma_short = vol_ma_short.iloc[-1]
    current_vol_ma_long = vol_ma_long.iloc[-1]
    
    # Calculate ratio of short MA to long MA
    ratio = current_vol_ma_short / current_vol_ma_long if current_vol_ma_long > 0 else 1.0
    
    # Determine trend
    if ratio > 1.2:
        trend = 'increasing'
    elif ratio < 0.8:
        trend = 'decreasing'
    else:
        trend = 'neutral'
    
    # Calculate strength (how far from 1.0)
    strength = abs(ratio - 1.0)
    
    return {
        'trend': trend,
        'strength': strength,
        'ratio': ratio
    }

def detect_volume_spikes(data: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detects significant volume spikes
    
    Args:
        data: DataFrame with OHLCV and technical indicators
        config: Configuration parameters
        
    Returns:
        List of dictionaries with volume spike information
    """
    if len(data) < config['volume']['ma_period'] * 2:
        return []
    
    # Get volume data
    volume = data['volume']
    
    # Calculate volume moving average
    vol_ma = volume.rolling(window=config['volume']['ma_period']).mean()
    
    # Calculate ratio of volume to moving average
    volume_ratio = volume / vol_ma
    
    # Detect spikes (volume more than 2x the moving average)
    spike_threshold = 2.0
    spike_indices = volume_ratio[volume_ratio > spike_threshold].index
    
    # Create list of spike information
    spikes = []
    for idx in spike_indices:
        if idx in data.index:
            spike_date = idx
            spike_volume = volume.loc[idx]
            spike_ratio = volume_ratio.loc[idx]
            
            # Determine if price moved up or down on spike day
            if idx > 0 and data.index[0] < idx:
                price_change = data['close'].loc[idx] / data['close'].shift().loc[idx] - 1
                price_direction = 'up' if price_change > 0 else 'down'
            else:
                price_direction = 'unknown'
            
            spikes.append({
                'date': spike_date,
                'volume': spike_volume,
                'ratio': spike_ratio,
                'price_direction': price_direction
            })
    
    return spikes
