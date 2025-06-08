"""
Market Regime Analysis Module - Momentum Analysis

This module provides the momentum analysis component of the market regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any

def analyze_momentum(data: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Analyzes the momentum component of the market regime
    
    Args:
        data: DataFrame with OHLCV and technical indicators
        config: Configuration parameters
        
    Returns:
        Momentum classification: 'positive', 'neutral', 'negative'
    """
    if len(data) < max(config['momentum']['rsi_period'], config['momentum']['macd_slow']):
        return 'neutral'  # Not enough data
    
    # Calculate RSI if not already present
    if 'rsi' not in data.columns:
        data['rsi'] = calculate_rsi(data, config['momentum']['rsi_period'])
    
    # Calculate MACD if not already present
    if 'macd' not in data.columns or 'macd_signal' not in data.columns:
        macd_data = calculate_macd(
            data, 
            config['momentum']['macd_fast'], 
            config['momentum']['macd_slow'], 
            config['momentum']['macd_signal']
        )
        data['macd'] = macd_data['macd']
        data['macd_signal'] = macd_data['signal']
        data['macd_hist'] = macd_data['histogram']
    
    # Get current values
    current_rsi = data['rsi'].iloc[-1]
    current_macd = data['macd'].iloc[-1]
    current_macd_signal = data['macd_signal'].iloc[-1]
    current_macd_hist = data['macd_hist'].iloc[-1]
    
    # Classify momentum based on RSI
    if current_rsi > config['momentum']['rsi_high']:
        rsi_signal = 'positive'
    elif current_rsi < config['momentum']['rsi_low']:
        rsi_signal = 'negative'
    else:
        rsi_signal = 'neutral'
    
    # Classify momentum based on MACD
    if current_macd > current_macd_signal and current_macd > 0:
        macd_signal = 'positive'
    elif current_macd < current_macd_signal and current_macd < 0:
        macd_signal = 'negative'
    else:
        macd_signal = 'neutral'
    
    # Combine signals (simple majority)
    signals = [rsi_signal, macd_signal]
    positive_count = signals.count('positive')
    negative_count = signals.count('negative')
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

def calculate_rsi(data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data: DataFrame with OHLCV data
        period: RSI period
        
    Returns:
        Series with RSI values
    """
    close = data['close']
    delta = close.diff()
    
    # Get gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> Dict[str, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        data: DataFrame with OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Dictionary with MACD, signal line, and histogram
    """
    close = data['close']
    
    # Calculate EMAs
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd = ema_fast - ema_slow
    
    # Calculate signal line
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd - signal
    
    return {
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    }

def analyze_momentum_oscillators(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyzes multiple momentum oscillators for a comprehensive view
    
    Args:
        data: DataFrame with OHLCV and technical indicators
        config: Configuration parameters
        
    Returns:
        Dictionary with momentum oscillator values
    """
    results = {}
    
    # Calculate RSI if not already present
    if 'rsi' not in data.columns:
        data['rsi'] = calculate_rsi(data, config['momentum']['rsi_period'])
    
    # Get current RSI
    results['rsi'] = data['rsi'].iloc[-1] if not data['rsi'].empty else 50.0
    
    # Calculate Rate of Change (ROC)
    periods = [5, 10, 20]
    for period in periods:
        roc_name = f'roc_{period}'
        if roc_name not in data.columns:
            data[roc_name] = (data['close'] / data['close'].shift(period) - 1) * 100
        
        results[roc_name] = data[roc_name].iloc[-1] if not data[roc_name].empty else 0.0
    
    # Calculate stochastic oscillator
    if 'stoch_k' not in data.columns or 'stoch_d' not in data.columns:
        stoch_period = 14
        k_period = 3
        d_period = 3
        
        # Calculate %K
        low_min = data['low'].rolling(window=stoch_period).min()
        high_max = data['high'].rolling(window=stoch_period).max()
        
        data['stoch_k'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D
        data['stoch_d'] = data['stoch_k'].rolling(window=d_period).mean()
    
    results['stoch_k'] = data['stoch_k'].iloc[-1] if not data['stoch_k'].empty else 50.0
    results['stoch_d'] = data['stoch_d'].iloc[-1] if not data['stoch_d'].empty else 50.0
    
    return results
