"""
Relative Strength calculation module.
Contains functions for calculating various relative strength metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

def standardize_periods(stock_data: pd.Series, market_data: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align stock and market data to ensure consistent dates and handle any missing values.
    
    Parameters:
    -----------
    stock_data : pd.Series
        Stock price data
    market_data : pd.Series
        Market index data
        
    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        Aligned stock and market data
        
    Raises:
    -------
    ValueError
        If either input is None, empty, or contains only NaN values
    """
    # Input validation
    if stock_data is None or market_data is None:
        raise ValueError("Input data cannot be None")
    
    if not isinstance(stock_data, pd.Series) or not isinstance(market_data, pd.Series):
        raise ValueError("Inputs must be pandas Series")
    
    if len(stock_data) == 0 or len(market_data) == 0:
        raise ValueError("Input data cannot be empty")
    
    if stock_data.isna().all() or market_data.isna().all():
        raise ValueError("Input data cannot contain only NaN values")
    
    # Ensure data is sorted by date
    stock_data = stock_data.sort_index()
    market_data = market_data.sort_index()
    
    # Find common date range
    common_dates = stock_data.index.intersection(market_data.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between stock and market data")
        
    # Align data
    stock_aligned = stock_data.loc[common_dates]
    market_aligned = market_data.loc[common_dates]
    
    # Handle missing values
    valid_data = ~(stock_aligned.isna() | market_aligned.isna())
    if not valid_data.any():
        raise ValueError("No valid data points after alignment")
    
    return stock_aligned[valid_data], market_aligned[valid_data]

def calculate_relative_strength_cumulative(
    stock_data: pd.Series,
    market_data: pd.Series,
    min_periods: int = 20
) -> float:
    """
    Calculate cumulative relative strength for a stock against a market index.
    
    Parameters:
    -----------
    stock_data : pd.Series
        Stock price data
    market_data : pd.Series
        Market index data
    min_periods : int
        Minimum number of periods required for calculation
        
    Returns:
    --------
    float
        Cumulative relative strength score
    """
    # Handle None, empty Series, or Series with only NaN values
    if (stock_data is None or market_data is None or
        not isinstance(stock_data, pd.Series) or not isinstance(market_data, pd.Series) or
        len(stock_data) == 0 or len(market_data) == 0 or
        stock_data.isna().all() or market_data.isna().all()):
        logger.warning("Invalid input data for RS calculation")
        return 0.0
    
    # Ensure data is properly aligned
    stock_aligned, market_aligned = standardize_periods(stock_data, market_data)
    
    # Check for minimum data requirement
    if len(stock_aligned) < min_periods:
        logger.warning(f"Insufficient data points: {len(stock_aligned)} < {min_periods}")
        return 0.0
    
    # Check for flat market (to avoid division by zero)
    if market_aligned.std() == 0 or stock_aligned.std() == 0:
        logger.warning("Zero standard deviation detected in data")
        return 0.0
    
    # Calculate normalized stock and market performance
    stock_perf = stock_aligned.iloc[-1] / stock_aligned.iloc[0]
    market_perf = market_aligned.iloc[-1] / market_aligned.iloc[0]
    
    logger.info(f"Stock performance: {stock_perf:.4f}")
    logger.info(f"Market performance: {market_perf:.4f}")
    
    # Calculate relative performance
    relative_perf = (stock_perf / market_perf) - 1
    
    logger.info(f"Raw relative performance: {relative_perf:.4f}")
    
    # Scale and bound the result
    rs = float(relative_perf * 100)
    bounded_rs = max(min(rs, 10.0), -10.0)  # Bound between -10 and 10
    
    logger.info(f"Final RS value: {bounded_rs:.4f}")
    return bounded_rs

def calculate_relative_strength_rolling(
    stock_data: pd.Series,
    market_data: pd.Series,
    window: int = 63
) -> pd.Series:
    """
    Calculate rolling relative strength for a stock against a market index.
    
    Parameters:
    -----------
    stock_data : pd.Series
        Stock price data
    market_data : pd.Series
        Market index data
    window : int
        Rolling window size in days
        
    Returns:
    --------
    pd.Series
        Series of rolling relative strength values
    """
    # Handle edge cases
    if (stock_data is None or market_data is None or
        not isinstance(stock_data, pd.Series) or not isinstance(market_data, pd.Series) or
        len(stock_data) == 0 or len(market_data) == 0 or
        stock_data.isna().all() or market_data.isna().all()):
        return pd.Series([0.0])
    
    # Ensure data is properly aligned
    stock_aligned, market_aligned = standardize_periods(stock_data, market_data)
    
    if len(stock_aligned) < window:
        logger.warning(f"Insufficient data points for rolling calculation: {len(stock_aligned)} < {window}")
        return pd.Series([0.0])
    
    # Calculate returns
    stock_returns = stock_aligned.pct_change()
    market_returns = market_aligned.pct_change()
    
    # Calculate rolling performance ratio
    rolling_stock = (1 + stock_returns).rolling(window=window).apply(lambda x: x.prod())
    rolling_market = (1 + market_returns).rolling(window=window).apply(lambda x: x.prod())
    
    # Calculate relative strength
    rs = (rolling_stock / rolling_market - 1) * 100
    
    # Bound the values
    rs = rs.clip(-10, 10)  # Bound between -10 and 10
    
    logger.info(f"Rolling RS range: {rs.min():.4f} to {rs.max():.4f}")
    return rs

def calculate_relative_strength_metrics(
    stock_data: pd.Series,
    market_data: Union[pd.Series, pd.DataFrame],
    window_sizes: Optional[Dict[str, int]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive relative strength metrics for a stock against market indices.
    
    Parameters:
    -----------
    stock_data : pd.Series
        Stock price data
    market_data : Union[pd.Series, pd.DataFrame]
        Market index data. Can be either a Series for a single index or DataFrame for multiple indices
    window_sizes : Optional[Dict[str, int]]
        Dictionary of window sizes for different calculations
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing relative strength metrics as float values
    """
    if window_sizes is None:
        window_sizes = {
            'roll': 63,     # ~3 months
            'mom': 252,     # ~1 year
            'short': 21,    # ~1 month
            'medium': 126,  # ~6 months
            'long': 252     # ~1 year
        }
    
    # Input validation with early return for edge cases
    if stock_data is None or market_data is None:
        return {
            'RS_CUM_MARKET': 0.0,
            'RS_ROLL_MARKET': 0.0,
            'RS_MOM_MARKET': 0.0
        }
    
    result = {}
    
    # Handle Series input
    if isinstance(market_data, pd.Series):
        market_name = "MARKET"
        market_data_dict = {market_name: market_data}
    else:
        market_data_dict = {col: market_data[col] for col in market_data.columns}
    
    # Calculate metrics for each market index
    for market_name, market_series in market_data_dict.items():
        # Standardize periods for calculations
        stock_aligned, market_aligned = standardize_periods(stock_data, market_series)
        
        # Calculate returns
        stock_returns = stock_aligned.pct_change()
        market_returns = market_aligned.pct_change()
        
        # 1. Basic RS Metrics
        # Cumulative RS
        cum_rs = calculate_relative_strength_cumulative(stock_aligned, market_aligned)
        result[f'RS_CUM_{market_name}'] = float(cum_rs)
        
        # Rolling RS
        roll_rs = calculate_relative_strength_rolling(stock_aligned, market_aligned, window=window_sizes['roll'])
        result[f'RS_ROLL_{market_name}'] = float(roll_rs.iloc[-1])
        
        # Momentum RS
        mom_rs = calculate_relative_strength_rolling(stock_aligned, market_aligned, window=window_sizes['mom'])
        result[f'RS_MOM_{market_name}'] = float(mom_rs.iloc[-1])
        
        # 2. Enhanced RS Metrics
        # Multiple timeframe RS
        for period, window in [('SHORT', 'short'), ('MEDIUM', 'medium'), ('LONG', 'long')]:
            rs = calculate_relative_strength_rolling(stock_aligned, market_aligned, window=window_sizes[window])
            result[f'RS_{period}_{market_name}'] = float(rs.iloc[-1])
        
        # 3. RS Trend Analysis
        # Calculate RS slope (trend direction and strength)
        rs_line = (stock_aligned / stock_aligned.iloc[0]) / (market_aligned / market_aligned.iloc[0])
        rs_slope = np.polyfit(np.arange(len(rs_line[-20:])), rs_line[-20:], 1)[0]
        result[f'RS_TREND_{market_name}'] = float(rs_slope)
        
        # 4. RS Volatility
        # Calculate how stable the RS relationship is
        rs_std = roll_rs.std()
        result[f'RS_VOLATILITY_{market_name}'] = float(rs_std)
        
        # 5. RS Mean Reversion
        # How far current RS is from its moving average
        rs_ma = roll_rs.rolling(window=20).mean()
        if not rs_ma.empty:
            rs_deviation = (roll_rs.iloc[-1] - rs_ma.iloc[-1]) / rs_ma.iloc[-1]
            result[f'RS_DEVIATION_{market_name}'] = float(rs_deviation)
        
        # 6. RS Momentum Factors
        # Rate of change in RS
        rs_roc = roll_rs.pct_change(periods=5)  # 5-day rate of change
        result[f'RS_ROC_{market_name}'] = float(rs_roc.iloc[-1])
        
        # RS Acceleration
        rs_roc_change = rs_roc.diff()
        result[f'RS_ACCELERATION_{market_name}'] = float(rs_roc_change.iloc[-1])
        
        # 7. RS Regime Detection
        # Determine if RS is in strong/weak regime
        rs_percentile = pd.Series(roll_rs).rank(pct=True).iloc[-1]
        result[f'RS_REGIME_{market_name}'] = float(rs_percentile)
    
    return result
