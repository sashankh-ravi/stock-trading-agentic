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
    try:
        # Handle None, empty Series, or Series with only NaN values
        if (stock_data is None or market_data is None or
            not isinstance(stock_data, pd.Series) or not isinstance(market_data, pd.Series) or
            len(stock_data) == 0 or len(market_data) == 0 or
            stock_data.isna().all() or market_data.isna().all()):
            return 0.0
        
        # Ensure data is properly aligned
        stock_aligned, market_aligned = standardize_periods(stock_data, market_data)
        
        # Check for minimum data requirement
        if len(stock_aligned) < min_periods:
            return 0.0
        
        # Check for flat market (to avoid division by zero)
        if market_aligned.std() == 0 or stock_aligned.std() == 0:
            return 0.0
        
        # Calculate relative performance
        relative_perf = (stock_aligned.iloc[-1] / stock_aligned.iloc[0]) / (market_aligned.iloc[-1] / market_aligned.iloc[0]) - 1
        
        # Scale and bound the result
        rs = float(relative_perf * 100)
        return max(min(rs, 10.0), -10.0)  # Bound between -10 and 10
        
    except Exception as e:
        logger.error(f"Error in calculate_relative_strength_cumulative: {str(e)}")
        return 0.0

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
    try:
        # Handle edge cases
        if (stock_data is None or market_data is None or
            not isinstance(stock_data, pd.Series) or not isinstance(market_data, pd.Series) or
            len(stock_data) == 0 or len(market_data) == 0 or
            stock_data.isna().all() or market_data.isna().all()):
            return pd.Series([0.0])
        
        # Ensure data is properly aligned
        stock_aligned, market_aligned = standardize_periods(stock_data, market_data)
        
        # Calculate returns
        stock_returns = stock_aligned.pct_change()
        market_returns = market_aligned.pct_change()
        
        # Calculate rolling means with minimum periods
        min_periods = max(2, window // 2)  # Ensure at least 2 periods for mean calculation
        stock_mean = stock_returns.rolling(window=window, min_periods=min_periods).mean()
        market_mean = market_returns.rolling(window=window, min_periods=min_periods).mean()
        
        # Calculate relative strength
        rs = (stock_mean - market_mean) * 100
        
        # Bound the values
        return rs.clip(-10, 10)  # Bound between -10 and 10
        
    except Exception as e:
        logger.error(f"Error in calculate_relative_strength_rolling: {str(e)}")
        return pd.Series([0.0])

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
            'roll': 63,  # ~3 months
            'mom': 252   # ~1 year
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
        try:
            # Standardize periods for calculations
            stock_aligned, market_aligned = standardize_periods(stock_data, market_series)
            
            # Calculate cumulative RS
            cum_rs = calculate_relative_strength_cumulative(stock_aligned, market_aligned)
            result[f'RS_CUM_{market_name}'] = float(cum_rs)
            
            # Calculate rolling RS
            roll_rs = calculate_relative_strength_rolling(stock_aligned, market_aligned, window=window_sizes['roll'])
            result[f'RS_ROLL_{market_name}'] = float(roll_rs.iloc[-1])
            
            # Calculate momentum RS
            mom_rs = calculate_relative_strength_rolling(stock_aligned, market_aligned, window=window_sizes['mom'])
            result[f'RS_MOM_{market_name}'] = float(mom_rs.iloc[-1])
            
        except Exception as e:
            logger.error(f"Error calculating RS metrics for {market_name}: {str(e)}")
            result[f'RS_CUM_{market_name}'] = 0.0
            result[f'RS_ROLL_{market_name}'] = 0.0
            result[f'RS_MOM_{market_name}'] = 0.0
    
    return result
