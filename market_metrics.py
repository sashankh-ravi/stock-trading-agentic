"""
Market Data Utilities

Functions for calculating market correlations and relative strength metrics
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_market_correlations(stock_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
    """Calculate correlations between stock and market indices"""
    correlations = {}
    
    if 'Close' not in stock_data.columns:
        return pd.Series(correlations)
        
    stock_returns = stock_data['Close'].pct_change(fill_method=None)
    
    for market_name in market_data.columns:
        try:
            market_series = market_data[market_name]
            
            # Skip if market data is missing
            if market_series.empty or market_series.isna().all():
                correlations[f'CORR_{market_name}'] = np.nan
                continue
                
            market_returns = market_series.pct_change(fill_method=None)
            
            # Align the series on their index (dates)
            common_dates = stock_returns.index.intersection(market_returns.index)
            if len(common_dates) > 1:
                # Remove NaN values before calculating correlation
                stock_clean = stock_returns[common_dates].dropna()
                market_clean = market_returns[common_dates].dropna()
                
                # Get common dates again after dropping NaN values
                final_dates = stock_clean.index.intersection(market_clean.index)
                
                if len(final_dates) > 1:
                    correlation = stock_clean[final_dates].corr(market_clean[final_dates])
                    correlations[f'CORR_{market_name}'] = correlation
                else:
                    correlations[f'CORR_{market_name}'] = np.nan
            else:
                correlations[f'CORR_{market_name}'] = np.nan
                
        except Exception as e:
            logger.warning(f"Error calculating correlation with {market_name}: {str(e)}")
            correlations[f'CORR_{market_name}'] = np.nan
            
    return pd.Series(correlations)

def calculate_relative_strength(stock_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
    """Calculate relative strength comparing stock performance to market indices"""
    rs_metrics = {}
    
    if 'Close' not in stock_data.columns:
        return pd.Series(rs_metrics)
        
    stock_returns = stock_data['Close'].pct_change(fill_method=None)
    
    for market_name in market_data.columns:
        try:
            market_series = market_data[market_name]
            
            # Skip if market data is missing
            if market_series.empty or market_series.isna().all():
                rs_metrics[f'RS_{market_name}'] = np.nan
                continue
                
            market_returns = market_series.pct_change(fill_method=None)
            
            # Align dates and remove NaN values
            common_dates = stock_returns.index.intersection(market_returns.index)
            if len(common_dates) > 1:
                stock_clean = stock_returns[common_dates].dropna()
                market_clean = market_returns[common_dates].dropna()
                
                # Get common dates again after dropping NaN values
                final_dates = stock_clean.index.intersection(market_clean.index)
                
                if len(final_dates) > 1:
                    # Calculate daily returns
                    stock_pct = stock_clean[final_dates].pct_change(fill_method=None)
                    benchmark_pct = market_clean[final_dates].pct_change(fill_method=None)
                    
                    # Calculate cumulative returns
                    stock_cum = (1 + stock_pct.fillna(0)).cumprod()
                    bench_cum = (1 + benchmark_pct.fillna(0)).cumprod()
                    
                    # Set starting point to 1
                    stock_cum = stock_cum / stock_cum.iloc[0] if not stock_cum.empty else pd.Series(index=stock_returns.index)
                    bench_cum = bench_cum / bench_cum.iloc[0] if not bench_cum.empty else pd.Series(index=stock_returns.index)
                    
                    # Calculate relative strength ratio
                    rs = stock_cum / bench_cum.replace(0, np.nan)
                    
                    # Fill any missing values with previous values
                    rs = rs.fillna(method='ffill')
                    rs_metrics[f'RS_{market_name}'] = rs.iloc[-1]  # Get latest value
                else:
                    rs_metrics[f'RS_{market_name}'] = np.nan
            else:
                rs_metrics[f'RS_{market_name}'] = np.nan
                
        except Exception as e:
            logger.warning(f"Error calculating relative strength for {market_name}: {str(e)}")
            rs_metrics[f'RS_{market_name}'] = np.nan
            
    return pd.Series(rs_metrics)
