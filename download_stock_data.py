"""
Stock data download module for the Nifty 500 Trading System.
This module provides functions for downloading and processing stock data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def download_stock_data(
    symbol: str, 
    period: str = '5y', 
    interval: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    auto_adjust: bool = True,
    prepost: bool = False,
    progress: bool = False,
    retry_count: int = 3,
    retry_delay: int = 5,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Download stock data with retry logic and caching.
    
    Args:
        symbol: Stock ticker symbol
        period: Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        start_date: Start date string in 'YYYY-MM-DD' format
        end_date: End date string in 'YYYY-MM-DD' format
        auto_adjust: Adjust all OHLC automatically
        prepost: Include Pre and Post market data
        progress: Display progress bar
        retry_count: Number of retry attempts
        retry_delay: Delay between retries in seconds
        cache_dir: Directory to cache downloaded data
        
    Returns:
        DataFrame with stock data
    """
    # Check if cached file exists
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        file_suffix = f"_{period}" if not (start_date and end_date) else f"_{start_date}_{end_date}"
        cache_file = cache_path / f"{symbol}{file_suffix}.parquet"
        
        if cache_file.exists():
            # Removed try-except to expose actual errors
            data = pd.read_parquet(cache_file)
            logger.info(f"Loaded cached data for {symbol}")
            return data
    
    # Download data with retries
    attempts = 0
    while attempts < retry_count:
        # Removed try-except to expose actual errors
        if start_date and end_date:
            data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=prepost,
                progress=progress
            )
        else:
            data = yf.download(
                symbol, 
                period=period, 
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=prepost,
                progress=progress
            )
            
        if data.empty:
            logger.warning(f"No data found for {symbol}")
            attempts += 1
            time.sleep(retry_delay)
            continue
            
        # Save to cache if needed
        if cache_dir and cache_file:
            # Removed try-except to expose actual errors
            data.to_parquet(cache_file)
            logger.info(f"Cached data for {symbol}")
                
        return data
        
    logger.error(f"Failed to download {symbol} after {retry_count} attempts")
    return pd.DataFrame()
