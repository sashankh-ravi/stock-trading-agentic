#!/usr/bin/env python3
"""
Clean Data Downloader for Nifty 500 Trading System

This script provides a clean and simple interface for downloading all necessary data
for the Nifty 500 trading system. It processes stocks in small batches with appropriate
delays to avoid rate limiting issues.

No try-except blocks are used to ensure all errors are visible.
"""

import os
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import functions from core modules
from download_nifty500_data import get_nifty500_symbols
from download_stock_data import download_stock_data

# Constants
DATA_DIR = Path("data")
HISTORICAL_DIR = DATA_DIR / "historical"
FUNDAMENTAL_DIR = DATA_DIR / "fundamental"
NEWS_DIR = DATA_DIR / "news"

# Create directories
for directory in [DATA_DIR, HISTORICAL_DIR, FUNDAMENTAL_DIR, NEWS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def download_historical_data(symbols, period='5y', batch_size=5, delay=5, batch_delay=30):
    """
    Download historical price data for all symbols in small batches with delays.
    
    Args:
        symbols: List of stock symbols
        period: Time period to download data for
        batch_size: Number of symbols to process in each batch
        delay: Delay between individual downloads (seconds)
        batch_delay: Delay between batches (seconds)
    """
    logger.info(f"Starting historical data download for {len(symbols)} symbols")
    
    total_batches = (len(symbols) + batch_size - 1) // batch_size
    successful = 0
    failed = 0
    
    for batch_idx in range(0, len(symbols), batch_size):
        batch = symbols[batch_idx:batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1
        
        logger.info(f"Processing batch {current_batch}/{total_batches}")
        
        for i, symbol in enumerate(batch):
            logger.info(f"Downloading historical data for {symbol}")
            
            # Skip if file already exists
            output_file = HISTORICAL_DIR / f"{symbol.replace('.NS', '')}_5y.parquet"
            if output_file.exists():
                logger.info(f"File already exists for {symbol}, skipping")
                successful += 1
                continue
            
            # Download data
            data = download_stock_data(symbol, period=period)
            
            if data is not None and not data.empty:
                # Save data
                data.to_parquet(output_file)
                logger.info(f"Successfully downloaded data for {symbol}")
                successful += 1
            else:
                logger.warning(f"Failed to download data for {symbol}, continuing to the next symbol")
                failed += 1
            
            # Add delay between downloads
            if i < len(batch) - 1:
                logger.info(f"Waiting {delay} seconds before next download...")
                time.sleep(delay)
        
        # Add delay between batches
        if current_batch < total_batches:
            logger.info(f"Batch {current_batch} complete. Waiting {batch_delay} seconds before next batch...")
            time.sleep(batch_delay)
    
    logger.info(f"Historical data download complete. Success: {successful}, Failed: {failed}")
    return successful, failed

def create_minimal_fundamental_data(symbol, sector_map, industry_map):
    """Create minimal fundamental data file for a symbol."""
    output_file = FUNDAMENTAL_DIR / f"{symbol.replace('.NS', '')}_fundamental.json"
    
    # Skip if file already exists
    if output_file.exists():
        logger.info(f"Fundamental data already exists for {symbol}, skipping")
        return True
    
    # Create basic data structure
    data = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "longName": symbol,
        "sector": sector_map.get(symbol.replace('.NS', ''), "Unknown"),
        "industry": industry_map.get(symbol.replace('.NS', ''), "Unknown"),
        "source": "manually_created"
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        import json
        json.dump(data, f, indent=2)
    
    logger.info(f"Created minimal fundamental data for {symbol}")
    return True

def download_all_data():
    """Download all necessary data for the trading system."""
    logger.info("=== Starting Complete Data Download ===")
    
    # Get Nifty 500 symbols
    logger.info("Getting Nifty 500 symbols")
    symbols, sector_map, industry_map = get_nifty500_symbols()
    
    # Download historical data
    download_historical_data(symbols)
    
    # Create fundamental data
    logger.info("Creating fundamental data files")
    for symbol in symbols:
        create_minimal_fundamental_data(symbol, sector_map, industry_map)
    
    logger.info("=== Data Download Complete ===")

if __name__ == "__main__":
    download_all_data()
