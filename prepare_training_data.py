"""
Script to download and prepare comprehensive training data for Nifty 500 stocks.
Includes all technical indicators, market correlations, and relative strength metrics.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import concurrent.futures
import yfinance as yf
from tqdm import tqdm

from download_nifty500_data import download_stock_data
from technical_indicators import TechnicalIndicators
from data_collector import DataCollector
from market_metrics import calculate_market_correlations
from relative_strength import calculate_relative_strength_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_training_data.log')
    ]
)

logger = logging.getLogger(__name__)

def validate_stock_data(df: pd.DataFrame, symbol: str) -> bool:
    """Validate downloaded stock data for completeness and quality"""
    if df is None or df.empty:
        logger.error(f"{symbol}: Empty dataframe")
        return False
        
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"{symbol}: Missing required columns")
        return False
        
    # Check for sufficient data points (minimum 1 year)
    min_days = 252  # Trading days in a year
    if len(df) < min_days:
        logger.error(f"{symbol}: Insufficient data points ({len(df)} < {min_days})")
        return False
        
    # Check for excessive missing values
    missing_pct = df[required_cols].isnull().mean() * 100
    if any(missing_pct > 10):  # Allow max 10% missing values
        logger.error(f"{symbol}: Excessive missing values: {missing_pct[missing_pct > 10]}")
        return False
        
    # Basic data validation
    if df is None or df.empty:
        logger.error(f"{symbol}: Empty dataframe")
        return False
        
    # Check for required columns
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in price_cols):
        logger.error(f"{symbol}: Missing required columns")
        return False
        
    try:
        # Remove any rows with NaN values in price columns
        valid_data = df.dropna(subset=price_cols)
        
        # Check for minimum data points (1 year)
        if len(valid_data) < 252:
            logger.error(f"{symbol}: Insufficient valid trading days ({len(valid_data)} < 252)")
            return False
            
        # Check for positive prices
        for col in ['Open', 'High', 'Low', 'Close']:
            if (valid_data[col] <= 0).any():
                logger.error(f"{symbol}: Non-positive prices found in {col}")
                return False
                
        # Check for valid volume
        if (valid_data['Volume'] < 0).any():
            logger.error(f"{symbol}: Negative volume found")
            return False
            
        # Check price relationships
        price_errors = valid_data[
            (valid_data['High'] < valid_data['Low']) |
            (valid_data['Close'] > valid_data['High']) |
            (valid_data['Close'] < valid_data['Low']) |
            (valid_data['Open'] > valid_data['High']) |
            (valid_data['Open'] < valid_data['Low'])
        ]
        
        if not price_errors.empty:
            first_error = price_errors.iloc[0]
            logger.error(f"{symbol}: Invalid price relationships found on {first_error.name}")
            logger.error(f"O: {first_error['Open']:.2f}, H: {first_error['High']:.2f}, L: {first_error['Low']:.2f}, C: {first_error['Close']:.2f}")
            return False
            
    except Exception as e:
        logger.error(f"{symbol}: Error validating price data: {str(e)}")
        return False
        
    return True

def validate_indicators(df: pd.DataFrame, symbol: str) -> bool:
    """Validate calculated technical indicators"""
    # Check for key indicators
    key_indicators = [
        'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR',
        'SMA20', 'SMA50', 'SMA200', 'VWAP'
    ]
    
    missing_indicators = [ind for ind in key_indicators if ind not in df.columns]
    if missing_indicators:
        logger.error(f"{symbol}: Missing indicators: {missing_indicators}")
        return False
        
    # Validate indicator ranges
    if 'RSI' in df.columns and not df['RSI'].between(0, 100).all():
        logger.error(f"{symbol}: RSI values out of range")
        return False
        
    # Check for excessive NaN values in indicators
    indicator_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    missing_pct = df[indicator_cols].isnull().mean() * 100
    if any(missing_pct > 20):  # Allow more NaN values for indicators due to lookback periods
        logger.error(f"{symbol}: Excessive missing values in indicators: {missing_pct[missing_pct > 20]}")
        return False
        
    return True

class TrainingDataPreparator:
    def __init__(self):
        self.data_collector = DataCollector()
        self.indicators = TechnicalIndicators()
        
        # Create data directories
        self.base_path = Path("data/training")
        self.raw_data_path = self.base_path / "raw"
        self.processed_data_path = self.base_path / "processed"
        
        for path in [self.base_path, self.raw_data_path, self.processed_data_path]:
            path.mkdir(parents=True, exist_ok=True)
            
    def get_nifty500_symbols(self) -> List[str]:
        """Get list of Nifty 500 symbols"""
        return self.data_collector._get_nifty500_symbols()
    
    def download_market_data(self) -> pd.DataFrame:
        """Download market indices data for 5 years"""
        indices = [
            '^NSEI',  # NIFTY 50
            '^BSESN',  # BSE SENSEX
            '^NSEBANK'  # NIFTY BANK
        ]
        
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Download all indices at once
        try:
            data = yf.download(
                indices,
                start=start_date,
                end=end_date,
                progress=False
            )['Close']  # Get only closing prices
            
            # Rename columns to remove the ^ prefix
            data.columns = [index.replace('^', '') for index in indices]
            return data
            
        except Exception as e:
            logger.error(f"Error downloading market data: {e}")
            return pd.DataFrame()
    
    def download_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download and process data for a single stock"""
        try:
            # Download with auto-adjust and back-adjust using both split and dividend info
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,  # Adjust all OHLC automatically
                back_adjust=True,  # Back-adjust data to match desired timeframe
                actions=True  # Include dividends and splits
            )
            
            if data is None or data.empty:
                logger.error(f"No data returned for {symbol}")
                return None
                
            # Validate the downloaded data
            if not validate_stock_data(data, symbol):
                return None
                
            # Save raw data
            data.to_parquet(self.raw_data_path / f"{symbol.replace('.', '_')}_raw.parquet")
            return data
                
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return None
    
    def process_stock_data(self, symbol: str, df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Process stock data with all required metrics"""
        try:
            # Calculate technical indicators
            df = self.indicators.calculate_all(df.copy())
            
            # Validate calculated indicators
            if not validate_indicators(df, symbol):
                return None
            
            # Calculate market correlations
            correlations = calculate_market_correlations(df['Close'], market_data)
            for market, corr in correlations.items():
                df[f'CORR_{market}'] = corr
            
            # Calculate relative strength metrics
            rs_metrics = calculate_relative_strength_metrics(df['Close'], market_data)
            for metric, value in rs_metrics.items():
                if isinstance(value, (int, float)):
                    df[metric] = value
                elif isinstance(value, pd.Series):
                    df[metric] = value
            
            # Add derived features
            df['VOL_MA_RATIO'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['CLOSE_MA_RATIO'] = df['Close'] / df['Close'].rolling(50).mean()
            df['HIGH_LOW_RANGE'] = (df['High'] - df['Low']) / df['Close']
            
            # Calculate returns for different horizons
            for days in [1, 5, 10, 20]:
                df[f'RETURN_{days}D'] = df['Close'].pct_change(days)
            
            # Save processed data
            df.to_parquet(self.processed_data_path / f"{symbol.replace('.', '_')}_processed.parquet")
            return df
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None

    def prepare_training_data(self):
        """Download and prepare training data for all Nifty 500 stocks"""
        # Get Nifty 500 symbols
        symbols = self.get_nifty500_symbols()
        if not symbols:
            logger.error("No symbols found")
            return
        
        logger.info(f"Found {len(symbols)} symbols")
        
        # Download market data first
        logger.info("Downloading market data...")
        market_data = self.download_market_data()
        if market_data.empty:
            logger.error("Failed to download market data")
            return
        
        # Save market data
        market_data.to_parquet(self.base_path / "market_data.parquet")
        logger.info(f"Saved market data with {len(market_data.columns)} indices")
        
        # Calculate dates for 5 years of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        
        # Download and process stock data
        success_count = 0
        failed_symbols = []
        
        logger.info("Downloading and processing stock data...")
        for symbol in tqdm(symbols):
            try:
                # Download data
                df = self.download_stock_data(symbol, start_date, end_date)
                if df is None or df.empty:
                    failed_symbols.append(symbol)
                    continue
                
                # Process data
                processed_df = self.process_stock_data(symbol, df, market_data)
                if processed_df is not None:
                    success_count += 1
                    logger.info(f"Successfully processed {symbol}")
                else:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                failed_symbols.append(symbol)
        
        # Save failed symbols for retry
        if failed_symbols:
            with open(self.base_path / "failed_downloads.txt", "w") as f:
                f.write("\n".join(failed_symbols))
        
        logger.info(f"Completed data preparation:")
        logger.info(f"Successfully processed: {success_count}")
        logger.info(f"Failed: {len(failed_symbols)}")
        if failed_symbols:
            logger.info("Failed symbols saved to failed_downloads.txt")
def main():
    """Main function to run the training data preparation"""
    logger.info("Starting training data preparation")
    
    preparator = TrainingDataPreparator()
    preparator.prepare_training_data()
    
    logger.info("Training data preparation completed")

if __name__ == "__main__":
    main()
