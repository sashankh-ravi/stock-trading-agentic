"""
Market Context Module

Handles market-wide analysis, correlations, and regime detection.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Market indices to track
GLOBAL_INDICES = {
    # Indian Indices
    '^NSEI': 'Nifty 50',
    '^BSESN': 'Sensex',
    '^NSMIDCP': 'Nifty Midcap',
    # Global Indices
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones',
    '^IXIC': 'Nasdaq'
}

# Commodities and fixed income
MARKET_INSTRUMENTS = {
    'GC=F': 'Gold',
    'CL=F': 'Crude Oil',
    'NG=F': 'Natural Gas',
    '^TNX': '10-yr Treasury'
}

def download_market_data(start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> pd.DataFrame:
    """Download market data for correlation analysis"""
    market_data = {}
    
    # Download indices
    for symbol, name in GLOBAL_INDICES.items():
        df = yf.download(symbol, start=start_date, end=end_date)
        if not df.empty:
            market_data[name] = df['Close']
    
    # Download commodities/bonds
    for symbol, name in MARKET_INSTRUMENTS.items():
        df = yf.download(symbol, start=start_date, end=end_date)
        if not df.empty:
            market_data[name] = df['Close']
    
    return pd.DataFrame(market_data)

def calculate_correlations(price_data: pd.DataFrame, 
                         market_data: pd.DataFrame,
                         window: int = 60) -> pd.DataFrame:
    """Calculate rolling correlations with market instruments"""
    correlations = pd.DataFrame(index=price_data.index)
    
    for column in market_data.columns:
        correlations[f'corr_{column}'] = (
            price_data['Close']
            .rolling(window)
            .corr(market_data[column])
        )
    
    return correlations

def detect_market_regime(market_data: pd.DataFrame,
                        window: int = 60) -> pd.Series:
    """Detect market regime using multiple indicators"""
    nifty = market_data['Nifty 50']
    
    # Trend
    sma200 = nifty.rolling(200).mean()
    trend = np.where(nifty > sma200, 1, -1)
    
    # Volatility
    volatility = nifty.rolling(window).std()
    vol_high = volatility > volatility.rolling(window).mean()
    
    # Momentum
    momentum = nifty.pct_change(window).gt(0)
    
    # Combine signals
    regime = pd.Series(index=market_data.index, dtype=str)
    regime.loc[(trend == 1) & momentum & ~vol_high] = 'bull_normal'
    regime.loc[(trend == 1) & momentum & vol_high] = 'bull_volatile'
    regime.loc[(trend == -1) & ~momentum & ~vol_high] = 'bear_normal'
    regime.loc[(trend == -1) & ~momentum & vol_high] = 'bear_volatile'
    regime.loc[regime.isna()] = 'neutral'
    
    return regime

def analyze_market_context(stock_data: pd.DataFrame,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """Analyze stock in market context"""
    # Get market data
    market_data = download_market_data(start_date, end_date)
    
    # Calculate correlations
    correlations = calculate_correlations(stock_data, market_data)
    
    # Detect regime
    regime = detect_market_regime(market_data)
    
    # Combine results
    result = stock_data.copy()
    result = pd.concat([result, correlations], axis=1)
    result['market_regime'] = regime
    
    return result

def process_market_context(input_dir: str = "data/processed",
                         output_dir: str = "data/market_regime") -> None:
    """Process all stocks with market context"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for file in input_path.glob("*_enriched.parquet"):
        logger.info(f"Adding market context to {file.name}")
        
        # Read enriched data
        df = pd.read_parquet(file)
        
        # Add market context
        df = analyze_market_context(df)
        
        # Save with market context
        output_file = output_path / file.name.replace('_enriched.parquet', '_market.parquet')
        df.to_parquet(output_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_market_context()
