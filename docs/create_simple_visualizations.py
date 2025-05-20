#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/create_simple_visualizations.py

"""
Generate simple candlestick patterns and technical indicator images for documentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mplfinance as mpf
from datetime import datetime, timedelta

# Create directory for images
image_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/images/patterns')
image_dir.mkdir(exist_ok=True, parents=True)

print(f"Creating images in: {image_dir}")

# Generate sample data
def generate_sample_data(pattern_type='bullish'):
    # Generate date range for the sample data
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    
    # Create base prices
    if pattern_type == 'bullish':
        closes = np.linspace(100, 90, 20).tolist() + np.linspace(90, 100, 10).tolist()
    elif pattern_type == 'bearish':
        closes = np.linspace(90, 100, 20).tolist() + np.linspace(100, 90, 10).tolist()
    else:  # neutral
        closes = np.linspace(90, 100, 15).tolist() + np.linspace(100, 90, 15).tolist()
    
    # Add some randomness
    closes = [c + np.random.normal(0, 1) for c in closes]
    
    # Generate OHLC data
    opens = [c + np.random.normal(0, 2) for c in closes]
    highs = [max(o, c) + abs(np.random.normal(0, 1)) for o, c in zip(opens, closes)]
    lows = [min(o, c) - abs(np.random.normal(0, 1)) for o, c in zip(opens, closes)]
    volumes = [1000000 + np.random.normal(0, 100000) for _ in closes]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=dates)
    
    return df

# Create and save pattern images
def create_pattern_image(pattern_name):
    print(f"Creating {pattern_name} pattern image...")
    
    # Create appropriate data based on pattern type
    if pattern_name in ['hammer', 'bullish_engulfing', 'morning_star']:
        df = generate_sample_data('bullish')
    elif pattern_name in ['shooting_star', 'bearish_engulfing', 'evening_star']:
        df = generate_sample_data('bearish')
    else:
        df = generate_sample_data('neutral')
    
    # Customize specific patterns in the data
    if pattern_name == 'doji':
        # Doji has open ≈ close
        idx = df.index[-5]
        df.loc[idx, 'Close'] = df.loc[idx, 'Open']
        df.loc[idx, 'High'] = df.loc[idx, 'Open'] * 1.02
        df.loc[idx, 'Low'] = df.loc[idx, 'Open'] * 0.98
    
    elif pattern_name == 'hammer':
        # Hammer has small body, long lower shadow
        idx = df.index[-5]
        df.loc[idx, 'Close'] = df.loc[idx, 'Open'] * 1.01
        df.loc[idx, 'High'] = df.loc[idx, 'Close'] * 1.005
        df.loc[idx, 'Low'] = df.loc[idx, 'Close'] * 0.95
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    mpf.plot(df, type='candle', style='yahoo', ax=ax)
    
    # Save the image
    plt.savefig(image_dir / f"{pattern_name}.png", dpi=150)
    plt.close()

# Create and save indicator images
def create_indicator_image(indicator_name):
    print(f"Creating {indicator_name} indicator image...")
    
    # Create sample data
    df = generate_sample_data()
    
    # Add indicator to data
    if indicator_name == 'bollinger_bands':
        # Calculate Bollinger Bands
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['StdDev'] = df['Close'].rolling(window=20).std()
        df['UpperBand'] = df['SMA20'] + (df['StdDev'] * 2)
        df['LowerBand'] = df['SMA20'] - (df['StdDev'] * 2)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(df, type='candle', style='yahoo', ax=ax)
        
        # Add Bollinger Bands to the plot
        ax.plot(df.index, df['SMA20'], label='SMA(20)', color='blue')
        ax.plot(df.index, df['UpperBand'], label='Upper Band', color='red')
        ax.plot(df.index, df['LowerBand'], label='Lower Band', color='red')
        ax.legend()
        
    elif indicator_name == 'macd':
        # Calculate MACD
        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        mpf.plot(df, type='candle', style='yahoo', ax=ax1)
        
        # Add MACD to the bottom plot
        ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax2.plot(df.index, df['Signal'], label='Signal', color='red')
        ax2.legend()
    
    # Save the image
    plt.savefig(image_dir / f"{indicator_name}.png", dpi=150)
    plt.close()

def main():
    # Generate candlestick pattern images
    patterns = ['doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing', 
                'morning_star', 'evening_star']
    
    for pattern in patterns:
        create_pattern_image(pattern)
    
    # Generate indicator images
    indicators = ['bollinger_bands', 'macd']
    
    for indicator in indicators:
        create_indicator_image(indicator)
    
    print("All images created successfully!")

if __name__ == "__main__":
    main()
