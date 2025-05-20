#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/create_missing_pattern_images.py

"""
Create missing pattern and indicator images for the documentation.
This script checks which patterns are referenced in the documentation
but don't have corresponding image files, and creates them.
"""

import os
import re
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Define paths
docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
image_dir = docs_dir / 'images' / 'patterns'
image_dir.mkdir(exist_ok=True, parents=True)

# Read documentation files to extract pattern references
pattern_files = [
    docs_dir / 'comprehensive_pattern_guide.md',
    docs_dir / 'comprehensive_pattern_guide_enhanced.md',
    docs_dir / 'indicator_explanations.md',
    docs_dir / 'indicator_explanations_enhanced.md',
    docs_dir / 'nifty500_trading_system_master_document.md'
]

def extract_patterns_from_docs():
    """Extract pattern names referenced in the documentation files."""
    pattern_references = set()
    img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    
    for file_path in pattern_files:
        if not file_path.exists():
            print(f"File does not exist: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find all image references
            for match in img_pattern.finditer(content):
                img_path = match.group(1)
                if 'patterns/' in img_path:
                    # Extract just the filename without extension
                    pattern_name = os.path.basename(img_path).split('.')[0]
                    pattern_references.add(pattern_name)
            
            print(f"Processed {file_path}, found references to patterns in this file")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return pattern_references

def get_existing_patterns():
    """Get list of existing pattern images."""
    existing_patterns = set()
    
    if image_dir.exists():
        for img_file in image_dir.glob('*.png'):
            pattern_name = img_file.stem  # filename without extension
            existing_patterns.add(pattern_name)
    
    return existing_patterns

def generate_sample_data(pattern_type='bullish'):
    """Generate sample OHLC data for patterns."""
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

def create_pattern(pattern_name):
    """Create and save a visualization for the specified pattern."""
    print(f"Creating visualization for {pattern_name}...")
    
    # Determine pattern type from name
    if any(term in pattern_name.lower() for term in ['bullish', 'hammer', 'morning', 'white']):
        pattern_type = 'bullish'
    elif any(term in pattern_name.lower() for term in ['bearish', 'shooting', 'evening', 'black', 'dark']):
        pattern_type = 'bearish'
    else:
        pattern_type = 'neutral'
    
    # Generate data
    df = generate_sample_data(pattern_type)
    
    # Customize specific patterns
    if 'doji' in pattern_name.lower():
        # Doji has open ≈ close
        idx = df.index[-5]
        df.loc[idx, 'Close'] = df.loc[idx, 'Open']
        df.loc[idx, 'High'] = df.loc[idx, 'Open'] * 1.02
        df.loc[idx, 'Low'] = df.loc[idx, 'Open'] * 0.98
    
    elif 'hammer' in pattern_name.lower() or 'hanging' in pattern_name.lower():
        # Hammer has small body, long lower shadow
        idx = df.index[-5]
        if pattern_type == 'bullish':
            df.loc[idx, 'Close'] = df.loc[idx, 'Open'] * 1.01
        else:
            df.loc[idx, 'Close'] = df.loc[idx, 'Open'] * 0.99
        df.loc[idx, 'High'] = df.loc[idx, 'Close'] * 1.005
        df.loc[idx, 'Low'] = df.loc[idx, 'Close'] * 0.95
    
    elif 'shooting' in pattern_name.lower() or 'inverted' in pattern_name.lower():
        # Shooting star has small body, long upper shadow
        idx = df.index[-5]
        if pattern_type == 'bearish':
            df.loc[idx, 'Close'] = df.loc[idx, 'Open'] * 0.99
        else:
            df.loc[idx, 'Close'] = df.loc[idx, 'Open'] * 1.01
        df.loc[idx, 'High'] = df.loc[idx, 'Close'] * 1.05
        df.loc[idx, 'Low'] = df.loc[idx, 'Close'] * 0.995
    
    elif 'engulfing' in pattern_name.lower():
        # Engulfing pattern (two candles)
        idx1 = df.index[-6]
        idx2 = df.index[-5]
        
        if 'bullish' in pattern_name.lower():
            # First candle: small bearish
            df.loc[idx1, 'Open'] = 100
            df.loc[idx1, 'Close'] = 99
            df.loc[idx1, 'High'] = 100.5
            df.loc[idx1, 'Low'] = 98.5
            
            # Second candle: larger bullish engulfing
            df.loc[idx2, 'Open'] = 98.8
            df.loc[idx2, 'Close'] = 100.2
            df.loc[idx2, 'High'] = 100.7
            df.loc[idx2, 'Low'] = 98.3
        else:
            # First candle: small bullish
            df.loc[idx1, 'Open'] = 99
            df.loc[idx1, 'Close'] = 100
            df.loc[idx1, 'High'] = 100.5
            df.loc[idx1, 'Low'] = 98.5
            
            # Second candle: larger bearish engulfing
            df.loc[idx2, 'Open'] = 100.2
            df.loc[idx2, 'Close'] = 98.8
            df.loc[idx2, 'High'] = 100.7
            df.loc[idx2, 'Low'] = 98.3
    
    elif 'morning_star' in pattern_name.lower():
        # Morning star (three candles)
        idx1 = df.index[-7]  # Bearish candle
        idx2 = df.index[-6]  # Star
        idx3 = df.index[-5]  # Bullish candle
        
        # First candle: long bearish
        df.loc[idx1, 'Open'] = 100
        df.loc[idx1, 'Close'] = 97
        df.loc[idx1, 'High'] = 100.5
        df.loc[idx1, 'Low'] = 96.5
        
        # Second candle: small doji/star
        df.loc[idx2, 'Open'] = 96.8
        df.loc[idx2, 'Close'] = 96.9
        df.loc[idx2, 'High'] = 97.1
        df.loc[idx2, 'Low'] = 96.5
        
        # Third candle: bullish candle
        df.loc[idx3, 'Open'] = 97.1
        df.loc[idx3, 'Close'] = 99
        df.loc[idx3, 'High'] = 99.5
        df.loc[idx3, 'Low'] = 96.9
    
    elif 'evening_star' in pattern_name.lower():
        # Evening star (three candles)
        idx1 = df.index[-7]  # Bullish candle
        idx2 = df.index[-6]  # Star
        idx3 = df.index[-5]  # Bearish candle
        
        # First candle: long bullish
        df.loc[idx1, 'Open'] = 97
        df.loc[idx1, 'Close'] = 100
        df.loc[idx1, 'High'] = 100.5
        df.loc[idx1, 'Low'] = 96.5
        
        # Second candle: small doji/star
        df.loc[idx2, 'Open'] = 100.2
        df.loc[idx2, 'Close'] = 100.1
        df.loc[idx2, 'High'] = 100.5
        df.loc[idx2, 'Low'] = 99.9
        
        # Third candle: bearish candle
        df.loc[idx3, 'Open'] = 99.9
        df.loc[idx3, 'Close'] = 98
        df.loc[idx3, 'High'] = 100.1
        df.loc[idx3, 'Low'] = 97.5
    
    # Create a slice of data around the pattern
    if 'engulfing' in pattern_name.lower():
        df_slice = df.iloc[-10:-2]
    elif 'morning_star' in pattern_name.lower() or 'evening_star' in pattern_name.lower():
        df_slice = df.iloc[-11:-2]
    else:
        df_slice = df.iloc[-10:-2]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    mpf.plot(df_slice, type='candle', style='yahoo', ax=ax, volume=False)
    
    # Highlight the pattern
    if 'engulfing' in pattern_name.lower():
        # Highlight the engulfing pattern
        ax.axvspan(df_slice.index[-2], df_slice.index[-1], alpha=0.2, color='yellow')
    elif 'morning_star' in pattern_name.lower() or 'evening_star' in pattern_name.lower():
        # Highlight the three-candle pattern
        ax.axvspan(df_slice.index[-3], df_slice.index[-1], alpha=0.2, color='yellow')
    else:
        # Highlight the single candle pattern
        ax.axvspan(df_slice.index[-2], df_slice.index[-1], alpha=0.2, color='yellow')
    
    # Add title
    pattern_title = pattern_name.replace('_', ' ').title()
    plt.title(f"{pattern_title} Pattern")
    
    # Save the image
    output_path = image_dir / f"{pattern_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Created {output_path}")
    return output_path

def create_indicator(indicator_name):
    """Create and save a visualization for the specified indicator."""
    print(f"Creating visualization for {indicator_name}...")
    
    # Generate data
    df = generate_sample_data('neutral')
    
    if 'macd' in indicator_name.lower():
        # MACD (Moving Average Convergence Divergence)
        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        mpf.plot(df, type='candle', style='yahoo', ax=ax1, volume=False)
        
        # Add MACD to the bottom plot
        ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax2.plot(df.index, df['Signal'], label='Signal', color='red')
        ax2.fill_between(df.index, df['MACD'] - df['Signal'], 0, 
                         where=(df['MACD'] >= df['Signal']), 
                         facecolor='green', alpha=0.5)
        ax2.fill_between(df.index, df['MACD'] - df['Signal'], 0, 
                         where=(df['MACD'] < df['Signal']), 
                         facecolor='red', alpha=0.5)
        ax2.legend()
        ax2.set_ylabel('MACD')
        ax2.grid(True, alpha=0.3)
        
        plt.title('MACD Indicator')
    
    elif 'bollinger' in indicator_name.lower():
        # Bollinger Bands
        period = 20
        # Calculate Bollinger Bands
        df['SMA20'] = df['Close'].rolling(window=period).mean()
        df['StdDev'] = df['Close'].rolling(window=period).std()
        df['UpperBand'] = df['SMA20'] + (df['StdDev'] * 2)
        df['LowerBand'] = df['SMA20'] - (df['StdDev'] * 2)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(df, type='candle', style='yahoo', ax=ax, volume=False)
        
        # Add Bollinger Bands to the plot
        ax.plot(df.index, df['SMA20'], label='SMA(20)', color='blue')
        ax.plot(df.index, df['UpperBand'], label='Upper Band', color='red')
        ax.plot(df.index, df['LowerBand'], label='Lower Band', color='red')
        ax.fill_between(df.index, df['UpperBand'], df['LowerBand'], alpha=0.1, color='blue')
        ax.legend()
        
        plt.title('Bollinger Bands')
    
    elif 'rsi' in indicator_name.lower():
        # RSI (Relative Strength Index)
        period = 14
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        mpf.plot(df, type='candle', style='yahoo', ax=ax1, volume=False)
        
        # Add RSI to the bottom plot
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.fill_between(df.index, df['RSI'], 70, 
                         where=(df['RSI'] >= 70), 
                         facecolor='red', alpha=0.5)
        ax2.fill_between(df.index, df['RSI'], 30, 
                         where=(df['RSI'] <= 30), 
                         facecolor='green', alpha=0.5)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI')
        ax2.grid(True, alpha=0.3)
        
        plt.title('Relative Strength Index (RSI)')
    
    elif 'atr' in indicator_name.lower():
        # ATR (Average True Range)
        period = 14
        # Calculate ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=period).mean()
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        mpf.plot(df, type='candle', style='yahoo', ax=ax1, volume=False)
        
        # Add ATR to the bottom plot
        ax2.plot(df.index, df['ATR'], label='ATR(14)', color='orange')
        ax2.set_ylabel('ATR')
        ax2.grid(True, alpha=0.3)
        
        plt.title('Average True Range (ATR)')
    
    else:
        # Generic technical indicator visualization
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(df, type='candle', style='yahoo', ax=ax, volume=False)
        
        # Add a moving average
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        ax.plot(df.index, df['SMA20'], label='SMA(20)', color='blue')
        ax.legend()
        
        plt.title(f"{indicator_name.replace('_', ' ').title()}")
    
    # Save the image
    output_path = image_dir / f"{indicator_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Created {output_path}")
    return output_path

def main():
    """Main function to check and create missing pattern images."""
    # Get patterns referenced in docs
    referenced_patterns = extract_patterns_from_docs()
    print(f"Found {len(referenced_patterns)} pattern references in documentation:")
    print(', '.join(sorted(referenced_patterns)))
    
    # Get existing pattern images
    existing_patterns = get_existing_patterns()
    print(f"\nFound {len(existing_patterns)} existing pattern images:")
    print(', '.join(sorted(existing_patterns)))
    
    # Find missing patterns
    missing_patterns = referenced_patterns - existing_patterns
    print(f"\nFound {len(missing_patterns)} missing pattern images to create:")
    print(', '.join(sorted(missing_patterns)))
    
    # Create missing pattern images
    for pattern_name in sorted(missing_patterns):
        if any(term in pattern_name.lower() for term in ['macd', 'rsi', 'bollinger', 'atr']):
            create_indicator(pattern_name)
        else:
            create_pattern(pattern_name)
    
    print("\nAll missing pattern images have been created!")

if __name__ == "__main__":
    main()
