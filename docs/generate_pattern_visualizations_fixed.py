#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/generate_pattern_visualizations_fixed.py

"""
Generate visual examples of candlestick patterns and technical indicators for documentation.
This script creates images illustrating key patterns and indicators used in the Nifty 500 Trading System.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.patches as patches
from pathlib import Path
try:
    import mplfinance as mpf
except ImportError:
    print("Installing mplfinance...")
    import subprocess
    subprocess.check_call(["pip", "install", "mplfinance"])
    import mplfinance as mpf

# Ensure the pattern_images directory exists
image_dir = Path(__file__).parent / 'images' / 'patterns'
image_dir.mkdir(exist_ok=True, parents=True)

def generate_prices(trend="neutral", days=30, volatility=0.015):
    """Generate synthetic price data for pattern visualization"""
    dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
    
    # Base price path with specified trend
    if trend == "uptrend":
        base = np.linspace(100, 110, days) + np.random.normal(0, volatility * 100, days).cumsum()
    elif trend == "downtrend":
        base = np.linspace(110, 100, days) + np.random.normal(0, volatility * 100, days).cumsum()
    else:  # neutral
        base = 100 + np.random.normal(0, volatility * 100, days).cumsum()
    
    # Generate OHLC data
    opens = base + np.random.normal(0, volatility * base, days)
    highs = np.maximum(base + np.random.normal(volatility * base, volatility * base, days), 
                      np.maximum(opens, base))
    lows = np.minimum(base + np.random.normal(-volatility * base, volatility * base, days), 
                     np.minimum(opens, base))
    closes = base
    volumes = np.random.normal(1000000, 200000, days) * (1 + np.abs(closes - opens) / opens)
    
    df = pd.DataFrame({
        'Date': pd.DatetimeIndex(dates),
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    return df

def insert_pattern(df, pattern_type, position=-5):
    """Insert a specific candlestick pattern into the synthetic data"""
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    # Convert negative position to actual index
    if position < 0:
        actual_position = len(data) + position
    else:
        actual_position = position
    
    if pattern_type == "doji":
        # Create a doji (open ≈ close)
        data.iloc[actual_position, data.columns.get_loc('Close')] = data.iloc[actual_position, data.columns.get_loc('Open')]
        data.iloc[actual_position, data.columns.get_loc('High')] = data.iloc[actual_position, data.columns.get_loc('Open')] * 1.02
        data.iloc[actual_position, data.columns.get_loc('Low')] = data.iloc[actual_position, data.columns.get_loc('Open')] * 0.98
    
    elif pattern_type == "hammer":
        # Create a hammer (small body, little/no upper shadow, long lower shadow)
        data.iloc[actual_position, data.columns.get_loc('Open')] = data.iloc[actual_position, data.columns.get_loc('Close')] * 1.01  # Small bearish body
        data.iloc[actual_position, data.columns.get_loc('High')] = data.iloc[actual_position, data.columns.get_loc('Open')] * 1.005  # Very small upper shadow
        data.iloc[actual_position, data.columns.get_loc('Low')] = data.iloc[actual_position, data.columns.get_loc('Close')] * 0.95   # Long lower shadow
    
    elif pattern_type == "shooting_star":
        # Create a shooting star (small body, long upper shadow, little/no lower shadow)
        data.iloc[actual_position, data.columns.get_loc('Open')] = data.iloc[actual_position, data.columns.get_loc('Close')] * 0.99  # Small bullish body
        data.iloc[actual_position, data.columns.get_loc('High')] = data.iloc[actual_position, data.columns.get_loc('Open')] * 1.05   # Long upper shadow
        data.iloc[actual_position, data.columns.get_loc('Low')] = data.iloc[actual_position, data.columns.get_loc('Close')] * 0.995  # Very small lower shadow
    
    elif pattern_type == "engulfing_bullish":
        # Create a bullish engulfing pattern (small bearish candle followed by large bullish candle)
        data.iloc[actual_position-1, data.columns.get_loc('Close')] = data.iloc[actual_position-1, data.columns.get_loc('Open')] * 0.98  # Bearish candle
        data.iloc[actual_position, data.columns.get_loc('Open')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 0.99    # Open below prev close
        data.iloc[actual_position, data.columns.get_loc('Close')] = data.iloc[actual_position-1, data.columns.get_loc('Open')] * 1.03    # Close above prev open
    
    elif pattern_type == "engulfing_bearish":
        # Create a bearish engulfing pattern (small bullish candle followed by large bearish candle)
        data.iloc[actual_position-1, data.columns.get_loc('Close')] = data.iloc[actual_position-1, data.columns.get_loc('Open')] * 1.02  # Bullish candle
        data.iloc[actual_position, data.columns.get_loc('Open')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 1.01    # Open above prev close
        data.iloc[actual_position, data.columns.get_loc('Close')] = data.iloc[actual_position-1, data.columns.get_loc('Open')] * 0.97    # Close below prev open
    
    elif pattern_type == "morning_star":
        # Create a morning star (bearish candle, small-bodied candle, bullish candle)
        data.iloc[actual_position-2, data.columns.get_loc('Open')] = data.iloc[actual_position-2, data.columns.get_loc('Close')] * 1.05  # Large bearish candle
        data.iloc[actual_position-1, data.columns.get_loc('Open')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 0.995  # Small body
        data.iloc[actual_position-1, data.columns.get_loc('High')] = data.iloc[actual_position-1, data.columns.get_loc('Open')] * 1.01
        data.iloc[actual_position-1, data.columns.get_loc('Low')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 0.99
        data.iloc[actual_position, data.columns.get_loc('Open')] = data.iloc[actual_position, data.columns.get_loc('Close')] * 0.97      # Bullish candle
        data.iloc[actual_position, data.columns.get_loc('Close')] = data.iloc[actual_position-2, data.columns.get_loc('Close')] * 1.02   # Close above first day
    
    elif pattern_type == "evening_star":
        # Create an evening star (bullish candle, small-bodied candle, bearish candle)
        data.iloc[actual_position-2, data.columns.get_loc('Open')] = data.iloc[actual_position-2, data.columns.get_loc('Close')] * 0.95  # Large bullish candle
        data.iloc[actual_position-1, data.columns.get_loc('Open')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 1.005  # Small body
        data.iloc[actual_position-1, data.columns.get_loc('High')] = data.iloc[actual_position-1, data.columns.get_loc('Open')] * 1.01
        data.iloc[actual_position-1, data.columns.get_loc('Low')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 0.99
        data.iloc[actual_position, data.columns.get_loc('Open')] = data.iloc[actual_position, data.columns.get_loc('Close')] * 1.03      # Bearish candle
        data.iloc[actual_position, data.columns.get_loc('Close')] = data.iloc[actual_position-2, data.columns.get_loc('Close')] * 0.98   # Close below first day
    
    elif pattern_type == "three_white_soldiers":
        # Create three white soldiers (three bullish candles, each opening within previous body and closing higher)
        data.iloc[actual_position-2, data.columns.get_loc('Open')] = data.iloc[actual_position-2, data.columns.get_loc('Close')] * 0.97  # First soldier
        data.iloc[actual_position-1, data.columns.get_loc('Open')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 0.98  # Second soldier
        data.iloc[actual_position-1, data.columns.get_loc('Close')] = data.iloc[actual_position-2, data.columns.get_loc('Close')] * 1.02  # Higher close
        data.iloc[actual_position, data.columns.get_loc('Open')] = data.iloc[actual_position, data.columns.get_loc('Close')] * 0.98      # Third soldier
        data.iloc[actual_position, data.columns.get_loc('Close')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 1.02   # Even higher close
    
    elif pattern_type == "three_black_crows":
        # Create three black crows (three bearish candles, each opening within previous body and closing lower)
        data.iloc[actual_position-2, data.columns.get_loc('Open')] = data.iloc[actual_position-2, data.columns.get_loc('Close')] * 1.03  # First crow
        data.iloc[actual_position-1, data.columns.get_loc('Open')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 1.02  # Second crow
        data.iloc[actual_position-1, data.columns.get_loc('Close')] = data.iloc[actual_position-2, data.columns.get_loc('Close')] * 0.98  # Lower close
        data.iloc[actual_position, data.columns.get_loc('Open')] = data.iloc[actual_position, data.columns.get_loc('Close')] * 1.02      # Third crow
        data.iloc[actual_position, data.columns.get_loc('Close')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 0.98   # Even lower close
    
    elif pattern_type == "harami_bullish":
        # Create a bullish harami (large bearish candle followed by small bullish candle contained within first candle)
        data.iloc[actual_position-1, data.columns.get_loc('Open')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 1.06  # Large bearish candle
        mid_point = (data.iloc[actual_position-1, data.columns.get_loc('Open')] + data.iloc[actual_position-1, data.columns.get_loc('Close')]) / 2
        data.iloc[actual_position, data.columns.get_loc('Open')] = mid_point * 0.99                      # Small body contained
        data.iloc[actual_position, data.columns.get_loc('Close')] = mid_point * 1.01                     # within first candle
    
    elif pattern_type == "harami_bearish":
        # Create a bearish harami (large bullish candle followed by small bearish candle contained within first candle)
        data.iloc[actual_position-1, data.columns.get_loc('Open')] = data.iloc[actual_position-1, data.columns.get_loc('Close')] * 0.94  # Large bullish candle
        mid_point = (data.iloc[actual_position-1, data.columns.get_loc('Open')] + data.iloc[actual_position-1, data.columns.get_loc('Close')]) / 2
        data.iloc[actual_position, data.columns.get_loc('Open')] = mid_point * 1.01                      # Small body contained
        data.iloc[actual_position, data.columns.get_loc('Close')] = mid_point * 0.99                     # within first candle
    
    return data

def create_pattern_visualization(pattern_name, trend="neutral"):
    """Create and save visualization of a specific candlestick pattern"""
    data = generate_prices(trend=trend)
    
    if pattern_name == "doji":
        data = insert_pattern(data, "doji")
        title = "Doji Candlestick Pattern"
        description = "A single candle with very small body (open ≈ close)"
    
    elif pattern_name == "hammer":
        data = insert_pattern(data, "hammer")
        title = "Hammer Candlestick Pattern"
        description = "Bullish reversal: small body, minimal upper shadow, long lower shadow"
    
    elif pattern_name == "shooting_star":
        data = insert_pattern(data, "shooting_star")
        title = "Shooting Star Candlestick Pattern"
        description = "Bearish reversal: small body, long upper shadow, minimal lower shadow"
    
    elif pattern_name == "engulfing_bullish":
        data = insert_pattern(data, "engulfing_bullish")
        title = "Bullish Engulfing Pattern"
        description = "Small bearish candle followed by large bullish candle that engulfs previous candle"
    
    elif pattern_name == "engulfing_bearish":
        data = insert_pattern(data, "engulfing_bearish")
        title = "Bearish Engulfing Pattern"
        description = "Small bullish candle followed by large bearish candle that engulfs previous candle"
    
    elif pattern_name == "morning_star":
        data = insert_pattern(data, "morning_star")
        title = "Morning Star Pattern"
        description = "Bullish reversal: bearish candle, small candle, bullish candle"
    
    elif pattern_name == "evening_star":
        data = insert_pattern(data, "evening_star")
        title = "Evening Star Pattern"
        description = "Bearish reversal: bullish candle, small candle, bearish candle"
    
    elif pattern_name == "three_white_soldiers":
        data = insert_pattern(data, "three_white_soldiers")
        title = "Three White Soldiers Pattern"
        description = "Bullish reversal: three consecutive bullish candles with progressive higher closes"
    
    elif pattern_name == "three_black_crows":
        data = insert_pattern(data, "three_black_crows")
        title = "Three Black Crows Pattern"
        description = "Bearish reversal: three consecutive bearish candles with progressive lower closes"
    
    elif pattern_name == "harami_bullish":
        data = insert_pattern(data, "harami_bullish")
        title = "Bullish Harami Pattern"
        description = "Large bearish candle followed by small bullish candle contained within first candle"
    
    elif pattern_name == "harami_bearish":
        data = insert_pattern(data, "harami_bearish")
        title = "Bearish Harami Pattern"
        description = "Large bullish candle followed by small bearish candle contained within first candle"
    
    # Convert data to format suitable for mplfinance
    df = data.copy()
    df.set_index('Date', inplace=True)
    df = df.iloc[-10:]  # Just show the last 10 bars
    
    # Create plot using mplfinance
    fig, ax = plt.subplots(figsize=(10, 6))
    mpf.plot(df, type='candle', style='yahoo', ax=ax, volume=False)
    
    # Highlight the pattern
    if "three" in pattern_name or "morning_star" in pattern_name or "evening_star" in pattern_name:
        pattern_range = 3
    elif "engulfing" in pattern_name or "harami" in pattern_name:
        pattern_range = 2
    else:
        pattern_range = 1
    
    # Add title and description
    plt.title(title, fontsize=14)
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=12)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(image_dir / f"{pattern_name}.png", dpi=150)
    plt.close()

def create_technical_indicator_visualization(indicator_name):
    """Create and save visualization of a technical indicator"""
    # Generate some price data
    days = 100
    data = generate_prices(days=days, trend="neutral" if "bands" in indicator_name else "uptrend")
    data.set_index('Date', inplace=True)
    
    if indicator_name == "bollinger_bands":
        # Calculate Bollinger Bands
        window = 20
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        mpf.plot(data, type='candle', style='yahoo', ax=ax, volume=False)
        
        # Add Bollinger Bands
        ax.plot(data.index, rolling_mean, label='20-day SMA (Middle Band)', color='red')
        ax.plot(data.index, upper_band, label='Upper Band (+2σ)', color='green', linestyle='--')
        ax.plot(data.index, lower_band, label='Lower Band (-2σ)', color='green', linestyle='--')
        
        # Add description
        title = "Bollinger Bands"
        description = "Volatility-based bands set at standard deviation levels from a moving average"
        
    elif indicator_name == "keltner_channels":
        # Calculate Keltner Channels
        window = 20
        ema = data['Close'].ewm(span=window).mean()
        
        # Calculate ATR
        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - data['Close'].shift(1))
        tr3 = abs(data['Low'] - data['Close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        upper_channel = ema + (atr * 2)
        lower_channel = ema - (atr * 2)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        mpf.plot(data, type='candle', style='yahoo', ax=ax, volume=False)
        
        # Add Keltner Channels
        ax.plot(data.index, ema, label='20-day EMA (Middle Line)', color='red')
        ax.plot(data.index, upper_channel, label='Upper Channel (EMA + 2×ATR)', color='purple', linestyle='--')
        ax.plot(data.index, lower_channel, label='Lower Channel (EMA - 2×ATR)', color='purple', linestyle='--')
        
        # Add description
        title = "Keltner Channels"
        description = "Volatility-based channels using EMA and ATR instead of standard deviation"
    
    elif indicator_name == "atr":
        # Calculate ATR
        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - data['Close'].shift(1))
        tr3 = abs(data['Low'] - data['Close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Create two-panel plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart in top panel
        mpf.plot(data, type='candle', style='yahoo', ax=ax1, volume=False)
        ax1.set_ylabel('Price')
        
        # ATR in bottom panel
        ax2.plot(data.index, atr, label='14-day ATR', color='red')
        ax2.set_ylabel('ATR')
        ax2.fill_between(data.index, 0, atr, color='red', alpha=0.2)
        
        # Add description
        title = "Average True Range (ATR)"
        description = "Measures market volatility by decomposing the entire range of an asset price for that period"
        
    elif indicator_name == "support_resistance":
        # Generate data with clearer support/resistance zones
        data = generate_prices(days=days, trend="neutral", volatility=0.01)
        data.set_index('Date', inplace=True)
        
        # Create artificial support/resistance levels
        support_level = data['Close'].min() * 1.02
        resistance_level = data['Close'].max() * 0.98
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        mpf.plot(data, type='candle', style='yahoo', ax=ax, volume=False)
        
        # Draw support and resistance lines
        ax.axhline(y=support_level, color='green', linestyle='-', linewidth=2, label='Support')
        ax.axhline(y=resistance_level, color='red', linestyle='-', linewidth=2, label='Resistance')
        
        # Add description
        title = "Support and Resistance Levels"
        description = "Price levels where a stock has historically had difficulty rising above (resistance) or falling below (support)"
    
    elif indicator_name == "macd":
        # Calculate MACD
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        # Create two-panel plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart in top panel
        mpf.plot(data, type='candle', style='yahoo', ax=ax1, volume=False)
        ax1.set_ylabel('Price')
        
        # MACD in bottom panel
        ax2.plot(data.index, macd_line, label='MACD Line (12-26)', color='blue')
        ax2.plot(data.index, signal_line, label='Signal Line (9)', color='red')
        
        # Plot histogram
        for i in range(len(histogram)):
            if histogram.iloc[i] >= 0:
                ax2.bar(data.index[i], histogram.iloc[i], color='green', width=1, alpha=0.5)
            else:
                ax2.bar(data.index[i], histogram.iloc[i], color='red', width=1, alpha=0.5)
        
        # Add zero line
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add description
        title = "Moving Average Convergence Divergence (MACD)"
        description = "Trend-following momentum indicator showing the relationship between two moving averages"
    
    # Set title and labels
    plt.suptitle(title, fontsize=14)
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=12)
    
    # Format axes
    for ax in fig.axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(image_dir / f"{indicator_name}.png", dpi=150)
    plt.close()

def main():
    """Generate all pattern and indicator visualizations"""
    print("Generating candlestick pattern visualizations...")
    
    # List of patterns to visualize
    candlestick_patterns = [
        "doji",
        "hammer",
        "shooting_star",
        "engulfing_bullish",
        "engulfing_bearish",
        "morning_star",
        "evening_star",
        "three_white_soldiers",
        "three_black_crows",
        "harami_bullish",
        "harami_bearish"
    ]
    
    # Create pattern visualizations
    for pattern in candlestick_patterns:
        print(f"Creating visualization for {pattern}...")
        create_pattern_visualization(pattern)
    
    print("\nGenerating technical indicator visualizations...")
    
    # List of indicators to visualize
    indicators = [
        "bollinger_bands",
        "keltner_channels",
        "atr",
        "support_resistance",
        "macd"
    ]
    
    # Create indicator visualizations
    for indicator in indicators:
        print(f"Creating visualization for {indicator}...")
        create_technical_indicator_visualization(indicator)
    
    print(f"\nAll visualizations saved to {image_dir}")

if __name__ == "__main__":
    main()
