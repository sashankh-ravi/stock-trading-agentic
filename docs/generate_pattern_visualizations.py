#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/generate_pattern_visualizations.py

"""
Generate visual examples of candlestick patterns and technical indicators for documentation.
This script creates images illustrating key patterns and indicators used in the Nifty 500 Trading System.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.patches as patches
from pathlib import Path
import mplfinance as mpf  # Import mplfinance for better candlestick charts

# Ensure the pattern_images directory exists
image_dir = Path(__file__).parent / 'images' / 'patterns'
image_dir.mkdir(exist_ok=True, parents=True)

def generate_prices(trend="neutral", days=30, volatility=0.015):
    """Generate synthetic price data for pattern visualization"""
    dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
    dates = [mdates.date2num(date) for date in dates]
    
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
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    return data

def insert_pattern(data, pattern_type, position=-5):
    """Insert a specific candlestick pattern into the synthetic data"""
    if pattern_type == "doji":
        # Create a doji (open ≈ close)
        data.loc[position, 'Close'] = data.loc[position, 'Open']
        data.loc[position, 'High'] = data.loc[position, 'Open'] * 1.02
        data.loc[position, 'Low'] = data.loc[position, 'Open'] * 0.98
    
    elif pattern_type == "hammer":
        # Create a hammer (small body, little/no upper shadow, long lower shadow)
        data.loc[position, 'Open'] = data.loc[position, 'Close'] * 1.01  # Small bearish body
        data.loc[position, 'High'] = data.loc[position, 'Open'] * 1.005  # Very small upper shadow
        data.loc[position, 'Low'] = data.loc[position, 'Close'] * 0.95   # Long lower shadow
    
    elif pattern_type == "shooting_star":
        # Create a shooting star (small body, long upper shadow, little/no lower shadow)
        data.loc[position, 'Open'] = data.loc[position, 'Close'] * 0.99  # Small bullish body
        data.loc[position, 'High'] = data.loc[position, 'Open'] * 1.05   # Long upper shadow
        data.loc[position, 'Low'] = data.loc[position, 'Close'] * 0.995  # Very small lower shadow
    
    elif pattern_type == "engulfing_bullish":
        # Create a bullish engulfing pattern (small bearish candle followed by large bullish candle)
        data.loc[position-1, 'Close'] = data.loc[position-1, 'Open'] * 0.98  # Bearish candle
        data.loc[position, 'Open'] = data.loc[position-1, 'Close'] * 0.99    # Open below prev close
        data.loc[position, 'Close'] = data.loc[position-1, 'Open'] * 1.03    # Close above prev open
    
    elif pattern_type == "engulfing_bearish":
        # Create a bearish engulfing pattern (small bullish candle followed by large bearish candle)
        data.loc[position-1, 'Close'] = data.loc[position-1, 'Open'] * 1.02  # Bullish candle
        data.loc[position, 'Open'] = data.loc[position-1, 'Close'] * 1.01    # Open above prev close
        data.loc[position, 'Close'] = data.loc[position-1, 'Open'] * 0.97    # Close below prev open
    
    elif pattern_type == "morning_star":
        # Create a morning star (bearish candle, small-bodied candle, bullish candle)
        data.loc[position-2, 'Open'] = data.loc[position-2, 'Close'] * 1.05  # Large bearish candle
        data.loc[position-1, 'Open'] = data.loc[position-1, 'Close'] * 0.995  # Small body
        data.loc[position-1, 'High'] = data.loc[position-1, 'Open'] * 1.01
        data.loc[position-1, 'Low'] = data.loc[position-1, 'Close'] * 0.99
        data.loc[position, 'Open'] = data.loc[position, 'Close'] * 0.97      # Bullish candle
        data.loc[position, 'Close'] = data.loc[position-2, 'Close'] * 1.02   # Close above first day
    
    elif pattern_type == "evening_star":
        # Create an evening star (bullish candle, small-bodied candle, bearish candle)
        data.loc[position-2, 'Open'] = data.loc[position-2, 'Close'] * 0.95  # Large bullish candle
        data.loc[position-1, 'Open'] = data.loc[position-1, 'Close'] * 1.005  # Small body
        data.loc[position-1, 'High'] = data.loc[position-1, 'Open'] * 1.01
        data.loc[position-1, 'Low'] = data.loc[position-1, 'Close'] * 0.99
        data.loc[position, 'Open'] = data.loc[position, 'Close'] * 1.03      # Bearish candle
        data.loc[position, 'Close'] = data.loc[position-2, 'Close'] * 0.98   # Close below first day
    
    elif pattern_type == "three_white_soldiers":
        # Create three white soldiers (three bullish candles, each opening within previous body and closing higher)
        data.loc[position-2, 'Open'] = data.loc[position-2, 'Close'] * 0.97  # First soldier
        data.loc[position-1, 'Open'] = data.loc[position-1, 'Close'] * 0.98  # Second soldier
        data.loc[position-1, 'Close'] = data.loc[position-2, 'Close'] * 1.02  # Higher close
        data.loc[position, 'Open'] = data.loc[position, 'Close'] * 0.98      # Third soldier
        data.loc[position, 'Close'] = data.loc[position-1, 'Close'] * 1.02   # Even higher close
    
    elif pattern_type == "three_black_crows":
        # Create three black crows (three bearish candles, each opening within previous body and closing lower)
        data.loc[position-2, 'Open'] = data.loc[position-2, 'Close'] * 1.03  # First crow
        data.loc[position-1, 'Open'] = data.loc[position-1, 'Close'] * 1.02  # Second crow
        data.loc[position-1, 'Close'] = data.loc[position-2, 'Close'] * 0.98  # Lower close
        data.loc[position, 'Open'] = data.loc[position, 'Close'] * 1.02      # Third crow
        data.loc[position, 'Close'] = data.loc[position-1, 'Close'] * 0.98   # Even lower close
    
    elif pattern_type == "harami_bullish":
        # Create a bullish harami (large bearish candle followed by small bullish candle contained within first candle)
        data.loc[position-1, 'Open'] = data.loc[position-1, 'Close'] * 1.06  # Large bearish candle
        mid_point = (data.loc[position-1, 'Open'] + data.loc[position-1, 'Close']) / 2
        data.loc[position, 'Open'] = mid_point * 0.99                      # Small body contained
        data.loc[position, 'Close'] = mid_point * 1.01                     # within first candle
    
    elif pattern_type == "harami_bearish":
        # Create a bearish harami (large bullish candle followed by small bearish candle contained within first candle)
        data.loc[position-1, 'Open'] = data.loc[position-1, 'Close'] * 0.94  # Large bullish candle
        mid_point = (data.loc[position-1, 'Open'] + data.loc[position-1, 'Close']) / 2
        data.loc[position, 'Open'] = mid_point * 1.01                      # Small body contained
        data.loc[position, 'Close'] = mid_point * 0.99                     # within first candle
    
    return data

def insert_multiday_pattern(data, pattern_type, position=-5):
    """Insert a specific multi-day candlestick pattern into the synthetic data"""
    if pattern_type == "three_day_reversal_bottom":
        # Create a three-day bullish reversal pattern
        # Three consecutive down days followed by a reversal
        data.loc[position-3, 'Open'] = data.loc[position-3, 'Close'] * 1.03
        data.loc[position-2, 'Open'] = data.loc[position-2, 'Close'] * 1.02
        data.loc[position-1, 'Open'] = data.loc[position-1, 'Close'] * 1.01
        data.loc[position, 'Open'] = data.loc[position, 'Close'] * 0.99
        data.loc[position, 'Close'] = data.loc[position-1, 'Close'] * 1.03
        data.loc[position, 'Volume'] = data.loc[position, 'Volume'] * 1.5  # Higher volume on reversal
    
    elif pattern_type == "key_reversal":
        # Create a key reversal day (bullish)
        # New low followed by close above previous day's high
        data.loc[position-2, 'Open'] = data.loc[position-2, 'Close'] * 1.02
        data.loc[position-1, 'Open'] = data.loc[position-1, 'Close'] * 1.03
        data.loc[position, 'Open'] = data.loc[position, 'Close'] * 1.01
        data.loc[position, 'Low'] = data.loc[position-1, 'Low'] * 0.98  # New low
        data.loc[position, 'Close'] = data.loc[position-1, 'High'] * 1.01  # Close above previous high
        data.loc[position, 'Volume'] = data.loc[position, 'Volume'] * 1.4  # Higher volume
    
    elif pattern_type == "island_reversal":
        # Create an island reversal (bullish)
        # Gap down, trading, gap up
        data.loc[position-2, 'Open'] = data.loc[position-2, 'Close'] * 1.03
        data.loc[position-1, 'Open'] = data.loc[position-2, 'Low'] * 0.97  # Gap down
        data.loc[position-1, 'High'] = data.loc[position-1, 'Open'] * 1.02
        data.loc[position-1, 'Low'] = data.loc[position-1, 'Open'] * 0.98
        data.loc[position, 'Low'] = data.loc[position-1, 'High'] * 1.02  # Gap up
        data.loc[position, 'Open'] = data.loc[position, 'Low'] * 1.01
        data.loc[position, 'Close'] = data.loc[position, 'Open'] * 1.03
    
    elif pattern_type == "three_outside_up":
        # Create a three outside up pattern
        # Bearish candle, bullish engulfing, confirmation candle
        data.loc[position-2, 'Open'] = data.loc[position-2, 'Close'] * 1.02  # Bearish
        data.loc[position-1, 'Open'] = data.loc[position-2, 'Close'] * 0.99  # Open below prev close
        data.loc[position-1, 'Close'] = data.loc[position-2, 'Open'] * 1.02  # Close above prev open (engulfing)
        data.loc[position, 'Open'] = data.loc[position, 'Close'] * 0.99
        data.loc[position, 'Close'] = data.loc[position-1, 'Close'] * 1.02  # Confirmation candle
    
    elif pattern_type == "volatility_squeeze":
        # Create a volatility squeeze pattern
        # Decreasing range between candles, followed by expansion
        for i in range(5, 0, -1):
            volatility_factor = 0.2 + (0.15 * i / 5)  # Decreasing volatility
            data.loc[position-i, 'High'] = data.loc[position-i, 'Open'] * (1 + volatility_factor/10)
            data.loc[position-i, 'Low'] = data.loc[position-i, 'Open'] * (1 - volatility_factor/10)
        
        # Create the breakout candle
        data.loc[position, 'Open'] = data.loc[position-1, 'Close'] * 0.99
        data.loc[position, 'Close'] = data.loc[position, 'Open'] * 1.04
        data.loc[position, 'High'] = data.loc[position, 'Close'] * 1.02
        data.loc[position, 'Volume'] = data.loc[position, 'Volume'] * 2  # Higher volume on breakout
    
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
    df['Date'] = pd.to_datetime([datetime.now() - timedelta(days=30-i) for i in range(len(df))])
    df.set_index('Date', inplace=True)
    df = df.iloc[-10:]  # Just show the last 10 bars
    
    # Create plot using mplfinance
    fig, ax = plt.subplots(figsize=(10, 6))
    mpf.plot(df, type='candle', style='yahoo', ax=ax, volume=False, show_nontrading=False)
    
    # Highlight the pattern
    if "three" in pattern_name:
        rect = patches.Rectangle((plot_data[-3, 0] - 0.3, min(plot_data[-3:, 3]) * 0.99), 3.6, 
                               max(plot_data[-3:, 2]) * 1.01 - min(plot_data[-3:, 3]) * 0.99, 
                               linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8)
        highlight_idx = range(-3, 0)
    elif "morning_star" in pattern_name or "evening_star" in pattern_name:
        rect = patches.Rectangle((plot_data[-3, 0] - 0.3, min(plot_data[-3:, 3]) * 0.99), 3.6, 
                               max(plot_data[-3:, 2]) * 1.01 - min(plot_data[-3:, 3]) * 0.99, 
                               linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8)
        highlight_idx = range(-3, 0)
    elif "engulfing" in pattern_name or "harami" in pattern_name:
        rect = patches.Rectangle((plot_data[-2, 0] - 0.3, min(plot_data[-2:, 3]) * 0.99), 2.6, 
                               max(plot_data[-2:, 2]) * 1.01 - min(plot_data[-2:, 3]) * 0.99, 
                               linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8)
        highlight_idx = range(-2, 0)
    else:
        rect = patches.Rectangle((plot_data[-1, 0] - 0.3, plot_data[-1, 3] * 0.99), 1.6, 
                               plot_data[-1, 2] * 1.01 - plot_data[-1, 3] * 0.99, 
                               linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8)
        highlight_idx = [-1]
    
    ax.add_patch(rect)
    
    # Set title and labels
    plt.title(title, fontsize=14)
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    # Add description
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=12)
    
    # Add annotations for key components of the pattern
    if "three" in pattern_name:
        for i, day in enumerate(highlight_idx):
            candle_x = plot_data[day, 0]
            candle_mid = (plot_data[day, 1] + plot_data[day, 4]) / 2
            plt.annotate(f"Candle {i+1}", 
                       xy=(candle_x, candle_mid),
                       xytext=(candle_x - 1.5, candle_mid + (0.01 * data.iloc[0]['Close'])),
                       arrowprops=dict(arrowstyle='->', color='blue'),
                       color='blue')
    elif "star" in pattern_name:
        plt.annotate("First Candle", 
                   xy=(plot_data[-3, 0], (plot_data[-3, 1] + plot_data[-3, 4]) / 2),
                   xytext=(plot_data[-3, 0] - 2, (plot_data[-3, 1] + plot_data[-3, 4]) / 2),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
        plt.annotate("Star", 
                   xy=(plot_data[-2, 0], (plot_data[-2, 1] + plot_data[-2, 4]) / 2),
                   xytext=(plot_data[-2, 0], max(plot_data[-2:, 2]) * 1.02),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
        plt.annotate("Third Candle", 
                   xy=(plot_data[-1, 0], (plot_data[-1, 1] + plot_data[-1, 4]) / 2),
                   xytext=(plot_data[-1, 0] + 2, (plot_data[-1, 1] + plot_data[-1, 4]) / 2),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
    elif "engulfing" in pattern_name:
        plt.annotate("First Candle", 
                   xy=(plot_data[-2, 0], (plot_data[-2, 1] + plot_data[-2, 4]) / 2),
                   xytext=(plot_data[-2, 0] - 2, (plot_data[-2, 1] + plot_data[-2, 4]) / 2),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
        plt.annotate("Engulfing Candle", 
                   xy=(plot_data[-1, 0], (plot_data[-1, 1] + plot_data[-1, 4]) / 2),
                   xytext=(plot_data[-1, 0] + 2, (plot_data[-1, 1] + plot_data[-1, 4]) / 2),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
    elif "harami" in pattern_name:
        plt.annotate("Mother Candle", 
                   xy=(plot_data[-2, 0], (plot_data[-2, 1] + plot_data[-2, 4]) / 2),
                   xytext=(plot_data[-2, 0] - 2, (plot_data[-2, 1] + plot_data[-2, 4]) / 2),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
        plt.annotate("Harami (Inside) Candle", 
                   xy=(plot_data[-1, 0], (plot_data[-1, 1] + plot_data[-1, 4]) / 2),
                   xytext=(plot_data[-1, 0] + 2, (plot_data[-1, 1] + plot_data[-1, 4]) / 2),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
    elif pattern_name == "hammer":
        plt.annotate("Small Body", 
                   xy=(plot_data[-1, 0], (plot_data[-1, 1] + plot_data[-1, 4]) / 2),
                   xytext=(plot_data[-1, 0] - 2, plot_data[-1, 2]),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
        plt.annotate("Long Lower Shadow", 
                   xy=(plot_data[-1, 0], (plot_data[-1, 3] + plot_data[-1, 4]) / 2),
                   xytext=(plot_data[-1, 0] + 2, (plot_data[-1, 3] + plot_data[-1, 4]) / 2),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
    elif pattern_name == "shooting_star":
        plt.annotate("Small Body", 
                   xy=(plot_data[-1, 0], (plot_data[-1, 1] + plot_data[-1, 4]) / 2),
                   xytext=(plot_data[-1, 0] - 2, plot_data[-1, 3]),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
        plt.annotate("Long Upper Shadow", 
                   xy=(plot_data[-1, 0], (plot_data[-1, 1] + plot_data[-1, 2]) / 2),
                   xytext=(plot_data[-1, 0] + 2, (plot_data[-1, 1] + plot_data[-1, 2]) / 2),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(image_dir / f"{pattern_name}.png", dpi=150)
    plt.close()

def create_multiday_pattern_visualization(pattern_name, trend="neutral"):
    """Create and save visualization of a multi-day candlestick pattern"""
    data = generate_prices(trend=trend)
    
    if pattern_name == "three_day_reversal_bottom":
        data = insert_multiday_pattern(data, "three_day_reversal_bottom")
        title = "Three-Day Reversal Bottom Pattern"
        description = "Bullish reversal: Three down days followed by a strong up day with higher volume"
    
    elif pattern_name == "key_reversal":
        data = insert_multiday_pattern(data, "key_reversal")
        title = "Key Reversal Day Pattern"
        description = "Bullish reversal: New low followed by close above previous day's high"
    
    elif pattern_name == "island_reversal":
        data = insert_multiday_pattern(data, "island_reversal")
        title = "Island Reversal Pattern"
        description = "Gap down, isolated trading, gap up - signals strong reversal potential"
    
    elif pattern_name == "three_outside_up":
        data = insert_multiday_pattern(data, "three_outside_up")
        title = "Three Outside Up Pattern"
        description = "Bullish reversal: Bearish candle, bullish engulfing, confirmation candle"
    
    elif pattern_name == "volatility_squeeze":
        data = insert_multiday_pattern(data, "volatility_squeeze")
        title = "Volatility Squeeze Pattern"
        description = "Period of decreasing volatility followed by a volatility expansion and breakout"
    
    # Convert data to format suitable for mplfinance
    df = data.copy()
    df['Date'] = pd.to_datetime([datetime.now() - timedelta(days=30-i) for i in range(len(df))])
    df.set_index('Date', inplace=True)
    
    # Create customized candlestick chart
    fig, ax = plt.subplots(figsize=(12, 6))
    mpf.plot(df, type='candle', style='yahoo', ax=ax, volume=False, show_nontrading=False)
    
    # Add annotations based on pattern type
    if pattern_name == "three_day_reversal_bottom":
        ax.annotate("Down Day 1", 
                   xy=(mdates.date2num(df.index[-4]), df['Low'].iloc[-4]),
                   xytext=(mdates.date2num(df.index[-4]), df['Low'].iloc[-4] * 0.97),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   color='red')
        ax.annotate("Down Day 2", 
                   xy=(mdates.date2num(df.index[-3]), df['Low'].iloc[-3]),
                   xytext=(mdates.date2num(df.index[-3]), df['Low'].iloc[-3] * 0.97),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   color='red')
        ax.annotate("Down Day 3", 
                   xy=(mdates.date2num(df.index[-2]), df['Low'].iloc[-2]),
                   xytext=(mdates.date2num(df.index[-2]), df['Low'].iloc[-2] * 0.97),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   color='red')
        ax.annotate("Reversal Day\n(Higher Volume)", 
                   xy=(mdates.date2num(df.index[-1]), df['High'].iloc[-1]),
                   xytext=(mdates.date2num(df.index[-1]), df['High'].iloc[-1] * 1.03),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   color='green')
    
    elif pattern_name == "key_reversal":
        ax.annotate("Previous Day", 
                   xy=(mdates.date2num(df.index[-2]), df['Close'].iloc[-2]),
                   xytext=(mdates.date2num(df.index[-2]), df['High'].iloc[-2] * 1.03),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
        ax.annotate("New Low", 
                   xy=(mdates.date2num(df.index[-1]), df['Low'].iloc[-1]),
                   xytext=(mdates.date2num(df.index[-1]) - 1, df['Low'].iloc[-1] * 0.98),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   color='red')
        ax.annotate("Close Above\nPrevious High", 
                   xy=(mdates.date2num(df.index[-1]), df['Close'].iloc[-1]),
                   xytext=(mdates.date2num(df.index[-1]) + 1, df['Close'].iloc[-1] * 1.03),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   color='green')
    
    elif pattern_name == "island_reversal":
        ax.annotate("Gap Down", 
                   xy=(mdates.date2num(df.index[-2]) - 0.3, (df['Low'].iloc[-3] + df['High'].iloc[-2])/2),
                   xytext=(mdates.date2num(df.index[-2]) - 1, (df['Low'].iloc[-3] + df['High'].iloc[-2])/2),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   color='red')
        ax.annotate("Island\n(Isolated Trading)", 
                   xy=(mdates.date2num(df.index[-2]), df['Close'].iloc[-2]),
                   xytext=(mdates.date2num(df.index[-2]), df['Low'].iloc[-2] * 0.95),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
        ax.annotate("Gap Up", 
                   xy=(mdates.date2num(df.index[-1]) - 0.3, (df['High'].iloc[-2] + df['Low'].iloc[-1])/2),
                   xytext=(mdates.date2num(df.index[-1]) - 1, (df['High'].iloc[-2] + df['Low'].iloc[-1])/2),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   color='green')
    
    elif pattern_name == "three_outside_up":
        ax.annotate("Bearish Candle", 
                   xy=(mdates.date2num(df.index[-3]), df['Close'].iloc[-3]),
                   xytext=(mdates.date2num(df.index[-3]), df['Low'].iloc[-3] * 0.97),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   color='red')
        ax.annotate("Bullish Engulfing", 
                   xy=(mdates.date2num(df.index[-2]), df['Close'].iloc[-2]),
                   xytext=(mdates.date2num(df.index[-2]) - 1, df['High'].iloc[-2] * 1.03),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   color='green')
        ax.annotate("Confirmation Candle", 
                   xy=(mdates.date2num(df.index[-1]), df['Close'].iloc[-1]),
                   xytext=(mdates.date2num(df.index[-1]) + 1, df['Close'].iloc[-1] * 1.03),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   color='green')
    
    elif pattern_name == "volatility_squeeze":
        ax.annotate("Decreasing Volatility", 
                   xy=(mdates.date2num(df.index[-4]), df['Close'].iloc[-4]),
                   xytext=(mdates.date2num(df.index[-4]) - 2, df['Close'].iloc[-4] * 1.03),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   color='blue')
        ax.annotate("Breakout on\nHigher Volume", 
                   xy=(mdates.date2num(df.index[-1]), df['High'].iloc[-1]),
                   xytext=(mdates.date2num(df.index[-1]) + 1, df['High'].iloc[-1] * 1.03),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   color='green')
    
    # Set title and labels
    plt.suptitle(title, fontsize=14)
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=12)
    
    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(image_dir / f"{pattern_name}.png", dpi=150)
    plt.close()

def create_technical_indicator_visualization(indicator_name):
    """Create and save visualization of a technical indicator"""
    # Generate some price data
    days = 100
    data = generate_prices(days=days, trend="neutral" if "bands" in indicator_name else "uptrend")
    dates = data['Date'].values
    
    if indicator_name == "bollinger_bands":
        # Calculate Bollinger Bands
        window = 20
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, data['Close'], label='Price', color='blue')
        ax.plot(dates, rolling_mean, label='20-day SMA (Middle Band)', color='red')
        ax.plot(dates, upper_band, label='Upper Band (+2σ)', color='green', linestyle='--')
        ax.plot(dates, lower_band, label='Lower Band (-2σ)', color='green', linestyle='--')
        
        # Fill between bands
        ax.fill_between(dates, upper_band, lower_band, color='green', alpha=0.1)
        
        # Annotate bands
        mid_idx = len(dates) // 2
        ax.annotate('Upper Band: Price + 2σ',
                  xy=(dates[mid_idx], upper_band.iloc[mid_idx]),
                  xytext=(dates[mid_idx], upper_band.iloc[mid_idx] * 1.05),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  color='black')
        
        ax.annotate('Middle Band: 20-day SMA',
                  xy=(dates[mid_idx], rolling_mean.iloc[mid_idx]),
                  xytext=(dates[mid_idx-10], rolling_mean.iloc[mid_idx] * 1.1),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  color='black')
        
        ax.annotate('Lower Band: Price - 2σ',
                  xy=(dates[mid_idx], lower_band.iloc[mid_idx]),
                  xytext=(dates[mid_idx], lower_band.iloc[mid_idx] * 0.95),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  color='black')
        
        # Highlight a squeeze and expansion
        squeeze_idx = data['Close'].rolling(window=10).std().iloc[30:60].idxmin()
        ax.annotate('Bollinger Squeeze\n(Low Volatility)',
                  xy=(dates[squeeze_idx-days], data['Close'].iloc[squeeze_idx-days]),
                  xytext=(dates[squeeze_idx-days-10], data['Close'].iloc[squeeze_idx-days] * 1.1),
                  arrowprops=dict(arrowstyle='->', color='red'),
                  color='red')
        
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
        ax.plot(dates, data['Close'], label='Price', color='blue')
        ax.plot(dates, ema, label='20-day EMA (Middle Line)', color='red')
        ax.plot(dates, upper_channel, label='Upper Channel (EMA + 2×ATR)', color='purple', linestyle='--')
        ax.plot(dates, lower_channel, label='Lower Channel (EMA - 2×ATR)', color='purple', linestyle='--')
        
        # Fill between channels
        ax.fill_between(dates, upper_channel, lower_channel, color='purple', alpha=0.1)
        
        # Annotate channels
        mid_idx = len(dates) // 2
        ax.annotate('Upper Channel: EMA + 2×ATR',
                  xy=(dates[mid_idx], upper_channel.iloc[mid_idx]),
                  xytext=(dates[mid_idx], upper_channel.iloc[mid_idx] * 1.05),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  color='black')
        
        ax.annotate('Middle Line: 20-day EMA',
                  xy=(dates[mid_idx], ema.iloc[mid_idx]),
                  xytext=(dates[mid_idx-10], ema.iloc[mid_idx] * 1.1),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  color='black')
        
        ax.annotate('Lower Channel: EMA - 2×ATR',
                  xy=(dates[mid_idx], lower_channel.iloc[mid_idx]),
                  xytext=(dates[mid_idx], lower_channel.iloc[mid_idx] * 0.95),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  color='black')
        
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
        ax1.plot(dates, data['Close'], label='Price', color='blue')
        ax1.set_ylabel('Price')
        
        # ATR in bottom panel
        ax2.plot(dates, atr, label='14-day ATR', color='red')
        ax2.set_ylabel('ATR')
        
        # Fill ATR for visualization
        ax2.fill_between(dates, 0, atr, color='red', alpha=0.2)
        
        # Annotations
        # Find a period of increasing volatility
        vol_increase_idx = atr.diff().iloc[30:70].idxmax()
        ax1.annotate('Price Volatility Increases',
                   xy=(dates[vol_increase_idx-days], data['Close'].iloc[vol_increase_idx-days]),
                   xytext=(dates[vol_increase_idx-days-10], data['Close'].iloc[vol_increase_idx-days] * 1.1),
                   arrowprops=dict(arrowstyle='->', color='purple'),
                   color='purple')
        
        ax2.annotate('ATR Increases',
                   xy=(dates[vol_increase_idx-days], atr.iloc[vol_increase_idx-days]),
                   xytext=(dates[vol_increase_idx-days-10], atr.iloc[vol_increase_idx-days] * 1.5),
                   arrowprops=dict(arrowstyle='->', color='purple'),
                   color='purple')
        
        # Add description
        title = "Average True Range (ATR)"
        description = "Measures market volatility by decomposing the entire range of an asset price for that period"
        
    elif indicator_name == "support_resistance":
        # Generate data with clearer support/resistance zones
        data = generate_prices(days=days, trend="neutral", volatility=0.01)
        dates = data['Date'].values
        
        # Create artificial support/resistance levels
        support_level = data['Close'].min() * 1.02
        resistance_level = data['Close'].max() * 0.98
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, data['Close'], label='Price', color='blue')
        
        # Draw support and resistance lines
        ax.axhline(y=support_level, color='green', linestyle='-', linewidth=2, label='Support')
        ax.axhline(y=resistance_level, color='red', linestyle='-', linewidth=2, label='Resistance')
        
        # Highlight areas where price interacts with support/resistance
        # Find points near support
        support_touches = []
        for i in range(len(data)):
            if abs(data['Low'].iloc[i] - support_level) / support_level < 0.005:
                support_touches.append(i)
        
        # Find points near resistance
        resistance_touches = []
        for i in range(len(data)):
            if abs(data['High'].iloc[i] - resistance_level) / resistance_level < 0.005:
                resistance_touches.append(i)
        
        # Annotate support/resistance touches
        if support_touches:
            idx = support_touches[0]
            ax.annotate('Support Test',
                      xy=(dates[idx], support_level),
                      xytext=(dates[idx-5], support_level * 0.95),
                      arrowprops=dict(arrowstyle='->', color='green'),
                      color='green')
        
        if resistance_touches:
            idx = resistance_touches[0]
            ax.annotate('Resistance Test',
                      xy=(dates[idx], resistance_level),
                      xytext=(dates[idx-5], resistance_level * 1.05),
                      arrowprops=dict(arrowstyle='->', color='red'),
                      color='red')
        
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
        ax1.plot(dates, data['Close'], label='Price', color='blue')
        ax1.set_ylabel('Price')
        
        # MACD in bottom panel
        ax2.plot(dates, macd_line, label='MACD Line (12-26)', color='blue')
        ax2.plot(dates, signal_line, label='Signal Line (9)', color='red')
        
        # Plot histogram
        for i in range(len(histogram)):
            if histogram.iloc[i] >= 0:
                ax2.bar(dates[i], histogram.iloc[i], color='green', width=1, alpha=0.5)
            else:
                ax2.bar(dates[i], histogram.iloc[i], color='red', width=1, alpha=0.5)
        
        # Add zero line
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Annotations
        # Find a crossover
        for i in range(1, len(macd_line)-1):
            if macd_line.iloc[i-1] < signal_line.iloc[i-1] and macd_line.iloc[i] > signal_line.iloc[i]:
                ax2.annotate('Bullish Crossover',
                           xy=(dates[i], macd_line.iloc[i]),
                           xytext=(dates[i-10], macd_line.iloc[i] * 3),
                           arrowprops=dict(arrowstyle='->', color='green'),
                           color='green')
                break
                
        # Find a divergence
        ax2.annotate('Histogram (MACD - Signal)',
                   xy=(dates[50], histogram.iloc[50]),
                   xytext=(dates[45], histogram.iloc[50] * 5),
                   arrowprops=dict(arrowstyle='->', color='purple'),
                   color='purple')
        
        # Add description
        title = "Moving Average Convergence Divergence (MACD)"
        description = "Trend-following momentum indicator showing the relationship between two moving averages"
    
    # Set title and labels
    plt.suptitle(title, fontsize=14)
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=12)
    
    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
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
    
    print("\nGenerating multi-day candlestick pattern visualizations...")
    
    # List of multi-day patterns to visualize
    multiday_patterns = [
        "three_day_reversal_bottom",
        "key_reversal",
        "island_reversal",
        "three_outside_up",
        "volatility_squeeze"
    ]
    
    # Create multi-day pattern visualizations
    for pattern in multiday_patterns:
        print(f"Creating visualization for {pattern}...")
        create_multiday_pattern_visualization(pattern)
    
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
