#!/usr/bin/env python3
"""
Generate comprehensive analysis and visualizations for RELIANCE.NS
This script creates detailed analysis for documentation purposes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from download_stock_data import download_stock_data
from technical_indicators import add_technical_indicators
from relative_strength import calculate_relative_strength_metrics
from market_regime import MarketRegimeAnalyzer, analyze_volatility_regime
import yfinance as yf

def analyze_reliance():
    """Comprehensive analysis of RELIANCE.NS for documentation"""
    
    print("🔍 Starting RELIANCE.NS Comprehensive Analysis...")
    print("=" * 60)
    
    # Check if we have existing test results
    test_results_path = "data/test_results/RELIANCE_comprehensive_test.parquet"
    
    if os.path.exists(test_results_path):
        print("📂 Loading existing RELIANCE test results...")
        data_with_indicators = pd.read_parquet(test_results_path)
        
        # Set Date column as index if it exists
        if 'Date' in data_with_indicators.columns:
            data_with_indicators['Date'] = pd.to_datetime(data_with_indicators['Date'])
            data_with_indicators = data_with_indicators.set_index('Date')
            
        print(f"✅ Loaded {len(data_with_indicators)} trading days with indicators")
        print(f"📅 Date range: {data_with_indicators.index[0].strftime('%Y-%m-%d')} to {data_with_indicators.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Latest close price: ₹{float(data_with_indicators['Close'].iloc[-1]):.2f}")
        print(f"📈 52-week high: ₹{float(data_with_indicators['High'].max()):.2f}")
        print(f"📉 52-week low: ₹{float(data_with_indicators['Low'].min()):.2f}")
        
    else:
        # 1. Download RELIANCE data
        print("\n📊 1. DOWNLOADING RELIANCE.NS DATA")
        print("-" * 40)
        
        symbol = "RELIANCE.NS"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        
        # Get basic data
        data = download_stock_data(
            symbol, 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("❌ Failed to download data")
            return
        
        print(f"✅ Downloaded {len(data)} trading days")
        print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Latest close price: ₹{float(data['Close'].iloc[-1]):.2f}")
        print(f"📈 52-week high: ₹{float(data['High'].max()):.2f}")
        print(f"📉 52-week low: ₹{float(data['Low'].min()):.2f}")
        
        # 2. Add technical indicators
        print("\n🔧 2. CALCULATING TECHNICAL INDICATORS")
        print("-" * 40)
        
        data_with_indicators = add_technical_indicators(data.copy())
    
    print(f"📊 Added technical indicators, total columns: {len(data_with_indicators.columns)}")
    
    # Get latest values for key indicators
    latest = data_with_indicators.iloc[-1]
    
    print(f"\n📈 LATEST TECHNICAL INDICATORS (as of {latest.name.strftime('%Y-%m-%d')}):")
    print(f"   RSI (14):           {latest.get('RSI', 'N/A'):.2f}")
    print(f"   MACD:               {latest.get('MACD', 'N/A'):.4f}")
    print(f"   MACD Signal:        {latest.get('MACD_Signal', 'N/A'):.4f}")
    print(f"   SMA 20:             ₹{latest.get('SMA_20', 0):.2f}")
    print(f"   SMA 50:             ₹{latest.get('SMA_50', 0):.2f}")
    print(f"   SMA 200:            ₹{latest.get('SMA_200', 0):.2f}")
    print(f"   Bollinger Upper:    ₹{latest.get('Bollinger_Upper', 0):.2f}")
    print(f"   Bollinger Lower:    ₹{latest.get('Bollinger_Lower', 0):.2f}")
    print(f"   ATR:                ₹{latest.get('ATR', 0):.2f}")
    print(f"   ADX:                {latest.get('ADX', 'N/A'):.2f}")
    print(f"   Volume Ratio (20):  {latest.get('Volume_Ratio_20', 0):.2f}")
    
    # Check for recent patterns
    print(f"\n🕯️ RECENT CANDLESTICK PATTERNS:")
    recent_data = data_with_indicators.tail(10)
    patterns = ['Doji', 'Hammer', 'Engulfing', 'Morning_Star', 'Evening_Star', 'Shooting_Star']
    for pattern in patterns:
        if pattern in recent_data.columns:
            pattern_days = recent_data[recent_data[pattern] != 0]
            if not pattern_days.empty:
                print(f"   {pattern}: {len(pattern_days)} occurrences in last 10 days")
                for date, row in pattern_days.iterrows():
                    print(f"      → {date.strftime('%Y-%m-%d')}: ₹{row['Close']:.2f}")
    
    # 3. Relative Strength Analysis
    print("\n📊 3. RELATIVE STRENGTH ANALYSIS")
    print("-" * 40)
    
    # Note: Skipping relative strength calculation as it requires market index data
    print(f"🎯 RELATIVE STRENGTH METRICS:")
    print(f"   ⚠️  Requires market index data (Nifty 50, sector indices)")
    print(f"   📊 Available in full pipeline with market data download")
    
    # 4. Market Regime Analysis
    print("\n🌊 4. MARKET REGIME ANALYSIS")
    print("-" * 40)
    
    analyzer = MarketRegimeAnalyzer()
    regime = analyzer.detect_regime(data_with_indicators)
    print(f"📋 CURRENT MARKET REGIME:")
    print(f"   Trend:              {regime.trend}")
    print(f"   Volatility:         {regime.volatility}")
    print(f"   Volume:             {regime.volume}")
    print(f"   Momentum:           {regime.momentum}")
    print(f"   Confidence:         {regime.confidence:.2f}")
    
    # 5. Generate Visualization
    print("\n📈 5. GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    # Create a comprehensive chart
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle('RELIANCE.NS - Comprehensive Technical Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Price and Moving Averages
    ax1 = axes[0]
    ax1.plot(data_with_indicators.index, data_with_indicators['Close'], label='Close Price', linewidth=2)
    if 'SMA_20' in data_with_indicators.columns:
        ax1.plot(data_with_indicators.index, data_with_indicators['SMA_20'], label='SMA 20', alpha=0.7)
    if 'SMA_50' in data_with_indicators.columns:
        ax1.plot(data_with_indicators.index, data_with_indicators['SMA_50'], label='SMA 50', alpha=0.7)
    if 'SMA_200' in data_with_indicators.columns:
        ax1.plot(data_with_indicators.index, data_with_indicators['SMA_200'], label='SMA 200', alpha=0.7)
    
    # Add Bollinger Bands
    if all(col in data_with_indicators.columns for col in ['Bollinger_Upper', 'Bollinger_Lower']):
        ax1.fill_between(data_with_indicators.index, 
                        data_with_indicators['Bollinger_Upper'], 
                        data_with_indicators['Bollinger_Lower'], 
                        alpha=0.1, color='gray', label='Bollinger Bands')
    
    ax1.set_title('Price Chart with Moving Averages & Bollinger Bands')
    ax1.set_ylabel('Price (₹)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RSI
    ax2 = axes[1]
    if 'RSI' in data_with_indicators.columns:
        ax2.plot(data_with_indicators.index, data_with_indicators['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.fill_between(data_with_indicators.index, 30, 70, alpha=0.1, color='yellow')
    
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MACD
    ax3 = axes[2]
    if all(col in data_with_indicators.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
        ax3.plot(data_with_indicators.index, data_with_indicators['MACD'], label='MACD', color='blue')
        ax3.plot(data_with_indicators.index, data_with_indicators['MACD_Signal'], label='Signal', color='red')
        ax3.bar(data_with_indicators.index, data_with_indicators['MACD_Hist'], 
               label='Histogram', alpha=0.3, color='green')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax3.set_title('MACD (Moving Average Convergence Divergence)')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volume
    ax4 = axes[3]
    ax4.bar(data_with_indicators.index, data_with_indicators['Volume'], alpha=0.6, color='lightblue', label='Volume')
    if 'Volume_SMA_20' in data_with_indicators.columns:
        ax4.plot(data_with_indicators.index, data_with_indicators['Volume_SMA_20'], 
                color='red', label='Volume SMA 20')
    
    ax4.set_title('Trading Volume')
    ax4.set_ylabel('Volume')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis for all subplots
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('docs/images/reliance_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved comprehensive chart: docs/images/reliance_comprehensive_analysis.png")
    
    # 6. Generate specific pattern examples
    print("\n🔍 6. ANALYZING SPECIFIC PATTERNS")
    print("-" * 40)
    
    # Find hammer patterns
    if 'Hammer' in data_with_indicators.columns:
        hammer_days = data_with_indicators[data_with_indicators['Hammer'] != 0].tail(5)
        if not hammer_days.empty:
            print(f"🔨 RECENT HAMMER PATTERNS:")
            for date, row in hammer_days.iterrows():
                print(f"   📅 {date.strftime('%Y-%m-%d')}: Close ₹{row['Close']:.2f}, "
                      f"Range ₹{row['High'] - row['Low']:.2f}")
    
    # Find engulfing patterns
    if 'Engulfing' in data_with_indicators.columns:
        engulfing_days = data_with_indicators[data_with_indicators['Engulfing'] != 0].tail(5)
        if not engulfing_days.empty:
            print(f"📊 RECENT ENGULFING PATTERNS:")
            for date, row in engulfing_days.iterrows():
                pattern_type = "Bullish" if row['Engulfing'] > 0 else "Bearish"
                print(f"   📅 {date.strftime('%Y-%m-%d')}: {pattern_type} Engulfing, "
                      f"Close ₹{row['Close']:.2f}")
    
    # 7. Risk Analysis
    print("\n⚠️  7. RISK ANALYSIS")
    print("-" * 40)
    
    # Calculate risk metrics
    returns = data_with_indicators['Close'].pct_change().dropna()
    
    print(f"📊 RISK METRICS (Annualized):")
    print(f"   Daily Volatility:   {returns.std():.4f} ({returns.std() * np.sqrt(252):.2%})")
    print(f"   Average Return:     {returns.mean():.4f} ({returns.mean() * 252:.2%})")
    print(f"   Sharpe Ratio:       {(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}")
    
    # Value at Risk (5%)
    var_5 = np.percentile(returns, 5)
    print(f"   VaR (5%):          {var_5:.4f} ({var_5:.2%})")
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    print(f"   Max Drawdown:      {max_dd:.2%}")
    
    # 8. Current Signals
    print("\n🚦 8. CURRENT TRADING SIGNALS")
    print("-" * 40)
    
    signals = []
    
    # RSI signals
    current_rsi = latest.get('RSI', 50)
    if current_rsi > 70:
        signals.append("🔴 RSI Overbought (>70) - Potential Sell Signal")
    elif current_rsi < 30:
        signals.append("🟢 RSI Oversold (<30) - Potential Buy Signal")
    else:
        signals.append(f"🟡 RSI Neutral ({current_rsi:.1f}) - No Strong Signal")
    
    # MACD signals
    current_macd = latest.get('MACD', 0)
    current_signal = latest.get('MACD_Signal', 0)
    if current_macd > current_signal:
        signals.append("🟢 MACD Above Signal - Bullish Momentum")
    else:
        signals.append("🔴 MACD Below Signal - Bearish Momentum")
    
    # Moving average signals
    current_price = latest.get('Close', 0)
    sma_20 = latest.get('SMA_20', 0)
    sma_50 = latest.get('SMA_50', 0)
    
    if current_price > sma_20 > sma_50:
        signals.append("🟢 Price Above Short-term MAs - Uptrend")
    elif current_price < sma_20 < sma_50:
        signals.append("🔴 Price Below Short-term MAs - Downtrend")
    else:
        signals.append("🟡 Mixed Moving Average Signals")
    
    for signal in signals:
        print(f"   {signal}")
    
    print(f"\n✅ Analysis Complete!")
    print("=" * 60)
    
    return data_with_indicators

if __name__ == "__main__":
    analyze_reliance()
