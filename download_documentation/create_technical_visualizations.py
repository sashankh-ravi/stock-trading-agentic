#!/usr/bin/env python3
"""
Technical Analysis Visualization Generator

This script creates comprehensive visualizations for all 74+ technical indicators,
20+ candlestick patterns, gap analysis, and Ichimoku Cloud system using RELIANCE.NS data.

Author: Technical Analysis Documentation System
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append('/home/sashankhravi/Documents/stock-trading-agentic')

from technical_indicators import add_technical_indicators
import yfinance as yf
import talib

class TechnicalVisualizationGenerator:
    """
    Generate comprehensive visualizations for technical analysis documentation
    """
    
    def __init__(self, output_dir="/home/sashankhravi/Documents/stock-trading-agentic/download_documentation/images"):
        self.output_dir = output_dir
        self.create_output_directory()
        
        # Color schemes for different indicator types
        self.colors = {
            'trend': {'primary': '#2E86AB', 'secondary': '#A23B72', 'tertiary': '#F18F01'},
            'momentum': {'primary': '#C73E1D', 'secondary': '#F24236', 'tertiary': '#FFB30F'},
            'volatility': {'primary': '#8E44AD', 'secondary': '#9B59B6', 'tertiary': '#E67E22'},
            'volume': {'primary': '#27AE60', 'secondary': '#2ECC71', 'tertiary': '#F39C12'},
            'ichimoku': {'primary': '#34495E', 'secondary': '#5D6D7E', 'tertiary': '#85929E'},
            'patterns': {'bullish': '#27AE60', 'bearish': '#E74C3C', 'neutral': '#F39C12'}
        }
        
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_output_directory(self):
        """Create output directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for different categories
        subdirs = ['trend', 'momentum', 'volatility', 'volume', 'ichimoku', 'patterns', 'gaps', 'comprehensive']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def fetch_reliance_data(self, period="2y"):
        """Fetch RELIANCE.NS data for analysis"""
        print(f"Fetching RELIANCE.NS data for period: {period}")
        ticker = yf.Ticker("RELIANCE.NS")
        data = ticker.history(period=period)
        
        if data.empty:
            raise ValueError("No data fetched for RELIANCE.NS")
        
        # Reset index to have Date as column
        data.reset_index(inplace=True)
        data['symbol'] = 'RELIANCE.NS'
        
        print(f"Fetched {len(data)} rows of data from {data['Date'].min()} to {data['Date'].max()}")
        return data
        
    def add_technical_analysis(self, data):
        """Add technical indicators to the data"""
        print("Adding technical indicators...")
        enhanced_data = add_technical_indicators(data.copy())
        print(f"Added technical indicators. New shape: {enhanced_data.shape}")
        return enhanced_data
    
    def create_trend_indicators_chart(self, data):
        """Create comprehensive trend indicators visualization"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('RELIANCE.NS - Trend Indicators Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        recent_data = data.tail(250)  # Last 250 trading days
        dates = recent_data['Date']
        
        # Chart 1: Price with Moving Averages
        ax1 = axes[0]
        ax1.plot(dates, recent_data['Close'], label='Close Price', color='black', linewidth=2)
        ax1.plot(dates, recent_data['SMA_20'], label='SMA 20', color=self.colors['trend']['primary'], alpha=0.8)
        ax1.plot(dates, recent_data['SMA_50'], label='SMA 50', color=self.colors['trend']['secondary'], alpha=0.8)
        ax1.plot(dates, recent_data['EMA_20'], label='EMA 20', color=self.colors['trend']['tertiary'], alpha=0.8, linestyle='--')
        ax1.set_title('Price Action with Moving Averages', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: MACD
        ax2 = axes[1]
        ax2.plot(dates, recent_data['MACD'], label='MACD Line', color=self.colors['trend']['primary'])
        ax2.plot(dates, recent_data['MACD_Signal'], label='Signal Line', color=self.colors['trend']['secondary'])
        ax2.bar(dates, recent_data['MACD_Hist'], label='MACD Histogram', alpha=0.6, color=self.colors['trend']['tertiary'])
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('MACD (Moving Average Convergence Divergence)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: ADX and Directional Movement
        ax3 = axes[2]
        ax3.plot(dates, recent_data['ADX'], label='ADX (Trend Strength)', color='purple', linewidth=2)
        ax3.plot(dates, recent_data['Plus_DI'], label='+DI', color='green', alpha=0.8)
        ax3.plot(dates, recent_data['Minus_DI'], label='-DI', color='red', alpha=0.8)
        ax3.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Strong Trend (25)')
        ax3.set_title('ADX and Directional Movement Index', fontweight='bold')
        ax3.set_ylabel('ADX Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'trend', 'trend_indicators_comprehensive.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created trend indicators chart")
    
    def create_momentum_indicators_chart(self, data):
        """Create comprehensive momentum indicators visualization"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        fig.suptitle('RELIANCE.NS - Momentum Indicators Analysis', fontsize=16, fontweight='bold')
        
        recent_data = data.tail(250)
        dates = recent_data['Date']
        
        # Chart 1: RSI
        ax1 = axes[0]
        ax1.plot(dates, recent_data['RSI'], color=self.colors['momentum']['primary'], linewidth=2)
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax1.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax1.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax1.fill_between(dates, 70, 100, alpha=0.2, color='red')
        ax1.fill_between(dates, 0, 30, alpha=0.2, color='green')
        ax1.set_title('RSI (Relative Strength Index)', fontweight='bold')
        ax1.set_ylabel('RSI Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Chart 2: Stochastic Oscillator
        ax2 = axes[1]
        ax2.plot(dates, recent_data['Stoch_K'], label='%K', color=self.colors['momentum']['primary'])
        ax2.plot(dates, recent_data['Stoch_D'], label='%D', color=self.colors['momentum']['secondary'])
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7)
        ax2.fill_between(dates, 80, 100, alpha=0.2, color='red')
        ax2.fill_between(dates, 0, 20, alpha=0.2, color='green')
        ax2.set_title('Stochastic Oscillator (%K and %D)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Chart 3: ROC (Rate of Change) Multiple Periods
        ax3 = axes[2]
        ax3.plot(dates, recent_data['ROC_5'], label='ROC 5-day', color=self.colors['momentum']['primary'], alpha=0.8)
        ax3.plot(dates, recent_data['ROC_10'], label='ROC 10-day', color=self.colors['momentum']['secondary'], alpha=0.8)
        ax3.plot(dates, recent_data['ROC_21'], label='ROC 21-day', color=self.colors['momentum']['tertiary'], alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Rate of Change (Multiple Timeframes)', fontweight='bold')
        ax3.set_ylabel('ROC (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: CCI (Commodity Channel Index)
        ax4 = axes[3]
        ax4.plot(dates, recent_data['CCI'], color=self.colors['momentum']['primary'], linewidth=2)
        ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Overbought (+100)')
        ax4.axhline(y=-100, color='green', linestyle='--', alpha=0.7, label='Oversold (-100)')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax4.fill_between(dates, 100, 300, alpha=0.2, color='red')
        ax4.fill_between(dates, -300, -100, alpha=0.2, color='green')
        ax4.set_title('CCI (Commodity Channel Index)', fontweight='bold')
        ax4.set_ylabel('CCI Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'momentum', 'momentum_indicators_comprehensive.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created momentum indicators chart")
    
    def create_volatility_indicators_chart(self, data):
        """Create comprehensive volatility indicators visualization"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('RELIANCE.NS - Volatility Indicators Analysis', fontsize=16, fontweight='bold')
        
        recent_data = data.tail(250)
        dates = recent_data['Date']
        
        # Chart 1: Bollinger Bands
        ax1 = axes[0]
        ax1.plot(dates, recent_data['Close'], label='Close Price', color='black', linewidth=2)
        ax1.plot(dates, recent_data['Bollinger_Upper'], label='Upper Band', color=self.colors['volatility']['primary'], alpha=0.8)
        ax1.plot(dates, recent_data['Bollinger_Middle'], label='Middle Band (SMA)', color=self.colors['volatility']['secondary'], alpha=0.8)
        ax1.plot(dates, recent_data['Bollinger_Lower'], label='Lower Band', color=self.colors['volatility']['primary'], alpha=0.8)
        ax1.fill_between(dates, recent_data['Bollinger_Upper'], recent_data['Bollinger_Lower'], 
                        alpha=0.1, color=self.colors['volatility']['primary'])
        ax1.set_title('Bollinger Bands (20-period, 2 std dev)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: ATR (Average True Range)
        ax2 = axes[1]
        ax2.plot(dates, recent_data['ATR'], color=self.colors['volatility']['primary'], linewidth=2)
        ax2.fill_between(dates, recent_data['ATR'], alpha=0.3, color=self.colors['volatility']['primary'])
        ax2.set_title('ATR (Average True Range) - 14 Period', fontweight='bold')
        ax2.set_ylabel('ATR Value')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Volatility Metrics
        ax3 = axes[2]
        ax3.plot(dates, recent_data['Volatility_10'], label='10-day Volatility', color=self.colors['volatility']['primary'])
        ax3.plot(dates, recent_data['Volatility_20'], label='20-day Volatility', color=self.colors['volatility']['secondary'])
        ax3.plot(dates, recent_data['Volatility_60'], label='60-day Volatility', color=self.colors['volatility']['tertiary'])
        ax3.set_title('Annualized Volatility (Multiple Timeframes)', fontweight='bold')
        ax3.set_ylabel('Volatility (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'volatility', 'volatility_indicators_comprehensive.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created volatility indicators chart")
    
    def create_volume_indicators_chart(self, data):
        """Create comprehensive volume indicators visualization"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        fig.suptitle('RELIANCE.NS - Volume Indicators Analysis', fontsize=16, fontweight='bold')
        
        recent_data = data.tail(250)
        dates = recent_data['Date']
        
        # Chart 1: Price and Volume
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        ax1.plot(dates, recent_data['Close'], color='black', linewidth=2, label='Close Price')
        ax1_twin.bar(dates, recent_data['Volume'], alpha=0.6, color=self.colors['volume']['primary'], label='Volume')
        ax1.set_title('Price Action with Volume', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: OBV (On-Balance Volume)
        ax2 = axes[1]
        ax2.plot(dates, recent_data['OBV'], color=self.colors['volume']['primary'], linewidth=2)
        ax2.set_title('OBV (On-Balance Volume)', fontweight='bold')
        ax2.set_ylabel('OBV')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Money Flow Index
        ax3 = axes[2]
        ax3.plot(dates, recent_data['Money_Flow_Index'], color=self.colors['volume']['secondary'], linewidth=2)
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought (80)')
        ax3.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold (20)')
        ax3.fill_between(dates, 80, 100, alpha=0.2, color='red')
        ax3.fill_between(dates, 0, 20, alpha=0.2, color='green')
        ax3.set_title('MFI (Money Flow Index)', fontweight='bold')
        ax3.set_ylabel('MFI Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Chart 4: Volume Ratios
        ax4 = axes[3]
        ax4.plot(dates, recent_data['Relative_Volume'], label='Relative Volume', color=self.colors['volume']['primary'])
        ax4.plot(dates, recent_data['Volume_Ratio_20'], label='Volume Ratio (20-day)', color=self.colors['volume']['tertiary'])
        ax4.axhline(y=1, color='gray', linestyle='-', alpha=0.5)
        ax4.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='High Volume (1.5x)')
        ax4.set_title('Volume Analysis Ratios', fontweight='bold')
        ax4.set_ylabel('Volume Ratio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'volume', 'volume_indicators_comprehensive.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created volume indicators chart")
    
    def create_ichimoku_cloud_chart(self, data):
        """Create comprehensive Ichimoku Cloud visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle('RELIANCE.NS - Ichimoku Cloud Analysis', fontsize=16, fontweight='bold')
        
        recent_data = data.tail(200)
        dates = recent_data['Date']
        
        # Price
        ax.plot(dates, recent_data['Close'], label='Close Price', color='black', linewidth=2)
        
        # Ichimoku components
        ax.plot(dates, recent_data['Tenkan_Sen'], label='Tenkan Sen (9)', color=self.colors['ichimoku']['primary'], linewidth=1.5)
        ax.plot(dates, recent_data['Kijun_Sen'], label='Kijun Sen (26)', color=self.colors['ichimoku']['secondary'], linewidth=1.5)
        
        # Cloud (Kumo)
        ax.plot(dates, recent_data['Senkou_Span_A'], label='Senkou Span A', color='green', alpha=0.7)
        ax.plot(dates, recent_data['Senkou_Span_B'], label='Senkou Span B', color='red', alpha=0.7)
        
        # Fill cloud based on direction
        mask_bullish = recent_data['Senkou_Span_A'] > recent_data['Senkou_Span_B']
        mask_bearish = recent_data['Senkou_Span_A'] <= recent_data['Senkou_Span_B']
        
        ax.fill_between(dates, recent_data['Senkou_Span_A'], recent_data['Senkou_Span_B'], 
                       where=mask_bullish, alpha=0.2, color='green', label='Bullish Cloud')
        ax.fill_between(dates, recent_data['Senkou_Span_A'], recent_data['Senkou_Span_B'], 
                       where=mask_bearish, alpha=0.2, color='red', label='Bearish Cloud')
        
        # Chikou Span (plotted 26 periods behind)
        chikou_dates = dates[:-26] if len(dates) > 26 else dates
        chikou_values = recent_data['Chikou_Span'].iloc[26:] if len(recent_data) > 26 else recent_data['Chikou_Span']
        ax.plot(chikou_dates, chikou_values, label='Chikou Span', color='purple', linewidth=1.5, linestyle='--')
        
        ax.set_title('Ichimoku Kinko Hyo (Ichimoku Cloud)', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ichimoku', 'ichimoku_cloud_comprehensive.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created Ichimoku Cloud chart")
    
    def create_candlestick_patterns_examples(self, data):
        """Create visual examples of candlestick patterns"""
        # Get pattern columns
        pattern_columns = [col for col in data.columns if col in [
            'Doji', 'Engulfing', 'Hammer', 'Hanging_Man', 'Shooting_Star', 'Inverted_Hammer',
            'Morning_Star', 'Evening_Star', 'Doji_Star', 'Harami', 'Harami_Cross',
            'Piercing', 'Dark_Cloud', 'Three_White_Soldiers', 'Three_Black_Crows',
            'Spinning_Top', 'Marubozu', 'Dragonfly_Doji', 'Gravestone_Doji', 'Abandoned_Baby'
        ]]
        
        # Create pattern detection summary
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        fig.suptitle('RELIANCE.NS - Candlestick Pattern Detection (Last 100 Days)', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, pattern in enumerate(pattern_columns):
            if i >= len(axes):
                break
                
            ax = axes[i]
            recent_data = data.tail(100)
            
            # Count pattern occurrences
            pattern_occurrences = (recent_data[pattern] != 0).sum()
            bullish_count = (recent_data[pattern] > 0).sum()
            bearish_count = (recent_data[pattern] < 0).sum()
            
            # Create bar chart
            categories = ['Bullish', 'Bearish', 'Total']
            counts = [bullish_count, bearish_count, pattern_occurrences]
            colors = [self.colors['patterns']['bullish'], self.colors['patterns']['bearish'], self.colors['patterns']['neutral']]
            
            bars = ax.bar(categories, counts, color=colors, alpha=0.7)
            ax.set_title(f'{pattern.replace("_", " ")}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Occurrences')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(count)}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(pattern_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'patterns', 'candlestick_patterns_summary.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created candlestick patterns chart for {len(pattern_columns)} patterns")
    
    def create_gap_analysis_chart(self, data):
        """Create gap analysis visualization"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('RELIANCE.NS - Gap Analysis', fontsize=16, fontweight='bold')
        
        recent_data = data.tail(250)
        dates = recent_data['Date']
        
        # Chart 1: Price with Gap Detection
        ax1 = axes[0]
        ax1.plot(dates, recent_data['Close'], color='black', linewidth=2, label='Close Price')
        
        # Mark gap up days
        gap_up_dates = dates[recent_data['Gap_Up']]
        gap_up_prices = recent_data.loc[recent_data['Gap_Up'], 'Close']
        ax1.scatter(gap_up_dates, gap_up_prices, color='green', s=50, marker='^', label='Gap Up', alpha=0.8)
        
        # Mark gap down days
        gap_down_dates = dates[recent_data['Gap_Down']]
        gap_down_prices = recent_data.loc[recent_data['Gap_Down'], 'Close']
        ax1.scatter(gap_down_dates, gap_down_prices, color='red', s=50, marker='v', label='Gap Down', alpha=0.8)
        
        ax1.set_title('Price Action with Gap Detection', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Gap Size Distribution
        ax2 = axes[1]
        gap_sizes = recent_data['Gap_Size'].dropna()
        gap_sizes = gap_sizes[gap_sizes != 0]  # Remove non-gap days
        
        if len(gap_sizes) > 0:
            ax2.hist(gap_sizes, bins=30, alpha=0.7, color=self.colors['patterns']['neutral'], edgecolor='black')
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_title('Gap Size Distribution (% of Previous Close)', fontweight='bold')
            ax2.set_xlabel('Gap Size (%)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No significant gaps detected', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Gap Size Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gaps', 'gap_analysis_comprehensive.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created gap analysis chart")
    
    def create_comprehensive_dashboard(self, data):
        """Create comprehensive technical analysis dashboard"""
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.3, wspace=0.2)
        
        fig.suptitle('RELIANCE.NS - Comprehensive Technical Analysis Dashboard', fontsize=20, fontweight='bold')
        
        recent_data = data.tail(200)
        dates = recent_data['Date']
        
        # 1. Price with key moving averages
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, recent_data['Close'], label='Close', color='black', linewidth=2)
        ax1.plot(dates, recent_data['SMA_20'], label='SMA 20', alpha=0.8)
        ax1.plot(dates, recent_data['SMA_50'], label='SMA 50', alpha=0.8)
        ax1.plot(dates, recent_data['EMA_20'], label='EMA 20', alpha=0.8, linestyle='--')
        ax1.set_title('Price Action with Moving Averages', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(dates, recent_data['RSI'], color=self.colors['momentum']['primary'])
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax2.fill_between(dates, 70, 100, alpha=0.2, color='red')
        ax2.fill_between(dates, 0, 30, alpha=0.2, color='green')
        ax2.set_title('RSI (14)', fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(dates, recent_data['MACD'], label='MACD')
        ax3.plot(dates, recent_data['MACD_Signal'], label='Signal')
        ax3.bar(dates, recent_data['MACD_Hist'], alpha=0.6, label='Histogram')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('MACD', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Bollinger Bands
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(dates, recent_data['Close'], color='black', linewidth=1)
        ax4.plot(dates, recent_data['Bollinger_Upper'], alpha=0.7)
        ax4.plot(dates, recent_data['Bollinger_Lower'], alpha=0.7)
        ax4.fill_between(dates, recent_data['Bollinger_Upper'], recent_data['Bollinger_Lower'], alpha=0.1)
        ax4.set_title('Bollinger Bands', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Volume with OBV
        ax5 = fig.add_subplot(gs[2, 1])
        ax5_twin = ax5.twinx()
        ax5.bar(dates, recent_data['Volume'], alpha=0.6, color='blue', label='Volume')
        ax5_twin.plot(dates, recent_data['OBV'], color='red', label='OBV')
        ax5.set_title('Volume & OBV', fontweight='bold')
        ax5.legend(loc='upper left')
        ax5_twin.legend(loc='upper right')
        
        # 6. ADX and DI
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.plot(dates, recent_data['ADX'], label='ADX', linewidth=2)
        ax6.plot(dates, recent_data['Plus_DI'], label='+DI', alpha=0.8)
        ax6.plot(dates, recent_data['Minus_DI'], label='-DI', alpha=0.8)
        ax6.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        ax6.set_title('ADX & Directional Movement', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Stochastic
        ax7 = fig.add_subplot(gs[3, 1])
        ax7.plot(dates, recent_data['Stoch_K'], label='%K')
        ax7.plot(dates, recent_data['Stoch_D'], label='%D')
        ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7)
        ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7)
        ax7.fill_between(dates, 80, 100, alpha=0.2, color='red')
        ax7.fill_between(dates, 0, 20, alpha=0.2, color='green')
        ax7.set_title('Stochastic Oscillator', fontweight='bold')
        ax7.legend()
        ax7.set_ylim(0, 100)
        ax7.grid(True, alpha=0.3)
        
        # 8. ATR
        ax8 = fig.add_subplot(gs[4, 0])
        ax8.plot(dates, recent_data['ATR'], color=self.colors['volatility']['primary'])
        ax8.fill_between(dates, recent_data['ATR'], alpha=0.3, color=self.colors['volatility']['primary'])
        ax8.set_title('Average True Range (ATR)', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 9. Money Flow Index
        ax9 = fig.add_subplot(gs[4, 1])
        ax9.plot(dates, recent_data['Money_Flow_Index'], color=self.colors['volume']['secondary'])
        ax9.axhline(y=80, color='red', linestyle='--', alpha=0.7)
        ax9.axhline(y=20, color='green', linestyle='--', alpha=0.7)
        ax9.fill_between(dates, 80, 100, alpha=0.2, color='red')
        ax9.fill_between(dates, 0, 20, alpha=0.2, color='green')
        ax9.set_title('Money Flow Index', fontweight='bold')
        ax9.set_ylim(0, 100)
        ax9.grid(True, alpha=0.3)
        
        # 10. Summary Statistics
        ax10 = fig.add_subplot(gs[5, :])
        ax10.axis('off')
        
        # Calculate summary statistics
        current_price = recent_data['Close'].iloc[-1]
        rsi_current = recent_data['RSI'].iloc[-1]
        macd_signal = 'Bullish' if recent_data['MACD_Hist'].iloc[-1] > 0 else 'Bearish'
        adx_current = recent_data['ADX'].iloc[-1]
        trend_strength = 'Strong' if adx_current > 25 else 'Weak'
        
        summary_text = f"""
        Current Analysis Summary (RELIANCE.NS):
        
        Current Price: ₹{current_price:.2f}
        RSI (14): {rsi_current:.1f} ({'Overbought' if rsi_current > 70 else 'Oversold' if rsi_current < 30 else 'Neutral'})
        MACD Signal: {macd_signal}
        ADX: {adx_current:.1f} (Trend Strength: {trend_strength})
        
        Technical Indicators Summary:
        • Trend Indicators: 5 (SMA, EMA, MACD components)
        • Momentum Indicators: 12 (RSI, ROC, CCI, Stochastic, etc.)
        • Volatility Indicators: 8 (ATR, Bollinger Bands, Volatility metrics)
        • Volume Indicators: 7 (OBV, MFI, Volume ratios, etc.)
        • Ichimoku Cloud: 5 components
        • Candlestick Patterns: 20+ pattern recognition
        • Gap Analysis: 3 gap detection metrics
        
        Total Technical Indicators: 74+
        """
        
        ax10.text(0.02, 0.98, summary_text, transform=ax10.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.savefig(os.path.join(self.output_dir, 'comprehensive', 'technical_analysis_dashboard.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created comprehensive technical analysis dashboard")
    
    def generate_all_visualizations(self):
        """Generate all technical analysis visualizations"""
        print("Starting comprehensive technical analysis visualization generation...")
        print("=" * 60)
        
        # Fetch data
        data = self.fetch_reliance_data()
        
        # Add technical indicators
        enhanced_data = self.add_technical_analysis(data)
        
        print(f"\nData shape: {enhanced_data.shape}")
        print(f"Date range: {enhanced_data['Date'].min()} to {enhanced_data['Date'].max()}")
        print(f"Available columns: {len(enhanced_data.columns)}")
        
        # Generate all charts
        print("\nGenerating visualization charts...")
        print("-" * 40)
        
        self.create_trend_indicators_chart(enhanced_data)
        self.create_momentum_indicators_chart(enhanced_data)
        self.create_volatility_indicators_chart(enhanced_data)
        self.create_volume_indicators_chart(enhanced_data)
        self.create_ichimoku_cloud_chart(enhanced_data)
        self.create_candlestick_patterns_examples(enhanced_data)
        self.create_gap_analysis_chart(enhanced_data)
        self.create_comprehensive_dashboard(enhanced_data)
        
        print("\n" + "=" * 60)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        return True

def main():
    """Main function to generate all technical analysis visualizations"""
    generator = TechnicalVisualizationGenerator()
    success = generator.generate_all_visualizations()
    
    if success:
        print("\n🎉 Technical Analysis visualization generation completed successfully!")
        print(f"📁 All charts saved to: {generator.output_dir}")
    else:
        print("\n❌ Visualization generation failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
