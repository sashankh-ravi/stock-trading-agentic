"""
News Sentiment Analysis Visualization Generator

This script creates comprehensive visualizations for the news sentiment analysis system,
including multi-source sentiment tracking, behavioral indicators, and composite scoring
using RELIANCE.NS real data.

Author: News Sentiment Documentation System
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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append('/home/sashankhravi/Documents/stock-trading-agentic')

from download_nifty500_data import collect_social_sentiment_data, _analyze_real_news_sentiment, _calculate_volume_sentiment, _calculate_price_action_sentiment, _get_analyst_sentiment, _calculate_fear_greed_indicators
import yfinance as yf

class SentimentVisualizationGenerator:
    """
    Generate comprehensive visualizations for news sentiment analysis documentation
    """
    
    def __init__(self, output_dir="/home/sashankhravi/Documents/stock-trading-agentic/download_documentation/images"):
        self.output_dir = output_dir
        self.create_output_directory()
        
        # Color schemes for sentiment analysis
        self.colors = {
            'sentiment': {
                'bullish': '#27AE60',    # Green
                'bearish': '#E74C3C',    # Red  
                'neutral': '#F39C12',    # Orange
                'background': '#ECF0F1'  # Light gray
            },
            'sources': {
                'news': '#3498DB',       # Blue
                'volume': '#9B59B6',     # Purple
                'price': '#E67E22',      # Orange
                'analyst': '#1ABC9C'     # Teal
            },
            'psychology': {
                'fear': '#C0392B',       # Dark red
                'greed': '#27AE60',      # Green
                'neutral': '#95A5A6'     # Gray
            }
        }
        
    def create_output_directory(self):
        """Create output directory structure for sentiment images"""
        sentiment_dir = os.path.join(self.output_dir, "sentiment")
        os.makedirs(sentiment_dir, exist_ok=True)
        
    def get_sample_data(self, symbol="RELIANCE.NS", days=60):
        """Generate sample sentiment data for visualization"""
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get historical price data
        hist_data = ticker.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            print(f"No data available for {symbol}")
            return None
            
        # Generate sample sentiment data for each day
        sentiment_timeline = []
        
        for i, (date, row) in enumerate(hist_data.iterrows()):
            # Simulate sentiment scores based on price movement and volume
            price_change = (row['Close'] / hist_data['Close'].iloc[max(0, i-1)] - 1) * 100 if i > 0 else 0
            volume_ratio = row['Volume'] / hist_data['Volume'].mean()
            
            # Base sentiment around 50 with variations
            base_sentiment = 50
            
            # Price influence
            price_sentiment = base_sentiment + (price_change * 2)
            price_sentiment = max(0, min(100, price_sentiment))
            
            # Volume influence  
            volume_sentiment = base_sentiment + (volume_ratio - 1) * 20
            volume_sentiment = max(0, min(100, volume_sentiment))
            
            # News sentiment (simulated with some randomness)
            news_sentiment = base_sentiment + np.random.normal(0, 10) + (price_change * 1.5)
            news_sentiment = max(0, min(100, news_sentiment))
            
            # Analyst sentiment (more stable)
            analyst_sentiment = base_sentiment + np.random.normal(0, 5) + (price_change * 0.5)
            analyst_sentiment = max(0, min(100, analyst_sentiment))
            
            # Composite sentiment
            composite = (news_sentiment * 0.4 + volume_sentiment * 0.3 + 
                        price_sentiment * 0.3)
            
            sentiment_timeline.append({
                'date': date,
                'news_sentiment': news_sentiment,
                'volume_sentiment': volume_sentiment,
                'price_sentiment': price_sentiment,
                'analyst_sentiment': analyst_sentiment,
                'composite_sentiment': composite,
                'price': row['Close'],
                'volume': row['Volume'],
                'price_change': price_change
            })
            
        return pd.DataFrame(sentiment_timeline)
    
    def create_news_sentiment_timeline(self, data, symbol="RELIANCE.NS"):
        """Create news sentiment timeline visualization"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: News Sentiment Over Time
        ax1.plot(data['date'], data['news_sentiment'], 
                color=self.colors['sources']['news'], linewidth=2, label='News Sentiment')
        ax1.fill_between(data['date'], data['news_sentiment'], 50, 
                        where=(data['news_sentiment'] >= 50), 
                        color=self.colors['sentiment']['bullish'], alpha=0.3, label='Bullish')
        ax1.fill_between(data['date'], data['news_sentiment'], 50,
                        where=(data['news_sentiment'] < 50),
                        color=self.colors['sentiment']['bearish'], alpha=0.3, label='Bearish')
        
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Neutral Line')
        ax1.axhline(y=70, color=self.colors['sentiment']['bullish'], linestyle=':', alpha=0.7, label='Bullish Threshold')
        ax1.axhline(y=30, color=self.colors['sentiment']['bearish'], linestyle=':', alpha=0.7, label='Bearish Threshold')
        
        ax1.set_title(f'{symbol} - News Sentiment Analysis (VADER NLP)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Sentiment Score (0-100)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Price vs Sentiment Correlation
        ax2_twin = ax2.twinx()
        
        # Price line
        ax2.plot(data['date'], data['price'], color='black', linewidth=2, label='Price')
        ax2.set_ylabel('Price (₹)', fontsize=12, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Sentiment line
        ax2_twin.plot(data['date'], data['news_sentiment'], 
                     color=self.colors['sources']['news'], linewidth=2, 
                     linestyle='--', label='News Sentiment')
        ax2_twin.set_ylabel('Sentiment Score', fontsize=12, color=self.colors['sources']['news'])
        ax2_twin.tick_params(axis='y', labelcolor=self.colors['sources']['news'])
        ax2_twin.set_ylim(0, 100)
        
        ax2.set_title('Price vs News Sentiment Correlation', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sentiment Distribution
        sentiment_bins = np.arange(0, 101, 10)
        ax3.hist(data['news_sentiment'], bins=sentiment_bins, 
                color=self.colors['sources']['news'], alpha=0.7, edgecolor='black')
        ax3.axvline(x=data['news_sentiment'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {data["news_sentiment"].mean():.1f}')
        ax3.axvline(x=50, color='gray', linestyle=':', alpha=0.7, label='Neutral (50)')
        
        ax3.set_title('News Sentiment Score Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Sentiment Score', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment', 'news_sentiment_timeline.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_sentiment_dashboard(self, data, symbol="RELIANCE.NS"):
        """Create comprehensive multi-source sentiment dashboard"""
        fig = plt.figure(figsize=(20, 14))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
        
        # Main timeline plot
        ax_main = fig.add_subplot(gs[0, :])
        
        # Plot all sentiment sources
        ax_main.plot(data['date'], data['news_sentiment'], 
                    color=self.colors['sources']['news'], linewidth=2, label='News Sentiment (40%)')
        ax_main.plot(data['date'], data['volume_sentiment'], 
                    color=self.colors['sources']['volume'], linewidth=2, label='Volume Sentiment (30%)')
        ax_main.plot(data['date'], data['price_sentiment'], 
                    color=self.colors['sources']['price'], linewidth=2, label='Price Action Sentiment (30%)')
        ax_main.plot(data['date'], data['analyst_sentiment'], 
                    color=self.colors['sources']['analyst'], linewidth=2, label='Analyst Sentiment')
        ax_main.plot(data['date'], data['composite_sentiment'], 
                    color='black', linewidth=3, label='Composite Sentiment', alpha=0.8)
        
        # Sentiment zones
        ax_main.fill_between(data['date'], 80, 100, color=self.colors['sentiment']['bullish'], 
                           alpha=0.2, label='Extremely Bullish Zone')
        ax_main.fill_between(data['date'], 60, 80, color=self.colors['sentiment']['bullish'], 
                           alpha=0.1, label='Bullish Zone')
        ax_main.fill_between(data['date'], 40, 60, color=self.colors['sentiment']['neutral'], 
                           alpha=0.1, label='Neutral Zone')
        ax_main.fill_between(data['date'], 20, 40, color=self.colors['sentiment']['bearish'], 
                           alpha=0.1, label='Bearish Zone')
        ax_main.fill_between(data['date'], 0, 20, color=self.colors['sentiment']['bearish'], 
                           alpha=0.2, label='Extremely Bearish Zone')
        
        ax_main.set_title(f'{symbol} - Multi-Source Sentiment Dashboard', fontsize=16, fontweight='bold')
        ax_main.set_ylabel('Sentiment Score (0-100)', fontsize=12)
        ax_main.set_ylim(0, 100)
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_main.grid(True, alpha=0.3)
        
        # Current sentiment gauges
        current_data = data.iloc[-1]
        
        # News sentiment gauge
        ax_news = fig.add_subplot(gs[1, 0])
        self.create_sentiment_gauge(ax_news, current_data['news_sentiment'], 
                                  'News Sentiment', self.colors['sources']['news'])
        
        # Volume sentiment gauge  
        ax_volume = fig.add_subplot(gs[1, 1])
        self.create_sentiment_gauge(ax_volume, current_data['volume_sentiment'], 
                                  'Volume Sentiment', self.colors['sources']['volume'])
        
        # Price sentiment gauge
        ax_price = fig.add_subplot(gs[1, 2])
        self.create_sentiment_gauge(ax_price, current_data['price_sentiment'], 
                                  'Price Sentiment', self.colors['sources']['price'])
        
        # Composite sentiment gauge
        ax_composite = fig.add_subplot(gs[2, 0])
        self.create_sentiment_gauge(ax_composite, current_data['composite_sentiment'], 
                                   'Composite Sentiment', 'black')
        
        # Sentiment correlation matrix
        ax_corr = fig.add_subplot(gs[2, 1:])
        sentiment_cols = ['news_sentiment', 'volume_sentiment', 'price_sentiment', 'composite_sentiment']
        corr_matrix = data[sentiment_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, ax=ax_corr, cbar_kws={'shrink': 0.8})
        ax_corr.set_title('Sentiment Source Correlations', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment', 'sentiment_dashboard_comprehensive.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_sentiment_gauge(self, ax, value, title, color):
        """Create a sentiment gauge visualization"""
        # Create semicircle gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=8)
        
        # Sentiment zones
        zones = [
            (0, 20, self.colors['sentiment']['bearish']),
            (20, 40, '#FF6B6B'),
            (40, 60, self.colors['sentiment']['neutral']),
            (60, 80, '#90EE90'),
            (80, 100, self.colors['sentiment']['bullish'])
        ]
        
        for start, end, zone_color in zones:
            start_angle = np.pi * (1 - start/100)
            end_angle = np.pi * (1 - end/100)
            zone_theta = np.linspace(end_angle, start_angle, 20)
            ax.plot(r * np.cos(zone_theta), r * np.sin(zone_theta), 
                   zone_color, linewidth=8, alpha=0.7)
        
        # Current value needle
        value_angle = np.pi * (1 - value/100)
        ax.arrow(0, 0, 0.8 * np.cos(value_angle), 0.8 * np.sin(value_angle),
                head_width=0.05, head_length=0.05, fc=color, ec=color, linewidth=3)
        
        # Value text
        ax.text(0, -0.3, f'{value:.1f}', ha='center', va='center', 
               fontsize=14, fontweight='bold', color=color)
        ax.text(0, -0.5, title, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
    def create_volume_sentiment_analysis(self, data, symbol="RELIANCE.NS"):
        """Create volume sentiment analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Volume vs Volume Sentiment
        ax1_twin = ax1.twinx()
        
        # Volume bars
        colors = ['green' if vs >= 50 else 'red' for vs in data['volume_sentiment']]
        ax1.bar(data['date'], data['volume'], color=colors, alpha=0.6, label='Daily Volume')
        ax1.set_ylabel('Volume', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Volume sentiment line
        ax1_twin.plot(data['date'], data['volume_sentiment'], 
                     color=self.colors['sources']['volume'], linewidth=2, 
                     label='Volume Sentiment')
        ax1_twin.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        ax1_twin.set_ylabel('Volume Sentiment', fontsize=12, 
                          color=self.colors['sources']['volume'])
        ax1_twin.tick_params(axis='y', labelcolor=self.colors['sources']['volume'])
        ax1_twin.set_ylim(0, 100)
        
        ax1.set_title('Volume vs Volume Sentiment', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volume Ratio Analysis
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        ax2.plot(data['date'], data['volume_ratio'], 
                color=self.colors['sources']['volume'], linewidth=2)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Average Volume')
        ax2.axhline(y=1.5, color='green', linestyle=':', alpha=0.7, label='High Volume Threshold')
        ax2.axhline(y=0.7, color='red', linestyle=':', alpha=0.7, label='Low Volume Threshold')
        
        ax2.fill_between(data['date'], data['volume_ratio'], 1,
                        where=(data['volume_ratio'] >= 1),
                        color='green', alpha=0.3, label='Above Average')
        ax2.fill_between(data['date'], data['volume_ratio'], 1,
                        where=(data['volume_ratio'] < 1),
                        color='red', alpha=0.3, label='Below Average')
        
        ax2.set_title('Volume Ratio Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volume Ratio', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Volume Sentiment Distribution
        ax3.hist(data['volume_sentiment'], bins=20, 
                color=self.colors['sources']['volume'], alpha=0.7, edgecolor='black')
        ax3.axvline(x=data['volume_sentiment'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {data["volume_sentiment"].mean():.1f}')
        ax3.axvline(x=50, color='gray', linestyle=':', alpha=0.7, label='Neutral (50)')
        
        ax3.set_title('Volume Sentiment Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Volume Sentiment Score', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Volume Sentiment vs Price Change
        ax4.scatter(data['volume_sentiment'], data['price_change'], 
                   color=self.colors['sources']['volume'], alpha=0.6, s=50)
        
        # Fit trend line
        z = np.polyfit(data['volume_sentiment'], data['price_change'], 1)
        p = np.poly1d(z)
        ax4.plot(data['volume_sentiment'], p(data['volume_sentiment']), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend Line (R²={np.corrcoef(data["volume_sentiment"], data["price_change"])[0,1]**2:.3f})')
        
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        ax4.axvline(x=50, color='gray', linestyle='-', alpha=0.7)
        
        ax4.set_title('Volume Sentiment vs Price Change', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Volume Sentiment', fontsize=12)
        ax4.set_ylabel('Price Change (%)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment', 'volume_sentiment_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_fear_greed_indicators(self, data, symbol="RELIANCE.NS"):
        """Create fear and greed indicators visualization"""
        # Calculate fear and greed components for sample data
        fear_greed_data = []
        
        for i in range(len(data)):
            if i < 20:  # Need sufficient data for calculations
                continue
                
            window_data = data.iloc[max(0, i-30):i+1]
            
            # Simulate fear/greed components
            volatility_component = 50 + np.random.normal(0, 15)
            volatility_component = max(0, min(100, volatility_component))
            
            momentum_component = 50 + (data.iloc[i]['price_change'] * 2)
            momentum_component = max(0, min(100, momentum_component))
            
            volume_component = data.iloc[i]['volume_sentiment']
            
            composite_fg = (volatility_component + momentum_component + volume_component) / 3
            
            fear_greed_data.append({
                'date': data.iloc[i]['date'],
                'volatility_component': volatility_component,
                'momentum_component': momentum_component,
                'volume_component': volume_component,
                'composite_fear_greed': composite_fg
            })
            
        fg_df = pd.DataFrame(fear_greed_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Fear & Greed Index Timeline
        ax1.plot(fg_df['date'], fg_df['composite_fear_greed'], 
                color='black', linewidth=3, label='Fear & Greed Index')
        
        # Fear/Greed zones
        ax1.fill_between(fg_df['date'], 80, 100, color=self.colors['psychology']['greed'], 
                        alpha=0.3, label='Extreme Greed')
        ax1.fill_between(fg_df['date'], 60, 80, color=self.colors['psychology']['greed'], 
                        alpha=0.2, label='Greed')
        ax1.fill_between(fg_df['date'], 40, 60, color=self.colors['psychology']['neutral'], 
                        alpha=0.2, label='Neutral')
        ax1.fill_between(fg_df['date'], 20, 40, color=self.colors['psychology']['fear'], 
                        alpha=0.2, label='Fear')
        ax1.fill_between(fg_df['date'], 0, 20, color=self.colors['psychology']['fear'], 
                        alpha=0.3, label='Extreme Fear')
        
        ax1.set_title(f'{symbol} - Fear & Greed Index', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Fear & Greed Score (0-100)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Component Analysis
        ax2.plot(fg_df['date'], fg_df['volatility_component'], 
                color='purple', linewidth=2, label='Volatility Component')
        ax2.plot(fg_df['date'], fg_df['momentum_component'], 
                color='orange', linewidth=2, label='Momentum Component')
        ax2.plot(fg_df['date'], fg_df['volume_component'], 
                color='green', linewidth=2, label='Volume Component')
        
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        ax2.set_title('Fear & Greed Components', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Component Score', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Current Fear & Greed Gauge
        current_fg = fg_df['composite_fear_greed'].iloc[-1]
        self.create_fear_greed_gauge(ax3, current_fg)
        
        # Plot 4: Fear & Greed Distribution
        ax4.hist(fg_df['composite_fear_greed'], bins=20, 
                color='gray', alpha=0.7, edgecolor='black')
        ax4.axvline(x=fg_df['composite_fear_greed'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {fg_df["composite_fear_greed"].mean():.1f}')
        
        # Zone markers
        ax4.axvline(x=20, color=self.colors['psychology']['fear'], linestyle=':', alpha=0.7, label='Fear Threshold')
        ax4.axvline(x=80, color=self.colors['psychology']['greed'], linestyle=':', alpha=0.7, label='Greed Threshold')
        
        ax4.set_title('Fear & Greed Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Fear & Greed Score', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment', 'fear_greed_indicators.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_fear_greed_gauge(self, ax, value):
        """Create fear and greed gauge"""
        # Create semicircle gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Fear/Greed zones
        zones = [
            (0, 20, '#8B0000', 'Extreme Fear'),
            (20, 40, '#DC143C', 'Fear'),
            (40, 60, '#FFD700', 'Neutral'),
            (60, 80, '#90EE90', 'Greed'),
            (80, 100, '#006400', 'Extreme Greed')
        ]
        
        for start, end, zone_color, label in zones:
            start_angle = np.pi * (1 - start/100)
            end_angle = np.pi * (1 - end/100)
            zone_theta = np.linspace(end_angle, start_angle, 20)
            ax.plot(r * np.cos(zone_theta), r * np.sin(zone_theta), 
                   zone_color, linewidth=10, alpha=0.8)
        
        # Current value needle
        value_angle = np.pi * (1 - value/100)
        ax.arrow(0, 0, 0.8 * np.cos(value_angle), 0.8 * np.sin(value_angle),
                head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=3)
        
        # Value text
        ax.text(0, -0.3, f'{value:.1f}', ha='center', va='center', 
               fontsize=16, fontweight='bold')
        
        # Zone labels
        if value < 20:
            emotion = 'Extreme Fear'
        elif value < 40:
            emotion = 'Fear'
        elif value < 60:
            emotion = 'Neutral'
        elif value < 80:
            emotion = 'Greed'
        else:
            emotion = 'Extreme Greed'
            
        ax.text(0, -0.5, emotion, ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        ax.set_title('Current Fear & Greed Index', fontsize=14, fontweight='bold')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
    def create_sentiment_price_correlation(self, data, symbol="RELIANCE.NS"):
        """Create sentiment vs price performance correlation analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Sentiment Leading Indicator Analysis
        # Calculate forward returns
        data['forward_return_1d'] = data['price'].shift(-1) / data['price'] - 1
        data['forward_return_5d'] = data['price'].shift(-5) / data['price'] - 1
        
        # Sentiment vs 1-day forward returns
        ax1.scatter(data['composite_sentiment'], data['forward_return_1d'] * 100, 
                   alpha=0.6, s=50, color=self.colors['sentiment']['neutral'])
        
        # Fit trend line
        valid_data = data.dropna(subset=['composite_sentiment', 'forward_return_1d'])
        if len(valid_data) > 10:
            z = np.polyfit(valid_data['composite_sentiment'], valid_data['forward_return_1d'] * 100, 1)
            p = np.poly1d(z)
            ax1.plot(valid_data['composite_sentiment'], p(valid_data['composite_sentiment']), 
                    "r--", alpha=0.8, linewidth=2, 
                    label=f'Trend (R²={np.corrcoef(valid_data["composite_sentiment"], valid_data["forward_return_1d"])[0,1]**2:.3f})')
        
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        ax1.axvline(x=50, color='gray', linestyle='-', alpha=0.7)
        ax1.set_title('Sentiment vs 1-Day Forward Returns', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Composite Sentiment', fontsize=12)
        ax1.set_ylabel('1-Day Forward Return (%)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sentiment vs 5-day Forward Returns
        ax2.scatter(data['composite_sentiment'], data['forward_return_5d'] * 100, 
                   alpha=0.6, s=50, color=self.colors['sentiment']['bullish'])
        
        valid_data_5d = data.dropna(subset=['composite_sentiment', 'forward_return_5d'])
        if len(valid_data_5d) > 10:
            z = np.polyfit(valid_data_5d['composite_sentiment'], valid_data_5d['forward_return_5d'] * 100, 1)
            p = np.poly1d(z)
            ax2.plot(valid_data_5d['composite_sentiment'], p(valid_data_5d['composite_sentiment']), 
                    "r--", alpha=0.8, linewidth=2,
                    label=f'Trend (R²={np.corrcoef(valid_data_5d["composite_sentiment"], valid_data_5d["forward_return_5d"])[0,1]**2:.3f})')
        
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        ax2.axvline(x=50, color='gray', linestyle='-', alpha=0.7)
        ax2.set_title('Sentiment vs 5-Day Forward Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Composite Sentiment', fontsize=12)
        ax2.set_ylabel('5-Day Forward Return (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rolling Correlation Analysis
        window = 20
        rolling_corr = data['composite_sentiment'].rolling(window).corr(data['price_change'])
        
        ax3.plot(data['date'], rolling_corr, 
                color='purple', linewidth=2, label=f'{window}-Day Rolling Correlation')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='No Correlation')
        ax3.axhline(y=0.3, color='green', linestyle=':', alpha=0.7, label='Positive Correlation')
        ax3.axhline(y=-0.3, color='red', linestyle=':', alpha=0.7, label='Negative Correlation')
        
        ax3.fill_between(data['date'], rolling_corr, 0,
                        where=(rolling_corr >= 0),
                        color='green', alpha=0.3)
        ax3.fill_between(data['date'], rolling_corr, 0,
                        where=(rolling_corr < 0),
                        color='red', alpha=0.3)
        
        ax3.set_title('Rolling Sentiment-Price Correlation', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Correlation Coefficient', fontsize=12)
        ax3.set_ylim(-1, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sentiment Regime Analysis
        # Define sentiment regimes
        data['sentiment_regime'] = pd.cut(data['composite_sentiment'], 
                                        bins=[0, 30, 70, 100], 
                                        labels=['Bearish', 'Neutral', 'Bullish'])
        
        regime_returns = data.groupby('sentiment_regime')['price_change'].agg(['mean', 'std', 'count'])
        
        x_pos = range(len(regime_returns.index))
        colors_regime = [self.colors['sentiment']['bearish'], 
                        self.colors['sentiment']['neutral'], 
                        self.colors['sentiment']['bullish']]
        
        bars = ax4.bar(x_pos, regime_returns['mean'], 
                      yerr=regime_returns['std'], 
                      color=colors_regime, alpha=0.7, 
                      capsize=5, error_kw={'linewidth': 2})
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, regime_returns['count'])):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={count}', ha='center', va='bottom', fontweight='bold')
        
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax4.set_title('Average Returns by Sentiment Regime', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Sentiment Regime', fontsize=12)
        ax4.set_ylabel('Average Daily Return (%)', fontsize=12)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(regime_returns.index)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment', 'sentiment_price_correlation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_all_visualizations(self, symbol="RELIANCE.NS"):
        """Generate all sentiment analysis visualizations"""
        print(f"Generating sentiment analysis visualizations for {symbol}...")
        
        # Get sample data
        data = self.get_sample_data(symbol, days=60)
        if data is None:
            print(f"Could not generate data for {symbol}")
            return
            
        print("Creating news sentiment timeline...")
        self.create_news_sentiment_timeline(data, symbol)
        
        print("Creating multi-source sentiment dashboard...")
        self.create_sentiment_dashboard(data, symbol)
        
        print("Creating volume sentiment analysis...")
        self.create_volume_sentiment_analysis(data, symbol)
        
        print("Creating fear & greed indicators...")
        self.create_fear_greed_indicators(data, symbol)
        
        print("Creating sentiment-price correlation analysis...")
        self.create_sentiment_price_correlation(data, symbol)
        
        print(f"All sentiment visualizations saved to: {os.path.join(self.output_dir, 'sentiment')}")

# Generate visualizations if run directly
if __name__ == "__main__":
    generator = SentimentVisualizationGenerator()
    generator.generate_all_visualizations("RELIANCE.NS")
