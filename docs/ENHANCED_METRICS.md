# Enhanced Metrics for Trading Analysis

This document outlines the comprehensive set of metrics collected by the enhanced data download system for the Nifty 500 Trading System.

## 1. Price Data

### Basic OHLCV Data
- **Open**: Opening price for the period
- **High**: Highest price during the period
- **Low**: Lowest price during the period
- **Close**: Closing price for the period
- **Volume**: Trading volume for the period

### Derived Time Series
- **Returns**: Daily percentage price changes
- **Log Returns**: Natural logarithm of returns for statistical analysis
- **Cumulative Returns**: Cumulative product of (1 + returns)

## 2. Technical Indicators

### Trend Indicators
- **Moving Averages**: SMA20, SMA50, SMA200, EMA20, EMA50, EMA200
- **MACD**: MACD line, Signal line, MACD histogram
- **ADX**: Average Directional Index for trend strength
- **Parabolic SAR**: Stop and Reverse indicator for trend changes
- **Ichimoku Cloud**: Tenkan, Kijun, Senkou A/B, Chikou Span

### Momentum Indicators
- **RSI**: Relative Strength Index (14-period)
- **Stochastic Oscillator**: %K and %D lines
- **Williams %R**: Williams Percent Range
- **CCI**: Commodity Channel Index
- **MFI**: Money Flow Index
- **DMI**: Directional Movement Index (DI+, DI-)

### Volatility Indicators
- **Bollinger Bands**: Upper, Middle, Lower bands
- **ATR**: Average True Range
- **Standard Deviation**: 20-day price standard deviation
- **Keltner Channels**: Upper, Middle, Lower bands

### Volume Indicators
- **OBV**: On-Balance Volume
- **VWAP**: Volume-Weighted Average Price
- **Volume Ratio**: Current volume relative to moving average
- **CMF**: Chaikin Money Flow
- **Volume Profile**: POC, Value Area High/Low

## 3. Fundamental Data

### Valuation Metrics
- **PE Ratio**: Price to Earnings ratio
- **Forward PE**: Forward-looking PE ratio
- **PB Ratio**: Price to Book ratio
- **PS Ratio**: Price to Sales ratio
- **EV/EBITDA**: Enterprise Value to EBITDA
- **EV/Revenue**: Enterprise Value to Revenue

### Financial Health Metrics
- **Debt to Equity**: Total debt relative to equity
- **Current Ratio**: Current assets to current liabilities
- **Quick Ratio**: Liquid assets to current liabilities
- **Interest Coverage**: EBIT to interest expenses

### Profitability Metrics
- **Profit Margin**: Net income as percentage of revenue
- **Operating Margin**: Operating income as percentage of revenue
- **ROE**: Return on Equity
- **ROA**: Return on Assets
- **ROIC**: Return on Invested Capital

### Growth Metrics
- **Revenue Growth**: Year-over-year revenue growth
- **Earnings Growth**: Year-over-year earnings growth
- **Dividend Growth**: Year-over-year dividend growth

## 4. Market Context Metrics

### Relative Strength Metrics
- **RS_vs_Market**: Stock performance relative to Nifty 50
- **RS_vs_Sector**: Stock performance relative to its sector
- **RS_Percentile**: Percentile rank of relative strength
- **RS_Slope**: Trend in relative strength (improving/deteriorating)

### Intermarket Correlations
- **Correlation_SPX**: Correlation with S&P 500
- **Correlation_Gold**: Correlation with gold
- **Correlation_Oil**: Correlation with crude oil
- **Correlation_USDINR**: Correlation with USD/INR forex rate
- **Correlation_10Y**: Correlation with 10-year treasury yield

### Market Regime Indicators
- **Market_Regime**: Combined market regime classification
- **Trend_Regime**: Market trend classification (Uptrend/Downtrend/Range-Bound)
- **Volatility_Regime**: Market volatility classification (High/Normal/Low)
- **Breadth_Indicators**: Market breadth metrics for overall market health

## 5. Options Data

### Options Chain Metrics
- **Put_Call_Ratio**: Volume ratio of puts to calls
- **Implied_Volatility**: Average implied volatility
- **IV_Skew**: Skew in implied volatility across strikes
- **IV_Term_Structure**: Term structure of implied volatility
- **Open_Interest_Profile**: Distribution of open interest

## 6. Sentiment Data

### News Sentiment
- **News_Sentiment_Score**: Aggregate sentiment from recent news
- **News_Volume**: Volume of news articles
- **News_Sources**: Diversity of news sources

### Alternative Data
- **Institutional_Holdings**: Changes in institutional ownership
- **Insider_Transactions**: Recent insider buying/selling
- **Social_Media_Sentiment**: Sentiment from social media platforms

## 7. Composite Metrics

### Statistical Factors
- **Z-Score_Price**: Standardized price relative to historical range
- **Z-Score_Volume**: Standardized volume relative to historical range
- **Momentum_Composite**: Combined momentum factor from multiple indicators
- **Value_Composite**: Combined value factor from multiple fundamental metrics
- **Quality_Composite**: Combined quality factor from profitability and health metrics

### Machine Learning Features
- **PCA_Components**: Principal component analysis factors
- **Anomaly_Scores**: Statistical anomaly detection metrics
- **Pattern_Recognition**: Identified technical patterns

## Integration with Predictive Models

All these metrics are available for integration with:
- Time series forecasting models
- Classification models for trade signals
- Regime-switching models
- Ensemble methods combining multiple strategies

This comprehensive set of metrics ensures that all relevant market information is captured for thorough analysis and robust trading strategies.
