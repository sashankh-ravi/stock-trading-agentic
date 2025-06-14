# Technical Analysis System: Mathematical Foundations and Implementation

## Overview

This document provides the technical foundation for a quantitative stock analysis system that processes daily OHLCV (Open, High, Low, Close, Volume) data from Indian equity markets (NSE). The system implements 74 technical indicators and 20 candlestick pattern recognition algorithms.

## Data Sources and Structure

### Data Provider
- **Primary Source**: Yahoo Finance API (`yfinance` library)
- **Market**: National Stock Exchange of India (NSE)
- **Frequency**: Daily
- **Trading Hours**: 09:15 - 15:30 IST

### Data Structure
```python
# Daily OHLCV Data Structure
{
    'Date': datetime,      # Trading date
    'Open': float,         # Opening price at 09:15 IST
    'High': float,         # Highest price during session
    'Low': float,          # Lowest price during session
    'Close': float,        # Closing price at 15:30 IST
    'Volume': int,         # Number of shares traded
    'Adj Close': float     # Price adjusted for dividends/splits
}
```

### Data Quality Metrics
- **Lookback Requirements**: 200 trading days minimum for moving averages
- **Missing Data Handling**: Forward fill for gaps < 3 days
- **Corporate Actions**: Automatic adjustment via Yahoo Finance
- **Quality Score**: Percentage of valid data points after indicator calculation

## Candlestick Analysis

### Mathematical Foundation

Each candlestick represents price action during one trading day:

```
Candlestick Components:
• Body = |Close - Open|
• Upper_Shadow = High - max(Open, Close)  
• Lower_Shadow = min(Open, Close) - Low
• Total_Range = High - Low
• Body_Percentage = Body / Total_Range
```

### Pattern Detection Algorithm

The system uses TA-Lib pattern recognition with the following detection criteria:

#### Doji Patterns
```python
# Detection Logic
doji_threshold = 0.1  # 10% of range
is_doji = (Body / Total_Range) < doji_threshold

# Subtypes
if Upper_Shadow < (Body * 0.1) and Lower_Shadow > (Body * 2):
    pattern = "DRAGONFLY_DOJI"  # Bullish at support
elif Lower_Shadow < (Body * 0.1) and Upper_Shadow > (Body * 2):
    pattern = "GRAVESTONE_DOJI"  # Bearish at resistance
else:
    pattern = "STANDARD_DOJI"   # Indecision
```

#### Hammer/Hanging Man
```python
# Detection Criteria
min_lower_shadow = Body * 2
max_upper_shadow = Body * 0.1
min_body_size = Total_Range * 0.1

is_hammer = (Lower_Shadow >= min_lower_shadow and 
             Upper_Shadow <= max_upper_shadow and
             Body >= min_body_size)

# Context determines classification:
# Hammer: After downtrend (bullish reversal)
# Hanging Man: After uptrend (bearish reversal)
```

#### Engulfing Patterns
```python
# Two-candle pattern
def detect_engulfing(day1, day2):
    day1_body_top = max(day1.open, day1.close)
    day1_body_bottom = min(day1.open, day1.close)
    day2_body_top = max(day2.open, day2.close)
    day2_body_bottom = min(day2.open, day2.close)
    
    # Bullish Engulfing
    bullish = (day1.close < day1.open and  # Day 1 bearish
               day2.close > day2.open and  # Day 2 bullish
               day2_body_bottom < day1_body_bottom and
               day2_body_top > day1_body_top)
    
    # Bearish Engulfing  
    bearish = (day1.close > day1.open and  # Day 1 bullish
               day2.close < day2.open and  # Day 2 bearish
               day2_body_bottom < day1_body_bottom and
               day2_body_top > day1_body_top)
    
    return bullish, bearish
```

### Pattern Frequency Analysis

Based on RELIANCE.NS analysis (492 trading days):

| Pattern | Occurrences | Frequency | Statistical Significance |
|---------|-------------|-----------|-------------------------|
| Spinning Top | 32 | 6.5% | High (χ² p < 0.01) |
| Doji | 16 | 3.3% | Moderate (χ² p < 0.05) |
| Harami | 8 | 1.6% | Low |
| Engulfing | 7 | 1.4% | Low |
| Hammer | 2 | 0.4% | Very Low |

## Technical Indicators

### Trend Indicators

#### Simple Moving Average (SMA)
```
Formula: SMA_n(t) = (1/n) × Σ(P_i) for i from (t-n+1) to t

Implementation:
SMA_20 = rolling_mean(Close, window=20)
SMA_50 = rolling_mean(Close, window=50)  
SMA_200 = rolling_mean(Close, window=200)

Properties:
• Lag: (n-1)/2 periods
• Smoothing: Equal weight to all periods
• Signal: Price crossover or MA crossover
```

#### Exponential Moving Average (EMA)
```
Formula: EMA_t = α × P_t + (1-α) × EMA_{t-1}
where α = 2/(n+1)

Initialization: EMA_0 = P_0

Smoothing Factor:
• EMA_12: α = 2/13 = 0.1538
• EMA_26: α = 2/27 = 0.0741

Properties:
• Less lag than SMA
• More weight to recent prices
• Responsive to trend changes
```

#### MACD (Moving Average Convergence Divergence)
```
Components:
MACD_Line = EMA_12 - EMA_26
Signal_Line = EMA_9(MACD_Line)
Histogram = MACD_Line - Signal_Line

Signal Generation:
• Bullish: MACD > Signal (crossover above)
• Bearish: MACD < Signal (crossover below)
• Momentum: Histogram slope direction
• Divergence: Price vs MACD direction conflict
```

### Momentum Indicators

#### Relative Strength Index (RSI)
```
Mathematical Foundation:
RS_t = (1/n) × Σ(Gains) / (1/n) × Σ(Losses)
RSI_t = 100 - (100 / (1 + RS_t))

Wilder's Smoothing:
Avg_Gain_t = ((n-1) × Avg_Gain_{t-1} + Current_Gain) / n
Avg_Loss_t = ((n-1) × Avg_Loss_{t-1} + Current_Loss) / n

where:
Current_Gain = max(0, Close_t - Close_{t-1})
Current_Loss = max(0, Close_{t-1} - Close_t)

Boundary Conditions:
• Range: [0, 100]
• Overbought: RSI > 70
• Oversold: RSI < 30
• Neutral: RSI ≈ 50
```

#### Commodity Channel Index (CCI)
```
Formula: CCI = (TP - SMA_TP) / (0.015 × MD)

where:
TP = (High + Low + Close) / 3  # Typical Price
SMA_TP = 20-period SMA of TP
MD = (1/20) × Σ|TP_i - SMA_TP|  # Mean Deviation

Scaling Factor: 0.015 ensures ~70% of values within ±100

Interpretation:
• CCI > +100: Strong upward movement
• CCI < -100: Strong downward movement  
• Normal range: ±100
```

### Volatility Indicators

#### Average True Range (ATR)
```
True Range Calculation:
TR_t = max(High_t - Low_t, 
           |High_t - Close_{t-1}|, 
           |Low_t - Close_{t-1}|)

ATR Calculation:
ATR_t = ((n-1) × ATR_{t-1} + TR_t) / n  # Wilder's smoothing

Normalization:
ATR_Percent = (ATR / Close) × 100

Applications:
• Position sizing: Risk = Capital × % / ATR
• Stop placement: Stop = Entry ± (ATR × multiplier)
• Volatility filtering: Trade when ATR > threshold
```

#### Bollinger Bands
```
Mathematical Construction:
Middle = SMA(Close, 20)
σ = sqrt((1/20) × Σ(Close_i - Middle)²)
Upper = Middle + (k × σ)
Lower = Middle - (k × σ)

Standard Parameters:
• Period: 20
• Standard Deviations (k): 2.0

Statistical Properties:
• ~95% of observations within 2σ bands
• ~99.7% of observations within 3σ bands

Derived Indicators:
%B = (Close - Lower) / (Upper - Lower)
Bandwidth = (Upper - Lower) / Middle
```

### Volume Indicators

#### On-Balance Volume (OBV)
```
Cumulative Calculation:
If Close_t > Close_{t-1}: OBV_t = OBV_{t-1} + Volume_t
If Close_t < Close_{t-1}: OBV_t = OBV_{t-1} - Volume_t  
If Close_t = Close_{t-1}: OBV_t = OBV_{t-1}

Initialization: OBV_0 = 0

Principle: Volume precedes price
Analysis: Divergence between OBV and price indicates potential reversal
```

#### Money Flow Index (MFI)
```
Multi-step Calculation:

1. Typical Price = (High + Low + Close) / 3
2. Money Flow = Typical_Price × Volume
3. Positive/Negative Classification:
   If TP_t > TP_{t-1}: Positive_MF = MF_t
   If TP_t < TP_{t-1}: Negative_MF = MF_t

4. 14-period Sums:
   PMF_14 = Σ(Positive_MF, 14)
   NMF_14 = Σ(Negative_MF, 14)

5. Money Flow Ratio = PMF_14 / NMF_14
6. MFI = 100 - (100 / (1 + MFR))

Range: [0, 100] (similar to RSI but volume-weighted)
```

## Market Regime Detection

### Methodology

The system classifies market conditions into discrete regimes using multiple indicators:

```python
def detect_regime(price_data, volume_data):
    # Trend Analysis
    adx = calculate_adx(price_data, period=14)
    di_plus = calculate_di_plus(price_data, period=14)
    di_minus = calculate_di_minus(price_data, period=14)
    
    # Volatility Analysis  
    atr_percentile = calculate_percentile_rank(atr_data, lookback=60)
    
    # Volume Analysis
    relative_volume = volume / volume_ma_20
    
    # Regime Classification
    if adx > 25:
        trend = 'strong_up' if di_plus > di_minus else 'strong_down'
    elif adx < 20:
        trend = 'sideways'
    else:
        trend = 'weak_trend'
    
    # Volatility Regime
    if atr_percentile > 80:
        volatility = 'high'
    elif atr_percentile < 20:
        volatility = 'low'  
    else:
        volatility = 'normal'
    
    return RegimeState(trend, volatility, volume_regime, confidence)
```

### Regime Categories

1. **Trend Classification**:
   - strong_up: ADX > 25, +DI > -DI
   - strong_down: ADX > 25, +DI < -DI
   - sideways: ADX < 20
   - weak_trend: 20 ≤ ADX ≤ 25

2. **Volatility Classification**:
   - high: ATR > 80th percentile (60-day lookback)
   - normal: 20th ≤ ATR ≤ 80th percentile  
   - low: ATR < 20th percentile

3. **Volume Classification**:
   - high: Volume > 1.5 × 20-day average
   - normal: 0.7 ≤ Volume ≤ 1.5 × 20-day average
   - low: Volume < 0.7 × 20-day average

## Performance Metrics

### Computational Performance

| Operation | Time (ms) | Memory (MB) |
|-----------|-----------|-------------|
| Data Download (492 days) | 750 | 2.1 |
| Technical Indicators (74) | 142 | 7.4 |
| Pattern Recognition (20) | 15 | 0.8 |
| Regime Detection | 8 | 0.3 |
| **Total Processing** | **915** | **10.6** |

### Data Quality Assessment

```python
def calculate_data_quality(df):
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    quality_score = (1 - missing_cells/total_cells) * 100
    return quality_score

# Example: RELIANCE.NS (492 days, 81 columns)
# Missing values: 1,292 out of 39,852 total cells
# Quality Score: 96.8%
```

### Statistical Validation

The system undergoes continuous validation:

1. **Cross-Correlation Analysis**: Identify redundant indicators
2. **Information Ratio**: Measure signal-to-noise ratio
3. **Walk-Forward Testing**: Out-of-sample performance validation
4. **Regime Stability**: Performance across market conditions

## Error Handling

### Data Validation
```python
def validate_ohlcv_data(df):
    """Validate basic OHLCV constraints"""
    assert (df['High'] >= df['Low']).all(), "High < Low violation"
    assert (df['High'] >= df['Open']).all(), "High < Open violation"  
    assert (df['High'] >= df['Close']).all(), "High < Close violation"
    assert (df['Low'] <= df['Open']).all(), "Low > Open violation"
    assert (df['Low'] <= df['Close']).all(), "Low > Close violation"
    assert (df['Volume'] >= 0).all(), "Negative volume violation"
    
    return True
```

### Missing Data Strategy
```python
def handle_missing_data(df, max_gap=3):
    """Forward fill gaps up to max_gap days"""
    for column in ['Open', 'High', 'Low', 'Close']:
        # Forward fill small gaps
        df[column] = df[column].fillna(method='ffill', limit=max_gap)
        
        # Flag remaining missing data
        remaining_na = df[column].isnull().sum()
        if remaining_na > 0:
            logger.warning(f"{remaining_na} missing values in {column}")
    
    return df
```

## Implementation Notes

### Dependencies
```python
# Core computation
import numpy as np
import pandas as pd
import talib

# Data acquisition  
import yfinance as yf

# Statistical analysis
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns
```

### Memory Optimization
```python
def optimize_dataframe(df):
    """Reduce memory usage by optimizing data types"""
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df
```

This technical documentation provides the mathematical and implementation foundation for understanding how the trading system processes market data and generates quantitative signals. Each indicator and pattern has been documented with its mathematical basis, implementation details, and practical considerations.
