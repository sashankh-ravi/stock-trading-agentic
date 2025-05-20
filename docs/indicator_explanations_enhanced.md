# Technical Indicator Explanations

This document provides comprehensive explanations of all technical indicators used in the Nifty 500 Trading System. Each indicator is explained with its calculation methodology, interpretation guidelines, and practical application within the trading system.

## Table of Contents

1. [Momentum Indicators](#momentum-indicators)
2. [Trend Indicators](#trend-indicators)
3. [Volatility Indicators](#volatility-indicators)
4. [Volume Indicators](#volume-indicators)
5. [Oscillators](#oscillators)
6. [Custom Indicators](#custom-indicators)
7. [Market Regime Indicators](#market-regime-indicators)

## Momentum Indicators

### CCI (Commodity Channel Index)
**Definition:** The Commodity Channel Index (CCI) measures the current price level relative to an average price level over a given period, normalized by typical price volatility.

**Calculation:**
```
CCI = (Typical Price - SMA of Typical Price) / (0.015 × Mean Deviation)
```
Where:
- Typical Price = (High + Low + Close) / 3
- SMA = Simple Moving Average
- Mean Deviation = Average of absolute deviations from SMA

**Visual Example:**
![CCI Indicator](images/technical_indicators.png)

**Interpretation:**
- Values above +100 indicate an overbought condition (potential reversal down)
- Values below -100 indicate an oversold condition (potential reversal up)
- CCI crossing from negative to positive territory signals bullish momentum
- CCI crossing from positive to negative territory signals bearish momentum

**Usage in the System:**
The trading system uses CCI for identifying potential price reversals and confirming trends when used alongside other indicators. The standard lookback period is 20 bars, but the system dynamically adjusts this based on market volatility.

### Williams %R
**Definition:** Williams %R is a momentum oscillator that measures overbought and oversold levels by comparing the close price to the high-low range over a specific period.

**Calculation:**
```
Williams %R = ((Highest High - Close) / (Highest High - Lowest Low)) × -100
```

**Visual Example:**
![Williams %R](images/technical_indicators.png)

**Interpretation:**
- Ranges from 0 to -100
- Values between -80 to -100 indicate oversold conditions
- Values between 0 to -20 indicate overbought conditions
- Divergences between price and Williams %R can signal potential reversals
- Often used as a leading indicator to anticipate market turns

**Usage in the System:**
The system incorporates Williams %R with adaptive thresholds that adjust based on market volatility, making it more reliable across different market conditions. It's particularly useful for identifying potential reversals in ranging markets.

### MFI (Money Flow Index)
**Definition:** The Money Flow Index combines price and volume data to create a volume-weighted RSI. It measures the flow of money into and out of a security over a specified period.

**Calculation:**
```
1. Calculate Typical Price = (High + Low + Close) / 3
2. Calculate Raw Money Flow = Typical Price × Volume
3. Determine Positive and Negative Money Flows based on price direction
4. Money Ratio = (14-day Positive Money Flow) / (14-day Negative Money Flow)
5. MFI = 100 - (100 / (1 + Money Ratio))
```

**Visual Example:**
![Money Flow Index](images/technical_indicators.png)

**Interpretation:**
- Ranges from 0 to 100
- Values over 80 indicate overbought conditions
- Values below 20 indicate oversold conditions
- Divergences between price and MFI can signal potential reversals
- MFI incorporates volume, making it more responsive to significant market moves

**Usage in the System:**
The system uses MFI as a volume-weighted momentum indicator to confirm price movements. When significant price moves are accompanied by strong volume (as indicated by MFI), the system assigns higher confidence to the signal. MFI is particularly valuable in identifying potential market tops and bottoms.

### DMI (Directional Movement Index)
**Definition:** The Directional Movement Index consists of three indicators: ADX (Average Directional Index), DI+ (Positive Directional Indicator), and DI- (Negative Directional Indicator). Together, they measure the strength and direction of a trend.

**Calculation:**
```
1. Calculate +DM and -DM (Directional Movement)
   +DM = Current High - Previous High (if positive, otherwise 0)
   -DM = Previous Low - Current Low (if positive, otherwise 0)
2. Calculate True Range (TR)
3. Calculate smoothed +DM, -DM, and TR
4. Calculate +DI and -DI
   +DI = 100 × (Smoothed +DM / Smoothed TR)
   -DI = 100 × (Smoothed -DM / Smoothed TR)
5. Calculate DX (Directional Index)
   DX = 100 × (|+DI - -DI| / |+DI + -DI|)
6. Calculate ADX (Average Directional Index)
   ADX = Moving average of DX (typically 14 periods)
```

**Visual Example:**
![DMI/ADX](images/trend_following.png)

**Interpretation:**
- ADX measures trend strength (regardless of direction)
  - ADX > 25 indicates a strong trend
  - ADX < 20 indicates a weak or absent trend
- +DI and -DI indicate trend direction
  - +DI crossing above -DI suggests bullish momentum
  - -DI crossing above +DI suggests bearish momentum
- The larger the spread between +DI and -DI, the stronger the trend

**Usage in the System:**
The system integrates DMI to determine both trend strength and direction. ADX is used as a filter to avoid false signals in ranging markets, while DI crossovers trigger trend-following entries. The combination of DMI components helps the system distinguish between trending and ranging markets, adjusting strategy weights accordingly.

## Volatility Indicators

### Bollinger Bands
**Definition:** Bollinger Bands consist of a middle band (SMA) with upper and lower bands set at standard deviation levels above and below the middle band.

**Calculation:**
```
Middle Band = 20-period SMA of closing prices
Upper Band = Middle Band + (20-period standard deviation of price × 2)
Lower Band = Middle Band - (20-period standard deviation of price × 2)
```

**Visual Example:**
![Bollinger Bands](images/patterns/bollinger_bands.png)

**Interpretation:**
- Price touching upper band may indicate overbought conditions
- Price touching lower band may indicate oversold conditions
- Bands widening indicate increasing volatility
- Bands narrowing indicate decreasing volatility
- "Band squeeze" (narrow bands) often precedes significant price movements

**Usage in the System:**
The system uses Bollinger Bands both as a volatility measure and for identifying potential reversals. Band width is tracked to identify volatility regimes, while price interactions with the bands generate trading signals. In low-volatility environments, the system may adjust band settings to be more sensitive.

### ATR (Average True Range)
**Definition:** ATR measures market volatility by decomposing the entire range of an asset price for a specific period, taking into account any gaps or limit moves. It provides a measure of market volatility without considering price direction.

**Calculation:**
```
1. Calculate True Range (TR) for each period:
   TR = max[(High - Low), abs(High - Previous Close), abs(Low - Previous Close)]
   
2. ATR = 14-period moving average of TR
   Can use simple, exponential, or Wilder's smoothing methods
```

**Visual Example:**
![ATR](images/volatility_regimes.png)

**Interpretation:**
- Higher ATR values indicate higher volatility and uncertainty
- Lower ATR values indicate lower volatility and more stable price movement
- ATR does not provide directional bias, only volatility magnitude
- Can be used to identify potential breakouts when ATR is contracting (coiling)
- Useful for sizing positions and setting stop-loss levels (e.g., 2x ATR from entry)
- ATR expansion often precedes significant price movements
- Changes in ATR can signal regime shifts in the market

**Usage in the System:**
The system employs ATR primarily for risk management. Stop-loss distances are calculated as multiples of ATR, ensuring they adapt to market volatility. Additionally, position sizing is inversely proportional to ATR, reducing exposure during highly volatile periods. The system also tracks ATR trends to identify potential volatility regime changes and adjusts strategy parameters accordingly.

### Keltner Channels
**Definition:** Keltner Channels are volatility-based envelopes set above and below an exponential moving average, using the Average True Range to set channel width. Unlike Bollinger Bands which use standard deviation, Keltner Channels use ATR, which can be more responsive to volatility changes.

**Calculation:**
```
Middle Line = 20-period EMA of typical price
Upper Channel Line = Middle Line + (2 × ATR(10))
Lower Channel Line = Middle Line - (2 × ATR(10))

Where:
Typical Price = (High + Low + Close) / 3
```

**Visual Example:**
![Keltner Channels](images/technical_indicators.png)

**Interpretation:**
- Price reaching the upper channel may indicate overbought conditions
- Price reaching the lower channel may indicate oversold conditions
- Price breaking above/below channels may signal trend continuation
- Channel width indicates market volatility

**Usage in the System:**
The system uses Keltner Channels alongside Bollinger Bands to identify high-probability reversal zones. When both indicators suggest overbought/oversold conditions, signals are given more weight. Additionally, the system tracks the relationship between these two indicators for divergence signals.

### Standard Deviation
**Definition:** Standard Deviation is a statistical measure of market volatility that quantifies how much prices are deviating from their average value over a specific lookback period. It provides essential insights into market volatility and is a core component of many other technical indicators, including Bollinger Bands.

**Calculation:**
```
1. Calculate mean (average) price over n periods:
   Mean = Sum of closing prices / n

2. Calculate the squared differences from the mean:
   Squared_Diff = (Price - Mean)²

3. Find the average of these squared differences:
   Variance = Sum of Squared_Diff / n

4. Take the square root to get the standard deviation:
   Standard Deviation = √Variance
```

**Visual Example:**
![Standard Deviation](images/volatility_regimes.png)

**Interpretation:**
- Higher standard deviation indicates higher volatility and uncertainty
- Lower standard deviation indicates lower volatility and more stable price movement
- Expanding standard deviation often precedes significant price moves
- Contracting standard deviation may indicate consolidation before a breakout
- Can be used to identify volatility cycles and regime shifts
- Often used as a volatility input for position sizing algorithms

**Usage in the System:**
The system uses standard deviation calculations for multiple purposes. It forms the basis for Bollinger Bands construction and volatility regime detection. The system also uses normalized standard deviation (comparing current volatility to historical levels) for dynamic position sizing and risk management. During high standard deviation periods, position sizes are reduced and stop-losses are tightened to account for the increased volatility.

```
1. Calculate the mean price over the selected period (n):
   Mean = Sum of prices / n
   
2. Calculate the sum of squared deviations from the mean:
   Sum of squared deviations = Sum[(Price - Mean)²]
   
3. Calculate the standard deviation:
   Standard Deviation = Square root of (Sum of squared deviations / n)
```

**Visual Example:**
![Standard Deviation](images/volatility_regimes.png)

**Interpretation:**
- Higher standard deviation indicates higher volatility and uncertainty
- Lower standard deviation indicates lower volatility and more stable prices
- Rising standard deviation often precedes significant market moves
- Contracting standard deviation (low volatility) may signal an impending breakout
- Can be used to identify volatility regimes for strategy adaptation
- Standard deviation is the basis for Bollinger Bands width calculation

**Usage in the System:**
The system incorporates standard deviation for multiple purposes. It's used to identify volatility regimes (low, medium, high) which determine position sizing and strategy selection. During low standard deviation periods, the system looks for potential breakout opportunities. The system also uses the ratio of current to historical standard deviation to determine if market conditions are normal or abnormal, adjusting risk parameters accordingly.

## Volume Indicators

### OBV (On-Balance Volume)
**Definition:** OBV is a cumulative indicator that adds volume on up days and subtracts volume on down days, creating a running total that shows whether volume is flowing in or out of a security.

**Calculation:**
```
If Close > Previous Close:
    OBV = Previous OBV + Current Volume
If Close < Previous Close:
    OBV = Previous OBV - Current Volume
If Close = Previous Close:
    OBV = Previous OBV
```

**Visual Example:**
![OBV](images/technical_indicators.png)

**Interpretation:**
- Rising OBV indicates positive volume pressure (bullish)
- Falling OBV indicates negative volume pressure (bearish)
- Divergence between OBV and price can signal potential reversals
- OBV often leads price movements

**Usage in the System:**
The system incorporates OBV primarily for confirmation of price trends and for identifying potential reversals through divergence analysis. When price makes a new high/low but OBV fails to confirm, the system flags this as a warning sign. Additionally, the system compares OBV across different timeframes for multi-timeframe confirmation.

### Volume Profile
**Definition:** Volume Profile is a horizontal histogram showing the amount of volume traded at specific price levels over a selected time period, helping to identify significant price zones.

**Calculation:**
Volume is sorted and displayed at each price level over the specified period.

**Visual Example:**
![Volume Profile](images/technical_indicators.png)

**Interpretation:**
- High volume nodes indicate significant price levels where substantial trading has occurred
- Point of Control (POC) is the price level with the highest traded volume
- Value Area contains 70% of the total trading volume
- Low volume nodes often indicate areas where price may move quickly

**Usage in the System:**
The system integrates Volume Profile analysis to identify key support/resistance levels and potential price targets. These volume-based levels are often more significant than traditional chart-based support/resistance. The system particularly focuses on the POC and Value Area extremes when calculating profit targets and stop-loss levels.

### VWAP (Volume-Weighted Average Price)
**Definition:** VWAP is the ratio of the value traded (price multiplied by volume) to total volume traded over a specific time period, providing the average price at which a security has traded throughout the day, weighted by volume.

**Calculation:**
```
VWAP = ∑(Price × Volume) / ∑(Volume)

Where:
Price = (High + Low + Close) / 3  # Typical Price
```

**Visual Example:**
![VWAP](images/technical_indicators.png)

**Interpretation:**
- Price above VWAP indicates bullish sentiment
- Price below VWAP indicates bearish sentiment
- Often used as a fair value reference by institutional traders
- Serves as a dynamic support/resistance level
- Distance from VWAP indicates potential overextension (mean reversion opportunity)
- Can be calculated for different timeframes (daily, weekly, monthly) for multi-timeframe analysis

**Usage in the System:**
For intraday strategies, the system uses VWAP as a reference point for determining market bias. Crossovers of price and VWAP are monitored for potential entry signals, while distance from VWAP helps gauge the potential for mean reversion. The system also tracks multiple time period VWAPs (weekly and monthly) for a more comprehensive analysis.

### Chaikin Money Flow (CMF)
**Definition:** Chaikin Money Flow measures buying and selling pressure over a specific period by combining price and volume information. It helps identify accumulation (buying) and distribution (selling) patterns.

**Calculation:**
```
1. Calculate Money Flow Multiplier (MFM):
   MFM = ((Close - Low) - (High - Close)) / (High - Low)

2. Calculate Money Flow Volume (MFV):
   MFV = MFM × Volume

3. Chaikin Money Flow = 21-day Sum of Money Flow Volume / 21-day Sum of Volume
```

**Visual Example:**
![Chaikin Money Flow](images/technical_indicators.png)

**Interpretation:**
- CMF values range between -1 and +1
- Positive CMF (above 0) indicates buying pressure (accumulation)
- Negative CMF (below 0) indicates selling pressure (distribution)
- Values above +0.1 show strong buying pressure
- Values below -0.1 show strong selling pressure
- Divergence between CMF and price can signal potential reversals
- Trend confirmation when CMF moves in the same direction as price

**Usage in the System:**
The system employs Chaikin Money Flow to assess the quality of price movements. Strong price advances with positive CMF are given higher probability of continuation than those with negative CMF (which may indicate distribution). The indicator is also used to detect potential divergences, where price makes new highs but CMF fails to confirm, warning of potential weakness. Additionally, the system uses CMF zero-line crossovers as supplementary signals to confirm trend changes identified by other indicators.

## Trend Indicators

### Moving Averages
**Definition:** Moving averages smooth price data to identify trend direction and potentially support and resistance levels. Common types include Simple Moving Average (SMA), Exponential Moving Average (EMA), and Weighted Moving Average (WMA).

**Calculation:**
```
SMA = Sum of closing prices over n periods / n
EMA = (Close - Previous EMA) × (2 / (n + 1)) + Previous EMA
```

**Visual Example:**
![Moving Averages](images/trend_following.png)

**Interpretation:**
- Price above MA indicates an uptrend
- Price below MA indicates a downtrend
- Crossovers between different MA periods can signal trend changes
- MA slope indicates trend strength
- Multiple MAs can form support/resistance zones

**Usage in the System:**
The system employs multiple moving averages across different timeframes. Fast and slow EMAs (8 and 21 periods) are used for generating trading signals, while longer-term SMAs (50 and 200 periods) establish the broader trend context. The system also tracks moving average convergence/divergence to gauge trend strength and potential reversals.

### MACD (Moving Average Convergence Divergence)
**Definition:** MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.

**Calculation:**
```
MACD Line = 12-period EMA - 26-period EMA
Signal Line = 9-period EMA of MACD Line
Histogram = MACD Line - Signal Line
```

**Visual Example:**
![MACD](images/patterns/macd.png)

**Interpretation:**
- MACD above zero indicates bullish momentum
- MACD below zero indicates bearish momentum
- MACD crossing above signal line suggests bullish signal
- MACD crossing below signal line suggests bearish signal
- Divergence between MACD and price can signal potential reversals

**Usage in the System:**
The system uses MACD in multiple ways: signal line crossovers for entry/exit points, zero line crossovers for trend confirmation, and histogram patterns for momentum analysis. Additionally, the system incorporates MACD histogram divergence for early reversal signals and monitors multiple timeframe MACD alignment for high-probability trades.

### Parabolic SAR
**Definition:** Parabolic SAR (Stop And Reverse) is a trend-following indicator that provides potential entry and exit points, appearing as dots above or below the price.

**Calculation:**
```
If Uptrend:
    SAR(today) = SAR(yesterday) + AF × (EP - SAR(yesterday))
If Downtrend:
    SAR(today) = SAR(yesterday) - AF × (SAR(yesterday) - EP)

Where:
- EP (Extreme Point) = Highest high in uptrend or lowest low in downtrend
- AF (Acceleration Factor) starts at 0.02 and increases by 0.02 each time a new EP is recorded, up to a maximum of 0.2
```

**Visual Example:**
![Parabolic SAR](images/trend_following.png)

**Interpretation:**
- SAR below price indicates an uptrend
- SAR above price indicates a downtrend
- When SAR flips from below price to above, it signals a potential sell
- When SAR flips from above price to below, it signals a potential buy
- Works best in trending markets

**Usage in the System:**
The system primarily uses Parabolic SAR for trailing stop-loss management in trending markets. In strong trends (confirmed by other indicators), SAR points serve as dynamic stop-loss levels. The system also monitors the distance between price and SAR to gauge trend strength and potential reversal proximity.

## Custom Indicators

### Trend Strength Index (TSI)
**Definition:** A custom indicator that combines multiple trend measures to quantify the strength and quality of the current market trend.

**Calculation:**
```
1. Calculate ADX and normalize to a 0-100 scale
2. Calculate Linear Regression Slope of price and normalize
3. Calculate MA alignment score (percentage of aligned MAs)
4. Calculate price location relative to key MAs
5. TSI = (ADX + Slope + MA_Alignment + Price_Location) / 4
```

**Visual Example:**
![Trend Strength Index](images/trend_strength_index.png)

**Interpretation:**
- TSI > 70 indicates a strong trend
- TSI < 30 indicates a weak or non-existent trend
- TSI slope provides insight into trend acceleration/deceleration
- Can be applied to any timeframe

**Usage in the System:**
The TSI serves as a key filter for activating trend-following strategies. When TSI exceeds threshold values, the system increases allocation to trend-following signals and reduces mean-reversion exposure. The system also tracks changes in TSI to identify early trend exhaustion signs.

### Market Regime Detector
**Definition:** A composite indicator that classifies current market conditions into defined regimes (trending up, trending down, ranging, or volatile) to optimize strategy selection.

**Calculation:**
```
1. Calculate volatility measures (ATR relative to historical ranges)
2. Calculate directional measures (ADX, linear regression R², etc.)
3. Apply a multivariate classification algorithm to determine the current regime
```

**Visual Example:**
![Market Regime Detector](images/regime_transitions.png)

**Interpretation:**
- Bullish Trend: Strong upside momentum with moderate-to-low volatility
- Bearish Trend: Strong downside momentum with moderate-to-high volatility
- Range-Bound: Low directional movement with low-to-moderate volatility
- Volatile/Uncertain: High volatility with unclear direction

**Usage in the System:**
The Market Regime Detector dynamically adjusts the weights of different strategy components based on the identified regime. In trending markets, the system increases allocation to trend-following strategies. In range-bound markets, mean-reversion strategies receive higher weights. In volatile/uncertain regimes, the system reduces overall exposure and implements tighter risk controls.

### Adaptive RSI
**Definition:** A modified version of the traditional Relative Strength Index that dynamically adjusts its parameters based on market volatility and trend characteristics.

**Calculation:**
```
1. Calculate market volatility using ATR
2. Adjust RSI period based on volatility (shorter periods in high volatility)
3. Adjust overbought/oversold thresholds based on trend strength
4. Calculate RSI using the adaptive parameters
```

**Visual Example:**
![Adaptive RSI](images/technical_indicators.png)

**Interpretation:**
- Values above dynamic overbought threshold signal potential reversal (or trend continuation in strong trends)
- Values below dynamic oversold threshold signal potential reversal (or trend continuation in strong downtrends)
- Crossovers of the midline (50) indicate momentum shifts
- Divergences with price remain significant signals

**Usage in the System:**
The Adaptive RSI provides more relevant signals across different market regimes by automatically adjusting its sensitivity. The system particularly focuses on RSI divergences in trending markets and overbought/oversold conditions in ranging markets. The adaptive thresholds help reduce false signals compared to the traditional fixed-threshold approach.

## Market Regime Indicators

### Volatility Regime Analysis
**Definition:** A framework for categorizing market periods based on volatility levels to adapt strategy parameters accordingly.

**Calculation:**
```python
def get_volatility_regime(prices, window=20):
    # Calculate returns
    returns = np.log(prices / prices.shift(1))
    
    # Calculate historical volatility
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    # Define regime thresholds based on historical quartiles
    vol_quartiles = volatility.rolling(window=252).quantile([0.25, 0.75])
    low_threshold = vol_quartiles.iloc[-1, 0]
    high_threshold = vol_quartiles.iloc[-1, 1]
    
    # Determine current regime
    current_vol = volatility.iloc[-1]
    if current_vol < low_threshold:
        return "low_volatility"
    elif current_vol > high_threshold:
        return "high_volatility"
    else:
        return "normal_volatility"
```

**Visual Example:**
![Volatility Regimes](images/volatility_regimes.png)

**Interpretation:**
- Low Volatility: Typically associated with complacency and trend continuation
- Normal Volatility: Balanced market conditions
- High Volatility: Associated with uncertainty and potential trend reversals

**Trading Applications:**
- Low Volatility: Employ trend-following strategies, wider stop-losses
- Normal Volatility: Balanced approach, standard strategy parameters
- High Volatility: Reduce position sizes, tighten stops, favor mean-reversion

**Implementation:**
```python
# Adjust position sizing based on volatility regime
def get_position_size(base_size, volatility_regime):
    if volatility_regime == "low_volatility":
        return base_size * 1.2  # Increase size in low volatility
    elif volatility_regime == "high_volatility":
        return base_size * 0.6  # Reduce size in high volatility
    else:
        return base_size  # Standard size in normal conditions
```

### Trend Regime Detection
**Definition:** A systematic approach to identifying the prevailing market trend direction and strength to optimize strategy selection.

**Calculation:**
```python
def get_trend_regime(prices, short_window=20, long_window=200):
    # Calculate moving averages
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    
    # Calculate ADX for trend strength
    high, low, close = prices_df['High'], prices_df['Low'], prices_df['Close']
    adx = talib.ADX(high, low, close, timeperiod=14)
    
    # Determine trend direction
    if short_ma.iloc[-1] > long_ma.iloc[-1]:
        direction = "uptrend"
    else:
        direction = "downtrend"
    
    # Determine trend strength
    if adx.iloc[-1] < 20:
        strength = "weak"
    elif adx.iloc[-1] < 40:
        strength = "moderate"
    else:
        strength = "strong"
    
    return f"{strength}_{direction}"
```

**Visual Example:**
![Trend Regimes](images/regime_transitions.png)

**Interpretation:**
- Strong Uptrend: Definitive bullish momentum with strong buyer dominance
- Moderate Uptrend: Bullish bias with occasional pullbacks
- Weak Uptrend: Bullish bias but lacking conviction
- Weak Downtrend: Bearish bias but lacking conviction
- Moderate Downtrend: Bearish bias with occasional rallies
- Strong Downtrend: Definitive bearish momentum with strong seller dominance

**Trading Applications:**
- Strong Trends: Emphasize trend-following, minimize counter-trend exposure
- Moderate Trends: Balanced approach with both trend and pullback strategies
- Weak Trends: Emphasize range-bound strategies, limit trend exposure

**Implementation:**
```python
# Select strategy based on trend regime
def select_strategy_weights(trend_regime):
    if "strong" in trend_regime:
        return {"trend_following": 0.7, "mean_reversion": 0.1, "breakout": 0.2}
    elif "moderate" in trend_regime:
        return {"trend_following": 0.5, "mean_reversion": 0.3, "breakout": 0.2}
    else:  # weak trend
        return {"trend_following": 0.2, "mean_reversion": 0.6, "breakout": 0.2}
```

---

## Integration of Multiple Indicators

The Nifty 500 Trading System employs a sophisticated approach to integrating multiple indicators, reducing false signals and enhancing overall strategy robustness:

### Correlation Analysis
**Description:** Systematic analysis of indicator correlations to avoid redundancy and over-reliance on similar signals.

**Visual Example:**
![Indicator Correlation](images/indicator_correlation.png)

**Implementation:**
```python
def analyze_indicator_correlations(indicators_df):
    """
    Calculate correlation matrix between different indicators
    and identify highly correlated pairs to avoid redundancy
    """
    corr_matrix = indicators_df.corr()
    
    # Identify highly correlated pairs (>0.8)
    highly_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                highly_correlated.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    return corr_matrix, highly_correlated
```

### Multi-Timeframe Analysis
**Description:** Analysis of indicators across multiple timeframes (daily, weekly, monthly) to identify stronger, more reliable signals.

**Implementation:**
```python
def multi_timeframe_confirmation(daily_signals, weekly_signals, monthly_signals):
    """
    Check if signals are aligned across multiple timeframes
    """
    confirmed_signals = {}
    
    for signal_name in daily_signals:
        if signal_name in weekly_signals and signal_name in monthly_signals:
            # Check if signal direction is the same across timeframes
            if (np.sign(daily_signals[signal_name]) == 
                np.sign(weekly_signals[signal_name]) == 
                np.sign(monthly_signals[signal_name])):
                
                # Confirmed signal is stronger
                confirmed_signals[signal_name] = daily_signals[signal_name] * 1.5
    
    return confirmed_signals
```

### Consensus Scoring
**Description:** A weighted scoring system that aggregates signals from different indicator categories to generate a final directional bias and conviction level.

**Implementation:**
```python
def calculate_consensus_score(trend_signals, momentum_signals, 
                             volatility_signals, volume_signals):
    """
    Calculate a weighted consensus score from different indicator categories
    """
    # Apply category weights
    weights = {
        'trend': 0.4,
        'momentum': 0.3,
        'volatility': 0.2,
        'volume': 0.1
    }
    
    # Calculate category scores (average of signals in each category)
    category_scores = {
        'trend': sum(trend_signals.values()) / len(trend_signals) if trend_signals else 0,
        'momentum': sum(momentum_signals.values()) / len(momentum_signals) if momentum_signals else 0,
        'volatility': sum(volatility_signals.values()) / len(volatility_signals) if volatility_signals else 0,
        'volume': sum(volume_signals.values()) / len(volume_signals) if volume_signals else 0
    }
    
    # Calculate weighted consensus
    consensus = sum(category_scores[cat] * weights[cat] for cat in weights)
    
    # Determine conviction level
    conviction = min(1.0, abs(consensus) * 2)  # Scale to 0-1
    
    return consensus, conviction
```

---

*Note: This guide integrates theoretical indicator knowledge with practical implementation details specific to the Nifty 500 Trading System. While based on established technical analysis principles, the system has been optimized for Indian market characteristics and behavior.*
