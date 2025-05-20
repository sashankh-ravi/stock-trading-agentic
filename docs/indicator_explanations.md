## Detailed Technical Indicator Explanations

### Momentum Indicators

#### CCI (Commodity Channel Index)
**Definition:** The Commodity Channel Index (CCI) measures the current price level relative to an average price level over a given period, normalized by typical price volatility.

**Calculation:**
```
CCI = (Typical Price - SMA of Typical Price) / (0.015 × Mean Deviation)
```
Where:
- Typical Price = (High + Low + Close) / 3
- SMA = Simple Moving Average
- Mean Deviation = Average of absolute deviations from SMA

**Interpretation:**
- Values above +100 indicate an overbought condition (potential reversal down)
- Values below -100 indicate an oversold condition (potential reversal up)
- CCI crossing from negative to positive territory signals bullish momentum
- CCI crossing from positive to negative territory signals bearish momentum

**Usage in the System:**
The trading system uses CCI for identifying potential price reversals and confirming trends when used alongside other indicators. The standard lookback period is 20 bars, but the system dynamically adjusts this based on market volatility.

#### Williams %R
**Definition:** Williams %R is a momentum oscillator that measures overbought and oversold levels by comparing the close price to the high-low range over a specific period.

**Calculation:**
```
Williams %R = ((Highest High - Close) / (Highest High - Lowest Low)) × -100
```

**Interpretation:**
- Ranges from 0 to -100
- Values between -80 to -100 indicate oversold conditions
- Values between 0 to -20 indicate overbought conditions
- Divergences between price and Williams %R can signal potential reversals
- Often used as a leading indicator to anticipate market turns

**Usage in the System:**
The system incorporates Williams %R with adaptive thresholds that adjust based on market volatility, making it more reliable across different market conditions. It's particularly useful for identifying potential reversals in ranging markets.

#### MFI (Money Flow Index)
**Definition:** The Money Flow Index combines price and volume data to create a volume-weighted RSI. It measures the flow of money into and out of a security over a specified period.

**Calculation:**
```
1. Calculate Typical Price = (High + Low + Close) / 3
2. Calculate Raw Money Flow = Typical Price × Volume
3. Determine Positive and Negative Money Flows based on price direction
4. Money Ratio = (14-day Positive Money Flow) / (14-day Negative Money Flow)
5. MFI = 100 - (100 / (1 + Money Ratio))
```

**Interpretation:**
- Ranges from 0 to 100
- Values over 80 indicate overbought conditions
- Values below 20 indicate oversold conditions
- Divergences between price and MFI can signal potential reversals
- MFI incorporates volume, making it more responsive to significant market moves

**Usage in the System:**
The system uses MFI as a volume-weighted momentum indicator to confirm price movements. When significant price moves are accompanied by strong volume (as indicated by MFI), the system assigns higher confidence to the signal. MFI is particularly valuable in identifying potential market tops and bottoms.

#### DMI (Directional Movement Index)
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

### Volatility Indicators

#### Bollinger Bands
**Definition:** Bollinger Bands consist of a middle band (SMA) with upper and lower bands set at standard deviation levels above and below the middle band.

**Calculation:**
```
Middle Band = 20-day SMA
Upper Band = Middle Band + (2 × 20-day Standard Deviation)
Lower Band = Middle Band - (2 × 20-day Standard Deviation)
```

**Interpretation:**
- Price reaching the upper band indicates overbought conditions
- Price reaching the lower band indicates oversold conditions
- Band width expansion indicates increased volatility
- Band width contraction indicates decreased volatility
- "Bollinger Band Squeeze" (narrow bands) often precedes significant price moves

**Usage in the System:**
The system uses Bollinger Bands for multiple purposes: volatility measurement, identification of overbought/oversold conditions, and detection of potential breakouts. The standard (20,2) configuration is used as a baseline, but the system dynamically adjusts the parameters based on market conditions.

#### ATR (Average True Range)
**Definition:** Average True Range measures market volatility by calculating the average range between high and low prices, accounting for gaps.

**Calculation:**
```
1. Calculate True Range (TR), which is the greatest of:
   - Current High - Current Low
   - |Current High - Previous Close|
   - |Current Low - Previous Close|
2. ATR = 14-day EMA or SMA of TR
```

**Interpretation:**
- Higher ATR values indicate higher volatility
- Lower ATR values indicate lower volatility
- ATR is not directional (does not indicate price direction)
- Often used for setting stop-loss levels and position sizing
- Can identify potential breakout points when ATR expands rapidly

**Usage in the System:**
ATR serves as the foundation for the system's risk management framework. It's used for:
1. Dynamic position sizing (smaller positions in higher volatility)
2. Adaptive stop-loss placement (wider stops in higher volatility)
3. Volatility breakout detection
4. Normalization of indicators across different securities

#### Keltner Channels
**Definition:** Keltner Channels are volatility-based bands placed above and below an EMA, using ATR to set channel width.

**Calculation:**
```
Middle Line = 20-day EMA
Upper Channel = Middle Line + (2 × ATR)
Lower Channel = Middle Line - (2 × ATR)
```

**Interpretation:**
- Price reaching the upper channel indicates potential overbought conditions
- Price reaching the lower channel indicates potential oversold conditions
- Channel width indicates volatility (wider = higher volatility)
- Breakouts above/below channels can signal strong trend continuation
- More responsive to volatility changes than Bollinger Bands

**Usage in the System:**
The system uses Keltner Channels alongside Bollinger Bands to confirm volatility conditions. When price breaks through both Bollinger Bands and Keltner Channels, the system assigns higher probability to the breakout signal. Additionally, the comparison between these two channel systems helps detect divergences in volatility readings.

#### Standard Deviation
**Definition:** Standard Deviation measures the dispersion of price values from their mean, providing a statistical measure of market volatility.

**Calculation:**
```
Standard Deviation = √(Σ(Close - Average Close)² / n)
```
Where n is the number of periods (typically 20).

**Interpretation:**
- Higher values indicate increased volatility
- Lower values indicate decreased volatility
- Can help identify potential consolidation periods before breakouts
- Often used to normalize other indicators
- Useful for comparing volatility across different time periods

**Usage in the System:**
The system employs rolling standard deviation calculations to:
1. Normalize indicator readings across different securities
2. Detect volatility regime changes
3. Adjust strategy parameters dynamically
4. Identify potential consolidation patterns
5. Calculate z-scores for various metrics

### Volume Indicators

#### OBV (On-Balance Volume)
**Definition:** On-Balance Volume is a cumulative indicator that adds volume on up days and subtracts volume on down days to identify buying and selling pressure.

**Calculation:**
```
If Close > Previous Close:
    OBV = Previous OBV + Current Volume
If Close < Previous Close:
    OBV = Previous OBV - Current Volume
If Close = Previous Close:
    OBV = Previous OBV
```

**Interpretation:**
- Rising OBV indicates buying pressure (positive)
- Falling OBV indicates selling pressure (negative)
- OBV leading price (making new highs before price) is bullish
- OBV divergence from price can signal potential reversals
- Confirmation of price trends when OBV moves in the same direction

**Usage in the System:**
The system analyzes OBV to confirm price trends and identify potential divergences. When price makes a new high but OBV fails to do so, it signals weakening momentum and potential reversal. OBV is particularly valuable for validating breakouts, as genuine breakouts should be accompanied by increasing volume.

#### Volume Profile
**Definition:** Volume Profile displays trading volume at specific price levels over a given time period, creating a histogram of volume distribution by price.

**Calculation:**
Volume is aggregated at different price levels to create a horizontal histogram.

**Interpretation:**
- High volume nodes indicate significant support/resistance levels
- "Point of Control" (highest volume level) is a major support/resistance
- "Value Area" (where 70% of trading occurred) defines the fair value range
- Low volume nodes indicate areas where price may move quickly
- Volume distribution shape can indicate accumulation or distribution patterns

**Usage in the System:**
The system constructs dynamic support and resistance levels based on Volume Profile analysis. These levels are incorporated into the pattern recognition strategy for trade entries and exits. Additionally, the system identifies low volume areas that may allow for rapid price movement, adjusting position sizing and stop placement accordingly.

#### Chaikin Money Flow
**Definition:** Chaikin Money Flow measures the money flow volume over a specified period to determine buying and selling pressure.

**Calculation:**
```
1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
2. Money Flow Volume = Money Flow Multiplier × Volume
3. Chaikin Money Flow = 21-day Sum of Money Flow Volume / 21-day Sum of Volume
```

**Interpretation:**
- Values above +0.1 indicate strong buying pressure
- Values below -0.1 indicate strong selling pressure
- Values between -0.1 and +0.1 indicate neutral conditions
- Divergences between price and CMF can signal potential reversals
- Often used to confirm price trends and identify accumulation/distribution

**Usage in the System:**
The system employs Chaikin Money Flow to assess the quality of price movements. Strong price advances with positive CMF are given higher probability of continuation than those with negative CMF. The indicator is also used to detect potential distribution and accumulation phases, which helps in timing entries and exits.

#### VWAP (Volume-Weighted Average Price)
**Definition:** VWAP calculates the average price a security has traded at throughout the day, based on both volume and price.

**Calculation:**
```
VWAP = Σ(Price × Volume) / Σ(Volume)
```

**Interpretation:**
- Price above VWAP indicates bullish intraday sentiment
- Price below VWAP indicates bearish intraday sentiment
- Institutional buying often occurs near or below VWAP
- Used as a benchmark for trade execution quality
- Mean-reversion tendency toward VWAP throughout the day

**Usage in the System:**
While primarily an intraday tool, the system incorporates multi-day VWAP calculations to identify significant price levels. For shorter-term trades, entries below VWAP in bullish conditions are prioritized for better average execution prices. The system also uses VWAP as a dynamic support/resistance level, particularly for intraday strategy adjustments.
