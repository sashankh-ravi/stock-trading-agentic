# Comprehensive Candlestick Pattern and Technical Indicator Guide

This document provides in-depth explanations of all candlestick patterns and technical indicators used in the Nifty 500 Trading System. Each pattern and indicator is explained in detail with visual characteristics, calculation methods, trading implications, and system-specific implementation notes.

## Table of Contents

1. [Single-Day Candlestick Patterns](#single-day-candlestick-patterns)
2. [Multi-Day Candlestick Patterns](#multi-day-candlestick-patterns)
3. [Price Action Patterns](#price-action-patterns)
4. [Technical Bands and Overlays](#technical-bands-and-overlays)
5. [Volatility Indicators](#volatility-indicators)
6. [Volume-Based Indicators](#volume-based-indicators)
7. [System-Specific Pattern Integration](#system-specific-pattern-integration)

## Single-Day Candlestick Patterns

### Bullish Patterns

#### Three White Soldiers (CDL3WHITESOLDIERS)
**Description:** Three consecutive bullish (white/green) candles, each opening within the previous candle's body and closing higher than the previous close. Represents strong buying pressure.

**Visual Characteristics:**
- Three consecutive bullish candles
- Each candle opens within the real body of the previous candle
- Each candle closes progressively higher
- Little to no upper shadows (ideally)

**Reliability:** 90-100%

**Trading Implications:**
- Strong bullish reversal signal, especially after a downtrend
- Indicates sustained buying pressure and momentum
- More reliable with increasing volume over the three days

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)
if result[-1] > 0:  # Bullish pattern
    patterns['CDL3WHITESOLDIERS'] = 1.0  # Highest reliability weight
```

**Context Requirements:**
- Most effective after a downtrend or in a consolidation area
- Confirmation by volume increase enhances reliability
- The longer the candles, the stronger the signal

---

#### Morning Star (CDLMORNINGSTAR)
**Description:** A three-candle bullish reversal pattern consisting of a large bearish candle, followed by a small-bodied candle (star) that gaps down, and completed by a bullish candle that gaps up and closes well into the first candle's body.

**Visual Characteristics:**
- First candle: Long bearish (red/black) candle
- Second candle: Small-bodied candle (doji or spinning top) with a gap down
- Third candle: Bullish (white/green) candle that closes well into the first candle's body
- Gaps between candles (though not strictly necessary in all markets)

**Reliability:** 90-100%

**Trading Implications:**
- Strong bullish reversal signal, especially at the end of a downtrend
- Indicates a shift from bearish to bullish sentiment
- The deeper the third candle penetrates into the first candle's body, the stronger the signal

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
if result[-1] > 0:  # Bullish pattern
    patterns['CDLMORNINGSTAR'] = 1.0  # Highest reliability weight
```

**Context Requirements:**
- Must occur after a definite downtrend
- The star should ideally have minimal body size
- Volume profile: decreasing on first candle, low on star, increasing on third candle

---

#### Piercing Pattern (CDLPIERCING)
**Description:** A two-candle bullish reversal pattern where a bearish candle is followed by a bullish candle that opens below the previous close but closes above the midpoint of the previous candle's body.

**Visual Characteristics:**
- First candle: Bearish (red/black) with substantial body
- Second candle: Bullish (white/green) that opens below previous low
- Second candle closes above the midpoint of the first candle's body
- Second candle penetrates at least 50% into the first candle's body

**Reliability:** 80-90%

**Trading Implications:**
- Moderate to strong bullish reversal signal
- Indicates buyers overcoming sellers after a downtrend
- The deeper the penetration into the first candle's body, the stronger the signal

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)
if result[-1] > 0:  # Bullish pattern
    patterns['CDLPIERCING'] = 0.9  # High reliability weight
```

**Context Requirements:**
- Should appear after a downtrend
- Higher reliability with confirmation from volume increase on second candle
- Works best in markets with defined sessions (with true gaps)

---

#### Bullish Engulfing (CDLENGULFING)
**Description:** A two-candle bullish reversal pattern where a small bearish candle is completely engulfed by a larger bullish candle that opens lower and closes higher than the previous candle.

**Visual Characteristics:**
- First candle: Small to medium bearish (red/black) candle
- Second candle: Larger bullish (white/green) candle
- Second candle completely engulfs the body of the first candle
- Second candle may or may not engulf the shadows of the first candle

**Reliability:** 80-85%

**Trading Implications:**
- Strong bullish reversal signal
- Indicates buyers overwhelmed sellers
- The larger the second candle relative to the first, the stronger the signal

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
if result[-1] > 0:  # Bullish pattern
    patterns['CDLENGULFING'] = 0.85  # High reliability weight
```

**Context Requirements:**
- Most effective at the bottom of a downtrend
- Confirmation by higher volume on engulfing candle increases reliability
- More reliable when the engulfing candle engulfs multiple previous candles

---

#### Hammer (CDLHAMMER)
**Description:** A single-candle bullish reversal pattern with a small body at the top, little or no upper shadow, and a long lower shadow that's at least twice the length of the body.

**Visual Characteristics:**
- Small body at the upper end of the trading range
- Little or no upper shadow
- Long lower shadow (at least 2x the body length)
- Body color is less important, but a bullish (white/green) body is slightly more bullish

**Reliability:** 75-80%

**Trading Implications:**
- Moderate bullish reversal signal
- Shows that sellers drove prices down but buyers regained control by close
- Requires confirmation by a bullish candle or strong volume

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
if result[-1] > 0:  # Bullish pattern
    patterns['CDLHAMMER'] = 0.8  # Medium-high reliability weight
```

**Context Requirements:**
- Must occur after a downtrend to be considered a reversal pattern
- More reliable when the lower shadow penetrates a support level
- Confirmation on the next candle is important for validation

---

#### Inverted Hammer (CDLINVERTEDHAMMER)
**Description:** A single-candle bullish reversal pattern resembling an upside-down hammer with a small body at the lower end, little or no lower shadow, and a long upper shadow.

**Visual Characteristics:**
- Small body at the lower end of the trading range
- Little or no lower shadow
- Long upper shadow (at least 2x the body length)
- Body color is less important, but a bullish (white/green) body is preferred

**Reliability:** 70-75%

**Trading Implications:**
- Moderate bullish reversal signal
- Shows buyers pushed prices up but couldn't sustain the high, yet sellers couldn't push back below the open
- Requires strong confirmation on following candle

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices)
if result[-1] > 0:  # Bullish pattern
    patterns['CDLINVERTEDHAMMER'] = 0.75  # Medium reliability weight
```

**Context Requirements:**
- Must appear after a downtrend
- Requires confirmation by a strong bullish candle on the following day
- Volume spike on the inverted hammer day increases reliability

### Bearish Patterns

#### Three Black Crows (CDL3BLACKCROWS)
**Description:** Three consecutive bearish (black/red) candles, each opening within the previous candle's body and closing lower than the previous close. Represents strong selling pressure.

**Visual Characteristics:**
- Three consecutive bearish candles
- Each candle opens within the real body of the previous candle
- Each candle closes progressively lower
- Little to no lower shadows (ideally)

**Reliability:** 90-100%

**Trading Implications:**
- Strong bearish reversal signal, especially after an uptrend
- Indicates sustained selling pressure and downward momentum
- More reliable with increasing volume over the three days

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)
if result[-1] < 0:  # Bearish pattern
    patterns['CDL3BLACKCROWS'] = -1.0  # Highest reliability weight
```

**Context Requirements:**
- Most effective after an uptrend
- More reliable when each candle closes near its low
- Volume should ideally increase with each candle

---

#### Evening Star (CDLEVENINGSTAR)
**Description:** A three-candle bearish reversal pattern consisting of a large bullish candle, followed by a small-bodied candle (star) that gaps up, and completed by a bearish candle that gaps down and closes well into the first candle's body.

**Visual Characteristics:**
- First candle: Long bullish (white/green) candle
- Second candle: Small-bodied candle (doji or spinning top) with a gap up
- Third candle: Bearish (red/black) candle that closes well into the first candle's body
- Gaps between candles (though not strictly necessary in all markets)

**Reliability:** 90-100%

**Trading Implications:**
- Strong bearish reversal signal, especially at the end of an uptrend
- Indicates a shift from bullish to bearish sentiment
- The deeper the third candle penetrates into the first candle's body, the stronger the signal

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
if result[-1] < 0:  # Bearish pattern
    patterns['CDLEVENINGSTAR'] = -1.0  # Highest reliability weight
```

**Context Requirements:**
- Must occur after a definite uptrend
- The star should ideally have minimal body size
- Volume profile: decreasing on first candle, low on star, increasing on third candle

---

#### Bearish Engulfing (CDLENGULFING)
**Description:** A two-candle bearish reversal pattern where a small bullish candle is completely engulfed by a larger bearish candle that opens higher and closes lower than the previous candle.

**Visual Characteristics:**
- First candle: Small to medium bullish (white/green) candle
- Second candle: Larger bearish (red/black) candle
- Second candle completely engulfs the body of the first candle
- Second candle may or may not engulf the shadows of the first candle

**Reliability:** 80-85%

**Trading Implications:**
- Strong bearish reversal signal
- Indicates sellers overwhelmed buyers
- The larger the second candle relative to the first, the stronger the signal

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
if result[-1] < 0:  # Bearish pattern
    patterns['CDLENGULFING'] = -0.85  # High reliability weight
```

**Context Requirements:**
- Most effective at the top of an uptrend
- Confirmation by higher volume on engulfing candle increases reliability
- More reliable when the engulfing candle engulfs multiple previous candles

---

#### Hanging Man (CDLHANGINGMAN)
**Description:** A single-candle bearish reversal pattern with a small body at the top, little or no upper shadow, and a long lower shadow that's at least twice the length of the body.

**Visual Characteristics:**
- Visually identical to a hammer, but occurs in an uptrend
- Small body at the upper end of the trading range
- Little or no upper shadow
- Long lower shadow (at least 2x the body length)
- Body color is less important, but a bearish (red/black) body is slightly more bearish

**Reliability:** 75-80%

**Trading Implications:**
- Moderate bearish reversal signal
- Shows that sellers pushed prices down during the session but buyers managed to push back
- Requires confirmation by a bearish candle or strong volume

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
if result[-1] < 0:  # Bearish pattern
    patterns['CDLHANGINGMAN'] = -0.8  # Medium-high reliability weight
```

**Context Requirements:**
- Must occur after an uptrend to be considered a reversal pattern
- More reliable with high volume
- Confirmation on the next candle is important for validation

---

#### Shooting Star (CDLSHOOTINGSTAR)
**Description:** A single-candle bearish reversal pattern with a small body at the lower end, little or no lower shadow, and a long upper shadow that's at least twice the length of the body.

**Visual Characteristics:**
- Visually identical to an inverted hammer, but occurs in an uptrend
- Small body at the lower end of the trading range
- Little or no lower shadow
- Long upper shadow (at least 2x the body length)
- Body color is less important, but a bearish (red/black) body is preferred

**Reliability:** 70-75%

**Trading Implications:**
- Moderate bearish reversal signal
- Shows buyers pushed prices up but couldn't sustain the high
- Requires confirmation on following candle

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
if result[-1] < 0:  # Bearish pattern
    patterns['CDLSHOOTINGSTAR'] = -0.75  # Medium reliability weight
```

**Context Requirements:**
- Must appear after an uptrend
- More reliable with high volume
- Requires confirmation on the following candle

---

#### Harami (CDLHARAMI)
**Description:** A two-candle pattern where a large candle is followed by a smaller candle that is completely contained within the body of the first candle. The pattern can be bullish or bearish depending on context.

**Visual Characteristics:**
- First candle: Large candle (bullish in a downtrend for bullish harami, bearish in an uptrend for bearish harami)
- Second candle: Smaller candle with body completely contained within the body of the first candle
- Second candle can be either color, but opposite of the first candle suggests stronger reversal

**Reliability:** 65-70%

**Trading Implications:**
- Moderate reversal signal
- Indicates a decrease in momentum and potential reversal
- Requires confirmation due to its lower reliability

**System Implementation:**
```python
# Implementation via TA-Lib
result = talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)
if result[-1] < 0:  # Bearish pattern
    patterns['CDLHARAMI'] = -0.7  # Medium-low reliability weight
elif result[-1] > 0:  # Bullish pattern
    patterns['CDLHARAMI'] = 0.7  # Medium-low reliability weight
```

**Context Requirements:**
- Needs to appear after a clear trend
- Often indicates indecision and requires confirmation
- More reliable when second candle is opposite color of first candle

## Multi-Day Candlestick Patterns

### Reversal Patterns

#### Three-Day Reversal Bottom
**Description:** A three-day bullish reversal pattern that occurs at the end of a downtrend, showing a transition from selling pressure to buying pressure.

**Visual Characteristics:**
- Occurs during a downtrend
- Three consecutive down days
- Third day opens lower and closes higher than previous day
- High volume on third day

**Reliability:** 80-85%

**Trading Implications:**
- Strong bullish reversal signal
- Indicates selling exhaustion and bullish momentum building
- Third day's close above previous day signals strength

**System Implementation:**
```python
def _detect_three_day_reversal_bottom(self, df: pd.DataFrame):
    """
    Detect a three-day bullish reversal pattern
    """
    # Check for downtrend
    if not (last_prices['ma5'].iloc[-1] < last_prices['ma20'].iloc[-1]):
        return None
        
    # Check for three consecutive down days, then a reversal
    if not (last_prices['is_up_day'].iloc[-4:-1].sum() == 0 and 
            last_prices['is_up_day'].iloc[-1]):
        return None
        
    # Third day opens lower and closes higher
    if not (last_prices['Open'].iloc[-1] < last_prices['Close'].iloc[-2] and 
            last_prices['Close'].iloc[-1] > last_prices['Open'].iloc[-1]):
        return None
        
    # High volume on reversal day
    volume_factor = last_prices['rel_volume'].iloc[-1]
    
    pattern_strength = min(0.6 + (0.4 * volume_factor / 2.0), 1.0)
    
    return (True, 'bullish', pattern_strength)
```

**Context Requirements:**
- Must occur after a defined downtrend
- Volume confirmation is critical
- Price should close above the previous day's close

---

#### Three-Day Reversal Top
**Description:** A three-day bearish reversal pattern that occurs at the end of an uptrend, showing a transition from buying pressure to selling pressure.

**Visual Characteristics:**
- Occurs during an uptrend
- Three consecutive up days
- Third day opens higher and closes lower than previous day
- High volume on third day

**Reliability:** 80-85%

**Trading Implications:**
- Strong bearish reversal signal
- Indicates buying exhaustion and bearish momentum building
- Third day's close below previous day signals weakness

**System Implementation:**
```python
def _detect_three_day_reversal_top(self, df: pd.DataFrame):
    """
    Detect a three-day bearish reversal pattern
    """
    # Similar to bottom reversal but in opposite context
    # Pattern detection logic with trend and volume confirmation
    # Returns (True, 'bearish', pattern_strength) if detected
```

**Context Requirements:**
- Must occur after a defined uptrend
- Volume confirmation is critical
- Price should close below the previous day's close

---

#### Key Reversal Day
**Description:** A single-day pattern that signals a potential trend reversal. In an uptrend, price makes a new high but closes near the low (bearish). In a downtrend, price makes a new low but closes near the high (bullish).

**Visual Characteristics:**
- Bullish version: In a downtrend, price makes a new low but reverses and closes above previous day's close
- Bearish version: In an uptrend, price makes a new high but reverses and closes below previous day's close
- Wide trading range (large high-low range)
- High volume

**Reliability:** 75-80%

**Trading Implications:**
- Moderate to strong reversal signal
- Indicates a significant intraday sentiment shift
- Often precedes a change in trend direction

**System Implementation:**
```python
def _detect_key_reversal(self, df: pd.DataFrame):
    """
    Detect key reversal day pattern
    """
    # Implementation logic for both bullish and bearish versions
    # Returns pattern direction and strength based on context
```

**Context Requirements:**
- Must occur after a defined trend
- High volume is essential for validation
- More reliable when price breaks a significant support/resistance level

### Continuation Patterns

#### Three Outside Up/Down
**Description:** A three-candle continuation pattern. The bullish version (up) starts with a bearish candle, followed by a bullish engulfing candle, and then a third bullish candle that closes above the second candle. The bearish version (down) is the opposite.

**Visual Characteristics:**
- Three Outside Up (Bullish):
  - First candle: Bearish
  - Second candle: Bullish engulfing
  - Third candle: Bullish, closing above second candle's close
- Three Outside Down (Bearish):
  - First candle: Bullish
  - Second candle: Bearish engulfing
  - Third candle: Bearish, closing below second candle's close

**Reliability:** 85-90%

**Trading Implications:**
- Strong continuation signal
- Confirms the engulfing pattern's reversal signal
- Provides higher reliability than a simple engulfing pattern

**System Implementation:**
```python
def _detect_three_outside_up(self, df: pd.DataFrame):
    """
    Detect Three Outside Up pattern (bullish)
    """
    # Check for bearish first candle
    # Check for bullish engulfing second candle
    # Check for bullish third candle closing above second
    # Volume confirmation
```

**Context Requirements:**
- Ideally occurs after a price correction in the direction of the larger trend
- More reliable with increasing volume on second and third candles
- Strong confirmation when third candle closes strongly in pattern direction

---

#### Three Inside Up/Down
**Description:** A three-candle continuation pattern. The bullish version (up) starts with a large bearish candle, followed by a bullish harami candle, and then a third bullish candle that closes above the second candle. The bearish version (down) is the opposite.

**Visual Characteristics:**
- Three Inside Up (Bullish):
  - First candle: Large bearish
  - Second candle: Bullish harami (small body inside first candle's body)
  - Third candle: Bullish, closing above second candle's close
- Three Inside Down (Bearish):
  - First candle: Large bullish
  - Second candle: Bearish harami (small body inside first candle's body)
  - Third candle: Bearish, closing below second candle's close

**Reliability:** 80-85%

**Trading Implications:**
- Moderate to strong continuation signal
- Confirms the harami pattern's reversal signal
- Third day confirms the reversal indicated by the harami

**System Implementation:**
```python
def _detect_three_inside_up(self, df: pd.DataFrame):
    """
    Detect Three Inside Up pattern (bullish)
    """
    # Implementation logic for the pattern detection
    # with volume and trend confirmation
```

**Context Requirements:**
- Best formed after a defined trend and pullback
- More reliable with volume confirmation
- Works best in trending markets

### Gap Patterns

#### Gap and Trap (Island Reversal)
**Description:** A pattern where price gaps in the direction of the trend, forms a small consolidation or island, then gaps back in the opposite direction, trapping traders who entered on the first gap.

**Visual Characteristics:**
- First gap in the direction of the trend
- Short consolidation period (1-3 candles)
- Second gap against the trend direction
- Island of price action isolated by the two gaps

**Reliability:** 85-90%

**Trading Implications:**
- Strong reversal signal
- Indicates a rapid sentiment shift
- Often marks significant turning points

**System Implementation:**
```python
def _detect_gap_and_trap(self, df: pd.DataFrame):
    """
    Detect Gap and Trap (Island Reversal) pattern
    """
    # Check for first gap (trend direction)
    # Check for consolidation pattern
    # Check for second gap (counter-trend)
    # Verify pattern with volume profile
```

**Context Requirements:**
- Typically occurs at market extremes
- More reliable with high volume on the reversal gap
- Best in markets where true gaps can form (not 24-hour markets)

---

#### Exhaustion Gap
**Description:** A gap that occurs at the end of a strong trend and is characterized by high volume but fails to continue the trend, signaling potential exhaustion.

**Visual Characteristics:**
- Gap in the direction of the prevailing trend
- Unusually high volume on the gap day
- Price fails to continue in gap direction
- Often followed by reversal price action

**Reliability:** 75-80%

**Trading Implications:**
- Moderate to strong reversal signal
- Indicates trend exhaustion
- Often marks the end of a trend phase

**System Implementation:**
```python
def _detect_exhaustion_gap(self, df: pd.DataFrame):
    """
    Detect Exhaustion Gap pattern
    """
    # Check for gap in trend direction
    # Verify high volume
    # Check for failure to continue trend
    # Confirm with reversal price action
```

**Context Requirements:**
- Occurs after an extended trend
- Volume spike is essential
- Failure to make significant new highs/lows after the gap

## Price Action Patterns

### Double Top/Bottom
**Description:** A reversal pattern where price makes two peaks/troughs at approximately the same level, with a moderate trough/peak in between.

**Visual Characteristics:**
- Double Top (Bearish):
  - Two distinct peaks at approximately the same price level
  - Moderate trough between peaks
  - Second peak often has lower volume
- Double Bottom (Bullish):
  - Two distinct troughs at approximately the same price level
  - Moderate peak between troughs
  - Second trough often has higher volume

**Reliability:** 75-80%

**Trading Implications:**
- Moderate reversal signal
- Indicates failure to break resistance/support
- Confirms when price breaks the neck line (middle trough/peak)

**System Implementation:**
```python
def _is_double_top(self, peaks: np.ndarray, volumes: np.ndarray,
                 threshold: float = 0.02) -> bool:
    """Check for double top pattern"""
    if len(peaks) < 2:
        return False
        
    price_diff = abs(peaks[1] - peaks[0]) / peaks[0]
    volume_decline = volumes[1] < volumes[0]
    
    return price_diff < threshold and volume_decline

def _is_double_bottom(self, troughs: np.ndarray, volumes: np.ndarray,
                    threshold: float = 0.02) -> bool:
    """Check for double bottom pattern"""
    if len(troughs) < 2:
        return False
        
    price_diff = abs(troughs[1] - troughs[0]) / troughs[0]
    volume_increase = volumes[1] > volumes[0]
    
    return price_diff < threshold and volume_increase
```

**Context Requirements:**
- Peaks/troughs should be separated by enough time (at least 10-20 bars)
- Volume pattern helps confirm (decreasing at second top, increasing at second bottom)
- Breakout from neckline with volume confirms the pattern

---

### Head and Shoulders
**Description:** A reversal pattern consisting of three peaks, with the middle peak (head) higher than the two surrounding peaks (shoulders), which are roughly equal in height.

**Visual Characteristics:**
- Head and Shoulders (Bearish):
  - Left shoulder: First peak in uptrend
  - Head: Higher peak
  - Right shoulder: Third peak, lower than head, similar to left shoulder
  - Neckline: Support line connecting the lows between shoulders and head
- Inverse Head and Shoulders (Bullish):
  - Left shoulder: First trough in downtrend
  - Head: Lower trough
  - Right shoulder: Third trough, higher than head, similar to left shoulder
  - Neckline: Resistance line connecting the highs between shoulders and head

**Reliability:** 80-85%

**Trading Implications:**
- Strong reversal signal
- Confirms when price breaks the neckline
- Target often measured as distance from head to neckline, projected from breakpoint

**System Implementation:**
```python
def _is_head_and_shoulders(self, peaks: np.ndarray, inverse: bool = False,
                        threshold: float = 0.05) -> bool:
    """Check for head and shoulders pattern"""
    if len(peaks) < 3:
        return False
        
    if not inverse:
        # Regular H&S: middle peak higher than shoulders
        return (peaks[1] > peaks[0] and
               peaks[1] > peaks[2] and
               abs(peaks[0] - peaks[2]) / peaks[0] < threshold)
    else:
        # Inverse H&S: middle trough lower than shoulders
        return (peaks[1] < peaks[0] and
               peaks[1] < peaks[2] and
               abs(peaks[0] - peaks[2]) / peaks[0] < threshold)
```

**Context Requirements:**
- Pattern should form after an extended trend
- Volume typically decreases with each peak in H&S top
- Volume typically increases with each trough in inverse H&S
- Neckline break should be accompanied by volume increase

---

### Cup and Handle
**Description:** A bullish continuation pattern resembling a cup (U-shape) followed by a handle (slight downward drift). Represents a consolidation period before continuation of an uptrend.

**Visual Characteristics:**
- Cup: Rounded U-shaped pattern over several weeks or months
- Handle: Short downward drift after the cup, typically lasting 1-4 weeks
- Breakout: Upward move past the resistance level established by the cup rim

**Reliability:** 80-85%

**Trading Implications:**
- Strong bullish continuation signal
- Confirms with breakout above handle resistance
- Target usually measured from cup depth added to breakout point

**System Implementation:**
```python
def _is_cup_and_handle(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
    """
    Detect cup and handle pattern
    """
    # Cup portion typically spans 30-40 bars
    cup_period = min(40, len(prices) - 10)
    
    # Find lowest point in cup range
    cup_low_idx = np.argmin(prices[:cup_period])
    
    # Check if cup low point is roughly in the middle (U-shape)
    if cup_low_idx < cup_period * 0.3 or cup_low_idx > cup_period * 0.7:
        return False
        
    # Left and right cup heights should be similar
    left_height = prices[0] - prices[cup_low_idx]
    right_height = prices[cup_period] - prices[cup_low_idx]
    
    if abs(left_height - right_height) / left_height > 0.3:
        return False
        
    # Handle portion should be a small pullback
    handle_start = cup_period
    handle_depth = (max(prices[handle_start:]) - min(prices[handle_start:])) / prices[handle_start]
    
    # Handle shouldn't be too deep
    if handle_depth > 0.15:
        return False
        
    # Check for breakout
    if prices[-1] > max(prices[5:cup_period]):
        # Volume confirmation
        if volumes[-1] > np.mean(volumes[-5:]):
            return True
            
    return False
```

**Context Requirements:**
- Must form after an uptrend
- Cup should be rounded (not V-shaped)
- Handle should retrace no more than 1/3 to 1/2 of the cup's advance
- Volume should increase on breakout

## Technical Bands and Overlays

### Bollinger Bands
**Description:** A volatility-based indicator consisting of a middle band (SMA) with upper and lower bands set at standard deviation levels above and below the middle band.

**Calculation:**
```
Middle Band = n-period SMA
Upper Band = Middle Band + (k × n-period standard deviation)
Lower Band = Middle Band - (k × n-period standard deviation)
```
Where:
- n is typically 20 periods
- k is typically 2 standard deviations

**Visual Characteristics:**
- Three bands that expand and contract based on volatility
- Upper band: Resistance in ranging markets, target in trending markets
- Lower band: Support in ranging markets, target in trending markets
- Width: Indicator of volatility

**Trading Applications:**
- Bollinger Band Squeeze: Bands narrow, indicating low volatility and potential breakout
- Overbought/Oversold: Price touching or exceeding bands may indicate extreme conditions
- Mean Reversion: Price tends to return to the middle band after touching outer bands
- Trend Strength: Strong trends can ride the bands

**System Implementation:**
```python
# Calculate Bollinger Bands
def calculate_bollinger_bands(prices, period=20, num_std=2):
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

# Detecting Bollinger Band squeeze
def detect_bollinger_squeeze(upper, lower, middle, lookback=20):
    band_width = (upper - lower) / middle
    if band_width[-1] < min(band_width[-lookback:-1]):
        return True
    return False
```

**Context Considerations:**
- More effective in ranging markets for mean reversion
- Band touches are more significant with low historical volatility
- Bands work best with other confirmation indicators
- Band width provides volatility context for other patterns

---

### Keltner Channels
**Description:** A volatility-based indicator similar to Bollinger Bands but uses Average True Range (ATR) instead of standard deviation to set the bands around a central EMA.

**Calculation:**
```
Middle Line = n-period EMA
Upper Channel = Middle Line + (m × ATR)
Lower Channel = Middle Line - (m × ATR)
```
Where:
- n is typically 20 periods for the EMA
- m is typically 1.5 or 2 multiplier for the ATR
- ATR is typically a 10 or 14-period Average True Range

**Visual Characteristics:**
- Three lines/bands that expand and contract based on volatility
- Less responsive to rapid price changes than Bollinger Bands
- More stable channel boundaries than Bollinger Bands
- Often used alongside Bollinger Bands for confirmation

**Trading Applications:**
- Channel Breakouts: Price breaking above/below channels suggests strong momentum
- Trend Direction: Price hugging upper/lower channel indicates strong trend
- Mean Reversion: Price tends to revert to middle line
- Volatility Squeeze: When Keltner Channels contain Bollinger Bands, a strong breakout often follows

**System Implementation:**
```python
# Calculate Keltner Channels
def calculate_keltner_channels(df, ema_period=20, atr_period=14, multiplier=2.0):
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['atr'] = calculate_atr(df, atr_period)
    df['upper_kc'] = df['ema'] + (multiplier * df['atr'])
    df['lower_kc'] = df['ema'] - (multiplier * df['atr'])
    return df
```

**Context Considerations:**
- More reliable in trending markets than Bollinger Bands
- Channel width indicates volatility similar to Bollinger Bands
- Less prone to false breakouts than Bollinger Bands
- Often used for trend trading strategies

---

### VWAP (Volume-Weighted Average Price)
**Description:** A volume-weighted average of price, typically calculated for a single trading session. It represents the average price at which a security has traded throughout the day, based on both volume and price.

**Calculation:**
```
VWAP = Σ(Price × Volume) / Σ(Volume)
```

**Visual Characteristics:**
- Single line overlaid on the price chart
- Resets at the beginning of each trading session
- Acts as a dynamic support/resistance level
- Represents "fair value" from an institutional perspective

**Trading Applications:**
- Institutional Benchmark: Used to assess execution quality
- Support/Resistance: Acts as intraday support/resistance
- Trend Confirmation: Price above/below VWAP suggests bullish/bearish bias
- Mean Reversion: Price tends to revert to VWAP during range-bound markets

**System Implementation:**
```python
def calculate_vwap(df):
    # Assuming df has columns: high, low, close, volume
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']
    
    # Cumulative values for the session
    df['cum_tp_vol'] = df.groupby(df.index.date)['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby(df.index.date)['volume'].cumsum()
    
    # VWAP calculation
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    return df
```

**Context Considerations:**
- Most relevant for intraday trading
- More significant in liquid markets with institutional participation
- Less useful for overnight positions or multiple day analysis
- Works best in markets with defined trading sessions

## Volatility Indicators

### ATR (Average True Range)
**Description:** A measure of market volatility that accounts for gaps in price movement. It calculates the average of the true range over a specified period.

**Calculation:**
```
True Range (TR) = Greatest of:
1. Current High - Current Low
2. |Current High - Previous Close|
3. |Current Low - Previous Close|

ATR = n-period average of TR
```

**Visual Characteristics:**
- Plotted as a single line below the price chart
- Higher values indicate higher volatility
- Lower values indicate lower volatility
- Does not indicate direction, only volatility

**Trading Applications:**
- Position Sizing: Adjust position size based on volatility
- Stop Loss Placement: Set stops at a multiple of ATR
- Breakout Confirmation: Increasing ATR confirms breakouts
- Volatility Contraction: Decreasing ATR may precede significant moves

**System Implementation:**
```python
def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range for position sizing"""
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    # Use TA-Lib for ATR calculation
    atr = talib.ATR(high, low, close, timeperiod=period)
    
    # Return the latest ATR value
    return atr[-1]
```

**Context Considerations:**
- ATR is non-directional (doesn't indicate bullish/bearish)
- Useful for comparing volatility across different securities
- Often used with other indicators for context
- Helps identify appropriate risk parameters for a given market

---

### Volatility Squeeze
**Description:** A pattern that occurs when volatility decreases significantly, often indicated by Bollinger Bands contracting inside Keltner Channels, signaling a potential explosive move.

**Calculation:**
- Bollinger Bands contract (width decreases)
- Keltner Channels remain relatively stable
- Bollinger Bands move inside Keltner Channels

**Visual Characteristics:**
- Narrowing of the Bollinger Bands
- Bollinger Bands inside Keltner Channels
- Followed by expansion of bands and significant price movement
- Often precedes major trend changes or continuation

**Trading Applications:**
- Breakout Anticipation: Prepare for significant move
- Options Strategies: Potentially profitable for options straddles/strangles
- Trend Detection: First move after squeeze often indicates new trend direction
- Risk Management: Adjust position size for expected volatility expansion

**System Implementation:**
```python
def _detect_volatility_squeeze(self, df: pd.DataFrame):
    """Detect volatility squeeze pattern"""
    # Calculate Bollinger Bands
    upper_bb, middle_bb, lower_bb = talib.BBANDS(
        df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    
    # Calculate Keltner Channels
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    ema = talib.EMA(typical_price, timeperiod=20)
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    upper_kc = ema + (atr * 1.5)
    lower_kc = ema - (atr * 1.5)
    
    # Identify squeeze
    squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    
    # Check if we're coming out of a squeeze
    if squeeze.iloc[-2] and not squeeze.iloc[-1]:
        # Determine direction based on momentum
        if df['Close'].iloc[-1] > ema.iloc[-1]:
            return (True, 'bullish', 0.8)
        else:
            return (True, 'bearish', 0.8)
    
    return None
```

**Context Considerations:**
- Works best in markets that alternate between low and high volatility
- More reliable with longer squeeze duration
- Breakout direction often confirmed by momentum indicators
- False signals possible in extremely calm or chaotic markets

## Volume-Based Indicators

### OBV (On-Balance Volume)
**Description:** A cumulative indicator that adds volume on up days and subtracts volume on down days, measuring buying and selling pressure.

**Calculation:**
```
If Close > Previous Close:
    OBV = Previous OBV + Current Volume
If Close < Previous Close:
    OBV = Previous OBV - Current Volume
If Close = Previous Close:
    OBV = Previous OBV
```

**Visual Characteristics:**
- Line graph plotted below the price chart
- Rising OBV suggests accumulation (buying pressure)
- Falling OBV suggests distribution (selling pressure)
- Divergences between OBV and price often signal potential reversals

**Trading Applications:**
- Trend Confirmation: OBV should move in the same direction as price in a healthy trend
- Divergence: OBV moving opposite to price suggests potential reversal
- Breakout Confirmation: Volume should increase during breakouts
- Support/Resistance: OBV can form its own trend lines and patterns

**System Implementation:**
```python
def calculate_obv(df):
    """Calculate On-Balance Volume"""
    df['daily_ret'] = df['close'].pct_change()
    df['direction'] = np.where(df['daily_ret'] > 0, 1, np.where(df['daily_ret'] < 0, -1, 0))
    df['direction_volume'] = df['direction'] * df['volume']
    df['obv'] = df['direction_volume'].cumsum()
    return df
```

**Context Considerations:**
- More reliable in trending markets
- Should be used with price action for confirmation
- Sensitive to high volume days
- Best used for intermediate to long-term analysis

---

### CMF (Chaikin Money Flow)
**Description:** A volume-weighted measure of accumulation/distribution over a specified period, indicating buying and selling pressure.

**Calculation:**
```
Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
Money Flow Volume = Money Flow Multiplier × Volume
CMF = 21-period Sum of Money Flow Volume / 21-period Sum of Volume
```

**Visual Characteristics:**
- Oscillator typically bounded between -1 and +1
- Positive values indicate buying pressure
- Negative values indicate selling pressure
- Zero line crossovers can signal shifts in momentum

**Trading Applications:**
- Trend Confirmation: Positive CMF confirms uptrends, negative CMF confirms downtrends
- Divergence: CMF diverging from price action suggests potential reversal
- Support/Resistance Breaks: CMF can confirm breakouts
- Volume Analysis: Provides volume context to price movements

**System Implementation:**
```python
def calculate_cmf(df, period=20):
    """Calculate Chaikin Money Flow"""
    # Money Flow Multiplier
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.replace([np.inf, -np.inf], 0)
    
    # Money Flow Volume
    mfv = mfm * df['volume']
    
    # Chaikin Money Flow
    df['cmf'] = mfv.rolling(period).sum() / df['volume'].rolling(period).sum()
    return df
```

**Context Considerations:**
- More reliable when used with other indicators
- Best for intermediate-term analysis
- Can be affected by large volume spikes
- Most effective in trending markets

## System-Specific Pattern Integration

### Pattern Weighting and Combination

The Nifty 500 Trading System integrates multiple pattern types using a weighted approach:

```python
def _combine_signals(self, pattern_signals, multi_day_patterns, price_patterns, 
                   volume_signal, levels, current_price):
    """Combine all signals into final signal and confidence"""
    # Weight for each component
    weights = {
        'candlestick': 0.25,
        'multi_day': 0.25,
        'price_pattern': 0.2,
        'volume': 0.15,
        'support_resistance': 0.15
    }
    
    # Calculate weighted signals for each component
    if pattern_signals:
        pattern_score = np.mean(list(pattern_signals.values()))
    else:
        pattern_score = 0

    if multi_day_patterns:
        multi_day_score = np.mean(list(multi_day_patterns.values()))
    else:
        multi_day_score = 0
        
    if price_patterns:
        price_score = np.mean(list(price_patterns.values()))
    else:
        price_score = 0
        
    # Calculate support/resistance signal
    sr_score = 0
    # ... support/resistance logic ...
        
    # Combine signals
    signal_strength = (
        weights['candlestick'] * pattern_score +
        weights['multi_day'] * multi_day_score +
        weights['price_pattern'] * price_score +
        weights['volume'] * volume_signal +
        weights['support_resistance'] * sr_score
    )
    
    # Calculate confidence based on agreement
    signals = [
        1 if s > 0 else -1 if s < 0 else 0
        for s in [pattern_score, multi_day_score, price_score, 
                 volume_signal, sr_score]
        if s != 0
    ]
    
    if signals:
        agreement = np.mean(np.abs(signals))
        confidence = min(1.0, agreement * abs(signal_strength))
    else:
        confidence = 0
        
    return signal_strength, confidence
```

### Context-Aware Pattern Adjustment

The system adjusts pattern weights based on market context:

```python
def _adjust_pattern_strength(self, pattern_name, base_strength, market_context, data):
    """
    Adjust pattern strength based on market context
    """
    # Get base weight for this pattern
    base_weight = self.pattern_weights.get(pattern_name, 75) / 100.0
    
    # Apply market context adjustments
    context_multiplier = 1.0
    
    # Different patterns are more relevant in different market contexts
    if pattern_name in ['three_day_reversal_bottom', 'three_day_reversal_top']:
        # Reversal patterns are more reliable at extremes
        if 'strong' in market_context['trend']:
            context_multiplier *= 1.2
        if market_context['volume'] == 'high':
            context_multiplier *= 1.3
    
    elif pattern_name in ['three_day_strength', 'three_outside_up']:
        # Continuation patterns are more reliable in established trends
        if market_context['trend'] in ['uptrend', 'downtrend']:
            context_multiplier *= 1.2
    
    # Ensure the final strength is in the range [0, 1]
    adjusted_strength = min(base_weight * base_strength * context_multiplier, 1.0)
    
    return adjusted_strength
```

### Pattern Sequencing and Confluence

The system also evaluates the reliability of patterns based on their sequence and confluence with other technical factors:

1. **Sequential Patterns**: Multiple patterns appearing in sequence receive higher weights
2. **Volume Confirmation**: Patterns with confirming volume profiles are given higher reliability scores
3. **Indicator Confluence**: Patterns that align with other technical indicators receive higher weights
4. **Support/Resistance Interaction**: Patterns forming near key support/resistance levels are emphasized
5. **Volatility Context**: Pattern reliability is adjusted based on the current volatility regime

This approach ensures that trading decisions are based on comprehensive technical analysis rather than isolated pattern signals, significantly improving the reliability of the system's predictions.
