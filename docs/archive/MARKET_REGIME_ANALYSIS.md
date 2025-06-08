# Market Regime Analysis

## Overview

The Market Regime Analysis module provides a comprehensive framework for detecting and analyzing different market regimes, regime transitions, and regime-specific trading parameters. It helps adapt trading strategies to current market conditions and provides performance metrics for each regime.

## Key Features

1. **Multi-factor Regime Detection**
   - Trend analysis (strong uptrend to strong downtrend)
   - Volatility analysis (low, normal, high)
   - Volume analysis (low, normal, high)
   - Momentum analysis (strong, weak, neutral)

2. **Regime Transition Detection**
   - Identifies major shifts between regimes
   - Detects specific transition types (e.g., volatility expansion, trend reversal)
   - Provides confidence scores for transitions

3. **Regime-Specific Parameters**
   - Adjusts position sizing based on volatility
   - Modifies stop-loss and take-profit levels
   - Adjusts strategy weights for different regimes
   - Customizes entry thresholds based on market conditions

4. **Regime Performance Analytics**
   - Calculates returns, volatility, and Sharpe ratios by regime
   - Measures win rates and drawdowns in different regimes
   - Provides insights for strategy optimization

## Usage Examples

### Basic Regime Detection

```python
from market_regime import MarketRegimeAnalyzer

# Initialize analyzer
analyzer = MarketRegimeAnalyzer()

# Detect current regime
current_regime = analyzer.detect_regime(market_data)

print(f"Current Trend: {current_regime.trend}")
print(f"Volatility: {current_regime.volatility}")
print(f"Volume: {current_regime.volume}")
print(f"Momentum: {current_regime.momentum}")
print(f"Confidence: {current_regime.confidence:.2f}")
```

### Detecting Regime Transitions

```python
# Check for regime transitions
is_transition, transition_type = analyzer.detect_regime_transition(historical_data)

if is_transition:
    print(f"Detected {transition_type} transition!")
```

### Getting Regime-Specific Parameters

```python
# Get optimal parameters for current regime
params = analyzer.get_regime_parameters(current_regime)

# Apply to trading strategy
position_size = params['position_sizing']['size']
stop_loss_multiplier = params['stop_loss']['atr_multiplier']
```

### Analyzing Regime Performance

```python
# Calculate performance metrics by regime
regime_metrics = analyzer.calculate_regime_metrics(historical_data, returns)

# Compare performance across regimes
for regime, metrics in regime_metrics.items():
    print(f"\n{regime.upper()}:")
    print(f"  Return: {metrics['return']:.4f}")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
```

## Regime Types

### Trend Regimes
- `strong_up`: Strong uptrend with price above all moving averages
- `moderate_up`: Moderate uptrend with price above medium-term MA
- `weak_up`: Weak uptrend with early signs of bullishness
- `neutral`: No clear trend direction
- `weak_down`: Weak downtrend with early signs of bearishness
- `moderate_down`: Moderate downtrend with price below medium-term MA
- `strong_down`: Strong downtrend with price below all moving averages

### Volatility Regimes
- `low`: Below 25th percentile of historical volatility
- `normal`: Between 25th and 75th percentile
- `high`: Above 75th percentile

### Volume Regimes
- `low`: Below 1 standard deviation from mean
- `normal`: Within 1 standard deviation from mean
- `high`: Above 1 standard deviation from mean

### Momentum Regimes
- `strong`: Strong momentum (RSI/MFI > 70 or positive MACD)
- `weak`: Weak momentum (RSI/MFI < 30 or negative MACD)
- `neutral`: Neutral momentum

## Implementation Details

The `MarketRegimeAnalyzer` class contains methods for:
- Analyzing trend conditions using moving averages
- Detecting volatility regimes using standard deviation of returns
- Evaluating volume conditions with rolling statistics
- Analyzing momentum using RSI, MFI, and MACD
- Calculating confidence scores for regime classifications
- Detecting transitions between regimes
- Generating regime-specific trading parameters
- Calculating performance metrics for each regime type

## Integration with Trading Strategies

Market regime analysis can be integrated with trading strategies to:
1. Filter signals based on current regime
2. Adjust position sizing based on volatility
3. Customize stop-loss and take-profit levels
4. Rebalance strategy weights for optimal performance
5. Provide regime context for trade decisions

## Performance Considerations

- The analyzer requires sufficient historical data for reliable regime detection
- Lookback periods are customizable based on trading timeframes
- Regime transitions are more reliable than absolute regime classifications
- Confidence scores help assess the reliability of regime detection
