# Trading System Documentation

## System Overview

This trading system combines multiple approaches:
1. Technical Analysis with correlation-aware indicators
2. Reinforcement Learning for strategy optimization
3. LLaMA-based financial reasoning

## Technical Indicators

### Trend Indicators
- **Moving Averages (SMA/EMA)**: Used for trend identification
  - SMA20: Short-term trend
  - SMA50: Medium-term trend
  - SMA200: Long-term trend
  Implementation: Using talib with optimized lookback periods

- **MACD**: Trend and momentum indicator
  - Fast EMA (12 periods)
  - Slow EMA (26 periods)
  - Signal line (9-period EMA of MACD)
  Best used for: Trend confirmation and momentum measurement

- **ADX**: Measures trend strength
  - Values > 25 indicate strong trend
  - Used with DMI for direction confirmation

### Momentum Indicators
- **RSI**: Measures overbought/oversold conditions
  - < 30: Oversold
  - > 70: Overbought
  Implementation: 14-period RSI with dynamic thresholds

- **Stochastic Oscillator**: Price position relative to range
  - %K: Fast stochastic
  - %D: Slow stochastic
  Best used with trend confirmation

### Volatility Indicators
- **Bollinger Bands**: Dynamic volatility bands
  - Middle Band: 20-period SMA
  - Upper/Lower Bands: ±2 standard deviations
  Used for: Volatility-based entries and exits

- **ATR**: Measures market volatility
  - Used for position sizing
  - Stop loss calculation
  Implementation: 14-period ATR with adaptive scaling

### Volume Indicators
- **OBV**: Cumulative volume flow
  - Confirms price trends
  - Divergence analysis

- **Volume Profile**: Price levels with highest trading activity
  - POC (Point of Control)
  - Value Area High/Low
  Implementation: Rolling window analysis

## Strategy Implementation

### 1. Correlation Analysis
```python
def get_uncorrelated_indicators(threshold=0.7):
    """
    Identifies relatively independent indicators to avoid redundant signals
    Returns groups of indicators with correlation < threshold
    """
```

### 2. Indicator Combinations
The system combines technical indicators based on:
- Signal strength and reliability
- Cross-validation across multiple timeframes
- Adaptive weighting based on market conditions

### 3. Signal Generation
Each indicator provides signals with:
- Direction (buy/sell/hold)
- Strength (0-1)
- Timeframe (short/medium/long)

## Strategy Implementation Guide

### 1. Trend Following Strategies

#### a. Multi-Timeframe Moving Average Strategy
```python
def multi_timeframe_ma_strategy(data):
    # Higher timeframe trend (1H)
    major_trend = data['SMA200'] > data['SMA50']
    
    # Lower timeframe signals (5min)
    entry_signal = (data['EMA20'] > data['EMA50']) & (data['RSI'] < 60)
    
    # Volume confirmation
    volume_confirmed = data['Volume'] > data['Volume'].rolling(20).mean()
    
    return entry_signal & major_trend & volume_confirmed
```

Key Components:
- Trend alignment across timeframes
- Volume confirmation for entries
- RSI filter to avoid overbought conditions
- Dynamic position sizing based on ATR

Risk Management:
- Stop Loss: 2 * ATR below entry
- Take Profit: 3 * ATR above entry
- Maximum position size: 2% account risk

#### b. Momentum Breakout Strategy
```python
def momentum_breakout_strategy(data):
    # Volatility threshold
    atr = calculate_atr(data, period=14)
    
    # Price breakout
    breakout = (data['Close'] - data['Close'].shift(1)) > 1.5 * atr
    
    # Volume surge
    volume_surge = data['Volume'] > 2 * data['Volume'].rolling(20).mean()
    
    # Trend confirmation
    trend_aligned = data['MACD'] > data['MACD_Signal']
    
    return breakout & volume_surge & trend_aligned
```

Position Management:
- Entry: Market order on breakout confirmation
- Initial Stop: Below previous swing low
- Trailing Stop: Parabolic SAR
- Scale out: 33% at 1:1, 33% at 2:1, remainder at 3:1

### 2. Mean Reversion Strategies

#### a. Oversold Bounce Strategy
```python
def oversold_bounce_strategy(data):
    # Multiple indicator confirmation
    rsi_oversold = data['RSI'] < 30
    stoch_oversold = (data['STOCH_K'] < 20) & (data['STOCH_D'] < 20)
    
    # Price near support
    near_support = data['Close'] < data['BB_Lower']
    
    # Volume divergence
    volume_divergence = (
        data['Close'].rolling(5).std().expanding().mean() >
        data['Volume'].rolling(5).std().expanding().mean()
    )
    
    return rsi_oversold & stoch_oversold & near_support & volume_divergence
```

Risk Parameters:
- Entry: Limit orders at support levels
- Stop Loss: Below recent low
- Take Profit: Mean + 1σ
- Position Size: Inverse volatility weighted

#### b. Volatility Mean Reversion
```python
def volatility_reversion_strategy(data):
    # Bollinger Band setup
    bb_width = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    extreme_width = bb_width > bb_width.rolling(100).mean() + 2 * bb_width.rolling(100).std()
    
    # Price extreme
    price_extreme = (
        (data['Close'] < data['BB_Lower']) |
        (data['Close'] > data['BB_Upper'])
    )
    
    # Volume confirmation
    normal_volume = (
        data['Volume'] < data['Volume'].rolling(20).mean() * 1.5
    )
    
    return extreme_width & price_extreme & normal_volume
```

### 3. Machine Learning Enhanced Strategies

#### a. LLaMA-Assisted Pattern Recognition
```python
def llama_pattern_strategy(data, model):
    # Get market context
    context = {
        'price_action': extract_price_patterns(data),
        'indicators': calculate_indicators(data),
        'market_regime': identify_regime(data)
    }
    
    # Get model analysis
    analysis = model.evaluate_strategy(context)
    
    # Combine with technical signals
    technical_valid = validate_technicals(data)
    
    return {
        'signal': analysis['recommendation'],
        'confidence': analysis['confidence'],
        'supporting_factors': analysis['reasoning']
    } if technical_valid else None
```

Key Features:
- Pattern recognition from historical data
- Real-time market regime classification
- Technical confirmation of AI signals
- Confidence-based position sizing

#### b. Hybrid RL-Technical Strategy
```python
def hybrid_rl_strategy(data, rl_model):
    # State preparation
    state = prepare_state(data)
    
    # Get RL action
    action = rl_model.predict(state)
    
    # Technical validation
    tech_signals = validate_technical_setup(data)
    
    # Risk assessment
    risk_score = calculate_risk_score(data)
    
    # Combined decision
    return {
        'action': action if tech_signals['valid'] else 'hold',
        'position_size': adjust_size_by_risk(risk_score),
        'stop_loss': calculate_adaptive_stop(data)
    }
```

### 4. Implementation Best Practices

#### a. Entry Conditions
1. Multiple Timeframe Confirmation
```python
def validate_entry(data):
    # Higher timeframe trend (1H)
    htf_trend = calculate_trend(resample_timeframe(data, '1H'))
    
    # Current timeframe setup (5min)
    current_setup = validate_setup(data)
    
    # Lower timeframe trigger (1min)
    ltf_trigger = get_entry_trigger(resample_timeframe(data, '1min'))
    
    return all([htf_trend, current_setup, ltf_trigger])
```

2. Volume Analysis
```python
def analyze_volume(data):
    return {
        'relative_volume': data['Volume'] / data['Volume'].rolling(20).mean(),
        'volume_trend': calculate_volume_trend(data),
        'price_volume_correlation': calculate_pv_correlation(data)
    }
```

#### b. Position Management
1. Dynamic Position Sizing
```python
def calculate_position_size(data, account_balance, risk_per_trade=0.02):
    # ATR-based stop distance
    atr = calculate_atr(data)
    stop_distance = 2 * atr
    
    # Calculate risk amount
    risk_amount = account_balance * risk_per_trade
    
    # Calculate position size
    price = data['Close'].iloc[-1]
    return (risk_amount / stop_distance) / price
```

2. Adaptive Stop Loss
```python
def calculate_adaptive_stop(data):
    # Base stop distance
    atr = calculate_atr(data)
    base_stop = 2 * atr
    
    # Volatility adjustment
    vol_multiplier = calculate_volatility_multiplier(data)
    
    # Support/Resistance levels
    key_levels = identify_key_levels(data)
    
    return optimize_stop_level(base_stop, vol_multiplier, key_levels)
```

#### c. Exit Management
1. Scaling Out
```python
def manage_exits(position, data):
    if position['profit'] >= position['initial_stop'] * 2:
        # Scale out 50%
        return execute_scale_out(position, 0.5)
    
    if position['profit'] >= position['initial_stop'] * 3:
        # Scale out remaining
        return execute_scale_out(position, 1.0)
```

2. Trailing Stop
```python
def update_trailing_stop(position, data):
    # ATR-based trail
    atr = calculate_atr(data)
    new_stop = data['Close'].iloc[-1] - (2 * atr)
    
    # Only move stop up
    if new_stop > position['current_stop']:
        position['current_stop'] = new_stop
```

### 5. Risk Management Framework

#### a. Pre-Trade Checklist
1. Market Conditions
```python
def validate_market_conditions():
    return {
        'volatility_suitable': check_volatility(),
        'liquidity_adequate': check_liquidity(),
        'spread_acceptable': check_spread(),
        'news_events_clear': check_upcoming_news()
    }
```

2. Position Risk
```python
def calculate_total_risk(new_position):
    return {
        'portfolio_heat': get_portfolio_risk(),
        'correlation_risk': get_correlation_exposure(),
        'sector_exposure': get_sector_risk(),
        'total_leverage': get_current_leverage()
    }
```

#### b. Monitoring and Adjustment
1. Real-time Risk Metrics
```python
def monitor_risk_metrics():
    return {
        'var': calculate_value_at_risk(),
        'sharpe': calculate_rolling_sharpe(),
        'max_drawdown': calculate_drawdown(),
        'win_rate': calculate_win_rate()
    }
```

2. Strategy Adjustment
```python
def adjust_strategy_parameters(metrics):
    if metrics['sharpe'] < 1.0:
        tighten_risk_parameters()
    if metrics['win_rate'] < 0.4:
        review_entry_conditions()
    if metrics['max_drawdown'] > 0.1:
        reduce_position_sizes()
```

### 6. Performance Analysis

#### a. Strategy Metrics
Track these key metrics for each strategy:
- Win Rate
- Profit Factor
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Average Win/Loss Ratio
- Recovery Factor

#### b. Optimization Process
1. Parameter Optimization
```python
def optimize_parameters(strategy, data):
    results = []
    for params in generate_parameter_combinations():
        performance = backtest_strategy(strategy, data, params)
        results.append({
            'params': params,
            'metrics': calculate_metrics(performance)
        })
    return find_optimal_parameters(results)
```

2. Adaptation Process
```python
def adapt_strategy(strategy, market_conditions):
    # Adjust parameters based on volatility
    if market_conditions['volatility'] > threshold:
        widen_stops()
        reduce_position_size()
    
    # Adjust based on trend strength
    if market_conditions['trend_strength'] < minimum:
        switch_to_ranging_mode()
    
    # Adjust based on volume
    if market_conditions['volume'] < average:
        increase_confirmation_requirements()
```

Remember:
1. Always validate strategy changes in a paper trading environment first
2. Monitor strategy degradation and adapt parameters
3. Keep detailed logs of all trades and decisions
4. Regular review and optimization of parameters
5. Maintain a comprehensive risk management framework

## LLaMA Financial Reasoning

### Training Process
1. **Data Collection**
   - Successful and failed trades
   - Market conditions
   - Technical setups
   - Outcomes and reasoning

2. **Fine-tuning**
   - Supervised fine-tuning (SFT)
   - Custom financial prompts
   - Strategy evaluation
   - Risk assessment

3. **Model Usage**
   - Real-time strategy evaluation
   - Market condition analysis
   - Risk factor identification
   - Trade setup validation

### Example Training Data
```json
{
    "market_data": {
        "prices": [...],
        "volumes": [...],
        "timeframe": "5min"
    },
    "indicators": {
        "RSI": 65.4,
        "MACD": 0.23,
        "BB_Upper": 125.6,
        ...
    },
    "outcome": {
        "profit_pct": 2.3,
        "hold_time": 12,
        "exit_reason": "take_profit"
    }
}
```

## Reinforcement Learning Implementation

### State Space
1. Technical Indicators
   - Trend indicators
   - Momentum indicators
   - Volatility indicators
   - Volume indicators

2. Market Context
   - Price levels
   - Volume profile
   - Time features
   - Market regime

### Action Space
1. Entry Actions
   - Long entry
   - Short entry
   - Hold

2. Position Sizing
   - Dynamic based on:
     * ATR
     * Account balance
     * Risk parameters

3. Exit Actions
   - Take profit
   - Stop loss
   - Trail stop

### Reward Function
```python
reward = (pnl * risk_adjusted_factor) - (holding_cost + transaction_cost)
```
Where:
- pnl: Realized/unrealized profit/loss
- risk_adjusted_factor: Based on Sharpe ratio
- holding_cost: Time decay
- transaction_cost: Slippage and fees

## Best Practices

### Risk Management
1. Position Sizing
   - Maximum 2% risk per trade
   - ATR-based stop loss
   - Account balance scaling

2. Market Conditions
   - Volatility adjustments
   - Trend strength requirements
   - Volume confirmation

### Strategy Selection
1. Market Regime
   - Trending: Moving average strategies
   - Ranging: Mean reversion strategies
   - Volatile: Option-based strategies

2. Timeframe Alignment
   - Multiple timeframe confirmation
   - Trend alignment check
   - Volume profile support

## Performance Monitoring

### Metrics Tracked
1. Returns
   - Absolute returns
   - Risk-adjusted returns (Sharpe)
   - Maximum drawdown

2. Trade Statistics
   - Win rate
   - Profit factor
   - Average win/loss

3. Strategy Health
   - Indicator effectiveness
   - Signal quality
   - Cost efficiency

### Adaptation
The system continuously adapts by:
1. Updating indicator weights
2. Fine-tuning LLaMA model
3. Adjusting RL parameters
4. Optimizing execution