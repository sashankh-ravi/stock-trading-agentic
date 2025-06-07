# Relative Strength Module Documentation

## Overview

The relative strength module provides tools for calculating and analyzing relative strength metrics between stocks and market indices. It implements multiple approaches to relative strength calculation and provides comprehensive error handling and data validation.

## Key Features

1. **Data Alignment**
   - Automatic date alignment between stock and market data
   - Handling of missing values and different time periods
   - Validation of data quality and sufficiency

2. **Multiple Calculation Methods**
   - Cumulative returns-based relative strength
   - Rolling window-based relative strength
   - Momentum-enhanced relative strength metrics

3. **Comprehensive Metrics**
   - RS_CUM: Cumulative relative strength
   - RS_ROLL: Rolling relative strength
   - RS_MOM: Relative strength momentum

## Usage Examples

### Basic Usage

```python
from relative_strength import calculate_relative_strength_metrics

# Calculate RS metrics
rs_metrics = calculate_relative_strength_metrics(
    stock_data=stock_df['Close'],
    market_data=market_df,
    window=90
)

print(rs_metrics)
```

### Advanced Usage

```python
from relative_strength import (
    calculate_relative_strength_cumulative,
    calculate_relative_strength_rolling
)

# Calculate cumulative RS
rs_cum = calculate_relative_strength_cumulative(
    stock_data,
    market_data,
    min_periods=20
)

# Calculate rolling RS
rs_roll = calculate_relative_strength_rolling(
    stock_data,
    market_data,
    window=90,
    min_periods=20
)
```

## Implementation Details

### Data Standardization

The module automatically handles:
- Date alignment between stock and market data
- Missing value removal
- Data validation and quality checks

### Error Handling

Comprehensive error handling for:
- Insufficient data points
- Missing values
- Division by zero
- Extreme values
- Data alignment issues

### Performance Considerations

- Efficient calculations using vectorized operations
- Minimal memory footprint
- Optimized for large datasets

## Function Reference

### `standardize_periods`

Aligns stock and market data to ensure consistent dates.

```python
def standardize_periods(
    stock_data: pd.Series,
    market_data: pd.Series
) -> Tuple[pd.Series, pd.Series]
```

### `calculate_relative_strength_cumulative`

Calculates relative strength using cumulative returns.

```python
def calculate_relative_strength_cumulative(
    stock_data: pd.Series,
    market_data: pd.Series,
    min_periods: int = 20
) -> float
```

### `calculate_relative_strength_rolling`

Calculates rolling relative strength.

```python
def calculate_relative_strength_rolling(
    stock_data: pd.Series,
    market_data: pd.Series,
    window: int = 90,
    min_periods: int = 20
) -> pd.Series
```

### `calculate_relative_strength_metrics`

Calculates comprehensive relative strength metrics.

```python
def calculate_relative_strength_metrics(
    stock_data: pd.Series,
    market_data: pd.DataFrame,
    window: int = 90,
    min_periods: int = 20
) -> Dict[str, float]
```

## Testing

The module includes comprehensive unit tests covering:
- Basic functionality
- Edge cases
- Error handling
- Data validation

Run tests using:
```bash
python -m unittest tests/test_relative_strength.py
```

## Known Limitations

1. Extreme Market Conditions
   - May not accurately reflect relative strength during extreme market volatility
   - Clipping is applied to handle extreme values

2. Data Requirements
   - Minimum periods required for reliable calculations
   - Both stock and market data must have sufficient overlap

3. Performance
   - Large rolling windows may impact performance with very large datasets
   - Memory usage increases with window size

## Future Improvements

1. Additional Metrics
   - Sector-relative strength
   - Risk-adjusted relative strength
   - Volume-weighted relative strength

2. Performance Optimizations
   - Parallel processing for multiple stocks
   - Caching of intermediate calculations
   - Optimized rolling window calculations

3. Enhanced Analytics
   - Regime detection
   - Adaptive window sizes
   - Machine learning integration
