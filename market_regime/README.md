# Market Regime Analysis Package

This package provides a comprehensive market regime analysis framework, structured in a modular way for better maintainability and extensibility.

## Implementation

The implementation is now organized as a package with the following files:
- `__init__.py`: Core components and classes for market regime analysis
- `trend.py`: Trend analysis functions
- `volatility.py`: Volatility analysis functions
- `volume.py`: Volume analysis functions  
- `momentum.py`: Momentum analysis functions

The original `market_regime.py` in the root directory is maintained for backward compatibility while the transition is completed. Future development should focus on this package structure.

## Features

- Multi-factor regime detection (trend, volatility, volume, momentum)
- Regime transition detection
- Regime-specific parameter adjustments
- Performance analytics by regime type
- Confidence scoring for regime classifications

## How to Use

```python
from market_regime import MarketRegimeAnalyzer

# Initialize analyzer
analyzer = MarketRegimeAnalyzer()

# Detect current regime
current_regime = analyzer.analyze(market_data)

# Get optimal parameters for current regime
params = analyzer.get_regime_parameters(current_regime)
```

## Migration Status

This package is a modular refactoring of the original `market_regime.py` file. The original file is maintained for backward compatibility until the migration is complete. 

## Testing

Unit tests for this package can be run with:

```bash
python -m unittest test_market_regime_package.py
```
