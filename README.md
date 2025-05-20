# Nifty500 Trading System

An advanced trading system that combines technical analysis, reinforcement learning, and financial reasoning for algorithmic trading on the Nifty 500 index.

## Features

- **Technical Analysis Framework**: Over 50 implemented technical indicators with correlation analysis
- **Reinforcement Learning Integration**: Strategy optimization using RL algorithms
- **Advanced Backtesting Engine**: Comprehensive backtesting with realistic market simulation
- **Risk Management System**: Position sizing, stop-loss, and drawdown management
- **Portfolio Management**: Multi-asset optimization and rebalancing

## Project Structure

- **`app.py`**: Main application entry point
- **`strategies/`**: Trading strategy implementations
  - Base strategy framework
  - Momentum, trend, and pattern-based strategies
  - Strategy combination logic
- **`backtesting/`**: Backtesting framework
  - Historical data simulation
  - Performance metrics
  - Test framework
- **`technical_indicators.py`**: Custom technical indicator implementations
- **`risk_manager.py`**: Risk management tools and calculations
- **`portfolio_manager.py`**: Portfolio optimization and management
- **`docs/`**: Comprehensive documentation
  - Detailed thesis on system design and implementation
  - Indicator explanations and pattern guides

## Getting Started

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (required for technical indicators)
# Instructions vary by platform - see https://mrjbq7.github.io/ta-lib/install.html
```

### Running the System

```bash
# Run the main application
python app.py

# Run backtests
python backtesting/nifty_backtester.py

# Train reinforcement learning model
python train.py
```

## Documentation

The complete documentation for this system is available in:
- **`docs/nifty500_trading_system_thesis.md`**: Comprehensive system design and implementation details
- **`docs/Nifty500_Trading_System_Complete.pdf`**: PDF version with visualizations and references

## License

This project is licensed under the MIT License - see the LICENSE file for details.
