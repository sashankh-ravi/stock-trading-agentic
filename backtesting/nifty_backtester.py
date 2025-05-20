"""
Backtesting Framework for NIFTY 500 Strategy

Features:
1. Time period separation (training/validation/testing)
2. Transaction costs
3. Risk metrics calculation
4. Performance attribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import logging
from strategies.nifty500_strategy import Nifty500Strategy
from technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class BacktestResults:
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: pd.Series = None
        self.metrics: Dict = {}
        self.attribution: Dict = {}

class NiftyBacktester:
    def __init__(self,
                strategy: Nifty500Strategy,
                initial_capital: float = 1000000,
                transaction_cost: float = 0.001):  # 10 bps per trade
        
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.indicators = TechnicalIndicators()
        
    def run_backtest(self,
                   data: Dict[str, pd.DataFrame],
                   start_date: datetime,
                   end_date: datetime) -> BacktestResults:
        """
        Run backtest on multiple stocks
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Dictionary of OHLCV DataFrames for each stock
        start_date : datetime
            Start date for backtest
        end_date : datetime
            End date for backtest
        """
        results = BacktestResults()
        portfolio = self._initialize_portfolio()
        
        # Filter data for backtest period
        filtered_data = {
            symbol: df[(df.index >= start_date) & (df.index <= end_date)]
            for symbol, df in data.items()
        }
        
        # Main backtest loop
        dates = sorted(set.union(*[set(df.index) for df in filtered_data.values()]))
        for date in dates:
            try:
                self._process_trading_day(date, filtered_data, portfolio, results)
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                
        # Calculate final metrics
        self._calculate_performance_metrics(results)
        
        return results
        
    def _initialize_portfolio(self) -> Dict:
        """Initialize portfolio state"""
        return {
            'cash': self.initial_capital,
            'positions': {},  # symbol -> {quantity, entry_price, stop_loss, take_profit}
            'equity': self.initial_capital,
            'sector_exposure': {}  # sector -> exposure
        }
        
    def _process_trading_day(self,
                          date: datetime,
                          data: Dict[str, pd.DataFrame],
                          portfolio: Dict,
                          results: BacktestResults):
        """Process single trading day"""
        # Update positions
        self._update_positions(date, data, portfolio, results)
        
        # Generate new signals
        for symbol, df in data.items():
            if date not in df.index:
                continue
                
            # Skip if already have position
            if symbol in portfolio['positions']:
                continue
                
            # Get historical data up to current date
            hist_data = df[df.index <= date].copy()
            if len(hist_data) < 50:  # Minimum required history
                continue
                
            # Generate signal
            signal = self.strategy.generate_signal(hist_data)
            
            # Execute trades based on signal
            if signal.action in ['buy', 'sell']:
                self._execute_trade(date, symbol, signal, portfolio, hist_data, results)
                
        # Update equity curve
        results.equity_curve[date] = portfolio['equity']
        
    def _update_positions(self,
                       date: datetime,
                       data: Dict[str, pd.DataFrame],
                       portfolio: Dict,
                       results: BacktestResults):
        """Update existing positions"""
        for symbol in list(portfolio['positions'].keys()):
            if symbol not in data or date not in data[symbol].index:
                continue
                
            position = portfolio['positions'][symbol]
            current_price = data[symbol].loc[date, 'Close']
            
            # Check stop loss and take profit
            if position['side'] == 'buy':
                if current_price <= position['stop_loss']:
                    self._close_position(date, symbol, current_price, 'stop_loss', portfolio, results)
                elif current_price >= position['take_profit']:
                    self._close_position(date, symbol, current_price, 'take_profit', portfolio, results)
            else:  # sell position
                if current_price >= position['stop_loss']:
                    self._close_position(date, symbol, current_price, 'stop_loss', portfolio, results)
                elif current_price <= position['take_profit']:
                    self._close_position(date, symbol, current_price, 'take_profit', portfolio, results)
                    
    def _execute_trade(self,
                     date: datetime,
                     symbol: str,
                     signal: 'Signal',
                     portfolio: Dict,
                     data: pd.DataFrame,
                     results: BacktestResults):
        """Execute new trade"""
        current_price = data['Close'].iloc[-1]
        
        # Calculate position size
        position_value = portfolio['equity'] * signal.position_size
        quantity = position_value / current_price
        
        # Check if enough cash
        total_cost = position_value * (1 + self.transaction_cost)
        if total_cost > portfolio['cash']:
            quantity *= (portfolio['cash'] / total_cost)
            
        if quantity <= 0:
            return
            
        # Update portfolio
        portfolio['cash'] -= position_value * (1 + self.transaction_cost)
        portfolio['positions'][symbol] = {
            'quantity': quantity,
            'entry_price': current_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'side': signal.action,
            'entry_date': date
        }
        
        # Record trade
        results.trades.append({
            'date': date,
            'symbol': symbol,
            'side': signal.action,
            'quantity': quantity,
            'price': current_price,
            'value': position_value,
            'confidence': signal.confidence,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        })
        
    def _close_position(self,
                      date: datetime,
                      symbol: str,
                      current_price: float,
                      reason: str,
                      portfolio: Dict,
                      results: BacktestResults):
        """Close existing position"""
        position = portfolio['positions'][symbol]
        
        # Calculate P&L
        if position['side'] == 'buy':
            pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']
            
        # Update cash
        position_value = current_price * position['quantity']
        portfolio['cash'] += position_value * (1 - self.transaction_cost)
        
        # Record trade exit
        results.trades.append({
            'date': date,
            'symbol': symbol,
            'side': 'close_' + position['side'],
            'quantity': position['quantity'],
            'price': current_price,
            'value': position_value,
            'pnl': pnl,
            'pnl_pct': pnl / (position['entry_price'] * position['quantity']),
            'reason': reason
        })
        
        # Remove position
        del portfolio['positions'][symbol]
        
    def _calculate_performance_metrics(self, results: BacktestResults):
        """Calculate backtest performance metrics"""
        if not results.trades:
            return
            
        # Basic metrics
        trades_df = pd.DataFrame(results.trades)
        results.metrics['total_trades'] = len(trades_df)
        results.metrics['win_rate'] = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        results.metrics['avg_return'] = trades_df['pnl_pct'].mean()
        results.metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(results.equity_curve)
        results.metrics['max_drawdown'] = self._calculate_max_drawdown(results.equity_curve)
        
        # Risk metrics
        results.metrics['volatility'] = results.equity_curve.pct_change().std() * np.sqrt(252)
        results.metrics['sortino_ratio'] = self._calculate_sortino_ratio(results.equity_curve)
        results.metrics['calmar_ratio'] = self._calculate_calmar_ratio(
            results.equity_curve, results.metrics['max_drawdown']
        )
        
        # Performance attribution
        results.attribution = self._calculate_attribution(trades_df)
        
    def _calculate_sharpe_ratio(self, equity_curve: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        returns = equity_curve.pct_change().dropna()
        excess_returns = returns - risk_free_rate/252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        return abs(drawdowns.min())
        
    def _calculate_sortino_ratio(self, equity_curve: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate Sortino ratio"""
        returns = equity_curve.pct_change().dropna()
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
    def _calculate_calmar_ratio(self, equity_curve: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return 0
        annual_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252/len(equity_curve)) - 1
        return annual_return / max_drawdown
        
    def _calculate_attribution(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate performance attribution by different factors"""
        attribution = {}
        
        # Attribution by signal type
        attribution['signal_types'] = trades_df.groupby('side')['pnl'].agg(['sum', 'count']).to_dict()
        
        # Attribution by reason
        if 'reason' in trades_df.columns:
            attribution['exit_reasons'] = trades_df.groupby('reason')['pnl'].agg(['sum', 'count']).to_dict()
            
        return attribution
