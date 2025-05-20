"""Backtesting Engine

A robust backtesting framework for evaluating trading strategies.
Features:
- Multiple strategy evaluation
- Strategy combination testing
- Realistic transaction costs
- Risk management
- Performance metrics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass
from .base_strategy import BaseStrategy, Signal
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a completed or open trade"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    stop_loss: float
    take_profit: float
    status: str  # 'open', 'closed', 'stopped', 'target_reached'
    metrics: Dict

@dataclass
class BacktestResults:
    """Contains all backtest results and metrics"""
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict
    signals: List[Signal]
    
class Backtester:
    def __init__(self,
                initial_capital: float = 1000000,
                transaction_cost: float = 0.001,  # 0.1% per trade
                max_positions: int = 5,
                max_position_size: float = 0.25):  # 25% max per position
        """Initialize backtester with parameters"""
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.reset()
        
    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions = {}  # symbol -> Trade
        self.trades = []
        self.equity_curve = []
        self.signals = []
        
    def run(self, 
           data: Dict[str, pd.DataFrame],
           strategies: Union[BaseStrategy, List[BaseStrategy]],
           weights: List[float] = None) -> BacktestResults:
        """Run backtest with given data and strategies"""
        if not isinstance(strategies, list):
            strategies = [strategies]
            
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
            
        self.reset()
        dates = self._get_common_dates(data)
        
        # Run through each date
        for date in dates:
            # Update positions first
            self._update_positions(data, date)
            
            # Generate and combine signals
            signals = self._generate_signals(data, date, strategies)
            combined_signals = self._combine_signals(signals, weights)
            
            # Execute signals
            self._execute_signals(combined_signals)
            
            # Record equity
            self.equity_curve.append(self._calculate_equity())
            
        return self._generate_results()
    
    def _get_common_dates(self, data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Get common dates across all data"""
        dates = None
        for df in data.values():
            if dates is None:
                dates = df.index
            else:
                dates = dates.intersection(df.index)
        return dates
    
    def _update_positions(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp):
        """Update open positions and check stops/targets"""
        for symbol, trade in list(self.positions.items()):
            if symbol not in data:
                continue
                
            current_price = data[symbol].loc[date, 'Close']
            
            # Check stop loss
            if trade.stop_loss > 0:
                if (trade.entry_price > trade.stop_loss and current_price <= trade.stop_loss) or \
                   (trade.entry_price < trade.stop_loss and current_price >= trade.stop_loss):
                    self._close_position(symbol, current_price, date, 'stopped')
                    continue
            
            # Check take profit
            if trade.take_profit > 0:
                if (trade.entry_price < trade.take_profit and current_price >= trade.take_profit) or \
                   (trade.entry_price > trade.take_profit and current_price <= trade.take_profit):
                    self._close_position(symbol, current_price, date, 'target_reached')
                    continue
    
    def _generate_signals(self, 
                         data: Dict[str, pd.DataFrame], 
                         date: pd.Timestamp,
                         strategies: List[BaseStrategy]) -> List[List[Signal]]:
        """Generate signals for each strategy"""
        all_signals = []
        for strategy in strategies:
            signals = []
            for symbol, df in data.items():
                historical = df[df.index <= date].copy()
                if len(historical) > strategy.lookback:
                    signal = strategy.generate_signal(historical)
                    if signal.action != 'hold':
                        signals.append(signal)
            all_signals.append(signals)
        return all_signals
    
    def _combine_signals(self, 
                        strategy_signals: List[List[Signal]], 
                        weights: List[float]) -> List[Signal]:
        """Combine signals from multiple strategies"""
        combined = {}
        for signals, weight in zip(strategy_signals, weights):
            for signal in signals:
                if signal.symbol not in combined:
                    combined[signal.symbol] = signal
                    combined[signal.symbol].confidence *= weight
                else:
                    # Average signals weighted by confidence
                    total_conf = combined[signal.symbol].confidence + signal.confidence * weight
                    if total_conf > 0:
                        for attr in ['price', 'stop_loss', 'take_profit', 'position_size']:
                            current = getattr(combined[signal.symbol], attr)
                            new = getattr(signal, attr)
                            weighted = (current * combined[signal.symbol].confidence + 
                                      new * signal.confidence * weight) / total_conf
                            setattr(combined[signal.symbol], attr, weighted)
                    combined[signal.symbol].confidence = total_conf
                    
        return list(combined.values())
    
    def _execute_signals(self, signals: List[Signal]):
        """Execute trading signals"""
        # Sort signals by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        for signal in signals:
            # Skip if max positions reached
            if len(self.positions) >= self.max_positions:
                break
                
            # Skip if already have position in this symbol
            if signal.symbol in self.positions:
                continue
            
            # Calculate position size
            available_capital = self.capital * self.max_position_size
            position_value = min(available_capital, 
                               signal.position_size * self.capital)
            
            # Open position
            if position_value > 0:
                trade = Trade(
                    entry_time=signal.timestamp,
                    exit_time=pd.NaT,
                    symbol=signal.symbol,
                    entry_price=signal.price,
                    exit_price=0.0,
                    position_size=position_value / signal.price,
                    pnl=0.0,
                    pnl_pct=0.0,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    status='open',
                    metrics=signal.metrics
                )
                self.positions[signal.symbol] = trade
                self.capital -= position_value * (1 + self.transaction_cost)
    
    def _close_position(self, 
                       symbol: str, 
                       price: float, 
                       date: pd.Timestamp,
                       status: str):
        """Close a position and record the trade"""
        trade = self.positions[symbol]
        trade.exit_time = date
        trade.exit_price = price
        trade.status = status
        
        # Calculate P&L
        position_value = trade.position_size * price
        entry_value = trade.position_size * trade.entry_price
        trade.pnl = position_value - entry_value - \
                    (position_value + entry_value) * self.transaction_cost
        trade.pnl_pct = trade.pnl / entry_value
        
        # Update capital
        self.capital += position_value * (1 - self.transaction_cost)
        
        # Record trade
        self.trades.append(trade)
        del self.positions[symbol]
    
    def _calculate_equity(self) -> float:
        """Calculate current equity including open positions"""
        equity = self.capital
        for trade in self.positions.values():
            position_value = trade.position_size * trade.entry_price
            equity += position_value
        return equity
    
    def _generate_results(self) -> BacktestResults:
        """Generate final backtest results and metrics"""
        equity_curve = pd.Series(self.equity_curve)
        returns = equity_curve.pct_change().dropna()
        
        # Close any remaining positions
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, self.positions[symbol].entry_price,
                               equity_curve.index[-1], 'closed')
        
        metrics = {
            'total_return': (equity_curve.iloc[-1] / self.initial_capital - 1),
            'max_drawdown': (equity_curve / equity_curve.cummax() - 1).min(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'win_rate': len([t for t in self.trades if t.pnl > 0]) / len(self.trades) if self.trades else 0,
            'avg_trade': np.mean([t.pnl for t in self.trades]) if self.trades else 0,
            'num_trades': len(self.trades)
        }
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=equity_curve,
            metrics=metrics,
            signals=self.signals
        )
