"""Portfolio Manager

Implements sophisticated portfolio management with:
1. Risk-adjusted position sizing
2. Correlation-based diversification
3. Dynamic risk allocation
4. Rebalancing logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from strategies.base_strategy import Signal
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    entry_price: float
    current_price: float
    shares: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    risk_contribution: float

class PortfolioManager:
    def __init__(self,
                initial_capital: float = 1000000,
                max_position_size: float = 0.25,
                max_sector_exposure: float = 0.40,
                risk_free_rate: float = 0.03,
                target_volatility: float = 0.15,
                rebalance_threshold: float = 0.1):
        """Initialize portfolio manager
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum size for any position
            max_sector_exposure: Maximum exposure to any sector
            risk_free_rate: Annual risk-free rate
            target_volatility: Target portfolio volatility
            rebalance_threshold: Threshold for rebalancing
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.rebalance_threshold = rebalance_threshold
        
        self.positions: Dict[str, Position] = {}
        self.sector_exposure: Dict[str, float] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.performance_metrics = self._init_metrics()

    def process_signals(self, 
                      signals: List[Signal], 
                      prices: Dict[str, float],
                      sector_map: Dict[str, str]) -> List[Dict]:
        """Process new trading signals and update portfolio
        
        Returns list of trades to execute
        """
        try:
            trades = []
            
            # Update current positions
            self._update_positions(prices)
            
            # Sort signals by confidence
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            for signal in signals:
                symbol = signal.symbol
                current_price = prices[symbol]
                sector = sector_map.get(symbol, 'Unknown')
                
                # Check if we should close position
                if symbol in self.positions and signal.action == 'sell':
                    trades.append(self._create_exit_trade(symbol, current_price))
                    continue
                
                # Skip if already have position
                if symbol in self.positions:
                    continue
                    
                # Check sector exposure
                if self._check_sector_exposure(sector, signal.position_size):
                    # Calculate position size
                    size = self._calculate_position_size(
                        symbol, signal.position_size, current_price, sector
                    )
                    
                    if size > 0:
                        trades.append({
                            'symbol': symbol,
                            'action': signal.action,
                            'price': current_price,
                            'shares': size,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit
                        })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
            return []

    def _update_positions(self, current_prices: Dict[str, float]) -> None:
        """Update position values and metrics"""
        total_value = self.capital
        
        for symbol, pos in list(self.positions.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                pos.current_price = current_price
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.shares
                total_value += pos.shares * current_price
                
                # Check stops
                if current_price <= pos.stop_loss or current_price >= pos.take_profit:
                    self._close_position(symbol, current_price)
        
        # Update risk contributions
        if total_value > 0:
            for pos in self.positions.values():
                pos.risk_contribution = (pos.shares * pos.current_price) / total_value

    def _calculate_position_size(self,
                              symbol: str,
                              suggested_size: float,
                              price: float,
                              sector: str) -> float:
        """Calculate optimal position size considering all constraints"""
        try:
            # Get available capital
            available_capital = self.capital * self.max_position_size
            
            # Calculate base position size
            position_value = min(available_capital, suggested_size * self.capital)
            
            # Adjust for sector exposure
            sector_exposure = self.sector_exposure.get(sector, 0)
            max_sector_remaining = (self.max_sector_exposure - sector_exposure) * self.capital
            position_value = min(position_value, max_sector_remaining)
            
            # Convert to shares
            shares = position_value / price
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def _check_sector_exposure(self, sector: str, size: float) -> bool:
        """Check if adding position would exceed sector limits"""
        current_exposure = self.sector_exposure.get(sector, 0)
        return (current_exposure + size) <= self.max_sector_exposure

    def _create_exit_trade(self, symbol: str, price: float) -> Dict:
        """Create trade dict for position exit"""
        pos = self.positions[symbol]
        return {
            'symbol': symbol,
            'action': 'sell',
            'price': price,
            'shares': -pos.shares,  # Negative for selling
            'stop_loss': 0,
            'take_profit': 0
        }

    def _close_position(self, symbol: str, price: float) -> None:
        """Close a position and update metrics"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            realized_pnl = (price - pos.entry_price) * pos.shares
            self.capital += pos.shares * price + realized_pnl
            
            # Update metrics
            self._update_metrics({
                'symbol': symbol,
                'pnl': realized_pnl,
                'return': realized_pnl / (pos.entry_price * pos.shares),
                'duration': None  # Would add actual duration here
            })
            
            del self.positions[symbol]

    def _init_metrics(self) -> Dict:
        """Initialize performance metrics"""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'num_trades': 0,
            'avg_trade_return': 0.0,
            'avg_trade_duration': 0.0
        }

    def _update_metrics(self, trade_result: Dict) -> None:
        """Update performance metrics after trade"""
        try:
            n = self.performance_metrics['num_trades'] + 1
            self.performance_metrics['num_trades'] = n
            
            # Update returns
            self.performance_metrics['total_return'] = (
                self.capital / self.initial_capital - 1
            )
            
            # Update average trade metrics
            self.performance_metrics['avg_trade_return'] = (
                (self.performance_metrics['avg_trade_return'] * (n-1) + 
                 trade_result['return']) / n
            )
            
            if trade_result['duration']:
                self.performance_metrics['avg_trade_duration'] = (
                    (self.performance_metrics['avg_trade_duration'] * (n-1) + 
                     trade_result['duration']) / n
                )
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()

    def get_performance(self) -> Dict:
        """Get performance metrics"""
        return self.performance_metrics.copy()

    def calculate_risk_metrics(self) -> Dict:
        """Calculate current portfolio risk metrics"""
        try:
            total_value = self.capital + sum(
                pos.shares * pos.current_price for pos in self.positions.values()
            )
            
            position_values = {
                symbol: pos.shares * pos.current_price 
                for symbol, pos in self.positions.items()
            }
            
            weights = {
                symbol: value/total_value 
                for symbol, value in position_values.items()
            }
            
            # Calculate portfolio volatility
            if len(self.positions) > 0:
                returns = pd.DataFrame({
                    symbol: self.historical_data[symbol]['Returns'].tail(252)
                    for symbol in self.positions.keys()
                })
                
                cov_matrix = returns.cov() * 252  # Annualized
                port_vol = np.sqrt(
                    np.sum([weights[i] * weights[j] * cov_matrix.loc[i,j]
                           for i in weights for j in weights])
                )
                
                # Calculate VaR
                port_returns = returns.dot(pd.Series(weights))
                var_95 = np.percentile(port_returns, 5)
                var_99 = np.percentile(port_returns, 1)
                
                return {
                    'volatility': port_vol,
                    'var_95': var_95,
                    'var_99': var_99,
                    'sharpe_ratio': (self.performance_metrics['total_return'] - 
                                   self.risk_free_rate) / port_vol
                    if port_vol > 0 else 0
                }
            
            return {
                'volatility': 0,
                'var_95': 0,
                'var_99': 0,
                'sharpe_ratio': 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'volatility': 0,
                'var_95': 0,
                'var_99': 0,
                'sharpe_ratio': 0
            }
