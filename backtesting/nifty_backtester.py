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
from strategies.base_strategy import Signal
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
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        results = BacktestResults()
        portfolio = self._initialize_portfolio()
        
        # Initialize equity curve
        dates = sorted(set.union(*[set(df.index) for df in data.values()]))
        results.equity_curve = pd.Series(index=dates, data=self.initial_capital)
        
        # Filter data for backtest period
        filtered_data = {
            symbol: df[(df.index.tz_localize(None) >= pd.Timestamp(start_date).tz_localize(None)) & 
                      (df.index.tz_localize(None) <= pd.Timestamp(end_date).tz_localize(None))]
            for symbol, df in data.items()
        }
        
        logger.info(f"Loaded data for {len(filtered_data)} symbols")
        
        # Main backtest loop
        dates = sorted(set.union(*[set(df.index) for df in filtered_data.values()]))
        total_dates = len(dates)
        processed_dates = 0
        
        for date in dates:
            try:
                self._process_trading_day(date, filtered_data, portfolio, results)
                processed_dates += 1
                if processed_dates % 100 == 0:
                    logger.debug(f"Processed {processed_dates}/{total_dates} trading days...")
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                continue
                
        # Calculate final metrics
        self._calculate_performance_metrics(results)
        
        # Log summary
        final_equity = results.equity_curve.iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100
        logger.info(f"Backtest completed. Total return: {total_return:.1f}%")
        logger.info(f"Final equity: {final_equity:,.0f}")
        logger.info(f"Total trades: {len(results.trades)}")
        
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
            hist_data = df[df.index.tz_localize(None) <= pd.Timestamp(date).tz_localize(None)].copy()
            if len(hist_data) < 50:  # Minimum required history
                continue
                
            # Generate signal
            signal = self.strategy.generate_signal(hist_data)
            signal.symbol = symbol  # Ensure symbol is set
            
            # Execute trades based on signal
            if signal.action in ['buy', 'sell']:
                self._execute_trade(date, symbol, signal, portfolio, hist_data, results)
        
        # Calculate total portfolio value (cash + positions)
        self._update_portfolio_equity(date, data, portfolio)
                
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
        
        # Detect market regime
        market_regime = self._detect_market_regime(data)
        
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
            'take_profit': signal.take_profit,
            'entry_date': date,
            'pnl': 0,  # Initialize PnL
            'pnl_pct': 0,  # Initialize PnL percentage
            'market_regime': market_regime  # Add market regime
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
        
        # Get market regime for the current position if available in data
        market_regime = 'unknown'
        if symbol in results.trades:
            # Find the most recent entry for this symbol
            for trade in reversed(results.trades):
                if trade.get('symbol') == symbol and trade.get('side') in ['buy', 'sell']:
                    market_regime = trade.get('market_regime', 'unknown')
                    break
        
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
            'entry_date': position['entry_date'],
            'exit_date': date,
            'reason': reason,
            'market_regime': market_regime
        })
        
        # Remove position
        del portfolio['positions'][symbol]
        
    def _calculate_performance_metrics(self, results: BacktestResults):
        """Calculate backtest performance metrics"""
        if not results.trades:
            results.metrics = {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0
            }
            return
            
        # Basic metrics
        trades_df = pd.DataFrame(results.trades)
        
        # Ensure pnl exists and is numeric
        trades_df['pnl'] = trades_df['pnl'].fillna(0).astype(float)
        trades_df['pnl_pct'] = trades_df['pnl_pct'].fillna(0).astype(float)
        
        # Add entry/exit dates for holding period calculation
        trades_df['entry_date'] = pd.to_datetime(trades_df['date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['date'])
        
        # Calculate equity curve returns
        returns = results.equity_curve.pct_change().fillna(0)
        
        # Base metrics
        results.metrics = {
            'total_trades': len(trades_df),
            'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df),
            'avg_return': trades_df['pnl_pct'].mean(),
            'sharpe_ratio': self._calculate_sharpe_ratio(results.equity_curve),
            'max_drawdown': self._calculate_max_drawdown(results.equity_curve),
            'volatility': returns.std() * np.sqrt(252),
            'sortino_ratio': self._calculate_sortino_ratio(results.equity_curve),
            'calmar_ratio': self._calculate_calmar_ratio(
                results.equity_curve, 
                self._calculate_max_drawdown(results.equity_curve)
            )
        }
        
        # Add extended metrics
        results.metrics.update(self._calculate_extended_metrics(results.equity_curve, trades_df))
        
        # Add market regime metrics
        results.metrics['regime_analysis'] = self._calculate_regime_metrics(returns)
        
        # Performance attribution
        results.attribution = self._calculate_attribution(trades_df)
        
        # Log key metrics
        logger.info(f"Backtest Results:")
        logger.info(f"Total Return: {(results.equity_curve.iloc[-1]/results.equity_curve.iloc[0] - 1)*100:.2f}%")
        logger.info(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
        logger.info(f"Win Rate: {results.metrics['win_rate']*100:.2f}%")
        logger.info(f"Profit Factor: {results.metrics['profit_factor']:.2f}")
        logger.info(f"Max Drawdown: {results.metrics['max_drawdown']*100:.2f}%")
        
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
    
    def _calculate_regime_metrics(self, returns: pd.Series) -> Dict[str, Dict]:
        """Calculate market regime-based performance metrics"""
        # Check if returns is empty
        if len(returns) == 0:
            return {
                'bull': {'return': 0, 'sharpe': 0, 'max_drawdown': 0, 'win_rate': 0},
                'neutral': {'return': 0, 'sharpe': 0, 'max_drawdown': 0, 'win_rate': 0},
                'bear': {'return': 0, 'sharpe': 0, 'max_drawdown': 0, 'win_rate': 0},
                'low_vol': {'return': 0, 'sharpe': 0, 'win_rate': 0},
                'normal_vol': {'return': 0, 'sharpe': 0, 'win_rate': 0},
                'high_vol': {'return': 0, 'sharpe': 0, 'win_rate': 0},
                'regime_stability': 0,
                'vol_regime_stability': 0
            }
            
        trend_regimes = pd.Series(index=returns.index)
        vol_regimes = pd.Series(index=returns.index)
        regime_metrics = {}
        
        # Calculate trend regimes
        sma = returns.rolling(50).mean()
        std = returns.rolling(50).std()
        
        # Initialize with empty strings to avoid type conversion issues
        trend_regimes = pd.Series(index=returns.index, dtype='object')
        trend_regimes.loc[returns > sma + 0.5*std] = 'bull'
        trend_regimes.loc[returns < sma - 0.5*std] = 'bear'
        trend_regimes.loc[trend_regimes.isnull()] = 'neutral'
        
        # Calculate volatility regimes
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        
        # Initialize with empty strings to avoid type conversion issues
        vol_regimes = pd.Series(index=returns.index, dtype='object')
        vol_regimes.loc[rolling_vol > vol_mean + vol_std] = 'high_vol'
        vol_regimes.loc[rolling_vol < vol_mean - vol_std] = 'low_vol'
        vol_regimes.loc[vol_regimes.isnull()] = 'normal_vol'
        
        # Calculate regime-specific metrics
        for regime in ['bull', 'neutral', 'bear']:
            regime_returns = returns[trend_regimes == regime]
            if len(regime_returns) > 0:
                regime_metrics[regime] = {
                    'return': regime_returns.mean() * 252,
                    'sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)) \
                        if regime_returns.std() != 0 else 0,
                    'max_drawdown': ((1 + regime_returns).cumprod().div(
                        (1 + regime_returns).cumprod().cummax()).min() - 1) \
                        if len(regime_returns) > 1 else 0,
                    'win_rate': (regime_returns > 0).mean()
                }
        
        # Volatility regime metrics
        for regime in ['low_vol', 'normal_vol', 'high_vol']:
            regime_returns = returns[vol_regimes == regime]
            if len(regime_returns) > 0:
                regime_metrics[regime] = {
                    'return': regime_returns.mean() * 252,
                    'sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)) \
                        if regime_returns.std() != 0 else 0,
                    'win_rate': (regime_returns > 0).mean()
                }
        
        # Regime stability and transitions
        regime_metrics['regime_stability'] = len(trend_regimes[trend_regimes != trend_regimes.shift(1)]) / len(trend_regimes)
        regime_metrics['vol_regime_stability'] = len(vol_regimes[vol_regimes != vol_regimes.shift(1)]) / len(vol_regimes)
        
        return regime_metrics
        
    def _calculate_extended_metrics(self, equity_curve: pd.Series, trades_df: pd.DataFrame) -> Dict:
        """Calculate extended performance metrics"""
        returns = equity_curve.pct_change().dropna()
        metrics = {}
        
        # Risk metrics
        if len(returns) > 0:
            metrics.update({
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1),
                'cvar_95': returns[returns <= np.percentile(returns, 5)].mean() if len(returns[returns <= np.percentile(returns, 5)]) > 0 else 0,
                'cvar_99': returns[returns <= np.percentile(returns, 1)].mean() if len(returns[returns <= np.percentile(returns, 1)]) > 0 else 0,
                'skewness': returns.skew() if len(returns) > 2 else 0,
                'kurtosis': returns.kurtosis() if len(returns) > 3 else 0,
                'avg_drawdown': self._calculate_average_drawdown(equity_curve),
                'drawdown_duration': self._calculate_drawdown_duration(equity_curve)
            })
        else:
            # Default values if no returns
            metrics.update({
                'var_95': 0,
                'var_99': 0,
                'cvar_95': 0,
                'cvar_99': 0,
                'skewness': 0,
                'kurtosis': 0,
                'avg_drawdown': 0,
                'drawdown_duration': 0
            })
        
        # Trade metrics
        if len(trades_df) > 0:
            profitable_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            metrics.update({
                'avg_profit': profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'largest_profit': profitable_trades['pnl'].max() if len(profitable_trades) > 0 else 0,
                'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
                'profit_factor': abs(profitable_trades['pnl'].sum() / losing_trades['pnl'].sum()) \
                    if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf'),
                'avg_holding_period': (trades_df['exit_date'] - trades_df['entry_date']).mean().days \
                    if 'exit_date' in trades_df.columns and 'entry_date' in trades_df.columns else 0
            })
            
        return metrics
        
    def _calculate_average_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate average drawdown"""
        drawdown_series = equity_curve / equity_curve.cummax() - 1
        drawdowns = []
        current_drawdown = 0
        
        for dd in drawdown_series:
            if dd < 0:
                current_drawdown = min(current_drawdown, dd)
            elif current_drawdown < 0:
                drawdowns.append(abs(current_drawdown))
                current_drawdown = 0
                
        return np.mean(drawdowns) if drawdowns else 0
        
    def _calculate_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Calculate average drawdown duration in days"""
        drawdown_series = equity_curve / equity_curve.cummax() - 1
        durations = []
        current_duration = 0
        in_drawdown = False
        
        for dd in drawdown_series:
            if dd < 0:
                current_duration += 1
                in_drawdown = True
            elif in_drawdown:
                durations.append(current_duration)
                current_duration = 0
                in_drawdown = False
                
        return int(np.mean(durations)) if durations else 0
    
    def _calculate_attribution(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate performance attribution by different factors"""
        attribution = {}
        
        # Attribution by signal type
        signal_stats = trades_df.groupby('side').agg({
            'pnl': ['sum', 'mean', 'count', 'std'],
            'pnl_pct': ['mean', 'std']
        })
        attribution['signal_types'] = signal_stats.to_dict()
        
        # Attribution by exit reason
        if 'reason' in trades_df.columns:
            reason_stats = trades_df.groupby('reason').agg({
                'pnl': ['sum', 'mean', 'count', 'std'],
                'pnl_pct': ['mean', 'std']
            })
            attribution['exit_reasons'] = reason_stats.to_dict()
            
        # Attribution by holding period
        if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
            trades_df['holding_period'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
            
            # First check if we have any trades with valid holding periods
            if trades_df.empty:
                logger.warning("No trades found in trades data for holding period analysis")
                attribution['holding_periods'] = {}
                return attribution
            elif 'holding_period' not in trades_df.columns:
                logger.warning("No holding_period column found in trades data")
                attribution['holding_periods'] = {}
                return attribution
            elif trades_df['holding_period'].isna().all():
                logger.warning("No valid holding periods found in trades data - all values are NaN")
                attribution['holding_periods'] = {}
                return attribution
            
            try:
                # Remove NaN values before bucketing
                trades_with_periods = trades_df.dropna(subset=['holding_period'])
                
                if trades_with_periods.empty:
                    logger.warning("All holding periods are NaN")
                    attribution['holding_periods'] = {}
                    return attribution
                
                # Check if we have enough unique values for quantile-based bucketing
                unique_periods = trades_with_periods['holding_period'].unique()
                n_unique = len(unique_periods)
                
                if n_unique >= 4:
                    # Enough unique values for 4 quantiles
                    try:
                        # Apply qcut to the filtered DataFrame first
                        trades_with_periods['holding_bucket'] = pd.qcut(trades_with_periods['holding_period'], 
                                                        q=4, 
                                                        labels=['very_short', 'short', 'medium', 'long'],
                                                        duplicates='drop')
                        # Merge the bucketing back to the original DataFrame
                        trades_df = trades_df.merge(trades_with_periods[['holding_period', 'holding_bucket']], 
                                                on='holding_period', how='left')
                    except ValueError as e:
                        # Handle case where qcut fails even with duplicates='drop'
                        logger.warning(f"Error with 4 quantiles: {e}, trying fewer bins")
                        n_unique = 3  # Force to try with 3 bins
                
                if 1 < n_unique < 4:
                    # Some unique values but not enough for 4 buckets
                    n_bins = min(n_unique, 3)
                    labels = ['short', 'medium', 'long'][:n_bins]
                    try:
                        # Apply qcut to the filtered DataFrame first
                        trades_with_periods['holding_bucket'] = pd.qcut(trades_with_periods['holding_period'],
                                                          q=n_bins,
                                                          labels=labels,
                                                          duplicates='drop')
                        # Merge the bucketing back to the original DataFrame
                        trades_df = trades_df.merge(trades_with_periods[['holding_period', 'holding_bucket']], 
                                                on='holding_period', how='left')
                    except ValueError as e:
                        logger.warning(f"Error with {n_bins} quantiles: {e}, falling back to manual buckets")
                        raise ValueError("Falling back to manual categorization")
                
                elif n_unique == 1:
                    # Only one unique value
                    trades_with_periods['holding_bucket'] = 'medium'
                    # Merge the bucketing back to the original DataFrame
                    trades_df = trades_df.merge(trades_with_periods[['holding_period', 'holding_bucket']], 
                                            on='holding_period', how='left')
                    
            except Exception as e:
                logger.warning(f"Error creating quantile-based holding period buckets: {e}")
                
                # Fall back to a predefined categorization
                def categorize_holding(days):
                    if pd.isna(days):
                        return None
                    if days <= 5:
                        return 'very_short'
                    elif days <= 20:
                        return 'short'
                    elif days <= 60:
                        return 'medium'
                    else:
                        return 'long'
                
                trades_df['holding_bucket'] = trades_df['holding_period'].apply(categorize_holding)
            
            # Filter out any NaN values from the groupby
            valid_buckets = trades_df.dropna(subset=['holding_bucket', 'pnl', 'pnl_pct'])
            
            # Check if we have any valid buckets before grouping
            if not valid_buckets.empty:
                # Make sure we have all required columns before grouping
                required_cols = ['holding_bucket', 'pnl', 'pnl_pct', 'holding_period']
                if all(col in valid_buckets.columns for col in required_cols):
                    try:
                        # Check if all buckets have enough entries to calculate stats
                        group_sizes = valid_buckets.groupby('holding_bucket').size()
                        valid_groups = group_sizes[group_sizes >= 2].index.tolist()
                        
                        if valid_groups:
                            # Filter to only include groups with enough data for stats
                            filtered_buckets = valid_buckets[valid_buckets['holding_bucket'].isin(valid_groups)]
                            
                            # Calculate statistics only for valid groups
                            agg_dict = {
                                'pnl': ['sum', 'mean', 'count'],
                                'pnl_pct': ['mean'],
                                'holding_period': 'mean'
                            }
                            
                            # Only include std if we have enough samples
                            for group in valid_groups:
                                group_data = valid_buckets[valid_buckets['holding_bucket'] == group]
                                if len(group_data) >= 3:  # Need at least 3 data points for meaningful std
                                    if 'pnl' not in agg_dict:
                                        agg_dict['pnl'] = []
                                    if 'std' not in agg_dict['pnl']:
                                        agg_dict['pnl'].append('std')
                                        
                                    if 'pnl_pct' not in agg_dict:
                                        agg_dict['pnl_pct'] = []
                                    if 'std' not in agg_dict['pnl_pct']:
                                        agg_dict['pnl_pct'].append('std')
                            
                            # Log the aggregation keys for debugging
                            logger.debug(f"Holding period aggregation dict: {agg_dict}")
                            
                            try:
                                period_stats = filtered_buckets.groupby('holding_bucket').agg(agg_dict)
                                attribution['holding_periods'] = period_stats.to_dict()
                            except Exception as e:
                                logger.error(f"Error in holding period aggregation: {e}")
                                # Fallback to simple aggregation without multi-level columns
                                try:
                                    simple_agg = {
                                        'pnl': 'mean',
                                        'pnl_pct': 'mean',
                                        'holding_period': 'mean'
                                    }
                                    period_stats = filtered_buckets.groupby('holding_bucket').agg(simple_agg)
                                    attribution['holding_periods'] = period_stats.to_dict()
                                except Exception as e2:
                                    logger.error(f"Fallback aggregation also failed: {e2}")
                                    attribution['holding_periods'] = {}
                        else:
                            logger.warning("No holding period buckets with enough entries for statistics")
                            attribution['holding_periods'] = {}
                    except Exception as e:
                        logger.error(f"Error calculating holding period statistics: {e}")
                        attribution['holding_periods'] = {}
                else:
                    logger.warning(f"Missing required columns for holding period analysis. Available: {valid_buckets.columns.tolist()}")
                    attribution['holding_periods'] = {}
            else:
                # No valid buckets, set an empty dict
                logger.warning("No valid holding period buckets found after filtering NaNs")
                attribution['holding_periods'] = {}
            
        # Attribution by win/loss
        trades_df['outcome'] = trades_df['pnl'].apply(lambda x: 'win' if x > 0 else 'loss')
        outcome_stats = trades_df.groupby('outcome').agg({
            'pnl': ['sum', 'mean', 'count', 'std'],
            'pnl_pct': ['mean', 'std']
        })
        attribution['outcomes'] = outcome_stats.to_dict()
        
        # Temporal attribution (monthly)
        if 'date' in trades_df.columns:
            # Convert to datetime without timezone info first to avoid warnings
            trades_df['month'] = pd.to_datetime(trades_df['date']).dt.tz_localize(None).dt.to_period('M')
            monthly_stats = trades_df.groupby('month').agg({
                'pnl': ['sum', 'mean', 'count', 'std'],
                'pnl_pct': ['mean', 'std']
            })
            attribution['monthly'] = monthly_stats.to_dict()
            
        return attribution
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect market regime based on trends and volatility
        
        Args:
            data: Price data with OHLCV columns
        
        Returns:
            Market regime classification
        """
        # Calculate key metrics
        returns = data['Close'].pct_change()
        volatility = returns.rolling(21).std() * np.sqrt(252)
        
        # For shorter timeframes, use a shorter MA pair
        if len(data) < 200:
            trend = data['Close'].rolling(20).mean() / data['Close'].rolling(50).mean() - 1
        else:
            trend = data['Close'].rolling(50).mean() / data['Close'].rolling(200).mean() - 1
        
        # Classify regime
        current_vol = volatility.iloc[-1] if not volatility.empty else 0
        current_trend = trend.iloc[-1] if not trend.empty else 0
        
        if current_trend > 0.05:
            if current_vol > volatility.quantile(0.75) if not volatility.empty else 0:
                return 'volatile_bull'
            return 'steady_bull'
        elif current_trend < -0.05:
            if current_vol > volatility.quantile(0.75) if not volatility.empty else 0:
                return 'volatile_bear'
            return 'steady_bear'
        else:
            if current_vol > volatility.quantile(0.75) if not volatility.empty else 0:
                return 'volatile_sideways'
            return 'steady_sideways'
    
    def _update_portfolio_equity(self, date: datetime, data: Dict[str, pd.DataFrame], portfolio: Dict):
        """Update portfolio equity based on current market prices"""
        # Start with cash
        total_equity = portfolio['cash']
        
        # Add value of all positions
        for symbol, position in portfolio['positions'].items():
            if symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date, 'Close']
                position_value = position['quantity'] * current_price
                total_equity += position_value
        
        # Update portfolio equity
        portfolio['equity'] = total_equity
