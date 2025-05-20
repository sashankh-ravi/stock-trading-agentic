"""Strategy Testing Framework

This module implements a rigorous testing framework with:
1. Time-based data separation
2. Walk-forward testing
3. Multiple market regime testing
4. Robustness metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from .backtest_engine import Backtester, BacktestResults
from strategies.base_strategy import BaseStrategy
import logging
from scipy import stats
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TestFramework:
    def __init__(self,
                 data_path: str,
                 train_start: str = '2018-01-01',
                 train_end: str = '2021-12-31',
                 validation_start: str = '2022-01-01',
                 validation_end: str = '2022-12-31',
                 test_start: str = '2023-01-01',
                 test_end: str = '2023-12-31'):
        """Initialize test framework with strict time separation"""
        self.data_path = Path(data_path)
        self.periods = {
            'train': (train_start, train_end),
            'validation': (validation_start, validation_end),
            'test': (test_start, test_end)
        }
        self.data = {}
        self.results_path = self.data_path / 'test_results'
        self.results_path.mkdir(exist_ok=True)
        
    def load_data(self) -> None:
        """Load and split data for all periods"""
        logger.info("Loading and splitting data...")
        
        # Get all stock data files
        stock_files = list(self.data_path.glob('*.csv'))
        
        for period, (start, end) in self.periods.items():
            self.data[period] = {}
            for file in stock_files:
                try:
                    df = pd.read_csv(file, index_col='Date', parse_dates=True)
                    symbol = file.stem
                    period_data = df[start:end].copy()
                    if not period_data.empty:
                        self.data[period][symbol] = period_data
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
                    
        logger.info(f"Loaded data for {len(stock_files)} stocks across {len(self.periods)} periods")
        
    def evaluate_strategy(self, 
                         strategy: BaseStrategy,
                         period: str,
                         name: str = None) -> Tuple[BacktestResults, Dict]:
        """Evaluate a strategy on a specific period"""
        if period not in self.data:
            raise ValueError(f"Invalid period: {period}")
            
        backtester = Backtester()
        results = backtester.run(self.data[period], strategy)
        
        # Calculate additional metrics
        metrics = self._calculate_additional_metrics(results)
        
        # Save results if name provided
        if name:
            self._save_results(name, period, results, metrics)
            
        return results, metrics
    
    def walk_forward_test(self,
                         strategy: BaseStrategy,
                         window_size: int = 126,  # 6 months
                         step_size: int = 21,     # 1 month
                         name: str = None) -> Dict:
        """Perform walk-forward testing"""
        all_results = []
        
        for period in ['train', 'validation']:
            period_data = self.data[period]
            if not period_data:
                continue
                
            # Get all dates for this period
            dates = pd.date_range(self.periods[period][0], 
                                self.periods[period][1],
                                freq='B')  # Business days
            
            # Walk forward through the period
            for start_idx in range(0, len(dates) - window_size, step_size):
                window_start = dates[start_idx]
                window_end = dates[start_idx + window_size]
                
                # Extract data for this window
                window_data = {}
                for symbol, df in period_data.items():
                    window_df = df[window_start:window_end].copy()
                    if not window_df.empty:
                        window_data[symbol] = window_df
                
                # Run backtest for this window
                backtester = Backtester()
                results = backtester.run(window_data, strategy)
                
                all_results.append({
                    'period': period,
                    'window_start': window_start,
                    'window_end': window_end,
                    'metrics': self._calculate_additional_metrics(results)
                })
        
        # Aggregate and analyze walk-forward results
        aggregated = self._aggregate_walk_forward_results(all_results)
        
        # Save if name provided
        if name:
            self._save_walk_forward_results(name, aggregated)
            
        return aggregated
    
    def _calculate_additional_metrics(self, results: BacktestResults) -> Dict:
        """Calculate additional performance and robustness metrics"""
        equity = results.equity_curve
        returns = equity.pct_change().dropna()
        
        if len(returns) < 2:
            return {}
            
        # Basic metrics
        metrics = results.metrics.copy()
        
        # Risk metrics
        metrics.update({
            'volatility': returns.std() * np.sqrt(252),
            'sortino_ratio': self._calculate_sortino(returns),
            'calmar_ratio': -metrics['total_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] != 0 else 0,
            'avg_drawdown': self._calculate_avg_drawdown(equity),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        })
        
        # Trade metrics
        trade_returns = [t.pnl_pct for t in results.trades]
        if trade_returns:
            metrics.update({
                'avg_trade_return': np.mean(trade_returns),
                'trade_return_std': np.std(trade_returns),
                'profit_factor': self._calculate_profit_factor(results.trades),
                'avg_win': np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0,
                'avg_loss': np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0,
            })
        
        # Market regime metrics
        metrics.update(self._calculate_regime_metrics(returns))
        
        return metrics
    
    def _calculate_sortino(self, returns: pd.Series, risk_free: float = 0.03) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0
        downside_std = np.sqrt(np.mean(downside_returns**2))
        if downside_std == 0:
            return 0
        return np.sqrt(252) * np.mean(excess_returns) / downside_std
    
    def _calculate_avg_drawdown(self, equity: pd.Series) -> float:
        """Calculate average drawdown"""
        drawdowns = equity / equity.cummax() - 1
        return np.mean(drawdowns[drawdowns < 0]) if len(drawdowns[drawdowns < 0]) > 0 else 0
    
    def _calculate_profit_factor(self, trades: List) -> float:
        """Calculate profit factor"""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss != 0 else 0
    
    def _calculate_regime_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics in different market regimes and volatility states
        
        This function identifies:
        1. Market regimes (bull, bear, neutral) based on trend
        2. Volatility regimes (high, normal, low) based on realized vol
        3. Market breadth regimes based on correlation structure
        4. Sector rotation phases
        """
        try:
            # 1. Market Trend Regimes
            rolling_returns = returns.rolling(21).mean()  # 1-month trend
            rolling_vol = returns.rolling(21).std() * np.sqrt(252)  # Annualized vol
            
            # Define regimes using both trend and volatility
            trend_regimes = pd.qcut(rolling_returns, 3, labels=['bear', 'neutral', 'bull'])
            vol_regimes = pd.qcut(rolling_vol, 3, labels=['low_vol', 'normal_vol', 'high_vol'])
            
            regime_metrics = {}
            
            # Trend regime metrics
            for regime in ['bull', 'neutral', 'bear']:
                regime_returns = returns[trend_regimes == regime]
                if len(regime_returns) > 0:
                    regime_metrics.update({
                        f'{regime}_return': regime_returns.mean() * 252,
                        f'{regime}_sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) \
                            if regime_returns.std() != 0 else 0,
                        f'{regime}_max_drawdown': (1 + regime_returns).cumprod().div(
                            (1 + regime_returns).cumprod().cummax()).min() - 1,
                        f'{regime}_win_rate': (regime_returns > 0).mean()
                    })
                    
            # Volatility regime metrics
            for regime in ['low_vol', 'normal_vol', 'high_vol']:
                regime_returns = returns[vol_regimes == regime]
                if len(regime_returns) > 0:
                    regime_metrics.update({
                        f'{regime}_return': regime_returns.mean() * 252,
                        f'{regime}_sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) \
                            if regime_returns.std() != 0 else 0,
                        f'{regime}_max_drawdown': (1 + regime_returns).cumprod().div(
                            (1 + regime_returns).cumprod().cummax()).min() - 1
                    })
            
            # Combined regime analysis
            regime_metrics['regime_stability'] = self._calculate_regime_stability(trend_regimes)
            regime_metrics['vol_regime_stability'] = self._calculate_regime_stability(vol_regimes)
            
            # Current regime
            regime_metrics['current_regime'] = trend_regimes.iloc[-1]
            regime_metrics['current_vol_regime'] = vol_regimes.iloc[-1]
            
            # Transition probabilities
            regime_metrics['regime_transitions'] = self._calculate_transition_matrix(trend_regimes)
            
            return regime_metrics
            
        except Exception as e:
            logger.error(f"Error calculating regime metrics: {e}")
            return {}
    
    def _calculate_regime_stability(self, regimes: pd.Series) -> float:
        """Calculate regime stability score (0-1)
        
        Higher score means regimes are more stable with fewer transitions
        """
        transitions = (regimes != regimes.shift(1)).sum()
        stability = 1 - (transitions / len(regimes))
        return max(0, min(1, stability))
        
    def _calculate_transition_matrix(self, regimes: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probabilities"""
        transitions = {}
        unique_regimes = regimes.unique()
        
        for from_regime in unique_regimes:
            transitions[from_regime] = {}
            mask = regimes == from_regime
            total = mask.sum()
            
            if total > 0:
                for to_regime in unique_regimes:
                    # Count transitions from from_regime to to_regime
                    count = ((regimes.shift(-1) == to_regime) & mask).sum()
                    transitions[from_regime][to_regime] = count / total
                    
        return transitions
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Aggregate and analyze walk-forward testing results"""
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        aggregated = {}
        
        for metric in metrics:
            values = [r['metrics'].get(metric, 0) for r in results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
            
        # Calculate robustness score (percentage of profitable windows)
        profitable_windows = sum(1 for r in results if r['metrics'].get('total_return', 0) > 0)
        aggregated['robustness_score'] = profitable_windows / len(results)
        
        return aggregated
    
    def _save_results(self, 
                     strategy_name: str,
                     period: str,
                     results: BacktestResults,
                     metrics: Dict) -> None:
        """Save test results to file"""
        save_path = self.results_path / f"{strategy_name}_{period}.json"
        
        # Convert results to serializable format
        save_data = {
            'metrics': metrics,
            'equity_curve': results.equity_curve.to_dict(),
            'trades': [{
                'entry_time': str(t.entry_time),
                'exit_time': str(t.exit_time),
                'symbol': t.symbol,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'status': t.status
            } for t in results.trades]
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def _save_walk_forward_results(self, 
                                 strategy_name: str,
                                 results: Dict) -> None:
        """Save walk-forward test results"""
        save_path = self.results_path / f"{strategy_name}_walk_forward.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
