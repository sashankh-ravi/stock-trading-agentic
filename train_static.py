"""
Static Strategy Training Framework for Nifty 500

This module implements:
1. Data loading and splitting (3y train, 1y validation, 1y test)
2. Walk-forward optimization
3. Strategy evaluation across market regimes
4. Performance measurement and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from sklearn.model_selection import TimeSeriesSplit
from technical_indicators import TechnicalIndicators
from backtesting.test_framework import TestFramework
from backtesting.nifty_backtester import NiftyBacktester
from strategies.nifty500_strategy import Nifty500Strategy

logger = logging.getLogger(__name__)

class StrategyTrainer:
    def __init__(self,
                 data_path: str,
                 train_start: str = '2020-01-01',
                 train_end: str = '2022-12-31',
                 validation_start: str = '2023-01-01',
                 validation_end: str = '2023-12-31',
                 test_start: str = '2024-01-01',
                 test_end: str = '2024-12-31'):
        """
        Initialize strategy trainer with time periods
        
        Args:
            data_path: Path to data directory
            train_start: Training period start date
            train_end: Training period end date 
            validation_start: Validation period start date
            validation_end: Validation period end date
            test_start: Test period start date 
            test_end: Test period end date
        """
        self.data_path = Path(data_path)
        self.periods = {
            'train': (train_start, train_end),
            'validation': (validation_start, validation_end),
            'test': (test_start, test_end)
        }
        
        # Initialize components
        self.test_framework = TestFramework(
            data_path=data_path,
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end
        )
        self.technical_indicators = TechnicalIndicators()
        self.backtester = NiftyBacktester(Nifty500Strategy())
        
        self.results_path = self.data_path / 'training_results'
        self.results_path.mkdir(exist_ok=True)

    def load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare data for all periods"""
        logger.info("Loading and preparing data...")
        self.test_framework.load_data()
        return self.test_framework.data

    def detect_market_regime(self, data: pd.DataFrame) -> str:
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
        trend = data['Close'].rolling(50).mean() / data['Close'].rolling(200).mean() - 1
        
        # Classify regime
        current_vol = volatility.iloc[-1]
        current_trend = trend.iloc[-1]
        
        if current_trend > 0.05:
            if current_vol > volatility.quantile(0.75):
                return 'volatile_bull'
            return 'steady_bull'
        elif current_trend < -0.05:
            if current_vol > volatility.quantile(0.75):
                return 'volatile_bear'
            return 'steady_bear'
        else:
            if current_vol > volatility.quantile(0.75):
                return 'volatile_sideways'
            return 'steady_sideways'

    def walk_forward_optimization(self, 
                                data: pd.DataFrame,
                                param_grid: Dict,
                                window_size: int = 252,  # 1 year
                                step_size: int = 63) -> List[Dict]:
        """
        Implement walk-forward optimization
        
        Args:
            data: Price data
            param_grid: Grid of parameters to test
            window_size: Size of training window
            step_size: Step size for moving window
        
        Returns:
            List of results for each window
        """
        results = []
        
        for i in range(0, len(data) - window_size - step_size, step_size):
            # Split data
            train_data = data.iloc[i:i+window_size]
            test_data = data.iloc[i+window_size:i+window_size+step_size]
            
            # Optimize parameters on training data
            best_params = self._grid_search(train_data, param_grid)
            
            # Test on out-of-sample data
            test_results = self.backtester.run_backtest(
                {'NIFTY500': test_data},
                start_date=test_data.index[0],
                end_date=test_data.index[-1]
            )
            
            results.append({
                'window_start': data.index[i],
                'window_end': data.index[i+window_size],
                'test_start': data.index[i+window_size],
                'test_end': data.index[i+window_size+step_size],
                'parameters': best_params,
                'performance': test_results.metrics,
                'market_regime': self.detect_market_regime(test_data)
            })
        
        return results

    def _grid_search(self, data: pd.DataFrame, param_grid: Dict) -> Dict:
        """
        Perform grid search for parameter optimization
        
        Args:
            data: Training data
            param_grid: Parameter grid to search
        
        Returns:
            Best parameters found
        """
        best_score = float('-inf')
        best_params = None
        
        for params in self._generate_param_combinations(param_grid):
            self.backtester.strategy.set_parameters(params)
            results = self.backtester.run_backtest(
                {'NIFTY500': data},
                start_date=data.index[0],
                end_date=data.index[-1]
            )
            
            # Calculate objective function (e.g., Sharpe ratio)
            score = self._calculate_objective(results)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params

    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations from grid"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = itertools.product(*values)
        
        return [dict(zip(keys, combo)) for combo in combinations]

    def _calculate_objective(self, results) -> float:
        """Calculate objective function for optimization"""
        if not results or not results.metrics:
            return float('-inf')
        
        sharpe = results.metrics.get('sharpe_ratio', 0)
        max_dd = abs(results.metrics.get('max_drawdown', 1))
        
        # Penalize large drawdowns
        return sharpe / max_dd if max_dd > 0 else sharpe

    def evaluate_strategy(self, 
                         data: pd.DataFrame,
                         parameters: Dict,
                         period_name: str = 'validation') -> Dict:
        """
        Evaluate strategy performance
        
        Args:
            data: Price data
            parameters: Strategy parameters
            period_name: Name of evaluation period
        
        Returns:
            Performance metrics
        """
        self.backtester.strategy.set_parameters(parameters)
        results = self.backtester.run_backtest(
            {'NIFTY500': data},
            start_date=data.index[0],
            end_date=data.index[-1]
        )
        
        # Add regime-specific analysis
        market_regime = self.detect_market_regime(data)
        regime_metrics = self._calculate_regime_metrics(results, market_regime)
        
        # Combine all metrics
        metrics = {
            'period': period_name,
            'market_regime': market_regime,
            **results.metrics,
            **regime_metrics
        }
        
        return metrics

    def _calculate_regime_metrics(self, results, regime: str) -> Dict:
        """Calculate regime-specific metrics"""
        if not results or not results.trades:
            return {}
        
        trades_df = pd.DataFrame(results.trades)
        regime_trades = trades_df[trades_df['market_regime'] == regime]
        
        return {
            'regime_win_rate': (regime_trades['pnl'] > 0).mean() if len(regime_trades) > 0 else 0,
            'regime_avg_return': regime_trades['pnl'].mean() if len(regime_trades) > 0 else 0,
            'regime_sharpe': (regime_trades['pnl'].mean() / regime_trades['pnl'].std() 
                            if len(regime_trades) > 0 and regime_trades['pnl'].std() != 0 
                            else 0),
            'regime_trades': len(regime_trades)
        }

    def run_training(self, param_grid: Dict) -> Dict:
        """
        Run complete training process
        
        Args:
            param_grid: Parameter grid for optimization
        
        Returns:
            Training results and best parameters
        """
        logger.info("Starting training process...")
        
        # Load data
        try:
            data = self.load_and_prepare_data()
            
            # Validate data
            for period in ['train', 'validation', 'test']:
                if period not in data or 'NIFTY500' not in data[period]:
                    raise ValueError(f"Missing data for {period} period")
                if len(data[period]['NIFTY500']) < 252:  # At least one year of data
                    raise ValueError(f"Insufficient data for {period} period")
                    
            logger.info("Data validation successful")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
        
        # Walk-forward optimization on training data
        try:
            logger.info("Starting walk-forward optimization...")
            wfo_results = self.walk_forward_optimization(
                data['train']['NIFTY500'],
                param_grid
            )
            if not wfo_results:
                raise ValueError("Walk-forward optimization returned no results")
            logger.info(f"Completed {len(wfo_results)} optimization windows")
        except Exception as e:
            logger.error(f"Walk-forward optimization failed: {str(e)}")
            raise
        
        # Find best parameters across all windows
        try:
            best_params = self._select_best_parameters(wfo_results)
            logger.info(f"Selected best parameters: {best_params}")
            
            # Validate parameters
            required_params = {'ema_short', 'ema_long', 'rsi_period', 'volume_ma'}
            if not all(param in best_params for param in required_params):
                raise ValueError("Best parameters missing required values")
        except Exception as e:
            logger.error(f"Parameter selection failed: {str(e)}")
            raise
        
        # Evaluate on validation set
        try:
            logger.info("Starting validation period evaluation...")
            validation_metrics = self.evaluate_strategy(
                data['validation']['NIFTY500'],
                best_params,
                'validation'
            )
            logger.info(f"Validation Sharpe ratio: {validation_metrics.get('sharpe_ratio', 0):.2f}")
        except Exception as e:
            logger.error(f"Validation evaluation failed: {str(e)}")
            raise
        
        # Only proceed to test set if validation meets minimum criteria
        if validation_metrics.get('sharpe_ratio', 0) < 0.5:
            logger.warning("Low validation performance - consider parameter tuning")
        
        # Evaluate on test set
        try:
            logger.info("Starting test period evaluation...")
            test_metrics = self.evaluate_strategy(
                data['test']['NIFTY500'],
                best_params,
                'test'
            )
            logger.info(f"Test Sharpe ratio: {test_metrics.get('sharpe_ratio', 0):.2f}")
        except Exception as e:
            logger.error(f"Test evaluation failed: {str(e)}")
            raise
        
        # Compile and analyze results
        results = {
            'wfo_results': wfo_results,
            'best_parameters': best_params,
            'validation_metrics': validation_metrics,
            'test_metrics': test_metrics,
            'performance_summary': {
                'train_windows': len(wfo_results),
                'avg_train_sharpe': np.mean([r['performance']['sharpe_ratio'] for r in wfo_results]),
                'validation_sharpe': validation_metrics.get('sharpe_ratio', 0),
                'test_sharpe': test_metrics.get('sharpe_ratio', 0),
                'market_regimes_encountered': list(set(r['market_regime'] for r in wfo_results))
            }
        }
        
        # Save results
        try:
            self._save_results(results)
            logger.info("Training results saved successfully")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise
        
        return results

    def _select_best_parameters(self, wfo_results: List[Dict]) -> Dict:
        """Select best parameters from walk-forward optimization results"""
        # Calculate average performance for each parameter set
        param_performance = {}
        
        for result in wfo_results:
            params_key = str(result['parameters'])
            if params_key not in param_performance:
                param_performance[params_key] = []
            param_performance[params_key].append(result['performance']['sharpe_ratio'])
        
        # Select parameters with best average performance
        best_params_key = max(param_performance.items(), 
                            key=lambda x: np.mean(x[1]))[0]
        
        return eval(best_params_key)  # Convert string back to dict

    def _save_results(self, results: Dict):
        """Save training results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        save_path = self.results_path / f'training_results_{timestamp}.json'
        
        # Convert results to serializable format
        serializable_results = {
            'wfo_results': [{
                'window_start': str(r['window_start']),
                'window_end': str(r['window_end']),
                'test_start': str(r['test_start']),
                'test_end': str(r['test_end']),
                'parameters': r['parameters'],
                'performance': r['performance'],
                'market_regime': r['market_regime']
            } for r in results['wfo_results']],
            'best_parameters': results['best_parameters'],
            'validation_metrics': results['validation_metrics'],
            'test_metrics': results['test_metrics']
        }
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")

def main():
    # Example usage
    trainer = StrategyTrainer(
        data_path='data',
        train_start='2021-01-01',  # Shorter training period
        train_end='2022-12-31',
        validation_start='2023-01-01',
        validation_end='2023-06-30',  # Shorter validation period
        test_start='2023-07-01',      # Shorter test period
        test_end='2023-12-31'
    )
    
    # Define parameter grid (simplified for faster testing)
    param_grid = {
        'ema_short': [9],
        'ema_long': [21],
        'macd_fast': [12],
        'macd_slow': [26],
        'macd_signal': [9],
        'rsi_period': [14],
        'rsi_overbought': [70],
        'rsi_oversold': [30],
        'volume_ma': [20],
        'atr_period': [14],
        'stop_loss': [2.0],
        'take_profit': [4.0]
    }
    
    # Run training
    results = trainer.run_training(param_grid)
    
    # Log key metrics
    logger.info("\nTraining Results:")
    logger.info(f"Best Parameters: {results['best_parameters']}")
    logger.info(f"Validation Sharpe: {results['validation_metrics']['sharpe_ratio']:.2f}")
    logger.info(f"Test Sharpe: {results['test_metrics']['sharpe_ratio']:.2f}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()