"""Run Comprehensive Strategy Tests

This script runs extensive tests on trading strategies using:
1. In-sample optimization (2018-2021)
2. Out-of-sample validation (2022)
3. Strict out-of-sample testing (2023+)
"""

import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from strategies.trend_strategy import TrendStrategy
from backtesting.test_framework import TestFramework
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_strategy_parameters() -> List[Dict]:
    """Generate strategy parameter combinations"""
    param_grid = {
        'lookback': [10, 20, 30],
        'stop_loss_pct': [0.02, 0.03, 0.05],
        'take_profit_pct': [0.05, 0.10, 0.15],
        'ema_short': [5, 9, 13],
        'ema_medium': [20, 21, 25],
        'ema_long': [50, 55, 60],
        'volume_ma': [10, 20, 30],
        'rsi_period': [7, 14, 21],
        'rsi_overbought': [70, 75, 80],
        'rsi_oversold': [20, 25, 30]
    }
    
    # Generate combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]

def test_strategy_config(config: Dict) -> Dict:
    """Test a single strategy configuration"""
    try:
        # Create strategy
        strategy = TrendStrategy(**config['parameters'])
        
        # Run tests
        framework = config['framework']
        train_results, train_metrics = framework.evaluate_strategy(
            strategy, 'train', f"config_{config['id']}")
        
        val_results, val_metrics = framework.evaluate_strategy(
            strategy, 'validation')
            
        # Calculate stability score
        stability = calculate_stability_score(train_metrics, val_metrics)
        
        return {
            'id': config['id'],
            'parameters': config['parameters'],
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'stability_score': stability
        }
        
    except Exception as e:
        logger.error(f"Error testing configuration {config['id']}: {e}")
        return None

def calculate_stability_score(train_metrics: Dict, val_metrics: Dict) -> float:
    """Calculate stability score between training and validation periods"""
    key_metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate']
    stability_scores = []
    
    for metric in key_metrics:
        if metric in train_metrics and metric in val_metrics:
            # Calculate relative difference
            diff = abs(train_metrics[metric] - val_metrics[metric])
            max_val = max(abs(train_metrics[metric]), abs(val_metrics[metric]))
            if max_val != 0:
                stability = 1 - (diff / max_val)
                stability_scores.append(stability)
    
    return np.mean(stability_scores) if stability_scores else 0

def analyze_results(results: List[Dict], save_path: Path) -> None:
    """Analyze and visualize test results"""
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Basic statistics
    stats = {
        'train_sharpe_mean': df['train_metrics'].apply(lambda x: x['sharpe_ratio']).mean(),
        'train_sharpe_std': df['train_metrics'].apply(lambda x: x['sharpe_ratio']).std(),
        'val_sharpe_mean': df['validation_metrics'].apply(lambda x: x['sharpe_ratio']).mean(),
        'val_sharpe_std': df['validation_metrics'].apply(lambda x: x['sharpe_ratio']).std(),
        'stability_mean': df['stability_score'].mean(),
        'stability_std': df['stability_score'].std()
    }
    
    # Save statistics
    with open(save_path / 'analysis_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, 
                    x=df['train_metrics'].apply(lambda x: x['sharpe_ratio']),
                    y=df['validation_metrics'].apply(lambda x: x['sharpe_ratio']),
                    alpha=0.5)
    plt.xlabel('Training Sharpe Ratio')
    plt.ylabel('Validation Sharpe Ratio')
    plt.title('Strategy Performance: Training vs Validation')
    plt.savefig(save_path / 'performance_scatter.png')
    plt.close()
    
    # Parameter importance analysis
    param_importance = analyze_parameter_importance(df)
    with open(save_path / 'parameter_importance.json', 'w') as f:
        json.dump(param_importance, f, indent=2)

def analyze_parameter_importance(results_df: pd.DataFrame) -> Dict:
    """Analyze which parameters have the most impact on performance"""
    importance_metrics = {}
    
    for param in results_df['parameters'].iloc[0].keys():
        param_values = results_df['parameters'].apply(lambda x: x[param])
        sharpe_ratios = results_df['validation_metrics'].apply(lambda x: x['sharpe_ratio'])
        
        # Calculate correlation
        correlation = np.corrcoef(param_values, sharpe_ratios)[0, 1]
        importance_metrics[param] = abs(correlation)
    
    # Normalize to 0-1 range
    max_importance = max(importance_metrics.values())
    if max_importance > 0:
        importance_metrics = {k: v/max_importance for k, v in importance_metrics.items()}
    
    return importance_metrics

def main():
    # Setup
    base_path = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50')
    data_path = base_path / 'data'
    results_path = base_path / 'test_results'
    results_path.mkdir(exist_ok=True)
    
    # Initialize test framework
    framework = TestFramework(
        data_path=data_path,
        train_start='2018-01-01',
        train_end='2021-12-31',
        validation_start='2022-01-01',
        validation_end='2022-12-31',
        test_start='2023-01-01',
        test_end='2023-12-31'
    )
    
    # Load data
    framework.load_data()
    
    # Generate parameter combinations
    param_combinations = generate_strategy_parameters()
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    # Prepare test configurations
    test_configs = [
        {
            'id': i,
            'parameters': params,
            'framework': framework
        }
        for i, params in enumerate(param_combinations)
    ]
    
    # Run tests in parallel
    results = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(test_strategy_config, test_configs))
    
    # Filter out failed tests
    results = [r for r in results if r is not None]
    
    # Analyze results
    analyze_results(results, results_path)
    
    # Select best configurations
    results_df = pd.DataFrame(results)
    best_configs = select_best_configurations(results_df)
    
    # Save best configurations
    with open(results_path / 'best_configurations.json', 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    logger.info("Testing completed. Check test_results directory for analysis.")

def select_best_configurations(results_df: pd.DataFrame, top_n: int = 5) -> Dict:
    """Select best configurations based on multiple criteria"""
    # Sort by different metrics
    by_sharpe = results_df.nlargest(top_n, 'validation_metrics').apply(
        lambda x: x['sharpe_ratio'])
    by_stability = results_df.nlargest(top_n, 'stability_score')
    
    # Combine and deduplicate
    best_configs = pd.concat([by_sharpe, by_stability]).drop_duplicates()
    
    return {
        'by_sharpe_ratio': by_sharpe.to_dict(),
        'by_stability': by_stability.to_dict()
    }

if __name__ == "__main__":
    main()
