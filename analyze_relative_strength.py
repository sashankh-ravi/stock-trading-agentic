"""
Script to analyze and validate relative strength metrics on real market data.
"""

import pandas as pd
import numpy as np
from relative_strength import calculate_relative_strength_metrics
from data_collector import DataCollector
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def load_test_data() -> tuple:
    """Load sample stock and market data for testing."""
    try:
        # Initialize data collector
        dc = DataCollector()
        
        # Get market data
        market_data = dc.get_market_data(['NIFTY50', 'BANKNIFTY'])
        
        # Get some sample stocks
        stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        stock_data = {
            stock: dc.get_stock_data(stock) 
            for stock in stocks
        }
        
        return stock_data, market_data
        
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return {}, pd.DataFrame()

def analyze_relative_strength(
    stock_data: Dict[str, pd.DataFrame],
    market_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze relative strength metrics for multiple stocks.
    
    Parameters:
    -----------
    stock_data : Dict[str, pd.DataFrame]
        Dictionary of stock price data frames
    market_data : pd.DataFrame
        Market indices data
        
    Returns:
    --------
    pd.DataFrame
        Analysis results
    """
    results = []
    
    for symbol, data in stock_data.items():
        try:
            # Calculate RS metrics
            rs_metrics = calculate_relative_strength_metrics(
                data['Close'],
                market_data,
                window=90
            )
            
            # Add basic stats
            metrics = {
                'symbol': symbol,
                'data_points': len(data),
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'price_return': (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1)
            }
            
            # Add RS metrics
            metrics.update(rs_metrics)
            
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            
    return pd.DataFrame(results)

def plot_relative_strength_analysis(results: pd.DataFrame):
    """Plot relative strength analysis results."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: RS vs Returns
        plt.subplot(2, 2, 1)
        plt.scatter(results['price_return'], results['RS_CUM_NIFTY50'])
        plt.xlabel('Price Return')
        plt.ylabel('RS vs NIFTY50')
        plt.title('RS vs Price Returns')
        
        # Plot 2: RS Distribution
        plt.subplot(2, 2, 2)
        rs_cols = [col for col in results.columns if col.startswith('RS_')]
        rs_data = results[rs_cols]
        sns.boxplot(data=rs_data)
        plt.xticks(rotation=45)
        plt.title('RS Metrics Distribution')
        
        # Plot 3: RS Momentum
        plt.subplot(2, 2, 3)
        mom_cols = [col for col in results.columns if 'MOM' in col]
        mom_data = results[mom_cols]
        sns.barplot(data=mom_data)
        plt.xticks(rotation=45)
        plt.title('RS Momentum Comparison')
        
        plt.tight_layout()
        plt.savefig('rs_analysis.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting results: {e}")

def main():
    """Main analysis function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Loading test data...")
    stock_data, market_data = load_test_data()
    
    if not stock_data or market_data.empty:
        logger.error("Failed to load test data")
        return
    
    logger.info("Analyzing relative strength...")
    results = analyze_relative_strength(stock_data, market_data)
    
    logger.info("Analysis Results:")
    print(results.to_string())
    
    logger.info("Plotting results...")
    plot_relative_strength_analysis(results)
    
    logger.info("Analysis complete. Check rs_analysis.png for visualizations.")

if __name__ == "__main__":
    main()
