#!/usr/bin/env python
"""
Master runner script for the comprehensive Nifty 500 data download and enrichment pipeline.
This script runs the complete pipeline with all components:

1. Downloads historical price data for Nifty 500 stocks
2. Calculates technical indicators
3. Enriches data with fundamental analysis
4. Adds news sentiment
5. Calculates intermarket correlations
6. Measures relative strength
7. Analyzes market regime
8. Performs validation and generates summary reports

Usage:
    python run_master_pipeline.py [--max_symbols N] [--start_date YYYY-MM-DD] [--end_date YYYY-MM-DD]
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import pandas as pd
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_run.log')
    ]
)
logger = logging.getLogger(__name__)

# Import necessary functions
from download_nifty500_data import (
    download_nifty500_data,
    get_fundamental_data,
    get_news_sentiment,
    get_intermarket_correlations,
    get_relative_strength
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the comprehensive Nifty 500 data pipeline')
    
    # Add arguments
    parser.add_argument('--max_symbols', type=int, default=None, 
                        help='Maximum number of symbols to process (for testing)')
    parser.add_argument('--start_date', type=str, 
                        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, 
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--output_dir', type=str, 
                        default='data/pipeline_output',
                        help='Output directory for pipeline results')
    parser.add_argument('--verification', action='store_true',
                        help='Run verification tests after pipeline execution')
                        
    return parser.parse_args()

def run_pipeline(args):
    """Run the comprehensive data pipeline"""
    print("\n" + "="*80)
    print("NIFTY 500 COMPREHENSIVE DATA PIPELINE")
    print("="*80)
    
    start_time = datetime.now()
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory structure
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    (output_path / "fundamental").mkdir(exist_ok=True)
    (output_path / "news").mkdir(exist_ok=True)
    (output_path / "correlations").mkdir(exist_ok=True)
    (output_path / "relative_strength").mkdir(exist_ok=True)
    
    print(f"\nParameters:")
    print(f"- Start Date: {args.start_date}")
    print(f"- End Date: {args.end_date}")
    print(f"- Output Directory: {args.output_dir}")
    print(f"- Max Symbols: {args.max_symbols if args.max_symbols else 'All'}")
    
    # Stage 1: Download price data and calculate technical indicators
    print("\n" + "-"*80)
    print("STAGE 1: Downloading price data and calculating technical indicators")
    print("-"*80)
    
    stage1_start = datetime.now()
    download_nifty500_data(args.start_date, args.end_date, args.output_dir, args.max_symbols)
    stage1_duration = (datetime.now() - stage1_start).total_seconds()
    print(f"\nStage 1 completed in {stage1_duration:.1f} seconds")
    
    # Load the downloaded data
    try:
        price_data = pd.read_parquet(output_path / "nifty500_complete.parquet")
        symbols = price_data['symbol'].unique().tolist()
        print(f"Successfully loaded price data for {len(symbols)} symbols")
    except Exception as e:
        logger.error(f"Error loading price data: {e}")
        print(f"Error loading price data. Pipeline cannot continue.")
        return False
    
    # Limit to smaller set for testing if needed
    if args.max_symbols and args.max_symbols < len(symbols):
        symbols = symbols[:args.max_symbols]
        
    # Stage 2: Enrich with fundamental data, news, correlations, and relative strength
    print("\n" + "-"*80)
    print("STAGE 2: Enriching data with fundamental analysis, news sentiment, correlations, and RS")
    print("-"*80)
    
    stage2_start = datetime.now()
    
    # Process each enrichment component
    enrichment_results = {
        'fundamental': {'success': 0, 'failure': 0},
        'news': {'success': 0, 'failure': 0},
        'correlations': {'success': 0, 'failure': 0},
        'relative_strength': {'success': 0, 'failure': 0}
    }
    
    for i, symbol in enumerate(symbols):
        print(f"\nProcessing enrichment for {symbol} ({i+1}/{len(symbols)})")
        
        # Get symbol's price data
        symbol_data = price_data[price_data['symbol'] == symbol]
        
        # Fundamental data
        try:
            print(f"Getting fundamental data for {symbol}...")
            fundamental = get_fundamental_data(symbol)
            with open(output_path / "fundamental" / f"{symbol.replace('.NS', '')}_fundamental.json", 'w') as f:
                json.dump(fundamental, f, indent=2)
            enrichment_results['fundamental']['success'] += 1
        except Exception as e:
            logger.error(f"Error getting fundamental data for {symbol}: {e}")
            enrichment_results['fundamental']['failure'] += 1
        
        # News sentiment
        try:
            print(f"Getting news sentiment for {symbol}...")
            sentiment = get_news_sentiment(symbol)
            with open(output_path / "news" / f"{symbol.replace('.NS', '')}_news.json", 'w') as f:
                json.dump(sentiment, f, indent=2)
            enrichment_results['news']['success'] += 1
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            enrichment_results['news']['failure'] += 1
        
        # Intermarket correlations
        try:
            print(f"Calculating intermarket correlations for {symbol}...")
            correlations = get_intermarket_correlations(symbol, symbol_data)
            with open(output_path / "correlations" / f"{symbol.replace('.NS', '')}_correlations.json", 'w') as f:
                json.dump(correlations, f, indent=2)
            enrichment_results['correlations']['success'] += 1
        except Exception as e:
            logger.error(f"Error calculating correlations for {symbol}: {e}")
            enrichment_results['correlations']['failure'] += 1
        
        # Relative strength
        try:
            print(f"Calculating relative strength for {symbol}...")
            rs_metrics = get_relative_strength(symbol, symbol_data)
            with open(output_path / "relative_strength" / f"{symbol.replace('.NS', '')}_rs.json", 'w') as f:
                json.dump(rs_metrics, f, indent=2)
            enrichment_results['relative_strength']['success'] += 1
        except Exception as e:
            logger.error(f"Error calculating relative strength for {symbol}: {e}")
            enrichment_results['relative_strength']['failure'] += 1
        
        # Avoid rate limiting
        if i < len(symbols) - 1:
            time.sleep(2)
    
    stage2_duration = (datetime.now() - stage2_start).total_seconds()
    print(f"\nStage 2 completed in {stage2_duration:.1f} seconds")
    
    # Print enrichment summary
    print("\nEnrichment Results:")
    for component, results in enrichment_results.items():
        total = results['success'] + results['failure']
        success_rate = (results['success'] / total * 100) if total > 0 else 0
        print(f"- {component.capitalize()}: {results['success']}/{total} successful ({success_rate:.1f}%)")
    
    # Run verification tests if requested
    if args.verification:
        print("\n" + "-"*80)
        print("STAGE 3: Running verification tests")
        print("-"*80)
        
        import subprocess
        result = subprocess.run(
            [sys.executable, "run_verification_tests.py"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.returncode != 0:
            print("Verification tests failed. See logs for details.")
    
    # Pipeline complete
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print("="*80)
    
    return True

if __name__ == "__main__":
    args = parse_arguments()
    success = run_pipeline(args)
    sys.exit(0 if success else 1)
