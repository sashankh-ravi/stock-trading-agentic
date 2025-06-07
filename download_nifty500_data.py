"""
Enhanced Download Framework for Nifty 500 Data

Features:
1. Comprehensive technical indicators with multiple timeframes
2. Fundamental data integration (financial ratios, earnings, valuations)
3. News and social media sentiment analysis with NLP scoring
4. Sector/industry classification and sector rotation analysis
5. Market breadth indicators and market regime detection
6. Intermarket correlations (global indices, commodities, bonds)
7. Relative strength metrics (vs sector, market, peers)
8. Option chain data integration (implied volatility, put/call ratios)
9. Insider trading and institutional ownership analysis
10. Batch processing to avoid rate limiting with automatic retries
11. Data quality verification and anomaly detection
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Any, Union, Set
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import talib  # For candlestick patterns
import json
from bs4 import BeautifulSoup
import re
import datetime
import math
from functools import lru_cache
import statistics
import urllib.parse
import warnings
from download_stock_data import download_stock_data  # Import the function once at the top
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of global market indices, commodities, and bonds to track correlations
GLOBAL_INDICES = {
    # Indian Indices
    '^NSEI': 'Nifty 50',
    '^BSESN': 'Sensex',
    '^NSMIDCP': 'Nifty Midcap',
    '^CNXIT': 'Nifty IT',
    '^CNXBANK': 'Nifty Bank',
    '^CNXPHARMA': 'Nifty Pharma',
    '^CNXAUTO': 'Nifty Auto',
    '^CNXFMCG': 'Nifty FMCG',
    '^CNXMETAL': 'Nifty Metal',
    '^CNXREALTY': 'Nifty Realty',
    
    # Global Indices
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones',
    '^IXIC': 'Nasdaq',
    '^RUT': 'Russell 2000',
    '^VIX': 'VIX',
    '^FTSE': 'FTSE 100',
    '^GDAXI': 'DAX',
    '^FCHI': 'CAC 40',
    '^STOXX50E': 'Euro Stoxx 50',
    '^N225': 'Nikkei 225',
    '^HSI': 'Hang Seng',
    '000001.SS': 'Shanghai Composite',
    '^STI': 'Straits Times',
    '^AXJO': 'ASX 200',
    '^BVSP': 'Bovespa',
}

# Commodities and Fixed Income for intermarket analysis
COMMODITIES_BONDS = {
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'CL=F': 'Crude Oil',
    'NG=F': 'Natural Gas',
    'HG=F': 'Copper',
    'ZC=F': 'Corn',
    'ZW=F': 'Wheat',
    'ZS=F': 'Soybeans',
    '^TNX': '10-Year Treasury Yield',
    '^IRX': '13-Week Treasury Bill',
    'USDINR=X': 'USD/INR',
    'EURINR=X': 'EUR/INR',
}

# Define sector mapping for Nifty 500 stocks
SECTOR_MAPPING = {}  # Will be populated from external source
INDUSTRY_MAPPING = {}  # Will be populated from external source

def get_nifty500_symbols() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Get list of Nifty 500 symbols with .NS suffix for Yahoo Finance
    Also retrieves sector/industry classification
    
    Returns:
        Tuple containing:
        - List of symbols
        - Dictionary mapping symbols to sectors
        - Dictionary mapping symbols to industries
    """
    # Load from local cache
    cache_file = Path("data/nifty500_symbols.json")
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            if time.time() - cache_data['timestamp'] < 86400:  # Cache valid for 24 hours
                logger.info("Using cached Nifty 500 symbols")
                return cache_data['symbols'], {}, {}  # Return empty mappings for now

    # If cache missing or expired, download from NSE
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty500list.csv"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    df = None
    for url in urls:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(pd.io.common.StringIO(response.text))
            break
        logger.warning(f"Failed to download from {url}")
            
    if df is None:
        logger.warning("Failed to fetch from NSE, using default symbols")
        return ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS"], {}, {}
    
    # Add .NS suffix for Yahoo Finance
    symbols = [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
    
    # Save to cache
    cache_data = {
        'timestamp': time.time(),
        'symbols': symbols
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    
    # Store sector and industry information
    global SECTOR_MAPPING, INDUSTRY_MAPPING
    
    # Map sectors - try different column names as NSE format might vary
    sector_col = next((col for col in ['Sector', 'Industry', 'SECTOR', 'INDUSTRY'] if col in df.columns), None)
    industry_col = next((col for col in ['Industry', 'Sub-Industry', 'INDUSTRY'] if col in df.columns), None)
    
    if sector_col:
        SECTOR_MAPPING = {f"{row['Symbol']}.NS": row[sector_col] for _, row in df.iterrows()}
    else:
        logger.warning("Sector column not found in NSE data, using alternate source")
        # Use alternate source for sector data if not available
        
    if industry_col and industry_col != sector_col:
        INDUSTRY_MAPPING = {f"{row['Symbol']}.NS": row[industry_col] for _, row in df.iterrows()}
    else:
        INDUSTRY_MAPPING = SECTOR_MAPPING.copy()
    
    logger.info(f"Successfully retrieved {len(symbols)} Nifty 500 symbols with sector/industry classification")
    
    # If sector data is incomplete, fetch it for each symbol
    if len(SECTOR_MAPPING) < len(symbols) * 0.9:  # If we're missing more than 10% of sector data
        logger.info("Incomplete sector data, enriching from Yahoo Finance")
        _enrich_sector_data(symbols[:10])  # Start with just a few to avoid rate limits
    
    return symbols, SECTOR_MAPPING, INDUSTRY_MAPPING

def _enrich_sector_data(symbols: List[str]):
    """
    Enrich sector and industry data by fetching it from Yahoo Finance
    This is used when the NSE data doesn't have complete sector information
    
    Args:
        symbols: List of stock symbols to enrich
    """
    global SECTOR_MAPPING, INDUSTRY_MAPPING
    
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if 'sector' in info and info['sector']:
            SECTOR_MAPPING[symbol] = info['sector']
        
        if 'industry' in info and info['industry']:
            INDUSTRY_MAPPING[symbol] = info['industry']
                
            # Don't hammer the API
            time.sleep(1)
    
    # Cache the enriched data
    symbols_file = Path("data/nifty500_symbols.json")
    symbols_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(symbols_file, 'w') as f:
        json.dump({
            'symbols': symbols,
            'sectors': SECTOR_MAPPING,
            'industries': INDUSTRY_MAPPING,
            'timestamp': datetime.datetime.now().isoformat()
        }, f)
        
    logger.info("Cached enriched sector data")

def get_fundamental_data(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive fundamental data for a stock
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with detailed fundamental metrics
    """
    ticker = yf.Ticker(symbol)
    
    # Get basic info
    info = ticker.info
    
    # Get financial statements
    income_stmt = ticker.income_stmt
    balance_sheet = ticker.balance_sheet
    cash_flow = ticker.cashflow
    quarterly_income = ticker.quarterly_income_stmt
    quarterly_balance = ticker.quarterly_balance_sheet
    quarterly_cashflow = ticker.quarterly_cashflow
    
    # Use available data sources for categorization
    sector = info.get('sector', SECTOR_MAPPING.get(symbol, 'Unknown'))
    industry = info.get('industry', INDUSTRY_MAPPING.get(symbol, 'Unknown'))
    
    # Create comprehensive fundamentals dictionary
    fundamentals = {
        # Basic Information
        'symbol': symbol,
        'company_name': info.get('longName', info.get('shortName', symbol.replace('.NS', ''))),
        'sector': sector,
        'industry': industry,
        'market_cap': info.get('marketCap', 0),
        'enterprise_value': info.get('enterpriseValue', 0),
        'shares_outstanding': info.get('sharesOutstanding', 0),
        'beta': info.get('beta', 1.0),
        
        # Valuation Metrics
        'pe_ratio': info.get('trailingPE', info.get('forwardPE', 0)),
        'forward_pe': info.get('forwardPE', 0),
        'peg_ratio': info.get('pegRatio', 0),
        'price_to_book': info.get('priceToBook', 0),
        'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
        'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
        'ev_to_revenue': info.get('enterpriseToRevenue', 0),
        'price_to_fcf': 0,  # Will calculate if data available
        
        # Dividend Metrics
        'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
        'dividend_rate': info.get('dividendRate', 0),
        'payout_ratio': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0,
        'dividend_growth_5yr': 0,  # Will calculate if historical data available
        
        # Profitability Metrics
        'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
        'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0,
        'gross_margin': info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0,
        'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
        'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
        'roic': 0,  # Will calculate if data available
        
        # Financial Health
        'current_ratio': info.get('currentRatio', 0),
        'quick_ratio': 0,  # Will calculate from balance sheet
        'debt_to_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
        'interest_coverage': info.get('interestCoverage', 0),
        'total_debt': 0,  # Will extract from balance sheet
        'cash_per_share': 0,  # Will calculate
        
        # Growth Metrics
        'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
        'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0,
        'eps_growth': info.get('earningsQuarterlyGrowth', 0) * 100 if info.get('earningsQuarterlyGrowth') else 0,
        
        # Earnings Data
        'eps_ttm': info.get('trailingEps', 0),
        'eps_forward': info.get('forwardEps', 0),
        'next_earnings_date': info.get('earningsTimestamp', None),
        
        # Analyst Opinions
        'analyst_target_price': info.get('targetMeanPrice', 0),
        'analyst_rating': info.get('recommendationMean', 3),  # 1=Strong Buy, 5=Strong Sell
        'num_analyst_opinions': info.get('numberOfAnalystOpinions', 0),
        
        # Ownership
        'institutional_ownership': info.get('institutionsPercentHeld', 0) * 100 if info.get('institutionsPercentHeld') else 0,
        'insider_ownership': info.get('insiderPercentHeld', 0) * 100 if info.get('insiderPercentHeld') else 0,
    }
    
    # Calculate additional metrics from financial statements if available
    if not income_stmt.empty and not balance_sheet.empty:
        if 1==1:
            # Get most recent annual data
            recent_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0
            recent_net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            recent_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
            recent_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
            recent_liab = balance_sheet.loc['Total Liab'].iloc[0] if 'Total Liab' in balance_sheet.index else 0
            
            # Calculate additional ratios
            fundamentals['asset_turnover'] = recent_revenue / recent_assets if recent_assets else 0
            fundamentals['equity_multiplier'] = recent_assets / recent_equity if recent_equity else 0
            fundamentals['debt_ratio'] = recent_liab / recent_assets if recent_assets else 0
            
            # Get debt from balance sheet
            fundamentals['total_debt'] = balance_sheet.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance_sheet.index else 0
            fundamentals['total_debt'] += balance_sheet.loc['Short Long Term Debt'].iloc[0] if 'Short Long Term Debt' in balance_sheet.index else 0
            
            # ROIC calculation
            ebit = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else recent_net_income
            invested_capital = recent_equity + fundamentals['total_debt']
            fundamentals['roic'] = (ebit * (1 - 0.25)) / invested_capital * 100 if invested_capital else 0  # Assuming 25% tax rate
            
            # Add quarterly data trends if available
            if not quarterly_income.empty and 'Total Revenue' in quarterly_income.index:
                recent_quarters = quarterly_income.loc['Total Revenue'].iloc[:4].tolist()
                fundamentals['quarterly_revenue_trend'] = recent_quarters
                
                # Calculate quarter-over-quarter growth
                if len(recent_quarters) > 1:
                    qoq_growth = [(recent_quarters[i] / recent_quarters[i+1] - 1) * 100 
                                for i in range(min(3, len(recent_quarters)-1))]
                    fundamentals['revenue_qoq_growth'] = qoq_growth
    
    return fundamentals

# Cache sector averages to avoid recalculating
@lru_cache(maxsize=32)
def _get_sector_averages(sector: str) -> Dict[str, float]:
    """
    Get average valuation metrics for a sector
    Used for relative valuation calculations
    
    Args:
        sector: Sector name
    
    Returns:
        Dictionary with average metrics for the sector
    """
    # This would ideally come from a database or API
    # For now, using hardcoded representative values for Indian markets
    sector_data = {
        'Information Technology': {
            'pe_ratio': 25.0,
            'price_to_book': 5.2,
            'price_to_sales': 3.5,
            'dividend_yield': 1.5
        },
        'Financial Services': {
            'pe_ratio': 18.0,
            'price_to_book': 2.5,
            'price_to_sales': 3.0,
            'dividend_yield': 1.8
        },
        'Consumer Goods': {
            'pe_ratio': 40.0,
            'price_to_book': 10.0,
            'price_to_sales': 3.8,
            'dividend_yield': 1.0
        },
        'Automobile': {
            'pe_ratio': 22.0,
            'price_to_book': 3.5,
            'price_to_sales': 1.5,
            'dividend_yield': 1.2
        },
        'Pharmaceuticals': {
            'pe_ratio': 30.0,
            'price_to_book': 4.5,
            'price_to_sales': 3.0,
            'dividend_yield': 0.8
        },
        'Energy': {
            'pe_ratio': 12.0,
            'price_to_book': 1.8,
            'price_to_sales': 1.0,
            'dividend_yield': 4.0
        },
        'Metals': {
            'pe_ratio': 10.0,
            'price_to_book': 1.5,
            'price_to_sales': 0.8,
            'dividend_yield': 3.0
        }
    }
    
    # Return sector data if available, otherwise average of all sectors
    if sector in sector_data:
        return sector_data[sector]
    else:
        # Calculate average across all sectors
        all_sectors = list(sector_data.values())
        avg_data = {}
        for metric in ['pe_ratio', 'price_to_book', 'price_to_sales', 'dividend_yield']:
            avg_data[metric] = sum(s.get(metric, 0) for s in all_sectors) / len(all_sectors)
        return avg_data

def get_news_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Get enhanced news sentiment for a stock with NLP-based analysis
    
    Args:
        symbol: Stock symbol with .NS suffix
        
    Returns:
        Dictionary with sentiment scores, news items, and trend analysis
    """
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return {
                'sentiment_score': 0, 
                'article_count': 0, 
                'recent_sentiment': 'neutral',
                'sentiment_trend': 'stable',
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            }
        
        # Positive and negative keyword lists (expanded for Indian market context)
        positive_keywords = [
            # Standard positive terms
            'rise', 'gain', 'grow', 'up', 'bull', 'outperform', 'beat', 'strong', 'positive',
            'surge', 'jump', 'soar', 'rally', 'exceed', 'boost', 'upgrade', 'buy', 'overweight',
            'recommend', 'profit', 'improve', 'success', 'promising', 'upside', 'opportunity',
            # India-specific positive terms
            'disinvestment', 'FDI', 'reform', 'digital', 'infrastructure', 'dividend', 'bonus',
            'export', 'innovation', 'unlock', 'expansion', 'partnership', 'merger'
        ]
        
        negative_keywords = [
            # Standard negative terms
            'fall', 'drop', 'decline', 'down', 'bear', 'underperform', 'miss', 'weak', 'negative',
            'plunge', 'tumble', 'sink', 'selloff', 'downgrade', 'sell', 'underweight', 'avoid',
            'loss', 'concern', 'risk', 'warning', 'trouble', 'dispute', 'downside', 'threat', 
            # India-specific negative terms
            'SEBI probe', 'investigation', 'promoter', 'pledge', 'debt', 'default', 'bankruptcy',
            'audit', 'tax', 'penalty', 'fine', 'litigation', 'strike', 'protest', 'COVID', 'lockdown'
        ]
        
        # Sentiment analysis for each article
        sentiment_scores = []
        dates = []
        all_articles = []
        sentiment_distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for article in news:
            # Extract title and publication date
            title = article.get('title', '').lower()
            pub_date = article.get('providerPublishTime', 0)
            
            # Convert timestamp to datetime
            if pub_date:
                pub_date = datetime.datetime.fromtimestamp(pub_date)
                dates.append(pub_date)
            
            # Calculate sentiment score
            score = 0
            
            # Check for positive/negative keywords
            for word in positive_keywords:
                if word in title:
                    score += 1
            
            for word in negative_keywords:
                if word in title:
                    score -= 1
            
            # Apply modifiers (e.g., "not good" should be negative)
            negation_words = ['not', 'no', 'never', 'neither', 'nor', 'without']
            for neg in negation_words:
                if neg in title:
                    # Find the next sentiment word after negation
                    neg_idx = title.find(neg)
                    rest_of_title = title[neg_idx:]
                    
                    # Check if any positive words follow the negation
                    for pos in positive_keywords:
                        if pos in rest_of_title:
                            score -= 2  # Flip the sentiment and strengthen it
                            break
            
            # Normalize score to range [-1, 1]
            if score > 0:
                normalized_score = min(score / 3, 1.0)  # Cap at 1.0
                sentiment_distribution['positive'] += 1
            elif score < 0:
                normalized_score = max(score / 3, -1.0)  # Cap at -1.0
                sentiment_distribution['negative'] += 1
            else:
                normalized_score = 0
                sentiment_distribution['neutral'] += 1
                
            sentiment_scores.append(normalized_score)
            
            # Store article data
            all_articles.append({
                'title': article.get('title', ''),
                'date': pub_date.isoformat() if pub_date else None,
                'source': article.get('publisher', ''),
                'url': article.get('link', ''),
                'sentiment': normalized_score
            })
        
        # Get overall sentiment metrics
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        if avg_sentiment > 0.2:
            sentiment = 'positive'
        elif avg_sentiment < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Analyze sentiment trend if we have multiple articles with dates
        sentiment_trend = 'stable'
        if len(sentiment_scores) > 1 and len(dates) > 1:
            # Sort articles by date
            sorted_scores = [score for _, score in sorted(zip(dates, sentiment_scores))]
            
            # Split into two halves (older vs newer)
            half = len(sorted_scores) // 2
            older_avg = np.mean(sorted_scores[:half]) if half > 0 else 0
            newer_avg = np.mean(sorted_scores[half:]) if half < len(sorted_scores) else 0
            
            # Determine trend
            diff = newer_avg - older_avg
            if diff > 0.3:
                sentiment_trend = 'improving'
            elif diff < -0.3:
                sentiment_trend = 'deteriorating'
            else:
                sentiment_trend = 'stable'
        
        # Calculate distribution percentages
        total_articles = len(sentiment_scores) if sentiment_scores else 1
        for key in sentiment_distribution:
            sentiment_distribution[key] = round(sentiment_distribution[key] * 100 / total_articles, 1)
        
        # Return comprehensive sentiment data
        return {
            'sentiment_score': round(avg_sentiment, 2),
            'article_count': len(news),
            'recent_sentiment': sentiment,
            'sentiment_trend': sentiment_trend,
            'sentiment_distribution': sentiment_distribution,
            'recent_articles': all_articles[:5] if all_articles else []
        }
    
    except Exception as e:
        logger.error(f"Error getting news sentiment for {symbol}: {e}")
        return {
            'sentiment_score': 0, 
            'article_count': 0, 
            'recent_sentiment': 'neutral',
            'sentiment_trend': 'stable',
            'error': str(e)
        }

def get_intermarket_correlations(symbol: str, symbol_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive correlations with global markets, commodities, and fixed income
    
    Args:
        symbol: Stock symbol with .NS suffix
        symbol_data: DataFrame with stock price data
    
    Returns:
        Dictionary with correlation values, including:
        - Global index correlations
        - Commodity correlations
        - Bond/yield correlations
        - FX correlations
        - Correlation trends and stability metrics
    """
    correlations = {
        'indices': {},
        'commodities': {},
        'bonds_fx': {},
        'correlation_stability': {},
        'macro_regime_correlations': {}
    }
    
    try:
        # Define period to match the symbol data
        start_date = symbol_data.index[0]
        end_date = symbol_data.index[-1]
        
        # First get all the external data at once
        reference_symbols = list(GLOBAL_INDICES.keys()) + list(COMMODITIES_BONDS.keys())
        
        # Download data for all reference assets
        all_ref_data = {}
        
        logger.info(f"Downloading reference data for intermarket analysis of {symbol}")
        
        for ref_chunk in [reference_symbols[i:i+10] for i in range(0, len(reference_symbols), 10)]:
            try:
                chunk_data = yf.download(
                    ref_chunk, 
                    start=start_date, 
                    end=end_date, 
                    progress=False,
                    group_by='ticker'
                )
                
                # Process each symbol in the chunk
                for ref_symbol in ref_chunk:
                    if ref_symbol in chunk_data:
                        # For each ticker, extract the Close prices
                        if isinstance(chunk_data[ref_symbol], pd.DataFrame) and 'Close' in chunk_data[ref_symbol]:
                            all_ref_data[ref_symbol] = chunk_data[ref_symbol]['Close']
                
                # Sleep to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error downloading chunk of reference data: {e}")
        
        # Get symbol's close prices
        symbol_close = symbol_data['Close']
        
        # Calculate correlations for different reference types
        for ref_symbol, ref_name in GLOBAL_INDICES.items():
            if ref_symbol in all_ref_data:
                correlations['indices'].update(
                    _calculate_correlation_metrics(symbol_close, all_ref_data[ref_symbol], ref_name)
                )
        
        for ref_symbol, ref_name in COMMODITIES_BONDS.items():
            if ref_symbol in all_ref_data:
                if 'Gold' in ref_name or 'Silver' in ref_name or 'Oil' in ref_name:
                    correlations['commodities'].update(
                        _calculate_correlation_metrics(symbol_close, all_ref_data[ref_symbol], ref_name)
                    )
                else:
                    correlations['bonds_fx'].update(
                        _calculate_correlation_metrics(symbol_close, all_ref_data[ref_symbol], ref_name)
                    )
        
        # Calculate correlation stability metrics
        correlations['correlation_stability'] = _calculate_correlation_stability(
            symbol_close, 
            {k: all_ref_data[k] for k in ['^NSEI', '^GSPC', 'GC=F', '^TNX'] if k in all_ref_data}
        )
        
        # Calculate correlations under different market regimes
        correlations['macro_regime_correlations'] = _calculate_regime_correlations(
            symbol_close,
            {k: all_ref_data[k] for k in ['^NSEI', '^VIX', 'GC=F'] if k in all_ref_data}
        )
        
    except Exception as e:
        logger.error(f"Error calculating intermarket correlations for {symbol}: {e}")
    
    return correlations

def _calculate_correlation_metrics(
    series1: pd.Series, 
    series2: pd.Series, 
    name: str
) -> Dict[str, float]:
    """
    Calculate multiple correlation metrics between two time series
    
    Args:
        series1: First time series (stock price)
        series2: Second time series (reference asset)
        name: Name of the reference asset
    
    Returns:
        Dictionary with correlation values for different time periods
    """
    result = {}
    
    # Align dates
    aligned_data = pd.concat([series1, series2], axis=1, join='inner')
    aligned_data.columns = ['stock', 'reference']
    
    if len(aligned_data) <= 5:  # Need at least a few data points
        return {f"{name}_corr": None}
    
    # Calculate return series for both assets
    aligned_data['stock_ret'] = aligned_data['stock'].pct_change()
    aligned_data['reference_ret'] = aligned_data['reference'].pct_change()
    
    # Drop NaN values
    aligned_data = aligned_data.dropna()
    
    if len(aligned_data) <= 5:  # Check again after dropping NaNs
        return {f"{name}_corr": None}
    
    # Full period price correlation
    result[f"{name}_price_corr"] = aligned_data[['stock', 'reference']].corr().iloc[0, 1]
    
    # Full period return correlation
    result[f"{name}_ret_corr"] = aligned_data[['stock_ret', 'reference_ret']].corr().iloc[0, 1]
    
    # Recent period correlations (if enough data)
    if len(aligned_data) >= 30:
        recent_data = aligned_data.tail(30)
        result[f"{name}_recent_price_corr"] = recent_data[['stock', 'reference']].corr().iloc[0, 1]
        result[f"{name}_recent_ret_corr"] = recent_data[['stock_ret', 'reference_ret']].corr().iloc[0, 1]
    
    # Longer-term correlations
    if len(aligned_data) >= 90:
        quarter_data = aligned_data.tail(90)
        result[f"{name}_quarter_ret_corr"] = quarter_data[['stock_ret', 'reference_ret']].corr().iloc[0, 1]
    
    # Lagged correlations (does reference asset lead the stock?)
    for lag in [1, 5, 10]:
        if len(aligned_data) > lag + 10:
            aligned_data[f'reference_ret_lag{lag}'] = aligned_data['reference_ret'].shift(lag)
            lag_corr = aligned_data[['stock_ret', f'reference_ret_lag{lag}']].dropna().corr().iloc[0, 1]
            result[f"{name}_lead{lag}_corr"] = lag_corr
    
    return result

def _calculate_correlation_stability(
    stock_series: pd.Series, 
    ref_data: Dict[str, pd.Series]
) -> Dict[str, float]:
    """
    Calculate correlation stability metrics
    
    Args:
        stock_series: Stock price series
        ref_data: Dictionary of reference asset price series
    
    Returns:
        Dictionary with correlation stability metrics
    """
    result = {}
    
    if not ref_data or len(stock_series) < 126:  # Need at least 6 months of data
        return {'correlation_stability_score': None}
        
        # Use Nifty 50 as primary reference if available
        ref_key = '^NSEI' if '^NSEI' in ref_data else list(ref_data.keys())[0]
        ref_series = ref_data[ref_key]
        
        # Align dates
        aligned_data = pd.concat([stock_series, ref_series], axis=1, join='inner')
        aligned_data.columns = ['stock', 'reference']
        
        if len(aligned_data) < 126:
            return {'correlation_stability_score': None}
        
        # Calculate rolling correlations
        aligned_data['stock_ret'] = aligned_data['stock'].pct_change()
        aligned_data['reference_ret'] = aligned_data['reference'].pct_change()
        aligned_data = aligned_data.dropna()
        
        roll_corr = aligned_data['stock_ret'].rolling(30).corr(aligned_data['reference_ret'])
        
        # Stability is measured by the standard deviation of rolling correlations
        # Lower values mean more stable correlations
        corr_stability = roll_corr.std()
        
        # Normalize to a 0-100 scale where 100 means perfectly stable
        corr_stability_score = 100 * (1 - min(corr_stability, 1))
        
        result['correlation_stability_score'] = corr_stability_score
        result['correlation_mean'] = roll_corr.mean()
        result['correlation_min'] = roll_corr.min()
        result['correlation_max'] = roll_corr.max()
        
    return result

def _calculate_regime_correlations(
    stock_series: pd.Series, 
    ref_data: Dict[str, pd.Series]
) -> Dict[str, float]:
    """
    Calculate correlations under different market regimes
    
    Args:
        stock_series: Stock price series
        ref_data: Dictionary of reference asset price series
    
    Returns:
        Dictionary with regime-specific correlations
    """
    result = {}
    
    if not ref_data or '^NSEI' not in ref_data or len(stock_series) < 126:
        return {'regime_correlation_available': False}
        
        # Get market and VIX data
        market_series = ref_data['^NSEI']
        vix_series = ref_data.get('^VIX')
        
        # Align dates
        data = pd.concat([stock_series, market_series], axis=1, join='inner')
        data.columns = ['stock', 'market']
        
        # Calculate returns
        data['stock_ret'] = data['stock'].pct_change()
        data['market_ret'] = data['market'].pct_change()
        
        # Add VIX if available
        if vix_series is not None:
            data = pd.concat([data, vix_series], axis=1, join='inner')
            data.columns = ['stock', 'market', 'vix', 'stock_ret', 'market_ret']
        
        data = data.dropna()
        
        if len(data) < 30:
            return {'regime_correlation_available': False}
        
        # Identify bull/bear regimes (trailing 30-day returns)
        data['market_trend'] = data['market'].pct_change(30)
        data['bull_market'] = data['market_trend'] > 0
        data['bear_market'] = data['market_trend'] < 0
        
        # Identify high/low volatility regimes if VIX is available
        if 'vix' in data.columns:
            data['high_vol'] = data['vix'] > data['vix'].quantile(0.7)
            data['low_vol'] = data['vix'] < data['vix'].quantile(0.3)
            
            # Calculate regime-specific correlations
            bull_high_vol = data[data['bull_market'] & data['high_vol']]
            bull_low_vol = data[data['bull_market'] & data['low_vol']]
            bear_high_vol = data[data['bear_market'] & data['high_vol']]
            bear_low_vol = data[data['bear_market'] & data['low_vol']]
            
            # Calculate correlations if enough data points
            if len(bull_high_vol) > 10:
                result['bull_high_vol_corr'] = bull_high_vol[['stock_ret', 'market_ret']].corr().iloc[0, 1]
                
            if len(bull_low_vol) > 10:
                result['bull_low_vol_corr'] = bull_low_vol[['stock_ret', 'market_ret']].corr().iloc[0, 1]
                
            if len(bear_high_vol) > 10:
                result['bear_high_vol_corr'] = bear_high_vol[['stock_ret', 'market_ret']].corr().iloc[0, 1]
                
            if len(bear_low_vol) > 10:
                result['bear_low_vol_corr'] = bear_low_vol[['stock_ret', 'market_ret']].corr().iloc[0, 1]
        
        # Bull/bear correlations regardless of volatility
        bull_market = data[data['bull_market']]
        bear_market = data[data['bear_market']]
        
        if len(bull_market) > 10:
            result['bull_market_corr'] = bull_market[['stock_ret', 'market_ret']].corr().iloc[0, 1]
            
        if len(bear_market) > 10:
            result['bear_market_corr'] = bear_market[['stock_ret', 'market_ret']].corr().iloc[0, 1]
        
        result['regime_correlation_available'] = True
    
    return result

def get_relative_strength(symbol: str, symbol_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive relative strength metrics
    
    Args:
        symbol: Stock symbol with .NS suffix
        symbol_data: DataFrame with stock price data
    
    Returns:
        Dictionary with relative strength metrics, including:
        - RS vs market indices
        - RS vs sector
        - RS momentum and trends
        - RS percentile rankings
    """
    rs_metrics = {
        'vs_market': {},
        'vs_sector': {},
        'rs_momentum': {},
        'rs_ranks': {}
    }
    
    # Use symbol data close price
    if symbol_data.empty or 'Close' not in symbol_data.columns:
        return {'error': 'No price data available'}
            
        stock_close = symbol_data['Close']
        start_date = stock_close.index[0]
        end_date = stock_close.index[-1]
        
        # Get sector for the symbol
        sector = SECTOR_MAPPING.get(symbol, None)
        
        # Download Nifty and sector index data
        index_symbols = ['^NSEI']  # Nifty 50
        
        # Add sector index if available
        sector_index = None
        if sector:
            sector_map = {
                'FINANCIAL SERVICES': '^CNXFINANCE',
                'IT': '^CNXIT',
                'PHARMA': '^CNXPHARMA',
                'HEALTHCARE': '^CNXPHARMA',
                'AUTOMOBILE': '^CNXAUTO',
                'CONSUMER GOODS': '^CNXFMCG',
                'METALS': '^CNXMETAL',
                'ENERGY': '^CNXENERGY',
                'CONSTRUCTION': '^CNXREALTY',
                'REALTY': '^CNXREALTY'
            }
            
            # Try to match sector to an index - ensure sector is a string or handled properly
            sector_str = None
            
            if isinstance(sector, str):
                sector_str = sector.upper()
            elif isinstance(sector, (list, tuple)) and len(sector) > 0:
                if isinstance(sector[0], str):
                    sector_str = sector[0].upper()
                else:
                    sector_str = str(sector[0]).upper()
            elif isinstance(sector, dict) and 'sector' in sector:
                sector_str = str(sector['sector']).upper()
            else:
                # If sector is None or some other unexpected type, use a default string
                logger.warning(f"Unexpected sector type for {symbol}: {type(sector)}")
                sector_str = "UNKNOWN"
                
            # Try to match the sector string to a sector index
            if sector_str:
                for sector_key, index_symbol in sector_map.items():
                    if sector_key in sector_str:
                        sector_index = index_symbol
                        index_symbols.append(sector_index)
                        break
        
        # Download index data
        index_data = {}
        for idx_symbol in index_symbols:
            if 1==1:
                data = yf.download(
                    idx_symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if not data.empty and 'Close' in data.columns:
                    index_data[idx_symbol] = data['Close']
        
        # Calculate relative strength vs Nifty
        if '^NSEI' in index_data:
            nifty_close = index_data['^NSEI']
            
            # Align dates
            aligned_data = pd.concat([stock_close, nifty_close], axis=1, join='inner')
            aligned_data.columns = ['stock', 'nifty']
            
            if not aligned_data.empty:
                # Normalize to starting values
                aligned_data['stock_norm'] = aligned_data['stock'] / aligned_data['stock'].iloc[0]
                aligned_data['nifty_norm'] = aligned_data['nifty'] / aligned_data['nifty'].iloc[0]
                
                # Calculate RS line (stock/index)
                aligned_data['rs_line'] = aligned_data['stock_norm'] / aligned_data['nifty_norm']
                
                # Calculate RS metrics for different timeframes
                for days in [5, 10, 21, 63, 126, 252]:
                    if len(aligned_data) >= days:
                        # RS over the period
                        rs_period = aligned_data['rs_line'].iloc[-1] / aligned_data['rs_line'].iloc[-min(days, len(aligned_data))]
                        rs_metrics['vs_market'][f'rs_{days}d'] = rs_period
                
                # RS new highs/lows
                if len(aligned_data) >= 20:
                    rs_new_high_20d = aligned_data['rs_line'].iloc[-1] >= aligned_data['rs_line'].tail(20).max()
                    rs_new_low_20d = aligned_data['rs_line'].iloc[-1] <= aligned_data['rs_line'].tail(20).min()
                    rs_metrics['vs_market']['rs_new_high_20d'] = rs_new_high_20d
                    rs_metrics['vs_market']['rs_new_low_20d'] = rs_new_low_20d
                
                # RS slope
                if len(aligned_data) >= 10:
                    rs_slope = np.polyfit(range(10), aligned_data['rs_line'].tail(10).values, 1)[0]
                    rs_metrics['vs_market']['rs_slope_10d'] = rs_slope
                
                # Add current RS value
                rs_metrics['vs_market']['rs_current'] = aligned_data['rs_line'].iloc[-1]
                
                # Add RS momentum (rate of change)
                if len(aligned_data) >= 14:
                    rs_metrics['rs_momentum']['rs_roc_14d'] = (
                        aligned_data['rs_line'].pct_change(periods=14).iloc[-1] * 100
                    )
        
        # Calculate relative strength vs sector
        if sector_index and sector_index in index_data:
            sector_close = index_data[sector_index]
            
            # Align dates
            sector_aligned = pd.concat([stock_close, sector_close], axis=1, join='inner')
            sector_aligned.columns = ['stock', 'sector']
            
            if not sector_aligned.empty:
                # Normalize to starting values
                sector_aligned['stock_norm'] = sector_aligned['stock'] / sector_aligned['stock'].iloc[0]
                sector_aligned['sector_norm'] = sector_aligned['sector'] / sector_aligned['sector'].iloc[0]
                
                # Calculate RS line (stock/sector)
                sector_aligned['rs_sector'] = sector_aligned['stock_norm'] / sector_aligned['sector_norm']
                
                # Calculate RS metrics for different timeframes
                for days in [5, 10, 21, 63, 126, 252]:
                    if len(sector_aligned) >= days:
                        # RS over the period
                        rs_period = sector_aligned['rs_sector'].iloc[-1] / sector_aligned['rs_sector'].iloc[-min(days, len(sector_aligned))]
                        rs_metrics['vs_sector'][f'rs_sector_{days}d'] = rs_period
                
                # Add current RS value
                rs_metrics['vs_sector']['rs_sector_current'] = sector_aligned['rs_sector'].iloc[-1]
        
        # Calculate RS ranks (percentile ranking against Nifty 500)
        # This would typically be done with a database of all stocks
        # Here we approximate using the information we have
        rs_metrics['rs_ranks']['rs_rank_vs_market'] = np.random.randint(1, 100)  # Placeholder
        rs_metrics['rs_ranks']['rs_rank_vs_sector'] = np.random.randint(1, 100)  # Placeholder
        
    return rs_metrics
    try:
        # Download data
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return None
            
        # Add symbol
        df['Symbol'] = symbol
        
        # 1. Trend Indicators
        # Moving Averages and Combinations
        for period in [5, 8, 10, 13, 20, 30, 50, 100, 200]:
            df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            df[f'HMA{period}'] = df['Close'].rolling(window=period//2).mean() * 2 - df['Close'].rolling(window=period).mean()
            
        # Moving Average Crosses
        df['Golden_Cross'] = (df['SMA50'] > df['SMA200']).astype(int)
        df['Death_Cross'] = (df['SMA50'] < df['SMA200']).astype(int)
        
        # MACD and Variants
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Pct'] = df['MACD'] / df['Close'] * 100
        
        # ADX and DMI Components
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr14 = true_range.rolling(14).sum()
        df['DI_Plus'] = 100 * (plus_dm.rolling(14).sum() / tr14)
        df['DI_Minus'] = abs(100 * (minus_dm.rolling(14).sum() / tr14))
        df['ADX'] = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX_Trend'] = np.where(df['ADX'] > 25, 'Strong', 'Weak')
        
        # Ichimoku Cloud
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Ichimoku_Base'] = (high_26 + low_26) / 2
        
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
        
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
        
        df['Ichimoku_Lagging'] = df['Close'].shift(-26)
        
        # Parabolic SAR
        periods = len(df)
        acceleration = 0.02
        maximum = 0.2
        sar = df['Close'].copy()
        direction = 1
        ep = df['Low'][0]
        af = acceleration
        
        for i in range(2, periods):
            temp_sar = sar[i-1] + af * (ep - sar[i-1])
            if direction == 1:
                if temp_sar > df['Low'].iloc[i]:
                    direction = -1
                    temp_sar = max(df['High'].iloc[i-2:i+1])
                    ep = df['Low'].iloc[i]
                    af = acceleration
            else:
                if temp_sar < df['High'].iloc[i]:
                    direction = 1
                    temp_sar = min(df['Low'].iloc[i-2:i+1])
                    ep = df['High'].iloc[i]
                    af = acceleration
            
            if direction == 1:
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]
                    af = min(af + acceleration, maximum)
            else:
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]
                    af = min(af + acceleration, maximum)
                    
            sar[i] = temp_sar
            
        df['PSAR'] = sar
        df['PSAR_Direction'] = np.where(df['PSAR'] < df['Close'], 1, -1)
        
        # 2. Momentum Indicators
        # RSI and Variants
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        # Standard RSI
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic RSI
        rsi = df['RSI']
        df['Stoch_RSI'] = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        df['Stoch_RSI_K'] = df['Stoch_RSI'].rolling(3).mean() * 100
        df['Stoch_RSI_D'] = df['Stoch_RSI_K'].rolling(3).mean()
        
        # Multi-timeframe RSI
        for period in [2, 5, 14, 21]:
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator with Multiple Timeframes
        for period in [14, 21, 50]:
            low_p = df['Low'].rolling(window=period).min()
            high_p = df['High'].rolling(window=period).max()
            df[f'STOCH_K_{period}'] = 100 * (df['Close'] - low_p) / (high_p - low_p)
            df[f'STOCH_D_{period}'] = df[f'STOCH_K_{period}'].rolling(window=3).mean()
            
        # Rate of Change (ROC)
        for period in [5, 10, 20, 50]:
            df[f'ROC_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
        
        # CCI with Multiple Timeframes
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        for period in [14, 20, 50]:
            ma_tp = tp.rolling(window=period).mean()
            md_tp = (abs(tp - ma_tp)).rolling(window=period).mean()
            df[f'CCI_{period}'] = (tp - ma_tp) / (0.015 * md_tp)
            
        # Williams %R
        for period in [14, 28]:
            highest_high = df['High'].rolling(window=period).max()
            lowest_low = df['Low'].rolling(window=period).min()
            df[f'WILLR_{period}'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
        
        # 3. Volatility Indicators
        # Bollinger Bands with Multiple Timeframes/Deviations
        for period in [10, 20, 50]:
            for std_dev in [2.0, 2.5, 3.0]:
                bb_ma = df['Close'].rolling(window=period).mean()
                bb_std = df['Close'].rolling(window=period).std()
                df[f'BB_upper_{period}_{int(std_dev*10)}'] = bb_ma + (bb_std * std_dev)
                df[f'BB_middle_{period}'] = bb_ma
                df[f'BB_lower_{period}_{int(std_dev*10)}'] = bb_ma - (bb_std * std_dev)
                df[f'BB_width_{period}'] = (df[f'BB_upper_{period}_{int(std_dev*10)}'] - 
                                          df[f'BB_lower_{period}_{int(std_dev*10)}']) / df[f'BB_middle_{period}']
                df[f'BB_%B_{period}'] = (df['Close'] - df[f'BB_lower_{period}_{int(std_dev*10)}']) / \
                                      (df[f'BB_upper_{period}_{int(std_dev*10)}'] - df[f'BB_lower_{period}_{int(std_dev*10)}'])
        
        # Keltner Channels with Multiple Timeframes
        for period in [10, 20, 50]:
            df[f'KC_middle_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'KC_upper_{period}'] = df[f'KC_middle_{period}'] + (df['ATR'] * 2)
            df[f'KC_lower_{period}'] = df[f'KC_middle_{period}'] - (df['ATR'] * 2)
            df[f'KC_width_{period}'] = (df[f'KC_upper_{period}'] - df[f'KC_lower_{period}']) / df[f'KC_middle_{period}']
            
        # Volatility Ratios and Regimes
        df['Historical_Vol'] = df['Close'].pct_change().rolling(window=21).std() * np.sqrt(252)
        df['Volatility_MA'] = df['Historical_Vol'].rolling(window=50).mean()
        df['Volatility_Regime'] = np.where(df['Historical_Vol'] > df['Volatility_MA'] * 1.2, 'High',
                                         np.where(df['Historical_Vol'] < df['Volatility_MA'] * 0.8, 'Low', 'Normal'))
        
        # Squeeze Momentum Indicator
        bb_ma = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        bb_upper = bb_ma + (bb_std * 2)
        bb_lower = bb_ma - (bb_std * 2)
        
        kc_ma = df['Close'].rolling(window=20).mean()
        kc_range = df['ATR'].rolling(window=20).mean()
        kc_upper = kc_ma + (kc_range * 1.5)
        kc_lower = kc_ma - (kc_range * 1.5)
        
        df['Squeeze_On'] = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        df['Squeeze_Off'] = ~df['Squeeze_On']
        
        # 4. Volume Indicators and Analysis
        # OBV and Variants
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
        df['OBV_ROC'] = df['OBV'].pct_change(periods=5) * 100
        
        # VWAP and Variants
        for period in [1, 5, 21]:  # Daily, Weekly, Monthly
            df[f'VWAP_{period}'] = (df['Close'] * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
            
        # Volume Profile
        price_buckets = pd.qcut(df['Close'], q=10, labels=False)
        volume_profile = df.groupby(price_buckets)['Volume'].sum()
        df['Volume_POC'] = price_buckets.map(volume_profile.idxmax())  # Point of Control
        
        # Volume Ratios and Analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_ROC'] = df['Volume'].pct_change(periods=1) * 100
        df['Volume_Trend'] = np.where(df['Volume_Ratio'] > 1.5, 'High',
                                    np.where(df['Volume_Ratio'] < 0.5, 'Low', 'Normal'))
        
        # Money Flow Index with Multiple Timeframes
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        for period in [14, 28, 50]:
            pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
            neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
            mf_ratio = pos_flow / neg_flow
            df[f'MFI_{period}'] = 100 - (100 / (1 + mf_ratio))
        
        # Enhanced Chaikin Money Flow
        mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mf_volume = mf_multiplier * df['Volume']
        for period in [10, 20, 50]:
            df[f'CMF_{period}'] = mf_volume.rolling(period).sum() / df['Volume'].rolling(period).sum()
            
        # Volume-Price Correlation
        df['Vol_Price_Corr'] = df['Close'].rolling(20).corr(df['Volume'])
        
        # Volume Zone Oscillator
        ma_volume = df['Volume'].rolling(window=5).mean()
        vol_z_score = (df['Volume'] - ma_volume) / df['Volume'].rolling(window=5).std()
        df['VZO'] = vol_z_score.rolling(window=5).mean()
        
        # Accumulation/Distribution Line
        df['ADL'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
        df['ADL'] = df['ADL'].cumsum()
        
        # 5. Market Regime and Pattern Recognition
        # Hurst Exponent (Trend vs Mean Reversion)
        def hurst(ts, lags=range(2, 100)):
            tau = []; lagvec = []
            for lag in lags:
                tau.append(np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))))
                lagvec.append(lag)
            m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
            hurst = m[0] if not np.isnan(m[0]) else 0.5
            return hurst

        df['Hurst'] = df['Close'].rolling(window=100).apply(
            lambda x: hurst(x.values), raw=False
        )
        df['Market_Type'] = np.where(df['Hurst'] > 0.6, 'Trending',
                                   np.where(df['Hurst'] < 0.4, 'Mean_Reverting', 'Random'))
        
        # Cycle Indicators
        # Ehlers' Hilbert Transform Dominant Cycle Period
        smooth = (df['Close'] + 2*df['Close'].shift(1) + 2*df['Close'].shift(2) + df['Close'].shift(3))/6
        detrender = (0.0962*smooth + 0.5769*smooth.shift(2) - 0.5769*smooth.shift(4) - 0.0962*smooth.shift(6))*(0.075*df['Close'].pct_change().rolling(20).std())
        q1 = (0.0962*detrender + 0.5769*detrender.shift(2) - 0.5769*detrender.shift(4) - 0.0962*detrender.shift(6))*0.075
        i1 = detrender.shift(3)
        df['Cycle_Period'] = np.where((q1**2 + i1**2) > 0, 
                                    2*np.pi/np.arctan2(q1, i1).abs(), 
                                    0)
        
        # Mesa Sine Wave
        df['Mesa_Sine'] = np.sin(2*np.pi*np.arange(len(df))/df['Cycle_Period'])
        df['Mesa_LeadSine'] = np.sin(2*np.pi*np.arange(len(df))/df['Cycle_Period'] + np.pi/4)
        
        # Pattern Recognition
        # Support and Resistance
        def find_peaks(series, window=20):
            peaks = []
            for i in range(window, len(series)-window):
                if series.iloc[i] == max(series.iloc[i-window:i+window+1]):
                    peaks.append(series.iloc[i])
                elif series.iloc[i] == min(series.iloc[i-window:i+window+1]):
                    peaks.append(series.iloc[i])
            return peaks if peaks else [series.mean()]
            
        df['Support_Level'] = pd.Series(find_peaks(df['Low'])).rolling(20).min()
        df['Resistance_Level'] = pd.Series(find_peaks(df['High'])).rolling(20).max()
        
        # Gap Analysis
        df['Gap_Up'] = df['Low'] > df['High'].shift(1)
        df['Gap_Down'] = df['High'] < df['Low'].shift(1)
        df['Gap_Size'] = np.where(df['Gap_Up'], df['Low'] - df['High'].shift(1),
                                np.where(df['Gap_Down'], df['High'] - df['Low'].shift(1), 0))
        
        # Performance Metrics
        # Returns across timeframes
        for period in [1, 5, 10, 21, 63, 126, 252]:  # Daily to Annual
            df[f'Return_{period}d'] = df['Close'].pct_change(periods=period)
            df[f'Vol_{period}d'] = df[f'Return_{period}d'].rolling(window=period).std() * np.sqrt(252/period)
            
        # Risk Metrics
        # Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall)
        for conf_level in [0.95, 0.99]:
            df[f'VaR_{int(conf_level*100)}'] = df['Daily_Return'].rolling(window=252).quantile(1-conf_level)
            df[f'CVaR_{int(conf_level*100)}'] = df['Daily_Return'].rolling(window=252).apply(
                lambda x: x[x <= x.quantile(1-conf_level)].mean()
            )
            
        # Maximum Drawdown and Recovery Metrics
        df['Drawdown'] = (df['Close'] / df['Close'].cummax() - 1)
        df['Max_Drawdown'] = df['Drawdown'].rolling(window=252).min()
        df['Days_in_Drawdown'] = df['Drawdown'].apply(lambda x: 0 if x >= 0 else 1).rolling(252).sum()
        
        # Sortino Ratio
        risk_free_rate = 0.03/252  # Assuming 3% annual risk-free rate
        excess_returns = df['Daily_Return'] - risk_free_rate
        downside_returns = np.where(df['Daily_Return'] < 0, df['Daily_Return'], 0)
        df['Sortino_Ratio'] = (excess_returns.rolling(252).mean() * 252) / \
                             (np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252))
        
        # Clean up any missing values and ensure data quality
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # 6. Candlestick Patterns
        # Single Candlestick Patterns
        df['DOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['HAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['SPINNING_TOP'] = talib.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])

        # Double Candlestick Patterns
        df['ENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        df['HARAMI'] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
        df['PIERCING_LINE'] = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])
        df['DARK_CLOUD_COVER'] = talib.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])

        # Triple Candlestick Patterns
        df['MORNING_STAR'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['EVENING_STAR'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['THREE_WHITE_SOLDIERS'] = talib.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
        df['THREE_BLACK_CROWS'] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])

        # Pattern Strength Index (combined signal from all patterns)
        pattern_cols = ['DOJI', 'HAMMER', 'SHOOTING_STAR', 'SPINNING_TOP', 'ENGULFING', 
                       'HARAMI', 'PIERCING_LINE', 'DARK_CLOUD_COVER', 'MORNING_STAR', 
                       'EVENING_STAR', 'THREE_WHITE_SOLDIERS', 'THREE_BLACK_CROWS']
        df['Pattern_Bull_Count'] = df[pattern_cols].apply(lambda x: sum(x > 0), axis=1)
        df['Pattern_Bear_Count'] = df[pattern_cols].apply(lambda x: sum(x < 0), axis=1)
        df['Pattern_Strength'] = (df['Pattern_Bull_Count'] - df['Pattern_Bear_Count']) / len(pattern_cols)

        # 7. Cross-Asset and Multi-Timeframe Correlations
        # Rolling correlations between indicators
        indicator_pairs = [
            ('RSI', 'STOCH_K_14'),
            ('MACD', 'RSI'),
            ('OBV', 'Close'),
            ('Volume', 'Close'),
            ('ATR', 'Historical_Vol')
        ]
        
        for ind1, ind2 in indicator_pairs:
            if ind1 in df.columns and ind2 in df.columns:
                df[f'Corr_{ind1}_{ind2}'] = df[ind1].rolling(window=20).corr(df[ind2])

        # Multi-timeframe momentum correlations
        for period1, period2 in [(5, 20), (10, 50), (20, 100)]:
            if f'ROC_{period1}' in df.columns and f'ROC_{period2}' in df.columns:
                df[f'Mom_Corr_{period1}_{period2}'] = (
                    df[f'ROC_{period1}'].rolling(window=20).corr(df[f'ROC_{period2}'])
                )

        # Volume-Price Impact
        df['Vol_Price_Impact'] = (df['Close'] - df['Open']) * df['Volume_Ratio']
        df['Vol_Price_Efficiency'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
        
        # Correlation-based regime detection
        if 'RSI' in df.columns and 'MACD' in df.columns and 'Volume_Ratio' in df.columns:
            df['Indicator_Correlation'] = df[['RSI', 'MACD', 'Volume_Ratio']].rolling(20).corr().unstack().mean()
            df['Correlation_Regime'] = np.where(df['Indicator_Correlation'] > 0.7, 'High',
                                              np.where(df['Indicator_Correlation'] < 0.3, 'Low', 'Medium'))

        return df



    except Exception as e:
        logger.error(f"Error downloading and processing data for {symbol}: {e}")
        return None

def download_nifty500_data(start_date: str, end_date: str, output_dir: str = "data/historical", max_symbols: int = None):
    """
    Download Nifty 500 stock data in batches with sleep time between batches.
    Calculates technical indicators and performance metrics for each stock.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_dir: Directory to save the data files
        max_symbols: Maximum number of symbols to process (useful for testing)
        
    The function:
    1. Downloads historical OHLCV data
    2. Calculates technical indicators (SMA, EMA, RSI, MACD, etc.)
    3. Computes performance metrics (returns, volatility, etc.)
    4. Processes in batches of 50 stocks with 30s delay between batches
    5. Saves both individual batch files and a consolidated file
    """
    if 1==1:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get Nifty 500 symbols and sector data
        symbols, sector_map, industry_map = get_nifty500_symbols()
        
        # Limit symbols if max_symbols is specified
        if max_symbols is not None:
            symbols = symbols[:max_symbols]
            logger.info(f"Limited to processing {max_symbols} symbols for testing")
        
        # Process in batches with appropriate batch size and delay
        batch_size = 5  # Small batch size to avoid rate limiting
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        all_data = []
        
        logger.info(f"Processing {len(symbols)} symbols in {total_batches} batches of {batch_size}")
        
        # Set up batch parameters
        batch_delay = 60  # seconds between batches
        individual_delay = 5  # seconds between individual stocks
        retry_delay = 120  # seconds to wait before retry
        max_retries = 3   # maximum number of retries for a batch
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, len(symbols))
            batch_symbols = symbols[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({batch_start} to {batch_end})")
            
            # Implement retry mechanism for batch processing
            retry_count = 0
            batch_success = False
            
            while not batch_success and retry_count < max_retries:
                if 1==1:
                    # Download data for current batch
                    batch_data = []
                    failed_symbols = []
                    
                    for symbol in batch_symbols:
                        if 1==1:
                            df = download_stock_data(symbol, start_date, end_date)
                            if df is not None and not df.empty:
                                batch_data.append(df)
                                logger.info(f"Successfully downloaded data for {symbol}")
                            else:
                                logger.warning(f"No data returned for {symbol}")
                                failed_symbols.append(symbol)
                        
                        # Add a longer delay between individual downloads to avoid rate limiting
                        logger.info(f"Waiting {individual_delay} seconds before next symbol...")
                        time.sleep(individual_delay)
                    
                    # If we have data for at least half the symbols, consider the batch successful
                    if len(batch_data) >= len(batch_symbols) / 2:
                        batch_success = True
                        
                        # Log failed symbols
                        if failed_symbols:
                            logger.warning(f"Failed to download data for {len(failed_symbols)} symbols in batch {batch_num + 1}: {failed_symbols}")
                    else:
                        retry_count += 1
                       
                        logger.warning(f"Too many failures in batch {batch_num + 1}, retrying ({retry_count}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    
                    # Combine batch data
                    if batch_data:
                        batch_df = pd.concat(batch_data)
                        all_data.append(batch_df)
                        
                        # Save batch data
                        batch_file = output_path / f"nifty500_data_batch_{batch_num + 1}.parquet"
                        batch_df.to_parquet(batch_file)
                        logger.info(f"Saved batch {batch_num + 1} to {batch_file}")
            
            # If all retries failed, log error but continue with next batch
            if not batch_success:
                logger.error(f"Failed to process batch {batch_num + 1} after {max_retries} retries")
            
            if batch_num < total_batches - 1:
                logger.info(f"Batch {batch_num + 1} completed. Sleeping for {batch_delay} seconds before next batch...")
                time.sleep(batch_delay)
        
        # Combine all data and save
        if all_data:
            final_df = pd.concat(all_data)
            final_file = output_path / "nifty500_complete.parquet"
            final_df.to_parquet(final_file)
            logger.info(f"Successfully saved complete dataset to {final_file}")
            # Print summary info - use any unique identifier column that exists
            symbol_col = None
            for col in ['symbol', 'Symbol', 'Ticker']:
                if col in final_df.columns:
                    symbol_col = col
                    break
            if symbol_col:
                logger.info(f"Total records: {len(final_df)}, Total symbols: {len(final_df[symbol_col].unique())}")
            else:
                logger.info(f"Total records: {len(final_df)}")
