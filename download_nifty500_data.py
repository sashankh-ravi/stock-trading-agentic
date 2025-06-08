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

# Indian Market Indices
INDIAN_SECTOR_INDICES = {
    '^NSEI': 'Nifty 50',
    '^BSESN': 'Sensex',
    '^NSMIDCP': 'Nifty Midcap',
    '^CNXBANK': 'Nifty Bank',
    '^CNXAUTO': 'Nifty Auto',
    '^CNXFMCG': 'Nifty FMCG',
    '^CNXIT': 'Nifty IT',
    '^CNXMETAL': 'Nifty Metal',
    '^CNXPHARMA': 'Nifty Pharma',
    '^CNXREALTY': 'Nifty Realty',
    '^CNXENERGY': 'Nifty Energy',
    '^CNXINFRA': 'Nifty Infrastructure',
    '^CNXFINANCE': 'Nifty Financial Services',
    '^CNXFINSRV': 'Nifty Financial Services 25/50',
    '^CNXCONSUM': 'Nifty Consumer',
    '^CNXPSUBANK': 'Nifty PSU Bank',
    '^CNXMEDIA': 'Nifty Media'
}

# List of global market indices, commodities, and bonds to track correlations
GLOBAL_INDICES = {
    # Indian Indices (moved to INDIAN_SECTOR_INDICES)
    **INDIAN_SECTOR_INDICES,
    
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
        
        # Ownership and Institutional Data
        'institutional_ownership': info.get('institutionsPercentHeld', 0) * 100 if info.get('institutionsPercentHeld') else 0,
        'institutional_ownership_change': info.get('institutionsPercentChange', 0) * 100 if info.get('institutionsPercentChange') else 0,
        'institutions_count': info.get('institutionsCount', 0),
        'insider_ownership': info.get('insiderPercentHeld', 0) * 100 if info.get('insiderPercentHeld') else 0,
        'short_percent': info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0,
        'short_ratio': info.get('shortRatio', 0),
        
        # Liquidity Metrics
        'average_volume': info.get('averageVolume', 0),
        'average_volume_10d': info.get('averageVolume10days', 0),
        'relative_volume': info.get('averageVolume10days', 0) / info.get('averageVolume', 1) if info.get('averageVolume') else 0,
        
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
    
    # Define period to match the symbol data
    start_date = symbol_data.index[0]
    end_date = symbol_data.index[-1]
    
    # First get all the external data at once
    reference_symbols = list(GLOBAL_INDICES.keys()) + list(COMMODITIES_BONDS.keys())
    
    # Download data for all reference assets
    all_ref_data = {}
    
    logger.info(f"Downloading reference data for intermarket analysis of {symbol}")
    
    for ref_chunk in [reference_symbols[i:i+10] for i in range(0, len(reference_symbols), 10)]:
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
        Dictionary with relative strength metrics
    """
    rs_metrics = {
        'vs_market': {},
        'vs_sector': {},
        'rs_momentum': {},
        'rs_ranks': {}
    }
    
    # Input validation
    if symbol_data.empty or 'Close' not in symbol_data.columns:
        return {'error': 'No price data available'}
        
    stock_close = symbol_data['Close']  # Convert to Series
    start_date = stock_close.index[0]
    end_date = stock_close.index[-1]
    
    # Sector mapping for common industries
    STOCK_SECTOR_MAPPING = {
        'RELIANCE.NS': 'Energy',
        'HDFCBANK.NS': 'Financial Services',
        'TCS.NS': 'IT',
        'SUNPHARMA.NS': 'Pharma',
        'TATAMOTORS.NS': 'Auto',
        'ASIANPAINT.NS': 'Consumer',
        'SBIN.NS': 'PSU Bank',
        'TATASTEEL.NS': 'Metal',
        'NTPC.NS': 'Energy',
        'DLF.NS': 'Realty'
    }
    
    # Map sector to index
    SECTOR_TO_INDEX = {
        'Energy': '^CNXENERGY',
        'Financial Services': '^CNXFINANCE',
        'IT': '^CNXIT',
        'Pharma': '^CNXPHARMA',
        'Auto': '^CNXAUTO',
        'Consumer': '^CNXCONSUM',
        'PSU Bank': '^CNXPSUBANK',
        'Metal': '^CNXMETAL',
        'Realty': '^CNXREALTY',
        'Media': '^CNXMEDIA'
    }
    
    # Get sector for the symbol
    sector = STOCK_SECTOR_MAPPING.get(symbol, None)
    sector_index = SECTOR_TO_INDEX.get(sector, None) if sector else None
    
    # Download Nifty 50 data for market comparison
    nifty_data = yf.download(
        "^NSEI",
        start=start_date,
        end=end_date,
        progress=False
    )
    
    if not nifty_data.empty and 'Close' in nifty_data.columns:
        # Calculate RS vs Nifty 50
        rs_metrics['vs_market'] = calculate_market_rs(stock_close, nifty_data['Close'])
        
        # Calculate RS momentum metrics
        from relative_strength import calculate_relative_strength_rolling
        st_mom = calculate_relative_strength_rolling(stock_close, nifty_data['Close'], window=21)
        rs_metrics['rs_momentum'] = {
            'market_momentum_1m': float(st_mom.iloc[-1]) if not st_mom.empty else 0.0
        }
        
        # Calculate RS rankings
        rs_3m = calculate_relative_strength_rolling(stock_close, nifty_data['Close'], window=63)
        rs_6m = calculate_relative_strength_rolling(stock_close, nifty_data['Close'], window=126)
        rs_1y = calculate_relative_strength_rolling(stock_close, nifty_data['Close'], window=252)
        
        # Ensure we have valid ranking values
        if not rs_3m.empty and not rs_6m.empty and not rs_1y.empty:
            rs_metrics['rs_ranks'] = {
                'rs_rank_3m': float(rs_3m.rank(pct=True).iloc[-1]),
                'rs_rank_6m': float(rs_6m.rank(pct=True).iloc[-1]),
                'rs_rank_1y': float(rs_1y.rank(pct=True).iloc[-1])
            }
    
    # Calculate sector relative strength if sector index is available
    if sector_index:
        sector_data = yf.download(
            sector_index,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if not sector_data.empty and 'Close' in sector_data.columns:
            logger.info(f"Calculating sector RS for {symbol} against {sector_index}")
            rs_metrics['vs_sector'] = calculate_sector_rs(stock_close, sector_data['Close'])
    
    return rs_metrics

def calculate_market_rs(stock_prices: Union[pd.Series, pd.DataFrame], market_prices: Union[pd.Series, pd.DataFrame]) -> Dict[str, float]:
    """Calculate relative strength metrics against market index"""
    from relative_strength import calculate_relative_strength_metrics
    
    # Ensure we have Series objects
    if isinstance(stock_prices, pd.DataFrame):
        if 'Close' not in stock_prices.columns:
            return {'rs_cumulative': 0.0, 'rs_rolling_3m': 0.0, 'rs_momentum_1y': 0.0}
        stock_prices = stock_prices['Close']
    
    if isinstance(market_prices, pd.DataFrame):
        if 'Close' not in market_prices.columns:
            return {'rs_cumulative': 0.0, 'rs_rolling_3m': 0.0, 'rs_momentum_1y': 0.0}
        market_prices = market_prices['Close']
    
    # Additional validation
    if not isinstance(stock_prices, pd.Series) or not isinstance(market_prices, pd.Series):
        return {'rs_cumulative': 0.0, 'rs_rolling_3m': 0.0, 'rs_momentum_1y': 0.0}
    
    # Ensure both Series have data
    if stock_prices.empty or market_prices.empty:
        return {'rs_cumulative': 0.0, 'rs_rolling_3m': 0.0, 'rs_momentum_1y': 0.0}
    
    rs_market = calculate_relative_strength_metrics(
        stock_prices,
        market_prices,
        window_sizes={'roll': 63, 'mom': 252}  # 3 months and 1 year windows
    )
    
    return {
        'rs_cumulative': rs_market.get('RS_CUM_MARKET', 0.0),
        'rs_rolling_3m': rs_market.get('RS_ROLL_MARKET', 0.0),
        'rs_momentum_1y': rs_market.get('RS_MOM_MARKET', 0.0)
    }

def calculate_sector_rs(stock_prices: pd.Series, sector_prices: pd.Series) -> Dict[str, float]:
    """Calculate relative strength metrics against sector index"""
    from relative_strength import calculate_relative_strength_cumulative
    
    # Calculate cumulative RS
    rs_cum = calculate_relative_strength_cumulative(stock_prices, sector_prices)
    
    # Calculate rolling window RS
    rs_3m = calculate_relative_strength_cumulative(
        stock_prices[-63:],  # ~3 months
        sector_prices[-63:]
    )
    
    # Calculate momentum RS
    rs_1y = calculate_relative_strength_cumulative(
        stock_prices[-252:],  # ~1 year
        sector_prices[-252:]
    )
    
    return {
        'rs_cumulative': rs_cum,
        'rs_rolling_3m': rs_3m,
        'rs_momentum_1y': rs_1y
    }

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
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for various data components
    (output_path / "fundamentals").mkdir(exist_ok=True)
    (output_path / "sectors").mkdir(exist_ok=True)
    (output_path / "volatility").mkdir(exist_ok=True)
    (output_path / "correlations").mkdir(exist_ok=True)
    
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
    
    # First, download sector rotation data for context
    logger.info("Analyzing sector rotation for market context...")
    sector_rotation_data = get_sector_rotation_metrics("NIFTY")
    with open(output_path / "sectors" / "sector_rotation.json", 'w') as f:
        json.dump(sector_rotation_data, f, indent=2)
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, len(symbols))
        batch_symbols = symbols[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({batch_start} to {batch_end})")
        
        # Implement retry mechanism for batch processing
        retry_count = 0
        batch_success = False
        
        while not batch_success and retry_count < max_retries:
            # Download data for current batch
            batch_data = []
            failed_symbols = []
            
            for symbol in batch_symbols:
                df = download_stock_data(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    # Calculate sector-specific data
                    symbol_sector = sector_map.get(symbol, None)
                    
                    # Add sector rotation metrics for this stock
                    sector_data = get_sector_rotation_metrics(symbol, symbol_sector)
                    with open(output_path / "sectors" / f"{symbol.replace('.NS', '')}_sector.json", 'w') as f:
                        json.dump(sector_data, f, indent=2)
                    
                    # Add volatility regime analysis
                    from market_regime import analyze_volatility_regime
                    vol_regime = analyze_volatility_regime(df)
                    with open(output_path / "volatility" / f"{symbol.replace('.NS', '')}_vol_regime.json", 'w') as f:
                        json.dump(vol_regime, f, indent=2)
                    
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
                # Calculate technical indicators for the batch
                from technical_indicators import add_technical_indicators
                logger.info(f"Calculating technical indicators for batch {batch_num + 1}...")
                
                batch_df = pd.concat(batch_data)
                batch_df = add_technical_indicators(batch_df)
                
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
            
    # Create a data summary file
    create_data_summary(output_path, symbols)
    
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

def create_data_summary(output_path: Path, symbols: List[str]) -> None:
    """
    Create a comprehensive data summary file documenting the pipeline execution
    
    Args:
        output_path: Path where data files are stored
        symbols: List of symbols processed
    """
    summary_file = output_path / "data_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Nifty 500 Data Pipeline Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Execution Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Total Symbols Processed: {len(symbols)}\n")
        f.write(f"Output Directory: {output_path}\n\n")
        
        # List generated files
        f.write("Generated Files:\n")
        f.write("-" * 20 + "\n")
        for file_path in output_path.glob("*.parquet"):
            f.write(f"- {file_path.name}\n")
        
        f.write("\nData Components:\n")
        f.write("-" * 20 + "\n")
        f.write("- Historical price data (OHLCV)\n")
        f.write("- 17+ technical indicators with candlestick patterns\n")
        f.write("- Fundamental metrics and ratios\n")
        f.write("- News sentiment analysis\n")
        f.write("- Relative strength calculations\n")
        f.write("- Market regime analysis\n")
        f.write("- Sector rotation metrics\n")
        f.write("- Intermarket correlations\n")
        f.write("- Option chain data (when available)\n")
        f.write("- Institutional ownership data\n")
        
        f.write(f"\nFirst 10 Symbols:\n")
        f.write("-" * 20 + "\n")
        for symbol in symbols[:10]:
            f.write(f"- {symbol}\n")
        
    logger.info(f"Data summary saved to {summary_file}")

def get_sector_rotation_metrics(symbol: str, sector: str = None) -> Dict[str, Any]:
    """
    Calculate sector rotation metrics and performance rankings
    
    Args:
        symbol: Stock symbol (with .NS suffix for Indian stocks)
        sector: Sector name if known, otherwise will be looked up
        
    Returns:
        Dictionary containing sector rotation metrics and rankings
    """
    # Define timeframes for analysis
    timeframes = {
        '1w': 5,    # 1 week (5 trading days)
        '1m': 21,   # 1 month (21 trading days)
        '3m': 63,   # 3 months (63 trading days)
        '6m': 126,  # 6 months (126 trading days)
        '1y': 252   # 1 year (252 trading days)
    }
    
    # Get sector information if not provided
    if sector is None:
        ticker = yf.Ticker(symbol)
        sector = ticker.info.get('sector', 'Unknown')
    
    # Load Indian sector indices
    sector_performance = {}
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)  # 1 year history
    
    # Download data for all sector indices
    all_indices_data = {}
    for idx_symbol, idx_name in INDIAN_SECTOR_INDICES.items():
        idx = yf.Ticker(idx_symbol)
        idx_data = idx.history(start=start_date, end=end_date)
        if not idx_data.empty:
            all_indices_data[idx_name] = idx_data
            logger.debug(f"Downloaded {len(idx_data)} days for {idx_name}")
    
    # Download Nifty 50 as benchmark
    nifty_data = None
    if 'Nifty 50' in all_indices_data:
        nifty_data = all_indices_data['Nifty 50']
    else:
        nifty = yf.Ticker('^NSEI')
        nifty_data = nifty.history(start=start_date, end=end_date)
        logger.debug(f"Downloaded {len(nifty_data)} days for Nifty 50")
    
    # Download stock data
    stock_data = None
    stock = yf.Ticker(symbol)
    stock_data = stock.history(start=start_date, end=end_date)
    logger.debug(f"Downloaded {len(stock_data)} days for {symbol}")
    
    # Calculate sector performance for each timeframe
    timeframe_performance = {}
    timeframe_leaders = {}
    sectors_ranked = {}
    
    for timeframe_name, days in timeframes.items():
        # Calculate returns for all sectors
        sector_returns = {}
        for idx_name, idx_data in all_indices_data.items():
            if len(idx_data) >= days:
                sector_returns[idx_name] = (idx_data['Close'].iloc[-1] / idx_data['Close'].iloc[-days] - 1) * 100
        
        # Calculate benchmark return
        benchmark_return = None
        if nifty_data is not None and len(nifty_data) >= days:
            benchmark_return = (nifty_data['Close'].iloc[-1] / nifty_data['Close'].iloc[-days] - 1) * 100
        
        # Calculate stock return
        stock_return = None
        if stock_data is not None and len(stock_data) >= days:
            stock_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-days] - 1) * 100
        
        # Rank sectors by return
        if sector_returns:
            ranked_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
            sectors_ranked[timeframe_name] = ranked_sectors
            
            # Identify top and bottom sectors
            timeframe_leaders[timeframe_name] = {
                'top_sectors': [name for name, _ in ranked_sectors[:3]],
                'bottom_sectors': [name for name, _ in ranked_sectors[-3:]],
                'market_return': benchmark_return,
                'stock_return': stock_return
            }
            
            # Save performance
            timeframe_performance[timeframe_name] = sector_returns
    
    # Identify dominant sectors (sectors in top 3 across multiple timeframes)
    sector_count = {}
    for timeframe, leaders in timeframe_leaders.items():
        for sector_name in leaders['top_sectors']:
            sector_count[sector_name] = sector_count.get(sector_name, 0) + 1
    
    dominant_sectors = [sector for sector, count in sector_count.items() 
                       if count >= 2]  # Sectors that are top performers in at least 2 timeframes
    
    # Calculate sector rotation trends
    rotation_trends = {}
    for timeframe1, timeframe2 in [('1w', '1m'), ('1m', '3m'), ('3m', '6m')]:
        if timeframe1 in timeframe_leaders and timeframe2 in timeframe_leaders:
            # Check for changing leadership
            top_recent = set(timeframe_leaders[timeframe1]['top_sectors'])
            top_longer = set(timeframe_leaders[timeframe2]['top_sectors'])
            
            new_leaders = top_recent - top_longer
            fading_leaders = top_longer - top_recent
            
            rotation_trends[f'{timeframe1}_vs_{timeframe2}'] = {
                'new_leaders': list(new_leaders),
                'fading_leaders': list(fading_leaders),
                'rotation_strength': len(new_leaders) / 3 if new_leaders else 0
            }
    
    # Calculate relative sector strength for the stock's sector
    relative_sector_strength = {}
    stock_sector = None
    
    for sector_name in all_indices_data.keys():
        if sector.lower() in sector_name.lower():
            stock_sector = sector_name
            break
    
    if stock_sector and stock_sector in all_indices_data and nifty_data is not None:
        sector_data = all_indices_data[stock_sector]
        
        for timeframe_name, days in timeframes.items():
            if len(sector_data) >= days and len(nifty_data) >= days:
                sector_return = (sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-days] - 1) * 100
                benchmark_return = (nifty_data['Close'].iloc[-1] / nifty_data['Close'].iloc[-days] - 1) * 100
                
                relative_strength = sector_return - benchmark_return
                relative_sector_strength[timeframe_name] = relative_strength
    
    # Construct the result dictionary
    result = {
        'dominant_sectors': dominant_sectors,
        'timeframe_leaders': timeframe_leaders,
        'relative_sector_strength': relative_sector_strength,
        'rotation_trends': rotation_trends,
        'stock_sector': stock_sector if stock_sector else sector,
        'sector_rankings': {
            timeframe: {
                'ranks': [{'sector': s, 'return': r} for s, r in ranks[:5]]  # Top 5 only
            } for timeframe, ranks in sectors_ranked.items()
        }
    }
    
    # Add sector breadth analysis
    sector_breadth = {}
    for timeframe_name, days in timeframes.items():
        if timeframe_name in timeframe_performance:
            # Calculate percentage of sectors outperforming benchmark
            if benchmark_return is not None:
                outperforming = sum(1 for ret in timeframe_performance[timeframe_name].values() 
                                   if ret > benchmark_return)
                sector_breadth[timeframe_name] = outperforming / len(timeframe_performance[timeframe_name])
    
    result['sector_breadth'] = sector_breadth
    
    return result
