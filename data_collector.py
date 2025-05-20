import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, api_key: str):
        """Initialize DataCollector with api key for potential premium data services"""
        self.api_key = api_key
        self.nifty500_symbols = self._get_nifty500_symbols()
        self._data_cache = {}
        
    def _get_nifty500_symbols(self) -> List[str]:
        """Fetch Nifty 500 symbols using a more reliable method via Wikipedia and yfinance verification"""
        try:
            # First try to load from disk cache if available
            try:
                with open('nifty500_symbols.json', 'r') as f:
                    cached_data = json.load(f)
                    if datetime.now().timestamp() - cached_data['timestamp'] < 24*60*60:  # 24 hour cache
                        return cached_data['symbols']
            except (FileNotFoundError, json.JSONDecodeError):
                pass
                
            # Scrape from Wikipedia as fallback (more stable than NSE website)
            url = "https://en.wikipedia.org/wiki/NIFTY_500"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table with company listings
            tables = soup.find_all('table', {'class': 'wikitable'})
            symbols = []
            
            for table in tables:
                rows = table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:  # Ensure row has enough columns
                        symbol = cols[1].text.strip()
                        if symbol:
                            symbols.append(symbol + '.NS')
                            
            # Verify symbols exist in yfinance
            verified_symbols = []
            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    if info.get('regularMarketPrice') is not None:
                        verified_symbols.append(symbol)
                    time.sleep(0.1)  # Rate limiting
                except:
                    continue
                    
            # Cache the results
            if verified_symbols:
                with open('nifty500_symbols.json', 'w') as f:
                    json.dump({
                        'symbols': verified_symbols,
                        'timestamp': datetime.now().timestamp()
                    }, f)
                    
            return verified_symbols
            
        except Exception as e:
            logger.error(f"Error fetching Nifty 500 symbols: {e}")
            return []

    def get_financial_metrics(self, symbol: str) -> Dict:
        """Get 10 years of financial metrics for a company"""
        try:
            stock = yf.Ticker(symbol)
            
            # Fetch financial statements
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cash_flow
            
            # Calculate key metrics
            metrics = {
                'revenue_growth': self._calculate_growth_rates(income_stmt.loc['Total Revenue']),
                'profit_margins': income_stmt.loc['Net Income'] / income_stmt.loc['Total Revenue'],
                'roe': income_stmt.loc['Net Income'] / balance_sheet.loc['Total Stockholder Equity'],
                'debt_to_equity': balance_sheet.loc['Total Liab'] / balance_sheet.loc['Total Stockholder Equity'],
                'current_ratio': balance_sheet.loc['Total Current Assets'] / balance_sheet.loc['Total Current Liabilities'],
                'fcf_yield': cash_flow.loc['Free Cash Flow'] / (stock.info.get('marketCap', 1)),
                'pe_ratio': stock.info.get('trailingPE', None),
                'pb_ratio': stock.info.get('priceToBook', None),
                'dividend_yield': stock.info.get('dividendYield', 0)
            }

            return metrics
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {e}")
            return {}

    def _calculate_growth_rates(self, series: pd.Series) -> List[float]:
        """Calculate year-over-year growth rates"""
        return [(series[i] - series[i+1]) / series[i+1] for i in range(len(series)-1)]

    def get_news_sentiment(self, symbol: str, days: int = 30) -> Dict:
        """Get news articles and their sentiment for the last n days"""
        try:
            # Get news articles from multiple sources
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                return {'sentiment_score': 0, 'articles': []}
            
            # Simple sentiment calculation based on news volume
            return {
                'sentiment_score': min(len(news) / 10, 1.0),  # Normalize by 10 articles
                'articles': news[:10]
            }
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {symbol}: {e}")
            return {'sentiment_score': 0, 'articles': []}

    def _analyze_with_llm(self, articles: List[str]) -> Dict:
        """Simplified analysis without LLM"""
        return {
            'sentiment_score': 0.5,  # Neutral sentiment
            'insights': [],
            'risks': [],
            'opportunities': []
        }

    def analyze_company(self, symbol: str) -> Dict:
        """Comprehensive analysis of a company"""
        metrics = self.get_financial_metrics(symbol)
        news_sentiment = self.get_news_sentiment(symbol)
        
        # Simple scoring based on financial metrics
        try:
            health_score = 50  # Default neutral score
            if metrics:
                # Basic scoring based on available metrics
                if metrics.get('current_ratio', 1) > 1.5:
                    health_score += 10
                if metrics.get('debt_to_equity', 2) < 1:
                    health_score += 10
                if metrics.get('profit_margins', [0])[-1] > 0.1:
                    health_score += 10
            
            result = {
                'health_score': health_score,
                'growth_score': 50,  # Neutral score
                'risk_score': 50,    # Neutral score
                'recommendation_score': 50,  # Neutral score
                'metrics': metrics,
                'news_sentiment': news_sentiment
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in company analysis: {e}")
            return {}

    def get_market_data(self, symbol: str, period: str = "10y", use_cache: bool = True) -> pd.DataFrame:
        """Get market data with technical indicators and caching
        
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        cache_key = f"{symbol}_{period}"
        
        # Try to get from cache first
        if use_cache and cache_key in self._data_cache:
            cached_data = self._data_cache[cache_key]
            cache_age = (datetime.now() - cached_data['timestamp']).total_seconds()
            
            # Return cached data if less than 1 hour old
            if cache_age < 3600:
                return cached_data['data']
        
        try:
            # Fetch new data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
                
            # Add technical indicators
            # 1. Moving averages
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()
            
            # 2. RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # 4. Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
            
            # 5. Volume indicators
            df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA20']
            
            # Cache the results
            self._data_cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now()
            }
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def prepare_training_data(self, 
                             symbols: List[str] = None, 
                             start_date: str = "2018-01-01",
                             end_date: str = None) -> Dict[str, Dict]:
        """Prepare comprehensive training data for all symbols with proper date ranges
        
        Args:
            symbols: List of stock symbols. If None, uses Nifty 500 symbols
            start_date: Start date for historical data in YYYY-MM-DD format
            end_date: End date for historical data in YYYY-MM-DD format. If None, uses today
            
        Returns:
            Dictionary containing analysis and market data for each symbol
        """
        if symbols is None:
            symbols = self.nifty500_symbols
            
        if not symbols:
            logger.error("No symbols provided and unable to fetch Nifty 500 symbols")
            return {}
            
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        training_data = {}
        failed_symbols = []
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            try:
                # Get market data
                market_data = self.get_market_data(symbol, period="max")
                
                if market_data.empty:
                    logger.warning(f"No market data available for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                    
                # Filter for date range
                market_data = market_data[start_date:end_date]
                
                if market_data.empty:
                    logger.warning(f"No data in specified date range for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Get company analysis
                analysis = self.analyze_company(symbol)
                
                if not analysis:
                    logger.warning(f"Unable to analyze company {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Store data
                training_data[symbol] = {
                    'analysis': analysis,
                    'market_data': market_data
                }
                
                # Respect API rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue
        
        # Log summary
        success_count = len(training_data)
        fail_count = len(failed_symbols)
        logger.info(f"Data collection complete. Successfully processed {success_count} symbols. Failed: {fail_count}")
        if failed_symbols:
            logger.info(f"Failed symbols: {', '.join(failed_symbols)}")
            
        return training_data