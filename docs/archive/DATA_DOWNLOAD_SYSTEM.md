# Data Download System Documentation

## Overview

The Nifty 500 data download system is a robust, fault-tolerant framework designed to collect, process, and maintain historical and real-time market data. The system handles data acquisition, cleaning, validation, and storage with automatic error recovery and data quality checks.

## System Architecture

### 1. Data Sources

#### Primary Sources
- NSE (National Stock Exchange)
  - Symbol list
  - Corporate actions
  - Market indices
- Yahoo Finance
  - Historical OHLCV data
  - Real-time quotes
  - Company information

#### Backup Sources
- BSE (Bombay Stock Exchange)
- Alternative data providers
- Local cache system

### 2. Core Components

#### 2.1 Symbol Management
```python
def get_nifty500_symbols() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Get current Nifty 500 constituents with sector/industry classification
    
    Returns:
    - List of symbols with .NS suffix
    - Sector mapping dictionary
    - Industry mapping dictionary
    """
    # Primary source: NSE website
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty500list.csv"
    ]
    
    # Cache management
    cache_file = Path("data/nifty500_symbols.json")
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            if time.time() - cache_data['timestamp'] < 86400:  # 24-hour validity
                return cache_data['symbols'], {}, {}
```

**Features:**
- Automatic cache management
- Multiple source fallback
- Error handling with retries
- Sector/industry classification

#### 2.2 Historical Data Download
```python
def download_stock_data(
    symbol: str, 
    period: str = "10y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download historical OHLCV data with retry mechanism
    
    Parameters:
    - symbol: Stock symbol with .NS suffix
    - period: Time period to download (default: 10 years)
    - interval: Data frequency (default: daily)
    
    Returns:
    - DataFrame with OHLCV data
    """
    @retry(
        stop_max_attempt_number=3,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000
    )
    def _download():
        return yf.download(
            symbol, 
            period=period, 
            interval=interval,
            progress=False
        )
```

**Key Features:**
- Configurable time periods and intervals
- Exponential backoff retry mechanism
- Automatic adjustment for splits/dividends
- Progress tracking for batch downloads

#### 2.3 Real-time Data Collection
```python
def get_realtime_data(symbol: str) -> Dict[str, Any]:
    """
    Get real-time market data and calculate metrics
    
    Returns dictionary with:
    - Current price and volume
    - Bid/Ask spread
    - Trading session information
    - Last update timestamp
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'price': info.get('regularMarketPrice'),
            'volume': info.get('regularMarketVolume'),
            'bid': info.get('bid'),
            'ask': info.get('ask'),
            'timestamp': info.get('regularMarketTime')
        }
    except Exception as e:
        logger.error(f"Error fetching real-time data: {e}")
        return None
```

### 3. Data Processing Pipeline

#### 3.1 Data Validation
```python
def validate_data(df: pd.DataFrame) -> bool:
    """
    Comprehensive data validation checks
    
    Validates:
    - Price continuity
    - Volume consistency
    - Trading calendar alignment
    - Corporate action adjustments
    """
    checks = [
        _check_price_continuity(df),
        _check_volume_consistency(df),
        _check_calendar_alignment(df),
        _check_corporate_actions(df)
    ]
    return all(checks)
```

#### 3.2 Data Cleaning
```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess OHLCV data
    
    Steps:
    1. Remove duplicates
    2. Handle missing values
    3. Adjust for splits/dividends
    4. Validate price/volume ranges
    """
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Handle missing values
    df['Close'] = df['Close'].fillna(method='ffill')
    df['Volume'] = df['Volume'].fillna(0)
    
    # Validate ranges
    df = _validate_price_ranges(df)
    df = _validate_volume_ranges(df)
    
    return df
```

### 4. Storage System

#### 4.1 Data Organization
```plaintext
data/
├── historical/
│   ├── daily/
│   │   └── {symbol}_5y.parquet
│   ├── weekly/
│   └── monthly/
├── realtime/
└── metadata/
    ├── symbols.json
    ├── sectors.json
    └── corporate_actions.json
```

#### 4.2 File Formats
- **Parquet**: Primary storage format for OHLCV data
  - Efficient compression
  - Column-oriented storage
  - Fast read/write operations

- **JSON**: Metadata and configuration storage
  - Human-readable
  - Easy version control
  - Flexible schema

### 5. Error Handling

#### 5.1 Retry Mechanism
```python
def download_with_retry(symbol: str) -> pd.DataFrame:
    """
    Download data with exponential backoff retry
    """
    @retry(
        stop_max_attempt_number=3,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000
    )
    def _download():
        return yf.download(symbol, period="5y")
    
    try:
        return _download()
    except Exception as e:
        logger.error(f"Failed to download {symbol}: {e}")
        return pd.DataFrame()
```

#### 5.2 Error Recovery
```python
def recover_failed_downloads(
    failed_symbols: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Attempt to recover failed downloads using alternative sources
    """
    recovered_data = {}
    for symbol in failed_symbols:
        try:
            # Try alternative data sources
            data = _try_alternative_sources(symbol)
            if not data.empty:
                recovered_data[symbol] = data
        except Exception as e:
            logger.error(f"Recovery failed for {symbol}: {e}")
    return recovered_data
```

### 6. Performance Optimization

#### 6.1 Parallel Downloads
```python
def parallel_download(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple symbols in parallel
    """
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {
            executor.submit(download_stock_data, symbol): symbol 
            for symbol in symbols
        }
        
        results = {}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
    
    return results
```

#### 6.2 Caching Strategy
```python
@lru_cache(maxsize=100)
def get_cached_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Retrieve data from cache with LRU caching
    """
    cache_key = f"{symbol}_{start_date}_{end_date}"
    if cache_key in _data_cache:
        return _data_cache[cache_key]
    
    data = download_stock_data(symbol, start_date, end_date)
    _data_cache[cache_key] = data
    return data
```

### 7. Monitoring and Logging

#### 7.1 Download Statistics
```python
def log_download_stats(stats: Dict[str, Any]):
    """
    Log download statistics and performance metrics
    """
    logger.info(
        f"Download Stats:\n"
        f"- Total symbols: {stats['total_symbols']}\n"
        f"- Successful: {stats['successful']}\n"
        f"- Failed: {stats['failed']}\n"
        f"- Time taken: {stats['time_taken']:.2f}s\n"
        f"- Average time per symbol: {stats['avg_time']:.2f}s"
    )
```

#### 7.2 Data Quality Metrics
```python
def calculate_quality_metrics(
    df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate data quality metrics
    
    Returns:
    - Completeness score
    - Consistency score
    - Timeliness score
    - Overall quality score
    """
    metrics = {
        'completeness': _calculate_completeness(df),
        'consistency': _calculate_consistency(df),
        'timeliness': _calculate_timeliness(df)
    }
    
    metrics['overall_quality'] = sum(metrics.values()) / len(metrics)
    return metrics
```

### 8. Usage Examples

#### 8.1 Basic Usage
```python
# Get Nifty 500 symbols
symbols, sectors, industries = get_nifty500_symbols()

# Download historical data for one symbol
data = download_stock_data('RELIANCE.NS', period='5y')

# Process and validate the data
if validate_data(data):
    clean_data = clean_data(data)
    store_data(clean_data, 'RELIANCE.NS')
```

#### 8.2 Batch Processing
```python
# Download data for multiple symbols
symbol_batch = symbols[:50]
results = parallel_download(symbol_batch)

# Process results
for symbol, data in results.items():
    if validate_data(data):
        processed_data = clean_data(data)
        store_data(processed_data, symbol)
```

### 9. Configuration

#### 9.1 System Parameters
```python
CONFIG = {
    'download': {
        'batch_size': 50,
        'max_retries': 3,
        'timeout': 30,
        'cache_expiry': 86400
    },
    'validation': {
        'min_data_points': 100,
        'max_missing_pct': 0.1,
        'price_jump_threshold': 0.2
    },
    'storage': {
        'format': 'parquet',
        'compression': 'snappy',
        'partition_cols': ['year', 'month']
    }
}
```

This comprehensive documentation covers all aspects of the data download system, from architecture to implementation details and usage examples.
