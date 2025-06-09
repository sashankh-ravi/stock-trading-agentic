import datetime

# This is a placeholder for the actual generation logic.
# In a real scenario, this would involve detailed text generation
# based on the plan above, incorporating information from the Python scripts.

current_year = datetime.datetime.now().year

reliance_example_date = "2024-07-15" # Placeholder date for examples
reliance_close_price = "2950.50"
reliance_rsi_value = "65.3"
reliance_sma20_value = "2905.75"
reliance_sma50_value = "2850.20"
reliance_pe_ratio = "28.5"
reliance_sector_pe = "25.0" # Example sector P/E
reliance_news_sentiment_score = "+7"
reliance_pcr_value = "0.85"
reliance_call_oi_strike = "3000"
reliance_put_oi_strike = "2900"
reliance_institutional_own_pct = "45.2"
reliance_rs_vs_nifty = "1.05 (outperformed by 5% over period)"
reliance_correlation_oil = "0.35"


markdown_content = f"""# Nifty 500 Data Download & Augmentation Pipeline: A Deep Dive

## Last Updated: {datetime.date.today().isoformat()}

## 1. Introduction

This document provides an exhaustive technical explanation of the `download_nifty500_data.py` module, which forms the backbone of our stock data acquisition and augmentation pipeline for the Nifty 500 universe. In the world of quantitative trading, the quality, breadth, and depth of data are paramount. A robust data pipeline is the first critical step towards developing, backtesting, and deploying successful trading strategies.

This pipeline is designed to:
*   Fetch comprehensive historical price and volume data.
*   Calculate a wide array of technical indicators.
*   Retrieve detailed fundamental data for each company.
*   Analyze news sentiment.
*   Gather option chain data for volatility and sentiment insights.
*   Track institutional ownership.
*   Perform relative strength analysis against benchmarks and sectors.
*   Assess overall market breadth and intermarket correlations.

The ultimate goal is to create a rich, multi-faceted dataset for each stock in the Nifty 500, enabling sophisticated analysis and informed trading decisions.

## 2. Core Architecture & Setup (`download_nifty500_data.py`)

The `download_nifty500_data.py` script orchestrates the entire data collection process. It leverages several key Python libraries and a structured configuration.

### 2.1. Key Libraries & Their Roles

*   **`yfinance`**: The primary workhorse for fetching data from Yahoo Finance. It provides access to historical stock prices, trading volumes, company fundamentals (income statements, balance sheets, cash flow), news headlines, option chain data, and institutional holder information.
    *   *Significance*: Offers a free and relatively comprehensive source for global financial data, including Indian equities with the `.NS` suffix.
*   **`pandas`**: Essential for data manipulation and analysis. All fetched data is typically loaded into pandas DataFrames for cleaning, transformation, calculation of indicators, and storage.
    *   *Significance*: Provides powerful and flexible data structures (DataFrame, Series) that are ideal for time-series and tabular financial data.
*   **`numpy`**: Used for numerical operations, especially array manipulations and mathematical functions that underpin many financial calculations.
    *   *Significance*: Offers efficient numerical computation, often used by pandas and other libraries under the hood.
*   **`talib` (Ta-Lib)**: A widely used library for technical analysis, specifically for calculating candlestick pattern recognitions (e.g., Doji, Hammer, Engulfing patterns). The script uses functions like `talib.CDLDOJI()`.
    *   *Significance*: Provides pre-built, optimized functions for many common technical indicators and patterns, saving development time and ensuring correctness.
*   **`requests`**: For making HTTP requests to fetch data from web sources, such as the Nifty 500 constituents list from the NSE India website.
    *   *Significance*: A standard library for interacting with web APIs and websites.
*   **`concurrent.futures`**: Enables parallel execution of tasks, particularly useful for downloading data for multiple stocks simultaneously, significantly speeding up the overall process. `ThreadPoolExecutor` is used.
    *   *Significance*: Improves performance by leveraging multi-threading for I/O-bound tasks like data downloading.
*   **`logging`**: Implements a logging mechanism to track the script's execution, record informational messages, warnings, and errors. This is crucial for monitoring the pipeline's health and debugging issues.
    *   *Significance*: Provides a structured way to get feedback from the running application.
*   **`pathlib`, `json`, `datetime`, `os`, `time`, `re`, `math`, `functools.lru_cache`**: These are standard Python utility modules used for file path manipulation, working with JSON data (e.g., for caching), date/time operations, interacting with the operating system, adding delays (rate limiting), regular expressions, mathematical calculations, and caching function results for performance, respectively.

### 2.2. Configuration

*   **Logging Setup**:
    ```python
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    ```
    This configures application-wide logging to output messages with a timestamp, log level, and the message itself. `INFO` level provides a good balance of detail.
*   **Global Market Definitions**:
    *   `INDIAN_SECTOR_INDICES`: A dictionary mapping Yahoo Finance tickers for Indian sector indices (e.g., `^CNXBANK` for Nifty Bank) to their names.
        *   *Purpose*: Used for sector-specific relative strength analysis and benchmarking.
    *   `GLOBAL_INDICES`: Includes major global stock market indices (e.g., S&P 500 `^GSPC`, FTSE 100 `^FTSE`).
        *   *Purpose*: Essential for intermarket correlation analysis to understand how Indian markets are influenced by or move with global trends.
    *   `COMMODITIES_BONDS`: Lists tickers for key commodities (e.g., Gold `GC=F`, Crude Oil `CL=F`) and bond yields (e.g., US 10-Year Treasury `^TNX`).
        *   *Purpose*: Also for intermarket analysis, as these assets can significantly impact equity markets and reflect economic conditions. For example, rising oil prices can affect inflation and specific sectors.

## 3. Phase 1: Symbol Universe & Metadata Acquisition

Before any stock-specific data can be downloaded, the pipeline must first identify the current constituents of the Nifty 500 index and gather relevant metadata like sector and industry classifications.

### 3.1. Fetching Nifty 500 Symbols (`get_nifty500_symbols`)

*   **Purpose**: To obtain an accurate and up-to-date list of stock symbols that are part of the Nifty 500 index. This list forms the universe of stocks for which data will be downloaded.
*   **Process**:
    1.  **Caching**:
        ```python
        cache_file = Path("data/nifty500_symbols.json")
        if cache_file.exists():
            # ... load from cache if recent ...
        ```
        The function first checks for a local cache file (`data/nifty500_symbols.json`). If the cached list is recent (e.g., less than 24 hours old), it's used to avoid redundant downloads and reduce load on the NSE server.
        *   *Significance*: Improves performance and respects data provider limits.
    2.  **Fetching from NSE India**: If the cache is missing or stale, the script attempts to download the list from the official NSE India website using URLs like:
        `"https://archives.nseindia.com/content/indices/ind_nifty500list.csv"`
        A `User-Agent` header is set in the `requests.get` call to mimic a browser, which can be necessary to access some websites.
        *   *Significance*: Ensures the symbol list is sourced from the authoritative provider.
    3.  **Symbol Formatting**: The downloaded symbols (e.g., "RELIANCE") are appended with `.NS` (e.g., "RELIANCE.NS") to make them compatible with Yahoo Finance's ticker format for Indian stocks.
    4.  **Sector & Industry Mapping**: The CSV file from NSE typically contains sector and industry information for each stock.
        ```python
        sector_col = next((col for col in ['Sector', 'Industry', ...] if col in df.columns), None)
        # ...
        SECTOR_MAPPING = {{f"{{row['Symbol']}}.NS": row[sector_col] for _, row in df.iterrows()}}
        ```
        This information is extracted and stored in global dictionaries `SECTOR_MAPPING` and `INDUSTRY_MAPPING`.
        *   *Significance*: Crucial for sector-based analysis, relative strength comparisons against sector peers, and building sector-rotation strategies.
*   **RELIANCE.NS Example**:
    *   When `get_nifty500_symbols` runs, it would identify "RELIANCE" in the NSE list.
    *   It would be formatted to "RELIANCE.NS".
    *   Its sector, say "Energy" or "Refineries" (depending on NSE's classification in the CSV), would be stored: `SECTOR_MAPPING['RELIANCE.NS'] = 'Energy'`.

### 3.2. Enriching Sector Data (`_enrich_sector_data`)

*   **Purpose**: If the sector or industry data obtained from the NSE CSV is incomplete or missing for some symbols, this function attempts to fill these gaps using Yahoo Finance.
*   **Process**:
    ```python
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if 'sector' in info and info['sector']:
        SECTOR_MAPPING[symbol] = info['sector']
    ```
    For each symbol with missing data, it fetches the `ticker.info` dictionary from Yahoo Finance and extracts the `sector` and `industry` fields if available.
*   **Significance**: Aims to ensure that every stock in the universe has associated sector and industry metadata, which is vital for comprehensive analysis. This also acts as a fallback.

## 4. Phase 2: Individual Stock Data Acquisition & Processing

Once the symbol universe is defined, the pipeline processes each stock individually to gather a wide array of data points. The `download_historical_data_for_symbol` function in `download_nifty500_data.py` often serves as a mini-orchestrator for fetching price/volume and then triggering technical indicator and relative strength calculations for a single symbol. It internally calls `download_stock_data` from `download_stock_data.py`.

### 4.A. Historical Price & Volume Data (via `download_stock_data.py`)

*   **Core Function**: The actual download is handled by `download_stock_data` (from `download_stock_data.py`), which is called by `download_historical_data_for_symbol` in the main `download_nifty500_data.py` script.
*   **Purpose**: To obtain historical Open, High, Low, Close prices, and Volume (OHLCV) for each stock. This is the foundational data for most forms of technical analysis and strategy backtesting.
*   **Process**:
    *   Utilizes `yf.Ticker(symbol).history(start=start_date, end=end_date, interval=interval, auto_adjust=True)`.
    *   `start_date`, `end_date`: Define the period for historical data (e.g., last 5 years).
    *   `interval`: Specifies the data frequency (e.g., '1d' for daily, '1h' for hourly).
    *   `auto_adjust=True`: This is crucial as it provides prices adjusted for corporate actions like stock splits and dividends.
        *   *Significance of Adjusted Prices*: Ensures that historical price series are comparable over time, preventing artificial jumps or drops due to corporate actions that don't reflect a change in the company's fundamental value. For example, after a 2:1 stock split, the price halves, but `auto_adjust=True` scales historical prices down to maintain continuity.
*   **Data Quality Checks (Conceptual, as performed in `test_single_stock_comprehensive.py`)**:
    *   **Non-empty data**: Ensure some data was returned.
    *   **Sufficient data points**: Check if the number of records is adequate for analysis.
    *   **Presence of OHLCV columns**: Verify that 'Open', 'High', 'Low', 'Close', 'Volume' are present.
    *   **Logical consistency**:
        *   `High` must be greater than or equal to `Low`.
        *   `High` should generally be greater than or equal to `Open` and `Close`.
        *   `Low` should generally be less than or equal to `Open` and `Close`.
    *   **Non-negative volume**: Volume traded cannot be negative.
    *   *Significance*: These checks help identify issues with the data source or the download process early on.
*   **RELIANCE.NS Example**:
    *   To fetch daily data for RELIANCE.NS for the past 2 years:
        `data = yf.Ticker("RELIANCE.NS").history(period="2y", interval="1d", auto_adjust=True)`
    *   A typical row in the resulting DataFrame for RELIANCE.NS on `{reliance_example_date}` might look like:
        `Date: {reliance_example_date}, Open: 2930.00, High: 2965.00, Low: 2925.00, Close: {reliance_close_price}, Volume: 5,200,100`
        (Note: Prices are adjusted if `auto_adjust=True`).

### 4.B. Technical Indicators

*   **Integration**: Technical indicators are typically calculated after the raw OHLCV data is fetched. The `download_historical_data_for_symbol` function calls `add_technical_indicators` (from `technical_indicators.py`) and also includes direct `talib` calls for candlestick patterns.
*   **Purpose**: To transform raw price and volume data into quantitative signals that can help identify trends, momentum, volatility, market turning points, and other trading opportunities. They form the core inputs for many trading strategies.
*   **Methodology**:
    *   Many standard indicators (SMA, EMA, MACD, etc.) are calculated using pandas rolling window functions (`.rolling().mean()`, `.ewm().mean()`).
    *   Specific candlestick patterns are identified using functions from the `talib` library (e.g., `talib.CDLDOJI`).

*   **Key Indicators Covered**:

    1.  **Simple Moving Average (SMA)**
        *   **Purpose**: Smooths out price data to identify the direction of the trend. Longer SMAs reflect long-term trends, shorter SMAs reflect short-term trends.
        *   **Formula**: Sum of closing prices over N periods / N.
        *   **Interpretation**: Price above SMA is often seen as bullish, below as bearish. Crossovers of different SMAs (e.g., 20-day SMA crossing above 50-day SMA - a "golden cross") can signal trend changes.
        *   **RELIANCE.NS Example**: "On `{reliance_example_date}`, RELIANCE.NS closed at {reliance_close_price}. Its 20-day SMA was `{reliance_sma20_value}` and 50-day SMA was `{reliance_sma50_value}`. Since the price is above both SMAs and the 20-day SMA is above the 50-day SMA, this suggests a bullish trend in the short to medium term."
        *   **Visualization**: A plot would show the closing price line with the 20-day and 50-day SMA lines overlaid. Crossovers and the price's position relative to the SMAs would be key visual cues.

    2.  **Exponential Moving Average (EMA)**
        *   **Purpose**: Similar to SMA but gives more weight to recent prices, making it more responsive to new information.
        *   **Formula**: (Close - Previous EMA) * Multiplier + Previous EMA, where Multiplier = 2 / (N + 1).
        *   **Interpretation**: Used like SMAs for trend identification and crossover signals, but reacts faster.
        *   **RELIANCE.NS Example**: "RELIANCE.NS's 20-day EMA on `{reliance_example_date}` might be [Value], potentially slightly different from its SMA due to the weighting, offering a quicker reflection of recent price action."

    3.  **Relative Strength Index (RSI)**
        *   **Purpose**: A momentum oscillator that measures the speed and change of price movements. It indicates overbought or oversold conditions.
        *   **Formula**: `100 - (100 / (1 + RS))`, where RS = (Average Gain over N periods) / (Average Loss over N periods). Typically N=14.
        *   **Interpretation**: Values range from 0 to 100.
            *   RSI > 70: Often considered overbought (potential for a price pullback).
            *   RSI < 30: Often considered oversold (potential for a price bounce).
            *   Divergences: Bullish divergence (price makes lower low, RSI makes higher low) or bearish divergence (price makes higher high, RSI makes lower high) can signal trend reversals.
        *   **RELIANCE.NS Example**: "On `{reliance_example_date}`, RELIANCE.NS had a 14-day RSI of `{reliance_rsi_value}`. This value is approaching the overbought territory but not yet extreme, suggesting strong upward momentum that warrants monitoring for signs of exhaustion."
        *   **Caveat**: In strong trends, RSI can remain in overbought/oversold territory for extended periods.

    4.  **Moving Average Convergence Divergence (MACD)**
        *   **Purpose**: A trend-following momentum indicator that shows the relationship between two EMAs of a security’s price (typically 12-period EMA and 26-period EMA).
        *   **Formula**: MACD Line = 12-period EMA - 26-period EMA. Signal Line = 9-period EMA of MACD Line. Histogram = MACD Line - Signal Line.
        *   **Interpretation**:
            *   Crossovers: MACD Line crossing above Signal Line is bullish; below is bearish.
            *   Zero Line Crossover: MACD Line crossing above zero is bullish; below is bearish.
            *   Divergences: Similar to RSI, divergences between MACD and price can signal trend changes.
        *   **RELIANCE.NS Example**: "For RELIANCE.NS on `{reliance_example_date}`, if the MACD line was [Value, e.g., 15.2] and the Signal line was [Value, e.g., 12.8], with the MACD line having recently crossed above the signal line, it would be a bullish signal."

    5.  **Bollinger Bands (BBands)**
        *   **Purpose**: Measure market volatility and identify overbought/oversold conditions relative to a moving average.
        *   **Formula**:
            *   Middle Band: N-period SMA (typically 20-day).
            *   Upper Band: Middle Band + (K * N-period Standard Deviation of Price) (typically K=2).
            *   Lower Band: Middle Band - (K * N-period Standard Deviation of Price).
        *   **Interpretation**:
            *   Price touching Upper Band: Potentially overbought.
            *   Price touching Lower Band: Potentially oversold.
            *   Band Squeeze: Low volatility, often precedes a significant price move.
            *   Band Expansion: High volatility.
        *   **RELIANCE.NS Example**: "If RELIANCE.NS price on `{reliance_example_date}` was trading near its Upper Bollinger Band ([Value]), it might suggest a short-term overbought condition or strong momentum if the bands are expanding."

    6.  **Average True Range (ATR)**
        *   **Purpose**: Measures market volatility by decomposing the entire range of an asset price for that period.
        *   **Formula**: Typically a 14-day smoothed moving average of True Range values. True Range = max[(High - Low), abs(High - Previous Close), abs(Low - Previous Close)].
        *   **Interpretation**: Higher ATR indicates higher volatility; lower ATR indicates lower volatility. Not directional. Often used for setting stop-loss levels or position sizing.
        *   **RELIANCE.NS Example**: "An ATR of [Value, e.g., 45.5] for RELIANCE.NS on `{reliance_example_date}` means the stock has an average daily price range of roughly 45.5 points over the last 14 days."

    7.  **Commodity Channel Index (CCI)**
        *   **Purpose**: A momentum-based oscillator used to help determine when an investment vehicle is reaching a condition of being overbought or oversold.
        *   **Formula**: (Typical Price - N-period SMA of Typical Price) / (0.015 * N-period Mean Deviation of Typical Price). Typical Price = (High + Low + Close) / 3.
        *   **Interpretation**: Values above +100 suggest overbought; below -100 suggest oversold.
        *   **RELIANCE.NS Example**: "A CCI reading of [Value, e.g., 120] for RELIANCE.NS would indicate an overbought condition."

    8.  **Average Directional Index (ADX), Plus Directional Indicator (+DI), Minus Directional Indicator (-DI)**
        *   **Purpose**: ADX measures trend strength (not direction). +DI and -DI measure trend direction.
        *   **Interpretation**:
            *   ADX > 25: Strong trend (either up or down). ADX < 20: Weak or non-trending market.
            *   +DI above -DI: Bullish trend. -DI above +DI: Bearish trend.
            *   Crossovers of +DI and -DI can signal trend changes, confirmed by ADX strength.
        *   **RELIANCE.NS Example**: "If RELIANCE.NS has an ADX of [Value, e.g., 30], +DI at [Value, e.g., 28], and -DI at [Value, e.g., 15], it indicates a strong uptrend."

    9.  **Money Flow Index (MFI)**
        *   **Purpose**: A momentum indicator that incorporates volume and price data. It's also known as volume-weighted RSI.
        *   **Interpretation**: Similar to RSI for overbought (>80) / oversold (<20) conditions and divergences.
        *   **RELIANCE.NS Example**: "An MFI of [Value, e.g., 75] for RELIANCE.NS suggests significant buying pressure."

    10. **On-Balance Volume (OBV)**
        *   **Purpose**: A momentum indicator that uses volume flow to predict changes in stock price.
        *   **Interpretation**: Rising OBV reflects positive volume pressure that can lead to higher prices. Falling OBV signals negative volume pressure. Divergences between OBV and price are noteworthy.
        *   **RELIANCE.NS Example**: "If RELIANCE.NS price is making new highs and its OBV is also making new highs, it confirms the uptrend."

    11. **Rate of Change (ROC)**
        *   **Purpose**: A momentum oscillator that measures the percentage change in price between the current price and the price N periods ago.
        *   **Interpretation**: Positive ROC indicates upward momentum; negative ROC indicates downward momentum. Zero line crossovers can be used as signals.
        *   **RELIANCE.NS Example**: "A 10-day ROC of [Value, e.g., 2.5%] for RELIANCE.NS means its price has increased by 2.5% over the last 10 trading days."

*   **Candlestick Patterns (Identified using `talib` functions like `talib.CDLDOJI`, `talib.CDLHAMMER`, etc.)**:
    *   **Purpose**: These are short-term patterns (1-3 candles) that can indicate potential reversals or continuations. The `download_nifty500_data.py` script calculates many of these.
    *   **Examples**:
        *   **Doji**: Indicates indecision. A day where open and close are very close.
            *   *RELIANCE.NS Example*: "A Doji pattern on `{reliance_example_date}` for RELIANCE.NS, after a strong uptrend, might signal that the buying momentum is waning."
        *   **Hammer**: Bullish reversal pattern typically found at the bottom of a downtrend. Small body, long lower shadow, little to no upper shadow.
            *   *RELIANCE.NS Example*: "If RELIANCE.NS formed a Hammer on `{reliance_example_date}` after a period of decline, it would suggest buyers stepped in, potentially reversing the downtrend."
        *   **Engulfing (Bullish/Bearish)**: A two-candle reversal pattern. A Bullish Engulfing has a small bearish candle followed by a large bullish candle that engulfs the prior one.
            *   *RELIANCE.NS Example*: "A Bullish Engulfing pattern for RELIANCE.NS would be a strong bottom reversal signal."
    *   **Interpretation**: The output from `talib` functions is typically a series of numbers (0 for no pattern, 100 for bullish pattern, -100 for bearish pattern on a given day).
    *   **Significance**: While individual patterns have limited reliability, they can be powerful when combined with other indicators or support/resistance levels.

### 4.C. Fundamental Data (`get_fundamental_data`)

*   **Purpose**: To evaluate a company's intrinsic value, financial performance, health, and growth prospects. Fundamental data is crucial for long-term investment decisions and for identifying undervalued or overvalued stocks.
*   **Process**: This function queries `yfinance` for a wealth of information:
    *   `ticker.info`: Provides a dictionary with many current metrics (market cap, P/E, dividend yield, etc.).
    *   `ticker.financials`, `ticker.quarterly_financials`: Income Statement data.
    *   `ticker.balance_sheet`, `ticker.quarterly_balance_sheet`: Balance Sheet data.
    *   `ticker.cashflow`, `ticker.quarterly_cashflow`: Cash Flow Statement data.
    The function then extracts and calculates various ratios and metrics.

*   **Key Metrics Covered**:

    1.  **Valuation Metrics**:
        *   **P/E Ratio (Price-to-Earnings)**: `info.get('trailingPE')`
            *   *Significance*: How much investors are willing to pay per rupee of earnings. High P/E can mean high growth expectations or overvaluation.
            *   *RELIANCE.NS Example*: "RELIANCE.NS has a P/E ratio of `{reliance_pe_ratio}`. Comparing this to its historical average and the sector average (`{reliance_sector_pe}` from `_get_sector_averages`) helps assess its current valuation."
        *   **P/B Ratio (Price-to-Book)**: `info.get('priceToBook')`
            *   *Significance*: Compares market value to book value. Useful for capital-intensive industries.
        *   **PEG Ratio (Price/Earnings-to-Growth)**: `info.get('pegRatio')`
            *   *Significance*: P/E ratio divided by earnings growth rate. A PEG around 1 might suggest fair valuation relative to growth.
        *   **EV/EBITDA (Enterprise Value-to-EBITDA)**: `info.get('enterpriseToEbitda')`
            *   *Significance*: Compares total company value (market cap + debt - cash) to its earnings before interest, taxes, depreciation, and amortization. Useful for comparing companies with different capital structures.
        *   **Price-to-Sales (P/S)**: `info.get('priceToSalesTrailing12Months')`
            *   *Significance*: Compares market cap to total revenue. Useful for growth companies not yet profitable.

    2.  **Profitability Metrics**:
        *   **Profit Margin**: `info.get('profitMargins') * 100`
            *   *Significance*: Net income / Revenue. Shows how much profit is generated per rupee of sales.
        *   **Operating Margin**: `info.get('operatingMargins') * 100`
            *   *Significance*: Operating income / Revenue. Efficiency in core business operations.
        *   **Return on Equity (ROE)**: `info.get('returnOnEquity') * 100`
            *   *Significance*: Net income / Shareholder Equity. How effectively shareholder money is used to generate profit.
        *   **Return on Assets (ROA)**: `info.get('returnOnAssets') * 100`
            *   *Significance*: Net income / Total Assets. How efficiently assets are used.
        *   **Return on Invested Capital (ROIC)**: Calculated using EBIT, tax rate, and invested capital (equity + debt).
            *   *Significance*: Measures profit generated on all capital invested. Often compared to WACC.

    3.  **Financial Health Metrics**:
        *   **Debt-to-Equity (D/E)**: `info.get('debtToEquity') / 100` (if yfinance provides it as percentage)
            *   *Significance*: Total Debt / Shareholder Equity. Measures financial leverage.
        *   **Current Ratio**: `info.get('currentRatio')`
            *   *Significance*: Current Assets / Current Liabilities. Measures short-term liquidity.

    4.  **Dividend Metrics**:
        *   **Dividend Yield**: `info.get('dividendYield') * 100`
            *   *Significance*: Annual dividend per share / Price per share. Return from dividends.
        *   **Payout Ratio**: `info.get('payoutRatio') * 100`
            *   *Significance*: Dividends per share / Earnings per share. Proportion of earnings paid as dividends.

    5.  **Growth Metrics**:
        *   **Revenue Growth**: `info.get('revenueGrowth') * 100` (often YoY or QoQ)
        *   **Earnings Growth**: `info.get('earningsGrowth') * 100`

    6.  **Ownership & Analyst Data**:
        *   **Institutional Ownership**: `info.get('institutionsPercentHeld') * 100`
        *   **Analyst Target Price**: `info.get('targetMeanPrice')`
        *   **Analyst Rating**: `info.get('recommendationMean')` (e.g., 1=Strong Buy, 5=Strong Sell)

*   **Sector Averages (`_get_sector_averages`)**:
    *   **Purpose**: To provide context for a stock's fundamental metrics. A P/E of 30 might be high for a utility but low for a tech company.
    *   **Methodology**: The script currently uses a hardcoded dictionary of representative average metrics for various Indian sectors.
        ```python
        sector_data = { 'Information Technology': {'pe_ratio': 25.0, ...}, ... }
        ```
        Ideally, this would be sourced from a dynamic, up-to-date database or API.
    *   **Significance**: Comparing a stock's metrics (e.g., P/E, P/B, Dividend Yield) to its sector average helps in relative valuation.
    *   **RELIANCE.NS Example**: "If RELIANCE.NS (Energy sector) has a P/E of `{reliance_pe_ratio}` and the average P/E for the Energy sector is `{reliance_sector_pe}`, this suggests RELIANCE.NS is trading at a [premium/discount/similar valuation] compared to its peers."

### 4.D. News Sentiment Analysis (`get_news_sentiment`)

*   **Purpose**: To quantify the general market sentiment towards a stock based on recent news headlines. Positive news can drive prices up, while negative news can drive them down.
*   **Process**:
    1.  Fetches news articles using `yf.Ticker(symbol).news`.
    2.  Performs a simple keyword-based sentiment analysis on each article title:
        *   A predefined list of `positive_keywords` (e.g., 'rise', 'gain', 'profit', 'upgrade') and `negative_keywords` (e.g., 'fall', 'loss', 'downgrade', 'concern') is used.
        *   Each title is scanned for these keywords. Positive keywords increment a score, negative keywords decrement it.
    3.  Aggregates scores to produce an overall sentiment score, counts of positive/negative/neutral articles, and attempts to identify a sentiment trend.
*   **Metrics**:
    *   `sentiment_score`: An aggregate numerical score.
    *   `article_count`: Total number of recent articles.
    *   `recent_sentiment`: Categorical ('positive', 'negative', 'neutral').
    *   `sentiment_trend`: ('improving', 'declining', 'stable').
    *   `sentiment_distribution`: Counts of positive, neutral, negative articles.
*   **RELIANCE.NS Example**: "For RELIANCE.NS, the news sentiment analysis might yield a `sentiment_score` of `{reliance_news_sentiment_score}` based on [X] articles. This could be categorized as '[Positive/Moderately Positive]' recent sentiment. An example positive headline might be 'RELIANCE.NS posts strong quarterly earnings'."
*   **Limitations**:
    *   Keyword-based sentiment is very basic and can be easily fooled by negation, sarcasm, or complex language.
    *   It doesn't understand context deeply.
    *   More advanced Natural Language Processing (NLP) models (e.g., using transformer-based models like BERT or FinBERT) would provide much more accurate sentiment. The script mentions "NLP scoring" in its docstring, implying an aspiration for this.

### 4.E. Option Chain Data (`get_option_chain_data`)

*   **Purpose**: To analyze options market activity for insights into market expectations, implied volatility, potential support/resistance levels, and overall sentiment.
*   **Process**:
    1.  Fetches available option expiry dates using `ticker.options`.
    2.  For each expiry (or a selection of them, e.g., near-term expiries), it fetches the call and put option chains using `ticker.option_chain(expiry_date)`.
    3.  Extracts data like `impliedVolatility`, `openInterest`, `volume`, `lastPrice` for each option contract.
    4.  Calculates aggregate metrics like Put-Call Ratio and identifies strikes with high Open Interest.
    5.  The `_calculate_max_pain` helper function attempts to find the strike price at which the maximum number of option holders (both call and put buyers) would lose money if the stock expires at that price.
*   **Key Metrics**:
    *   **Implied Volatility (IV)**: The market's forecast of likely movement in a security's price. High IV suggests expectations of large price swings.
        *   *Significance*: Important for option pricing and risk assessment.
    *   **Put-Call Ratio (PCR)**: Ratio of trading volume or open interest of put options to call options.
        *   *Significance*: Often used as a contrarian indicator. A very high PCR might suggest bearish sentiment is overdone (potentially bullish), and a very low PCR might suggest bullish sentiment is overdone (potentially bearish).
    *   **Open Interest (OI)**: Total number of outstanding option contracts that have not been settled.
        *   *Significance*: High OI at certain strike prices can indicate potential support (for puts) or resistance (for calls) levels, as these are points where many market participants have a vested interest.
    *   **Max Pain**: The strike price where the greatest number of options (in terms of dollar value) would expire worthless.
        *   *Significance*: Some theories suggest that the underlying stock price will tend to gravitate towards the max pain strike price as expiration approaches.
*   **RELIANCE.NS Example**: "For RELIANCE.NS options expiring on [Expiry Date]:
    *   The overall Put-Call Ratio (Open Interest) is `{reliance_pcr_value}`.
    *   The highest Call OI is at the `{reliance_call_oi_strike}` strike, and the highest Put OI is at the `{reliance_put_oi_strike}` strike.
    *   The average Implied Volatility for near-the-money options is [IV Value]%.
    *   The calculated Max Pain strike is [Max Pain Strike Value]."

### 4.F. Institutional Ownership (`get_institutional_ownership`)

*   **Purpose**: To track the buying and selling activity of large financial institutions (mutual funds, pension funds, insurance companies). Significant institutional ownership can provide stability to a stock's price, and changes in their holdings can signal shifts in "smart money" sentiment.
*   **Process**: Fetches data using `yf.Ticker(symbol).institutional_holders`. This typically provides a list of top institutional holders and their reported positions and changes. The script aggregates this to get overall percentages.
*   **Metrics**:
    *   `percent_held_by_institutions`: Percentage of the company's outstanding shares held by institutions.
    *   `number_of_institutions_holding_stock`: Count of reporting institutions.
    *   (Potentially) Top holders, recent changes in their positions.
*   **RELIANCE.NS Example**: "As of the latest reporting, `{reliance_institutional_own_pct}`% of RELIANCE.NS's outstanding shares are held by [Number] institutions. Major holders include [Example Institution 1] and [Example Institution 2]."
*   **Significance**: High and increasing institutional ownership is often seen as a positive sign, indicating confidence in the company's prospects. Conversely, significant selling by institutions can be a warning.

## 5. Phase 3: Market-Wide & Relational Analysis

Beyond individual stock data, the pipeline also gathers information to understand the broader market context and how stocks perform relative to benchmarks or other asset classes.

### 5.A. Relative Strength (RS)

*   **Integration**: The `calculate_relative_strength_metrics` function (from `relative_strength.py`) is called, likely within `download_historical_data_for_symbol` or by the main orchestrator, after fetching the stock's and benchmark's historical data.
*   **Purpose**: To measure a stock's performance compared to a benchmark (e.g., Nifty 50, a sector index) or other stocks. Stocks showing strong relative strength (outperforming the benchmark) are often preferred by momentum investors.
*   **Methodology**:
    *   Typically involves calculating the ratio of the stock's price to the benchmark's price over time. An upward trending RS line indicates outperformance.
    *   The script might calculate RS over different periods (e.g., 3-month, 6-month, 1-year).
    *   It can also involve ranking stocks based on their RS.
*   **Metrics**:
    *   **RS Ratio**: `Stock Price / Benchmark Price`.
    *   **RS Percentile/Rank**: Stock's RS rank within a universe (e.g., Nifty 500).
    *   **RS Trend**: Whether the RS line is trending up, down, or flat.
*   **RELIANCE.NS Example**: "RELIANCE.NS's relative strength line versus the Nifty 50 has been [trending upwards/downwards/flat] over the past 6 months. Its current RS ratio is [Value]. This indicates it has [outperformed/underperformed/performed in line with] the broader market. Compared to the Nifty Energy index, its RS is [Description, e.g., `{reliance_rs_vs_nifty}`]."
*   **Significance**: Identifying market leaders (strong RS) or laggards (weak RS) is a key component of many trading strategies. Sector RS helps identify strong/weak sectors.

### 5.B. Market Breadth (`calculate_market_breadth`)

*   **Purpose**: To assess the health and underlying strength of a market move. Market indices can sometimes be skewed by a few large-cap stocks, while breadth indicators show how widely the move is distributed across the market.
*   **Process**: This function requires price data from a significant portion of the market (e.g., all Nifty 500 stocks).
    1.  For each stock in the universe, determine if it advanced or declined on a given day.
    2.  Calculate metrics like:
        *   **Advance-Decline Line (A/D Line)**: A cumulative sum of (Advancing Stocks - Declining Stocks).
        *   **New Highs-New Lows (NH-NL)**: Number of stocks making 52-week highs minus those making 52-week lows.
        *   **McClellan Oscillator/Summation Index**: More complex breadth indicators based on smoothed A/D data.
*   **Significance**:
    *   **Confirmation**: If the market index is rising and the A/D Line is also rising, it confirms the rally's strength.
    *   **Divergence**: If the index is rising but the A/D Line is falling (negative divergence), it can be a warning sign that the rally is losing steam and is not supported by broad market participation.
*   **Example**: "On `{reliance_example_date}`, while the Nifty 50 index [rose/fell] by [X] points, the Nifty 500 Advance-Decline Line [also rose, indicating broad participation / fell, indicating a divergence]." (This is a market-wide metric, not specific to RELIANCE.NS, but provides context for its price action).

### 5.C. Intermarket Correlations (`calculate_intermarket_correlations`)

*   **Purpose**: To understand the relationships between different asset classes (e.g., Indian equities vs. US equities, equities vs. commodities like oil or gold, equities vs. bonds, equities vs. currencies like USD/INR). These relationships can shift and provide valuable clues about market sentiment and economic trends.
*   **Process**:
    1.  Fetches historical data for the assets defined in `GLOBAL_INDICES` and `COMMODITIES_BONDS`.
    2.  Calculates rolling correlation coefficients (e.g., using `.rolling(window=N).corr()`) between:
        *   Nifty 50 and these global assets.
        *   Individual stocks (like RELIANCE.NS) and selected global assets.
*   **Significance**:
    *   **Risk-On/Risk-Off**: Correlations can indicate prevailing market sentiment. For example, a negative correlation between equities and gold might strengthen during risk-off periods (gold seen as a safe haven).
    *   **Economic Linkages**: Correlation with oil prices can be important for oil-importing countries like India and for specific sectors.
    *   **Global Influence**: Correlation with major global indices like the S&P 500 shows how much Indian markets are influenced by global trends.
*   **RELIANCE.NS Example**: "The 60-day rolling correlation between RELIANCE.NS and Brent Crude Oil (placeholder for CL=F) is currently `{reliance_correlation_oil}`. This suggests a [positive/negative/weak] short-term relationship. Its correlation with the S&P 500 (`^GSPC`) is [Value], indicating [degree] of co-movement with US markets."

## 6. Phase 4: Orchestration & Data Output

The entire process of fetching and processing data for all Nifty 500 stocks is managed by a main coordinating function.

### 6.1. Main Function (`download_nifty500_data_comprehensive`)

*   **Purpose**: This is the top-level function that orchestrates the entire data download and augmentation pipeline for all stocks in the Nifty 500 universe.
*   **Process**:
    1.  **Get Symbol List**: Calls `get_nifty500_symbols()` to obtain the list of stocks to process.
    2.  **Parallel Processing**: It uses `concurrent.futures.ThreadPoolExecutor` to download and process data for multiple stocks in parallel. This significantly reduces the total time taken.
        ```python
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_symbol_data, symbol) for symbol in symbols]
            # ... collect results ...
        ```
        (where `process_symbol_data` would be a helper function that calls `download_historical_data_for_symbol`, `get_fundamental_data`, etc. for one symbol).
    3.  **Error Handling**: Robust error handling (e.g., `try-except` blocks within the processing loop for each symbol) is essential to ensure that an issue with one stock doesn't halt the entire pipeline. Errors are typically logged.
    4.  **Rate Limiting**: Implicitly or explicitly, the code should respect API rate limits of data providers like Yahoo Finance. This might involve adding small delays (`time.sleep()`) between requests if not using a library that handles this.
    5.  **Data Aggregation & Storage**:
        *   The processed data for each stock (OHLCV, technical indicators, fundamentals, etc.) is typically combined into a pandas DataFrame.
        *   These DataFrames are then saved to disk, often in an efficient file format like Parquet. The script aims to save data to paths like `data/nifty500_data_comprehensive/{symbol_name}.parquet`.
        *   *Parquet File Format*: Chosen for its efficiency in terms of storage space and read/write speed, especially for large datasets and analytical queries. It's a columnar storage format.
*   **Output Structure**: Each Parquet file (e.g., `RELIANCE.parquet`) would contain a table where rows represent time periods (e.g., daily) and columns include:
    *   OHLCV data
    *   All calculated technical indicators
    *   Relative strength values
    *   (Fundamental data might be stored separately or joined if it's point-in-time, or if historical fundamentals are fetched, they could be time-series as well).

## 7. Leveraging the Test Script (`test_single_stock_comprehensive.py`)

The `test_single_stock_comprehensive.py` script plays a vital role in validating the integrity and correctness of the entire data download and augmentation pipeline.

*   **Purpose**:
    *   To perform an end-to-end test of all major data fetching and calculation components for a single stock (hardcoded to `RELIANCE.NS` in the provided snippet, but can be randomized).
    *   To ensure that each module (`technical_indicators.py`, `relative_strength.py`, `market_regime.py` etc. as used by the main download script) integrates correctly and produces sensible output.
    *   To provide a quick way to debug issues in the pipeline by focusing on one stock.
*   **How it Works**:
    The script is structured into several test functions, each targeting a specific aspect of the pipeline:
    1.  `select_test_symbol()`: Selects the stock to test (e.g., "RELIANCE.NS").
    2.  `test_basic_data_download()`: Tests OHLCV download and basic quality checks.
    3.  `test_technical_indicators()`: Tests the addition of various technical indicators.
    4.  `test_fundamental_data()`: Tests fetching of key fundamental metrics.
    5.  `test_news_sentiment()`: Tests the news fetching and basic sentiment scoring.
    6.  `test_relative_strength()`: Tests RS calculation against a benchmark (Nifty 50).
    7.  `test_market_regime()`: Tests the market regime classification logic.
    8.  `test_data_quality()`: Performs more in-depth data quality checks on the processed data.
    The main `run_comprehensive_test()` function calls these test functions sequentially.
*   **Interpreting Output**:
    *   **Logs**: The script heavily uses `logging` to output information about each test's progress and status (e.g., "✓ Basic data download test PASSED"). Any assertion failures will stop the script and indicate an error.
    *   **Summary Statistics**: At the end, it prints a summary of key results for the tested stock.
    *   **Saved Parquet File**: Crucially, it saves the fully processed DataFrame for the test stock to a file like `data/test_results/RELIANCE_comprehensive_test.parquet`.
        *   *Significance*: This output file is extremely valuable. You can load this Parquet file into a pandas DataFrame to manually inspect all the calculated values (OHLCV, every technical indicator, RS values, etc.) for `RELIANCE.NS` for each date. This allows for detailed verification and understanding of how each metric is computed.
*   **How to Use for This Documentation's Examples**:
    The specific values for `RELIANCE.NS` used as examples throughout this document (e.g., RSI value on a certain date, P/E ratio) are conceptual placeholders. To get **actual, live values**:
    1.  Ensure your environment is set up with all required libraries (`requirements.txt`).
    2.  Run the `test_single_stock_comprehensive.py` script.
    3.  Observe the logs for summary values or, for more detail, load the generated `RELIANCE_comprehensive_test.parquet` file:
        ```python
        import pandas as pd
        df_reliance = pd.read_parquet('data/test_results/RELIANCE_comprehensive_test.parquet')
        print(df_reliance.tail()) # To see recent data with all indicators
        # You can then pick specific values for specific dates.
        ```
    This allows you to populate the examples in this documentation with real, verifiable data.

## 8. Conclusion

The `download_nifty500_data.py` pipeline provides a robust and comprehensive framework for acquiring and preparing a rich dataset essential for quantitative stock market analysis and trading strategy development. By systematically fetching historical prices, calculating a diverse set of technical indicators, retrieving fundamental data, analyzing news sentiment, and considering market-wide factors, it lays a strong foundation for data-driven decision-making.

The use of parallel processing, caching, and structured error handling makes the pipeline efficient and resilient. The accompanying test script (`test_single_stock_comprehensive.py`) ensures ongoing validation and provides a clear way to inspect the generated data for any given stock.

This detailed dataset enables:
*   In-depth backtesting of trading strategies.
*   Development of sophisticated stock screening models.
*   Machine learning applications for price prediction or pattern recognition.
*   Comprehensive risk management analysis.

**Potential Future Enhancements**:
*   Integration of alternative data sources (e.g., economic indicators, corporate filings NLP, social media sentiment).
*   More advanced NLP for news sentiment (e.g., using pre-trained financial models like FinBERT).
*   Storing data in a dedicated time-series database (e.g., InfluxDB, TimescaleDB) for better query performance and scalability.
*   Automated scheduling and monitoring of the download pipeline.
*   Generation of historical point-in-time fundamental data to avoid lookahead bias in backtesting.

## 9. Appendix: Quick Formula Reference (Selected Technical Indicators)

| Indicator                 | Brief Formula Concept                                                                 | Typical Period(s) |
|---------------------------|---------------------------------------------------------------------------------------|-------------------|
| **SMA**                   | Sum of N closing prices / N                                                           | 20, 50, 200       |
| **EMA**                   | (Close - Prev_EMA) * (2/(N+1)) + Prev_EMA                                             | 12, 20, 26, 50    |
| **RSI**                   | 100 - (100 / (1 + (Avg Gain / Avg Loss)))                                             | 14                |
| **MACD Line**             | EMA(12) - EMA(26)                                                                     | 12, 26            |
| **MACD Signal Line**      | EMA(9) of MACD Line                                                                   | 9 (of MACD)       |
| **Bollinger Middle Band** | SMA(N)                                                                                | 20                |
| **Bollinger Upper/Lower** | SMA(N) +/- (StdDev(N) * K)                                                            | 20 (N), 2 (K)     |
| **ATR**                   | Smoothed MA of True Range; True Range = max(H-L, abs(H-PrevC), abs(L-PrevC))           | 14                |
| **OBV**                   | Cumulative sum of volume (added if close up, subtracted if close down)                | N/A               |
| **ROC**                   | ((Close - Close N periods ago) / Close N periods ago) * 100                          | 10, 12            |
| **CCI**                   | (Typical Price - SMA(TP)) / (0.015 * Mean Deviation(TP)); TP=(H+L+C)/3                | 20                |
| **MFI**                   | 100 - (100 / (1 + Money Flow Ratio)); MFR = (14-period Positive MF / 14-period Negative MF) | 14                |

*(Note: This is a simplified representation. Refer to standard financial texts for precise calculation details and nuances.)*

"""
# This is where the content would be written to a file if the tool was available.
# For now, it's just stored in the markdown_content variable.
# print(f"Markdown content generated with {len(markdown_content)} characters.")
