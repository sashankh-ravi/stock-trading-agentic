"""
Technical Indicators Module

This module implements a comprehensive set of technical indicators for stock market analysis.
Each indicator is documented with its calculation method, interpretation, and typical usage.

Categories:
1. Trend Indicators
2. Momentum Indicators
3. Volatility Indicators
4. Volume Indicators
5. Cycle Indicators
6. Pattern Recognition
7. Custom Composite Indicators
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class IndicatorSignal:
    """Represents a signal from a technical indicator"""
    name: str
    value: float
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0 to 1
    timeframe: str  # 'short', 'medium', 'long'

class TechnicalIndicators:
    """
    Comprehensive technical analysis toolkit with correlation-aware indicator combinations
    """
    
    def __init__(self):
        # Define indicator groups and their correlations
        self.indicator_groups = {
            'trend': {
                'sma': ['SMA20', 'SMA50', 'SMA200'],
                'ema': ['EMA20', 'EMA50', 'EMA200'],
                'macd': ['MACD', 'MACD_Signal', 'MACD_Hist'],
                'adx': ['ADX'],
                'parabolic_sar': ['SAR'],
                'ichimoku': ['Tenkan', 'Kijun', 'Senkou_A', 'Senkou_B', 'Chikou']
            },
            'momentum': {
                'rsi': ['RSI'],
                'stochastic': ['STOCH_K', 'STOCH_D'],
                'williams_r': ['WILLR'],
                'cci': ['CCI'],
                'mfi': ['MFI'],
                'dmi': ['DI_PLUS', 'DI_MINUS']
            },
            'volatility': {
                'bollinger': ['BB_Upper', 'BB_Middle', 'BB_Lower'],
                'atr': ['ATR'],
                'standard_dev': ['STD20'],
                'keltner': ['KC_Upper', 'KC_Middle', 'KC_Lower']
            },
            'volume': {
                'obv': ['OBV'],
                'vwap': ['VWAP'],
                'ad': ['AD'],
                'cmf': ['CMF'],
                'volume_profile': ['VOL_POC', 'VOL_VAH', 'VOL_VAL']
            },
            'cycle': {
                'hurst': ['HURST'],
                'hilbert': ['HT_TRENDLINE', 'HT_SINE'],
                'mesa': ['MESA_SINE', 'MESA_LEADSIN']
            }
        }
        
        # Initialize correlation matrix
        self.correlation_matrix = None
        
    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all technical indicators
        """
        df = data.copy()
        
        # Trend Indicators
        self._add_moving_averages(df)
        self._add_macd(df)
        self._add_adx(df)
        self._add_parabolic_sar(df)
        self._add_ichimoku(df)
        
        # Momentum Indicators
        self._add_rsi(df)
        self._add_stochastic(df)
        self._add_williams_r(df)
        self._add_cci(df)
        self._add_mfi(df)
        self._add_dmi(df)
        
        # Volatility Indicators
        self._add_bollinger_bands(df)
        self._add_atr(df)
        self._add_keltner_channels(df)
        
        # Volume Indicators
        self._add_volume_indicators(df)
        self._add_volume_profile(df)
        
        # Cycle Indicators
        self._add_cycle_indicators(df)
        
        # Calculate correlations
        self._update_correlation_matrix(df)
        
        return df
        
    def _add_moving_averages(self, df: pd.DataFrame):
        """Add various moving averages"""
        periods = [20, 50, 200]
        for period in periods:
            df[f'SMA{period}'] = talib.SMA(df['Close'], timeperiod=period)
            df[f'EMA{period}'] = talib.EMA(df['Close'], timeperiod=period)
            
    def _add_macd(self, df: pd.DataFrame):
        """
        Add MACD indicator
        MACD = 12-period EMA - 26-period EMA
        Signal = 9-period EMA of MACD
        """
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
    def _add_adx(self, df: pd.DataFrame):
        """
        Add Average Directional Index (ADX)
        Measures trend strength regardless of direction
        """
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        
    def _add_parabolic_sar(self, df: pd.DataFrame):
        """
        Add Parabolic SAR
        Indicates potential reversals in price movement
        """
        df['SAR'] = talib.SAR(df['High'], df['Low'])
        
    def _add_ichimoku(self, df: pd.DataFrame):
        """
        Add Ichimoku Cloud indicators
        A comprehensive trend trading system
        """
        # Conversion Line (Tenkan-sen)
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Tenkan'] = (high_9 + low_9) / 2
        
        # Base Line (Kijun-sen)
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Kijun'] = (high_26 + low_26) / 2
        
        # Leading Span A (Senkou Span A)
        df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
        
        # Leading Span B (Senkou Span B)
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
        
        # Lagging Span (Chikou)
        df['Chikou'] = df['Close'].shift(-26)
        
    def _add_rsi(self, df: pd.DataFrame):
        """
        Add Relative Strength Index (RSI)
        Measures momentum by comparing recent gains to recent losses
        """
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
    def _add_stochastic(self, df: pd.DataFrame):
        """
        Add Stochastic Oscillator
        Compares closing price to price range over time
        """
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
            df['High'], df['Low'], df['Close']
        )
        
    def _add_williams_r(self, df: pd.DataFrame):
        """
        Add Williams %R
        Momentum indicator measuring overbought/oversold levels
        """
        df['WILLR'] = talib.WILLR(
            df['High'], df['Low'], df['Close'], timeperiod=14
        )
        
    def _add_cci(self, df: pd.DataFrame):
        """
        Add Commodity Channel Index (CCI)
        Measures deviation from statistical mean
        """
        df['CCI'] = talib.CCI(
            df['High'], df['Low'], df['Close'], timeperiod=20
        )
        
    def _add_mfi(self, df: pd.DataFrame):
        """
        Add Money Flow Index (MFI)
        Volume-weighted RSI
        """
        df['MFI'] = talib.MFI(
            df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14
        )
        
    def _add_dmi(self, df: pd.DataFrame):
        """
        Add Directional Movement Index (DMI)
        Measures trend direction and strength
        """
        df['DI_PLUS'] = talib.PLUS_DI(
            df['High'], df['Low'], df['Close'], timeperiod=14
        )
        df['DI_MINUS'] = talib.MINUS_DI(
            df['High'], df['Low'], df['Close'], timeperiod=14
        )
        
    def _add_bollinger_bands(self, df: pd.DataFrame):
        """
        Add Bollinger Bands
        Volatility bands based on standard deviation
        """
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
            df['Close'], timeperiod=20
        )
        
    def _add_atr(self, df: pd.DataFrame):
        """
        Add Average True Range (ATR)
        Measures volatility
        """
        df['ATR'] = talib.ATR(
            df['High'], df['Low'], df['Close'], timeperiod=14
        )
        
    def _add_keltner_channels(self, df: pd.DataFrame):
        """
        Add Keltner Channels
        Volatility-based bands using ATR
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['KC_Middle'] = talib.SMA(typical_price, timeperiod=20)
        atr = df['ATR']
        df['KC_Upper'] = df['KC_Middle'] + (2 * atr)
        df['KC_Lower'] = df['KC_Middle'] - (2 * atr)
        
    def _add_volume_indicators(self, df: pd.DataFrame):
        """Add volume-based indicators"""
        # On-Balance Volume (OBV)
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Volume Weighted Average Price (VWAP)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Accumulation/Distribution Line
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Chaikin Money Flow
        df['CMF'] = talib.ADOSC(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
    def _add_volume_profile(self, df: pd.DataFrame):
        """
        Add Volume Profile indicators
        Analyzes price levels with highest trading activity
        """
        def calculate_poc(prices, volumes):
            # Point of Control - price level with highest volume
            unique_prices = np.unique(prices)
            volume_per_price = {p: sum(volumes[prices == p]) for p in unique_prices}
            poc = max(volume_per_price.items(), key=lambda x: x[1])[0]
            return poc
        
        window = 20
        df['VOL_POC'] = df.rolling(window=window).apply(
            lambda x: calculate_poc(x['Close'], x['Volume'])
        )
        
        # Value Area High and Low (70% of volume)
        def calculate_va(prices, volumes):
            sorted_idx = np.argsort(prices)
            cum_vol = np.cumsum(volumes[sorted_idx])
            total_vol = cum_vol[-1]
            
            # Find price levels containing 70% of volume
            vol_threshold = total_vol * 0.7
            idx = np.searchsorted(cum_vol, vol_threshold)
            
            if idx >= len(prices):
                return prices.max(), prices.min()
                
            return prices[sorted_idx[idx]], prices[sorted_idx[0]]
        
        df[['VOL_VAH', 'VOL_VAL']] = df.rolling(window=window).apply(
            lambda x: pd.Series(calculate_va(x['Close'].values, x['Volume'].values))
        )
        
    def _add_cycle_indicators(self, df: pd.DataFrame):
        """Add cycle-based indicators"""
        # Hurst Exponent (trend strength)
        def hurst(prices):
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag])))
                   for lag in lags]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]  # Hurst exponent is the slope
            
        df['HURST'] = df['Close'].rolling(window=100).apply(
            lambda x: hurst(x.values)
        )
        
        # Hilbert Transform
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['Close'])
        df['HT_SINE'], _ = talib.HT_SINE(df['Close'])
        
        # MESA Sine Wave
        df['MESA_SINE'], df['MESA_LEADSIN'] = talib.HT_SINE(df['Close'])
        
    def _update_correlation_matrix(self, df: pd.DataFrame):
        """
        Update correlation matrix between indicators
        Helps identify redundant indicators
        """
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.correlation_matrix = df[numeric_cols].corr()
        
    def get_uncorrelated_indicators(self, threshold: float = 0.7) -> Dict[str, List[str]]:
        """
        Identifies relatively independent indicators to avoid redundant signals
        
        Parameters:
        -----------
        threshold : float, default=0.7
            Correlation threshold above which indicators are considered redundant
            
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary of indicator groups, where each group contains uncorrelated indicators
            within that category (trend, momentum, volatility, etc.)
        """
        if self.correlation_matrix is None:
            return {}
            
        uncorrelated_groups = {}
        
        # Process each indicator category separately
        for category, indicators in self.indicator_groups.items():
            # Flatten the indicators in this category
            category_indicators = [ind for group in indicators.values() for ind in group]
            
            # Filter indicators that exist in the correlation matrix
            available_indicators = [ind for ind in category_indicators 
                                 if ind in self.correlation_matrix.columns]
            
            if not available_indicators:
                continue
                
            # Find uncorrelated indicators within this category
            selected = []
            remaining = available_indicators.copy()
            
            while remaining:
                current = remaining.pop(0)
                selected.append(current)
                
                # Remove highly correlated indicators
                remaining = [
                    ind for ind in remaining
                    if abs(self.correlation_matrix.loc[current, ind]) < threshold
                ]
            
            uncorrelated_groups[category] = selected
        
        return uncorrelated_groups
        
    def get_optimal_indicator_set(self, df: pd.DataFrame) -> List[str]:
        """
        Get optimal set of indicators using PCA
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with all indicators
            
        Returns:
        --------
        List[str]
            List of most important indicators
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols].fillna(0))
        
        # Apply PCA
        pca = PCA(n_components=0.95)  # Explain 95% of variance
        pca.fit(scaled_data)
        
        # Get feature importance
        importance = np.abs(pca.components_).sum(axis=0)
        importance = importance / importance.sum()
        
        # Select top indicators
        important_idx = importance.argsort()[-10:][::-1]  # Top 10 indicators
        return numeric_cols[important_idx].tolist()
        
    def get_indicator_signals(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """
        Get trading signals from all indicators
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with technical indicators
            
        Returns:
        --------
        List[IndicatorSignal]
            List of signals from each indicator
        """
        signals = []
        current_price = df['Close'].iloc[-1]
        
        # Trend Signals
        self._add_trend_signals(df, signals)
        
        # Momentum Signals
        self._add_momentum_signals(df, signals)
        
        # Volatility Signals
        self._add_volatility_signals(df, signals)
        
        # Volume Signals
        self._add_volume_signals(df, signals)
        
        # Cycle Signals
        self._add_cycle_signals(df, signals)
        
        return signals
        
    def _add_trend_signals(self, df: pd.DataFrame, signals: List[IndicatorSignal]):
        """Add signals from trend indicators"""
        # MACD
        if not df['MACD'].empty and not df['MACD_Signal'].empty:
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_Signal'].iloc[-1]
            hist = macd - signal
            
            signals.append(IndicatorSignal(
                name='MACD',
                value=hist,
                signal='buy' if hist > 0 and macd > 0 else 'sell' if hist < 0 and macd < 0 else 'hold',
                strength=min(abs(hist) / df['Close'].iloc[-1], 1),
                timeframe='medium'
            ))
            
        # Moving Averages
        for period in [20, 50, 200]:
            if f'SMA{period}' in df.columns:
                sma = df[f'SMA{period}'].iloc[-1]
                price = df['Close'].iloc[-1]
                signals.append(IndicatorSignal(
                    name=f'SMA{period}',
                    value=sma,
                    signal='buy' if price > sma else 'sell',
                    strength=min(abs(price - sma) / price, 1),
                    timeframe='long' if period == 200 else 'medium' if period == 50 else 'short'
                ))
                
    def _add_momentum_signals(self, df: pd.DataFrame, signals: List[IndicatorSignal]):
        """Add signals from momentum indicators"""
        # RSI
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            signals.append(IndicatorSignal(
                name='RSI',
                value=rsi,
                signal='buy' if rsi < 30 else 'sell' if rsi > 70 else 'hold',
                strength=min(abs(rsi - 50) / 50, 1),
                timeframe='short'
            ))
            
        # Stochastic
        if 'STOCH_K' in df.columns and 'STOCH_D' in df.columns:
            k = df['STOCH_K'].iloc[-1]
            d = df['STOCH_D'].iloc[-1]
            signals.append(IndicatorSignal(
                name='Stochastic',
                value=k,
                signal='buy' if k > d and k < 20 else 'sell' if k < d and k > 80 else 'hold',
                strength=min(abs(k - 50) / 50, 1),
                timeframe='short'
            ))
            
    def _add_volatility_signals(self, df: pd.DataFrame, signals: List[IndicatorSignal]):
        """Add signals from volatility indicators"""
        # Bollinger Bands
        if all(x in df.columns for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            price = df['Close'].iloc[-1]
            upper = df['BB_Upper'].iloc[-1]
            lower = df['BB_Lower'].iloc[-1]
            signals.append(IndicatorSignal(
                name='Bollinger',
                value=price,
                signal='buy' if price < lower else 'sell' if price > upper else 'hold',
                strength=min(abs(price - df['BB_Middle'].iloc[-1]) / price, 1),
                timeframe='medium'
            ))
            
    def _add_volume_signals(self, df: pd.DataFrame, signals: List[IndicatorSignal]):
        """Add signals from volume indicators"""
        # OBV
        if 'OBV' in df.columns:
            obv = df['OBV'].iloc[-1]
            obv_sma = df['OBV'].rolling(20).mean().iloc[-1]
            signals.append(IndicatorSignal(
                name='OBV',
                value=obv,
                signal='buy' if obv > obv_sma else 'sell',
                strength=min(abs(obv - obv_sma) / abs(obv_sma), 1),
                timeframe='medium'
            ))
            
    def _add_cycle_signals(self, df: pd.DataFrame, signals: List[IndicatorSignal]):
        """Add signals from cycle indicators"""
        # Hurst Exponent
        if 'HURST' in df.columns:
            hurst = df['HURST'].iloc[-1]
            signals.append(IndicatorSignal(
                name='Hurst',
                value=hurst,
                signal='buy' if hurst > 0.6 else 'sell' if hurst < 0.4 else 'hold',
                strength=min(abs(hurst - 0.5), 0.5) * 2,
                timeframe='long'
            ))
            
    def get_best_indicator_combinations(self, df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """
        Get the best combinations of indicators based on historical performance
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data and indicators
        top_n : int
            Number of top combinations to return
            
        Returns:
        --------
        List[Dict]
            List of best indicator combinations with their performance metrics
        """
        # Get uncorrelated indicators
        uncorrelated = self.get_uncorrelated_indicators()
        
        # Get optimal indicators from PCA
        optimal = self.get_optimal_indicator_set(df)
        
        # Combine both sets
        candidate_indicators = list(set(uncorrelated + optimal))
        
        # Test different combinations
        combinations = []
        for i in range(3, min(7, len(candidate_indicators) + 1)):
            for combo in itertools.combinations(candidate_indicators, i):
                performance = self._evaluate_indicator_combination(df, combo)
                combinations.append({
                    'indicators': combo,
                    'performance': performance
                })
                
        # Sort by performance and return top N
        combinations.sort(key=lambda x: x['performance']['sharpe_ratio'], reverse=True)
        return combinations[:top_n]
        
    def _evaluate_indicator_combination(self, df: pd.DataFrame, indicators: List[str]) -> Dict:
        """Evaluate performance of an indicator combination"""
        signals = []
        returns = []
        
        for i in range(100, len(df)):
            window = df.iloc[:i]
            signal = self._get_combined_signal(window[indicators])
            
            if signal != 'hold':
                future_return = (df['Close'].iloc[i+1] - df['Close'].iloc[i]) / df['Close'].iloc[i]
                returns.append(future_return if signal == 'buy' else -future_return)
                
        if not returns:
            return {'sharpe_ratio': 0, 'win_rate': 0, 'avg_return': 0}
            
        returns = np.array(returns)
        
        return {
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() != 0 else 0,
            'win_rate': (returns > 0).mean(),
            'avg_return': returns.mean()
        }
        
    def _get_combined_signal(self, indicators_data: pd.DataFrame) -> str:
        """Get combined signal from multiple indicators"""
        signals = []
        for indicator in indicators_data.columns:
            if indicator in self.indicator_interpretation:
                signal = self.indicator_interpretation[indicator](
                    indicators_data[indicator].iloc[-1]
                )
                signals.append(signal)
                
        # Vote for final signal
        buy_votes = signals.count('buy')
        sell_votes = signals.count('sell')
        
        if buy_votes > len(signals) * 0.6:
            return 'buy'
        elif sell_votes > len(signals) * 0.6:
            return 'sell'
        return 'hold'
        
    # Indicator interpretation rules
    indicator_interpretation = {
        'RSI': lambda x: 'buy' if x < 30 else 'sell' if x > 70 else 'hold',
        'MACD': lambda x: 'buy' if x > 0 else 'sell' if x < 0 else 'hold',
        'ADX': lambda x: 'buy' if x > 25 else 'hold',
        # Add more interpretation rules as needed
    }