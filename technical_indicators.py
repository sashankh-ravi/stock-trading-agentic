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
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools

logger = logging.getLogger(__name__)

# Function to add technical indicators to dataframe - can be used in batch processing
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard technical indicators to a dataframe of stock data
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Process each symbol separately
    symbols = result['symbol'].unique() if 'symbol' in result.columns else [None]
    
    for symbol in symbols:
        # Filter data for this symbol
        if symbol:
            symbol_data = result[result['symbol'] == symbol].copy()
        else:
            symbol_data = result.copy()
        
        # Skip if not enough data
        if len(symbol_data) < 30:
            logger.warning(f"Not enough data for {symbol} to calculate indicators")
            continue
            
        # Trend indicators
        symbol_data['SMA_20'] = talib.SMA(symbol_data['Close'], timeperiod=20)
        symbol_data['SMA_50'] = talib.SMA(symbol_data['Close'], timeperiod=50)
        symbol_data['SMA_200'] = talib.SMA(symbol_data['Close'], timeperiod=200)
        symbol_data['EMA_20'] = talib.EMA(symbol_data['Close'], timeperiod=20)
        symbol_data['EMA_50'] = talib.EMA(symbol_data['Close'], timeperiod=50)
        
        # Momentum indicators
        symbol_data['RSI'] = talib.RSI(symbol_data['Close'], timeperiod=14)
        symbol_data['MACD'], symbol_data['MACD_Signal'], symbol_data['MACD_Hist'] = talib.MACD(
            symbol_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        symbol_data['ROC'] = talib.ROC(symbol_data['Close'], timeperiod=10)  # Rate of Change
        symbol_data['CCI'] = talib.CCI(symbol_data['High'], symbol_data['Low'], symbol_data['Close'], timeperiod=14)  # Commodity Channel Index
        
        # Volatility indicators
        symbol_data['ATR'] = talib.ATR(symbol_data['High'], symbol_data['Low'], symbol_data['Close'], timeperiod=14)
        symbol_data['Bollinger_Upper'], symbol_data['Bollinger_Middle'], symbol_data['Bollinger_Lower'] = talib.BBANDS(
            symbol_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        symbol_data['Bollinger_Width'] = (symbol_data['Bollinger_Upper'] - symbol_data['Bollinger_Lower']) / symbol_data['Bollinger_Middle']
        
        # Volume indicators
        if 'Volume' in symbol_data.columns:
            symbol_data['OBV'] = talib.OBV(symbol_data['Close'], symbol_data['Volume'])
            symbol_data['Money_Flow_Index'] = talib.MFI(
                symbol_data['High'], symbol_data['Low'], 
                symbol_data['Close'], symbol_data['Volume'], timeperiod=14
            )
            # Volume relative to moving average (volume spike detection)
            symbol_data['Volume_SMA_20'] = talib.SMA(symbol_data['Volume'], timeperiod=20)
            symbol_data['Volume_Ratio_20'] = symbol_data['Volume'] / symbol_data['Volume_SMA_20']
            # Daily trading value (price * volume)
            symbol_data['Trading_Value'] = symbol_data['Close'] * symbol_data['Volume']
            
        # Trend strength and direction indicators
        symbol_data['ADX'] = talib.ADX(symbol_data['High'], symbol_data['Low'], symbol_data['Close'], timeperiod=14)
        symbol_data['Plus_DI'] = talib.PLUS_DI(symbol_data['High'], symbol_data['Low'], symbol_data['Close'], timeperiod=14)
        symbol_data['Minus_DI'] = talib.MINUS_DI(symbol_data['High'], symbol_data['Low'], symbol_data['Close'], timeperiod=14)
        
        # Candlestick patterns - Comprehensive set of 20 most reliable patterns
        # Basic reversal patterns
        symbol_data['Doji'] = talib.CDLDOJI(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Engulfing'] = talib.CDLENGULFING(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Hammer'] = talib.CDLHAMMER(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Hanging_Man'] = talib.CDLHANGINGMAN(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Shooting_Star'] = talib.CDLSHOOTINGSTAR(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Inverted_Hammer'] = talib.CDLINVERTEDHAMMER(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        
        # Star patterns (strong reversal signals)
        symbol_data['Morning_Star'] = talib.CDLMORNINGSTAR(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Evening_Star'] = talib.CDLEVENINGSTAR(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Doji_Star'] = talib.CDLDOJISTAR(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        
        # Harami patterns (inside day reversals)
        symbol_data['Harami'] = talib.CDLHARAMI(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Harami_Cross'] = talib.CDLHARAMICROSS(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        
        # Piercing and dark cloud patterns
        symbol_data['Piercing'] = talib.CDLPIERCING(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Dark_Cloud'] = talib.CDLDARKCLOUDCOVER(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        
        # Multiple candle patterns
        symbol_data['Three_White_Soldiers'] = talib.CDL3WHITESOLDIERS(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Three_Black_Crows'] = talib.CDL3BLACKCROWS(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        
        # Advanced patterns
        symbol_data['Spinning_Top'] = talib.CDLSPINNINGTOP(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Marubozu'] = talib.CDLMARUBOZU(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Dragonfly_Doji'] = talib.CDLDRAGONFLYDOJI(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Gravestone_Doji'] = talib.CDLGRAVESTONEDOJI(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        symbol_data['Abandoned_Baby'] = talib.CDLABANDONEDBABY(symbol_data['Open'], symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
        
        # Gap analysis
        symbol_data['Gap_Up'] = symbol_data['Open'] > symbol_data['High'].shift(1)
        symbol_data['Gap_Down'] = symbol_data['Open'] < symbol_data['Low'].shift(1)
        symbol_data['Gap_Size'] = (symbol_data['Open'] - symbol_data['Close'].shift(1)) / symbol_data['Close'].shift(1) * 100
        
        # Add daily returns
        symbol_data['Daily_Return'] = symbol_data['Close'].pct_change() * 100
        
        # Calculate volatility metrics
        symbol_data['Volatility_20'] = symbol_data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        symbol_data['Volatility_10'] = symbol_data['Daily_Return'].rolling(window=10).std() * np.sqrt(252)
        symbol_data['Volatility_60'] = symbol_data['Daily_Return'].rolling(window=60).std() * np.sqrt(252)
        symbol_data['Volatility_Ratio'] = symbol_data['Volatility_10'] / symbol_data['Volatility_60']
        
        # Range expansion/contraction
        symbol_data['Daily_Range'] = (symbol_data['High'] - symbol_data['Low']) / symbol_data['Low'] * 100
        symbol_data['Range_SMA_20'] = talib.SMA(symbol_data['Daily_Range'], timeperiod=20)
        symbol_data['Range_Expansion'] = symbol_data['Daily_Range'] / symbol_data['Range_SMA_20']
        
        # Advanced momentum indicators
        symbol_data['Momentum'] = symbol_data['Close'] / symbol_data['Close'].shift(10) - 1
        symbol_data['ROC_5'] = talib.ROC(symbol_data['Close'], timeperiod=5)
        symbol_data['ROC_10'] = talib.ROC(symbol_data['Close'], timeperiod=10)
        symbol_data['ROC_21'] = talib.ROC(symbol_data['Close'], timeperiod=21)
        symbol_data['ROC_63'] = talib.ROC(symbol_data['Close'], timeperiod=63)
        
        # Add Stochastic indicators
        symbol_data['Stoch_K'], symbol_data['Stoch_D'] = talib.STOCH(symbol_data['High'], 
                                                                   symbol_data['Low'], 
                                                                   symbol_data['Close'],
                                                                   fastk_period=14, 
                                                                   slowk_period=3, 
                                                                   slowd_period=3)
        
        symbol_data['StochRSI'] = talib.STOCHRSI(symbol_data['Close'], 
                                               timeperiod=14, 
                                               fastk_period=5, 
                                               fastd_period=3)[0]
        
        # Add Ichimoku Cloud indicators
        high_9 = symbol_data['High'].rolling(window=9).max()
        low_9 = symbol_data['Low'].rolling(window=9).min()
        symbol_data['Tenkan_Sen'] = (high_9 + low_9) / 2
        
        high_26 = symbol_data['High'].rolling(window=26).max()
        low_26 = symbol_data['Low'].rolling(window=26).min()
        symbol_data['Kijun_Sen'] = (high_26 + low_26) / 2
        
        symbol_data['Senkou_Span_A'] = ((symbol_data['Tenkan_Sen'] + symbol_data['Kijun_Sen']) / 2).shift(26)
        
        high_52 = symbol_data['High'].rolling(window=52).max()
        low_52 = symbol_data['Low'].rolling(window=52).min()
        symbol_data['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
        
        symbol_data['Chikou_Span'] = symbol_data['Close'].shift(-26)
        
        # Add liquidity metrics
        if 'Volume' in symbol_data.columns:
            # Relative volume compared to 20-day average
            symbol_data['Relative_Volume'] = symbol_data['Volume'] / symbol_data['Volume'].rolling(window=20).mean()
            
            # Money flow metrics
            typical_price = (symbol_data['High'] + symbol_data['Low'] + symbol_data['Close']) / 3
            money_flow = typical_price * symbol_data['Volume']
            
            # Positive and negative money flow
            pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            # Money flow ratio and index (14-period)
            pos_flow_sum = pos_flow.rolling(window=14).sum()
            neg_flow_sum = neg_flow.rolling(window=14).sum()
            symbol_data['Money_Flow_Ratio'] = pos_flow_sum / neg_flow_sum
            symbol_data['Money_Flow_Index_14'] = 100 - (100 / (1 + symbol_data['Money_Flow_Ratio']))
            
            # Chaikin Money Flow (20-period)
            money_flow_multiplier = ((symbol_data['Close'] - symbol_data['Low']) - (symbol_data['High'] - symbol_data['Close'])) / (symbol_data['High'] - symbol_data['Low'])
            money_flow_volume = money_flow_multiplier * symbol_data['Volume']
            symbol_data['Chaikin_Money_Flow_20'] = money_flow_volume.rolling(window=20).sum() / symbol_data['Volume'].rolling(window=20).sum()
        
        # Update the main dataframe
        if symbol:
            # Update the specific rows for this symbol with the new columns
            mask = result['symbol'] == symbol
            for col in symbol_data.columns:
                if col not in result.columns:
                    result[col] = np.nan
                result.loc[mask, col] = symbol_data[col].values
        else:
            result = symbol_data
            
    return result

@dataclass
class IndicatorSignal:
    """Represents a signal from a technical indicator"""
    indicator: str
    signal_type: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0 to 1
    timeframe: str  # '5m', '15m', '1h', '1d', etc.
    price_level: float
    description: str
    confidence: float  # 0 to 1

@dataclass
class MarketRegime:
    """Represents the current market regime"""
    trend: str  # 'uptrend', 'downtrend', 'sideways'
    volatility: str  # 'low', 'normal', 'high'
    volume: str  # 'low', 'normal', 'high'
    momentum: str  # 'strong', 'weak', 'neutral'
    confidence: float  # 0 to 1

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
                'adx': ['ADX', 'DI_PLUS', 'DI_MINUS'],
                'ichimoku': ['TENKAN', 'KIJUN', 'SENKOU_A', 'SENKOU_B', 'CHIKOU'],
                'supertrend': ['SUPERTREND', 'SUPERTREND_DIRECTION']
            },
            'momentum': {
                'rsi': ['RSI'],
                'stoch': ['STOCH_K', 'STOCH_D'],
                'cci': ['CCI'],
                'williams_r': ['WILLR'],
                'roc': ['ROC'],
                'mfi': ['MFI']
            },
            'volume': {
                'standard': ['OBV', 'AD'],
                'advanced': ['CMF', 'VWAP', 'EMV', 'VQI'],
                'price_volume': ['PVT', 'NVI', 'PVI']
            },
            'volatility': {
                'bands': ['BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER'],
                'atr': ['ATR', 'ATR_PERCENT'],
                'standard_dev': ['STD20'],
                'keltner': ['KC_UPPER', 'KC_MIDDLE', 'KC_LOWER']
            },
            'custom': {
                'regime': ['TREND_STRENGTH', 'VOL_REGIME', 'MOMENTUM_QUALITY'],
                'composite': ['VAM', 'VQI', 'TSI']
            }
        }
        
        # Timeframe configurations
        self.timeframes = {
            'intraday': ['5m', '15m', '30m', '1h', '4h'],
            'daily': ['1d'],
            'weekly': ['1w'],
            'monthly': ['1M']
        }
        
        # Market regime thresholds
        self.regime_thresholds = {
            'volatility': {
                'low': 10,  # percentile
                'high': 90  # percentile
            },
            'volume': {
                'low': 0.7,  # relative to moving average
                'high': 1.5  # relative to moving average
            },
            'trend': {
                'strong': 25,  # ADX threshold
                'neutral': 20  # ADX threshold
            }
        }
    
    def compute_all_indicators(self, df: pd.DataFrame, timeframe: str = '1d') -> pd.DataFrame:
        """Compute all technical indicators for a given timeframe."""
        
        # Extract price data
        close = df['Close'].iloc[:, 0].to_numpy(dtype=np.float64)
        high = df['High'].iloc[:, 0].to_numpy(dtype=np.float64)
        low = df['Low'].iloc[:, 0].to_numpy(dtype=np.float64)
        open_price = df['Open'].iloc[:, 0].to_numpy(dtype=np.float64)
        volume = df['Volume'].iloc[:, 0].to_numpy(dtype=np.float64)
        
        # 1. Trend Indicators
        df['SMA20'] = talib.SMA(close, timeperiod=20)
        df['SMA50'] = talib.SMA(close, timeperiod=50)
        df['SMA200'] = talib.SMA(close, timeperiod=200)
        df['EMA20'] = talib.EMA(close, timeperiod=20)
        df['EMA50'] = talib.EMA(close, timeperiod=50)
        df['EMA200'] = talib.EMA(close, timeperiod=200)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(close)
        
        # ADX and DI
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['DI_PLUS'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['DI_MINUS'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Ichimoku Cloud
        df['TENKAN'] = self._ichimoku_conversion(high, low, 9)
        df['KIJUN'] = self._ichimoku_conversion(high, low, 26)
        df['SENKOU_A'] = (df['TENKAN'] + df['KIJUN']) / 2
        df['SENKOU_B'] = self._ichimoku_conversion(high, low, 52)
        df['CHIKOU'] = pd.Series(close).shift(-26)
        
        # 2. Momentum Indicators
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=20)
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        df['ROC'] = talib.ROC(close, timeperiod=10)
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # 3. Volume Indicators
        df['OBV'] = talib.OBV(close, volume)
        df['AD'] = talib.AD(high, low, close, volume)
        df['CMF'] = self._chaikin_money_flow(high, low, close, volume, period=20)
        df['VWAP'] = self._vwap(high, low, close, volume)
        df['EMV'] = self._calculate_emv(high, low, volume)
        df['VQI'] = self._calculate_vqi(df, lookback=20)
        
        # 4. Volatility Indicators
        df['BBANDS_UPPER'], df['BBANDS_MIDDLE'], df['BBANDS_LOWER'] = talib.BBANDS(close)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['ATR_PERCENT'] = (df['ATR'] / close) * 100
        df['STD20'] = talib.STDDEV(close, timeperiod=20)
        
        # Keltner Channels
        typical_price = (high + low + close) / 3
        df['KC_MIDDLE'] = talib.EMA(typical_price, timeperiod=20)
        atr = talib.ATR(high, low, close, timeperiod=20)
        df['KC_UPPER'] = df['KC_MIDDLE'] + (2 * atr)
        df['KC_LOWER'] = df['KC_MIDDLE'] - (2 * atr)
        
        # 5. Advanced Adaptive Indicators
        df['FRAMA'] = self._calculate_frama(close, 16)  # Fractal Adaptive Moving Average
        df['KAMA'] = self._calculate_kama(close, 10, 2, 30)  # Kaufman Adaptive Moving Average  
        df['ZLEMA'] = self._calculate_zlema(close, 21)  # Zero Lag Exponential Moving Average
        df['VIDYA'] = self._calculate_vidya(close, 14)  # Variable Index Dynamic Average
        
        # 6. Custom Composite Indicators
        df['TREND_STRENGTH'] = self._calculate_trend_strength(df)
        df['VOL_REGIME'] = self._calculate_volatility_regime(df)
        df['MOMENTUM_QUALITY'] = self._calculate_momentum_quality(df)
        df['VAM'] = self._calculate_vam(df)  # Volatility Adjusted Momentum
        df['TSI'] = self._calculate_tsi(df)  # Trend Strength Index
        
        # Add market regime detection
        regime = self.detect_market_regime(df)
        df['MARKET_TREND'] = regime.trend
        df['MARKET_VOLATILITY'] = regime.volatility
        df['MARKET_VOLUME'] = regime.volume
        df['MARKET_MOMENTUM'] = regime.momentum
        df['REGIME_CONFIDENCE'] = regime.confidence
        
        return df
    
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime using multiple indicators."""
        # Trend Analysis
        adx = df['ADX'].iloc[-1]
        di_plus = df['DI_PLUS'].iloc[-1]
        di_minus = df['DI_MINUS'].iloc[-1]
        
        if adx > self.regime_thresholds['trend']['strong']:
            trend = 'uptrend' if di_plus > di_minus else 'downtrend'
        elif adx < self.regime_thresholds['trend']['neutral']:
            trend = 'sideways'
        else:
            trend = 'undefined'
        
        # Volatility Analysis
        current_atr = df['ATR_PERCENT'].iloc[-1]
        atr_percentile = stats.percentileofscore(df['ATR_PERCENT'].dropna(), current_atr)
        
        if atr_percentile > self.regime_thresholds['volatility']['high']:
            volatility = 'high'
        elif atr_percentile < self.regime_thresholds['volatility']['low']:
            volatility = 'low'
        else:
            volatility = 'normal'
        
        # Volume Analysis
        volume_ma = df['Volume'].rolling(window=20).mean()
        relative_volume = df['Volume'].iloc[-1] / volume_ma.iloc[-1]
        
        if relative_volume > self.regime_thresholds['volume']['high']:
            volume = 'high'
        elif relative_volume < self.regime_thresholds['volume']['low']:
            volume = 'low'
        else:
            volume = 'normal'
        
        # Momentum Analysis
        rsi = df['RSI'].iloc[-1]
        mfi = df['MFI'].iloc[-1]
        macd_hist = df['MACD_Hist'].iloc[-1]
        
        if (rsi > 60 and mfi > 60) or macd_hist > 0:
            momentum = 'strong'
        elif (rsi < 40 and mfi < 40) or macd_hist < 0:
            momentum = 'weak'
        else:
            momentum = 'neutral'
        
        # Calculate regime confidence
        trend_conf = min(adx / 100, 1.0)
        vol_conf = abs(atr_percentile - 50) / 50
        momentum_conf = abs(rsi - 50) / 50
        
        confidence = (trend_conf + vol_conf + momentum_conf) / 3
        
        return MarketRegime(
            trend=trend,
            volatility=volatility,
            volume=volume,
            momentum=momentum,
            confidence=confidence
        )
    
    def _calculate_vqi(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Calculate Volume Quality Index."""
        # Price-volume correlation
        price_change = df['Close'].pct_change()
        volume_relative = df['Volume'] / df['Volume'].rolling(lookback).mean()
        
        # Volume trend from OBV
        obv_trend = df['OBV'].diff().rolling(lookback).mean()
        
        # Volume consistency
        volume_std = df['Volume'].rolling(lookback).std() / df['Volume'].rolling(lookback).mean()
        
        # Combine into volume quality
        vqi = (0.4 * np.sign(price_change) * np.sign(volume_relative - 1) +
               0.4 * np.sign(obv_trend) +
               0.2 * (1 - volume_std))
        
        return vqi
    
    def _calculate_emv(self, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Ease of Movement (EMV)"""
        high_low = (high + low) / 2
        move = high_low - np.roll(high_low, 1)
        ratio = volume / (high - low)
        emv = move / ratio
        return pd.Series(emv).rolling(window=14).mean()
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite trend strength indicator."""
        # Directional movement
        ema_direction = np.sign(df['EMA20'] - df['EMA50'])
        
        # ADX strength scaling
        adx_strength = df['ADX'] / 100
        
        # MACD signal
        macd_signal = np.sign(df['MACD'] - df['MACD_Signal'])
        
        # Combine indicators
        trend_strength = (0.4 * ema_direction + 
                        0.4 * adx_strength * np.sign(df['DI_PLUS'] - df['DI_MINUS']) + 
                        0.2 * macd_signal)
        
        return trend_strength
    
    def _calculate_vam(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volatility Adjusted Momentum."""
        # Price momentum
        returns = df['Close'].pct_change()
        momentum = returns.rolling(window=14).mean()
        
        # Volatility scaling
        volatility = returns.rolling(window=14).std()
        
        # Adjust momentum by volatility
        vam = momentum / volatility
        
        return vam

    def _ichimoku_conversion(self, high: np.ndarray, low: np.ndarray, period: int) -> pd.Series:
        """Calculate Ichimoku conversion line."""
        period_high = pd.Series(high).rolling(window=period).max()
        period_low = pd.Series(low).rolling(window=period).min()
        return (period_high + period_low) / 2

    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """Calculate Supertrend indicator."""
        atr = talib.ATR(df['High'].iloc[:, 0], df['Low'].iloc[:, 0], df['Close'].iloc[:, 0], timeperiod=period)
        
        # Calculate basic upper and lower bands
        basic_upperband = (df['High'].iloc[:, 0] + df['Low'].iloc[:, 0]) / 2 + multiplier * atr
        basic_lowerband = (df['High'].iloc[:, 0] + df['Low'].iloc[:, 0]) / 2 - multiplier * atr
        
        # Initialize final upper and lower bands
        final_upperband = basic_upperband.copy()
        final_lowerband = basic_lowerband.copy()
        
        for i in range(1, len(df)):
            if basic_upperband[i] < final_upperband[i-1] and df['Close'].iloc[i-1, 0] <= final_upperband[i-1]:
                final_upperband[i] = final_upperband[i-1]
            if basic_lowerband[i] > final_lowerband[i-1] and df['Close'].iloc[i-1, 0] >= final_lowerband[i-1]:
                final_lowerband[i] = final_lowerband[i-1]
                
        # Calculate Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(len(df)):
            if i == 0:
                supertrend[i] = final_upperband[i]
                direction[i] = 1
            else:
                if supertrend[i-1] == final_upperband[i-1]:
                    if df['Close'].iloc[i, 0] > final_upperband[i]:
                        supertrend[i] = final_lowerband[i]
                        direction[i] = -1
                    else:
                        supertrend[i] = final_upperband[i]
                        direction[i] = 1
                else:
                    if df['Close'].iloc[i, 0] < final_lowerband[i]:
                        supertrend[i] = final_upperband[i]
                        direction[i] = 1
                    else:
                        supertrend[i] = final_lowerband[i]
                        direction[i] = -1
        
        return supertrend, direction

    def _chaikin_money_flow(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                        volume: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Chaikin Money Flow"""
        high_low = high - low
        high_low[high_low == 0] = 0.01  # Avoid division by zero
        mf_multiplier = ((close - low) - (high - close)) / high_low
        mf_volume = mf_multiplier * volume
        cmf = pd.Series(mf_volume).rolling(window=period).sum() / pd.Series(volume).rolling(window=period).sum()
        return cmf

    def _vwap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
          volume: np.ndarray) -> np.ndarray:
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = pd.Series(typical_price * volume).cumsum() / pd.Series(volume).cumsum()
        return vwap

    def _supertrend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                period: int = 10, multiplier: float = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate SuperTrend indicator"""
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        # Basic Upper and Lower Bands
        basic_upper = (high + low) / 2 + (multiplier * atr)
        basic_lower = (high + low) / 2 - (multiplier * atr)
        
        # Initialize SuperTrend
        supertrend = np.zeros_like(close)
        direction = np.zeros_like(close)  # 1 for uptrend, -1 for downtrend
        
        # First value
        if close[0] <= basic_upper[0]:
            supertrend[0] = basic_upper[0]
            direction[0] = -1
        else:
            supertrend[0] = basic_lower[0]
            direction[0] = 1
        
        # Calculate SuperTrend
        for i in range(1, len(close)):
            if close[i-1] <= supertrend[i-1]:  # Downtrend
                supertrend[i] = min(basic_upper[i], supertrend[i-1])
                if close[i] > supertrend[i]:
                    direction[i] = 1
                else:
                    direction[i] = -1
            else:  # Uptrend
                supertrend[i] = max(basic_lower[i], supertrend[i-1])
                if close[i] < supertrend[i]:
                    direction[i] = -1
                else:
                    direction[i] = 1
        
        return supertrend, direction

    def _demark_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate DeMark Setup and Countdown"""
        setup = np.zeros_like(close)
        countdown = np.zeros_like(close)
        
        # Setup calculation
        for i in range(4, len(close)):
            if close[i] > close[i-4]:
                setup[i] = min(setup[i-1] + 1, 9) if setup[i-1] > 0 else 1
            elif close[i] < close[i-4]:
                setup[i] = max(setup[i-1] - 1, -9) if setup[i-1] < 0 else -1
            else:
                setup[i] = 0
        
        # Countdown calculation
        setup_complete = False
        count = 0
        for i in range(4, len(close)):
            if setup[i] >= 9:
                if not setup_complete:
                    setup_complete = True
                    count = 0
                if close[i] > high[i-2]:
                    count += 1
                    countdown[i] = min(count, 13)
            elif setup[i] <= -9:
                if not setup_complete:
                    setup_complete = True
                    count = 0
                if close[i] < low[i-2]:
                    count += 1
                    countdown[i] = min(count, -13)
        
        return setup, countdown

    def _add_pivot_points(self, df: pd.DataFrame) -> None:
        """Add pivot points and support/resistance levels"""
        high = df['High'].iloc[:, 0]
        low = df['Low'].iloc[:, 0]
        close = df['Close'].iloc[:, 0]
        
        # Calculate Pivot Point (PP)
        df['PP'] = (high + low + close) / 3
        
        # Calculate Support and Resistance levels
        df['R1'] = (2 * df['PP']) - low
        df['S1'] = (2 * df['PP']) - high
        df['R2'] = df['PP'] + (high - low)
        df['S2'] = df['PP'] - (high - low)
        df['R3'] = high + 2 * (df['PP'] - low)
        df['S3'] = low - 2 * (high - df['PP'])

    def _add_fibonacci_levels(self, df: pd.DataFrame) -> None:
        """Add Fibonacci retracement levels"""
        high = df['High'].iloc[:, 0]
        low = df['Low'].iloc[:, 0]
        
        # Calculate range
        price_range = high - low
        
        # Calculate Fibonacci levels
        df['FIB_382'] = high - (price_range * 0.382)
        df['FIB_500'] = high - (price_range * 0.500)
        df['FIB_618'] = high - (price_range * 0.618)