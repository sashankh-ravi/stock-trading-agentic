import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    strategy_name: str
    timestamp: pd.Timestamp

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        pass

    def calculate_support_resistance(self, data: pd.DataFrame, n_levels: int = 3) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels using multiple methods"""
        prices = data['Close'].values
        
        # Method 1: Pivot Points
        highs = data['High'].values
        lows = data['Low'].values
        pivot = (highs[-1] + lows[-1] + prices[-1]) / 3
        r1 = 2 * pivot - lows[-1]
        s1 = 2 * pivot - highs[-1]
        
        # Method 2: Fractal Analysis
        peaks = []
        troughs = []
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                peaks.append(prices[i])
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                troughs.append(prices[i])
        
        # Method 3: Density Clustering
        all_prices = np.concatenate([highs, lows])
        kde = stats.gaussian_kde(all_prices)
        x_grid = np.linspace(min(all_prices), max(all_prices), 100)
        density = kde(x_grid)
        
        # Find local maxima in density
        peaks_idx = np.where((density[1:-1] > density[:-2]) & 
                           (density[1:-1] > density[2:]))[0] + 1
        level_prices = x_grid[peaks_idx]
        
        # Combine all levels and sort
        resistance_levels = sorted(set([r1] + peaks + level_prices[level_prices > prices[-1]].tolist()))
        support_levels = sorted(set([s1] + troughs + level_prices[level_prices < prices[-1]].tolist()))
        
        # Return n strongest levels
        return (support_levels[:n_levels], resistance_levels[-n_levels:])

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, rsi_period: int = 14, ma_short: int = 20, ma_long: int = 50):
        self.rsi_period = rsi_period
        self.ma_short = ma_short
        self.ma_long = ma_long
        
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        df = data.copy()
        
        # Calculate indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=self.rsi_period)
        df['MA_short'] = talib.SMA(df['Close'], timeperiod=self.ma_short)
        df['MA_long'] = talib.SMA(df['Close'], timeperiod=self.ma_long)
        df['MACD'], df['Signal'], _ = talib.MACD(df['Close'])
        
        current_price = df['Close'].iloc[-1]
        
        # Trend strength
        atr = talib.ATR(df['High'], df['Low'], df['Close'])
        trend_strength = atr.iloc[-1] / current_price
        
        # Generate signal
        rsi = df['RSI'].iloc[-1]
        ma_cross = df['MA_short'].iloc[-1] > df['MA_long'].iloc[-1]
        macd_cross = df['MACD'].iloc[-1] > df['Signal'].iloc[-1]
        
        # Calculate confidence
        confidence_factors = [
            1 if ma_cross else 0,
            1 if macd_cross else 0,
            1 if (rsi > 50 and rsi < 70) else 0
        ]
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Determine action
        if confidence > 0.5 and ma_cross and macd_cross:
            action = 'buy'
        elif rsi > 70 or (not ma_cross and not macd_cross):
            action = 'sell'
        else:
            action = 'hold'
            
        # Calculate stop loss and take profit
        atr_multiple = 2
        stop_loss = current_price * (1 - 0.05)  # 5% stop loss
        take_profit = current_price * (1 + 0.15)  # 15% take profit
        
        return TradingSignal(
            symbol=data.name if hasattr(data, 'name') else 'Unknown',
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe='short_term',
            strategy_name='trend_following',
            timestamp=df.index[-1]
        )

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, std_dev: float = 2.0):
        self.lookback = lookback
        self.std_dev = std_dev
        
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        df = data.copy()
        
        # Calculate Bollinger Bands
        df['MA'] = talib.SMA(df['Close'], timeperiod=self.lookback)
        df['STD'] = df['Close'].rolling(window=self.lookback).std()
        df['Upper'] = df['MA'] + (self.std_dev * df['STD'])
        df['Lower'] = df['MA'] - (self.std_dev * df['STD'])
        
        # Calculate RSI for confirmation
        df['RSI'] = talib.RSI(df['Close'])
        
        current_price = df['Close'].iloc[-1]
        upper_band = df['Upper'].iloc[-1]
        lower_band = df['Lower'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        # Generate signal
        if current_price > upper_band and rsi > 70:
            action = 'sell'
            confidence = min((current_price - upper_band) / upper_band, 1)
        elif current_price < lower_band and rsi < 30:
            action = 'buy'
            confidence = min((lower_band - current_price) / lower_band, 1)
        else:
            action = 'hold'
            confidence = 0.0
            
        # Calculate stop loss and take profit
        if action == 'buy':
            stop_loss = current_price * (1 - 0.05)
            take_profit = df['MA'].iloc[-1]
        elif action == 'sell':
            stop_loss = current_price * 1.05
            take_profit = df['MA'].iloc[-1]
        else:
            stop_loss = take_profit = current_price
            
        return TradingSignal(
            symbol=data.name if hasattr(data, 'name') else 'Unknown',
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe='short_term',
            strategy_name='mean_reversion',
            timestamp=df.index[-1]
        )

class BreakoutStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        df = data.copy()
        current_price = df['Close'].iloc[-1]
        
        # Calculate support and resistance
        support_levels, resistance_levels = self.calculate_support_resistance(df)
        
        # Calculate volume indicators
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['ADL'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume confirmation
        volume_increasing = df['Volume'].iloc[-1] > df['Volume'].iloc[-2]
        obv_increasing = df['OBV'].iloc[-1] > df['OBV'].iloc[-2]
        
        # Find nearest levels
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price*1.1)
        nearest_support = max([s for s in support_levels if s < current_price], default=current_price*0.9)
        
        # Check for breakouts
        resistance_breakout = current_price > nearest_resistance and volume_increasing and obv_increasing
        support_breakdown = current_price < nearest_support and volume_increasing
        
        # Generate signal
        if resistance_breakout:
            action = 'buy'
            confidence = min((current_price - nearest_resistance) / nearest_resistance, 0.9)
            stop_loss = nearest_resistance * 0.95
            take_profit = current_price * 1.15
        elif support_breakdown:
            action = 'sell'
            confidence = min((nearest_support - current_price) / nearest_support, 0.9)
            stop_loss = nearest_support * 1.05
            take_profit = current_price * 0.85
        else:
            action = 'hold'
            confidence = 0.0
            stop_loss = take_profit = current_price
            
        return TradingSignal(
            symbol=data.name if hasattr(data, 'name') else 'Unknown',
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe='short_term',
            strategy_name='breakout',
            timestamp=df.index[-1]
        )

class PatternRecognitionStrategy(BaseStrategy):
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        df = data.copy()
        current_price = df['Close'].iloc[-1]
        
        # Calculate various candlestick patterns
        patterns = {
            'engulfing': talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close']),
            'hammer': talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close']),
            'morning_star': talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close']),
            'evening_star': talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close']),
            'doji': talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        }
        
        # Count bullish and bearish patterns
        bullish_patterns = sum(1 for p in patterns.values() if p.iloc[-1] > 0)
        bearish_patterns = sum(1 for p in patterns.values() if p.iloc[-1] < 0)
        
        # Calculate confidence based on pattern consensus
        total_patterns = len(patterns)
        bullish_confidence = bullish_patterns / total_patterns
        bearish_confidence = bearish_patterns / total_patterns
        
        # Generate signal
        if bullish_confidence > 0.3:
            action = 'buy'
            confidence = bullish_confidence
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.15
        elif bearish_confidence > 0.3:
            action = 'sell'
            confidence = bearish_confidence
            stop_loss = current_price * 1.05
            take_profit = current_price * 0.85
        else:
            action = 'hold'
            confidence = 0.0
            stop_loss = take_profit = current_price
            
        return TradingSignal(
            symbol=data.name if hasattr(data, 'name') else 'Unknown',
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe='short_term',
            strategy_name='pattern_recognition',
            timestamp=df.index[-1]
        )

class VolumeProfileStrategy(BaseStrategy):
    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins
        
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        df = data.copy()
        current_price = df['Close'].iloc[-1]
        
        # Calculate Volume Profile
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), self.n_bins)
        volume_profile = np.zeros(self.n_bins-1)
        
        for i in range(len(df)):
            idx = np.digitize(df['Close'].iloc[i], price_bins) - 1
            volume_profile[idx] += df['Volume'].iloc[i]
            
        # Find Point of Control (price level with highest volume)
        poc_idx = np.argmax(volume_profile)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        # Calculate Value Area (70% of volume)
        total_volume = volume_profile.sum()
        value_area_volume = total_volume * 0.7
        
        cumsum = 0
        value_area_indices = []
        sorted_indices = np.argsort(volume_profile)[::-1]
        
        for idx in sorted_indices:
            cumsum += volume_profile[idx]
            value_area_indices.append(idx)
            if cumsum >= value_area_volume:
                break
                
        value_area_low = price_bins[min(value_area_indices)]
        value_area_high = price_bins[max(value_area_indices) + 1]
        
        # Generate signal based on price position relative to Value Area
        if current_price < value_area_low:
            # Price below Value Area - potential buy
            action = 'buy'
            confidence = min((value_area_low - current_price) / value_area_low, 0.9)
            stop_loss = current_price * 0.95
            take_profit = poc_price
        elif current_price > value_area_high:
            # Price above Value Area - potential sell
            action = 'sell'
            confidence = min((current_price - value_area_high) / value_area_high, 0.9)
            stop_loss = current_price * 1.05
            take_profit = poc_price
        else:
            action = 'hold'
            confidence = 0.0
            stop_loss = take_profit = current_price
            
        return TradingSignal(
            symbol=data.name if hasattr(data, 'name') else 'Unknown',
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe='short_term',
            strategy_name='volume_profile',
            timestamp=df.index[-1]
        )

class MultiStrategyTrader:
    def __init__(self, min_confidence: float = 0.6):
        self.strategies = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            PatternRecognitionStrategy(),
            VolumeProfileStrategy()
        ]
        self.min_confidence = min_confidence
        
    def analyze(self, data: pd.DataFrame) -> Dict:
        signals = []
        
        # Get signals from all strategies
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(data)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.__class__.__name__}: {e}")
                
        if not signals:
            return None
            
        # Aggregate signals
        buy_signals = [s for s in signals if s.action == 'buy']
        sell_signals = [s for s in signals if s.action == 'sell']
        
        # Calculate consensus
        total_strategies = len(self.strategies)
        buy_confidence = sum(s.confidence for s in buy_signals) / total_strategies
        sell_confidence = sum(s.confidence for s in sell_signals) / total_strategies
        
        # Make final decision
        if buy_confidence > self.min_confidence and buy_confidence > sell_confidence:
            action = 'buy'
            confidence = buy_confidence
            signals_used = buy_signals
        elif sell_confidence > self.min_confidence and sell_confidence > buy_confidence:
            action = 'sell'
            confidence = sell_confidence
            signals_used = sell_signals
        else:
            action = 'hold'
            confidence = max(buy_confidence, sell_confidence)
            signals_used = signals
            
        # Calculate consensus stop loss and take profit
        if signals_used:
            stop_loss = np.median([s.stop_loss for s in signals_used])
            take_profit = np.median([s.take_profit for s in signals_used])
        else:
            stop_loss = take_profit = data['Close'].iloc[-1]
            
        return {
            'action': action,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': data.index[-1],
            'price': data['Close'].iloc[-1],
            'signals': [{'strategy': s.strategy_name, 'action': s.action, 'confidence': s.confidence} 
                       for s in signals]
        }
        
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Backtest the multi-strategy approach"""
        results = []
        position = None
        capital = initial_capital
        trades = []
        
        for i in range(100, len(data)):  # Start after warmup period
            window = data.iloc[:i+1]
            analysis = self.analyze(window)
            
            if analysis is None:
                continue
                
            current_price = data['Close'].iloc[i]
            
            # Check existing position
            if position:
                # Check stop loss and take profit
                if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                    profit = (current_price - position['entry_price']) * position['shares']
                    capital += position['shares'] * current_price
                    trades.append({
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': profit,
                        'profit_pct': (current_price - position['entry_price']) / position['entry_price'],
                        'exit_reason': 'stop_loss' if current_price <= position['stop_loss'] else 'take_profit'
                    })
                    position = None
                    
            # Enter new position
            elif analysis['action'] in ['buy', 'sell'] and analysis['confidence'] > self.min_confidence:
                risk_per_trade = capital * 0.02  # Risk 2% per trade
                position_size = risk_per_trade / (current_price * 0.05)  # Based on 5% stop loss
                shares = int(position_size)
                
                if shares > 0:
                    position = {
                        'entry_price': current_price,
                        'shares': shares,
                        'stop_loss': analysis['stop_loss'],
                        'take_profit': analysis['take_profit']
                    }
                    capital -= shares * current_price
                    
            results.append({
                'timestamp': data.index[i],
                'price': current_price,
                'capital': capital + (position['shares'] * current_price if position else 0),
                'action': analysis['action'],
                'confidence': analysis['confidence']
            })
            
        # Calculate performance metrics
        df_results = pd.DataFrame(results)
        returns = df_results['capital'].pct_change()
        
        return {
            'total_return': (df_results['capital'].iloc[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'max_drawdown': (df_results['capital'].cummax() - df_results['capital']).max() / df_results['capital'].cummax().max(),
            'win_rate': sum(1 for t in trades if t['profit'] > 0) / len(trades) if trades else 0,
            'trades': trades,
            'results': df_results
        }