"""
Integrated NIFTY 500 Trading Strategy

Core Features:
1. Multi-timeframe analysis
2. Volume-price correlation
3. Risk-adjusted position sizing
4. Dynamic stop loss management
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Tuple, List
from strategies.base_strategy import Signal
from backtesting.base_strategy import BaseStrategy
from technical_indicators import TechnicalIndicators
import logging

logger = logging.getLogger(__name__)

class Nifty500Strategy(BaseStrategy):
    def __init__(self,
                lookback: int = 20,
                stop_loss_pct: float = 0.05,
                take_profit_pct: float = 0.15,
                volume_threshold: float = 1.5,
                risk_per_trade: float = 0.02,  # 2% risk per trade
                max_sector_exposure: float = 0.20):  # 20% max sector exposure
        
        super().__init__()
        self.lookback = lookback
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.volume_threshold = volume_threshold
        self.risk_per_trade = risk_per_trade
        self.max_sector_exposure = max_sector_exposure
        
        # Initialize technical indicators
        self.indicators = TechnicalIndicators()
        
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal for a single stock"""
        # Skip if not enough data
        if len(data) < self.lookback:
            return Signal(
                timestamp=data.index[-1],
                symbol="",  # Symbol not provided in single-stock case
                action="hold",
                confidence=0,
                price=data['Close'].iloc[-1],
                stop_loss=0,
                take_profit=0,
                position_size=0,
                metrics={}
            )
        
        # Use the provided data directly without recalculating indicators
        df_with_indicators = data
        
        # If some columns are missing, use defaults
        has_macd = 'MACD' in df_with_indicators.columns and 'MACD_Signal' in df_with_indicators.columns
        has_adx = 'ADX' in df_with_indicators.columns
        has_rsi = 'RSI' in df_with_indicators.columns
        has_stoch = 'STOCH_K' in df_with_indicators.columns and 'STOCH_D' in df_with_indicators.columns
        has_cci = 'CCI' in df_with_indicators.columns
        has_bb = 'BB_Upper' in df_with_indicators.columns and 'BB_Lower' in df_with_indicators.columns
        has_atr = 'ATR' in df_with_indicators.columns
        
        # Analyze components with simplified fallbacks when indicators are missing
        trend_score = 0
        if has_macd and has_adx:
            trend_score = self._analyze_trend(df_with_indicators)
        else:
            # Simple trend based on price movement
            sma20 = df_with_indicators['Close'].rolling(20).mean().iloc[-1] if len(df_with_indicators) >= 20 else df_with_indicators['Close'].mean()
            sma50 = df_with_indicators['Close'].rolling(50).mean().iloc[-1] if len(df_with_indicators) >= 50 else df_with_indicators['Close'].mean()
            trend_score = 1 if sma20 > sma50 else -1
            trend_score *= 0.7  # Reduce confidence for simplified calculation
        
        momentum_score = 0
        if has_rsi and has_stoch and has_cci:
            momentum_score = self._analyze_momentum(df_with_indicators)
        else:
            # Simple momentum based on returns
            returns = df_with_indicators['Close'].pct_change(5).iloc[-1] * 20  # Normalized to -1 to 1 range
            momentum_score = np.clip(returns, -1, 1)
        
        volume_score = 0
        if 'Volume' in df_with_indicators.columns and 'OBV' in df_with_indicators.columns:
            volume_score = self._analyze_volume(df_with_indicators)
        else:
            # Neutral volume score if volume data is missing
            volume_score = 0
        
        volatility_score = 0
        risk_metrics = {}
        if has_atr and has_bb:
            volatility_score, risk_metrics = self._analyze_volatility(df_with_indicators)
        else:
            # Simple volatility based on std
            volatility = df_with_indicators['Close'].pct_change().rolling(20).std().iloc[-1] if len(df_with_indicators) >= 20 else 0.02
            volatility_score = 1 - min(volatility * 20, 1)  # Lower volatility is better
            risk_metrics = {'atr': volatility * df_with_indicators['Close'].iloc[-1]}
        
        # Combine signals
        total_score = self._combine_signals(
            trend_score, momentum_score, volume_score, volatility_score
        )
        
        current_price = data['Close'].iloc[-1]
        
        # Determine action based on score
        if total_score > 0.6:
            action = "buy"
            confidence = total_score
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            position_size = self._calculate_position_size(
                current_price, 
                risk_metrics['atr'], 
                abs(total_score)
            )
        elif total_score < -0.6:
            action = "sell"
            confidence = abs(total_score)
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
            position_size = self._calculate_position_size(
                current_price, 
                risk_metrics['atr'], 
                abs(total_score)
            )
        else:
            action = "hold"
            confidence = abs(total_score)
            stop_loss = 0
            take_profit = 0
            position_size = 0
            
        return Signal(
            timestamp=data.index[-1],
            symbol="",  # Symbol not provided in single-stock case
            action=action,
            confidence=confidence,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            metrics={
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'volatility_score': volatility_score,
                'total_score': total_score
            }
        )
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate trading signals for all symbols"""
        signals = []
        
        try:
            for symbol, df in data.items():
                if len(df) < self.lookback:
                    signals.append(self._create_neutral_signal(df, symbol))
                    continue
                
                # Calculate technical indicators
                df_with_indicators = self.indicators.calculate_all(df.copy())
                
                # Analyze components
                trend_score = self._analyze_trend(df_with_indicators)
                momentum_score = self._analyze_momentum(df_with_indicators)
                volume_score = self._analyze_volume(df_with_indicators)
                volatility_score, risk_metrics = self._analyze_volatility(df_with_indicators)
                
                # Combine signals
                total_score = self._combine_signals(
                    trend_score, momentum_score, volume_score, volatility_score
                )
                
                current_price = df['Close'].iloc[-1]
                timestamp = df.index[-1]
                
                if abs(total_score) > 0.6:  # Signal threshold
                    direction = 1 if total_score > 0 else -1
                    size = self._calculate_position_size(
                        current_price,
                        risk_metrics['atr'],
                        abs(total_score)
                    )
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=direction,
                        size=size,
                        type='ENTRY',
                        reason=f'Combined score: {total_score:.2f}',
                        price=current_price,
                        params={
                            'trend_score': trend_score,
                            'momentum_score': momentum_score,
                            'volume_score': volume_score,
                            'volatility_score': volatility_score,
                            'stop_loss': current_price * (1 - direction * self.stop_loss_pct),
                            'take_profit': current_price * (1 + direction * self.take_profit_pct)
                        }
                    ))
                else:
                    signals.append(self._create_neutral_signal(df, symbol))
                
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            # Return neutral signals on error
            for symbol in data.keys():
                signals.append(self._create_neutral_signal(data[symbol], symbol))
        
        return signals
    
    def _create_neutral_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        """Create a neutral (no trade) signal"""
        return Signal(
            timestamp=data.index[-1],
            symbol=symbol,
            direction=0,
            size=0,
            type='NEUTRAL',
            reason='No clear signal or insufficient data',
            price=data['Close'].iloc[-1],
            params=None
        )
    
    def _analyze_trend(self, data: pd.DataFrame) -> float:
        """Analyze trend strength and direction"""
        try:
            # EMA trend alignment
            ema_20 = data['Close'].ewm(span=20, adjust=False).mean()
            ema_50 = data['Close'].ewm(span=50, adjust=False).mean()
            ema_200 = data['Close'].ewm(span=200, adjust=False).mean()
            
            trend_aligned = (ema_20 > ema_50).iloc[-1] and (ema_50 > ema_200).iloc[-1]
            
            # ADX for trend strength (> 25 indicates strong trend)
            adx = data['ADX'].iloc[-1]
            trend_strength = min(adx / 50, 1.0)  # Normalize to 0-1
            
            # MACD trend
            macd_trend = 1 if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else -1
            
            # Combine scores
            base_score = trend_strength * macd_trend
            if trend_aligned:
                base_score *= 1.2  # Boost score if EMAs aligned
                
            return np.clip(base_score, -1, 1)
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return 0.0
            
    def _analyze_momentum(self, data: pd.DataFrame) -> float:
        """Analyze price momentum"""
        try:
            # RSI
            rsi = data['RSI'].iloc[-1]
            rsi_score = (rsi - 50) / 50  # Normalize to -1 to 1
            
            # Stochastic
            stoch_k = data['STOCH_K'].iloc[-1]
            stoch_d = data['STOCH_D'].iloc[-1]
            stoch_score = (stoch_k - 50) / 50
            
            # CCI
            cci = data['CCI'].iloc[-1]
            cci_score = np.clip(cci / 100, -1, 1)
            
            # Combine momentum signals with weights
            momentum_score = (
                0.4 * rsi_score +
                0.3 * stoch_score +
                0.3 * cci_score
            )
            
            return np.clip(momentum_score, -1, 1)
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return 0.0
            
    def _analyze_volume(self, data: pd.DataFrame) -> float:
        """Analyze volume patterns"""
        try:
            # Volume surge analysis
            volume_ma = data['Volume'].rolling(20).mean()
            volume_ratio = data['Volume'].iloc[-1] / volume_ma.iloc[-1]
            volume_surge = (volume_ratio - 1) / self.volume_threshold
            
            # OBV trend
            obv = data['OBV'].diff()
            obv_trend = 1 if obv.iloc[-1] > 0 else -1
            
            # Volume-price correlation
            price_changes = data['Close'].pct_change()
            volume_changes = data['Volume'].pct_change()
            corr = price_changes.corr(volume_changes)
            
            # Combine volume signals
            volume_score = (
                0.4 * np.clip(volume_surge, -1, 1) +
                0.3 * obv_trend +
                0.3 * np.clip(corr, -1, 1)
            )
            
            return np.clip(volume_score, -1, 1)
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return 0.0
            
    def _analyze_volatility(self, data: pd.DataFrame) -> Tuple[float, Dict]:
        """Analyze volatility conditions"""
        try:
            # ATR calculation
            atr = data['ATR'].iloc[-1]
            atr_pct = atr / data['Close'].iloc[-1]
            
            # Bollinger Bands
            bb_width = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            bb_position = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # Volatility score (-1 to 1, positive means good entry volatility)
            volatility_score = np.clip(1 - (atr_pct * 100), -1, 1)  # Lower volatility is better for our strategy
            
            risk_metrics = {
                'atr': atr,
                'atr_pct': atr_pct,
                'bb_width': bb_width.iloc[-1],
                'bb_position': bb_position.iloc[-1]
            }
            
            return volatility_score, risk_metrics
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return 0.0, {'atr': 0, 'atr_pct': 0, 'bb_width': 0, 'bb_position': 0}
            
    def _combine_signals(self,
                       trend_score: float,
                       momentum_score: float,
                       volume_score: float,
                       volatility_score: float) -> float:
        """Combine all signals into final score"""
        # Weights for each component
        weights = {
            'trend': 0.35,
            'momentum': 0.25,
            'volume': 0.25,
            'volatility': 0.15
        }
        
        final_score = (
            weights['trend'] * trend_score +
            weights['momentum'] * momentum_score +
            weights['volume'] * volume_score +
            weights['volatility'] * volatility_score
        )
        
        return np.clip(final_score, -1, 1)
        
    def _calculate_position_size(self,
                              current_price: float,
                              atr: float,
                              signal_confidence: float) -> float:
        """Calculate position size using Kelly Criterion and ATR"""
        # Kelly position sizing
        win_rate = signal_confidence  # Use signal confidence as win rate
        risk_reward = self.take_profit_pct / self.stop_loss_pct
        kelly_fraction = win_rate - ((1 - win_rate) / risk_reward)
        
        # Limit maximum position size
        max_position = self.risk_per_trade / (atr / current_price)
        kelly_position = kelly_fraction * max_position
        
        return np.clip(kelly_position, 0, max_position)
