import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yfinance as yf
from technical_strategy import MultiStrategyTrader
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import threading
from queue import Queue
import warnings
from sklearn.preprocessing import StandardScaler
from typing import Optional
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConformalPredictor:
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        self.calibration_scores = []
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model and calibrate prediction intervals"""
        # Split data for training and calibration
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_cal_scaled = self.scaler.transform(X_cal)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate nonconformity scores on calibration set
        cal_predictions = self.model.predict(X_cal_scaled)
        self.calibration_scores = np.abs(y_cal - cal_predictions)
        
    def predict(self, X: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions with confidence intervals"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Calculate prediction intervals
        q = np.quantile(self.calibration_scores, confidence)
        lower = predictions - q
        upper = predictions + q
        
        return predictions, lower, upper

class RealtimeMonitor:
    def __init__(self, symbols: List[str], update_interval: int = 5):
        """
        Initialize real-time monitor
        Args:
            symbols: List of stock symbols to monitor
            update_interval: Update interval in minutes
        """
        self.symbols = symbols
        self.update_interval = update_interval
        self.trader = MultiStrategyTrader(min_confidence=0.6)
        self.conformal_predictors = {}
        self.signal_queue = Queue()
        self.stop_event = threading.Event()
        self.history = {}  # Store historical predictions and actual outcomes
        
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for prediction"""
        features = []
        
        # Technical indicators
        close = data['Close'].values
        volume = data['Volume'].values
        high = data['High'].values
        low = data['Low'].values
        
        # Price-based features
        features.extend([
            np.mean(close[-20:]),  # 20-period MA
            np.std(close[-20:]),   # 20-period volatility
            (close[-1] - close[-20]) / close[-20],  # 20-period return
            (close[-1] - low[-20:].min()) / close[-1],  # Distance from low
            (high[-20:].max() - close[-1]) / close[-1],  # Distance from high
            (high[-1] - low[-1]) / low[-1],  # Daily volatility
            np.mean([(high[i] - low[i]) / low[i] for i in range(-5, 0)]),  # Average volatility
        ])
        
        # Volume-based features
        features.extend([
            np.mean(volume[-20:]),  # Average volume
            volume[-1] / np.mean(volume[-20:]),  # Relative volume
            np.corrcoef(close[-20:], volume[-20:])[0, 1],  # Price-volume correlation
        ])
        
        # Momentum features
        returns = np.diff(close) / close[:-1]
        features.extend([
            np.mean(returns[-5:]),  # Short-term momentum
            np.mean(returns[-20:]),  # Medium-term momentum
            np.sum(returns[-5:] > 0) / 5,  # Up-day ratio
        ])
        
        return np.array(features)
        
    def train_predictors(self, lookback_days: int = 365):
        """Train conformal predictors for each symbol"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for symbol in self.symbols:
            try:
                # Fetch historical data
                stock = yf.Ticker(symbol)
                data = stock.history(start=start_date, end=end_date, interval='5m')
                
                if len(data) < 100:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Prepare features and targets
                X = []
                y = []
                
                for i in range(20, len(data)-1):
                    window = data.iloc[:i]
                    features = self._prepare_features(window)
                    target = (data.iloc[i+1]['Close'] - data.iloc[i]['Close']) / data.iloc[i]['Close']
                    
                    X.append(features)
                    y.append(target)
                
                X = np.array(X)
                y = np.array(y)
                
                # Train predictor
                predictor = ConformalPredictor()
                predictor.fit(X, y)
                self.conformal_predictors[symbol] = predictor
                
                logger.info(f"Trained predictor for {symbol}")
                
                # Initialize history storage
                self.history[symbol] = {
                    'predictions': [],
                    'actuals': [],
                    'confidence_intervals': []
                }
                
            except Exception as e:
                logger.error(f"Error training predictor for {symbol}: {e}")
                
    def _monitor_symbol(self, symbol: str):
        """Monitor a single symbol"""
        try:
            # Fetch recent data
            stock = yf.Ticker(symbol)
            data = stock.history(period='1d', interval='5m')
            
            if len(data) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                return
            
            # Get trading signals
            analysis = self.trader.analyze(data)
            
            if analysis is None:
                return
                
            # Get price prediction if predictor exists
            prediction_info = {}
            if symbol in self.conformal_predictors:
                features = self._prepare_features(data)
                pred, lower, upper = self.conformal_predictors[symbol].predict(
                    features.reshape(1, -1)
                )
                prediction_info = {
                    'predicted_return': pred[0],
                    'lower_bound': lower[0],
                    'upper_bound': upper[0],
                    'timestamp': datetime.now()
                }
                
                # Store prediction
                self.history[symbol]['predictions'].append(prediction_info)
                
                # Update actual returns for previous predictions
                self._update_prediction_accuracy(symbol, data)
            
            # Combine signals and predictions
            signal = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': data['Close'].iloc[-1],
                'action': analysis['action'],
                'confidence': analysis['confidence'],
                'stop_loss': analysis['stop_loss'],
                'take_profit': analysis['take_profit'],
                'prediction': prediction_info,
                'strategy_signals': analysis['signals']
            }
            
            # Add to queue
            self.signal_queue.put(signal)
            
        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {e}")
            
    def _update_prediction_accuracy(self, symbol: str, current_data: pd.DataFrame):
        """Update prediction accuracy metrics"""
        if symbol not in self.history:
            return
            
        history = self.history[symbol]
        current_price = current_data['Close'].iloc[-1]
        
        # Update actuals for previous predictions
        for pred in history['predictions']:
            if 'actual_return' not in pred and pred['timestamp'] < datetime.now() - timedelta(minutes=5):
                # Find the actual return
                try:
                    timestamp = pred['timestamp']
                    historical_price = current_data.loc[current_data.index <= timestamp].iloc[-1]['Close']
                    actual_return = (current_price - historical_price) / historical_price
                    pred['actual_return'] = actual_return
                    history['actuals'].append(actual_return)
                    
                    # Store confidence interval performance
                    in_interval = (actual_return >= pred['lower_bound'] and 
                                actual_return <= pred['upper_bound'])
                    history['confidence_intervals'].append(in_interval)
                except Exception as e:
                    logger.error(f"Error updating prediction accuracy: {e}")
                    
    def get_prediction_metrics(self, symbol: str) -> Optional[Dict]:
        """Get prediction performance metrics for a symbol"""
        if symbol not in self.history:
            return None
            
        history = self.history[symbol]
        predictions = [p['predicted_return'] for p in history['predictions'] 
                     if 'actual_return' in p]
        actuals = [p['actual_return'] for p in history['predictions'] 
                  if 'actual_return' in p]
        
        if not predictions or not actuals:
            return None
            
        # Calculate metrics
        mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        interval_coverage = np.mean(history['confidence_intervals']) if history['confidence_intervals'] else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'interval_coverage': interval_coverage,
            'total_predictions': len(predictions)
        }
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        logger.info("Starting real-time monitoring...")
        
        while not self.stop_event.is_set():
            threads = []
            
            # Create threads for each symbol
            for symbol in self.symbols:
                thread = threading.Thread(target=self._monitor_symbol, args=(symbol,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Process signals
            signals = []
            while not self.signal_queue.empty():
                signals.append(self.signal_queue.get())
            
            if signals:
                self._process_signals(signals)
            
            # Wait for next update
            time.sleep(self.update_interval * 60)
            
    def _process_signals(self, signals: List[Dict]):
        """Process and display trading signals with enhanced metrics"""
        # Sort by confidence and predicted return
        def signal_score(signal):
            pred_return = signal.get('prediction', {}).get('predicted_return', 0)
            return signal['confidence'] * (1 + pred_return if pred_return else 1)
            
        signals.sort(key=signal_score, reverse=True)
        
        logger.info("\n=== Trading Signals Update ===")
        logger.info(f"Timestamp: {datetime.now()}")
        
        for signal in signals:
            if signal['action'] != 'hold' and signal['confidence'] > 0.6:
                pred_info = signal['prediction']
                pred_str = ""
                if pred_info:
                    pred_str = (
                        f"Predicted Return: {pred_info['predicted_return']:.2%} "
                        f"[{pred_info['lower_bound']:.2%}, {pred_info['upper_bound']:.2%}]"
                    )
                    
                    # Get prediction metrics
                    metrics = self.get_prediction_metrics(signal['symbol'])
                    if metrics:
                        pred_str += f"\nPrediction Metrics:"
                        pred_str += f"\n  - MSE: {metrics['mse']:.6f}"
                        pred_str += f"\n  - MAE: {metrics['mae']:.6f}"
                        pred_str += f"\n  - Interval Coverage: {metrics['interval_coverage']:.2%}"
                
                logger.info(f"\nSymbol: {signal['symbol']}")
                logger.info(f"Action: {signal['action'].upper()}")
                logger.info(f"Confidence: {signal['confidence']:.2f}")
                logger.info(f"Price: {signal['price']:.2f}")
                logger.info(f"Stop Loss: {signal['stop_loss']:.2f}")
                logger.info(f"Take Profit: {signal['take_profit']:.2f}")
                if pred_str:
                    logger.info(pred_str)
                
                # Show supporting strategies
                supporting_strategies = [s for s in signal['strategy_signals'] 
                                      if s['action'] == signal['action']]
                if supporting_strategies:
                    logger.info("\nSupporting Strategies:")
                    for strat in supporting_strategies:
                        logger.info(f"- {strat['strategy']}: {strat['confidence']:.2f}")
                        
                # Save signal to file
                self._save_signal(signal)
                
    def _save_signal(self, signal: Dict):
        """Save trading signal to file for historical analysis"""
        try:
            filename = f"signals_{datetime.now().strftime('%Y%m%d')}.json"
            signal_data = {
                'timestamp': signal['timestamp'].isoformat(),
                'symbol': signal['symbol'],
                'action': signal['action'],
                'price': signal['price'],
                'confidence': signal['confidence'],
                'prediction': signal['prediction'],
                'supporting_strategies': [s for s in signal['strategy_signals'] 
                                       if s['action'] == signal['action']]
            }
            
            with open(filename, 'a') as f:
                f.write(json.dumps(signal_data) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.stop_event.set()
        logger.info("Stopping monitoring...")

# Example usage
if __name__ == "__main__":
    # Get Nifty 500 symbols (example subset)
    nifty_symbols = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "HDFC.NS", "ITC.NS", "SBIN.NS", "BAJFINANCE.NS"
    ]
    
    # Initialize and start monitor
    monitor = RealtimeMonitor(nifty_symbols)
    
    # Train predictors
    logger.info("Training predictors...")
    monitor.train_predictors()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        logger.info("Monitoring stopped by user")