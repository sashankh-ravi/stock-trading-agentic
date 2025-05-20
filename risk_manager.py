"""Risk Manager

Implements dynamic risk management with:
1. Kelly Criterion optimization
2. Dynamic position sizing
3. Adaptive stop losses
4. Portfolio-level risk control
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self,
                target_monthly_return: float = 0.15,  # 15%
                max_loss_pct: float = 0.05,          # 5%
                max_portfolio_var: float = 0.02,      # 2%
                confidence_level: float = 0.95,
                max_leverage: float = 1.5):
        """Initialize risk manager
        
        Args:
            target_monthly_return: Target monthly return
            max_loss_pct: Maximum allowed loss
            max_portfolio_var: Maximum portfolio variance
            confidence_level: VaR confidence level
            max_leverage: Maximum allowed leverage
        """
        self.target_monthly_return = target_monthly_return
        self.max_loss_pct = max_loss_pct
        self.max_portfolio_var = max_portfolio_var
        self.confidence_level = confidence_level
        self.max_leverage = max_leverage
        
        # Historical performance tracking
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        
    def calculate_position_size(self,
                             price: float,
                             stop_loss: float,
                             portfolio_value: float,
                             volatility: float,
                             win_rate: float,
                             correlation: float) -> Tuple[float, Dict]:
        """Calculate optimal position size using enhanced Kelly Criterion
        
        Returns:
            Tuple of (position_size, metrics)
        """
        try:
            # Calculate base position size using Kelly
            win_amount = self.target_monthly_return * portfolio_value
            loss_amount = self.max_loss_pct * portfolio_value
            
            kelly_fraction = (win_rate * win_amount - (1 - win_rate) * loss_amount) / win_amount
            
            # Adjust for volatility
            vol_adjustment = np.exp(-volatility)  # Reduce size when volatility is high
            
            # Adjust for correlation
            corr_adjustment = 1 - abs(correlation)  # Reduce size when correlation is high
            
            # Calculate final position size
            position_size = kelly_fraction * vol_adjustment * corr_adjustment
            
            # Apply constraints
            position_size = max(0.0, min(position_size, self.max_leverage))
            
            # Calculate actual position value
            position_value = portfolio_value * position_size
            
            # Calculate number of shares
            shares = position_value / price if price > 0 else 0
            
            metrics = {
                'kelly_fraction': kelly_fraction,
                'vol_adjustment': vol_adjustment,
                'corr_adjustment': corr_adjustment,
                'final_size': position_size,
                'shares': shares
            }
            
            return position_size, metrics
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0, {}

    def calculate_adaptive_stops(self,
                              price: float,
                              volatility: float,
                              atr: float,
                              trend_strength: float) -> Tuple[float, float]:
        """Calculate adaptive stop loss and take profit levels
        
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            # Base stops using ATR
            base_stop = 2 * atr
            base_target = 3 * atr  # 1.5:1 reward-risk ratio
            
            # Adjust for volatility
            vol_factor = np.clip(volatility / 0.2, 0.5, 2.0)  # 0.2 is average volatility
            base_stop *= vol_factor
            base_target *= vol_factor
            
            # Adjust for trend strength
            trend_factor = np.clip(1 + trend_strength, 0.5, 2.0)
            base_target *= trend_factor
            
            # Calculate final levels
            stop_loss = price - base_stop
            take_profit = price + base_target
            
            # Ensure minimum profit target
            min_target = price * (1 + self.target_monthly_return)
            take_profit = max(take_profit, min_target)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating adaptive stops: {e}")
            return price * 0.95, price * 1.15  # Default 5% stop, 15% target

    def calculate_portfolio_risk(self,
                              positions: List[Dict],
                              correlations: pd.DataFrame,
                              volatilities: pd.Series) -> Dict:
        """Calculate portfolio-level risk metrics
        
        Returns:
            Dict of risk metrics
        """
        try:
            if not positions:
                return {
                    'portfolio_var': 0,
                    'var_95': 0,
                    'expected_shortfall': 0,
                    'risk_contribution': {}
                }
            
            # Calculate portfolio weights
            total_value = sum(p['value'] for p in positions)
            weights = {p['symbol']: p['value']/total_value for p in positions}
            
            # Calculate portfolio variance
            port_var = 0
            for i, pos_i in enumerate(positions):
                sym_i = pos_i['symbol']
                for pos_j in positions:
                    sym_j = pos_j['symbol']
                    port_var += (weights[sym_i] * weights[sym_j] *
                               volatilities[sym_i] * volatilities[sym_j] *
                               correlations.loc[sym_i, sym_j])
            
            # Calculate VaR
            port_vol = np.sqrt(port_var)
            z_score = norm.ppf(self.confidence_level)
            var_95 = total_value * port_vol * z_score
            
            # Calculate Expected Shortfall (CVaR)
            es_95 = total_value * port_vol * (norm.pdf(z_score)/(1-self.confidence_level))
            
            # Calculate risk contribution of each position
            risk_contrib = {}
            for pos in positions:
                sym = pos['symbol']
                mrc = 0  # Marginal risk contribution
                for pos_j in positions:
                    sym_j = pos_j['symbol']
                    mrc += (weights[sym_j] * volatilities[sym] * volatilities[sym_j] *
                           correlations.loc[sym, sym_j])
                risk_contrib[sym] = weights[sym] * mrc / port_var if port_var > 0 else 0
            
            return {
                'portfolio_var': port_var,
                'portfolio_vol': port_vol,
                'var_95': var_95,
                'expected_shortfall': es_95,
                'risk_contribution': risk_contrib
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {
                'portfolio_var': 0,
                'var_95': 0,
                'expected_shortfall': 0,
                'risk_contribution': {}
            }

    def update_trade_history(self, trade: Dict) -> None:
        """Update trade history and recalculate metrics"""
        self.trade_history.append(trade)
        
        # Keep only recent history
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

    def get_trading_metrics(self) -> Dict:
        """Calculate trading performance metrics"""
        try:
            if not self.trade_history:
                return {}
            
            trades = pd.DataFrame(self.trade_history)
            
            # Calculate key metrics
            metrics = {
                'win_rate': len(trades[trades['pnl'] > 0]) / len(trades),
                'avg_win': trades[trades['pnl'] > 0]['pnl'].mean(),
                'avg_loss': abs(trades[trades['pnl'] < 0]['pnl'].mean()),
                'profit_factor': (trades[trades['pnl'] > 0]['pnl'].sum() /
                               abs(trades[trades['pnl'] < 0]['pnl'].sum())
                               if len(trades[trades['pnl'] < 0]) > 0 else float('inf')),
                'avg_trade': trades['pnl'].mean(),
                'std_trade': trades['pnl'].std()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {}

    def adjust_risk_parameters(self) -> None:
        """Dynamically adjust risk parameters based on performance"""
        try:
            metrics = self.get_trading_metrics()
            if not metrics:
                return
            
            # Adjust position sizing based on win rate
            if metrics['win_rate'] < 0.4:
                self.max_leverage *= 0.8  # Reduce leverage when win rate is low
            elif metrics['win_rate'] > 0.6:
                self.max_leverage = min(self.max_leverage * 1.2, 2.0)
                
            # Adjust stop loss based on average loss
            if metrics['avg_loss'] > self.max_loss_pct * 1.2:  # Loss too large
                self.max_loss_pct *= 0.9  # Tighten stops
            elif metrics['avg_loss'] < self.max_loss_pct * 0.8:  # Room for more risk
                self.max_loss_pct = min(self.max_loss_pct * 1.1, 0.07)
                
        except Exception as e:
            logger.error(f"Error adjusting risk parameters: {e}")

    def check_portfolio_health(self,
                            portfolio_value: float,
                            positions: List[Dict]) -> Tuple[bool, str]:
        """Check overall portfolio health
        
        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            # Check drawdown
            if self.portfolio_history:
                max_value = max(h['value'] for h in self.portfolio_history)
                drawdown = (portfolio_value - max_value) / max_value
                if drawdown < -0.15:  # 15% drawdown
                    return False, "Excessive drawdown detected"
            
            # Check position concentration
            if positions:
                max_position = max(p['value'] for p in positions)
                if max_position / portfolio_value > 0.25:  # 25% in single position
                    return False, "Position too concentrated"
            
            # Check leverage
            total_exposure = sum(p['value'] for p in positions)
            if total_exposure / portfolio_value > self.max_leverage:
                return False, "Excessive leverage"
            
            return True, "Portfolio healthy"
            
        except Exception as e:
            logger.error(f"Error checking portfolio health: {e}")
            return False, f"Error in health check: {str(e)}"
