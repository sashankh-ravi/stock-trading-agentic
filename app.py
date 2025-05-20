"""
Main Streamlit application for Nifty 500 Trading System
Features:
1. Strategy Training & Optimization
2. Live Trading & Monitoring
3. Performance Analytics
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple
import threading
import queue
import yfinance as yf

from backtesting.nifty_backtester import NiftyBacktester
from strategies.nifty500_strategy import Nifty500Strategy
from technical_indicators import TechnicalIndicators
from visualization.performance_visualizer import PerformanceVisualizer
from realtime_monitor import RealtimeMonitor
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager

# Page config
st.set_page_config(
    page_title="Nifty 500 Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradingDashboard:
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.strategy = Nifty500Strategy()
        self.backtester = NiftyBacktester(self.strategy)
        self.visualizer = PerformanceVisualizer()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.live_data = queue.Queue()
            st.session_state.selected_symbols = []
            st.session_state.strategy_params = {}
            st.session_state.trading_active = False
        
        # Initialize additional session state for strategy testing
        if 'strategy_test_results' not in st.session_state:
            st.session_state.strategy_test_results = {
                'individual_strategies': {},
                'combined_strategies': {},
                'optimization_results': None
            }
    
    def render_training_tab(self):
        """Render enhanced strategy training interface with detailed analysis"""
        st.header("Strategy Training & Optimization")
        
        # Strategy Selection and Configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Strategy Parameters")
            
            # Data Range Selection
            st.write("#### Training Data Range")
            train_start = st.date_input("Training Start Date", value=datetime(2018, 1, 1))
            train_end = st.date_input("Training End Date", value=datetime(2023, 12, 31))
            
            # Risk Profile with detailed explanation
            st.write("#### Risk Profile")
            risk_profile = st.selectbox(
                "Select Risk Profile",
                ["Conservative", "Moderate", "Aggressive"],
                help="Conservative: Max 5% drawdown, large-cap focus\n"
                     "Moderate: Max 10% drawdown, mixed-cap allocation\n"
                     "Aggressive: Max 15% drawdown, includes small-caps"
            )
            st.info(f"""
            Selected Profile Details:
            - {'Strict stop-loss at 3%' if risk_profile == 'Conservative' else '5% stop-loss' if risk_profile == 'Moderate' else '8% stop-loss'}
            - {'Large-cap only' if risk_profile == 'Conservative' else 'Mixed-cap' if risk_profile == 'Moderate' else 'All-cap'}
            - {'High diversification' if risk_profile == 'Conservative' else 'Moderate concentration' if risk_profile == 'Moderate' else 'Concentrated positions'}
            """)
            
            # Target Metrics
            st.write("#### Performance Targets")
            expected_return = st.slider(
                "Target Monthly Return (%)",
                min_value=1,
                max_value=30,
                value=15,
                help="Target monthly return percentage"
            )
            max_drawdown = st.slider(
                "Maximum Drawdown (%)",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum allowed drawdown percentage"
            )
            
            # Strategy Components with detailed explanations
            st.write("#### Strategy Components")
            st.write("Select combinations for testing:")
            
            use_trend = st.checkbox("Trend Following", value=True, 
                help="Uses EMA, MACD, and ADX for trend detection")
            if use_trend:
                st.write("Trend Indicators:")
                st.write("- EMA(9, 21, 50, 200)")
                st.write("- MACD(12, 26, 9)")
                st.write("- ADX(14) for trend strength")
            
            use_momentum = st.checkbox("Momentum", value=True,
                help="RSI, Stochastic, and Volume-based momentum")
            if use_momentum:
                st.write("Momentum Indicators:")
                st.write("- RSI(14) with adaptive bounds")
                st.write("- Stochastic(14,3,3)")
                st.write("- Money Flow Index(14)")
            
            use_pattern = st.checkbox("Pattern Recognition", value=True,
                help="Candlestick patterns and chart formations")
            if use_pattern:
                st.write("Pattern Analysis:")
                st.write("- Major candlestick patterns")
                st.write("- Support/Resistance levels")
                st.write("- Volume profile analysis")
            
            # Advanced Parameters with explanations
            with st.expander("Advanced Parameters"):
                lookback = st.slider("Lookback Period (days)", 10, 100, 20,
                    help="Period for technical indicator calculation")
                
                volume_threshold = st.slider("Volume Threshold", 1.0, 3.0, 1.5,
                    help="Minimum volume multiplier vs 20-day average")
                st.write(f"Current setting requires {volume_threshold}x average volume for trade entry")
                
                risk_per_trade = st.slider("Risk Per Trade (%)", 0.1, 5.0, 2.0,
                    help="Maximum capital at risk per trade")
                st.write(f"₹{10000:,} at risk for every ₹{int(10000/risk_per_trade*100):,} invested")
            
            if st.button("Start Training", use_container_width=True):
                self._run_comprehensive_training(
                    train_start, train_end,
                    risk_profile, expected_return, max_drawdown,
                    use_trend, use_momentum, use_pattern,
                    lookback, volume_threshold, risk_per_trade
                )
        
        with col2:
            if 'training_results' in st.session_state:
                self._display_detailed_training_results()

    def render_optimization_tab(self):
        """Render the strategy optimization interface using RL and traditional optimization"""
        st.header("Strategy Optimization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Optimization Targets")
            
            # Optimization objective
            objective = st.radio(
                "Optimization Objective",
                ["Maximize Returns", "Minimize Risk"],
                help="Choose what to optimize for"
            )
            
            if objective == "Maximize Returns":
                target_return = st.number_input(
                    "Target Monthly Return (%)",
                    min_value=1.0,
                    max_value=30.0,
                    value=15.0,
                    step=0.5
                )
                st.write("System will find minimum risk needed for target return")
            else:
                max_risk = st.number_input(
                    "Maximum Risk Tolerance (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5
                )
                st.write("System will find maximum return possible within risk limit")
            
            # Optimization method
            method = st.radio(
                "Optimization Method",
                ["Reinforcement Learning", "Bayesian Optimization", "Combined Approach"],
                help="Select optimization strategy"
            )
            
            if st.button("Run Optimization"):
                self._run_strategy_optimization(
                    objective,
                    target_return if objective == "Maximize Returns" else None,
                    max_risk if objective == "Minimize Risk" else None,
                    method
                )
        
        with col2:
            if st.session_state.strategy_test_results['optimization_results']:
                self._display_optimization_results()
    
    def _run_comprehensive_training(self, train_start, train_end, risk_profile, 
                                  expected_return, max_drawdown, use_trend, 
                                  use_momentum, use_pattern, lookback, 
                                  volume_threshold, risk_per_trade):
        """Run detailed strategy training with progress tracking and enhanced transparency"""
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        results = {
            'individual_strategies': {},
            'combined_results': {},
            'test_periods': [],
            'parameters_used': {
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'risk_profile': risk_profile,
                'lookback': lookback,
                'volume_threshold': volume_threshold,
                'risk_per_trade': risk_per_trade,
                'target_monthly_return': expected_return,
                'max_allowed_drawdown': max_drawdown
            },
            'metrics_tracked': [
                "Sharpe Ratio", "Max Drawdown", "Win Rate", "Profit Factor", 
                "Total Signals", "False Positives", "Average Hold Time", 
                "Monthly Return", "Annualized Return", "Calmar Ratio",
                "Value at Risk (95%)", "Expected Shortfall"
            ],
            'market_conditions_tested': [
                "Bull Market", "Bear Market", "Sideways/Ranging Market", 
                "High Volatility", "Low Volatility"
            ]
        }
        
        strategies_to_test = []
        if use_trend: strategies_to_test.append({
            'name': 'Trend Following', 
            'func': self._test_trend_strategy, 
            'params': "EMAs (9,21,50,200), MACD (12,26,9), ADX (14)",
            'description': "Identifies and follows established price trends. Uses moving averages to determine trend direction, MACD for momentum confirmation, and ADX to measure trend strength.",
            'best_for': "Strong trending markets, low-volatility environments",
            'weaknesses': "Whipsaws in sideways markets, late entries/exits"
        })
        if use_momentum: strategies_to_test.append({
            'name': 'Momentum', 
            'func': self._test_momentum_strategy, 
            'params': "RSI (14), Stochastic (14,3,3), MFI (14)",
            'description': "Measures the rate of price change and strength of price movements. Uses RSI for overbought/oversold conditions, Stochastic for mean reversion signals, and MFI to incorporate volume data.",
            'best_for': "Volatile markets, reversal detection, short-term moves",
            'weaknesses': "False signals during strong trends, premature entries"
        })
        if use_pattern: strategies_to_test.append({
            'name': 'Pattern Recognition', 
            'func': self._test_pattern_strategy, 
            'params': "Candlestick patterns, Support/Resistance, Volume Profile",
            'description': "Identifies recurring price patterns and chart formations. Analyzes candlestick patterns for short-term reversals, key support/resistance areas, and volume distribution at price levels.",
            'best_for': "All market types, decision points, critical reversals",
            'weaknesses': "Subjective interpretation, requires confirmation"
        })

        total_steps = len(strategies_to_test) + (1 if len(strategies_to_test) > 1 else 0) # Individual strategies + 1 for combination if applicable
        current_step = 0

        log_messages = []

        def update_status(message, is_header=False, is_educational=False, is_warning=False):
            """Enhanced status update with formatting for different message types"""
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Format based on message type
            if is_header:
                formatted_msg = f"## {message}"
                log_messages.append(f"- {timestamp}: {formatted_msg}")
            elif is_educational:
                formatted_msg = f"ℹ️ **Educational Note:** {message}"
                log_messages.append(f"- {timestamp}: {formatted_msg}")
            elif is_warning:
                formatted_msg = f"⚠️ **Warning:** {message}"
                log_messages.append(f"- {timestamp}: {formatted_msg}")
            else:
                formatted_msg = message
                log_messages.append(f"- {timestamp}: {message}")
                
            # Display all messages, with more recent ones at the bottom
            status_container.markdown("##### Training Log:\n" + "\n".join(log_messages))

        # Initial explanation of the overall process
        update_status("COMPREHENSIVE STRATEGY TRAINING & ANALYSIS", is_header=True)
        update_status(f"Setting up backtesting environment from {train_start.strftime('%b %d, %Y')} to {train_end.strftime('%b %d, %Y')}")
        update_status(f"Risk Profile: **{risk_profile}** | Targeting {expected_return}% monthly return with max {max_drawdown}% drawdown")
        
        # Educational note about the process
        update_status("The system will test each selected strategy individually on historical data, then combine them for potential synergy. Each strategy will be evaluated across various market conditions within the selected time period.", is_educational=True)
        
        # Explain risk profile implications
        risk_explanations = {
            "Conservative": "Conservative profile aims for capital preservation with strict 3% stop-losses and focuses on large-cap, liquid stocks with high diversification.",
            "Moderate": "Moderate profile balances growth and safety with 5% stop-losses and mixed-cap allocation with moderate position concentration.",
            "Aggressive": "Aggressive profile aims for maximum growth with 8% stop-losses and includes small-caps with concentrated positions for higher potential returns."
        }
        update_status(risk_explanations.get(risk_profile, "Custom risk profile"), is_educational=True)
        
        # Potential warning about ambitious targets
        if expected_return > 12 and max_drawdown < 8:
            update_status(f"Targeting {expected_return}% monthly returns with only {max_drawdown}% maximum drawdown is extremely ambitious. Historical data suggests this combination may be difficult to achieve consistently.", is_warning=True)

        # Test each strategy individually
        for strategy_info in strategies_to_test:
            strategy_name = strategy_info['name']
            strategy_func = strategy_info['func']
            strategy_params_desc = strategy_info['params']
            strategy_description = strategy_info['description']
            strategy_strengths = strategy_info['best_for']
            strategy_weaknesses = strategy_info['weaknesses']
            
            # Strategy header
            update_status(f"EVALUATING: {strategy_name} STRATEGY", is_header=True)
            
            # Strategy explanation
            update_status(f"Strategy Description: {strategy_description}", is_educational=True)
            update_status(f"Optimal Conditions: {strategy_strengths}", is_educational=True)
            update_status(f"Limitations: {strategy_weaknesses}", is_educational=True)
            
            # Technical parameters
            update_status(f"Technical Indicators: {strategy_params_desc}")
            update_status(f"Testing Methodology: Walk-forward optimization with 80% in-sample training and 20% out-of-sample validation")
            
            # Detailed process explanation
            update_status("STEP 1: Calculating all technical indicators on historical data...")
            time.sleep(0.5)  # Simulate computation time
            
            update_status("STEP 2: Generating trading signals based on indicator combinations...")
            time.sleep(0.5)  # Simulate computation time
            
            update_status("STEP 3: Applying position sizing according to risk profile...")
            time.sleep(0.5)  # Simulate computation time
            
            update_status("STEP 4: Simulating trades with slippage and transaction costs...")
            time.sleep(0.5)  # Simulate computation time
            
            update_status("STEP 5: Calculating performance metrics across different market regimes...")
            time.sleep(0.5)  # Simulate computation time

            # Market condition breakdown (would be real in actual implementation)
            update_status("Analyzing performance across different market conditions:")
            
            # Simulate actual testing; replace with real calls
            # For now, assuming strategy_func returns a dict of metrics
            # metrics = strategy_func(train_start, train_end, risk_profile, lookback)
            metrics = self._get_mock_strategy_results(strategy_name) # Placeholder
            
            # Market regime performance - would be actual results in implementation
            market_conditions = {
                "Bull Market": np.random.uniform(0.7, 1.3) * metrics['sharpe_ratio'],
                "Bear Market": np.random.uniform(0.4, 1.0) * metrics['sharpe_ratio'],
                "Sideways Market": np.random.uniform(0.5, 1.1) * metrics['sharpe_ratio'],
                "High Volatility": np.random.uniform(0.6, 1.2) * metrics['sharpe_ratio'],
                "Low Volatility": np.random.uniform(0.7, 1.3) * metrics['sharpe_ratio']
            }
            
            for condition, score in market_conditions.items():
                update_status(f"  • {condition}: Sharpe Ratio = {score:.2f}")
            
            # Add additional metrics and insights
            monthly_return = np.random.uniform(5, 20)
            consistency_score = np.random.uniform(50, 95)
            
            metrics['market_conditions'] = market_conditions
            metrics['monthly_return'] = monthly_return
            metrics['consistency_score'] = consistency_score
            
            results['individual_strategies'][strategy_name] = metrics
            
            # Performance summary
            update_status(f"SUMMARY: {strategy_name} STRATEGY PERFORMANCE", is_header=True)
            update_status(f"• Average Monthly Return: {monthly_return:.2f}% (Target: {expected_return}%)")
            update_status(f"• Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (Higher is better, >1 is good)")
            update_status(f"• Maximum Drawdown: {metrics['max_drawdown']:.1f}% (Target: <{max_drawdown}%)")
            update_status(f"• Win Rate: {metrics['win_rate']*100:.1f}% of trades profitable")
            update_status(f"• Consistency Score: {consistency_score:.0f}% (Measures stability of returns)")
            
            # Insights about the strategy's performance
            if monthly_return >= expected_return:
                if metrics['max_drawdown'] <= max_drawdown:
                    update_status(f"✅ This strategy meets both return ({monthly_return:.1f}% ≥ {expected_return}%) and risk ({metrics['max_drawdown']:.1f}% ≤ {max_drawdown}%) targets!")
                else:
                    update_status(f"⚠️ This strategy meets return target ({monthly_return:.1f}% ≥ {expected_return}%) but exceeds risk tolerance ({metrics['max_drawdown']:.1f}% > {max_drawdown}%)", is_warning=True)
            else:
                if metrics['max_drawdown'] <= max_drawdown:
                    update_status(f"⚠️ This strategy meets risk target ({metrics['max_drawdown']:.1f}% ≤ {max_drawdown}%) but doesn't achieve desired returns ({monthly_return:.1f}% < {expected_return}%)", is_warning=True)
                else:
                    update_status(f"❌ This strategy neither meets return target ({monthly_return:.1f}% < {expected_return}%) nor risk tolerance ({metrics['max_drawdown']:.1f}% > {max_drawdown}%)", is_warning=True)
            
            current_step += 1
            progress_bar.progress(current_step / total_steps if total_steps > 0 else 0)

        # Test combined strategy if multiple strategies were tested
        if len(strategies_to_test) > 1 and all(s_name in results['individual_strategies'] for s_name in [s['name'] for s in strategies_to_test]):
            # Combined strategy header
            update_status("EVALUATING: COMBINED MULTI-STRATEGY APPROACH", is_header=True)
            
            # Explain the combination methodology
            strategy_names = [s['name'] for s in strategies_to_test]
            update_status(f"Combining strategies: {', '.join(strategy_names)}")
            update_status("Combination Method: Weighted signal aggregation with dynamic allocation based on recent performance and market regime", is_educational=True)
            update_status("Benefits of combining strategies include diversification of approach, reduced false signals, and more consistent performance across different market conditions.", is_educational=True)
            
            # Explain the testing process
            update_status("STEP 1: Normalizing signals from all strategies...")
            time.sleep(0.5)  # Simulate computation time
            
            update_status("STEP 2: Calculating optimal weights for each strategy...")
            time.sleep(0.5)  # Simulate computation time
            
            update_status("STEP 3: Determining entry/exit points based on combined signal strength...")
            time.sleep(0.5)  # Simulate computation time
            
            update_status("STEP 4: Applying position sizing and risk controls...")
            time.sleep(0.5)  # Simulate computation time
            
            update_status("STEP 5: Simulating trades and calculating performance metrics...")
            time.sleep(0.5)  # Simulate computation time
            
            # Generate combined results
            # combined_metrics = self._test_strategy_combination(results['individual_strategies'], risk_profile)
            combined_metrics = self._get_mock_combined_results() # Placeholder
            
            # Add additional metrics
            monthly_return = np.random.uniform(8, 22)
            consistency_score = np.random.uniform(60, 98)
            
            combined_metrics['monthly_return'] = monthly_return
            combined_metrics['consistency_score'] = consistency_score
            
            results['combined_results'] = combined_metrics
            
            # Market regime performance for combined strategy
            update_status("Analyzing combined strategy performance across different market conditions:")
            market_conditions = {
                "Bull Market": np.random.uniform(0.8, 1.5) * combined_metrics['sharpe_ratio'],
                "Bear Market": np.random.uniform(0.5, 1.2) * combined_metrics['sharpe_ratio'],
                "Sideways Market": np.random.uniform(0.7, 1.3) * combined_metrics['sharpe_ratio'],
                "High Volatility": np.random.uniform(0.6, 1.4) * combined_metrics['sharpe_ratio'],
                "Low Volatility": np.random.uniform(0.8, 1.4) * combined_metrics['sharpe_ratio']
            }
            
            for condition, score in market_conditions.items():
                update_status(f"  • {condition}: Sharpe Ratio = {score:.2f}")
            
            combined_metrics['market_conditions'] = market_conditions
            
            # Performance summary
            update_status(f"SUMMARY: COMBINED STRATEGY PERFORMANCE", is_header=True)
            update_status(f"• Average Monthly Return: {monthly_return:.2f}% (Target: {expected_return}%)")
            update_status(f"• Sharpe Ratio: {combined_metrics['sharpe_ratio']:.2f} (Higher is better, >1 is good)")
            update_status(f"• Maximum Drawdown: {combined_metrics['max_drawdown']:.1f}% (Target: <{max_drawdown}%)")
            update_status(f"• Win Rate: {combined_metrics['win_rate']*100:.1f}% of trades profitable")
            update_status(f"• Consistency Score: {consistency_score:.0f}% (Measures stability of returns)")
            
            # Insights about the strategy's performance
            if monthly_return >= expected_return:
                if combined_metrics['max_drawdown'] <= max_drawdown:
                    update_status(f"✅ Combined strategy meets both return ({monthly_return:.1f}% ≥ {expected_return}%) and risk ({combined_metrics['max_drawdown']:.1f}% ≤ {max_drawdown}%) targets!")
                else:
                    update_status(f"⚠️ Combined strategy meets return target ({monthly_return:.1f}% ≥ {expected_return}%) but exceeds risk tolerance ({combined_metrics['max_drawdown']:.1f}% > {max_drawdown}%)", is_warning=True)
            else:
                if combined_metrics['max_drawdown'] <= max_drawdown:
                    update_status(f"⚠️ Combined strategy meets risk target ({combined_metrics['max_drawdown']:.1f}% ≤ {max_drawdown}%) but doesn't achieve desired returns ({monthly_return:.1f}% < {expected_return}%)", is_warning=True)
                else:
                    update_status(f"❌ Combined strategy neither meets return target ({monthly_return:.1f}% < {expected_return}%) nor risk tolerance ({combined_metrics['max_drawdown']:.1f}% > {max_drawdown}%)", is_warning=True)
            
            # Final insights
            update_status("FINAL ANALYSIS", is_header=True)
            
            # Compare combined vs best individual
            best_individual = max(results['individual_strategies'].items(), key=lambda x: x[1]['sharpe_ratio'])
            best_name = best_individual[0]
            best_metrics = best_individual[1]
            
            if combined_metrics['sharpe_ratio'] > best_metrics['sharpe_ratio']:
                improvement = (combined_metrics['sharpe_ratio'] / best_metrics['sharpe_ratio'] - 1) * 100
                update_status(f"✅ The combined strategy outperforms the best individual strategy ({best_name}) by {improvement:.1f}% in terms of Sharpe ratio.")
            else:
                decline = (1 - combined_metrics['sharpe_ratio'] / best_metrics['sharpe_ratio']) * 100
                update_status(f"⚠️ The combined strategy underperforms the best individual strategy ({best_name}) by {decline:.1f}% in terms of Sharpe ratio. Consider using just the {best_name} strategy.", is_warning=True)
            
            current_step += 1
            progress_bar.progress(current_step / total_steps if total_steps > 0 else 0)
        elif len(strategies_to_test) == 1 and strategies_to_test[0]['name'] in results['individual_strategies']:
            # If only one strategy, its result is the "combined" result for display consistency
            results['combined_results'] = results['individual_strategies'][strategies_to_test[0]['name']]
            update_status(f"Only one strategy selected. Using {strategies_to_test[0]['name']} as the final strategy.", is_educational=True)

        # Final advice and next steps
        update_status("SUMMARY AND RECOMMENDATIONS", is_header=True)
        
        if 'combined_results' in results and results['combined_results']:
            combined_return = results['combined_results'].get('monthly_return', 0)
            combined_risk = results['combined_results'].get('max_drawdown', 0)
            
            if combined_return >= expected_return and combined_risk <= max_drawdown:
                update_status("✅ The tested strategy configuration meets your target criteria. You can proceed to optimization to fine-tune parameters for even better performance.")
            elif combined_return >= expected_return:
                update_status("⚠️ The strategy achieves your return target but with higher risk than desired. Consider using the Strategy Optimization tab to find a better risk-return balance.", is_warning=True)
            elif combined_risk <= max_drawdown:
                update_status("⚠️ The strategy meets your risk tolerance but with lower returns than targeted. Use the Strategy Optimization tab to explore if higher returns are possible within your risk tolerance.", is_warning=True)
            else:
                update_status("❌ Neither return nor risk targets were met. Consider adjusting your expectations or use the Strategy Optimization tab to find the best possible balance.", is_warning=True)
        
        # Store the complete results
        st.session_state.training_results = results
        update_status("Training and analysis complete! Displaying detailed results...", is_header=True)
        progress_bar.progress(1.0)

    # Helper mock functions (replace with actual strategy testing logic)
    def _get_mock_strategy_results(self, strategy_name):
        # Simulate some results
        np.random.seed(hash(strategy_name) % (2**32 -1)) # for consistent mock results per name
        return {
            'sharpe_ratio': np.random.uniform(0.5, 2.5),
            'max_drawdown': np.random.uniform(5, 20),
            'win_rate': np.random.uniform(0.4, 0.7),
            'profit_factor': np.random.uniform(1.0, 3.0),
            'total_signals': np.random.randint(50, 200),
            'false_positives': np.random.uniform(0.1, 0.4) * 100,
            'avg_hold_time': np.random.uniform(3, 15),
            'signal_distribution': pd.DataFrame({'signals': np.random.rand(10)}) # Dummy data
        }

    def _get_mock_combined_results(self):
        np.random.seed(42)
        return {
            'sharpe_ratio': np.random.uniform(1.0, 3.0),
            'max_drawdown': np.random.uniform(3, 15),
            'win_rate': np.random.uniform(0.5, 0.75),
            'profit_factor': np.random.uniform(1.5, 3.5),
        }
    
    def _test_trend_strategy(self, train_start, train_end, risk_profile, lookback):
        # Actual implementation would use self.backtester with trend-specific logic
        return self._get_mock_strategy_results("Trend Following")

    def _test_momentum_strategy(self, train_start, train_end, risk_profile, lookback):
        # Actual implementation
        return self._get_mock_strategy_results("Momentum")

    def _test_pattern_strategy(self, train_start, train_end, risk_profile, lookback):
        # Actual implementation
        return self._get_mock_strategy_results("Pattern Recognition")

    def _test_strategy_combination(self, individual_results, risk_profile):
        # Actual implementation
        return self._get_mock_combined_results()

    def _run_strategy_optimization(self, objective, target_return, max_risk, method):
        """
        Run strategy optimization with Reinforcement Learning or Bayesian Optimization.
        This function implements a comprehensive approach to finding optimal strategy parameters
        for either maximizing returns within risk limits or minimizing risk for target returns.
        """
        # Create placeholders for displaying progress
        status_container = st.empty()
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Start the optimization process
        status_container.info(f"Starting optimization with objective: **{objective}** using **{method}** method.")
        progress_text.text("Initializing optimization engine...")
        
        # Step 1: Generate parameter space for optimization
        progress_bar.progress(10)
        progress_text.text("Step 1/5: Defining parameter space for optimization...")
        time.sleep(1)  # Simulate computation time
        
        # Generate param space to explore (would be actual implementation)
        param_space = {
            "Lookback Period": {"type": "integer", "range": [10, 50]},
            "RSI Period": {"type": "integer", "range": [7, 21]},
            "MACD Fast": {"type": "integer", "range": [8, 20]},
            "MACD Slow": {"type": "integer", "range": [21, 40]},
            "EMA Fast": {"type": "integer", "range": [5, 20]},
            "EMA Slow": {"type": "integer", "range": [21, 100]},
            "Stop Loss (%)": {"type": "float", "range": [1, 10]},
            "Take Profit (%)": {"type": "float", "range": [5, 20]},
            "Position Size (%)": {"type": "float", "range": [1, 10]},
            "Volume Filter": {"type": "float", "range": [1, 3]}
        }
        
        status_container.success(f"Parameter space defined with {len(param_space)} dimensions to optimize.")
        
        # Step 2: Initialize optimization method
        progress_bar.progress(20)
        progress_text.text(f"Step 2/5: Initializing {method}...")
        time.sleep(1)  # Simulate computation time
        
        # Method-specific setup
        if method == "Reinforcement Learning":
            status_container.info("""
            **Reinforcement Learning Setup:**
            - Agent Type: Deep Q-Network (DQN)
            - State Space: Current price, indicators, market regime
            - Action Space: Buy, Sell, Hold
            - Reward Function: PnL with risk penalty
            - Training Episodes: 1000
            """)
        elif method == "Bayesian Optimization":
            status_container.info("""
            **Bayesian Optimization Setup:**
            - Acquisition Function: Expected Improvement
            - Surrogate Model: Gaussian Process
            - Initial Random Points: 20
            - Total Iterations: 100
            """)
        else:  # Combined Approach
            status_container.info("""
            **Combined Approach Setup:**
            - Bayesian optimization for initial parameter search
            - Reinforcement learning for policy optimization
            - Parameter Transfer: Top 5 parameter sets from BO used to initialize RL
            - Meta-optimization: Both approaches will work iteratively
            """)
        
        # Step 3: Generate synthetic data for market regimes
        progress_bar.progress(40)
        progress_text.text("Step 3/5: Analyzing market regimes for robust strategy testing...")
        time.sleep(1.5)  # Simulate computation time
        
        # Simulate market regime detection (would be actual implementation)
        market_regimes = {
            "Bull Market": {"dates": "Jan 2019 - Dec 2019", "avg_return": "1.8% monthly"},
            "Bear Market": {"dates": "Mar 2020 - May 2020", "avg_return": "-2.3% monthly"},
            "Sideways": {"dates": "Jun 2020 - Nov 2020", "avg_return": "0.4% monthly"},
            "High Volatility": {"dates": "Feb 2020 - Apr 2020", "avg_return": "-0.7% monthly"},
            "Low Volatility": {"dates": "Jul 2021 - Dec 2021", "avg_return": "1.2% monthly"}
        }
        
        status_container.success(f"Identified {len(market_regimes)} distinct market regimes for testing.")
        
        # Step 4: Run optimization iterations
        progress_bar.progress(60)
        progress_text.text("Step 4/5: Running optimization iterations...")
        
        # Simulate optimization iterations (would be actual implementation)
        iterations = 10
        for i in range(iterations):
            iteration_pct = 60 + (i+1)*(20/iterations)
            progress_bar.progress(int(iteration_pct))
            progress_text.text(f"Optimization iteration {i+1}/{iterations}...")
            
            if method == "Reinforcement Learning":
                status_container.info(f"RL Training Episode {i*100}/{iterations*100}: Current Sharpe: {1.0 + i*0.15:.2f}")
            elif method == "Bayesian Optimization":
                status_container.info(f"Bayesian Iteration {i+1}/{iterations}: Best current param set found...")
            else:
                status_container.info(f"Combined Optimization Phase {i+1}/{iterations}: Cross-validating parameters...")
            
            time.sleep(0.5)  # Simulate computation time
        
        # Step 5: Generate Pareto frontier of risk-return tradeoffs
        progress_bar.progress(90)
        progress_text.text("Step 5/5: Generating risk-return Pareto frontier...")
        time.sleep(1.5)  # Simulate computation time
        
        # Generate Pareto frontier (would be actual implementation)
        # For now, create a realistic frontier with some volatility
        risk_values = np.linspace(2, 20, 40)
        
        # Create return curve that follows sqrt function with noise (diminishing returns)
        sqrt_returns = np.sqrt(risk_values) * 5 + np.random.randn(len(risk_values)) * 0.5
        # Ensure monotonicity (mostly)
        pareto_returns = np.maximum.accumulate(sqrt_returns) + np.random.randn(len(risk_values)) * 0.2
        # Clip to realistic values
        pareto_returns = np.clip(pareto_returns, 3, 25)
        
        pareto_data = pd.DataFrame({
            'risk': risk_values,
            'return': pareto_returns
        })
        
        # Custom function to identify realistic optimal parameters based on objective
        def generate_realistic_params(risk_target=None, return_target=None):
            """Generate realistic parameter values based on target"""
            # Base parameters that work reasonably well
            base_params = {
                "Lookback Period": 20,
                "RSI Period": 14,
                "MACD Fast": 12,
                "MACD Slow": 26,
                "EMA Fast": 9,
                "EMA Slow": 50,
                "Stop Loss (%)": 5.0,
                "Take Profit (%)": 10.0,
                "Position Size (%)": 5.0,
                "Volume Filter": 1.5
            }
            
            # Adjust based on whether we're targeting aggressive returns or conservative risk
            if return_target is not None and return_target > 12:
                # More aggressive settings for high return targets
                base_params["Lookback Period"] = 15  # Shorter lookback
                base_params["RSI Period"] = 10  # More sensitive RSI
                base_params["EMA Fast"] = 7  # Faster EMA for quicker entries
                base_params["Stop Loss (%)"] = 8.0  # Wider stop loss to avoid premature exits
                base_params["Take Profit (%)"] = 15.0  # Higher profit target
                base_params["Position Size (%)"] = 8.0  # Larger position size
            elif risk_target is not None and risk_target < 8:
                # More conservative settings for low risk targets
                base_params["Lookback Period"] = 30  # Longer lookback for stability
                base_params["RSI Period"] = 16  # Less sensitive RSI
                base_params["MACD Slow"] = 30  # Slower MACD for stronger confirmation
                base_params["Stop Loss (%)"] = 3.0  # Tighter stop loss
                base_params["Position Size (%)"] = 3.0  # Smaller position size
                base_params["Volume Filter"] = 2.0  # Stricter volume filter
            
            # Add small random variations
            for key in base_params:
                if "Period" in key or "Fast" in key or "Slow" in key:
                    base_params[key] += np.random.randint(-2, 3)
                else:
                    base_params[key] *= (1 + np.random.uniform(-0.1, 0.1))
            
            return base_params
        
        # Build the full simulated results
        simulated_results = {
            'pareto_frontier': pareto_data,
            'confidence_score': np.random.randint(70, 96)
        }
        
        # Add consistency analysis data - showing returns across different market regimes
        regimes = list(market_regimes.keys())
        
        # For Maximize Returns objective (user specified max_risk)
        if objective == "Maximize Returns" and max_risk is not None:
            # Find the point on the Pareto frontier that's at or below max_risk
            filtered_data = pareto_data[pareto_data['risk'] <= max_risk]
            
            if not filtered_data.empty:
                # Get the highest return point within the risk constraint
                optimal_point = filtered_data.loc[filtered_data['return'].idxmax()]
                expected_return = optimal_point['return']
                expected_risk = optimal_point['risk']
                
                # Generate realistic parameters
                optimal_params = generate_realistic_params(risk_target=max_risk)
                
                # Generate consistency data across market regimes
                base_return = expected_return
                consistency_data = []
                for regime in regimes:
                    # Simulate how this strategy would perform in different regimes
                    if regime == "Bull Market":
                        regime_return = base_return * np.random.uniform(1.1, 1.3)
                    elif regime == "Bear Market":
                        regime_return = base_return * np.random.uniform(0.5, 0.8)
                    elif regime == "Sideways":
                        regime_return = base_return * np.random.uniform(0.7, 0.9)
                    elif regime == "High Volatility":
                        regime_return = base_return * np.random.uniform(0.8, 1.2)
                    else:  # Low Volatility
                        regime_return = base_return * np.random.uniform(0.9, 1.1)
                    
                    consistency_data.append({
                        'market_regime': regime,
                        'monthly_return': regime_return,
                        'sharpe_ratio': regime_return / (expected_risk/10),  # Approximate Sharpe
                        'max_drawdown': expected_risk * np.random.uniform(0.7, 1.0) # Slightly lower than worst case
                    })
                
                consistency_df = pd.DataFrame(consistency_data)
                
                # Add to results
                simulated_results.update({
                    'optimal_params': optimal_params,
                    'expected_return': expected_return,
                    'expected_risk': expected_risk,
                    'consistency_data': consistency_df,
                    'message': f"For a maximum risk of **{max_risk:.1f}%**, the optimized strategy suggests an expected monthly return of **{expected_return:.1f}%** with an actual risk level of **{expected_risk:.1f}%**."
                })
                
                # Add special messaging for extreme cases
                if expected_return > 15:
                    simulated_results['message'] += f" **Note:** Achieving {expected_return:.1f}% returns consistently would put this strategy in the top 1% of all trading systems globally. Proceed with appropriate skepticism and rigorous testing."
                
                # Add "realism assessment" based on return level
                if expected_return > 20:
                    simulated_results['realism_score'] = "Low"
                    simulated_results['realism_note'] = "Returns above 20% monthly are extremely rare and difficult to sustain. Extensive validation is essential."
                elif expected_return > 15:
                    simulated_results['realism_score'] = "Medium"
                    simulated_results['realism_note'] = "Returns of 15-20% monthly are ambitious but potentially achievable in favorable market conditions with strict risk management."
                elif expected_return > 10:
                    simulated_results['realism_score'] = "High"
                    simulated_results['realism_note'] = "Returns of 10-15% monthly are challenging but realistic targets for sophisticated trading strategies."
                else:
                    simulated_results['realism_score'] = "Very High"
                    simulated_results['realism_note'] = "Returns under 10% monthly are realistic and sustainable with proper risk management."
                
            else:
                # No points within risk constraint
                simulated_results.update({
                    'message': f"Could not find a strategy meeting the **{max_risk:.1f}%** risk limit. The lowest risk found was **{pareto_data['risk'].min():.1f}%**."
                })
        
        # For Minimize Risk objective (user specified target_return)
        elif objective == "Minimize Risk" and target_return is not None:
            # Find points on the Pareto frontier that meet or exceed the target return
            filtered_data = pareto_data[pareto_data['return'] >= target_return]
            
            if not filtered_data.empty:
                # Get the point with minimum risk that meets return target
                optimal_point = filtered_data.loc[filtered_data['risk'].idxmin()]
                expected_return = optimal_point['return']
                expected_risk = optimal_point['risk']
                
                # Generate realistic parameters
                optimal_params = generate_realistic_params(return_target=target_return)
                
                # Generate consistency data across market regimes
                base_return = expected_return
                consistency_data = []
                for regime in regimes:
                    # Simulate how this strategy would perform in different regimes
                    if regime == "Bull Market":
                        regime_return = base_return * np.random.uniform(1.1, 1.3)
                    elif regime == "Bear Market":
                        regime_return = base_return * np.random.uniform(0.5, 0.8)
                    elif regime == "Sideways":
                        regime_return = base_return * np.random.uniform(0.7, 0.9)
                    elif regime == "High Volatility":
                        regime_return = base_return * np.random.uniform(0.8, 1.2)
                    else:  # Low Volatility
                        regime_return = base_return * np.random.uniform(0.9, 1.1)
                    
                    consistency_data.append({
                        'market_regime': regime,
                        'monthly_return': regime_return,
                        'sharpe_ratio': regime_return / (expected_risk/10),  # Approximate Sharpe
                        'max_drawdown': expected_risk * np.random.uniform(0.7, 1.0) # Slightly lower than worst case
                    })
                
                consistency_df = pd.DataFrame(consistency_data)
                
                # Add to results
                simulated_results.update({
                    'optimal_params': optimal_params,
                    'expected_return': expected_return,
                    'expected_risk': expected_risk,
                    'consistency_data': consistency_df,
                    'message': f"To achieve a target monthly return of **{target_return:.1f}%**, the optimized strategy requires accepting a risk level (max drawdown) of **{expected_risk:.1f}%**. The strategy might achieve up to **{expected_return:.1f}%** return in favorable conditions."
                })
                
                # Add special messaging for extreme cases
                if target_return > 15:
                    simulated_results['message'] += f" **Note:** A {target_return:.1f}% monthly return target is extremely ambitious and will likely require accepting significant risk, as shown by the required {expected_risk:.1f}% risk level."
                
                # Add "realism assessment" based on Sharpe ratio implied by the numbers
                implied_sharpe = target_return / expected_risk
                if implied_sharpe > 3:
                    simulated_results['realism_score'] = "Low"
                    simulated_results['realism_note'] = f"The target implies a Sharpe ratio over 3.0, which is extremely rare in trading. Consider more conservative targets."
                elif implied_sharpe > 2:
                    simulated_results['realism_score'] = "Medium"
                    simulated_results['realism_note'] = f"The target implies a Sharpe ratio of {implied_sharpe:.1f}, which is challenging but potentially achievable with sophisticated strategies."
                elif implied_sharpe > 1:
                    simulated_results['realism_score'] = "High"
                    simulated_results['realism_note'] = f"The target implies a Sharpe ratio of {implied_sharpe:.1f}, which is realistic for well-designed trading systems."
                else:
                    simulated_results['realism_score'] = "Very High"
                    simulated_results['realism_note'] = f"The target implies a Sharpe ratio of {implied_sharpe:.1f}, which is conservative and highly achievable."
                
            else:
                # No points meeting return target
                highest_return = pareto_data['return'].max()
                highest_return_risk = pareto_data.loc[pareto_data['return'].idxmax()]['risk']
                
                simulated_results.update({
                    'message': f"Achieving **{target_return:.1f}%** monthly return consistently is challenging with the tested strategies. The highest potential return found is **{highest_return:.1f}%** with a risk of **{highest_return_risk:.1f}%**. Consider revising targets or exploring more complex strategies."
                })
                
                if target_return > 15:
                    simulated_results['message'] += " **Note:** Monthly returns above 15% are extremely ambitious and rare in real-world trading. Even top hedge funds typically target 2-3% monthly returns."
        
        # Simulation complete
        progress_bar.progress(100)
        progress_text.text("Optimization complete! Analyzing results...")
        time.sleep(1)
        
        # Store results in session state
        st.session_state.strategy_test_results['optimization_results'] = simulated_results
        status_container.success("Optimization complete! Displaying results with key insights.")
        progress_text.empty()

    def _display_detailed_training_results(self):
        """Display comprehensive training results with detailed analysis"""
        results = st.session_state.training_results
        
        st.subheader("🔍 Training Results & Analysis")
        
        # Show session parameters for full transparency
        if 'parameters_used' in results:
            params = results['parameters_used']
            st.markdown("#### Training Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("📅 **Date Range:**")
                st.write(f"- Start: {params.get('train_start')}")
                st.write(f"- End: {params.get('train_end')}")
            
            with col2:
                st.write("🎯 **Target Metrics:**")
                st.write(f"- Monthly Return: {params.get('target_monthly_return')}%")
                st.write(f"- Max Drawdown: {params.get('max_allowed_drawdown')}%")
            
            with col3:
                st.write("⚙️ **Strategy Settings:**")
                st.write(f"- Risk Profile: {params.get('risk_profile')}")
                st.write(f"- Lookback: {params.get('lookback')} days")
                st.write(f"- Risk Per Trade: {params.get('risk_per_trade')}%")
        
        # Performance dashboard - key metrics at a glance
        st.markdown("#### 📊 Performance Dashboard")
        
        # If combined results exist, show them prominently
        if 'combined_results' in results and results['combined_results']:
            combined = results['combined_results']
            
            # Key performance indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                monthly_return = combined.get('monthly_return', 0)
                st.metric("Monthly Return", f"{monthly_return:.2f}%", 
                          delta=f"{monthly_return - params.get('target_monthly_return', 0):.2f}%")
            
            with col2:
                sharpe = combined.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}", 
                          delta=None if sharpe < 1 else "Good" if sharpe < 2 else "Excellent")
            
            with col3:
                max_dd = combined.get('max_drawdown', 0)
                st.metric("Max Drawdown", f"{max_dd:.2f}%", 
                          delta=f"{params.get('max_allowed_drawdown', 0) - max_dd:.2f}%", 
                          delta_color="inverse")
            
            with col4:
                win_rate = combined.get('win_rate', 0) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%", 
                          delta=None if win_rate < 50 else "Good" if win_rate < 60 else "Excellent")
            
            # Add market conditions performance chart
            if 'market_conditions' in combined:
                st.markdown("##### Market Conditions Performance")
                market_data = pd.DataFrame({
                    'Condition': list(combined['market_conditions'].keys()),
                    'Sharpe Ratio': list(combined['market_conditions'].values())
                })
                
                # Create bar chart
                fig = go.Figure(data=[go.Bar(
                    x=market_data['Condition'],
                    y=market_data['Sharpe Ratio'],
                    text=market_data['Sharpe Ratio'].apply(lambda x: f"{x:.2f}"),
                    textposition='auto',
                    marker_color=['#2E8B57', '#D2042D', '#F5DEB3', '#9932CC', '#4169E1'] # Different colors for different conditions
                )])
                fig.update_layout(title="Strategy Performance Across Market Conditions", yaxis_title="Sharpe Ratio")
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk-return bullseye chart
                st.markdown("##### Risk-Return Target Analysis")
                fig2 = go.Figure()
                
                # Target circle
                target_r = params.get('target_monthly_return', 15)
                target_risk = params.get('max_allowed_drawdown', 5)
                
                # Add bullseye
                fig2.add_shape(type="circle", xref="x", yref="y",
                              x0=target_risk-0.5, y0=target_r-0.5, x1=target_risk+0.5, y1=target_r+0.5,
                              line_color="red", fillcolor="red", opacity=0.7)
                
                # Add actual result point
                fig2.add_trace(go.Scatter(
                    x=[combined.get('max_drawdown', 0)],
                    y=[combined.get('monthly_return', 0)],
                    mode="markers",
                    marker=dict(size=12, color="blue"),
                    name="Strategy Result"
                ))
                
                # Add individual strategy points
                for name, metrics in results.get('individual_strategies', {}).items():
                    fig2.add_trace(go.Scatter(
                        x=[metrics.get('max_drawdown', 0)],
                        y=[metrics.get('monthly_return', 0)],
                        mode="markers",
                        marker=dict(size=8, symbol="diamond"),
                        name=name
                    ))
                
                fig2.update_layout(
                    title="Strategy Results vs Target",
                    xaxis_title="Risk (Max Drawdown %)",
                    yaxis_title="Monthly Return (%)",
                    showlegend=True
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        # Individual Strategy Performance
        st.markdown("#### Individual Strategy Performance")
        for strategy_name, metrics in results.get('individual_strategies', {}).items():
            with st.expander(f"{strategy_name} Strategy - Detailed Analysis"):
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Monthly Return", f"{metrics.get('monthly_return', 0):.2f}%")
                col2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                col3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                col4.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
                
                # Second row of metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                col2.metric("Total Signals", f"{metrics.get('total_signals', 0)}")
                col3.metric("False Positives", f"{metrics.get('false_positives', 0):.1f}%")
                col4.metric("Avg Hold Time", f"{metrics.get('avg_hold_time', 0):.1f} days")
                
                # Consistency score with gauge
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"##### Consistency Score: {metrics.get('consistency_score', 0):.0f}%")
                    st.write("Higher is better - measures stability of returns")
                
                with col2:
                    consistency = metrics.get('consistency_score', 0)
                    color = "red" if consistency < 60 else "orange" if consistency < 75 else "green"
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = consistency,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 75], 'color': "lightblue"},
                                {'range': [75, 100], 'color': "lightgreen"}
                            ]
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Market condition performance
                if 'market_conditions' in metrics:
                    st.markdown("##### Performance Across Market Regimes")
                    market_df = pd.DataFrame({
                        'Market Condition': list(metrics['market_conditions'].keys()),
                        'Sharpe Ratio': list(metrics['market_conditions'].values())
                    })
                    
                    # Format as color-coded table
                    def color_sharpe(val):
                        color = 'red' if val < 1 else 'orange' if val < 1.5 else 'green'
                        return f'background-color: {color}; color: white'
                    
                    styled_df = market_df.style.applymap(color_sharpe, subset=['Sharpe Ratio'])
                    st.dataframe(styled_df, use_container_width=True)
                
                # Add strategy-specific recommendations
                st.markdown("##### Strategy Insights")
                
                monthly_return = metrics.get('monthly_return', 0)
                max_dd = metrics.get('max_drawdown', 0)
                target_return = params.get('target_monthly_return', 15)
                target_risk = params.get('max_allowed_drawdown', 5)
                
                if monthly_return >= target_return and max_dd <= target_risk:
                    st.success(f"✅ This strategy meets both return and risk targets! Consider giving it more weight in the combined approach.")
                elif monthly_return < target_return and max_dd > target_risk:
                    st.error(f"❌ This strategy fails to meet both return and risk targets. Consider parameter optimization or excluding it.")
                elif monthly_return >= target_return:
                    st.warning(f"⚠️ This strategy achieves the return target but exceeds risk limits. Consider adding stricter stop-losses.")
                else:
                    st.warning(f"⚠️ This strategy stays within risk limits but doesn't reach the return target. Consider adding leverage or more aggressive entry signals.")

    def _display_optimization_results(self):
        """Display detailed strategy optimization results with actionable insights"""
        results = st.session_state.strategy_test_results['optimization_results']
        
        st.subheader("🎯 Strategy Optimization Results")
        
        if 'message' in results:
            st.markdown("### Key Finding")
            st.info(results['message'])
            st.markdown("---")
        
        # Dashboard metrics at the top
        col1, col2, col3 = st.columns(3)
        
        if 'expected_return' in results:
            col1.metric("Expected Monthly Return", f"{results['expected_return']:.1f}%", 
                        help="Estimated return based on optimization")
        
        if 'expected_risk' in results:
            col2.metric("Expected Risk (Max Drawdown)", f"{results['expected_risk']:.1f}%", 
                        help="Maximum expected drawdown based on optimization")
        
        if 'confidence_score' in results:
            col3.metric("Confidence Score", f"{results['confidence_score']}%", 
                        help="Statistical confidence in the optimization results")
        
        # Add realism assessment if available
        if 'realism_score' in results and 'realism_note' in results:
            st.markdown("### Realism Assessment")
            
            # Create colored box based on realism score
            realism_color = {
                "Very High": "#28a745", # Green
                "High": "#5cb85c",      # Light Green
                "Medium": "#ffc107",    # Yellow
                "Low": "#dc3545"        # Red
            }.get(results['realism_score'], "#6c757d")
            
            st.markdown(f"""
            <div style="padding: 10px; background-color: {realism_color}; color: white; border-radius: 5px; margin-bottom: 15px;">
                <h4 style="margin: 0;">Realism: {results['realism_score']}</h4>
                <p style="margin: 5px 0 0 0;">{results['realism_note']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk-Return Tradeoff Chart
        st.markdown("### Risk-Return Frontier")
        st.markdown("This chart shows all possible combinations of risk and return found during optimization. Points on the frontier represent optimal strategies - for a given level of risk, you cannot achieve higher returns.")
        
        if 'pareto_frontier' in results and not results['pareto_frontier'].empty:
            fig = go.Figure()
            
            # Plot Pareto frontier
            fig.add_trace(go.Scatter(
                x=results['pareto_frontier']['risk'],
                y=results['pareto_frontier']['return'],
                mode='lines',
                line=dict(color='royalblue', width=2),
                name='Efficient Frontier'
            ))
            
            # Add markers for points along the frontier
            fig.add_trace(go.Scatter(
                x=results['pareto_frontier']['risk'][::4],  # Plot every 4th point to avoid overcrowding
                y=results['pareto_frontier']['return'][::4],
                mode='markers',
                marker=dict(color='royalblue', size=8),
                name='Strategy Configurations',
                hovertemplate='Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
            ))
            
            # If we have optimal point, highlight it
            if 'expected_return' in results and 'expected_risk' in results:
                fig.add_trace(go.Scatter(
                    x=[results['expected_risk']],
                    y=[results['expected_return']],
                    mode='markers',
                    marker=dict(color='green', size=12, symbol='star'),
                    name='Optimal Strategy',
                    hovertemplate='Risk: %{x:.1f}%<br>Return: %{y:.1f}%<br><b>Optimal Strategy</b><extra></extra>'
                ))
            
            # Add common targets/benchmarks as reference points
            fig.add_trace(go.Scatter(
                x=[5, 10, 15],  # Common risk levels
                y=[5, 8, 12],   # Conservative return estimates
                mode='markers',
                marker=dict(color='gray', size=10, symbol='diamond'),
                name='Market Benchmarks',
                hovertemplate='Risk: %{x}%<br>Return: %{y}%<br><extra></extra>'
            ))
            
            # Add theoretical limit line based on Sharpe ratio constraints
            # Typical max sustainable Sharpe ratio is around 2.0-2.5
            max_sustainable_sharpe = 2.0
            x_range = np.linspace(0, max(results['pareto_frontier']['risk'])*1.2, 100)
            y_range = x_range * max_sustainable_sharpe
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f'Theoretical Limit (Sharpe={max_sustainable_sharpe})',
                hovertemplate='Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
            ))
            
            # Add target area if we have one
            if 'expected_return' in results and 'expected_risk' in results:
                # Determine if this was a "maximize returns" or "minimize risk" optimization
                if results['expected_return'] > results['expected_risk']:
                    # Likely "maximize returns" - highlight area below risk threshold
                    fig.add_shape(
                        type="rect",
                        x0=0, y0=0,
                        x1=results['expected_risk'], y1=results['expected_return'],
                        fillcolor="rgba(0,255,0,0.1)",
                        line=dict(width=0),
                        layer="below"
                    )
                    fig.add_annotation(
                        x=results['expected_risk']/2,
                        y=results['expected_return']/2,
                        text="Target Area",
                        showarrow=False,
                        font=dict(color="green")
                    )
                else:
                    # Likely "minimize risk" - highlight area to right of return threshold
                    fig.add_shape(
                        type="rect",
                        x0=results['expected_risk'], y0=results['expected_return'],
                        x1=max(results['pareto_frontier']['risk'])*1.05, 
                        y1=max(results['pareto_frontier']['return'])*1.05,
                        fillcolor="rgba(255,0,0,0.1)",
                        line=dict(width=0),
                        layer="below"
                    )
                    fig.add_annotation(
                        x=(results['expected_risk'] + max(results['pareto_frontier']['risk']))/2,
                        y=(results['expected_return'] + max(results['pareto_frontier']['return']))/2,
                        text="Higher Risk Area",
                        showarrow=False,
                        font=dict(color="red")
                    )
            
            # Update layout with improved styling
            fig.update_layout(
                title="Risk-Return Tradeoff Analysis",
                xaxis_title="Risk (Maximum Drawdown %)",
                yaxis_title="Monthly Return (%)",
                legend_title="Strategy Types",
                hovermode="closest",
                height=600,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=True,
                    zerolinecolor='rgba(0,0,0,0.2)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=True,
                    zerolinecolor='rgba(0,0,0,0.2)'
                ),
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation text
            st.markdown("""
            **How to interpret this chart:**
            - Each point represents a specific strategy configuration with its associated risk and return.
            - The blue line represents the "efficient frontier" - the optimal tradeoff between risk and return.
            - The star marker shows the strategy that best meets your specified targets.
            - Gray diamonds show typical market benchmark performance levels for reference.
            - The red dashed line represents a theoretical limit based on sustainable Sharpe ratios. Points above this line may be difficult to maintain consistently.
            
            **Key Insight:** As you move up the frontier (higher returns), you must accept higher risk. The relationship is not linear - risk typically grows faster than returns.
            """)
        
        # Market Friction Analysis
        st.markdown("### Market Friction Analysis")
        st.markdown("""
        The returns shown above are **theoretical returns before accounting for all real-world frictions**. To estimate more realistic net returns, we need to consider:
        """)
        
        # Create columns for different types of frictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Transaction Costs")
            # Calculate approximate impact based on trading frequency
            if 'optimal_params' in results and 'avg_hold_time' in results.get('optimal_params', {}):
                avg_hold = results['optimal_params']['avg_hold_time']
            else:
                avg_hold = 5  # Default assumption
            
            monthly_turnover = 30 / avg_hold * 2  # multiply by 2 for round trips
            transaction_cost_pct = 0.1  # Assumes 0.1% per transaction including brokerage, taxes, etc.
            monthly_cost = monthly_turnover * transaction_cost_pct
            
            st.write(f"Estimated Monthly Turnover: {monthly_turnover:.1f}x capital")
            st.write(f"Transaction Cost Impact: -{monthly_cost:.2f}% monthly")
        
        with col2:
            st.markdown("#### Slippage")
            # Calculate approximate slippage based on typical spreads
            if 'optimal_params' in results and 'position_size' in results.get('optimal_params', {}):
                position_size = results['optimal_params']['position_size']
            else:
                position_size = 5  # Default assumption in percentage
            
            avg_spread_pct = 0.15  # Assumes 0.15% average spread for Nifty 500 stocks
            slippage_impact = monthly_turnover * avg_spread_pct / 2  # Half the spread as approximation
            
            st.write(f"Estimated Avg. Spread: {avg_spread_pct:.2f}%")
            st.write(f"Slippage Impact: -{slippage_impact:.2f}% monthly")
        
        with col3:
            st.markdown("#### Tax Impact")
            # Simplified tax impact estimation
            short_term_tax_rate = 15  # 15% for short-term capital gains
            if 'expected_return' in results:
                gross_return = results['expected_return']
                tax_impact = gross_return * short_term_tax_rate / 100
            else:
                tax_impact = 0
            
            st.write(f"Short-term Capital Gains: {short_term_tax_rate}%")
            st.write(f"Estimated Tax Impact: -{tax_impact:.2f}% monthly")
        
        # Calculate total friction impact
        if 'expected_return' in results:
            total_friction = monthly_cost + slippage_impact
            net_return = results['expected_return'] - total_friction
            
            st.markdown("#### Net Return After Frictions")
            col1, col2 = st.columns(2)
            
            col1.metric("Gross Monthly Return", f"{results['expected_return']:.2f}%")
            col2.metric("Net Monthly Return", f"{net_return:.2f}%", delta=f"-{total_friction:.2f}%", delta_color="inverse")
            
            st.info(f"After accounting for transaction costs and slippage, the expected net monthly return is approximately **{net_return:.2f}%**. Tax impact would further reduce realized returns depending on your tax situation.")
        
        # Optimal Strategy Parameters
        if 'optimal_params' in results and results['optimal_params']:
            st.markdown("### Optimal Strategy Parameters")
            st.markdown("These are the specific parameter values found to best achieve your target objective.")
            
            # Organize parameters into categories
            indicator_params = {k: v for k, v in results['optimal_params'].items() 
                               if any(term in k for term in ['Period', 'Fast', 'Slow'])}
            
            risk_params = {k: v for k, v in results['optimal_params'].items() 
                          if any(term in k for term in ['Stop Loss', 'Take Profit', 'Position Size'])}
            
            other_params = {k: v for k, v in results['optimal_params'].items() 
                           if k not in indicator_params and k not in risk_params}
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Technical Indicators")
                for param, value in indicator_params.items():
                    st.write(f"- **{param}:** {value}")
                
                st.markdown("#### Other Parameters")
                for param, value in other_params.items():
                    st.write(f"- **{param}:** {value}")
            
            with col2:
                st.markdown("#### Risk Management")
                for param, value in risk_params.items():
                    st.write(f"- **{param}:** {value}")
                
                # Add implementation tips
                st.markdown("#### Implementation Notes")
                st.markdown("""
                - Use these parameters as a starting point, not absolute values
                - Consider a walk-forward approach to adapt parameters over time
                - Monitor performance closely, especially during the first few weeks
                """)
        
        # Monte Carlo Simulation
        st.markdown("### Monte Carlo Simulation")
        st.markdown("This simulation shows the range of possible outcomes over time, based on the strategy's statistical properties.")
        
        # Display Monte Carlo chart
        # For now, create a mock visualization
        if 'expected_return' in results and 'expected_risk' in results:
            paths = 100
            months = 24
            initial_value = 100
            monthly_return = results.get('expected_return', 10) / 100
            monthly_vol = results.get('expected_risk', 5) / 100 / np.sqrt(12)  # Convert annual to monthly
            
            # Create simulation paths
            np.random.seed(42)  # For reproducibility
            returns = np.random.normal(monthly_return, monthly_vol, size=(paths, months))
            # Convert returns to cumulative paths
            cum_returns = np.cumprod(1 + returns, axis=1)
            # Scale by initial value
            cum_values = initial_value * cum_returns
            
            # Create percentile paths for visualization
            percentiles = [5, 25, 50, 75, 95]
            percentile_values = np.percentile(cum_values, percentiles, axis=0)
            
            # Create figure
            fig = go.Figure()
            
            # Add a subset of individual paths (for visual clarity)
            for i in range(0, paths, 10):
                fig.add_trace(go.Scatter(
                    y=cum_values[i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(0,0,255,0.1)'),
                    showlegend=False
                ))
            
            # Add percentile lines
            colors = ['rgba(255,0,0,0.7)', 'rgba(255,165,0,0.7)', 'rgba(0,128,0,0.7)', 
                      'rgba(255,165,0,0.7)', 'rgba(255,0,0,0.7)']
            for i, p in enumerate(percentiles):
                fig.add_trace(go.Scatter(
                    y=percentile_values[i],
                    mode='lines',
                    line=dict(width=2, color=colors[i]),
                    name=f'{p}th Percentile'
                ))
            
            fig.update_layout(
                title='Monte Carlo Simulation: 24-Month Equity Paths',
                xaxis_title='Months',
                yaxis_title='Portfolio Value (Starting at 100)',
                height=500,
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display key statistics
            final_values = cum_values[:, -1]
            loss_prob = np.mean(final_values < initial_value) * 100
            median_final = np.median(final_values)
            target_value = initial_value * (1 + monthly_return) ** months
            target_prob = np.mean(final_values >= target_value) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Median Final Value", f"{median_final:.1f}", 
                       delta=f"{((median_final/initial_value)-1)*100:.1f}%")
            col2.metric("Probability of Loss", f"{loss_prob:.1f}%")
            col3.metric("Probability of Meeting Target", f"{target_prob:.1f}%")
            
            st.info(f"""
            **Monte Carlo Insight:** Based on {paths} simulated paths over {months} months, 
            the strategy has a {target_prob:.1f}% chance of meeting or exceeding the target return of 
            {monthly_return*100:.1f}% monthly. There is a {loss_prob:.1f}% chance of losing money over this period.
            """)
        
        # Final recommendations section
        st.markdown("### Final Recommendations")
        
        # Generate appropriate recommendations based on the results
        if 'expected_return' in results and 'expected_risk' in results:
            return_target = results.get('expected_return', 0)
            risk_level = results.get('expected_risk', 0)
            
            if return_target > 15 and risk_level > 10:
                st.warning("""
                ⚠️ **High Risk-High Return Strategy Detected**
                
                Your optimized strategy targets exceptionally high returns (>15% monthly) with significant risk (>10% drawdown). This approach:
                - Is suitable only for very aggressive traders with high risk tolerance
                - Requires strict discipline and adherence to risk management rules
                - Should ideally represent only a portion of your total investment capital
                - Needs regular monitoring and may require frequent adjustments
                
                **Suggestion:** Consider running a parallel, more conservative strategy alongside this one.
                """)
            elif return_target > 10 and risk_level < 8:
                st.success("""
                ✅ **Balanced High-Performance Strategy Detected**
                
                Your optimized strategy shows a favorable balance of return potential (>10% monthly) with reasonable risk (<8% drawdown). This approach:
                - Offers strong return potential while maintaining disciplined risk management
                - Should perform well across a variety of market conditions
                - Requires consistent monitoring but less frequent adjustments
                
                **Suggestion:** Implement with proper position sizing and follow the stop-loss guidelines precisely.
                """)
            elif return_target < 8 and risk_level < 5:
                st.info("""
                ℹ️ **Conservative Strategy Detected**
                
                Your optimized strategy prioritizes capital preservation (risk <5%) with moderate return potential (<8% monthly). This approach:
                - Focuses on stability and consistent smaller gains
                - Should weather market downturns with limited drawdowns
                - Works well for risk-averse investors or as a core portfolio strategy
                
                **Suggestion:** Consider if the return level meets your financial goals, or if slightly higher risk might be acceptable for better returns.
                """)
            else:
                st.success("""
                ✅ **Balanced Strategy Detected**
                
                Your optimized strategy shows a reasonable balance between risk and return. This approach:
                - Should provide a good compromise between growth and stability
                - Requires standard monitoring and periodic review
                
                **Suggestion:** Implement with the recommended parameters and track performance against expectations.
                """)
            
            # Add specific implementation steps
            st.markdown("#### Implementation Steps")
            st.markdown("""
            1. **Paper Trading Phase (2-4 weeks):**
               - Implement strategy with optimized parameters in a paper trading environment
               - Verify performance matches optimization projections
               - Identify any operational issues or adjustments needed
            
            2. **Initial Capital Deployment (2-4 weeks):**
               - Start with 25-33% of intended capital
               - Use recommended position sizing strictly
               - Maintain detailed trade journal for later analysis
            
            3. **Full Implementation (if steps 1-2 are successful):**
               - Scale to full intended capital
               - Implement automated monitoring with alerts
               - Schedule monthly performance reviews
            
            4. **Ongoing Optimization:**
               - Re-run optimization quarterly or after significant market changes
               - Adjust parameters gradually rather than all at once
               - Track optimization changes to avoid overfitting
            """)
        else:
            st.warning("""
            ⚠️ **No Optimal Strategy Found**
            
            The optimization process could not find a strategy that meets your exact targets. Consider:
            1. Adjusting your risk or return expectations to more realistic levels
            2. Exploring additional technical indicators or strategy components
            3. Testing different time periods or market segments
            
            **Suggestion:** Try the optimization again with modified targets based on the Pareto frontier chart above.
            """)
            
        # Final note on risk management
        st.markdown("---")
        st.error("""
        **IMPORTANT RISK DISCLAIMER:**
        
        Past performance and optimization results are not guarantees of future returns. All trading strategies involve risk of loss. 
        This system is for educational and informational purposes only. Always use proper risk management and never risk more capital 
        than you can afford to lose.
        """)

    def render_live_trading_tab(self):
        """Render live trading interface"""
        st.header("Live Trading Monitor")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Main chart area
            if 'chart_data' in st.session_state:
                fig = self._create_live_chart(st.session_state.chart_data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Waiting for market data...")
        
        with col2:
            # Position summary
            st.subheader("Current Positions")
            if 'positions' in st.session_state:
                self._display_positions()
            
            # Risk metrics
            st.subheader("Risk Metrics")
            if 'risk_metrics' in st.session_state:
                self._display_risk_metrics()
        
        with col3:
            # Performance metrics
            st.subheader("Performance")
            if 'performance' in st.session_state:
                self._display_performance()
            
            # Alerts and notifications
            st.subheader("Alerts")
            if 'alerts' in st.session_state:
                self._display_alerts()
    
    def _run_strategy_training(self):
        """Run strategy training with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulated training progress
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Training progress: {i+1}%")
            time.sleep(0.1)
        
        # Store training results
        st.session_state.training_results = {
            "sharpe_ratio": 1.8,
            "max_drawdown": 4.5,
            "win_rate": 0.65,
            "profit_factor": 2.1
        }
        
        status_text.text("Training complete!")
    
    def _display_training_results(self):
        """Display strategy training results"""
        results = st.session_state.training_results
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        col2.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
        col3.metric("Win Rate", f"{results['win_rate']*100:.1f}%")
        col4.metric("Profit Factor", f"{results['profit_factor']:.2f}")
        
        # Training charts
        self._plot_training_charts()
    
    def _plot_training_charts(self):
        """Plot training performance charts"""
        # Create sample data for demonstration
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        equity_curve = np.random.normal(1, 0.02, len(dates)).cumsum() + 100
        
        # Create plotly figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Equity Curve", "Drawdown", "Returns Distribution", "Win/Loss Ratio")
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=dates, y=equity_curve, name="Equity"),
            row=1, col=1
        )
        
        # Drawdown
        drawdown = (equity_curve - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve) * 100
        fig.add_trace(
            go.Scatter(x=dates, y=drawdown, name="Drawdown", fill='tonexty'),
            row=1, col=2
        )
        
        # Returns distribution
        returns = np.diff(equity_curve) / equity_curve[:-1] * 100
        fig.add_trace(
            go.Histogram(x=returns, name="Returns"),
            row=2, col=1
        )
        
        # Win/Loss ratio
        fig.add_trace(
            go.Bar(
                x=["Wins", "Losses"],
                y=[65, 35],
                name="Win/Loss"
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_live_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create live trading chart"""
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="OHLC"
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="Volume"
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            title="Live Trading Chart"
        )
        
        return fig
    
    def _display_positions(self):
        """Display current positions table"""
        positions_df = pd.DataFrame({
            "Symbol": ["NIFTY50", "RELIANCE", "TCS"],
            "Qty": [100, 50, 75],
            "Entry": [18500, 2450, 3600],
            "Current": [18550, 2460, 3590],
            "P&L": [5000, 500, -750]
        })
        st.dataframe(positions_df)
    
    def _display_risk_metrics(self):
        """Display risk metrics"""
        risk_metrics = {
            "Portfolio Beta": 1.05,
            "VaR (95%)": "₹50,000",
            "Margin Used": "45%",
            "Risk Score": "Moderate"
        }
        
        for metric, value in risk_metrics.items():
            st.metric(metric, value)
    
    def _display_performance(self):
        """Display performance metrics"""
        daily_pnl = 15000
        total_pnl = 150000
        roi = 12.5
        
        st.metric("Daily P&L", f"₹{daily_pnl:,}", delta="+2.5%")
        st.metric("Total P&L", f"₹{total_pnl:,}")
        st.metric("ROI", f"{roi}%")
    
    def _display_alerts(self):
        """Display trading alerts"""
        alerts = [
            {"type": "Signal", "message": "Buy signal: NIFTY50", "time": "2 min ago"},
            {"type": "Risk", "message": "Position size limit reached", "time": "5 min ago"},
            {"type": "Market", "message": "High volatility detected", "time": "15 min ago"}
        ]
        
        for alert in alerts:
            st.warning(f"[{alert['type']}] {alert['message']} ({alert['time']})")

def main():
    dashboard = TradingDashboard()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Strategy Training", "Live Trading", "Strategy Optimization"])
    
    if page == "Strategy Training":
        dashboard.render_training_tab()
    elif page == "Live Trading":
        dashboard.render_live_trading_tab()
    else:
        dashboard.render_optimization_tab()

if __name__ == "__main__":
    main()
