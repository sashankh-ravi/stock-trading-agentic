"""Performance Visualization and Reporting

Implements comprehensive visualization and reporting functionality:
1. Performance metrics and charts
2. Risk analytics
3. Strategy attribution
4. Market regime analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceVisualizer:
    def __init__(self, save_path: str = 'reports'):
        """Initialize visualizer with save path"""
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')  # Updated to use the new style name
        sns.set_palette("husl")

    def create_performance_report(self,
                               equity_curve: pd.Series,
                               trades: List[Dict],
                               metrics: Dict,
                               regime_analysis: Dict,
                               strategy_weights: Dict,
                               benchmark: Optional[pd.Series] = None) -> str:
        """Create comprehensive performance report
        
        Returns:
            Path to saved report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.save_path / f"report_{timestamp}.html"
            
            # Create plotly figure
            fig = go.Figure()
            
            # 1. Equity Curve
            self._add_equity_curve(fig, equity_curve, benchmark)
            
            # 2. Drawdown Chart
            self._add_drawdown_chart(fig, equity_curve)
            
            # 3. Strategy Attribution
            self._add_strategy_attribution(fig, strategy_weights)
            
            # 4. Regime Analysis
            self._add_regime_analysis(fig, regime_analysis)
            
            # 5. Trade Analysis
            self._add_trade_analysis(fig, trades)
            
            # Add performance metrics table
            self._add_metrics_table(fig, metrics)
            
            # Save report
            fig.write_html(str(report_path))
            
            # Create additional plots
            self._create_supplementary_plots(
                equity_curve, trades, metrics, regime_analysis, timestamp
            )
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error creating performance report: {e}")
            return ""

    def _add_equity_curve(self,
                        fig: go.Figure,
                        equity: pd.Series,
                        benchmark: Optional[pd.Series] = None) -> None:
        """Add equity curve to figure"""
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                name="Strategy",
                line=dict(color='blue', width=2)
            )
        )
        
        if benchmark is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark.values,
                    name="Benchmark",
                    line=dict(color='gray', width=2, dash='dash')
                )
            )
            
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Value",
            showlegend=True
        )

    def _add_drawdown_chart(self,
                          fig: go.Figure,
                          equity: pd.Series) -> None:
        """Add drawdown chart"""
        drawdown = (equity / equity.cummax() - 1) * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown",
                line=dict(color='red', width=1),
                yaxis="y2"
            )
        )
        
        fig.update_layout(
            yaxis2=dict(
                title="Drawdown %",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    def _add_strategy_attribution(self,
                               fig: go.Figure,
                               weights: Dict[str, List[float]]) -> None:
        """Add strategy attribution chart"""
        df = pd.DataFrame(weights)
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df.values,
                name="Strategy Weights",
                text=df.values.round(2),
                textposition='auto',
            )
        )
        
        fig.update_layout(
            title="Strategy Attribution",
            barmode='stack',
            showlegend=True
        )

    def _add_regime_analysis(self,
                          fig: go.Figure,
                          regime_analysis: Dict) -> None:
        """Add regime analysis visualization"""
        regimes = pd.DataFrame(regime_analysis)
        
        # Create heatmap of regime transitions
        if 'transitions' in regime_analysis:
            transitions = pd.DataFrame(regime_analysis['transitions'])
            
            fig.add_trace(
                go.Heatmap(
                    z=transitions.values,
                    x=transitions.columns,
                    y=transitions.index,
                    colorscale='RdYlBu',
                    name="Regime Transitions"
                )
            )

    def _add_trade_analysis(self,
                         fig: go.Figure,
                         trades: List[Dict]) -> None:
        """Add trade analysis visualizations"""
        df = pd.DataFrame(trades)
        
        # Plot trade returns distribution
        if 'return' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['return'],
                    nbinsx=50,
                    name="Trade Returns",
                    opacity=0.75
                )
            )
            
            fig.update_layout(
                title="Trade Returns Distribution",
                xaxis_title="Return",
                yaxis_title="Count"
            )

    def _add_metrics_table(self,
                         fig: go.Figure,
                         metrics: Dict) -> None:
        """Add performance metrics table"""
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        list(metrics.keys()),
                        [f"{v:.2%}" if isinstance(v, float) else v
                         for v in metrics.values()]
                    ],
                    fill_color='lavender',
                    align='left'
                )
            )
        )

    def _create_supplementary_plots(self,
                                 equity: pd.Series,
                                 trades: List[Dict],
                                 metrics: Dict,
                                 regime_analysis: Dict,
                                 timestamp: str) -> None:
        """Create additional analysis plots"""
        # 1. Monthly Returns Heatmap
        self._plot_monthly_returns(equity, timestamp)
        
        # 2. Rolling Statistics
        self._plot_rolling_stats(equity, timestamp)
        
        # 3. Regime Performance
        self._plot_regime_performance(regime_analysis, timestamp)
        
        # 4. Risk Analytics
        self._plot_risk_analytics(equity, metrics, timestamp)

    def _plot_monthly_returns(self,
                           equity: pd.Series,
                           timestamp: str) -> None:
        """Create monthly returns heatmap"""
        returns = equity.pct_change()
        monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        
        # Reshape to year x month
        monthly_matrix = monthly_returns.to_frame()
        monthly_matrix.index = pd.MultiIndex.from_arrays([
            monthly_matrix.index.year,
            monthly_matrix.index.month
        ])
        monthly_matrix = monthly_matrix.unstack()
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            monthly_matrix,
            annot=True,
            fmt='.1%',
            center=0,
            cmap='RdYlGn'
        )
        plt.title('Monthly Returns Heatmap')
        plt.savefig(self.save_path / f'monthly_returns_{timestamp}.png')
        plt.close()

    def _plot_rolling_stats(self,
                         equity: pd.Series,
                         timestamp: str) -> None:
        """Plot rolling statistics"""
        returns = equity.pct_change()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Rolling Sharpe Ratio
        rolling_sharpe = (
            returns.rolling(252).mean() / returns.rolling(252).std()
        ) * np.sqrt(252)
        rolling_sharpe.plot(ax=ax1)
        ax1.set_title('Rolling 1-Year Sharpe Ratio')
        ax1.grid(True)
        
        # Rolling Volatility
        rolling_vol = returns.rolling(63).std() * np.sqrt(252)
        rolling_vol.plot(ax=ax2)
        ax2.set_title('Rolling 3-Month Volatility')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_path / f'rolling_stats_{timestamp}.png')
        plt.close()

    def _plot_regime_performance(self,
                              regime_analysis: Dict,
                              timestamp: str) -> None:
        """Plot regime-specific performance"""
        regimes = pd.DataFrame(regime_analysis)
        
        plt.figure(figsize=(10, 6))
        for regime in regimes.columns:
            if '_return' in regime:
                plt.bar(
                    regime.replace('_return', ''),
                    regimes[regime],
                    alpha=0.7
                )
                
        plt.title('Performance by Market Regime')
        plt.ylabel('Return')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_path / f'regime_performance_{timestamp}.png')
        plt.close()

    def _plot_risk_analytics(self,
                          equity: pd.Series,
                          metrics: Dict,
                          timestamp: str) -> None:
        """Plot risk analytics"""
        returns = equity.pct_change()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Returns Distribution
        sns.histplot(returns, kde=True, ax=ax1)
        ax1.set_title('Returns Distribution')
        ax1.grid(True)
        
        # Drawdown Analysis
        drawdown = (equity / equity.cummax() - 1) * 100
        drawdown.plot(ax=ax2)
        ax2.set_title('Drawdown Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_path / f'risk_analytics_{timestamp}.png')
        plt.close()
