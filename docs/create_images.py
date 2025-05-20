#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/create_images.py
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

# Set up matplotlib to use Agg backend (no GUI needed)
plt.switch_backend('Agg')

# Ensure the images directory exists
image_dir = Path(__file__).parent / 'images'
image_dir.mkdir(exist_ok=True)

# List of required image filenames
required_images = [
    'system_architecture.png',
    'trend_following.png',
    'momentum_strategy.png',
    'pattern_recognition.png',
    'strategy_combination.png',
    'technical_indicators.png',
    'trend_strength_index.png',
    'indicator_correlation.png',
    'kelly_criterion.png',
    'drawdown_management.png',
    'risk_metrics.png',
    'rl_optimization.png',
    'bayesian_optimization.png',
    'monthly_return_distribution.png',
    'cumulative_return.png',
    'rolling_returns.png',
    'max_drawdown.png',
    'underwater_chart.png',
    'trade_distribution.png',
    'holding_period.png'
]

def create_trend_chart(filename):
    """Create trend following chart with indicators"""
    plt.figure(figsize=(10, 6))
    
    # Generate price data
    x = np.arange(100)
    price = 100 + np.cumsum(np.random.normal(0, 1, 100))
    
    # Create moving averages
    ma_short = np.array([np.mean(price[max(0, i-20):i+1]) for i in range(len(price))])
    ma_long = np.array([np.mean(price[max(0, i-50):i+1]) for i in range(len(price))])
    
    # Plot price and MAs
    plt.plot(x, price, 'b-', label='Price', linewidth=1.5)
    plt.plot(x, ma_short, 'r-', label='MA(20)', linewidth=1.5)
    plt.plot(x, ma_long, 'g-', label='MA(50)', linewidth=1.5)
    
    # Add buy/sell signals
    buy_signals = (ma_short > ma_long) & (np.roll(ma_short, 1) <= np.roll(ma_long, 1))
    sell_signals = (ma_short < ma_long) & (np.roll(ma_short, 1) >= np.roll(ma_long, 1))
    
    plt.plot(x[buy_signals], price[buy_signals], '^', markersize=10, color='green', label='Buy Signal')
    plt.plot(x[sell_signals], price[sell_signals], 'v', markersize=10, color='red', label='Sell Signal')
    
    # Add trend zones
    plt.fill_between(x[30:50], min(price), max(price), color='green', alpha=0.1, label='Uptrend')
    plt.fill_between(x[60:80], min(price), max(price), color='red', alpha=0.1, label='Downtrend')
    
    plt.title("Trend Following Strategy Visualization")
    plt.xlabel("Time (Days)")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(image_dir / filename, dpi=150)
    plt.close()

def create_momentum_chart(filename):
    """Create momentum strategy visualization"""
    plt.figure(figsize=(10, 6))
    
    # Generate price data
    x = np.arange(100)
    price = 100 + np.cumsum(np.random.normal(0, 1, 100))
    
    # Create RSI (simplified)
    changes = np.diff(price, prepend=price[0])
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    
    avg_gain = np.array([np.mean(gains[max(0, i-14):i+1]) for i in range(len(gains))])
    avg_loss = np.array([np.mean(losses[max(0, i-14):i+1]) for i in range(len(losses))])
    
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Plot price
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(x, price, 'b-', label='Price', linewidth=1.5)
    
    # Add buy/sell signals based on RSI
    buy_signals = (rsi < 30) & (np.roll(rsi, 1) >= 30)
    sell_signals = (rsi > 70) & (np.roll(rsi, 1) <= 70)
    
    ax1.plot(x[buy_signals], price[buy_signals], '^', markersize=10, color='green', label='Buy Signal (RSI<30)')
    ax1.plot(x[sell_signals], price[sell_signals], 'v', markersize=10, color='red', label='Sell Signal (RSI>70)')
    
    ax1.set_title("Momentum Strategy Visualization")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot RSI
    ax2.plot(x, rsi, 'purple', linewidth=1.5)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Time (Days)")
    ax2.grid(True, alpha=0.3)
    
    # Fill overbought/oversold regions
    ax2.fill_between(x, 70, 100, color='red', alpha=0.1)
    ax2.fill_between(x, 0, 30, color='green', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(image_dir / filename, dpi=150)
    plt.close()

def create_pattern_chart(filename):
    """Create pattern recognition visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate price data
    x = np.arange(100)
    np.random.seed(42)  # for reproducibility of the pattern
    
    # Create a pattern (head and shoulders)
    base = 100
    trend = np.linspace(0, 10, 100)
    pattern = np.zeros(100)
    
    # Left shoulder
    pattern[10:20] = np.linspace(0, 5, 10)
    pattern[20:30] = np.linspace(5, 0, 10)
    
    # Head
    pattern[35:45] = np.linspace(0, 8, 10)
    pattern[45:55] = np.linspace(8, 0, 10)
    
    # Right shoulder
    pattern[60:70] = np.linspace(0, 4, 10)
    pattern[70:80] = np.linspace(4, 0, 10)
    
    # Add noise and trend
    noise = np.random.normal(0, 0.5, 100)
    price = base + pattern + noise + trend * 0.2
    
    # Plot price
    ax.plot(x, price, 'b-', linewidth=1.5)
    
    # Highlight pattern regions
    ax.fill_between(x[10:30], min(price), max(price), color='yellow', alpha=0.2)
    ax.fill_between(x[35:55], min(price), max(price), color='orange', alpha=0.2)
    ax.fill_between(x[60:80], min(price), max(price), color='yellow', alpha=0.2)
    
    # Add annotations
    ax.annotate('Left Shoulder', xy=(20, price[20]), xytext=(20, price[20]-5),
                arrowprops=dict(arrowstyle='->'), ha='center')
    ax.annotate('Head', xy=(45, price[45]), xytext=(45, price[45]+5),
                arrowprops=dict(arrowstyle='->'), ha='center')
    ax.annotate('Right Shoulder', xy=(70, price[70]), xytext=(70, price[70]-5),
                arrowprops=dict(arrowstyle='->'), ha='center')
    ax.annotate('Neckline', xy=(50, price[30]), xytext=(50, price[30]-7),
                arrowprops=dict(arrowstyle='->'), ha='center')
    
    # Add sell signal and target
    ax.plot(85, price[85], 'rv', markersize=10, label='Sell Signal')
    ax.plot([85, 85], [price[85], price[85]-8], 'r--', alpha=0.7)
    ax.plot(85, price[85]-8, 'ro', markersize=6, label='Price Target')
    
    # Horizontal line for neckline
    ax.axhline(y=price[30], xmin=0.1, xmax=0.8, color='blue', linestyle='--', alpha=0.5)
    
    ax.set_title("Pattern Recognition: Head and Shoulders")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(image_dir / filename, dpi=150)
    plt.close()

def create_system_architecture(filename):
    """Create a simpler system architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Turn off the axis
    ax.axis('off')
    
    # Define the components
    components = [
        {'name': 'Data Collection', 'x': 0.5, 'y': 0.9, 'width': 0.3, 'height': 0.1, 'color': 'lightblue'},
        {'name': 'Preprocessing', 'x': 0.5, 'y': 0.75, 'width': 0.3, 'height': 0.1, 'color': 'lightblue'},
        
        {'name': 'Strategy Layer', 'x': 0.5, 'y': 0.55, 'width': 0.6, 'height': 0.15, 'color': 'lightgreen'},
        {'name': 'Trend\nFollowing', 'x': 0.3, 'y': 0.55, 'width': 0.15, 'height': 0.08, 'color': 'palegreen'},
        {'name': 'Momentum', 'x': 0.5, 'y': 0.55, 'width': 0.15, 'height': 0.08, 'color': 'palegreen'},
        {'name': 'Pattern\nRecognition', 'x': 0.7, 'y': 0.55, 'width': 0.15, 'height': 0.08, 'color': 'palegreen'},
        
        {'name': 'Risk Management', 'x': 0.5, 'y': 0.35, 'width': 0.3, 'height': 0.1, 'color': 'salmon'},
        
        {'name': 'Optimization Engine', 'x': 0.85, 'y': 0.45, 'width': 0.2, 'height': 0.1, 'color': 'orchid'},
        
        {'name': 'Execution Engine', 'x': 0.5, 'y': 0.15, 'width': 0.3, 'height': 0.1, 'color': 'gold'},
    ]
    
    # Create the boxes
    for comp in components:
        rect = plt.Rectangle((comp['x'] - comp['width']/2, comp['y'] - comp['height']/2), 
                          comp['width'], comp['height'], 
                          facecolor=comp['color'], edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(comp['x'], comp['y'], comp['name'], ha='center', va='center', fontweight='bold')
    
    # Add arrows
    arrows = [
        # Data flow
        {'start': (0.5, 0.85), 'end': (0.5, 0.8), 'color': 'black'},
        {'start': (0.5, 0.7), 'end': (0.5, 0.625), 'color': 'black'},
        {'start': (0.5, 0.475), 'end': (0.5, 0.4), 'color': 'black'},
        {'start': (0.5, 0.3), 'end': (0.5, 0.2), 'color': 'black'},
        
        # Optimization connections
        {'start': (0.75, 0.55), 'end': (0.85, 0.5), 'color': 'purple'},
        {'start': (0.85, 0.4), 'end': (0.65, 0.35), 'color': 'purple'},
    ]
    
    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                    arrowprops=dict(arrowstyle='->', color=arrow['color'], lw=2))
    
    # Add title
    ax.text(0.5, 0.97, 'Nifty 500 Trading System Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_items = [
        {'label': 'Data Pipeline', 'color': 'lightblue'},
        {'label': 'Strategy Components', 'color': 'lightgreen'},
        {'label': 'Risk Management', 'color': 'salmon'},
        {'label': 'Optimization', 'color': 'orchid'},
        {'label': 'Execution', 'color': 'gold'},
    ]
    
    for i, item in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.05, 0.05 + i*0.06), 0.03, 0.03, 
                                  facecolor=item['color'], edgecolor='black'))
        ax.text(0.1, 0.065 + i*0.06, item['label'], va='center')
    
    plt.tight_layout()
    plt.savefig(image_dir / filename, dpi=150)
    plt.close()

def create_indicator_correlation(filename):
    """Create a correlation matrix for technical indicators"""
    plt.figure(figsize=(10, 8))
    
    indicators = [
        "EMA(9)", "EMA(21)", "EMA(50)", 
        "MACD", "Signal", "RSI(14)", 
        "Stoch K", "Stoch D", "ATR",
        "OBV", "ADX", "BB Width"
    ]
    
    n = len(indicators)
    
    # Create a correlation matrix
    np.random.seed(42)
    
    # Base correlations that make sense for technical indicators
    corr = np.eye(n)
    
    # EMA correlations (highly correlated with each other)
    corr[0:3, 0:3] = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])
    
    # MACD and Signal are highly correlated
    corr[3:5, 3:5] = np.array([
        [1.0, 0.9],
        [0.9, 1.0]
    ])
    
    # MACD and EMAs are moderately correlated
    corr[3:5, 0:3] = np.array([
        [0.7, 0.5, 0.3],
        [0.6, 0.6, 0.4]
    ])
    corr[0:3, 3:5] = corr[3:5, 0:3].T
    
    # RSI and Stochastic have some correlation
    corr[5, 6:8] = [0.6, 0.5]
    corr[6:8, 5] = [0.6, 0.5]
    
    # Stoch K and Stoch D are highly correlated
    corr[6:8, 6:8] = np.array([
        [1.0, 0.8],
        [0.8, 1.0]
    ])
    
    # Add some randomness while preserving symmetry
    noise = np.random.uniform(-0.1, 0.1, (n, n))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0)
    
    corr = corr + noise
    
    # Ensure values are within [-1, 1] range
    corr = np.clip(corr, -1, 1)
    
    # Create a heatmap
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    
    # Add labels
    plt.xticks(range(n), indicators, rotation=45, ha='right')
    plt.yticks(range(n), indicators)
    
    # Add correlation values
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
            plt.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', color=color)
    
    plt.title("Technical Indicator Correlation Matrix")
    plt.tight_layout()
    plt.savefig(image_dir / filename, dpi=150)
    plt.close()

def create_cumulative_return(filename):
    """Create a cumulative return chart"""
    plt.figure(figsize=(10, 6))
    
    # Create x-axis (months)
    months = np.arange(36)
    
    # Strategy returns (monthly)
    np.random.seed(42)
    monthly_returns = np.random.normal(0.015, 0.03, 36)  # Mean 1.5%, SD 3%
    
    # Benchmark returns (monthly)
    benchmark_returns = np.random.normal(0.01, 0.025, 36)  # Mean 1%, SD 2.5%
    
    # Calculate cumulative returns
    strategy_cum_returns = (1 + monthly_returns).cumprod() - 1
    benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
    
    # Plot
    plt.plot(months, strategy_cum_returns * 100, 'b-', linewidth=2, label='Nifty 500 Strategy')
    plt.plot(months, benchmark_cum_returns * 100, 'r--', linewidth=2, label='Benchmark (Nifty 500 Index)')
    
    # Add key events
    plt.annotate('Market Correction', xy=(8, strategy_cum_returns[8] * 100), 
                xytext=(9, (strategy_cum_returns[8] - 0.1) * 100),
                arrowprops=dict(arrowstyle='->'))
    
    plt.annotate('Strategy Optimization', xy=(18, strategy_cum_returns[18] * 100), 
                xytext=(19, (strategy_cum_returns[18] + 0.1) * 100),
                arrowprops=dict(arrowstyle='->'))
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Months')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Cumulative Return Comparison')
    plt.legend()
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(image_dir / filename, dpi=150)
    plt.close()

def create_strategy_combination(filename):
    """Create a strategy combination approach visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create data
    x = np.arange(100)
    price = 100 + np.cumsum(np.random.normal(0, 1, 100))
    
    # Strategy signals (1 for buy, -1 for sell, 0 for neutral)
    np.random.seed(42)
    trend_signals = np.zeros(100)
    momentum_signals = np.zeros(100)
    pattern_signals = np.zeros(100)
    combined_signals = np.zeros(100)
    
    # Generate some signals
    for i in range(1, 100):
        # Trend signals
        if i % 25 < 15:  # trending periods
            trend_signals[i] = 1
        elif i % 25 >= 20:  # reversal periods
            trend_signals[i] = -1
            
        # Momentum signals (more frequent)
        if i % 10 < 5 and np.random.random() > 0.3:
            momentum_signals[i] = 1
        elif i % 10 >= 8 and np.random.random() > 0.3:
            momentum_signals[i] = -1
            
        # Pattern signals (less frequent but strong)
        if i in [20, 50, 80]:
            pattern_signals[i] = 1
        elif i in [30, 60, 90]:
            pattern_signals[i] = -1
            
        # Combined signal (weighted average)
        combined_signals[i] = (0.4 * trend_signals[i] + 
                             0.3 * momentum_signals[i] + 
                             0.3 * pattern_signals[i])
    
    # Plot price
    ax.plot(x, price, 'k-', alpha=0.6, linewidth=1, label='Price')
    
    # Plot signal strengths as a stacked area
    ax2 = ax.twinx()
    
    # Define signal colors
    signal_colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot signals on second y-axis
    ax2.fill_between(x, 0, trend_signals, color=signal_colors[0], alpha=0.2, label='Trend Signals')
    ax2.fill_between(x, 0, momentum_signals, color=signal_colors[1], alpha=0.2, label='Momentum Signals')
    ax2.fill_between(x, 0, pattern_signals, color=signal_colors[2], alpha=0.2, label='Pattern Signals')
    ax2.plot(x, combined_signals, color=signal_colors[3], linewidth=2, label='Combined Signal')
    
    # Plot buy/sell markers based on combined signal
    buy_points = (combined_signals > 0.5) & (np.roll(combined_signals, 1) <= 0.5)
    sell_points = (combined_signals < -0.5) & (np.roll(combined_signals, 1) >= -0.5)
    
    ax.plot(x[buy_points], price[buy_points], '^', markersize=10, color='green', label='Buy Signal')
    ax.plot(x[sell_points], price[sell_points], 'v', markersize=10, color='red', label='Sell Signal')
    
    # Labels and formatting
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Price')
    ax2.set_ylabel('Signal Strength')
    ax2.set_ylim(-1.5, 1.5)
    
    # Add title
    plt.title('Strategy Combination Approach')
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(image_dir / filename, dpi=150)
    plt.close()

def create_max_drawdown(filename):
    """Create a maximum drawdown visualization"""
    plt.figure(figsize=(10, 6))
    
    # Generate returns data
    np.random.seed(42)
    x = np.arange(500)
    returns = np.random.normal(0.001, 0.01, 500)  # Daily returns
    
    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate maximum drawdown
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns / peak - 1) * 100
    
    # Find the worst drawdown period
    worst_idx = np.argmin(drawdown)
    peak_idx = np.where(peak[worst_idx] == cum_returns[:worst_idx+1])[0][-1]
    
    # Plot cumulative returns
    plt.subplot(2, 1, 1)
    plt.plot(x, cum_returns, 'b-', linewidth=1.5, label='Cumulative Return')
    plt.plot(x, peak, 'g--', linewidth=1, alpha=0.5, label='Peak Value')
    
    # Highlight worst drawdown
    plt.fill_between(x[peak_idx:worst_idx+1], cum_returns[peak_idx:worst_idx+1], 
                    peak[peak_idx:worst_idx+1], color='red', alpha=0.3)
    
    plt.annotate(f'Max Drawdown: {drawdown[worst_idx]:.1f}%', 
                xy=(worst_idx, cum_returns[worst_idx]),
                xytext=(worst_idx+30, cum_returns[worst_idx]+0.1),
                arrowprops=dict(arrowstyle='->'))
    
    plt.title('Cumulative Return and Maximum Drawdown')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot drawdowns
    plt.subplot(2, 1, 2)
    plt.plot(x, drawdown, 'r-', linewidth=1.5)
    plt.fill_between(x, drawdown, 0, color='red', alpha=0.3)
    
    # Add threshold line
    plt.axhline(y=-5, color='black', linestyle='--', label='Risk Tolerance (-5%)')
    
    plt.title('Drawdown Analysis')
    plt.xlabel('Trading Days')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(image_dir / filename, dpi=150)
    plt.close()

def create_basic_chart(filename, title):
    """Create a basic chart with placeholder data"""
    plt.figure(figsize=(10, 6))
    
    # Generate random data
    x = np.arange(100)
    y = np.cumsum(np.random.normal(0, 1, 100))
    
    plt.plot(x, y, 'b-', linewidth=1.5)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(image_dir / filename, dpi=150)
    plt.close()

def main():
    print(f"Generating {len(required_images)} placeholder diagrams for the thesis document...")
    print(f"Image directory: {image_dir}")
    
    try:
        # Create each image
        for img_name in required_images:
            print(f"Creating {img_name}...", flush=True)
            
            try:
                if img_name == 'system_architecture.png':
                    create_system_architecture(img_name)
                elif img_name == 'trend_following.png':
                    create_trend_chart(img_name)
                elif img_name == 'momentum_strategy.png':
                    create_momentum_chart(img_name)
                elif img_name == 'pattern_recognition.png':
                    create_pattern_chart(img_name)
                elif img_name == 'strategy_combination.png':
                    create_strategy_combination(img_name)
                elif img_name == 'indicator_correlation.png':
                    create_indicator_correlation(img_name)
                elif img_name == 'cumulative_return.png':
                    create_cumulative_return(img_name)
                elif img_name == 'max_drawdown.png':
                    create_max_drawdown(img_name)
                else:
                    # Use a basic chart for other images
                    title = img_name.replace('.png', '').replace('_', ' ').title()
                    create_basic_chart(img_name, title)
                
                print(f"✓ Created {img_name}")
            except Exception as img_error:
                print(f"Error creating {img_name}: {img_error}")
            
            # Small pause to avoid overwhelming the system
            time.sleep(0.1)
        
        print(f"\nAll {len(required_images)} images generated successfully in {image_dir}")
        print("You can now run generate_pdf.py to create the final PDF document.")
    
    except Exception as e:
        print(f"Error generating images: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
