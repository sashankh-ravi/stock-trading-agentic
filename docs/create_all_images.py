#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/create_all_images.py

"""
Helper script to create all required images for the thesis document.
"""

from pathlib import Path
import sys
import os

def create_all_images():
    """Create all required images for the thesis document"""
    try:
        # Import the create_images module
        sys.path.append(str(Path(__file__).parent))
        
        from create_images import required_images, image_dir
        from create_images import (
            create_system_architecture,
            create_trend_chart,
            create_momentum_chart,
            create_pattern_chart,
            create_indicator_correlation,
            create_cumulative_return,
            create_max_drawdown,
            create_basic_chart
        )
        
        print(f"Generating {len(required_images)} images for the thesis document...")
        
        # System Architecture
        create_system_architecture('system_architecture.png')
        
        # Trading Strategies
        create_trend_chart('trend_following.png')
        create_momentum_chart('momentum_strategy.png')
        create_pattern_chart('pattern_recognition.png')
        create_basic_chart('strategy_combination.png', 'Strategy Combination Approach')
        
        # Technical Indicators
        create_basic_chart('technical_indicators.png', 'Technical Indicators')
        create_basic_chart('trend_strength_index.png', 'Trend Strength Index')
        create_indicator_correlation('indicator_correlation.png')
        
        # Risk Management
        create_basic_chart('kelly_criterion.png', 'Kelly Criterion Optimization')
        create_basic_chart('drawdown_management.png', 'Drawdown Management')
        create_basic_chart('risk_metrics.png', 'Risk Metrics')
        
        # Optimization
        create_basic_chart('rl_optimization.png', 'Reinforcement Learning Optimization')
        create_basic_chart('bayesian_optimization.png', 'Bayesian Optimization')
        
        # Performance Metrics
        create_basic_chart('monthly_return_distribution.png', 'Monthly Return Distribution')
        create_cumulative_return('cumulative_return.png')
        create_basic_chart('rolling_returns.png', 'Rolling Returns')
        create_max_drawdown('max_drawdown.png')
        create_basic_chart('underwater_chart.png', 'Underwater Chart')
        create_basic_chart('trade_distribution.png', 'Trade Distribution')
        create_basic_chart('holding_period.png', 'Holding Period Analysis')
        
        print(f"All {len(required_images)} images have been generated in: {image_dir}")
        
        # Verify images were created
        created_images = list(image_dir.glob('*.png'))
        print(f"Created {len(created_images)} image files")
        
        # Check for any missing images
        missing = [img for img in required_images if not (image_dir / img).exists()]
        if missing:
            print(f"WARNING: {len(missing)} images are missing:")
            for img in missing:
                print(f"  - {img}")
            return False
        else:
            print("All required images were created successfully")
            return True
    
    except Exception as e:
        print(f"Error creating images: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_all_images()
    sys.exit(0 if success else 1)
