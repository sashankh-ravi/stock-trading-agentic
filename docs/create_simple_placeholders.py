#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/create_simple_placeholders.py

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set the path to the images directory
current_dir = Path(__file__).parent.absolute()
image_dir = current_dir / 'images'

# Ensure the directory exists
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# List of missing images from the warnings
missing_images = [
    "volatility_regimes.png",
    "regime_transitions.png",
    "regime_parameters.png",
    "walk_forward.png",
    "cross_validation.png",
    "monte_carlo.png",
    "live_trading_architecture.png",
    "streamlit_dashboard.png",
    "config_panel.png",
    "performance_analytics.png",
    "tca.png",
    "bull_market_case.png",
    "bear_market_case.png",
    "sideways_market_case.png"
]

def create_simple_placeholder(filename, title):
    """Create a simple matplotlib-based placeholder image"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw a rectangle
    rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='black')
    ax.add_patch(rect)
    
    # Add title
    ax.text(0.5, 0.5, title, 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=20,
            transform=ax.transAxes)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    
    # Save the figure
    output_path = image_dir / filename
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"Created placeholder: {output_path}")
    return output_path

def create_all_missing_images():
    """Create placeholder images for all missing images"""
    print("Creating placeholder images for missing diagrams...")
    
    created_count = 0
    for image_name in missing_images:
        if not os.path.exists(image_dir / image_name):
            # Convert filename to a nice title (replace underscores with spaces, capitalize)
            title = " ".join(word.capitalize() for word in image_name.replace(".png", "").split("_"))
            create_simple_placeholder(image_name, title)
            created_count += 1
    
    print(f"Created {created_count} placeholder images in {image_dir}")

if __name__ == "__main__":
    create_all_missing_images()
