#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/create_missing_images.py

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

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

# Function to create a placeholder image with text
def create_placeholder_with_text(filename, title, size=(800, 600), bg_color="white", text_color="black"):
    # Create a new image with white background
    image = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # Just use the default font to avoid font issues
    font = ImageFont.load_default()
    
    # Draw a title
    draw.text((size[0]/2, size[1]/2), title, fill=text_color, font=font, anchor="mm" if hasattr(draw, "textbbox") else None)
    
    # Position text in the center
    w, h = draw.textsize(title, font=font) if hasattr(draw, 'textsize') else (size[0]/2, size[1]/2)
    position = ((size[0]-w)/2, (size[1]-h)/2)
    
    # Draw the text
    draw.text(position, title, fill=text_color, font=font)
    
    # Add a border
    draw.rectangle([(0, 0), (size[0]-1, size[1]-1)], outline=text_color)
    
    # Save the image
    output_path = image_dir / filename
    image.save(output_path)
    print(f"Created placeholder image: {output_path}")
    return output_path

# Function to create a volatility regimes plot
def create_volatility_regimes():
    plt.figure(figsize=(10, 6))
    
    # Simulated data
    np.random.seed(42)
    days = 500
    x = np.arange(days)
    
    # Create artificial regimes
    low_vol = np.random.normal(0, 0.5, size=days//4)
    high_vol = np.random.normal(0, 2.0, size=days//4)
    medium_vol = np.random.normal(0, 1.0, size=days//2)
    
    # Combine the regimes
    y = np.concatenate([low_vol, high_vol, medium_vol])
    
    # Cumulative returns
    cumret = np.cumsum(y)
    
    # Plot the data
    plt.plot(x, cumret, label='Price')
    
    # Shade regions for different regimes
    plt.axvspan(0, days//4, alpha=0.2, color='green', label='Low Volatility')
    plt.axvspan(days//4, days//2, alpha=0.2, color='red', label='High Volatility')
    plt.axvspan(days//2, days, alpha=0.2, color='yellow', label='Medium Volatility')
    
    plt.title('Market Volatility Regimes')
    plt.xlabel('Time (Days)')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    output_path = image_dir / 'volatility_regimes.png'
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Created: {output_path}")

# Function to create a regime transitions plot
def create_regime_transitions():
    plt.figure(figsize=(8, 6))
    
    # Create a transition matrix visualization
    transition_matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.6, 0.1],
        [0.2, 0.3, 0.5]
    ])
    
    regimes = ['Low Volatility', 'Medium Volatility', 'High Volatility']
    
    sns.heatmap(transition_matrix, annot=True, cmap='YlGnBu', fmt='.2f',
                xticklabels=regimes, yticklabels=regimes)
    
    plt.title('Regime Transition Probabilities')
    plt.xlabel('To Regime')
    plt.ylabel('From Regime')
    
    # Save the figure
    plt.tight_layout()
    output_path = image_dir / 'regime_transitions.png'
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Created: {output_path}")

# Create bull, bear, and sideways market case studies
def create_market_case_studies():
    # Create bull market case
    plt.figure(figsize=(10, 6))
    x = np.arange(100)
    y_bull = np.cumsum(np.random.normal(0.2, 1, size=100))  # Positive drift
    plt.plot(x, y_bull)
    plt.title('Bull Market Case Study')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(image_dir / 'bull_market_case.png', dpi=100)
    plt.close()
    print(f"Created: {image_dir / 'bull_market_case.png'}")
    
    # Create bear market case
    plt.figure(figsize=(10, 6))
    y_bear = np.cumsum(np.random.normal(-0.2, 1.2, size=100))  # Negative drift
    plt.plot(x, y_bear)
    plt.title('Bear Market Case Study')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(image_dir / 'bear_market_case.png', dpi=100)
    plt.close()
    print(f"Created: {image_dir / 'bear_market_case.png'}")
    
    # Create sideways market case
    plt.figure(figsize=(10, 6))
    y_sideways = np.cumsum(np.random.normal(0, 0.7, size=100))  # No drift
    plt.plot(x, y_sideways)
    plt.title('Sideways Market Case Study')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(image_dir / 'sideways_market_case.png', dpi=100)
    plt.close()
    print(f"Created: {image_dir / 'sideways_market_case.png'}")

# Create an architecture diagram for live trading
def create_live_trading_architecture():
    # Create a custom diagram
    image = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(image)
    
    # Just use the default font
    font = ImageFont.load_default()
    
    # Draw title
    draw.text((400, 30), "Live Trading Architecture", fill="black", font=title_font, anchor="mm")
    
    # Draw boxes for components
    components = [
        {"name": "Data Feed", "pos": (200, 100), "size": (200, 80)},
        {"name": "Strategy Engine", "pos": (400, 200), "size": (200, 80)},
        {"name": "Risk Manager", "pos": (400, 300), "size": (200, 80)},
        {"name": "Order Manager", "pos": (400, 400), "size": (200, 80)},
        {"name": "Broker API", "pos": (400, 500), "size": (200, 80)},
        {"name": "Monitoring", "pos": (600, 200), "size": (200, 80)},
    ]
    
    # Draw each component
    for comp in components:
        x, y = comp["pos"]
        w, h = comp["size"]
        # Draw box
        draw.rectangle([(x-w/2, y-h/2), (x+w/2, y+h/2)], outline="black", width=2)
        # Draw text
        draw.text((x, y), comp["name"], fill="black", font=font, anchor="mm")
    
    # Draw arrows connecting components
    arrows = [
        ((200, 140), (400, 180)),  # Data Feed to Strategy
        ((400, 240), (400, 280)),  # Strategy to Risk
        ((400, 340), (400, 380)),  # Risk to Order
        ((400, 440), (400, 480)),  # Order to Broker
        ((500, 200), (580, 200)),  # Strategy to Monitoring
    ]
    
    for start, end in arrows:
        draw.line([start, end], fill="black", width=2)
        # Draw arrowhead
        x1, y1 = start
        x2, y2 = end
        angle = np.arctan2(y2-y1, x2-x1)
        x3 = x2 - 10 * np.cos(angle - np.pi/6)
        y3 = y2 - 10 * np.sin(angle - np.pi/6)
        x4 = x2 - 10 * np.cos(angle + np.pi/6)
        y4 = y2 - 10 * np.sin(angle + np.pi/6)
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill="black")
    
    # Save the image
    output_path = image_dir / 'live_trading_architecture.png'
    image.save(output_path)
    print(f"Created: {output_path}")

# Function to create unique diagrams for each missing image
def create_all_missing_images():
    print("Creating missing images...")
    
    # Call specialized functions for some images
    create_volatility_regimes()
    create_regime_transitions()
    create_market_case_studies()
    create_live_trading_architecture()
    
    # Create generic placeholders for the rest
    for image_name in missing_images:
        if not os.path.exists(image_dir / image_name):
            # Convert filename to a nice title (replace underscores with spaces, capitalize)
            title = " ".join(word.capitalize() for word in image_name.replace(".png", "").split("_"))
            create_placeholder_with_text(image_name, title)
    
    print(f"All missing images created in {image_dir}")

if __name__ == "__main__":
    create_all_missing_images()
