#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/test_plot.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set non-interactive backend
plt.switch_backend('Agg')

print("Creating test plot...")

# Create directory
image_dir = Path(__file__).parent / 'images'
image_dir.mkdir(exist_ok=True)

# Create a simple plot
plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Test Plot')
plt.savefig(image_dir / 'test_plot.png')
plt.close()

print(f"Plot saved to {image_dir / 'test_plot.png'}")
