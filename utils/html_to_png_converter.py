#!/usr/bin/env python3
"""
HTML to PNG Converter for Mermaid Diagrams

This script converts HTML files containing Mermaid diagrams to PNG images.
It requires Chrome or Chromium to be installed with the headless mode capability.

Usage:
    python html_to_png_converter.py

Requirements:
    pip install selenium pillow webdriver_manager
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import argparse

def convert_html_to_png(html_file, output_png=None, wait_time=3):
    """
    Convert an HTML file with a Mermaid diagram to a PNG image
    
    Args:
        html_file: Path to the HTML file
        output_png: Path to save the PNG (if None, uses the same name as HTML but with .png)
        wait_time: Time to wait for diagram rendering in seconds
    
    Returns:
        Path to the generated PNG file
    """
    if output_png is None:
        output_png = os.path.splitext(html_file)[0] + '.png'
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Initialize Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Load the HTML file
        driver.get(f"file://{os.path.abspath(html_file)}")
        
        # Wait for the diagram to render
        time.sleep(wait_time)
        
        # Get the diagram element
        diagram_element = driver.find_element("css selector", "#diagram")
        
        # Take a screenshot of the diagram
        diagram_element.screenshot(output_png)
        
        print(f"Converted {html_file} to {output_png}")
        return output_png
    finally:
        driver.quit()

def main():
    parser = argparse.ArgumentParser(description="Convert HTML Mermaid diagrams to PNG images")
    parser.add_argument("--dir", type=str, default="docs/images", 
                        help="Directory containing HTML files (default: docs/images)")
    parser.add_argument("--wait", type=int, default=3,
                        help="Wait time in seconds for diagram rendering (default: 3)")
    args = parser.parse_args()
    
    # Find all HTML files in the specified directory
    html_files = []
    for root, _, files in os.walk(args.dir):
        for file in files:
            if file.endswith(".html"):
                html_files.append(os.path.join(root, file))
    
    if not html_files:
        print(f"No HTML files found in {args.dir}")
        return
    
    print(f"Found {len(html_files)} HTML files to convert:")
    for html_file in html_files:
        print(f"  - {html_file}")
    
    # Convert each HTML file to PNG
    for html_file in html_files:
        convert_html_to_png(html_file, wait_time=args.wait)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
