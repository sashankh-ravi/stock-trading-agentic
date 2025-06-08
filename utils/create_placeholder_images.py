#!/usr/bin/env python3
"""
Create placeholder PNG images with text for Mermaid diagrams

This script creates placeholder PNG images with text explaining how to generate
the actual diagrams from HTML files.

Usage:
    python create_placeholder_images.py
"""

from PIL import Image, ImageDraw, ImageFont
import os
import glob
import argparse

def create_placeholder_image(output_path, width=800, height=600, title=""):
    """
    Create a placeholder PNG image with instructional text
    
    Args:
        output_path: Path to save the PNG image
        width: Image width
        height: Image height
        title: Title of the diagram
    """
    # Create a white image
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Use default font
    try:
        font_large = ImageFont.truetype("Arial", 24)
        font_small = ImageFont.truetype("Arial", 16)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw title
    title_text = f"Diagram: {title}" if title else "Diagram Placeholder"
    draw.text((width//2, 100), title_text, fill=(0, 0, 0), anchor="mm", font=font_large)
    
    # Draw instruction
    instructions = [
        "This is a placeholder for a Mermaid diagram.",
        "To generate the actual diagram:",
        "",
        "1. Run the HTML to PNG converter:",
        "   python html_to_png_converter.py",
        "",
        "2. Or open the corresponding HTML file in a browser",
        "   and take a screenshot."
    ]
    
    y_pos = height // 2 - 80
    for line in instructions:
        draw.text((width//2, y_pos), line, fill=(0, 0, 0), anchor="mm", font=font_small)
        y_pos += 30
    
    # Save the image
    image.save(output_path)
    print(f"Created placeholder image: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create placeholder PNG images for Mermaid diagrams")
    parser.add_argument("--dir", type=str, default="docs/images", 
                        help="Directory containing HTML files (default: docs/images)")
    args = parser.parse_args()
    
    # Find all HTML files in the specified directory
    html_files = glob.glob(os.path.join(args.dir, "*.html"))
    
    if not html_files:
        print(f"No HTML files found in {args.dir}")
        return
    
    # Create placeholder PNG for each HTML file
    for html_file in html_files:
        png_file = os.path.splitext(html_file)[0] + '.png'
        title = os.path.splitext(os.path.basename(html_file))[0].replace('_', ' ').title()
        create_placeholder_image(png_file, title=title)
    
    print(f"Created {len(html_files)} placeholder images")

if __name__ == "__main__":
    main()
