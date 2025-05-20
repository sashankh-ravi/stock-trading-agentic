#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/create_pdf_with_images.py

from markdown_pdf import MarkdownPdf
from pathlib import Path
import os
import re

def main():
    """Create a PDF from markdown with images"""
    # Get the current directory
    current_dir = Path(__file__).parent
    image_dir = current_dir / 'images'
    
    # Set paths
    md_path = current_dir / 'nifty500_trading_system_thesis.md'
    output_pdf = current_dir / 'Nifty500_Thesis_with_Images.pdf'
    
    print(f"Reading markdown from: {md_path}")
    
    # Read the markdown file
    with open(md_path, 'r') as f:
        md_content = f.read()
    
    # Update image paths to be absolute
    def fix_image_path(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        
        if img_path.startswith('./images/'):
            img_filename = os.path.basename(img_path)
            abs_img_path = str(image_dir / img_filename)
            return f'![{alt_text}]({abs_img_path})'
        return match.group(0)
    
    md_content_fixed = re.sub(r'!\[(.*?)\]\((.*?)\)', fix_image_path, md_content)
    
    # Write to a temporary file
    temp_md_path = current_dir / 'temp_thesis_with_images.md'
    with open(temp_md_path, 'w') as f:
        f.write(md_content_fixed)
    
    print(f"Updated markdown with absolute image paths saved to: {temp_md_path}")
    
    # Convert to PDF using markdown-pdf
    try:
        print("Converting to PDF...")
        md_pdf = MarkdownPdf()
        md_pdf.convert(str(temp_md_path), str(output_pdf))
        print(f"PDF created successfully: {output_pdf}")
    except Exception as e:
        print(f"Error converting to PDF: {e}")
    
    # Clean up temporary file
    try:
        os.remove(temp_md_path)
    except Exception as e:
        print(f"Could not remove temporary file: {e}")

if __name__ == '__main__':
    main()
