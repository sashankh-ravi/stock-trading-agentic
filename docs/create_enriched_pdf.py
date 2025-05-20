#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/create_enriched_pdf.py

from markdown_pdf import MarkdownPdf
from pathlib import Path
import os
import re

def main():
    """Create a PDF from the enriched markdown with images"""
    # Get the current directory
    current_dir = Path(__file__).parent
    image_dir = current_dir / 'images'
    
    # Set paths
    md_path = current_dir / 'nifty500_trading_system_thesis_enriched.md'
    output_pdf = current_dir / 'Nifty500_Trading_System_Detailed.pdf'
    
    print(f"Reading enriched markdown from: {md_path}")
    
    # Check if the enriched markdown exists
    if not md_path.exists():
        print(f"Error: Enriched markdown file not found at {md_path}")
        print("Running merge_indicator_explanations.py first...")
        
        # Run the merge script to create the enriched markdown
        import subprocess
        result = subprocess.run(['python', 'merge_indicator_explanations.py'], 
                               cwd=current_dir, 
                               capture_output=True, 
                               text=True)
        
        if result.returncode != 0:
            print(f"Error running merge script: {result.stderr}")
            return
        
        print(result.stdout)
        
        # Check again if the file exists
        if not md_path.exists():
            print(f"Error: Enriched markdown file still not found after running merge script")
            return
    
    # Read the markdown file
    with open(md_path, 'r') as f:
        md_content = f.read()
    
    print(f"Successfully read {len(md_content)} characters from enriched markdown")
    
    # Update image paths to be absolute
    def fix_image_path(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        
        if img_path.startswith('./images/'):
            img_filename = os.path.basename(img_path)
            abs_img_path = str(image_dir / img_filename)
            print(f"Fixed image path: {img_filename}")
            return f'![{alt_text}]({abs_img_path})'
        return match.group(0)
    
    md_content_fixed = re.sub(r'!\[(.*?)\]\((.*?)\)', fix_image_path, md_content)
    
    # Write to a temporary file
    temp_md_path = current_dir / 'temp_enriched_thesis.md'
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
        print(f"You may need to install additional dependencies")
        print(f"Try: conda activate stock_env && pip install markdown-pdf PyMuPDF")
    
    # Keep temporary file for debugging
    print(f"Keeping temporary file at {temp_md_path} for debugging")

if __name__ == '__main__':
    main()
