#!/usr/bin/env python3
"""
Simple Thesis Generator

This script combines the main thesis content with indicator explanations
and generates a clean PDF without complicated logic.
"""

import markdown
import pdfkit
import os
import re
from pathlib import Path
import sys

def generate_pdf():
    """Generate a clean PDF combining all content"""
    print("Starting simple PDF generation process...")
    
    # Get the current script's directory
    current_dir = Path(__file__).parent.absolute()
    print(f"Working directory: {current_dir}")
    
    # Define paths
    thesis_path = current_dir / 'nifty500_trading_system_thesis.md'
    indicators_path = current_dir / 'indicator_explanations.md'
    candlestick_path = current_dir / 'candlestick_pattern_details.md'
    output_path = current_dir / 'nifty500_trading_system_merged.md'
    output_pdf = current_dir / 'Nifty500_Trading_System_Complete.pdf'
    temp_html = current_dir / 'thesis_temp.html'
    
    # Read the thesis content
    print(f"Reading thesis content from: {thesis_path}")
    with open(thesis_path, 'r', encoding='utf-8') as f:
        thesis_content = f.read()
        
    # Read the indicator explanations
    print(f"Reading indicator explanations from: {indicators_path}")
    with open(indicators_path, 'r', encoding='utf-8') as f:
        indicator_content = f.read()
        
    # Read the candlestick pattern details
    print(f"Reading candlestick pattern details from: {candlestick_path}")
    with open(candlestick_path, 'r', encoding='utf-8') as f:
        candlestick_content = f.read()
    
    # Find where to insert the indicator explanations
    # Technical Indicators section 4.1 is likely the right place
    sections = thesis_content.split("## 4. Technical Indicators")
    
    if len(sections) != 2:
        print("Warning: Could not find section '## 4. Technical Indicators' in the thesis")
        merged_content = thesis_content + "\n\n" + indicator_content + "\n\n" + candlestick_content
    else:
        head = sections[0] + "## 4. Technical Indicators"
        tail = sections[1]
        
        # Find where to insert candlestick explanations
        # Look for Pattern Recognition section
        pattern_section_match = re.search(r'### 3\.3 Pattern Recognition Strategy', thesis_content)
        
        merged_content = head + "\n\n" + indicator_content + tail
        
        # If found, insert candlestick content in appropriate place
        if pattern_section_match:
            # Split at the pattern recognition section
            sections = merged_content.split('### 3.3 Pattern Recognition Strategy')
            if len(sections) == 2:
                # Add candlestick section after pattern recognition intro
                pattern_head = sections[0] + '### 3.3 Pattern Recognition Strategy'
                pattern_tail = sections[1]
                
                # Look for subsections to place the candlestick content appropriately
                subsection_match = re.search(r'#### 3\.3\.1', pattern_tail)
                if subsection_match:
                    # If subsection exists, insert our content as 3.3.3
                    subsection_parts = pattern_tail.split('#### 3.3.2')
                    if len(subsection_parts) == 2:
                        subsection_head = subsection_parts[0] + '#### 3.3.2'
                        subsection_tail = subsection_parts[1]
                        
                        # Format candlestick content with proper heading
                        formatted_candlestick = "\n\n#### 3.3.3 Candlestick Pattern Integration\n\n" + candlestick_content.split('# ')[1] if '# ' in candlestick_content else candlestick_content
                        
                        pattern_tail = subsection_head + formatted_candlestick + subsection_tail
                else:
                    # If no subsections, just append
                    formatted_candlestick = "\n\n#### 3.3.1 Candlestick Pattern Integration\n\n" + candlestick_content.split('# ')[1] if '# ' in candlestick_content else candlestick_content
                    pattern_tail = formatted_candlestick + pattern_tail
                    
                merged_content = pattern_head + pattern_tail
    
    # Write the merged content to a temporary file
    print(f"Writing merged content to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    
    # Fix image references
    print("Processing markdown and fixing image references...")
    with open(output_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
        
    # Replace relative image paths with absolute paths
    image_dir = current_dir / 'images'
    fixed_content = md_content.replace('](./images/', f']({image_dir}/')
    
    # Convert markdown to HTML
    print("Converting markdown to HTML...")
    html_content = markdown.markdown(fixed_content, extensions=['tables', 'fenced_code', 'codehilite'])
    
    # Add some basic styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Nifty 500 Trading System</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                line-height: 1.6;
                margin: 40px;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3, h4 {{ color: #333366; }}
            code {{ 
                background: #f4f4f4; 
                padding: 2px 5px;
                border-radius: 3px;
            }}
            pre {{ 
                background: #f4f4f4; 
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            img {{ max-width: 100%; }}
            table {{ 
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{ 
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Write HTML to a temporary file
    print(f"Writing HTML to temporary file: {temp_html}")
    with open(temp_html, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    # Convert HTML to PDF using pdfkit
    print(f"Converting HTML to PDF: {output_pdf}")
    try:
        # Configure pdfkit options
        options = {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'encoding': 'UTF-8',
            'no-outline': None,
            'enable-local-file-access': None,
        }
        
        # Generate PDF
        pdfkit.from_file(str(temp_html), str(output_pdf), options=options)
        print(f"PDF generation complete: {output_pdf}")
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("Trying alternative approach...")
        
        try:
            # Try using wkhtmltopdf directly if installed
            import subprocess
            cmd = [
                'wkhtmltopdf',
                '--page-size', 'A4',
                '--margin-top', '20mm',
                '--margin-right', '20mm',
                '--margin-bottom', '20mm',
                '--margin-left', '20mm',
                '--encoding', 'UTF-8',
                '--enable-local-file-access',
                str(temp_html),
                str(output_pdf)
            ]
            subprocess.run(cmd, check=True)
            print(f"PDF generation complete (alternative method): {output_pdf}")
        except Exception as e2:
            print(f"Alternative approach failed: {e2}")
            print("Please install wkhtmltopdf if not already installed")
    
    return True

if __name__ == "__main__":
    generate_pdf()
