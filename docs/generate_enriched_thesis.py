#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/generate_enriched_thesis.py

import markdown
import pdfkit
import os
import re
from pathlib import Path
import sys

def generate_enriched_thesis_pdf():
    """
    Generate an enriched thesis PDF with detailed indicator explanations
    """
    print("Starting PDF generation process with enriched content...")
    
    # Get the current script's directory
    current_dir = Path(__file__).parent.absolute()
    print(f"Working directory: {current_dir}")
    
    # Define paths
    thesis_path = current_dir / 'nifty500_trading_system_thesis.md'
    indicators_path = current_dir / 'indicator_explanations.md'
    output_pdf = current_dir / 'Nifty500_Trading_System_Enriched.pdf'
    image_dir = current_dir / 'images'
    
    # Verify required files exist
    if not thesis_path.exists():
        print(f"Error: Thesis file not found at {thesis_path}")
        return False
    
    if not indicators_path.exists():
        print(f"Error: Indicator explanations file not found at {indicators_path}")
        return False
    
    # Read the thesis content
    print(f"Reading thesis content from: {thesis_path}")
    try:
        with open(thesis_path, 'r', encoding='utf-8') as f:
            thesis_content = f.read()
    except Exception as e:
        print(f"Error reading thesis file: {e}")
        return False
    
    # Read the indicator explanations
    print(f"Reading indicator explanations from: {indicators_path}")
    try:
        with open(indicators_path, 'r', encoding='utf-8') as f:
            indicator_content = f.read()
    except Exception as e:
        print(f"Error reading indicator explanations file: {e}")
        return False
    
    # Locate the Technical Indicators section in the thesis
    technical_section_pattern = r'(### 4\.1 Indicator Categories.*?)(### 4\.2 Custom Composite Indicators)'
    match = re.search(technical_section_pattern, thesis_content, re.DOTALL)
    
    if not match:
        print("Could not find Technical Indicators section in thesis")
        return False
    
    # Extract the section
    original_section = match.group(1)
    
    # Extract the content from the indicator explanations that we want to insert
    # We'll take the detailed explanations but skip the header
    explanation_section_pattern = r'## Detailed Technical Indicator Explanations\s*(.*)'
    explanation_match = re.search(explanation_section_pattern, indicator_content, re.DOTALL)
    
    if not explanation_match:
        print("Could not find explanation content")
        return False
    
    explanation_content = explanation_match.group(1).strip()
    
    # Create the new section
    new_section = original_section + "\n\n### 4.1.5 Detailed Indicator Explanations\n\n" + explanation_content + "\n\n"
    
    # Replace the old section with the new section
    enriched_thesis = thesis_content.replace(original_section, new_section)
    
    # Process the markdown to fix image references
    print("Processing markdown and fixing image references...")
    
    # Replace relative image paths with absolute paths
    def img_path_replacer(match):
        alt_text = match.group(1)
        rel_path = match.group(2)
        
        if rel_path.startswith('./images/'):
            img_name = os.path.basename(rel_path)
            abs_img_path = str(image_dir / img_name)
            
            # Verify the image exists
            if os.path.exists(abs_img_path):
                print(f"Found image: {img_name}")
                return f'![{alt_text}]({abs_img_path})'
            else:
                print(f"Warning: Image not found: {img_name}")
                return match.group(0)  # Return unchanged if image not found
        
        return match.group(0)  # Return unchanged for non-local images
    
    # Fix image paths in markdown
    enriched_thesis = re.sub(r'!\[(.*?)\]\((.*?)\)', img_path_replacer, enriched_thesis)
    
    # Write the enriched thesis to a temporary file
    temp_md_path = current_dir / 'enriched_thesis_temp.md'
    with open(temp_md_path, 'w', encoding='utf-8') as f:
        f.write(enriched_thesis)
    print(f"Saved enriched thesis to: {temp_md_path}")
    
    # Convert markdown to HTML
    print("Converting markdown to HTML...")
    html_content = markdown.markdown(
        enriched_thesis, 
        extensions=[
            'tables', 
            'fenced_code', 
            'codehilite', 
            'toc',
            'nl2br'  # Convert newlines to <br>
        ]
    )
    
    # Add academic styling with improved image handling
    styled_html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Nifty 500 Trading System: A Comprehensive Analysis</title>
        <style>
            @page {{
                size: A4;
                margin: 2.5cm;
            }}
            body {{ 
                font-family: 'Times New Roman', Times, serif; 
                line-height: 1.6;
                text-align: justify;
                font-size: 11pt;
            }}
            h1 {{ 
                font-size: 18pt; 
                text-align: center;
                margin-top: 2cm;
                margin-bottom: 1.5cm;
            }}
            h2 {{ 
                font-size: 14pt; 
                margin-top: 20px; 
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            h3 {{ 
                font-size: 12pt;
                margin-top: 15px;
            }}
            h4 {{ 
                font-size: 11pt;
                font-style: italic;
                margin-top: 10px;
            }}
            p {{ 
                margin-bottom: 10px;
            }}
            code {{ 
                font-family: Consolas, Monaco, 'Courier New', monospace;
                background-color: #f5f5f5; 
                padding: 2px 5px; 
                border-radius: 3px; 
                font-size: 0.9em;
            }}
            pre {{ 
                background-color: #f8f8f8; 
                padding: 10px; 
                border-radius: 5px; 
                overflow-x: auto;
                white-space: pre-wrap;
                margin: 15px 0;
                border: 1px solid #eee;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            blockquote {{
                border-left: 4px solid #ddd;
                padding-left: 15px;
                margin-left: 0;
                color: #555;
            }}
            
            /* Special styling for images */
            img {{
                max-width: 100%;
                height: auto !important;
                display: block;
                margin: 20px auto;
                page-break-inside: avoid;
            }}
            
            /* Image figure style */
            .figure {{
                text-align: center;
                margin: 25px 0;
                page-break-inside: avoid;
            }}
            .figure img {{
                border: 1px solid #eee;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                max-width: 90%;
            }}
            .figure-caption {{
                text-align: center;
                font-style: italic;
                font-size: 0.9em;
                margin-top: 8px;
            }}
            
            /* Table of contents */
            .toc {{
                background-color: #f9f9f9;
                padding: 15px;
                margin: 20px 0;
                border: 1px solid #eee;
                border-radius: 5px;
            }}
            .toc ul {{
                padding-left: 20px;
            }}
            .toc li {{
                margin-bottom: 5px;
            }}
            
            /* Links */
            a {{
                color: #0066cc;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            
            /* Page breaks */
            .page-break {{
                page-break-after: always;
            }}
            
            /* References section */
            .references {{
                margin-top: 30px;
            }}
            .references p {{
                padding-left: 25px;
                text-indent: -25px;
                margin-bottom: 8px;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    '''
    
    # Improve image references in the HTML directly
    # This replaces standard markdown image tags with more complex figure elements
    styled_html = re.sub(
        r'<p><img alt="([^"]*)" src="([^"]*)"[^>]*></p>',
        r'<div class="figure"><img alt="\1" src="\2"><div class="figure-caption">\1</div></div>',
        styled_html
    )
    
    # Create a temporary HTML file
    temp_html_path = current_dir / 'enriched_thesis_temp.html'
    with open(temp_html_path, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    print(f"Saved temporary HTML to: {temp_html_path}")
    
    # Configure PDF options
    options = {
        'page-size': 'A4',
        'margin-top': '25mm',
        'margin-right': '25mm',
        'margin-bottom': '25mm',
        'margin-left': '25mm',
        'encoding': 'UTF-8',
        'title': 'Nifty 500 Trading System: A Comprehensive Analysis',
        'footer-right': '[page] of [topage]',
        'footer-font-size': '9',
        'header-left': 'Nifty 500 Trading System',
        'header-line': True,
        'header-spacing': '5',
        'enable-local-file-access': True,  # Important for local file access!
        'enable-smart-shrinking': True,
        'print-media-type': True,
        'no-outline': None,
        'disable-javascript': False  # Enable JavaScript for better rendering
    }
    
    # Generate the PDF
    print(f"Generating PDF at: {output_pdf}")
    try:
        pdfkit.from_file(str(temp_html_path), str(output_pdf), options=options)
        print(f"PDF generation successful!")
        print(f"PDF saved to: {output_pdf}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("PDF generation failed.")
        print("Troubleshooting tips:")
        print("1. Ensure wkhtmltopdf is installed correctly")
        print("2. Check for path issues with images")
        print("3. Examine the temporary HTML file for issues")
        return False
    
    # Verify PDF was created
    if os.path.exists(output_pdf):
        print(f"Success! PDF file created at: {output_pdf}")
        print(f"File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
        return True
    else:
        print("Error: PDF file was not created.")
        return False

if __name__ == "__main__":
    success = generate_enriched_thesis_pdf()
    sys.exit(0 if success else 1)
