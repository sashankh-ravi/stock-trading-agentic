#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/fix_images.py

import os
import re
from pathlib import Path
import pdfkit

def main():
    # Get the current directory
    current_dir = Path(__file__).parent
    image_dir = current_dir / 'images'
    
    # Read the markdown file
    md_path = current_dir / 'nifty500_trading_system_thesis.md'
    print(f"Reading markdown from: {md_path}")
    
    with open(md_path, 'r') as f:
        md_content = f.read()
    
    # Create a direct HTML file with proper image references
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nifty 500 Trading System Thesis</title>
        <style>
            body { 
                font-family: 'Times New Roman', Times, serif; 
                margin: 60px; 
                line-height: 1.5;
                text-align: justify;
            }
            h1 { 
                font-size: 24pt; 
                text-align: center;
                color: #000;
                margin-bottom: 30px;
            }
            h2 { 
                font-size: 18pt; 
                color: #000; 
                margin-top: 40px; 
                border-bottom: 1px solid #ccc;
                padding-bottom: 5px;
            }
            h3 { 
                font-size: 14pt; 
                color: #000; 
                margin-top: 25px;
            }
            img {
                max-width: 90%;
                display: block;
                margin: 20px auto;
                border: 1px solid #ddd;
                padding: 5px;
            }
            .figure-caption {
                text-align: center;
                font-style: italic;
                margin-top: 8px;
                margin-bottom: 20px;
                font-size: 0.9em;
            }
            code {
                background-color: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
    """
    
    # Process the markdown content line by line to create basic HTML
    lines = md_content.split('\n')
    for line in lines:
        # Convert headings
        if line.startswith('# '):
            html_content += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith('## '):
            html_content += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith('### '):
            html_content += f"<h3>{line[4:]}</h3>\n"
        # Convert images
        elif '![' in line and '](' in line:
            match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
            if match:
                alt_text = match.group(1)
                img_path = match.group(2)
                img_filename = os.path.basename(img_path)
                abs_img_path = str(image_dir / img_filename)
                
                html_content += f'<div><img src="{abs_img_path}" alt="{alt_text}">\n'
                html_content += f'<div class="figure-caption">{alt_text}</div></div>\n'
        # Convert code blocks (simplified)
        elif line.startswith('```'):
            html_content += '<pre><code>\n'
            # Skip this opening line and find the closing ```
        elif line.startswith('`'):
            # Inline code
            line = re.sub(r'`(.*?)`', r'<code>\1</code>', line)
            html_content += f'<p>{line}</p>\n'
        # Regular paragraph text
        elif line.strip() != '':
            html_content += f'<p>{line}</p>\n'
        # Empty line (paragraph break)
        else:
            html_content += '<br>\n'
    
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML to a temporary file
    html_path = current_dir / 'thesis_with_images.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML file created: {html_path}")
    
    # Generate PDF with wkhtmltopdf
    output_pdf = current_dir / 'Nifty500_Trading_System_Thesis_with_Images.pdf'
    
    options = {
        'page-size': 'A4',
        'margin-top': '25mm',
        'margin-right': '25mm',
        'margin-bottom': '25mm',
        'margin-left': '25mm',
        'encoding': 'UTF-8',
        'enable-local-file-access': None,
    }
    
    try:
        pdfkit.from_file(str(html_path), str(output_pdf), options=options)
        print(f"PDF generated successfully: {output_pdf}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == '__main__':
    main()
