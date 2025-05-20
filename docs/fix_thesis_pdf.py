#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/fix_thesis_pdf.py

import os
import re
import subprocess
from pathlib import Path

def generate_thesis_pdf_with_images():
    """
    Generate a thesis PDF with proper image inclusion using a custom approach
    that avoids the limitations of the markdown module.
    """
    # Get the current directory
    current_dir = Path(__file__).parent
    image_dir = current_dir / 'images'
    
    # Check if images exist
    if not image_dir.exists() or not any(image_dir.glob('*.png')):
        print(f"Warning: No images found in {image_dir}")
    else:
        print(f"Found images directory: {image_dir}")
        image_files = list(image_dir.glob('*.png'))
        print(f"Number of images found: {len(image_files)}")
    
    # Read the markdown file
    md_path = current_dir / 'nifty500_trading_system_thesis.md'
    print(f"Reading markdown from: {md_path}")
    
    with open(md_path, 'r') as f:
        md_content = f.read()
    
    # Create a temporary HTML file with proper image references
    html_path = current_dir / 'thesis_temp.html'
    
    # Replace markdown image references with HTML img tags using absolute paths
    def replace_image_refs(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        
        # Extract the image filename from the path
        img_filename = os.path.basename(img_path)
        abs_img_path = image_dir / img_filename
        
        if abs_img_path.exists():
            return f'<div class="image-container"><img src="{abs_img_path}" alt="{alt_text}"><div class="image-caption">{alt_text}</div></div>'
        else:
            print(f"Warning: Image not found: {abs_img_path}")
            return f'<div class="image-container">[Missing Image: {alt_text}]</div>'
    
    # Replace image references
    content_with_img_tags = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image_refs, md_content)
    
    # Convert other markdown elements using pandoc
    with open(current_dir / 'temp_thesis.md', 'w') as f:
        f.write(content_with_img_tags)

    # Create HTML template with proper styling
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Nifty 500 Trading System: A Comprehensive Analysis</title>
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
        h4 { 
            font-size: 12pt; 
            color: #000; 
            margin-top: 20px;
            font-style: italic;
        }
        p { 
            text-indent: 0;
            margin-bottom: 12px;
        }
        code { 
            font-family: Consolas, Monaco, 'Courier New', monospace;
            background-color: #f5f5f5; 
            padding: 2px 5px; 
            border-radius: 3px; 
            font-size: 0.9em;
        }
        pre { 
            background-color: #f5f5f5; 
            padding: 15px; 
            border-radius: 5px; 
            overflow-x: auto; 
            line-height: 1.3;
            margin: 20px 0;
            border: 1px solid #ddd;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        blockquote {
            border-left: 4px solid #ccc;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
        }
        .image-container {
            text-align: center;
            margin: 30px 0;
        }
        .image-caption {
            font-style: italic;
            text-align: center;
            margin-top: 8px;
            font-size: 0.9em;
            color: #555;
        }
    </style>
</head>
<body>
    {{CONTENT}}
</body>
</html>
    """
    
    # Use pandoc to convert markdown to HTML
    try:
        print("Converting markdown to HTML using pandoc...")
        result = subprocess.run(
            ["pandoc", str(current_dir / "temp_thesis.md"), "-o", str(current_dir / "temp_content.html")],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Read the converted HTML content
        with open(current_dir / "temp_content.html", "r") as f:
            html_content = f.read()
        
        # Insert content into the template
        full_html = html_template.replace("{{CONTENT}}", html_content)
        
        # Write the full HTML to a file
        with open(html_path, "w") as f:
            f.write(full_html)
        
        print(f"HTML file created: {html_path}")
        
        # Generate PDF from HTML
        print("Generating PDF...")
        options = {
            'page-size': 'A4',
            'margin-top': '25mm',
            'margin-right': '25mm',
            'margin-bottom': '25mm',
            'margin-left': '25mm',
            'encoding': 'UTF-8',
            'enable-local-file-access': '',
            'footer-right': '[page] of [topage]',
            'footer-font-size': '9',
            'header-left': 'Nifty 500 Trading System',
            'header-line': '',
            'header-spacing': '5',
        }
        
        # Build wkhtmltopdf command
        cmd = ["wkhtmltopdf"]
        for key, value in options.items():
            if value:
                cmd.extend([f"--{key}", value])
            else:
                cmd.append(f"--{key}")
        
        cmd.extend([str(html_path), str(current_dir / "Nifty500_Trading_System_Thesis_with_Images.pdf")])
        
        # Execute the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("PDF created successfully: Nifty500_Trading_System_Thesis_with_Images.pdf")
        
        # Clean up temporary files
        for temp_file in ["temp_thesis.md", "temp_content.html", "thesis_temp.html"]:
            try:
                (current_dir / temp_file).unlink()
            except:
                pass
        
    except subprocess.CalledProcessError as e:
        print(f"Error running pandoc or wkhtmltopdf: {e}")
        print(f"Standard output: {e.stdout}")
        print(f"Standard error: {e.stderr}")
        
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    generate_thesis_pdf_with_images()
