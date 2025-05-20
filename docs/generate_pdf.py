import markdown
import pdfkit
import os
import re
from pathlib import Path

def convert_md_to_pdf():
    try:
        # Get the current script's directory
        current_dir = Path(__file__).parent
        
        # Read the markdown file
        md_path = current_dir / 'nifty500_trading_system_thesis.md'
        print(f"Reading markdown from: {md_path}")
        with open(md_path, 'r') as f:
            md_content = f.read()
        
        # Fix image paths to use absolute paths for wkhtmltopdf
        image_dir = current_dir / 'images'
        # Replace relative image paths with absolute paths
        def img_path_replacer(match):
            img_path = match.group(2)
            if img_path.startswith('./images/'):
                img_name = img_path.split('/')[-1]
                abs_path = str(image_dir / img_name)
                return f'![{match.group(1)}]({abs_path})'
            return match.group(0)
        
        md_content = re.sub(r'!\[(.*?)\]\((.*?)\)', img_path_replacer, md_content)
        
        # Convert markdown to HTML with extensions for tables and code highlighting
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite', 'toc'])
    
    except Exception as e:
        print(f"Error reading or processing markdown file: {e}")
        return
    
    # Add some basic styling with academic formatting
    styled_html = f'''
    <html>
    <head>
        <style>
            body {{ 
                font-family: 'Times New Roman', Times, serif; 
                margin: 60px; 
                line-height: 1.5;
                text-align: justify;
            }}
            h1 {{ 
                font-size: 24pt; 
                text-align: center;
                color: #000;
                margin-bottom: 30px;
            }}
            h2 {{ 
                font-size: 18pt; 
                color: #000; 
                margin-top: 40px; 
                border-bottom: 1px solid #ccc;
                padding-bottom: 5px;
            }}
            h3 {{ 
                font-size: 14pt; 
                color: #000; 
                margin-top: 25px;
            }}
            h4 {{ 
                font-size: 12pt; 
                color: #000; 
                margin-top: 20px;
                font-style: italic;
            }}
            p {{ 
                text-indent: 0;
                margin-bottom: 12px;
            }}
            code {{ 
                font-family: Consolas, Monaco, 'Courier New', monospace;
                background-color: #f5f5f5; 
                padding: 2px 5px; 
                border-radius: 3px; 
                font-size: 0.9em;
            }}
            pre {{ 
                background-color: #f5f5f5; 
                padding: 15px; 
                border-radius: 5px; 
                overflow-x: auto; 
                line-height: 1.3;
                margin: 20px 0;
                border: 1px solid #ddd;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 12px;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 20px auto;
            }}
            blockquote {{
                border-left: 4px solid #ccc;
                padding-left: 20px;
                margin-left: 0;
                color: #555;
            }}
            .caption {{
                text-align: center;
                font-style: italic;
                margin-top: 5px;
                font-size: 0.9em;
            }}
            a {{
                color: #0066cc;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .references p {{
                padding-left: 30px;
                text-indent: -30px;
            }}
            .image-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .image-caption {{
                font-style: italic;
                text-align: center;
                margin-top: 8px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    '''
    
    # Configure PDF options
    options = {
        'page-size': 'A4',
        'margin-top': '25mm',
        'margin-right': '25mm',
        'margin-bottom': '25mm',
        'margin-left': '25mm',
        'encoding': 'UTF-8',
        'no-outline': None,
        'enable-local-file-access': True,
        'title': 'Nifty 500 Trading System: A Comprehensive Analysis',
        'footer-right': '[page] of [topage]',
        'footer-font-size': '9',
        'header-font-size': '9',
        'header-left': 'Nifty 500 Trading System',
        'header-line': True,
        'header-spacing': '5',
    }
    
    # Convert HTML to PDF
    output_file = current_dir / 'Nifty500_Trading_System_Thesis.pdf'
    try:
        pdfkit.from_string(styled_html, str(output_file), options=options)
        print(f"PDF generated successfully: {output_file}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("Check if wkhtmltopdf is installed and in your PATH")

if __name__ == "__main__":
    convert_md_to_pdf()
