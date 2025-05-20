#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/generate_thesis_pdf.py
"""
Advanced PDF generation script for the Nifty 500 Trading System thesis document.
This script handles image embedding properly and ensures high-quality academic formatting.
"""

import os
import sys
import re
import markdown
from pathlib import Path
import time
import shutil

# Try to import PDF generation libraries in order of preference
try:
    # First option: WeasyPrint (best image handling)
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    PDF_ENGINE = "weasyprint"
    print("Using WeasyPrint for PDF generation")
except ImportError:
    try:
        # Second option: pdfkit (wkhtmltopdf)
        import pdfkit
        PDF_ENGINE = "pdfkit"
        print("Using pdfkit (wkhtmltopdf) for PDF generation")
    except ImportError:
        try:
            # Third option: markdown-pdf
            from markdown_pdf import MarkdownPdf
            PDF_ENGINE = "markdown-pdf"
            print("Using markdown-pdf for PDF generation")
        except ImportError:
            print("ERROR: No PDF generation library is available.")
            print("Please install one of: WeasyPrint, pdfkit, or markdown-pdf")
            print("Example: pip install weasyprint")
            sys.exit(1)

def verify_images():
    """Verify all required images exist and have proper content"""
    current_dir = Path(__file__).parent
    image_dir = current_dir / 'images'
    
    # List of required images based on the markdown file
    thesis_path = current_dir / 'nifty500_trading_system_thesis.md'
    with open(thesis_path, 'r') as f:
        content = f.read()
    
    # Extract image references using regex
    image_refs = re.findall(r'!\[(.*?)\]\((.*?)\)', content)
    required_images = []
    
    for _, img_path in image_refs:
        if img_path.startswith('./images/'):
            img_name = img_path.split('/')[-1]
            required_images.append(img_name)
    
    # Check if all required images exist
    missing_images = []
    for img_name in required_images:
        img_path = image_dir / img_name
        if not img_path.exists():
            missing_images.append(img_name)
    
    if missing_images:
        print(f"WARNING: {len(missing_images)} required images are missing:")
        for img_name in missing_images:
            print(f"  - {img_name}")            # Try to regenerate missing images
            print("\nAttempting to regenerate missing images...")
            try:
                # Try to use the separate create_all_images.py script
                import subprocess
                result = subprocess.run(
                    ["python3", str(Path(__file__).parent / "create_all_images.py")],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print("Images regenerated successfully")
                else:
                    print(f"Failed to regenerate images: {result.stderr}")
                    print("Please run create_all_images.py manually")
                    return False
            except Exception as e:
                print(f"Failed to regenerate images: {e}")
                print("Please run create_all_images.py manually")
                return False
    
    # Verify image sizes
    small_images = []
    for img_name in os.listdir(image_dir):
        if img_name.endswith('.png'):
            img_path = image_dir / img_name
            size_kb = os.path.getsize(img_path) / 1024
            if size_kb < 5:  # Less than 5KB might indicate an empty or corrupted image
                small_images.append((img_name, size_kb))
    
    if small_images:
        print("\nWARNING: Some images are suspiciously small in size:")
        for img_name, size_kb in small_images:
            print(f"  - {img_name}: {size_kb:.2f} KB")
    
    return True

def fix_image_paths(md_content):
    """Fix image paths to use absolute paths"""
    current_dir = Path(__file__).parent
    image_dir = current_dir / 'images'
    
    # Replace relative image paths with absolute paths
    def img_path_replacer(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        
        if img_path.startswith('./images/'):
            img_name = img_path.split('/')[-1]
            abs_path = str(image_dir / img_name)
            return f'![{alt_text}]({abs_path})'
        return match.group(0)
    
    return re.sub(r'!\[(.*?)\]\((.*?)\)', img_path_replacer, md_content)

def create_pdf_with_weasyprint(md_content, output_path):
    """Generate PDF using WeasyPrint - best for image handling"""
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content, 
        extensions=['tables', 'fenced_code', 'codehilite', 'toc']
    )
    
    # Add academic styling with CSS
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Nifty 500 Trading System: A Comprehensive Analysis</title>
        <style>
            @page {{
                size: A4;
                margin: 2.5cm 2.5cm 2.5cm 2.5cm;
                @top-left {{
                    content: "Nifty 500 Trading System";
                    font-size: 9pt;
                }}
                @top-right {{
                    content: "";
                }}
                @bottom-center {{
                    content: counter(page) " of " counter(pages);
                    font-size: 9pt;
                }}
            }}
            body {{ 
                font-family: 'Times New Roman', Times, serif; 
                font-size: 12pt;
                line-height: 1.5;
                text-align: justify;
            }}
            h1 {{ 
                font-size: 24pt; 
                text-align: center;
                margin-bottom: 30px;
            }}
            h2 {{ 
                font-size: 18pt; 
                margin-top: 40px; 
                border-bottom: 1px solid #ccc;
                padding-bottom: 5px;
            }}
            h3 {{ font-size: 14pt; margin-top: 25px; }}
            h4 {{ font-size: 12pt; margin-top: 20px; font-style: italic; }}
            p {{ text-indent: 0; margin-bottom: 12px; }}
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
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px 12px; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            img {{ 
                max-width: 100%; 
                height: auto; 
                display: block; 
                margin: 20px auto; 
            }}
            figcaption {{
                text-align: center;
                font-style: italic;
                font-size: 0.9em;
                margin-top: 8px;
            }}
            blockquote {{ border-left: 4px solid #ccc; padding-left: 20px; margin-left: 0; color: #555; }}
            a {{ color: #0066cc; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .references p {{ padding-left: 30px; text-indent: -30px; }}
            
            /* image container for proper display */
            .figure {{ 
                margin: 20px auto;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    font_config = FontConfiguration()
    HTML(string=html_with_style).write_pdf(
        output_path,
        stylesheets=[],
        font_config=font_config
    )

def create_pdf_with_pdfkit(md_content, output_path):
    """Generate PDF using pdfkit (wkhtmltopdf)"""
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content, 
        extensions=['tables', 'fenced_code', 'codehilite', 'toc']
    )
    
    # Add academic styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Nifty 500 Trading System: A Comprehensive Analysis</title>
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
            h3 {{ font-size: 14pt; color: #000; margin-top: 25px; }}
            h4 {{ font-size: 12pt; color: #000; margin-top: 20px; font-style: italic; }}
            p {{ text-indent: 0; margin-bottom: 12px; }}
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
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px 12px; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            img {{ 
                max-width: 100%; 
                height: auto; 
                display: block; 
                margin: 20px auto; 
            }}
            blockquote {{ border-left: 4px solid #ccc; padding-left: 20px; margin-left: 0; color: #555; }}
            .caption {{ text-align: center; font-style: italic; margin-top: 5px; font-size: 0.9em; }}
            a {{ color: #0066cc; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .references p {{ padding-left: 30px; text-indent: -30px; }}
            .image-container {{ text-align: center; margin: 20px 0; }}
            .image-caption {{ font-style: italic; text-align: center; margin-top: 8px; font-size: 0.9em; }}
            
            /* Fix for image display */
            figure {{ margin: 20px auto; text-align: center; }}
            figcaption {{ font-style: italic; font-size: 0.9em; margin-top: 8px; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
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
        'dpi': 300,  # Higher DPI for better quality
        'image-quality': 100  # Maximum image quality
    }
    
    # Convert HTML to PDF
    pdfkit.from_string(styled_html, output_path, options=options)

def create_pdf_with_markdown_pdf(md_content, output_path):
    """Generate PDF using markdown-pdf"""
    # Create a temporary fixed markdown file
    current_dir = Path(__file__).parent
    temp_md_path = current_dir / 'temp_thesis_with_fixed_images.md'
    
    with open(temp_md_path, 'w') as f:
        f.write(md_content)
    
    try:
        # Convert to PDF
        md_pdf = MarkdownPdf()
        md_pdf.convert(str(temp_md_path), output_path)
    finally:
        # Clean up temporary file
        if temp_md_path.exists():
            os.remove(temp_md_path)

def create_pdf():
    """Main function to create the PDF"""
    start_time = time.time()
    
    current_dir = Path(__file__).parent
    thesis_path = current_dir / 'nifty500_trading_system_thesis.md'
    output_pdf = current_dir / 'Nifty500_Trading_System_Thesis.pdf'
    
    print(f"Starting PDF generation process from: {thesis_path}")
    
    # Step 1: Verify that all images exist and have proper content
    print("\nStep 1: Verifying images...")
    if not verify_images():
        print("ERROR: Image verification failed. Please fix the missing images.")
        sys.exit(1)
    
    # Step 2: Read and fix the markdown content
    print("\nStep 2: Reading and processing markdown content...")
    with open(thesis_path, 'r') as f:
        md_content = f.read()
    
    # Fix image paths to use absolute paths
    md_content = fix_image_paths(md_content)
    
    # Step 3: Generate the PDF using the available engine
    print(f"\nStep 3: Generating PDF using {PDF_ENGINE}...")
    try:
        if PDF_ENGINE == "weasyprint":
            create_pdf_with_weasyprint(md_content, str(output_pdf))
        elif PDF_ENGINE == "pdfkit":
            create_pdf_with_pdfkit(md_content, str(output_pdf))
        elif PDF_ENGINE == "markdown-pdf":
            create_pdf_with_markdown_pdf(md_content, str(output_pdf))
    except Exception as e:
        print(f"ERROR: Failed to generate PDF: {e}")
        sys.exit(1)
    
    # Step 4: Verify the PDF was created successfully
    if not output_pdf.exists():
        print("ERROR: The PDF file was not created")
        sys.exit(1)
    
    pdf_size_kb = os.path.getsize(output_pdf) / 1024
    print(f"\nPDF file created successfully: {output_pdf}")
    print(f"PDF file size: {pdf_size_kb:.2f} KB")
    
    if pdf_size_kb < 100:
        print("WARNING: The PDF file size is smaller than expected for a document with images")
    
    # Step 5: Create a backup copy
    backup_pdf = current_dir / f'Nifty500_Thesis_Backup_{int(time.time())}.pdf'
    shutil.copy2(output_pdf, backup_pdf)
    print(f"Backup copy created: {backup_pdf}")
    
    end_time = time.time()
    print(f"\nPDF generation completed in {end_time - start_time:.2f} seconds")
    print("\nNow you can open the PDF to verify all content and images are displayed correctly")
    
    return True

if __name__ == "__main__":
    try:
        success = create_pdf()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
