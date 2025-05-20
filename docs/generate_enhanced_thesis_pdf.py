#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/generate_enhanced_thesis_pdf.py

"""
Generate an enhanced PDF version of the Nifty 500 Trading System thesis,
including all pattern and indicator visualizations.
"""

import os
import subprocess
import shutil
import re
from pathlib import Path

# Set up paths
docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
output_dir = docs_dir / 'output'
output_dir.mkdir(exist_ok=True)

# Source files
master_doc = docs_dir / 'nifty500_trading_system_master_document.md'
pattern_guide = docs_dir / 'comprehensive_pattern_guide_enhanced.md'
indicator_guide = docs_dir / 'indicator_explanations_enhanced.md'

# Output file
output_md = output_dir / 'nifty500_trading_system_complete.md'
output_pdf = docs_dir / 'Nifty500_Trading_System_Complete_Enhanced.pdf'

# Create combined markdown file
def create_combined_markdown():
    """Combine all documentation into a single markdown file."""
    print("Creating combined markdown file...")
    
    with open(output_md, 'w', encoding='utf-8') as outfile:
        # Start with the master document
        with open(master_doc, 'r', encoding='utf-8') as f:
            master_content = f.read()
        
        # Replace references to other docs with actual content
        master_content = re.sub(r'\[Comprehensive Pattern Guide\]\(comprehensive_pattern_guide_enhanced\.md\)', 
                               "## Comprehensive Pattern Guide", master_content)
        master_content = re.sub(r'\[Technical Indicator Explanations\]\(indicator_explanations_enhanced\.md\)', 
                               "## Technical Indicator Explanations", master_content)
        
        # Write the master content
        outfile.write(master_content)
        outfile.write("\n\n")
        
        # Append pattern guide (omitting front matter)
        with open(pattern_guide, 'r', encoding='utf-8') as f:
            pattern_content = f.read()
            pattern_content = re.sub(r'^<!-- filepath.*?-->\s*?# Comprehensive.*?##', "##", pattern_content, flags=re.DOTALL)
            outfile.write("\n\n")
            outfile.write("# Comprehensive Pattern Guide\n")
            outfile.write(pattern_content)
        
        # Append indicator guide (omitting front matter)
        with open(indicator_guide, 'r', encoding='utf-8') as f:
            indicator_content = f.read()
            indicator_content = re.sub(r'^<!-- filepath.*?-->\s*?# Technical.*?##', "##", indicator_content, flags=re.DOTALL)
            outfile.write("\n\n")
            outfile.write("# Technical Indicator Explanations\n")
            outfile.write(indicator_content)
    
    # Process the combined file to fix image paths
    fix_image_paths(output_md)
    
    print(f"Combined markdown file created: {output_md}")

def fix_image_paths(file_path):
    """Fix image paths to use absolute paths for Pandoc."""
    print("Fixing image paths...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix image paths - replace relative paths with absolute paths
    content = content.replace('](images/', f']({docs_dir}/images/')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def generate_pdf():
    """Generate PDF from the combined markdown file using Pandoc."""
    print("Generating PDF...")
    
    # Command to generate PDF with Pandoc
    cmd = [
        'pandoc',
        str(output_md),
        '--pdf-engine=xelatex',
        '-o', str(output_pdf),
        '--toc',
        '--toc-depth=3',
        '--highlight-style=tango',
        '-V', 'geometry:margin=1in',
        '-V', 'fontsize=11pt',
        '-V', 'links-as-notes=true',
        '-V', 'colorlinks=true',
        '--standalone'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"PDF successfully generated: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating PDF: {e}")
    except FileNotFoundError:
        print("Error: Pandoc is not installed or not found in PATH")
        print("Attempting alternative method...")
        try_alternative_pdf_generation()

def try_alternative_pdf_generation():
    """Try an alternative method to generate PDF if Pandoc fails."""
    try:
        # Try using markdown-pdf if available
        cmd = ['markdown-pdf', str(output_md), '-o', str(output_pdf)]
        subprocess.run(cmd, check=True)
        print(f"PDF generated using markdown-pdf: {output_pdf}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Alternative PDF generation method failed")
        print("Installing required package and trying again...")
        
        try:
            # Try installing a Python package for PDF generation
            subprocess.run(['pip', 'install', 'weasyprint', 'markdown'], check=True)
            
            # Create a temporary HTML file
            html_path = docs_dir / 'thesis_temp.html'
            
            # Convert Markdown to HTML
            with open(output_md, 'r', encoding='utf-8') as md_file:
                import markdown
                html_content = markdown.markdown(
                    md_file.read(),
                    extensions=['tables', 'fenced_code', 'toc']
                )
            
            # Add CSS styling
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Nifty 500 Trading System</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 2em; }}
                    h1, h2, h3 {{ color: #333366; }}
                    code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; }}
                    .toc {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="toc">
                    <h2>Table of Contents</h2>
                    <!-- TOC will be generated by JavaScript -->
                </div>
                {html_content}
            </body>
            </html>
            """
            
            with open(html_path, 'w', encoding='utf-8') as html_file:
                html_file.write(html_content)
            
            # Convert HTML to PDF
            from weasyprint import HTML
            HTML(str(html_path)).write_pdf(str(output_pdf))
            
            print(f"PDF generated using WeasyPrint: {output_pdf}")
            
        except Exception as e:
            print(f"Error in alternative PDF generation: {e}")
            print("PDF generation failed. Please install Pandoc or WeasyPrint manually.")

def main():
    """Main function to generate the enhanced thesis PDF."""
    print("Starting enhanced thesis PDF generation...")
    
    # Create combined markdown
    create_combined_markdown()
    
    # Generate PDF
    generate_pdf()
    
    print("Enhanced thesis PDF generation completed!")

if __name__ == "__main__":
    main()
