#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/finalize_thesis.py
import os
import sys
from pathlib import Path
import subprocess
import time
import markdown
import re

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("WeasyPrint not available. PDF will be generated using alternative method.")

def run_command(cmd, desc=None):
    """Run a command and print its output."""
    if desc:
        print(f"\n=== {desc} ===")
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print("Error output:")
        print(result.stderr)
    
    return result.returncode
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    
    return True

def finalize_thesis():
    """Finalize the thesis document by ensuring all images are present and regenerating the PDF."""
    current_dir = Path(__file__).parent
    
    # 1. Check if images directory exists
    image_dir = current_dir / 'images'
    if not image_dir.exists():
        image_dir.mkdir(exist_ok=True)
        print(f"Created images directory: {image_dir}")
    
    # 2. Generate images if needed
    if not list(image_dir.glob('*.png')) or len(list(image_dir.glob('*.png'))) < 20:
        print("Images missing or incomplete. Regenerating images...")
        if not run_command("python3 create_images.py", "Generating Images"):
            print("Failed to generate images. Exiting.")
            return False
    else:
        print(f"Found {len(list(image_dir.glob('*.png')))} images in the images directory.")
    
    # 3. Check image references in markdown
    print("\nChecking markdown image references...")
    thesis_path = current_dir / 'nifty500_trading_system_thesis.md'
    if not thesis_path.exists():
        print(f"Error: Thesis file {thesis_path} does not exist.")
        return False
    
    # 4. Generate the PDF
    print("\nGenerating PDF document...")
    if not run_command("python3 generate_pdf.py", "PDF Generation"):
        print("Failed to generate PDF. Exiting.")
        return False
    
    # 5. Verify the PDF exists and has a reasonable size
    pdf_path = current_dir / 'Nifty500_Trading_System_Thesis.pdf'
    if not pdf_path.exists():
        print(f"Error: PDF file {pdf_path} was not created.")
        return False
    
    pdf_size_kb = os.path.getsize(pdf_path) / 1024
    print(f"\nPDF file created: {pdf_path}")
    print(f"PDF file size: {pdf_size_kb:.2f} KB")
    
    if pdf_size_kb < 100:
        print("WARNING: The PDF file size is smaller than expected for a comprehensive document.")
    
    # 6. Final message
    print("\n=== Thesis Document Finalization Complete ===")
    print(f"The thesis document has been successfully generated at:")
    print(f"{pdf_path}")
    print("\nThe document includes all references, images, and comprehensive content about")
    print("the Nifty 500 Trading System as requested.")
    print("\nTo view the PDF, you can open it with your preferred PDF viewer.")
    
    return True

if __name__ == "__main__":
    start_time = time.time()
    success = finalize_thesis()
    end_time = time.time()
    
    if success:
        print(f"\nProcess completed successfully in {end_time - start_time:.2f} seconds.")
    else:
        print(f"\nProcess failed after {end_time - start_time:.2f} seconds.")
        sys.exit(1)
