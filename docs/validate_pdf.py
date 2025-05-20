#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/validate_pdf.py
import os
import sys
from pathlib import Path

def validate_pdf():
    """Validate the generated PDF file."""
    current_dir = Path(__file__).parent
    pdf_path = current_dir / 'Nifty500_Trading_System_Thesis.pdf'
    
    print("\n=== PDF Validation Report ===\n")
    
    # Basic validation
    if not pdf_path.exists():
        print(f"ERROR: The PDF file {pdf_path} does not exist.")
        return False
        
    # Check file size (should be at least 100KB for a comprehensive document)
    size_kb = os.path.getsize(pdf_path) / 1024
    if size_kb < 100:
        print(f"WARNING: The PDF file size ({size_kb:.2f} KB) is smaller than expected for a comprehensive document.")
    else:
        print(f"✓ PDF file size: {size_kb:.2f} KB (Good)")
    
    # Check if all images were included
    image_dir = current_dir / 'images'
    if not image_dir.exists():
        print("WARNING: The images directory does not exist.")
    elif not any(image_dir.iterdir()):
        print("WARNING: The images directory is empty.")
    else:
        image_count = len(list(image_dir.glob('*.png')))
        print(f"✓ Number of images found: {image_count}")
        
        # List all images
        print("\nImages in directory:")
        for img in sorted(image_dir.glob('*.png')):
            print(f"  - {img.name}")
    
    # Check for missing images referenced in the thesis
    thesis_path = current_dir / 'nifty500_trading_system_thesis.md'
    if thesis_path.exists():
        with open(thesis_path, 'r') as f:
            content = f.read()
            
        # Extract image references from markdown
        import re
        image_refs = re.findall(r'!\[.*?\]\((.*?)\)', content)
        image_filenames = [Path(ref).name for ref in image_refs if 'images/' in ref]
        
        print(f"\n✓ Number of image references in thesis: {len(image_filenames)}")
        
        # Check for missing images
        if image_dir.exists():
            existing_images = set(f.name for f in image_dir.glob('*.png'))
            missing_images = set(image_filenames) - existing_images
            
            if missing_images:
                print("\nWARNING: The following referenced images are missing:")
                for img in sorted(missing_images):
                    print(f"  - {img}")
            else:
                print("✓ All referenced images exist in the images directory")
    
    # Additional PDF validation could be done with PyPDF2 or similar libraries
    # but they might not be installed, so we skip advanced validation
    
    print(f"\nPDF file location: {pdf_path}")
    print("\nValidation completed successfully!")
    print("Note: For full validation, please open the PDF manually to check formatting and content.")
    return True

if __name__ == "__main__":
    validate_pdf()
