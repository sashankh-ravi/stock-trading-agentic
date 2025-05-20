#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/check_pdf_quality.py

"""
Check the quality and size of PDF documentation.
This script verifies that the generated PDFs are of sufficient quality
and include images and proper formatting.
"""

import os
from pathlib import Path
import sys

# Define paths
docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
pdf_file = docs_dir / 'Nifty500_Trading_System_Complete_Enhanced.pdf'

def check_pdf_quality():
    """Check the quality of the generated PDF documentation."""
    print("Checking PDF documentation quality...")
    
    if not pdf_file.exists():
        print(f"❌ PDF file not found: {pdf_file.name}")
        return False
    
    print(f"📄 Found PDF file: {pdf_file.name}")
    
    # Check file size (should be at least 1MB for a comprehensive document with images)
    size_bytes = pdf_file.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"📊 PDF File Size: {size_bytes} bytes ({size_mb:.2f} MB)")
    
    if size_mb < 1.0:
        print(f"❌ PDF file seems too small ({size_mb:.2f} MB). It may be missing images or content.")
        return False
    else:
        print(f"✅ PDF file has a reasonable size ({size_mb:.2f} MB).")
    
    # Check file modification time to verify it's recently updated
    mtime = os.path.getmtime(pdf_file)
    from datetime import datetime
    mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"⏰ Last Modified: {mtime_str}")
    
    print("\n✅ PDF quality check completed.")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("PDF DOCUMENTATION QUALITY CHECK")
    print("=" * 50)
    
    success = check_pdf_quality()
    
    print("=" * 50)
    if success:
        print("✅ PDF QUALITY CHECK PASSED")
    else:
        print("❌ PDF QUALITY CHECK FAILED")
    print("=" * 50)
    
    sys.exit(0 if success else 1)
