#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/verify_cross_references.py

"""
Verify cross-references in documentation files.
This script checks that links between documentation files are valid
and that all referenced files and images exist.
"""

import os
import re
import sys
from pathlib import Path

# Define paths
docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
images_dir = docs_dir / 'images'
patterns_dir = images_dir / 'patterns'

# Main documentation files
main_docs = [
    'index.md',
    'nifty500_trading_system_master_document.md',
    'comprehensive_pattern_guide_enhanced.md',
    'indicator_explanations_enhanced.md'
]

# Regex patterns
link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

def verify_cross_references():
    """Verify all cross-references in documentation files."""
    print("Verifying cross-references in documentation files...")
    
    # Collect all available files and images
    available_files = set(f.name for f in docs_dir.glob('*.md'))
    available_images = set(f"images/{f.name}" for f in images_dir.glob('*.png'))
    available_images.update(set(f"images/patterns/{f.name}" for f in patterns_dir.glob('*.png')))
    
    all_errors = []
    all_warnings = []
    
    # Check each main documentation file
    for doc_file in main_docs:
        doc_path = docs_dir / doc_file
        if not doc_path.exists():
            all_errors.append(f"❌ Documentation file not found: {doc_file}")
            continue
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check links to other documents
        for match in link_pattern.finditer(content):
            link_text, url = match.groups()
            
            # Skip external links and anchors
            if url.startswith(('http:', 'https:', '#')):
                continue
            
            # Check if the linked file exists
            if url.endswith('.md') and url not in available_files:
                all_errors.append(f"❌ In {doc_file}: Link to non-existent file: {url}")
            elif url.endswith('.pdf') and not (docs_dir / url).exists():
                all_errors.append(f"❌ In {doc_file}: Link to non-existent PDF: {url}")
        
        # Check image references
        for match in image_pattern.finditer(content):
            alt_text, image_path = match.groups()
            
            # Skip external images
            if image_path.startswith(('http:', 'https:')):
                continue
            
            # Normalize image path if needed
            if not image_path.startswith('images/'):
                image_path = f"images/{image_path}"
            
            # Check if the image exists
            if image_path not in available_images and not (docs_dir / image_path).exists():
                all_errors.append(f"❌ In {doc_file}: Reference to non-existent image: {image_path}")
        
        print(f"✅ Checked {doc_file}")
    
    # Print summary
    print("\n=== Cross-Reference Verification Summary ===")
    
    if all_errors:
        print("\nErrors:")
        for error in all_errors:
            print(f"  {error}")
    
    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"  {warning}")
    
    if not all_errors and not all_warnings:
        print("\n✅ All cross-references are valid!")
    
    return len(all_errors) == 0

if __name__ == "__main__":
    success = verify_cross_references()
    sys.exit(0 if success else 1)
