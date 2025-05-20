#!/usr/bin/env python3
"""Clean up duplicate and temporary files in the docs directory"""

import os
import re
from pathlib import Path

# Files to keep (one per category)
KEEP_FILES = {
    'generate': 'generate_complete_thesis.py',  # Main PDF generation script
    'create': 'create_pdf_with_images.py',      # Main image handling script
    'merge': 'merge_indicator_explanations.py', # Main content merging script
    'fix': None,                               # Remove all fix_* files
    'thesis_temp': None,                        # Remove all temp files
    'validate': None                            # Remove all validation scripts
}

def main():
    """Clean up redundant files in the docs directory"""
    docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
    
    # Group files by their prefix
    file_groups = {}
    for file_path in docs_dir.glob('*.py'):
        file_name = file_path.name
        prefix = re.match(r'^([a-z]+)_', file_name)
        if prefix:
            prefix = prefix.group(1)
            if prefix not in file_groups:
                file_groups[prefix] = []
            file_groups[prefix].append(file_path)
    
    # Delete duplicate files, keeping only the ones in KEEP_FILES
    for prefix, files in file_groups.items():
        if prefix in KEEP_FILES:
            keep_file = KEEP_FILES[prefix]
            if keep_file is None:  # Remove all files in this category
                for file_path in files:
                    print(f"Removing {file_path}")
                    os.remove(file_path)
            else:
                for file_path in files:
                    if file_path.name != keep_file:
                        print(f"Removing {file_path}")
                        os.remove(file_path)
    
    print("Cleanup complete")

if __name__ == '__main__':
    main()
