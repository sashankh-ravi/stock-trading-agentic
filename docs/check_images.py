#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/check_images.py
import re
from pathlib import Path

def check_markdown_images():
    """Check that all images referenced in the markdown file exist."""
    current_dir = Path(__file__).parent
    thesis_path = current_dir / 'nifty500_trading_system_thesis.md'
    image_dir = current_dir / 'images'
    
    print("=== Markdown Image Reference Check ===\n")
    
    # Get all images in the images directory
    if not image_dir.exists():
        print(f"Error: Image directory {image_dir} does not exist.")
        return
    
    existing_images = set(f.name for f in image_dir.glob('*.png'))
    print(f"Found {len(existing_images)} images in the images directory.")
    
    # Extract image references from the markdown file
    if not thesis_path.exists():
        print(f"Error: Thesis file {thesis_path} does not exist.")
        return
    
    with open(thesis_path, 'r') as f:
        content = f.read()
    
    image_refs = re.findall(r'!\[.*?\]\((.*?)\)', content)
    image_filenames = [Path(ref).name for ref in image_refs if 'images/' in ref]
    
    print(f"Found {len(image_filenames)} image references in the markdown file.")
    
    # Check for missing images
    missing_images = set(image_filenames) - existing_images
    if missing_images:
        print("\nWARNING: The following referenced images are missing:")
        for img in sorted(missing_images):
            print(f"  - {img}")
    else:
        print("\nAll referenced images exist in the images directory.")
    
    # Check for unused images
    unused_images = existing_images - set(image_filenames)
    if unused_images:
        print("\nNOTE: The following images exist but are not referenced in the markdown:")
        for img in sorted(unused_images):
            print(f"  - {img}")
    else:
        print("\nAll images in the directory are referenced in the markdown.")
    
    print("\nCheck completed successfully.")

if __name__ == "__main__":
    check_markdown_images()
