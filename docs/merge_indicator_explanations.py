#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/merge_indicator_explanations.py

import os
import re
from pathlib import Path

def main():
    """
    Merge the indicator explanations into the thesis markdown
    """
    print("Starting merge process...")
    
    # Get the current script's directory
    current_dir = Path(__file__).parent.absolute()
    print(f"Working directory: {current_dir}")
    
    # Define paths
    thesis_path = current_dir / 'nifty500_trading_system_thesis.md'
    indicators_path = current_dir / 'indicator_explanations.md'
    output_path = current_dir / 'nifty500_trading_system_thesis_enriched.md'
    
    # Verify required files exist
    if not thesis_path.exists():
        print(f"Error: Thesis file not found at {thesis_path}")
        return False
    
    if not indicators_path.exists():
        print(f"Error: Indicator explanations file not found at {indicators_path}")
        return False
    
    # Read the thesis content
    print(f"Reading thesis content from: {thesis_path}")
    try:
        with open(thesis_path, 'r', encoding='utf-8') as f:
            thesis_content = f.read()
    except Exception as e:
        print(f"Error reading thesis file: {e}")
        return False
    
    # Read the indicator explanations
    print(f"Reading indicator explanations from: {indicators_path}")
    try:
        with open(indicators_path, 'r', encoding='utf-8') as f:
            indicator_content = f.read()
    except Exception as e:
        print(f"Error reading indicator explanations file: {e}")
        return False
    
    # Locate the Technical Indicators section in the thesis
    technical_section_pattern = r'(### 4\.1 Indicator Categories.*?)(### 4\.2 Custom Composite Indicators)'
    match = re.search(technical_section_pattern, thesis_content, re.DOTALL)
    
    if not match:
        print("Could not find Technical Indicators section in thesis")
        return False
    
    # Extract the section
    original_section = match.group(1)
    after_section = match.group(2)
    
    # Extract the content from the indicator explanations that we want to insert
    # We'll take the detailed explanations but skip the header
    explanation_section_pattern = r'## Detailed Technical Indicator Explanations\s*(.*)'
    explanation_match = re.search(explanation_section_pattern, indicator_content, re.DOTALL)
    
    if not explanation_match:
        print("Could not find explanation content")
        return False
    
    explanation_content = explanation_match.group(1).strip()
    
    # Create the new section
    new_section = original_section + "\n\n### 4.1.5 Detailed Indicator Explanations\n\n" + explanation_content + "\n\n"
    
    # Replace the old section with the new section
    enriched_thesis = thesis_content.replace(original_section + after_section, new_section + after_section)
    
    # Write the enriched thesis to a new file
    print(f"Writing enriched thesis to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(enriched_thesis)
        print(f"Successfully created enriched thesis markdown at: {output_path}")
        return True
    except Exception as e:
        print(f"Error writing enriched thesis file: {e}")
        return False

if __name__ == "__main__":
    success = main()
    print(f"Process {'succeeded' if success else 'failed'}")
