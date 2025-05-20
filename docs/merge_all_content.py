#!/usr/bin/env python3
"""
Simple Content Merger

This script combines the main thesis content with indicator explanations
without attempting PDF generation, ensuring all content is properly merged.
"""

import os
from pathlib import Path
import re
import shutil

def merge_content():
    """Merge content from multiple files into a single, comprehensive markdown document"""
    print("Starting content merge process...")
    
    # Get the current script's directory
    current_dir = Path(__file__).parent.absolute()
    print(f"Working directory: {current_dir}")
    
    # Define paths
    thesis_path = current_dir / 'nifty500_trading_system_thesis.md'
    indicators_path = current_dir / 'indicator_explanations.md'
    candlestick_path = current_dir / 'candlestick_pattern_details.md'
    output_path = current_dir / 'Nifty500_Trading_System_Complete.md'
    
    # Make a backup of the original thesis file
    backup_path = current_dir / 'nifty500_trading_system_thesis_original.md'
    if not backup_path.exists():
        print(f"Creating backup of original thesis at: {backup_path}")
        shutil.copy2(thesis_path, backup_path)
    
    # Read the thesis content
    print(f"Reading thesis content from: {thesis_path}")
    with open(thesis_path, 'r', encoding='utf-8') as f:
        thesis_content = f.read()
        
    # Read the indicator explanations
    print(f"Reading indicator explanations from: {indicators_path}")
    with open(indicators_path, 'r', encoding='utf-8') as f:
        indicator_content = f.read()
        
    # Read the candlestick pattern details
    print(f"Reading candlestick pattern details from: {candlestick_path}")
    with open(candlestick_path, 'r', encoding='utf-8') as f:
        candlestick_content = f.read()
    
    # Find where to insert the indicator explanations in section 4
    indicator_insert_point = re.search(r'## 4\. Technical Indicators\s+', thesis_content)
    if indicator_insert_point:
        # Split at the insertion point
        insert_pos = indicator_insert_point.end()
        thesis_first_part = thesis_content[:insert_pos]
        thesis_second_part = thesis_content[insert_pos:]
        
        # Prepare the indicator content (excluding the heading which is already in the thesis)
        if indicator_content.startswith('## Detailed Technical Indicator Explanations'):
            clean_indicator_content = indicator_content.replace('## Detailed Technical Indicator Explanations', '')
        else:
            clean_indicator_content = indicator_content
            
        # Combine the parts
        thesis_content = f"{thesis_first_part}\n{clean_indicator_content}\n{thesis_second_part}"
    else:
        print("Warning: Could not find section '## 4. Technical Indicators' in the thesis")
        
    # Find where to insert the candlestick pattern explanations in section 3.3
    candlestick_insert_point = re.search(r'### 3\.3 Pattern Recognition Strategy\s+', thesis_content)
    if candlestick_insert_point:
        # We want to find where to insert after any existing subsections
        pattern_section_text = thesis_content[candlestick_insert_point.start():]
        next_section_match = re.search(r'### 3\.4|## 4\.', pattern_section_text)
        
        if next_section_match:
            # Find the end of section 3.3
            end_of_section = candlestick_insert_point.start() + next_section_match.start()
            
            # Check for existing subsections
            subsections = re.finditer(r'#### 3\.3\.\d+', thesis_content[candlestick_insert_point.start():end_of_section])
            subsection_numbers = [int(re.search(r'#### 3\.3\.(\d+)', m.group()).group(1)) for m in subsections]
            
            new_subsection_num = max(subsection_numbers) + 1 if subsection_numbers else 1
            
            # Format the candlestick content
            if candlestick_content.startswith('# Candlestick Pattern Implementation'):
                # Remove the title and use our own
                clean_candlestick = candlestick_content.split('\n', 1)[1]
            else:
                clean_candlestick = candlestick_content
                
            # Create the formatted section
            formatted_section = f"\n\n#### 3.3.{new_subsection_num} Candlestick Pattern Integration\n\n{clean_candlestick}"
            
            # Split thesis content and insert
            thesis_content = thesis_content[:end_of_section] + formatted_section + thesis_content[end_of_section:]
        else:
            # If no next section, add to the end of the file
            formatted_section = f"\n\n#### 3.3.1 Candlestick Pattern Integration\n\n{candlestick_content}"
            thesis_content += formatted_section
    else:
        print("Warning: Could not find section '### 3.3 Pattern Recognition Strategy' in the thesis")
    
    # Write the merged content
    print(f"Writing merged content to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(thesis_content)
    
    # Create an enriched thesis file with proper formatting
    enriched_path = current_dir / 'nifty500_trading_system_thesis_enriched.md'
    print(f"Writing enriched thesis to: {enriched_path}")
    with open(enriched_path, 'w', encoding='utf-8') as f:
        f.write(thesis_content)
    
    print(f"Content merge complete. Files created:")
    print(f"1. {output_path} - Full merged content")
    print(f"2. {enriched_path} - Enriched thesis")
    print("\nTo generate a PDF, you can run one of the following commands:")
    print("Option 1: pandoc -s Nifty500_Trading_System_Complete.md -o Nifty500_Trading_System_Complete.pdf")
    print("Option 2: python generate_complete_thesis.py (if properly configured)")
    
    return True

if __name__ == "__main__":
    merge_content()
