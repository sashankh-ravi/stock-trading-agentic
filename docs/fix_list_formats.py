#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/fix_list_formats.py

import re
import sys

def fix_list_indentation(input_file, output_file):
    print(f"Reading from {input_file}")
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Check current indentation
    list_items = re.findall(r'^(\s*)([*\-+]|\d+\.)\s', content, re.MULTILINE)
    print(f"Found {len(list_items)} list items")
    
    indentation_counts = {}
    for spaces, marker in list_items:
        if len(spaces) in indentation_counts:
            indentation_counts[len(spaces)] += 1
        else:
            indentation_counts[len(spaces)] = 1
    
    print("Current indentation levels:")
    for spaces, count in sorted(indentation_counts.items()):
        print(f"- {spaces} spaces: {count} items")
    
    # Fix 1-space indentation (should be 0)
    if 1 in indentation_counts:
        content = re.sub(r'^( )([*\-+]|\d+\.)', r'\2', content, flags=re.MULTILINE)
        print(f"Fixed {indentation_counts[1]} items with 1-space indentation")
    
    # Fix 2-3 space indentation (should be 0 or 4)
    for i in range(2, 4):
        if i in indentation_counts:
            # If closer to 0, make it 0
            content = re.sub(r'^( {' + str(i) + '})([*\-+]|\d+\.)', r'\2', content, flags=re.MULTILINE)
            print(f"Fixed {indentation_counts[i]} items with {i}-space indentation to 0 spaces")
    
    # Fix 5+ space indentation (make it 4 if it's closer to 4)
    for i in range(5, 8):
        if i in indentation_counts:
            content = re.sub(r'^( {' + str(i) + '})([*\-+]|\d+\.)', r'    \2', content, flags=re.MULTILINE)
            print(f"Fixed {indentation_counts[i]} items with {i}-space indentation to 4 spaces")
    
    # Write the fixed content
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Fixed list indentation in {output_file}")

if __name__ == "__main__":
    print("Script started")
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document.md"
        output_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document_fixed.md"
    
    print(f"Using input file: {input_file}")
    print(f"Using output file: {output_file}")
    
    try:
        fix_list_indentation(input_file, output_file)
        print("Script completed successfully")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
