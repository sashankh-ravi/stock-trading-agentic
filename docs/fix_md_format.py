#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/fix_md_format.py

import re
import sys

def fix_list_indentation(input_file, output_file):
    print(f"Reading from {input_file}")
    
    # Read the file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_code_block = False
    one_space_count = 0
    five_space_count = 0
    
    for line in lines:
        # Skip code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            fixed_lines.append(line)
            continue
        
        if in_code_block:
            fixed_lines.append(line)
            continue
        
        original_line = line
        
        # Check for 1-space indentation
        if line.startswith(' -') or line.startswith(' *'):
            line = line[1:]  # Remove the single space
            one_space_count += 1
        
        # Check for 5-space indentation
        if line.startswith('     -') or line.startswith('     *'):
            line = '    ' + line[5:]  # Replace 5 spaces with 4
            five_space_count += 1
        
        fixed_lines.append(line)
    
    print(f"Fixed {one_space_count} lines with 1-space indentation")
    print(f"Fixed {five_space_count} lines with 5-space indentation")
    
    # Write the fixed file
    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed list indentation in {output_file}")

if __name__ == "__main__":
    print("Script started")
    input_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document.md"
    output_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document_fixed.md"
    
    try:
        fix_list_indentation(input_file, output_file)
        print("Script completed successfully")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
