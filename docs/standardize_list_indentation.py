#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/standardize_list_indentation.py

import re
import sys

def standardize_list_indentation(input_file, output_file):
    print(f"Reading from {input_file}")
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Explicitly handle the most common list indentation cases
    # 1. Convert 1-space indents to 0 spaces (top-level)
    # 2. Convert 5-space indents to 4 spaces (nested level)
    
    lines = content.splitlines()
    fixed_lines = []
    in_code_block = False
    
    print(f"Processing {len(lines)} lines")
    
    for line in lines:
        original_line = line
        
        # Don't modify code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            fixed_lines.append(line)
            continue
            
        if in_code_block:
            fixed_lines.append(line)
            continue
        
        # Fix 1-space indentation to 0 spaces (make it top-level)
        if re.match(r'^ [-*]', line):  # Exactly 1 space before list marker
            line = line[1:]  # Remove the single space
            print(f"Fixed 1-space indent: '{original_line}' -> '{line}'")
            
        # Fix 5-space indentation to 4 spaces (make it properly nested)
        elif re.match(r'^     [-*]', line):  # Exactly 5 spaces before list marker
            line = line.replace('     ', '    ', 1)
            print(f"Fixed 5-space indent: '{original_line}' -> '{line}'")
            
        fixed_lines.append(line)
    
    print(f"Writing to {output_file}")
    with open(output_file, 'w') as f:
        f.write("\n".join(fixed_lines))
    
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
        standardize_list_indentation(input_file, output_file)
        print("Script completed successfully")
    except Exception as e:
        print(f"Script error: {str(e)}")
        import traceback
        traceback.print_exc()
