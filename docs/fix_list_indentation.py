#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/fix_list_indentation.py

import re
import os

def fix_list_indentation(input_file, output_file):
    print(f"Reading from {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        print(f"Read {len(lines)} lines from input file")
        
        # Fix list indentation
        fixed_lines = []
        
        for i in range(len(lines)):
            line = lines[i]
            
            # If line starts with spaces followed by list marker (- or *)
            if re.match(r'^\s+[-*]', line):
                # Count spaces
                spaces = len(re.match(r'^(\s+)', line).group(1))
                
                # If it's a nested list item (parent list item exists)
                is_nested = False
                for j in range(i-1, max(0, i-5), -1):
                    if re.match(r'^[-*]', lines[j].strip()):
                        is_nested = True
                        break
                
                if is_nested and spaces != 4:
                    # Make it exactly 4 spaces for nested list items
                    line = re.sub(r'^\s+', '    ', line)
                elif not is_nested and spaces > 0:
                    # Make it 0 spaces for top-level list items
                    line = re.sub(r'^\s+', '', line)
            
            fixed_lines.append(line)
        
        print(f"Writing to {output_file}")
        with open(output_file, 'w') as f:
            f.writelines(fixed_lines)
        
        print(f"Fixed list indentation in {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    input_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document.md"
    output_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document_fixed.md"
    fix_list_indentation(input_file, output_file)
