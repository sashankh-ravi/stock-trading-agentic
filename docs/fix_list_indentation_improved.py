#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/fix_list_indentation_improved.py

import re
import os
import sys

def fix_list_indentation(input_file, output_file):
    print(f"Reading from {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Track list depth to apply proper indentation
        fixed_lines = []
        in_code_block = False
        
        lines = content.splitlines()
        print(f"Read {len(lines)} lines from input file")
        
        for i, line in enumerate(lines):
            # Don't modify code blocks
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                fixed_lines.append(line)
                continue
                
            if in_code_block:
                fixed_lines.append(line)
                continue
            
            # Process list items - matches both ordered (1. ) and unordered (- or *) lists
            list_match = re.match(r'^(\s*)([*\-]|\d+\.)\s', line)
            if list_match:
                spaces = len(list_match.group(1))
                marker = list_match.group(2)
                content_after_marker = line[list_match.end():]
                
                # Determine nesting level by looking at previous non-empty lines
                level = 0
                for j in range(i-1, max(0, i-10), -1):
                    prev_line = lines[j].strip()
                    if not prev_line:  # Skip empty lines
                        continue
                    
                    # Check if previous line is a list item
                    prev_match = re.match(r'^(\s*)([*\-]|\d+\.)\s', lines[j])
                    if prev_match:
                        prev_spaces = len(prev_match.group(1))
                        if prev_spaces < spaces:  # Current item is nested under this one
                            level = 1 + (prev_spaces // 4)  # Calculate level based on previous indentation
                            break
                        elif prev_spaces == spaces:  # Same level as previous item
                            level = prev_spaces // 4
                            break
                        # If prev_spaces > spaces, keep looking further up
                    else:
                        # Previous line is not a list item, so current item is top-level
                        level = 0
                        break
                
                # Apply appropriate indentation
                if level == 0:
                    indentation = ""  # Top level: no indentation
                else:
                    indentation = "    " * level  # 4 spaces per level for nested items
                
                fixed_line = f"{indentation}{marker} {content_after_marker}"
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        print(f"Writing to {output_file}")
        with open(output_file, 'w') as f:
            f.write("\n".join(fixed_lines))
        
        print(f"Fixed list indentation in {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")

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
        print(f"Script error: {str(e)}")
        import traceback
        traceback.print_exc()
