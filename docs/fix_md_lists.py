#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/fix_md_lists.py

import re
import sys

def fix_markdown_lists(input_file, output_file):
    # Read content
    print(f"Reading from {input_file}")
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Replace specific patterns
    print("Fixing list indentation...")
    
    # Fix 1-space indented lists to 0-space (top level)
    pattern1 = re.compile(r'^( )[-*]', re.MULTILINE)
    fixed1 = re.sub(pattern1, r'-', content)
    print(f"Fixed {len(content) - len(fixed1)} 1-space indents")
    
    # Fix 5-space indented lists to 4-space (nested level)
    pattern2 = re.compile(r'^(     )([-*])', re.MULTILINE)
    fixed2 = re.sub(pattern2, r'    \2', fixed1)
    print(f"Fixed {len(fixed1) - len(fixed2)} 5-space indents")
    
    # Write output
    print(f"Writing to {output_file}")
    with open(output_file, 'w') as f:
        f.write(fixed2)
    
    print(f"Completed: Fixed list indentation in {output_file}")

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
        fix_markdown_lists(input_file, output_file)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
