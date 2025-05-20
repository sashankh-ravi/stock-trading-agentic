#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/fix_markdown_formatting.py

import re
import os

def fix_markdown_formatting(input_file, output_file):
    print(f"Reading from {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return
    
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        print(f"Read {len(content)} characters from input file")
        
        # Fix missing blank lines around headings
        content = re.sub(r'([^\n])\n(#+\s+[^\n]+)', r'\1\n\n\2', content)
        content = re.sub(r'(#+\s+[^\n]+)\n([^\n])', r'\1\n\n\2', content)
        
        # Fix missing blank lines around lists
        content = re.sub(r'([^\n])\n([-*]\s+[^\n]+)', r'\1\n\n\2', content)
        content = re.sub(r'([-*]\s+[^\n]+)\n([^\n#-*])', r'\1\n\n\2', content)
        
        # Fix missing blank lines around code blocks
        content = re.sub(r'([^\n])\n```', r'\1\n\n```', content)
        content = re.sub(r'```\n([^\n])', r'```\n\n\1', content)
        
        # Normalize list indentation - use simple approach to standardize all list items to 4 spaces
        lines = content.split('\n')
        for i in range(len(lines)):
            # If it's a list item with 1, 2, 3, or 5+ spaces, standardize to 4 spaces or 0 for top level
            if re.match(r'^\s+[-*]\s+', lines[i]):
                # Count the number of leading spaces
                leading_spaces = len(re.match(r'^(\s+)', lines[i]).group(1))
                
                # If 1-3 spaces, it's probably meant to be a top-level list item
                if 1 <= leading_spaces <= 3:
                    lines[i] = re.sub(r'^\s+', '', lines[i])
                # If 5+ spaces, it's probably meant to be a sub-list item with 4 spaces
                elif leading_spaces >= 5:
                    lines[i] = re.sub(r'^\s+', '    ', lines[i])
        
        content = '\n'.join(lines)
        
        print(f"Writing to {output_file}")
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"Fixed markdown formatting in {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    input_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document.md"
    output_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document_fixed.md"
    fix_markdown_formatting(input_file, output_file)
