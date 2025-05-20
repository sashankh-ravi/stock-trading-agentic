#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/fix_all_md_issues.py

import re
import sys
import os

def fix_list_indentation(content):
    """Fix list indentation in Markdown content."""
    print("Fixing list indentation...")
    
    # Process content line by line for better control
    lines = content.split('\n')
    fixed_lines = []
    in_code_block = False
    one_space_fixes = 0
    five_space_fixes = 0
    
    for line in lines:
        # Skip code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            fixed_lines.append(line)
            continue
        
        if in_code_block:
            fixed_lines.append(line)
            continue
        
        # Fix 1-space list items
        if re.match(r'^ [-*+]', line):
            line = line[1:]  # Remove the single space
            one_space_fixes += 1
        
        # Fix 5-space list items to have 4 spaces
        elif re.match(r'^     [-*+]', line):
            line = '    ' + line[5:]  # Replace 5 spaces with 4
            five_space_fixes += 1
        
        fixed_lines.append(line)
    
    print(f"Fixed {one_space_fixes} items with 1-space indentation")
    print(f"Fixed {five_space_fixes} items with 5-space indentation")
    
    return '\n'.join(fixed_lines)

def fix_blank_lines_around_elements(content):
    """Fix blank lines around headings, lists, and code blocks."""
    print("Fixing blank lines around elements...")
    
    # Ensure blank line before headings (except at beginning of file)
    content = re.sub(r'([^\n])\n(#{1,6} )', r'\1\n\n\2', content)
    
    # Ensure blank line after headings
    content = re.sub(r'(#{1,6} .*)\n([^#\n])', r'\1\n\n\2', content)
    
    # Ensure blank line before lists that don't follow headings or blank lines
    content = re.sub(r'([^\n#])\n([-*+] )', r'\1\n\n\2', content)
    content = re.sub(r'([^\n#])\n(\d+\. )', r'\1\n\n\2', content)
    
    # Ensure blank line before code blocks
    content = re.sub(r'([^\n])\n(```)', r'\1\n\n\2', content)
    
    # Ensure blank line after code blocks
    content = re.sub(r'(```)\n([^\n])', r'\1\n\n\2', content)
    
    # Fix too many blank lines (more than 2) between elements
    content = re.sub(r'\n{3,}', r'\n\n', content)
    
    return content

def fix_numbered_list_prefix(content):
    """Fix numbered list prefix issues."""
    print("Fixing numbered list prefix issues...")
    
    lines = content.split('\n')
    fixed_lines = []
    in_code_block = False
    in_numbered_list = False
    current_number = 1
    current_indent = ""
    
    for line in lines:
        # Skip code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            fixed_lines.append(line)
            continue
        
        if in_code_block:
            fixed_lines.append(line)
            continue
        
        # Check for numbered list items
        numbered_match = re.match(r'^(\s*)(\d+)\.(\s+)(.+)$', line)
        if numbered_match:
            indent, number, spaces, content = numbered_match.groups()
            
            # Start of a new list or continuing at a different indent level
            if not in_numbered_list or indent != current_indent:
                in_numbered_list = True
                current_number = 1
                current_indent = indent
            
            # Fix the number to be sequential
            line = f"{indent}{current_number}.{spaces}{content}"
            current_number += 1
        else:
            # Check if this is an empty line or non-list content
            if not line.strip() or not re.match(r'^\s*[-*+]', line):
                # Only reset if it's truly the end of a list (empty line)
                if not line.strip():
                    in_numbered_list = False
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_all_markdown_issues(input_file, output_file):
    """Fix all markdown linting issues in one pass."""
    print(f"Reading from {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Apply all fixes
        content = fix_list_indentation(content)
        content = fix_blank_lines_around_elements(content)
        content = fix_numbered_list_prefix(content)
        
        # Write the fixed content
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"Fixed markdown issues in {output_file}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_list_indentation(file_path):
    """Check list indentation in a markdown file."""
    print(f"Checking list indentation in {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find all list items and count their indentation
        list_items = []
        in_code_block = False
        for line in content.split('\n'):
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
                
            if in_code_block:
                continue
                
            # Check for list items (unordered and ordered)
            list_match = re.match(r'^(\s*)([-*+]|\d+\.)(\s+)', line)
            if list_match:
                indentation = len(list_match.group(1))
                list_items.append((indentation, line))
        
        # Count indentation levels
        indentation_levels = {}
        for indent, line in list_items:
            if indent not in indentation_levels:
                indentation_levels[indent] = 0
            indentation_levels[indent] += 1
        
        print(f"Found {len(list_items)} list items")
        print("Indentation levels:")
        for indent, count in sorted(indentation_levels.items()):
            print(f"- {indent} spaces: {count} items")
            
        # Report inconsistencies
        inconsistencies = []
        for indent in indentation_levels:
            if indent not in [0, 4, 8, 12]:  # Standard indentation levels
                inconsistencies.append(indent)
        
        if inconsistencies:
            print("Inconsistent indentation levels found:")
            for indent in inconsistencies:
                print(f"- {indent} spaces: should be 0, 4, 8, or 12 spaces")
        else:
            print("No inconsistent indentation levels found!")
            
        return len(inconsistencies) == 0
    except Exception as e:
        print(f"Error checking indentation: {str(e)}")
        return False

if __name__ == "__main__":
    print("Script started")
    
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document.md"
        output_file = "/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/nifty500_trading_system_master_document_final.md"
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    try:
        # Fix all markdown issues
        success = fix_all_markdown_issues(input_file, output_file)
        
        if success:
            print("\nVerifying the fixed file:")
            check_list_indentation(output_file)
        
        print("Script completed")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
