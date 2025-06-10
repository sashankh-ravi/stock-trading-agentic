#!/usr/bin/env python3
"""
Continue fixing markdown formatting issues in data_download.md
- Remove duplicate section headers with (2), (3), etc.
- Fix broken indentation in code blocks
- Fix numbered lists and bullet points
"""

import re

def fix_remaining_issues(file_path):
    """Fix remaining markdown formatting issues"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove duplicate section markers (2), (3), etc.
    content = re.sub(r'(#{1,6}\s+[^#\n]+)\s+\(\d+\)', r'\1', content)
    
    # Fix broken dictionary indentation in code blocks
    def fix_code_block_formatting(match):
        code_content = match.group(1)
        lines = code_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Comments should keep minimal indentation
            if stripped.startswith('#'):
                fixed_lines.append('    ' + stripped)
                continue
            
            # Dictionary/list items inside functions
            if "': " in stripped and not stripped.startswith('def ') and not stripped.startswith('return '):
                if not line.startswith('        '):  # Ensure proper dict indentation
                    fixed_lines.append('        ' + stripped)
                else:
                    fixed_lines.append(line.rstrip())
                continue
            
            # Function definitions and return statements
            if stripped.startswith(('def ', 'return ', 'if ', 'for ', 'while ', 'try:', 'except:', 'finally:')):
                if not line.startswith('    '):
                    fixed_lines.append('    ' + stripped)
                else:
                    fixed_lines.append(line.rstrip())
                continue
            
            # Variable assignments and other code
            if '=' in stripped or stripped.endswith(':') or stripped.startswith(('import ', 'from ')):
                if not line.startswith('    '):
                    fixed_lines.append('    ' + stripped)
                else:
                    fixed_lines.append(line.rstrip())
                continue
            
            # Keep line as is if already properly formatted
            fixed_lines.append(line.rstrip())
        
        return '\n'.join(fixed_lines)
    
    # Apply to all Python code blocks
    content = re.sub(r'```python\n(.*?)\n```', 
                    lambda m: f"```python\n{fix_code_block_formatting(m)}\n```", 
                    content, flags=re.DOTALL)
    
    # Fix list items that should be properly indented
    content = re.sub(r'\n\n\s*-\s+([A-Z])', r'\n    - \1', content)
    
    # Fix feature lists in docstrings
    content = re.sub(r'(Features:\n)\s*-\s+', r'\1    - ', content)
    
    # Fix broken numbered lists
    content = re.sub(r'\n\n(\d+)\.\s+([A-Z])', r'\n\1. \2', content)
    
    # Fix trailing whitespace and multiple blank lines
    content = re.sub(r' +\n', '\n', content)  # Remove trailing whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 consecutive newlines
    
    return content

def main():
    file_path = "/home/sashankhravi/Documents/stock-trading-agentic/data_download.md"
    
    print("Fixing remaining markdown formatting issues...")
    fixed_content = fix_remaining_issues(file_path)
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Remaining markdown formatting issues fixed!")

if __name__ == "__main__":
    main()
