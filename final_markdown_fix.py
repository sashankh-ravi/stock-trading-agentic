#!/usr/bin/env python3
"""
Final comprehensive markdown formatting fix for data_download.md
This script will fix all remaining formatting issues in one pass
"""

import re

def comprehensive_markdown_fix(file_path):
    """Apply comprehensive formatting fixes"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Remove all duplicate section markers
    content = re.sub(r'(#{1,6}\s+[^#\n]+)\s+\(\d+\)', r'\1', content)
    
    # 2. Fix Python code blocks with broken indentation
    def fix_python_code_block(match):
        code = match.group(1)
        lines = code.split('\n')
        fixed_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                fixed_lines.append('')
                continue
                
            # Handle different types of code lines
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Docstrings
                if indent_level > 0:
                    fixed_lines.append('    ' + stripped)
                else:
                    fixed_lines.append(stripped)
            elif stripped.startswith('def ') or stripped.startswith('class '):
                # Function/class definitions
                fixed_lines.append(stripped)
                indent_level = 1
            elif stripped.startswith(('if ', 'for ', 'while ', 'try:', 'with ', 'elif ', 'else:', 'except:', 'finally:')):
                # Control structures
                base_indent = '    ' * indent_level
                fixed_lines.append(base_indent + stripped)
                if stripped.endswith(':'):
                    indent_level += 1
            elif stripped.startswith('return '):
                # Return statements
                base_indent = '    ' * indent_level
                fixed_lines.append(base_indent + stripped)
            elif stripped.startswith('#'):
                # Comments
                base_indent = '    ' * max(0, indent_level)
                fixed_lines.append(base_indent + stripped)
            elif "': " in stripped and not stripped.startswith(('def ', 'class ', 'import ', 'from ')):
                # Dictionary items
                base_indent = '    ' * max(1, indent_level + 1)
                fixed_lines.append(base_indent + stripped)
            elif stripped.startswith('lambda:'):
                # Lambda functions
                base_indent = '    ' * max(1, indent_level + 1)
                fixed_lines.append(base_indent + stripped)
            elif stripped in ['}', ']', ')', '},']:
                # Closing brackets
                base_indent = '    ' * max(0, indent_level)
                fixed_lines.append(base_indent + stripped)
            else:
                # Regular code lines
                if indent_level > 0:
                    base_indent = '    ' * indent_level
                    fixed_lines.append(base_indent + stripped)
                else:
                    fixed_lines.append(stripped)
        
        return '\n'.join(fixed_lines)
    
    # Apply Python code block fixes
    content = re.sub(r'```python\n(.*?)\n```', 
                    lambda m: f"```python\n{fix_python_code_block(m)}\n```", 
                    content, flags=re.DOTALL)
    
    # 3. Fix numbered lists with consistent numbering
    def fix_numbered_list(text):
        # Find patterns like "1. item\n1. item\n1. item" and fix numbering
        pattern = r'(\d+)\.\s+([^\n]+)(?:\n\1\.\s+([^\n]+))*'
        
        def replace_numbers(match):
            lines = match.group(0).split('\n')
            result = []
            for i, line in enumerate(lines):
                if line.strip():
                    content_match = re.match(r'\d+\.\s+(.+)', line)
                    if content_match:
                        result.append(f"{i+1}. {content_match.group(1)}")
                    else:
                        result.append(line)
            return '\n'.join(result)
        
        return re.sub(pattern, replace_numbers, text)
    
    content = fix_numbered_list(content)
    
    # 4. Fix bullet points with proper indentation
    content = re.sub(r'\n\n\s*-\s+([A-Z])', r'\n    - \1', content)
    
    # 5. Fix feature lists in docstrings
    content = re.sub(r'(Features:\n)\s*-\s+', r'\1    - ', content)
    
    # 6. Clean up excessive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 7. Remove trailing whitespace
    content = re.sub(r' +\n', '\n', content)
    
    return content

def main():
    file_path = "/home/sashankhravi/Documents/stock-trading-agentic/data_download.md"
    
    print("Applying comprehensive markdown formatting fixes...")
    
    # Read current content
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Apply fixes
    fixed_content = comprehensive_markdown_fix(file_path)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Comprehensive markdown formatting completed!")
    print(f"Original size: {len(original_content)} characters")
    print(f"Fixed size: {len(fixed_content)} characters")

if __name__ == "__main__":
    main()
