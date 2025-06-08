#!/usr/bin/env python3
import re
import os
import subprocess
from pathlib import Path

def extract_mermaid_diagrams(md_file_path, output_dir):
    """Extract Mermaid diagrams from a Markdown file and save them as separate .mmd files."""
    with open(md_file_path, 'r') as f:
        content = f.read()
    
    # Extract the title from the Markdown file
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else Path(md_file_path).stem
    
    # Clean the title for use as a filename
    title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_').lower()
    
    # Find Mermaid code blocks
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
    
    if not mermaid_blocks:
        print(f"No Mermaid diagrams found in {md_file_path}")
        return []
    
    output_files = []
    for i, diagram in enumerate(mermaid_blocks):
        # Create a filename based on the markdown filename and diagram index
        base_filename = f"{title}_{i+1}"
        output_path = os.path.join(output_dir, f"{base_filename}.mmd")
        
        with open(output_path, 'w') as f:
            f.write(diagram)
        
        output_files.append((output_path, base_filename))
        print(f"Extracted diagram {i+1} from {md_file_path} to {output_path}")
    
    return output_files

def convert_mermaid_to_png(mmd_file, output_dir, output_name):
    """Convert a .mmd file to a .png file using Mermaid CLI."""
    output_path = os.path.join(output_dir, f"{output_name}.png")
    
    # Use mmdc command to convert the Mermaid file to PNG
    try:
        subprocess.run(['mmdc', '-i', mmd_file, '-o', output_path], check=True)
        print(f"Converted {mmd_file} to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {mmd_file} to PNG: {e}")
        return None
    except FileNotFoundError:
        print("mmdc command not found. Trying alternative approach...")
        # Alternative approach using the mermaid-cli Python package
        with open(mmd_file, 'r') as f:
            mermaid_code = f.read()
        
        # Create a temporary HTML file
        html_path = os.path.join(output_dir, f"{output_name}_temp.html")
        with open(html_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{output_name}</title>
                <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                <script>mermaid.initialize({{startOnLoad:true}});</script>
            </head>
            <body>
                <div class="mermaid">
                {mermaid_code}
                </div>
            </body>
            </html>
            """)
        
        print(f"Created HTML file: {html_path}")
        print(f"Please manually convert {html_path} to {output_path}")
        return html_path

def main():
    # Directory paths
    docs_dir = Path("/home/sashankhravi/Documents/stock-trading-agentic/docs/images")
    output_dir = Path("/home/sashankhravi/Documents/stock-trading-agentic/docs/images/extracted")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Markdown files with diagrams
    md_files = [
        docs_dir / "comprehensive_testing_architecture.md",
        docs_dir / "data_pipeline_architecture.md",
        docs_dir / "market_regime_analysis.md",
        docs_dir / "technical_indicators_classification.md"
    ]
    
    for md_file in md_files:
        if md_file.exists():
            # Extract diagrams
            extracted_files = extract_mermaid_diagrams(md_file, output_dir)
            
            # Convert each extracted diagram to PNG
            for mmd_file, output_name in extracted_files:
                convert_mermaid_to_png(mmd_file, output_dir, output_name)
        else:
            print(f"File not found: {md_file}")

if __name__ == "__main__":
    main()
