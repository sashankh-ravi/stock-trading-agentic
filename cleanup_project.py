#!/usr/bin/env python3
"""
Cleanup script for the stock_rl_agent_nifty_50 project.
This script removes unnecessary files to prepare the project for GitHub.
"""

import os
import shutil
import glob
from pathlib import Path

def clean_pyc_and_pycache():
    """Remove Python bytecode files and __pycache__ directories"""
    print("Removing Python bytecode files and __pycache__ directories...")
    
    # Get the project root directory
    project_root = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50')
    
    # Find and remove __pycache__ directories
    for cache_dir in project_root.glob('**/__pycache__'):
        print(f"Removing: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    # Find and remove .pyc and .pyo files
    for pyc_file in project_root.glob('**/*.py[co]'):
        print(f"Removing: {pyc_file}")
        os.remove(pyc_file)
    
    print("Python bytecode and caches removed.")

def clean_docs_folder():
    """Clean up the docs folder by removing unnecessary files"""
    print("\nCleaning up docs folder...")
    
    docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
    
    # List of files to keep
    essential_docs = [
        'nifty500_trading_system_thesis.md',           # Main thesis document
        'comprehensive_pattern_guide_enhanced.md',      # Enhanced pattern guide
        'indicator_explanations_enhanced.md',           # Enhanced indicator explanations
        'generate_complete_thesis.py',                  # Script to generate the thesis PDF
        'index.md',                                     # Main index file
        'thesis_formatting_summary.md',                 # Summary of formatting changes
        'Nifty500_Trading_System_Complete.pdf',         # Final compiled PDF
        'images'                                        # Images directory
    ]
    
    # Remove temporary and unnecessary files
    for pattern in [
        'temp_*.md', 'temp_*.html', '*_backup*.md', '*_fixed.md', 
        '*_merged.md', '*.html', 'Nifty500_Thesis_Backup_*.pdf'
    ]:
        for file_path in docs_dir.glob(pattern):
            if file_path.name not in essential_docs:
                print(f"Removing: {file_path}")
                os.remove(file_path)
    
    # Remove unnecessary Python scripts
    unnecessary_scripts = [
        'cleanup_docs.py', 'fix_pdf_formatting.py', 'format_thesis.py',
        'standardize_list_indentation.py', 'test_plot.py', 'pdf_validator.py',
        'verify_cross_references.py', 'simple_thesis_generator.py',
        'merge_indicator_explanations.py', 'finalize_thesis.py',
        'generate_thesis_pdf.py', 'create_pdf_with_images.py'
    ]
    
    for script in unnecessary_scripts:
        script_path = docs_dir / script
        if script_path.exists():
            print(f"Removing: {script_path}")
            os.remove(script_path)
            
    # Additional redundant docs to remove
    redundant_docs = [
        'nifty500_trading_system_master_document.md',       # Redundant with thesis.md
        'nifty500_trading_system_master_document_final.md', # Redundant with thesis.md
        'nifty500_trading_system_thesis_enriched.md',       # Redundant with thesis.md
        'comprehensive_pattern_guide.md',                   # Keep only enhanced version
        'indicator_explanations.md',                        # Keep only enhanced version
        'markdown_fixes_summary.md',                        # Summary not needed for GitHub
        'missing_sections.md',                              # Working file not needed
        'pattern_enhancement_report.md',                    # Report not needed for GitHub
        'training_documentation.md',                        # Internal document
        'updated_toc.md',                                   # Working file not needed
        'documentation_status_report.md'                    # Status report not needed
    ]
    
    for doc in redundant_docs:
        doc_path = docs_dir / doc
        if doc_path.exists():
            print(f"Removing redundant doc: {doc_path}")
            os.remove(doc_path)
    
    # Clean output directory
    output_dir = docs_dir / 'output'
    if output_dir.exists() and output_dir.is_dir():
        unnecessary_outputs = [
            'nifty500_trading_system_complete.md'           # Redundant with the PDF
        ]
        
        for output_file in unnecessary_outputs:
            output_path = output_dir / output_file
            if output_path.exists():
                print(f"Removing: {output_path}")
                os.remove(output_path)
                
        # Remove output directory if empty
        if not any(output_dir.iterdir()):
            print(f"Removing empty directory: {output_dir}")
            output_dir.rmdir()
    
    print("Docs folder cleaned up.")

def clean_project_root():
    """Clean up the project root directory"""
    print("\nCleaning up project root...")
    
    project_root = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50')
    
    # Remove duplicate and unnecessary files
    unnecessary_files = [
        'ta-lib-0.4.0-src.tar.gz',
        'ta-lib-0.4.0-src.tar.gz.1'
    ]
    
    for file in unnecessary_files:
        file_path = project_root / file
        if file_path.exists():
            print(f"Removing: {file_path}")
            os.remove(file_path)
    
    print("Project root cleaned up.")

def main():
    """Main function to clean up the project"""
    print("Starting project cleanup...")
    
    # Clean Python bytecode and caches
    clean_pyc_and_pycache()
    
    # Clean up the docs folder
    clean_docs_folder()
    
    # Clean up the project root
    clean_project_root()
    
    # Verify project is ready for GitHub
    verify_github_readiness()
    
    print("\nProject cleanup completed successfully!")
    print("The project is now ready for GitHub.")

def verify_github_readiness():
    """Verify that the project is ready for GitHub"""
    print("\nVerifying GitHub readiness...")
    
    project_root = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50')
    
    # Check for essential files
    essential_files = [
        'README.md',
        'LICENSE',
        '.gitignore',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in essential_files:
        if not (project_root / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: The following essential files are missing: {', '.join(missing_files)}")
    else:
        print("All essential GitHub files are present.")
    
    # Check for potentially sensitive or unnecessary large files
    large_files = []
    for file_path in project_root.glob('**/*'):
        if file_path.is_file() and file_path.stat().st_size > 10 * 1024 * 1024:  # Files larger than 10 MB
            large_files.append(str(file_path.relative_to(project_root)))
    
    if large_files:
        print(f"Warning: The following large files may not be suitable for GitHub: {', '.join(large_files)}")
    else:
        print("No large files detected that might cause issues with GitHub.")
    
    print("GitHub readiness verification completed.")

if __name__ == "__main__":
    main()
