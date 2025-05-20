#!/usr/bin/env python3
#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/validate_documentation.py

"""
Validation script for the Nifty 500 Trading System documentation.
This script checks for:
1. Broken links in documentation files
2. Missing images referenced in documentation
3. Consistency between different documentation files
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Define paths
docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
images_dir = docs_dir / 'images'
patterns_dir = images_dir / 'patterns'

# Define documentation files to check
main_docs = [
    docs_dir / 'index.md',
    docs_dir / 'nifty500_trading_system_master_document.md',
    docs_dir / 'comprehensive_pattern_guide_enhanced.md',
    docs_dir / 'indicator_explanations_enhanced.md',
]

# Patterns for finding references
link_pattern = re.compile(r'\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)')
image_pattern = re.compile(r'!\[(?P<alt_text>[^\]]*)\]\((?P<image_path>[^)]+)\)')
heading_pattern = re.compile(r'^#+\s+(.+?)$', re.MULTILINE)

class DocumentationValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Track files and resources
        self.all_files = set()
        self.all_images = set()
        self.referenced_files = defaultdict(set)
        self.referenced_images = defaultdict(set)
        self.headings = defaultdict(set)

    def scan_filesystem(self):
        """Scan filesystem to catalog available files and images."""
        print("Scanning filesystem...")
        
        # Scan documentation files
        for file in docs_dir.glob('*.md'):
            self.all_files.add(file.name)
        
        # Scan images
        for img in images_dir.glob('*.png'):
            self.all_images.add(f"images/{img.name}")
        
        for img in patterns_dir.glob('*.png'):
            self.all_images.add(f"images/patterns/{img.name}")
        
        print(f"Found {len(self.all_files)} documentation files")
        print(f"Found {len(self.all_images)} images")
    
    def parse_documents(self):
        """Parse all main documentation files to extract references and headings."""
        print("\nParsing documentation files...")
        
        for doc_path in main_docs:
            if not doc_path.exists():
                self.errors.append(f"Documentation file not found: {doc_path}")
                continue
            
            print(f"Parsing {doc_path.name}...")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract links to other documents
            for match in link_pattern.finditer(content):
                url = match.group('url')
                if not url.startswith(('http', '#')):
                    self.referenced_files[doc_path.name].add(url)
            
            # Extract image references
            for match in image_pattern.finditer(content):
                image_path = match.group('image_path')
                self.referenced_images[doc_path.name].add(image_path)
            
            # Extract headings
            for match in heading_pattern.finditer(content):
                heading = match.group(1).strip()
                self.headings[doc_path.name].add(heading)
    
    def check_document_references(self):
        """Check if all referenced documents exist."""
        print("\nChecking document references...")
        
        for doc, references in self.referenced_files.items():
            for ref in references:
                # Skip anchor references and external links
                if ref.startswith(('#', 'http')):
                    continue
                
                # Check if the referenced file exists
                if ref not in self.all_files and not (docs_dir / ref).exists():
                    self.errors.append(f"In {doc}: Referenced file not found: {ref}")
    
    def check_image_references(self):
        """Check if all referenced images exist."""
        print("\nChecking image references...")
        
        for doc, images in self.referenced_images.items():
            for img in images:
                # Normalize path
                img_path = img
                if not img.startswith(('images/', '/')):
                    img_path = f"images/{img}"
                
                # Check if the image exists
                if img_path not in self.all_images and not (docs_dir / img_path).exists():
                    self.errors.append(f"In {doc}: Referenced image not found: {img}")
    
    def check_heading_consistency(self):
        """Check for consistency in headings across documents."""
        print("\nChecking heading consistency...")
        
        # Look for similar headings across documents
        all_headings = defaultdict(list)
        for doc, headings in self.headings.items():
            for heading in headings:
                all_headings[heading.lower()].append((doc, heading))
        
        # Check for inconsistent capitalization
        for heading, occurrences in all_headings.items():
            if len(occurrences) > 1:
                unique_headings = set(h for _, h in occurrences)
                if len(unique_headings) > 1:
                    docs = [doc for doc, _ in occurrences]
                    self.warnings.append(f"Inconsistent heading formatting: '{heading}' in {', '.join(docs)}")
    
    def check_unused_images(self):
        """Check for images that exist but are not referenced."""
        print("\nChecking for unused images...")
        
        # Collect all referenced images
        all_referenced = set()
        for images in self.referenced_images.values():
            for img in images:
                # Normalize path
                if not img.startswith(('images/', '/')):
                    img = f"images/{img}"
                all_referenced.add(img)
        
        # Find unused images
        unused = self.all_images - all_referenced
        if unused:
            self.info.append(f"Found {len(unused)} unused images:")
            for img in sorted(unused):
                self.info.append(f"  - {img}")
    
    def check_pdf_files(self):
        """Check if PDF files referenced in documentation exist."""
        print("\nChecking PDF files...")
        
        pdf_files = list(docs_dir.glob('*.pdf'))
        pdf_names = [pdf.name for pdf in pdf_files]
        
        # Check if the main enhanced PDF exists
        if 'Nifty500_Trading_System_Complete_Enhanced.pdf' in pdf_names:
            self.info.append("Enhanced PDF documentation found")
        else:
            self.warnings.append("Enhanced PDF documentation not found")
        
        # Check for references to non-existent PDFs
        for doc, references in self.referenced_files.items():
            for ref in references:
                if ref.endswith('.pdf') and ref not in pdf_names:
                    self.errors.append(f"In {doc}: Referenced PDF not found: {ref}")
    
    def validate(self):
        """Run all validation checks."""
        self.scan_filesystem()
        self.parse_documents()
        
        self.check_document_references()
        self.check_image_references()
        self.check_heading_consistency()
        self.check_unused_images()
        self.check_pdf_files()
        
        return len(self.errors) == 0
    
    def print_report(self):
        """Print validation report."""
        print("\n" + "="*60)
        print("DOCUMENTATION VALIDATION REPORT")
        print("="*60)
        
        if self.errors:
            print("\n🚫 ERRORS:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.info:
            print("\nℹ️ INFORMATION:")
            for info in self.info:
                print(f"  • {info}")
        
        print("\n" + "="*60)
        if not self.errors:
            print("✅ Documentation validation PASSED!")
        else:
            print(f"❌ Documentation validation FAILED with {len(self.errors)} errors.")
        print("="*60)

def main():
    """Main function for validating documentation."""
    print("Starting documentation validation...")
    validator = DocumentationValidator()
    
    if validator.validate():
        validator.print_report()
        return 0
    else:
        validator.print_report()
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""
Validation script for the Nifty 500 Trading System documentation.
This script checks for:
1. Broken links in documentation files
2. Missing images referenced in documentation
3. Consistency between different documentation files
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Define paths
docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
images_dir = docs_dir / 'images'
patterns_dir = images_dir / 'patterns'

# Define documentation files to check
main_docs = [
    docs_dir / 'index.md',
    docs_dir / 'nifty500_trading_system_master_document.md',
    docs_dir / 'comprehensive_pattern_guide_enhanced.md',
    docs_dir / 'indicator_explanations_enhanced.md',
]

# Patterns for finding references
link_pattern = re.compile(r'\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)')
image_pattern = re.compile(r'!\[(?P<alt_text>[^\]]*)\]\((?P<image_path>[^)]+)\)')
heading_pattern = re.compile(r'^#+\s+(.+?)$', re.MULTILINE)

class DocumentationValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Track files and resources
        self.all_files = set()
        self.all_images = set()
        self.referenced_files = defaultdict(set)
        self.referenced_images = defaultdict(set)
        self.headings = defaultdict(set)

    def scan_filesystem(self):
        """Scan filesystem to catalog available files and images."""
        print("Scanning filesystem...")
        
        # Scan documentation files
        for file in docs_dir.glob('*.md'):
            self.all_files.add(file.name)
        
        # Scan images
        for img in images_dir.glob('*.png'):
            self.all_images.add(f"images/{img.name}")
        
        for img in patterns_dir.glob('*.png'):
            self.all_images.add(f"images/patterns/{img.name}")
        
        print(f"Found {len(self.all_files)} documentation files")
        print(f"Found {len(self.all_images)} images")
    
    def parse_documents(self):
        """Parse all main documentation files to extract references and headings."""
        print("\nParsing documentation files...")
        
        for doc_path in main_docs:
            if not doc_path.exists():
                self.errors.append(f"Documentation file not found: {doc_path}")
                continue
            
            print(f"Parsing {doc_path.name}...")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract links to other documents
            for match in link_pattern.finditer(content):
                url = match.group('url')
                if not url.startswith(('http', '#')):
                    self.referenced_files[doc_path.name].add(url)
            
            # Extract image references
            for match in image_pattern.finditer(content):
                image_path = match.group('image_path')
                self.referenced_images[doc_path.name].add(image_path)
            
            # Extract headings
            for match in heading_pattern.finditer(content):
                heading = match.group(1).strip()
                self.headings[doc_path.name].add(heading)
    
    def check_document_references(self):
        """Check if all referenced documents exist."""
        print("\nChecking document references...")
        
        for doc, references in self.referenced_files.items():
            for ref in references:
                # Skip anchor references and external links
                if ref.startswith(('#', 'http')):
                    continue
                
                # Check if the referenced file exists
                if ref not in self.all_files and not (docs_dir / ref).exists():
                    self.errors.append(f"In {doc}: Referenced file not found: {ref}")
    
    def check_image_references(self):
        """Check if all referenced images exist."""
        print("\nChecking image references...")
        
        for doc, images in self.referenced_images.items():
            for img in images:
                # Normalize path
                img_path = img
                if not img.startswith(('images/', '/')):
                    img_path = f"images/{img}"
                
                # Check if the image exists
                if img_path not in self.all_images and not (docs_dir / img_path).exists():
                    self.errors.append(f"In {doc}: Referenced image not found: {img}")
    
    def check_heading_consistency(self):
        """Check for consistency in headings across documents."""
        print("\nChecking heading consistency...")
        
        # Look for similar headings across documents
        all_headings = defaultdict(list)
        for doc, headings in self.headings.items():
            for heading in headings:
                all_headings[heading.lower()].append((doc, heading))
        
        # Check for inconsistent capitalization
        for heading, occurrences in all_headings.items():
            if len(occurrences) > 1:
                unique_headings = set(h for _, h in occurrences)
                if len(unique_headings) > 1:
                    docs = [doc for doc, _ in occurrences]
                    self.warnings.append(f"Inconsistent heading formatting: '{heading}' in {', '.join(docs)}")
    
    def check_unused_images(self):
        """Check for images that exist but are not referenced."""
        print("\nChecking for unused images...")
        
        # Collect all referenced images
        all_referenced = set()
        for images in self.referenced_images.values():
            for img in images:
                # Normalize path
                if not img.startswith(('images/', '/')):
                    img = f"images/{img}"
                all_referenced.add(img)
        
        # Find unused images
        unused = self.all_images - all_referenced
        if unused:
            self.info.append(f"Found {len(unused)} unused images:")
            for img in sorted(unused):
                self.info.append(f"  - {img}")
    
    def check_pdf_files(self):
        """Check if PDF files referenced in documentation exist."""
        print("\nChecking PDF files...")
        
        pdf_files = list(docs_dir.glob('*.pdf'))
        pdf_names = [pdf.name for pdf in pdf_files]
        
        # Check if the main enhanced PDF exists
        if 'Nifty500_Trading_System_Complete_Enhanced.pdf' in pdf_names:
            self.info.append("Enhanced PDF documentation found")
        else:
            self.warnings.append("Enhanced PDF documentation not found")
        
        # Check for references to non-existent PDFs
        for doc, references in self.referenced_files.items():
            for ref in references:
                if ref.endswith('.pdf') and ref not in pdf_names:
                    self.errors.append(f"In {doc}: Referenced PDF not found: {ref}")
    
    def validate(self):
        """Run all validation checks."""
        self.scan_filesystem()
        self.parse_documents()
        
        self.check_document_references()
        self.check_image_references()
        self.check_heading_consistency()
        self.check_unused_images()
        self.check_pdf_files()
        
        return len(self.errors) == 0
    
    def print_report(self):
        """Print validation report."""
        print("\n" + "="*60)
        print("DOCUMENTATION VALIDATION REPORT")
        print("="*60)
        
        if self.errors:
            print("\n🚫 ERRORS:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.info:
            print("\nℹ️ INFORMATION:")
            for info in self.info:
                print(f"  • {info}")
        
        print("\n" + "="*60)
        if not self.errors:
            print("✅ Documentation validation PASSED!")
        else:
            print(f"❌ Documentation validation FAILED with {len(self.errors)} errors.")
        print("="*60)

def main():
    """Main function for validating documentation."""
    print("Starting documentation validation...")
    validator = DocumentationValidator()
    
    if validator.validate():
        validator.print_report()
        return 0
    else:
        validator.print_report()
        return 1

if __name__ == "__main__":
    sys.exit(main())
