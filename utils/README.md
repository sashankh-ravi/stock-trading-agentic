# Utility Scripts

This directory contains utility scripts for maintenance and diagram generation.

## Scripts

- `html_to_png_converter.py`: Converts HTML Mermaid diagrams to PNG images
- `create_placeholder_images.py`: Creates placeholder PNG images with instructions
- `extract_diagrams.py`: Extracts Mermaid diagrams from Markdown to HTML

## Usage

### Converting HTML to PNG

1. Install required dependencies:
   ```bash
   pip install selenium pillow webdriver_manager
   ```

2. Run the converter:
   ```bash
   python html_to_png_converter.py
   ```

### Creating Placeholder Images

```bash
python create_placeholder_images.py
```

### Extracting Diagrams from Markdown

```bash
python extract_diagrams.py [input_markdown_file] [output_html_file]
```
