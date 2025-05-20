import re

filename = 'nifty500_trading_system_master_document.md'
with open(filename, 'r') as f:
    content = f.read()

# Check for headings
print("Checking headings...")
headings = re.findall(r'^(#{1,6} .+)$', content, re.MULTILINE)
heading_counts = {}
for heading in headings:
    if heading in heading_counts:
        heading_counts[heading] += 1
    else:
        heading_counts[heading] = 1

print("Duplicate headings:")
for heading, count in heading_counts.items():
    if count > 1:
        print(f"- {heading} (appears {count} times)")

# Check for link fragments
print("\nChecking link fragments...")
links = re.findall(r'\]\(#([^\)]+)\)', content)
print(f"Total link fragments: {len(links)}")

# Create list of valid fragment targets
valid_fragments = []
for heading in headings:
    # Convert heading to fragment target format
    slug = heading.lower()
    # Remove #s at the beginning
    slug = re.sub(r'^#+\s+', '', slug)
    # Replace spaces with dashes
    slug = slug.replace(' ', '-')
    # Remove non-alphanumeric characters except dashes
    slug = re.sub(r'[^a-z0-9\-]', '', slug)
    valid_fragments.append(slug)

invalid_links = []
for link in links:
    if link not in valid_fragments:
        invalid_links.append(link)

print(f"Invalid link fragments: {len(invalid_links)}")
for link in invalid_links[:20]:  # Limit to 20 to avoid excessive output
    print(f"- #{link}")

# Check for list indentation and spacing issues
print("\nChecking list formatting...")
list_items = []
in_code_block = False

# Process line by line for more accurate detection
for line in content.splitlines():
    if line.strip().startswith("```"):
        in_code_block = not in_code_block
        continue
        
    if in_code_block:
        continue
    
    # Check for list marker at the beginning of non-empty line
    list_match = re.match(r'^(\s*)([-*+]|\d+\.)(\s+)', line)
    if list_match and line.strip():
        indent_spaces = len(list_match.group(1))
        list_items.append((indent_spaces, line))

print(f"Total list items: {len(list_items)}")

# Check for inconsistent indentation
indentation_counts = {}
for spaces, line in list_items:
    if spaces in indentation_counts:
        indentation_counts[spaces] += 1
    else:
        indentation_counts[spaces] = 1

print("List indentation levels:")
for spaces, count in sorted(indentation_counts.items()):
    print(f"- {spaces} spaces: {count} items")
