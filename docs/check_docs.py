#!/usr/bin/env python3
#!/usr/bin/env python3
# filepath: /home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs/check_docs.py

"""
Simple script to check documentation completeness
"""

import os
from pathlib import Path

def main():
    docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
    
    # Check main documentation files
    print("Checking main documentation files...")
    main_docs = [
        'index.md',
        'nifty500_trading_system_master_document.md',
        'comprehensive_pattern_guide_enhanced.md',
        'indicator_explanations_enhanced.md',
    ]
    
    for doc in main_docs:
        path = docs_dir / doc
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {doc} exists ({size/1024:.1f} KB)")
        else:
            print(f"❌ {doc} is missing")
    
    # Check PDF documentation
    print("\nChecking PDF documentation...")
    pdf_file = docs_dir / 'Nifty500_Trading_System_Complete_Enhanced.pdf'
    if pdf_file.exists():
        size = pdf_file.stat().st_size
        print(f"✅ Enhanced PDF exists ({size/1024/1024:.1f} MB)")
    else:
        print(f"❌ Enhanced PDF is missing")
    
    # Check image directories
    print("\nChecking image directories...")
    img_dir = docs_dir / 'images'
    if img_dir.exists():
        num_images = len(list(img_dir.glob('*.png')))
        print(f"✅ Main images directory exists with {num_images} images")
    else:
        print(f"❌ Main images directory is missing")
    
    patterns_dir = img_dir / 'patterns'
    if patterns_dir.exists():
        num_patterns = len(list(patterns_dir.glob('*.png')))
        print(f"✅ Pattern images directory exists with {num_patterns} pattern images")
    else:
        print(f"❌ Pattern images directory is missing")
    
    # Check for key pattern images
    print("\nChecking key pattern images...")
    key_patterns = [
        'hammer.png',
        'doji.png',
        'bullish_engulfing.png',
        'bearish_engulfing.png',
        'morning_star.png',
        'evening_star.png',
        'shooting_star.png',
    ]
    
    for pattern in key_patterns:
        path = patterns_dir / pattern
        if path.exists():
            print(f"✅ {pattern} exists")
        else:
            print(f"❌ {pattern} is missing")
    
    # Check for key indicator images
    print("\nChecking key indicator images...")
    key_indicators = [
        'macd.png',
        'bollinger_bands.png',
        'rsi.png',
        'moving_averages.png'
    ]
    
    for indicator in key_indicators:
        path = img_dir / indicator
        if path.exists():
            print(f"✅ {indicator} exists")
        else:
            print(f"❌ {indicator} is missing")
    
    print("\nDocumentation check completed.")

"""
Simple script to check documentation completeness
"""

import os
from pathlib import Path

def main():
    docs_dir = Path('/home/sashankhravi/Documents/stock_rl_agent_nifty_50/docs')
    
    # Check main documentation files
    print("Checking main documentation files...")
    main_docs = [
        'index.md',
        'nifty500_trading_system_master_document.md',
        'comprehensive_pattern_guide_enhanced.md',
        'indicator_explanations_enhanced.md',
    ]
    
    for doc in main_docs:
        path = docs_dir / doc
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {doc} exists ({size/1024:.1f} KB)")
        else:
            print(f"❌ {doc} is missing")
    
    # Check PDF documentation
    print("\nChecking PDF documentation...")
    pdf_file = docs_dir / 'Nifty500_Trading_System_Complete_Enhanced.pdf'
    if pdf_file.exists():
        size = pdf_file.stat().st_size
        print(f"✅ Enhanced PDF exists ({size/1024/1024:.1f} MB)")
    else:
        print(f"❌ Enhanced PDF is missing")
    
    # Check image directories
    print("\nChecking image directories...")
    img_dir = docs_dir / 'images'
    if img_dir.exists():
        num_images = len(list(img_dir.glob('*.png')))
        print(f"✅ Main images directory exists with {num_images} images")
    else:
        print(f"❌ Main images directory is missing")
    
    patterns_dir = img_dir / 'patterns'
    if patterns_dir.exists():
        num_patterns = len(list(patterns_dir.glob('*.png')))
        print(f"✅ Pattern images directory exists with {num_patterns} pattern images")
    else:
        print(f"❌ Pattern images directory is missing")
    
    # Check for key pattern images
    print("\nChecking key pattern images...")
    key_patterns = [
        'hammer.png',
        'doji.png',
        'bullish_engulfing.png',
        'bearish_engulfing.png',
        'morning_star.png',
        'evening_star.png',
        'shooting_star.png',
        'macd.png',
        'bollinger_bands.png'
    ]
    
    for pattern in key_patterns:
        path = patterns_dir / pattern
        if path.exists():
            print(f"✅ {pattern} exists")
        else:
            print(f"❌ {pattern} is missing")
    
    print("\nDocumentation check completed.")

if __name__ == "__main__":
    main()
