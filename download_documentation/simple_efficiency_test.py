#!/usr/bin/env python3
import yfinance as yf

print("Testing RELIANCE.NS efficiency data availability...")

ticker = yf.Ticker("RELIANCE.NS")
info = ticker.info

# Check the key fields needed for efficiency calculations
fields_to_check = [
    'totalRevenue',
    'totalAssets', 
    'netReceivables',
    'inventory',
    'costOfRevenue'
]

print("\nChecking basic info fields:")
for field in fields_to_check:
    value = info.get(field)
    print(f"{field}: {value}")

# Try to calculate asset turnover
revenue = info.get('totalRevenue')
assets = info.get('totalAssets')

print(f"\nAsset Turnover calculation:")
print(f"Revenue: {revenue}")
print(f"Assets: {assets}")

if revenue and assets and assets != 0:
    asset_turnover = revenue / assets
    print(f"Asset Turnover: {asset_turnover:.2f}x")
else:
    print("Cannot calculate Asset Turnover - missing data")

print("\nTrying financial statements...")
print("Getting balance sheet...")
balance_sheet = ticker.balance_sheet
print(f"Balance sheet empty: {balance_sheet.empty}")

print("Getting financials...")
financials = ticker.financials  
print(f"Financials empty: {financials.empty}")
