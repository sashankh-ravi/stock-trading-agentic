#!/usr/bin/env python3
"""
Debug script to check what efficiency data is available for RELIANCE.NS
This will help determine if the empty efficiency plot is due to data unavailability 
or a code issue.
"""

import yfinance as yf
import pandas as pd

def debug_efficiency_data(symbol="RELIANCE.NS"):
    """
    Comprehensive debugging of efficiency-related data availability
    """
    print(f"🔍 DEBUGGING EFFICIENCY DATA FOR {symbol}")
    print("="*60)
    
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    print("\n📊 RAW DATA AVAILABILITY CHECK:")
    print("-" * 40)
    
    # Check all possible data fields that could be used for efficiency calculations
    efficiency_fields = {
        'totalRevenue': 'Total Revenue',
        'totalAssets': 'Total Assets', 
        'netReceivables': 'Net Receivables',
        'inventory': 'Inventory',
        'costOfRevenue': 'Cost of Revenue (COGS)',
        'totalCurrentAssets': 'Current Assets',
        'totalCurrentLiabilities': 'Current Liabilities',
        'accountsReceivable': 'Accounts Receivable',
        'totalCash': 'Total Cash'
    }
    
    available_data = {}
    
    for field, description in efficiency_fields.items():
        value = info.get(field)
        available_data[field] = value
        status = "✅ Available" if value is not None else "❌ Not Available"
        print(f"{description:25}: {status:15} | Value: {value}")
    
    print("\n🧮 EFFICIENCY CALCULATIONS ATTEMPT:")
    print("-" * 40)
    
    # 1. Asset Turnover = Revenue / Total Assets
    revenue = available_data.get('totalRevenue')
    total_assets = available_data.get('totalAssets')
    
    if revenue and total_assets and total_assets != 0:
        asset_turnover = revenue / total_assets
        print(f"✅ Asset Turnover: {asset_turnover:.2f}x")
        print(f"   Formula: {revenue:,} ÷ {total_assets:,} = {asset_turnover:.2f}")
    else:
        print(f"❌ Asset Turnover: Cannot calculate")
        print(f"   Revenue: {revenue}, Total Assets: {total_assets}")
    
    # 2. Receivables Turnover = Revenue / Net Receivables
    net_receivables = available_data.get('netReceivables')
    accounts_receivable = available_data.get('accountsReceivable')
    
    receivables = net_receivables or accounts_receivable
    
    if revenue and receivables and receivables != 0:
        receivables_turnover = revenue / receivables
        print(f"✅ Receivables Turnover: {receivables_turnover:.2f}x")
        print(f"   Formula: {revenue:,} ÷ {receivables:,} = {receivables_turnover:.2f}")
    else:
        print(f"❌ Receivables Turnover: Cannot calculate")
        print(f"   Revenue: {revenue}, Receivables: {receivables}")
    
    # 3. Inventory Turnover = COGS / Inventory
    cogs = available_data.get('costOfRevenue')
    inventory = available_data.get('inventory')
    
    if cogs and inventory and inventory != 0:
        inventory_turnover = cogs / inventory
        print(f"✅ Inventory Turnover: {inventory_turnover:.2f}x")
        print(f"   Formula: {cogs:,} ÷ {inventory:,} = {inventory_turnover:.2f}")
    else:
        print(f"❌ Inventory Turnover: Cannot calculate")
        print(f"   COGS: {cogs}, Inventory: {inventory}")
    
    print("\n📈 ALTERNATIVE DATA SOURCES CHECK:")
    print("-" * 40)
    
    # Check financial statements directly (without try-except to see actual errors)
    # Try to get quarterly financials
    quarterly_financials = ticker.quarterly_financials
    annual_financials = ticker.financials
    quarterly_balance_sheet = ticker.quarterly_balance_sheet
    annual_balance_sheet = ticker.balance_sheet
    
    print(f"Quarterly Financials: {'✅ Available' if not quarterly_financials.empty else '❌ Empty'}")
    print(f"Annual Financials: {'✅ Available' if not annual_financials.empty else '❌ Empty'}")
    print(f"Quarterly Balance Sheet: {'✅ Available' if not quarterly_balance_sheet.empty else '❌ Empty'}")
    print(f"Annual Balance Sheet: {'✅ Available' if not annual_balance_sheet.empty else '❌ Empty'}")
    
    if not annual_balance_sheet.empty and not annual_financials.empty:
        print("\n🔍 DIRECT FINANCIAL STATEMENT DATA:")
        print("-" * 30)
        
        # Check if we can find efficiency data in financial statements
        balance_sheet_items = annual_balance_sheet.index.tolist()
        income_items = annual_financials.index.tolist()
        
        print("📋 Available Balance Sheet Items:")
        for item in balance_sheet_items[:10]:  # Show first 10 items
            print(f"   - {item}")
        if len(balance_sheet_items) > 10:
            print(f"   ... and {len(balance_sheet_items) - 10} more items")
            
        print("\n📋 Available Income Statement Items:")
        for item in income_items[:10]:  # Show first 10 items
            print(f"   - {item}")
        if len(income_items) > 10:
            print(f"   ... and {len(income_items) - 10} more items")
    
    print("\n💡 CONCLUSIONS:")
    print("-" * 40)
    
    efficiency_available = 0
    if revenue and total_assets and total_assets != 0:
        efficiency_available += 1
    if revenue and receivables and receivables != 0:
        efficiency_available += 1
    if cogs and inventory and inventory != 0:
        efficiency_available += 1
        
    if efficiency_available == 0:
        print("❌ NO EFFICIENCY METRICS CAN BE CALCULATED")
        print("   This confirms the empty plot is due to data unavailability, not code error")
        print("   Yahoo Finance may not provide sufficient balance sheet data for RELIANCE.NS")
    elif efficiency_available < 3:
        print(f"⚠️  PARTIAL EFFICIENCY DATA: {efficiency_available}/3 metrics available")
        print("   Some efficiency metrics can be calculated, but not all")
    else:
        print("✅ ALL EFFICIENCY METRICS CAN BE CALCULATED")
        print("   If plot is empty, this indicates a code issue")
    
    return available_data

if __name__ == "__main__":
    # Test with RELIANCE.NS
    print("Testing with RELIANCE.NS...")
    reliance_data = debug_efficiency_data("RELIANCE.NS")
    
    print("\n" + "="*60)
    
    # Test with a US stock for comparison
    print("\nTesting with AAPL (US stock for comparison)...")
    aapl_data = debug_efficiency_data("AAPL")
