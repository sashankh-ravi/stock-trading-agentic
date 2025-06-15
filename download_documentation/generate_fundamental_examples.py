#!/usr/bin/env python3
"""
Fundamental Analysis Visua    # 1. VALUATION METRICS (8 metrics) with detailed explanations  
    print("\\n📊 VALUATION METRICS:")
    print("   📅 Update Frequency: Real-time for ratios (as stock price changes), Quarterly for fundamentals")
    print("   💾 Data Storage: Daily stock price + quarterly financial data = daily updated ratios")
    
    pe_ratio = info.get('trailingPE')
    fundamental_data['pe_ratio'] = pe_ratio
    if pe_ratio:
        print(f"   P/E Ratio: {pe_ratio:.2f}")
        print(f"   Formula: Market Price per Share ÷ Earnings per Share")
        print(f"   Interpretation: Investors pay ₹{pe_ratio:.2f} for every ₹1 of earnings")
        print(f"   📊 Calculation: Current stock price ÷ TTM EPS (updated daily)")Generator

This script generates comprehensive visualizations and examples for the 
fundamental analysis documentation, including:
- Financial metrics calculations
- Real examples from RELIANCE.NS
- Interactive plots and charts
- Mathematical formulas and explanations

Author: Financial Data Analysis Pipeline
Date: June 15, 2025
"""

import sys
import os

# Add the parent directory to the path to import from download_nifty500_data.py  
sys.path.insert(0, '/home/sashankhravi/Documents/stock-trading-agentic')

from download_nifty500_data import get_fundamental_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_fundamental_data_with_examples(symbol: str = "RELIANCE.NS") -> dict:
    """
    Fetch fundamental data using the actual function from download_nifty500_data.py
    and add detailed examples for documentation
    """
    print(f"🔍 Analyzing fundamental data for {symbol}")
    
    # Use the actual function from your main code
    fundamental_data = get_fundamental_data(symbol)
    
    if not fundamental_data:
        print(f"❌ No data available for {symbol}")
        return {}
    
    # Display the data with explanations
    print("\\n📊 VALUATION METRICS:")
    
    pe_ratio = fundamental_data.get('pe_ratio')
    if pe_ratio:
        print(f"   P/E Ratio: {pe_ratio:.2f}")
        print(f"   Formula: Market Price per Share ÷ Earnings per Share")
        print(f"   Interpretation: Investors pay ₹{pe_ratio:.2f} for every ₹1 of earnings")
    
    pb_ratio = fundamental_data.get('pb_ratio')
    if pb_ratio:
        print(f"   P/B Ratio: {pb_ratio:.2f}")
        print(f"   Formula: Market Price per Share ÷ Book Value per Share")
        print(f"   Interpretation: Stock trades at {pb_ratio:.2f}x its book value")
    
    market_cap = fundamental_data.get('market_cap')
    if market_cap:
        market_cap_cr = market_cap / 10000000  # Convert to crores
        print(f"   Market Cap: ₹{market_cap_cr:,.0f} crores")
        print(f"   Formula: Shares Outstanding × Current Stock Price")
    
    enterprise_value = fundamental_data.get('enterprise_value')
    if enterprise_value:
        ev_cr = enterprise_value / 10000000
        print(f"   Enterprise Value: ₹{ev_cr:,.0f} crores")
        print(f"   Formula: Market Cap + Total Debt - Cash and Cash Equivalents")
    
    ev_ebitda = fundamental_data.get('ev_ebitda')
    if ev_ebitda:
        print(f"   EV/EBITDA: {ev_ebitda:.2f}")
        print(f"   Formula: Enterprise Value ÷ EBITDA")
        print(f"   Interpretation: Company valued at {ev_ebitda:.2f}x its annual EBITDA")
    
    # PROFITABILITY METRICS
    print("\\n💰 PROFITABILITY METRICS:")
    print("   📅 Update Frequency: Quarterly (when financial statements are released)")
    print("   💾 Data Storage: One value per quarter, using TTM (Trailing Twelve Months) data")
    
    profit_margin = fundamental_data.get('profit_margin')
    if profit_margin:
        print(f"   Profit Margin: {profit_margin:.2f}%")
        print(f"   Formula: (Net Income ÷ Total Revenue) × 100")
        print(f"   Interpretation: Company keeps {profit_margin:.2f}¢ as profit for every ₹1 of revenue")
        print(f"   📊 Calculation: Uses TTM net income and TTM revenue")
    
    roe = fundamental_data.get('roe')
    if roe:
        print(f"   Return on Equity (ROE): {roe:.2f}%")
        print(f"   Formula: (Net Income ÷ Shareholders' Equity) × 100")
        print(f"   Interpretation: Generates {roe:.2f}% return on shareholders' investments")
    
    roa = fundamental_data.get('roa')
    if roa:
        print(f"   Return on Assets (ROA): {roa:.2f}%")
        print(f"   Formula: (Net Income ÷ Total Assets) × 100")
        print(f"   Interpretation: Generates {roa:.2f}% return on total assets")
    
    operating_margin = fundamental_data.get('operating_margin')
    if operating_margin:
        print(f"   Operating Margin: {operating_margin:.2f}%")
        print(f"   Formula: (Operating Income ÷ Total Revenue) × 100")
    
    gross_margin = fundamental_data.get('gross_margin')
    if gross_margin:
        print(f"   Gross Margin: {gross_margin:.2f}%")
        print(f"   Formula: ((Revenue - COGS) ÷ Revenue) × 100")
    
    ebitda_margin = fundamental_data.get('ebitda_margin')
    if ebitda_margin:
        print(f"   EBITDA Margin: {ebitda_margin:.2f}%")
        print(f"   Formula: (EBITDA ÷ Total Revenue) × 100")
        print(f"   EBITDA = Earnings Before Interest, Taxes, Depreciation, Amortization")
    
    fcf_margin = fundamental_data.get('fcf_margin')
    if fcf_margin:
        print(f"   Free Cash Flow Margin: {fcf_margin:.2f}%")
        print(f"   Formula: (Free Cash Flow ÷ Total Revenue) × 100")
    
    # FINANCIAL HEALTH METRICS
    print("\\n🏥 FINANCIAL HEALTH METRICS:")
    
    debt_to_equity = fundamental_data.get('debt_to_equity')
    if debt_to_equity:
        print(f"   Debt-to-Equity Ratio: {debt_to_equity:.2f}")
        print(f"   Formula: Total Debt ÷ Total Shareholders' Equity")
        print(f"   Interpretation: Company has ₹{debt_to_equity:.2f} of debt for every ₹1 of equity")
    
    current_ratio = fundamental_data.get('current_ratio')
    if current_ratio:
        print(f"   Current Ratio: {current_ratio:.2f}")
        print(f"   Formula: Current Assets ÷ Current Liabilities")
        print(f"   Interpretation: Company has ₹{current_ratio:.2f} in current assets for every ₹1 of current liabilities")
    
    quick_ratio = fundamental_data.get('quick_ratio')
    if quick_ratio:
        print(f"   Quick Ratio: {quick_ratio:.2f}")
        print(f"   Formula: (Current Assets - Inventory) ÷ Current Liabilities")
        print(f"   Interpretation: Measures liquidity without relying on inventory conversion")
    
    # GROWTH METRICS
    print("\\n📈 GROWTH METRICS:")
    
    revenue_growth = fundamental_data.get('revenue_growth')
    if revenue_growth:
        print(f"   Revenue Growth: {revenue_growth:.2f}%")
        print(f"   Formula: ((Current Revenue - Previous Revenue) ÷ Previous Revenue) × 100")
    
    earnings_growth = fundamental_data.get('earnings_growth')
    if earnings_growth:
        print(f"   Earnings Growth: {earnings_growth:.2f}%")
        print(f"   Formula: ((Current Earnings - Previous Earnings) ÷ Previous Earnings) × 100")
    
    # EFFICIENCY METRICS
    print("\\n⚡ EFFICIENCY METRICS:")
    print("   📅 Update Frequency: Quarterly (when financial statements are released)")
    print("   💾 Data Storage: One value per quarter, calculated from annual/quarterly financials")
    
    asset_turnover = fundamental_data.get('asset_turnover')
    if asset_turnover:
        print(f"   Asset Turnover: {asset_turnover:.2f}")
        print(f"   Formula: Total Revenue ÷ Total Assets")
        print(f"   Interpretation: Generates ₹{asset_turnover:.2f} in revenue for every ₹1 of assets")
        print(f"   📊 Calculation: Uses TTM (Trailing Twelve Months) revenue and latest quarter assets")
    
    receivables_turnover = fundamental_data.get('receivables_turnover')
    if receivables_turnover:
        print(f"   Receivables Turnover: {receivables_turnover:.2f}x")
        print(f"   📊 Calculation: TTM revenue ÷ current quarter receivables")
    
    inventory_turnover = fundamental_data.get('inventory_turnover')
    if inventory_turnover:
        print(f"   Inventory Turnover: {inventory_turnover:.2f}x")
        print(f"   📊 Calculation: TTM cost of goods sold ÷ current quarter inventory")
    
    # DIVIDEND METRICS
    print("\\n💸 DIVIDEND METRICS:")
    
    dividend_yield = fundamental_data.get('dividend_yield')
    if dividend_yield:
        print(f"   Dividend Yield: {dividend_yield:.2f}%")
        print(f"   Formula: (Annual Dividend per Share ÷ Stock Price) × 100")
        print(f"   Interpretation: Provides {dividend_yield:.2f}% annual return through dividends")
    
    payout_ratio = fundamental_data.get('payout_ratio')
    if payout_ratio:
        print(f"   Payout Ratio: {payout_ratio:.2f}%")
        print(f"   Formula: (Dividends per Share ÷ Earnings per Share) × 100")
    
    # PER SHARE METRICS
    print("\\n📊 PER SHARE METRICS:")
    
    shares_outstanding = fundamental_data.get('shares_outstanding')
    if shares_outstanding:
        shares_cr = shares_outstanding / 10000000  # Convert to crores
        print(f"   Shares Outstanding: {shares_cr:.2f} crores")
    
    eps = fundamental_data.get('eps')
    if eps:
        print(f"   Earnings per Share (EPS): ₹{eps:.2f}")
        print(f"   Formula: Net Income ÷ Shares Outstanding")
    
    book_value_per_share = fundamental_data.get('book_value_per_share')
    if book_value_per_share:
        print(f"   Book Value per Share: ₹{book_value_per_share:.2f}")
        print(f"   Formula: Total Shareholders' Equity ÷ Shares Outstanding")
    
    fcf_per_share = fundamental_data.get('fcf_per_share')
    if fcf_per_share:
        print(f"   Free Cash Flow per Share: ₹{fcf_per_share:.2f}")
        print(f"   Formula: Free Cash Flow ÷ Shares Outstanding")
    
    return fundamental_data

def create_fundamental_metrics_visualization(data: dict, symbol: str = "RELIANCE.NS"):
    """
    Create comprehensive visualizations for fundamental metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Fundamental Analysis Dashboard - {symbol}', fontsize=16, fontweight='bold')
    
    # 1. Profitability Metrics
    profitability_metrics = {
        'Profit Margin': data.get('profit_margin', 0),
        'Operating Margin': data.get('operating_margin', 0),
        'Gross Margin': data.get('gross_margin', 0),
        'ROE': data.get('roe', 0),
        'ROA': data.get('roa', 0)
    }
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(profitability_metrics.keys(), profitability_metrics.values(), 
                   color=['#2E8B57', '#32CD32', '#90EE90', '#FF6347', '#FF4500'])
    ax1.set_title('Profitability Metrics (%)', fontweight='bold')
    ax1.set_ylabel('Percentage (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. Financial Health Ratios
    interest_cov = data.get('interest_coverage', 0)
    interest_cov_safe = min(interest_cov, 20) if interest_cov is not None else 0
    
    health_metrics = {
        'Current Ratio': data.get('current_ratio', 0) or 0,
        'Quick Ratio': data.get('quick_ratio', 0) or 0,
        'D/E Ratio': data.get('debt_to_equity', 0) or 0,
        'Interest Coverage': interest_cov_safe
    }
    
    ax2 = axes[0, 1]
    bars2 = ax2.bar(health_metrics.keys(), health_metrics.values(),
                   color=['#4169E1', '#6495ED', '#FF6347', '#32CD32'])
    ax2.set_title('Financial Health Ratios', fontweight='bold')
    ax2.set_ylabel('Ratio')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
    
    # 3. Valuation Metrics
    pe_safe = min(data.get('pe_ratio', 0) or 0, 50)
    ev_ebitda_safe = min(data.get('ev_ebitda', 0) or 0, 30)
    
    valuation_metrics = {
        'P/E Ratio': pe_safe,
        'P/B Ratio': data.get('pb_ratio', 0) or 0,
        'EV/EBITDA': ev_ebitda_safe
    }
    
    ax3 = axes[0, 2]
    bars3 = ax3.bar(valuation_metrics.keys(), valuation_metrics.values(),
                   color=['#9370DB', '#8A2BE2', '#9932CC'])
    ax3.set_title('Valuation Metrics', fontweight='bold')
    ax3.set_ylabel('Multiple')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}x', ha='center', va='bottom')
    
    # 4. Growth Metrics
    growth_metrics = {
        'Revenue Growth': data.get('revenue_growth', 0) or 0,
        'Earnings Growth': data.get('earnings_growth', 0) or 0,
        'EBITDA Growth': data.get('ebitda_growth', 0) or 0,
        'FCF Growth': data.get('fcf_growth', 0) or 0
    }
    
    ax4 = axes[1, 0]
    colors = ['green' if x >= 0 else 'red' for x in growth_metrics.values()]
    bars4 = ax4.bar(growth_metrics.keys(), growth_metrics.values(), color=colors, alpha=0.7)
    ax4.set_title('Growth Metrics (%)', fontweight='bold')
    ax4.set_ylabel('Growth Rate (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    for bar in bars4:
        height = bar.get_height()
        if abs(height) > 0.1:
            ax4.text(bar.get_x() + bar.get_width()/2., 
                    height + (1 if height >= 0 else -2),
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    # 5. Efficiency Metrics
    efficiency_data = []
    efficiency_labels = []
    
    if data.get('asset_turnover'):
        efficiency_data.append(data['asset_turnover'])
        efficiency_labels.append('Asset Turnover')
    
    if data.get('receivables_turnover'):
        rt_val = data['receivables_turnover']
        if rt_val is not None:
            efficiency_data.append(min(rt_val, 50))  # Cap for visualization
            efficiency_labels.append('Receivables Turnover')
    
    if data.get('inventory_turnover'):
        it_val = data['inventory_turnover']
        if it_val is not None:
            efficiency_data.append(min(it_val, 50))  # Cap for visualization
            efficiency_labels.append('Inventory Turnover')
    
    ax5 = axes[1, 1]
    if efficiency_data:
        bars5 = ax5.bar(efficiency_labels, efficiency_data, color=['#FF8C00', '#FFA500', '#FFD700'])
        ax5.set_title('Efficiency Metrics', fontweight='bold')
        ax5.set_ylabel('Turnover Ratio')
        ax5.tick_params(axis='x', rotation=45)
        
        for bar in bars5:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}x', ha='center', va='bottom')
    else:
        ax5.text(0.5, 0.5, 'Efficiency Data\nNot Available', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Efficiency Metrics', fontweight='bold')
    
    # 6. Per Share Metrics
    per_share_metrics = {}
    per_share_labels = []
    
    if data.get('eps'):
        per_share_metrics['EPS'] = data['eps']
        per_share_labels.append('EPS (₹)')
    
    if data.get('book_value_per_share'):
        per_share_metrics['Book Value'] = data['book_value_per_share']
        per_share_labels.append('Book Value (₹)')
    
    if data.get('fcf_per_share'):
        per_share_metrics['FCF'] = data['fcf_per_share']
        per_share_labels.append('FCF (₹)')
    
    ax6 = axes[1, 2]
    
    if per_share_metrics:
        values = list(per_share_metrics.values())
        labels = list(per_share_metrics.keys())
        
        bars6 = ax6.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax6.set_title('Per Share Metrics (₹)', fontweight='bold')
        ax6.set_ylabel('Value (₹)')
        
        # Add value labels on bars
        for bar in bars6:
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'₹{height:.2f}', ha='center', va='bottom')
    else:
        ax6.text(0.5, 0.5, 'No Per Share\nData Available', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Per Share Metrics', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/sashankhravi/Documents/stock-trading-agentic/download_documentation/fundamental_analysis_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return "Dashboard created successfully"

def create_metric_explanation_table():
    """
    Create a detailed table of all fundamental metrics with explanations
    """
    metrics_data = {
        'Category': ['Valuation', 'Valuation', 'Valuation', 'Valuation', 'Valuation',
                    'Profitability', 'Profitability', 'Profitability', 'Profitability', 'Profitability',
                    'Financial Health', 'Financial Health', 'Financial Health', 'Financial Health',
                    'Growth', 'Growth', 'Growth', 'Growth',
                    'Efficiency', 'Efficiency', 'Efficiency',
                    'Dividend', 'Dividend', 'Dividend',
                    'Per Share', 'Per Share', 'Per Share', 'Per Share'],
        
        'Metric': ['P/E Ratio', 'P/B Ratio', 'PEG Ratio', 'EV/EBITDA', 'EV/Revenue',
                  'Profit Margin', 'Operating Margin', 'Gross Margin', 'ROE', 'ROA',
                  'Debt-to-Equity', 'Current Ratio', 'Quick Ratio', 'Interest Coverage',
                  'Revenue Growth', 'Earnings Growth', 'EBITDA Growth', 'FCF Growth',
                  'Asset Turnover', 'Receivables Turnover', 'Inventory Turnover',
                  'Dividend Yield', 'Payout Ratio', 'Dividend Growth',
                  'EPS', 'Book Value/Share', 'FCF/Share', 'Shares Outstanding'],
        
        'Formula': ['Market Price ÷ EPS', 'Market Price ÷ Book Value per Share', 'P/E Ratio ÷ Growth Rate',
                   'Enterprise Value ÷ EBITDA', 'Enterprise Value ÷ Revenue',
                   '(Net Income ÷ Revenue) × 100', '(Operating Income ÷ Revenue) × 100',
                   '((Revenue - COGS) ÷ Revenue) × 100', '(Net Income ÷ Equity) × 100',
                   '(Net Income ÷ Total Assets) × 100',
                   'Total Debt ÷ Total Equity', 'Current Assets ÷ Current Liabilities',
                   '(Current Assets - Inventory) ÷ Current Liabilities', 'EBIT ÷ Interest Expense',
                   '((Current Revenue - Previous Revenue) ÷ Previous Revenue) × 100',
                   '((Current Earnings - Previous Earnings) ÷ Previous Earnings) × 100',
                   '((Current EBITDA - Previous EBITDA) ÷ Previous EBITDA) × 100',
                   '((Current FCF - Previous FCF) ÷ Previous FCF) × 100',
                   'Revenue ÷ Total Assets', 'Revenue ÷ Accounts Receivable',
                   'COGS ÷ Average Inventory',
                   '(Annual Dividend ÷ Stock Price) × 100', '(Dividend ÷ EPS) × 100',
                   '((Current Dividend - Previous Dividend) ÷ Previous Dividend) × 100',
                   'Net Income ÷ Shares Outstanding', 'Shareholders Equity ÷ Shares Outstanding',
                   'Free Cash Flow ÷ Shares Outstanding', 'Total number of shares issued'],
        
        'Good Range': ['10-20', '1-3', '<1', '8-15', '1-3',
                      '>10%', '>15%', '>20%', '>15%', '>10%',
                      '<1', '>1.5', '>1', '>5', '>10%', '>10%', '>10%', '>10%',
                      '>1', '>6', '>4',
                      '2-6%', '30-60%', '>5%',
                      'Positive', 'Positive', 'Positive', 'Market dependent']
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Create formatted table
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code by category
    colors = {'Valuation': '#E6F3FF', 'Profitability': '#E6FFE6', 'Financial Health': '#FFE6E6',
              'Growth': '#FFF0E6', 'Efficiency': '#F0E6FF', 'Dividend': '#FFFFE6', 'Per Share': '#F0F0F0'}
    
    for i, category in enumerate(df['Category']):
        for j in range(len(df.columns)):
            table[(i+1, j)].set_facecolor(colors.get(category, '#FFFFFF'))
    
    # Header styling
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Comprehensive Fundamental Metrics Reference Table', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('/home/sashankhravi/Documents/stock-trading-agentic/download_documentation/fundamental_metrics_table.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to generate all visualizations and examples
    """
    print("🚀 Generating Fundamental Analysis Documentation Materials")
    print("="*60)
    
    # Get fundamental data with detailed examples
    fundamental_data = get_fundamental_data_with_examples("RELIANCE.NS")
    
    print("\\n" + "="*60)
    print("📊 Creating Visualizations...")
    
    # Create comprehensive dashboard
    result = create_fundamental_metrics_visualization(fundamental_data, "RELIANCE.NS")
    
    # Create metrics reference table
    create_metric_explanation_table()
    
    print(f"\\n✅ Documentation materials generated successfully!")
    print(f"📊 Dashboard created for RELIANCE.NS")
    print(f"📁 Files saved in: /home/sashankhravi/Documents/stock-trading-agentic/download_documentation/")
    print("   - fundamental_analysis_dashboard.png")
    print("   - fundamental_metrics_table.png")

if __name__ == "__main__":
    main()
