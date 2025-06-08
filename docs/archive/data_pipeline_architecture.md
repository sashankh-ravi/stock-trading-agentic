# Data Pipeline Architecture

The diagram below illustrates the complete Nifty 500 data download pipeline architecture:

```mermaid
flowchart TD
    A[Start: Nifty 500 Symbols List] --> B[Batch Processing]
    B --> C[download_stock_data.py]
    C --> D[Rate-Limited API Calls]
    D --> E[Raw OHLCV Data]
    E --> F[technical_indicators.py]
    F --> G[97+ Technical Indicators]
    E --> H[market_regime.py]
    H --> I[Market Regime Analysis]
    E --> J[relative_strength.py]
    J --> K[Relative Strength Metrics]
    G --> L[Enhanced Dataset]
    I --> L
    K --> L
    E --> M[Fundamental Data Extraction]
    M --> L
    E --> N[News Sentiment Analysis]
    N --> L
    L --> O[Data Quality Validation]
    O --> P[Save Processed Data]
    P --> Q[Generate Summary Reports]
    Q --> R[End: Production-Ready Dataset]

    style A fill:#f9d71c,stroke:#333,stroke-width:2px
    style B fill:#f9a11c,stroke:#333,stroke-width:2px
    style C fill:#f97a1c,stroke:#333,stroke-width:2px
    style E fill:#66c2a5,stroke:#333,stroke-width:2px
    style F fill:#8da0cb,stroke:#333,stroke-width:2px
    style G fill:#8da0cb,stroke:#333,stroke-width:2px
    style H fill:#fc8d62,stroke:#333,stroke-width:2px
    style I fill:#fc8d62,stroke:#333,stroke-width:2px
    style J fill:#e78ac3,stroke:#333,stroke-width:2px
    style K fill:#e78ac3,stroke:#333,stroke-width:2px
    style L fill:#a6d854,stroke:#333,stroke-width:2px
    style R fill:#33a02c,stroke:#333,stroke-width:2px
```

## Pipeline Components

1. **Input Processing**: Nifty 500 symbols are loaded and processed in batches to respect API rate limits
2. **Data Acquisition**: Raw price and volume data is downloaded for each stock
3. **Technical Analysis**: 97+ technical indicators are calculated across multiple categories
4. **Market Context**: Market regime and relative strength analysis add contextual information
5. **Fundamental Integration**: Company information and financial metrics are incorporated
6. **Data Validation**: Comprehensive quality checks ensure data integrity
7. **Output Generation**: Processed data is saved in multiple formats with summary reports

This architecture ensures complete coverage of the Nifty 500 index while maintaining high data quality and computational efficiency.
