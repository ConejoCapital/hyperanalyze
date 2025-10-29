# Hyperliquid Market Microstructure Dashboard - Phase 1

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the dashboard:**
```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Features

### Phase 1 Visualizations (All Implemented!)

1. **🔥 Dynamic Order Book Heatmap**
   - Time × Price × Depth visualization
   - Shows liquidity evolution across price levels
   - Order book imbalance indicator
   - Customizable time resolution and price bins

2. **🔄 Maker vs Taker Flow Analysis**
   - Stacked volume charts (maker vs taker)
   - Maker ratio evolution over time
   - Cumulative fee analysis
   - Trade count distribution

3. **📏 Spread Analysis Dashboard**
   - Multi-asset spread comparison
   - Spread vs volume correlation
   - Spread distribution histograms
   - Cross-asset spread metrics

4. **📊 Volume Profile by Asset**
   - Horizontal volume distribution (volume at price)
   - Buyer vs seller initiated volumes
   - Point of Control (POC) identification
   - Value Area calculation (70% volume zone)

5. **👥 Trader Analytics** (Account-Level)
   - Top traders by volume
   - Maker ratio distribution
   - Trade count vs volume analysis
   - Individual trader drill-down
   - P&L tracking and analysis

6. **📋 Data Explorer**
   - Interactive data filtering
   - Coin-level summary statistics
   - Raw trade data viewer

---

## 🎯 Key Insights You Can Extract

### Individual Ticker Level
- **Price dynamics**: How prices move across different liquidity zones
- **Spread behavior**: Tight vs wide spread periods
- **Volume concentration**: Where most trading occurs (POC)
- **Order flow**: Aggressive buying vs selling pressure

### Account Level (Who Moves the Market)
- **Top traders identification**: Ranked by total volume
- **Trading style**: Maker vs taker ratios reveal strategy
- **P&L performance**: Winners and losers
- **Coin preferences**: What each trader focuses on
- **Position patterns**: Net long/short tendencies

### Market Microstructure
- **Liquidity provision**: Maker activity indicates market health
- **Price impact**: How large trades affect prices
- **Bid-ask dynamics**: Spread widening during volatility
- **Order book imbalance**: Predictive signal for price direction

---

## 📁 File Structure

```
hyperorderbook/
├── data_loader.py           # Data loading & preprocessing pipeline
├── visualizations.py        # All Phase 1 visualization classes
├── dashboard.py             # Main Streamlit dashboard
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── VISUALIZATION_RECOMMENDATIONS.md  # Full recommendations doc
│
├── Hyperliquid Data Expanded/
│   ├── misc_events_20251027_1700-1800.json
│   └── node_fills_20251027_1700-1800.json
│
└── processed_data.parquet  # (Generated) Cached processed data
```

---

## 🔧 Usage Guide

### Basic Usage

1. **Launch the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

2. **First-time loading:**
   - The app will process the raw JSON data (takes 2-5 minutes for full dataset)
   - Processed data is cached as `processed_data.parquet` for faster subsequent loads
   - Progress shown in real-time

3. **Testing with limited data:**
   - Check "Limit data for testing" in sidebar
   - Set max lines (e.g., 10,000 blocks)
   - Faster for experimentation

### Navigation

**Sidebar:** Configure data settings and view overview metrics

**Tabs:**
- **Order Book Heatmap:** Select coin, time resolution, price bins
- **Maker vs Taker Flow:** Analyze liquidity provision patterns
- **Spread Analysis:** Compare up to 5 coins simultaneously
- **Volume Profile:** See volume distribution across price levels
- **Trader Analytics:** Identify top traders and drill into individual accounts
- **Data Explorer:** Browse raw data and summary statistics

---

## 🎨 Visualization Details

### 1. Order Book Heatmap
- **Purpose:** Understand market depth evolution
- **How to use:**
  - Select high-volume coin (ETH, BTC, etc.)
  - Adjust time resolution based on activity level
  - Look for:
    - Liquidity gaps (white spaces)
    - Support/resistance zones (concentrated color bands)
    - Bid/ask imbalance shifts

### 2. Maker vs Taker Flow
- **Purpose:** Assess market making activity
- **Interpretation:**
  - High maker ratio (>50%) = healthy liquidity provision
  - Taker spikes = aggressive order flow
  - Negative cumulative fees = net maker rebates received

### 3. Spread Analysis
- **Purpose:** Market efficiency indicator
- **Key metrics:**
  - Low spread (<5 bps) = tight, efficient market
  - Spread spike = volatility or low liquidity
  - Negative correlation with volume expected

### 4. Volume Profile
- **Purpose:** Identify fair value zones
- **How to read:**
  - POC (yellow line) = accepted price level
  - Value Area = 70% of volume (major acceptance zone)
  - Asymmetric buyer/seller bars = directional bias

### 5. Trader Analytics
- **Purpose:** Identify market movers and whales
- **Key insights:**
  - Top 20 traders often represent 50%+ of volume
  - Maker ratio reveals strategy:
    - >80% = pure market maker
    - 50-80% = mixed strategy
    - <50% = aggressive taker
  - P&L distribution shows skill vs luck

---

## 💡 Analysis Tips

### Finding Market Movers

1. Go to **Trader Analytics** tab
2. Sort by total volume
3. Check maker ratio:
   - Low ratio (<30%) = likely whale/informed trader
   - High ratio (>70%) = market maker
4. Click individual trader for detailed analysis

### Analyzing Specific Events

1. **Data Explorer** tab
2. Filter by coin and time
3. Look for:
   - Large trades (high notional)
   - Rapid sequences (potential algo)
   - Price level tests (repeated at same price)

### Comparing Assets

1. **Spread Analysis** tab
2. Select 3-5 related assets (e.g., ETH, BTC, SOL)
3. Compare:
   - Relative spread levels
   - Spread volatility
   - Volume-spread relationship

---

## ⚡ Performance Optimization

### First Load (Full Dataset ~45K blocks)
- **Expected time:** 2-5 minutes
- **Memory usage:** ~2-4 GB RAM
- **Output:** `processed_data.parquet` (~500 MB)

### Subsequent Loads
- **Expected time:** 5-10 seconds
- Loads from cached Parquet file

### Tips for Large Datasets
1. Use "Limit data for testing" during development
2. Delete `processed_data.parquet` if source data changes
3. For very large files (>100K blocks), consider:
   - Processing in chunks
   - Using Dask for parallel processing
   - Sampling for initial exploration

---

## 📊 Sample Insights from Oct 27, 2025 Data

Based on the 17:00-18:00 UTC hour:

- **Total trades:** ~500K-1M events
- **Unique traders:** Thousands of active addresses
- **Most active coins:** ETH, BTC, HYPE, IOTA, USUAL, etc.
- **Maker/Taker split:** Varies by asset (typically 40-60%)

---

## 🔮 Next Steps (Phase 2 & 3)

### Phase 2: Advanced Analytics
- Market impact visualization (Kyle's lambda)
- Order flow imbalance (OFI)
- Multi-asset correlation matrix
- Liquidity depth evolution

### Phase 3: Research-Grade
- Toxicity indicators
- 3D visualizations
- Hawkes process modeling
- Price level lifetime analysis

See `VISUALIZATION_RECOMMENDATIONS.md` for full roadmap.

---

## 🛠️ Customization

### Adding New Visualizations

1. Create new class in `visualizations.py`:
   ```python
   class MyNewViz:
       def __init__(self, df: pd.DataFrame):
           self.df = df
       
       def create_viz(self) -> go.Figure:
           # Your visualization code
           pass
   ```

2. Add to `dashboard.py`:
   ```python
   with tab:
       viz = MyNewViz(df)
       fig = viz.create_viz()
       st.plotly_chart(fig)
   ```

### Modifying Data Loader

Edit `data_loader.py` to add:
- New computed fields
- Additional preprocessing
- Custom aggregations

---

## 📚 References

- **Hyperliquid Docs:** https://hyperliquid.gitbook.io/hyperliquid-docs/
- **GitHub:** https://github.com/hyperliquid-dex
- **Python SDK:** https://github.com/hyperliquid-dex/hyperliquid-python-sdk

---

## 🐛 Troubleshooting

### Dashboard won't start
```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

### Out of memory
- Enable "Limit data for testing"
- Reduce max lines to 5,000-10,000
- Close other applications

### Slow performance
- Ensure `processed_data.parquet` exists
- Check if antivirus is scanning files
- Try reducing visualization complexity (fewer price bins, longer time resolution)

### Data not loading
- Verify data path in sidebar
- Check file permissions
- Ensure JSON is not corrupted

---

## 📧 Support

For issues related to:
- **Hyperliquid data:** Check official docs
- **Dashboard bugs:** Review code and error messages
- **Feature requests:** See Phase 2/3 roadmap

---

## 📄 License

This is a prototype dashboard for market microstructure analysis. Customize as needed for your research or trading operations.

---

**Built with ❤️ for Hyperliquid market analysis**

*Last updated: October 29, 2025*

