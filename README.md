# HyperAnalyze 📊

> **Historical Analytics Tool for Hyperliquid**  
> Advanced market microstructure analysis platform for Hyperliquid L1 DEX

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io)

**Live Demo:** [Coming Soon]

---

## 🎯 Overview

HyperAnalyze is a professional-grade market microstructure analysis platform built specifically for Hyperliquid. It provides deep insights into orderbook dynamics, wallet behavior, liquidation patterns, and market efficiency.

### Key Features

- 🔥 **Dynamic Order Book Heatmaps** - Visualize liquidity evolution over time
- 🔄 **Maker vs Taker Flow Analysis** - Track liquidity provision patterns
- 📏 **Multi-Asset Spread Analysis** - Compare market efficiency across coins
- 📊 **Volume Profile** - Identify fair value zones and support/resistance
- 👥 **Trader Analytics** - Identify whales and track top market participants
- 🎯 **Wallet Impact Analysis** - Discover which wallets move each orderbook
- 💥 **Liquidation Heatmaps** - Map dangerous price levels and liquidation zones
- 📋 **Interactive Data Explorer** - Dive deep into raw trade data

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ConejoCapital/hyperanalyze.git
cd hyperanalyze

# Install dependencies
pip3 install -r requirements.txt
```

### Run the Dashboard

```bash
# Launch the Streamlit app
streamlit run dashboard.py

# Or use the launcher script
./run_dashboard.sh
```

The dashboard will open at **http://localhost:8501**

---

## 📊 Features in Detail

### 1. Order Book Heatmap
Visualize market depth evolution across time and price levels:
- Time × Price × Depth 3D visualization
- Bid/ask imbalance indicators
- Support/resistance zone identification
- Liquidity gap detection

### 2. Maker vs Taker Flow
Analyze liquidity provision vs consumption:
- Stacked volume charts
- Maker ratio trends
- Fee economics analysis
- Market health indicators

### 3. Spread Analysis
Compare bid-ask spreads across multiple assets:
- Real-time spread tracking (basis points)
- Spread vs volume correlation
- Cross-asset efficiency comparison
- Market quality metrics

### 4. Volume Profile
Distribution of trading volume across price levels:
- Horizontal volume bars (buyer vs seller)
- Point of Control (POC) identification
- Value Area calculation (70% volume zone)
- High/Low Volume Nodes (HVN/LVN)

### 5. Trader Analytics
Track top market participants:
- Top 50 traders by volume
- Maker vs taker ratio analysis
- P&L tracking
- Individual trader drill-down
- Trading pattern identification

### 6. Wallet Impact Analysis ⭐
**Research Feature:** Identify which wallets move each orderbook:
- Volume-based ranking with taker ratio coloring
- Average price impact per trade
- Aggressive trading volume (orderbook consumption)
- Trade size distribution patterns
- **Answers:** "Who controls the orderbook?"

### 7. Liquidation Analysis ⭐
**Research Feature:** Map liquidation-prone price levels:
- Liquidation hotspot scatter plots
- Price-level concentration histograms
- Cumulative close volume tracking
- Close direction distribution
- **Answers:** "Where do liquidations happen?"

### 8. Data Explorer
Interactive data filtering and exploration:
- Coin-level summary statistics
- Advanced filtering by coin, side, time
- Raw trade data viewer
- Export capabilities

---

## 📁 Project Structure

```
hyperanalyze/
├── dashboard.py              # Main Streamlit application
├── data_loader.py            # Data pipeline & preprocessing
├── visualizations.py         # All visualization classes
├── requirements.txt          # Python dependencies
├── run_dashboard.sh          # Launch script
├── test_installation.py      # Installation verification
│
├── Hyperliquid Data Expanded/
│   ├── node_fills_*.json    # Trade execution data
│   └── misc_events_*.json   # Other blockchain events
│
├── Documentation/
│   ├── QUICK_START.md
│   ├── PHASE1_README.md
│   ├── RESEARCH_FEATURES.md
│   └── VISUALIZATION_RECOMMENDATIONS.md
│
└── processed_data.parquet   # Cached processed data (auto-generated)
```

---

## 🎓 Use Cases

### For Traders
- **Identify whale activity** - Track large market movers
- **Find liquidation zones** - Avoid getting liquidated
- **Analyze spread efficiency** - Choose liquid markets
- **Monitor maker/taker flow** - Gauge market sentiment

### For Researchers
- **Market microstructure studies** - Academic research
- **Orderbook concentration** - Decentralization metrics
- **Liquidity provision** - Market making analysis
- **Price discovery** - Information asymmetry research

### For Market Makers
- **Spread analysis** - Optimize quoting strategies
- **Volume profile** - Identify profitable price levels
- **Adverse selection** - Toxicity indicators
- **Competition analysis** - Track other MMs

### For Quants
- **Historical backtesting** - Strategy development
- **Market impact** - Execution cost analysis
- **Liquidity dynamics** - Depth evolution patterns
- **Correlation studies** - Cross-asset relationships

---

## 🔬 Research Questions Answered

✅ **"Who controls the BTC orderbook?"**  
→ Wallet Impact → Top 15 control X%

✅ **"Where will ETH liquidations trigger?"**  
→ Liquidations → Check histogram peaks

✅ **"Is this wallet a market manipulator?"**  
→ Wallet Impact + Trader Detail + P&L patterns

✅ **"What price levels have the most leverage?"**  
→ Liquidations → High-Risk Price Levels table

✅ **"Are liquidation cascades happening?"**  
→ Liquidations → Cumulative volume steep slopes

✅ **"Who has the biggest price impact per trade?"**  
→ Wallet Impact → Price impact scatter

---

## 📊 Data Format

HyperAnalyze works with Hyperliquid historical data:

### Supported Files
- `node_fills_*.json` - Trade execution events (primary data source)
- `misc_events_*.json` - Other blockchain events

### Data Schema
```json
{
  "local_time": "2025-10-27T17:00:00.063787823",
  "block_time": "2025-10-27T16:59:59.797563798",
  "block_number": 777010829,
  "events": [
    [
      "0x4264b5a132e4f263d6de2e0d01512a99ea21ec6e",
      {
        "coin": "ETH",
        "px": "4215.9",
        "sz": "0.3565",
        "side": "B",
        "crossed": true,
        "fee": "0.676335",
        "closedPnl": "0.0",
        ...
      }
    ]
  ]
}
```

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit (Python web framework)
- **Visualization:** Plotly (interactive charts)
- **Data Processing:** Pandas, NumPy
- **Storage:** Parquet (fast columnar format)
- **Performance:** Numba (JIT compilation)

---

## ⚙️ Configuration

### Data Path
Update in the sidebar or directly in `dashboard.py`:
```python
data_path = "Hyperliquid Data Expanded/node_fills_YYYYMMDD_HHMM-HHMM.json"
```

### Performance Settings
For large datasets, enable data limiting:
- Check "Limit data for testing" in sidebar
- Set max lines (e.g., 5,000-10,000 blocks)
- Loads in ~30 seconds vs 2-5 minutes

### Caching
First load processes raw JSON and creates `processed_data.parquet`:
- **First load:** 2-5 minutes (full dataset)
- **Subsequent loads:** 5-10 seconds (from cache)
- Delete cache file to reprocess data

---

## 📈 Sample Insights

From Oct 27, 2025 (17:00-18:00 UTC):
- **350K+ trades** across 187 coins
- **5,000+ unique traders**
- **Top 5 coins:** BTC ($7.4M), ETH ($2.2M), TRUMP ($1.4M), SOL ($918K), HYPE ($876K)
- **Top 20 traders** = 40-50% of all volume
- **Market concentration:** Varies by asset
- **Liquidation zones:** Identified at key price levels

---

## 🎯 Roadmap

### ✅ Phase 1: Complete
- Dynamic Order Book Heatmap
- Maker vs Taker Flow Analysis
- Spread Analysis Dashboard
- Volume Profile by Asset
- Trader Analytics
- Wallet Impact Analysis
- Liquidation Heatmaps

### 🚧 Phase 2: Advanced Analytics (Planned)
- Market impact visualization (Kyle's lambda)
- Order flow imbalance (OFI)
- Multi-asset correlation matrix
- Liquidity depth evolution
- Real-time WebSocket integration

### 🔮 Phase 3: Research-Grade (Future)
- Toxicity indicators
- 3D order book surface
- Hawkes process modeling
- Price level lifetime analysis
- Spoofing detection algorithms

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Install dev dependencies
pip3 install -r requirements.txt

# Run tests
python3 test_installation.py

# Launch development server
streamlit run dashboard.py --server.runOnSave=true
```

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Hyperliquid Docs:** https://hyperliquid.gitbook.io/hyperliquid-docs/
- **Hyperliquid GitHub:** https://github.com/hyperliquid-dex
- **Python SDK:** https://github.com/hyperliquid-dex/hyperliquid-python-sdk
- **Streamlit:** https://streamlit.io

---

## ⚠️ Disclaimer

This tool is for research and analysis purposes only. Not financial advice. Past performance does not indicate future results. Trade at your own risk.

---

## 📧 Contact

For questions, issues, or feature requests:
- **GitHub Issues:** https://github.com/ConejoCapital/hyperanalyze/issues
- **Repository:** https://github.com/ConejoCapital/hyperanalyze

---

## 🌟 Acknowledgments

Built with data from [Hyperliquid](https://hyperliquid.xyz), the performant L1 blockchain with a native DEX.

Special thanks to the Hyperliquid community for feedback and support.

---

**⭐ If you find this tool useful, please consider starring the repository!**

*Last updated: October 29, 2025*
