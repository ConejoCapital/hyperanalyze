# 📖 Order Book Analysis Plan

## 🎯 Overview

This document outlines the comprehensive plan for building advanced order book visualizations using the **50 million order events** dataset (`Hyperliquid_orderbooks.json.gz`).

**Data Coverage:**
- **Time Period:** October 27, 2025, 17:00-18:00 (1 hour)
- **Events:** ~50,000,000 order book events
- **Size:** 430MB compressed, ~3-4GB uncompressed
- **Scope:** All limit orders (placements, fills, cancels, rejections)

---

## 📊 Data Structure

```json
{
  "BLOCK_NUMBER": "777010829",
  "DATETIME": "2025-10-27 16:59:59.797",
  "VALUE": {
    "order": {
      "coin": "BTC",
      "side": "B",              // B = bid, A = ask
      "limitPx": "115554.0",    // Price
      "sz": "0.29483",          // Current size
      "origSz": "0.29483",      // Original size
      "oid": 214107582267,      // Order ID (unique)
      "timestamp": 1761584399797,
      "tif": "Alo",             // Time in Force
      "orderType": "Limit",
      "cloid": "...",           // Client order ID
      "isPositionTpsl": false,
      "isTrigger": false,
      "reduceOnly": false,
      "triggerPx": "0.0"
    },
    "status": "open",           // open, filled, canceled, badAloPxRejected
    "user": "0x...",            // Wallet address
    "hash": "0x...",            // Transaction hash (if filled)
    "builder": null,
    "time": "2025-10-27T16:59:59.797563798"
  }
}
```

**Key Fields:**
- `oid` - Track order lifecycle (placed → filled/canceled)
- `status` - Order state changes
- `side` - Bid (B) vs Ask (A)
- `limitPx` - Price level
- `sz` vs `origSz` - Partial fills detection
- `user` - Wallet tracking
- `timestamp` - High-precision timing

---

## 🚀 Phase 1: Core Order Book Reconstruction (Today)

**Goal:** Build foundational order book state tracking and 5 killer visualizations.

### 1. Full Level 2 Order Book Reconstruction ⭐⭐⭐⭐⭐

**What:** Reconstruct the complete order book state at any timestamp.

**Visualizations:**
- **Animated Depth Chart** - Watch bids/asks evolve over time
- **Order Book Snapshots** - Freeze-frame at key moments
- **Best Bid/Ask Tracking** - Mid-price and spread evolution

**Metrics:**
- Best bid/ask prices
- Mid-price: `(best_bid + best_ask) / 2`
- Spread: `best_ask - best_bid`
- Spread %: `(spread / mid_price) * 100`

**Research Value:**
- Understand liquidity dynamics
- Identify tight vs wide spread periods
- Track market maker activity

---

### 2. Liquidity Heatmap (Price × Time) ⭐⭐⭐⭐⭐

**What:** 2D heatmap showing liquidity concentration across price levels and time.

**Visualization:**
- **X-axis:** Time (minute-by-minute)
- **Y-axis:** Price levels (relative to mid-price)
- **Color:** Total liquidity depth (log scale)
- **Separate panels:** Bids (green) vs Asks (red)

**Features:**
- Zoom into specific time windows
- Identify "walls" of liquidity
- Track support/resistance levels
- See liquidity appearance/disappearance

**Research Value:**
- Where do large orders cluster?
- When does liquidity dry up?
- What price levels attract the most depth?

---

### 3. Order Book Imbalance Over Time ⭐⭐⭐⭐

**What:** Measure bid vs ask pressure at multiple depth levels.

**Metrics:**
- **Imbalance Ratio:** `(bid_depth - ask_depth) / (bid_depth + ask_depth)`
- **Range:** -1 (all asks) to +1 (all bids)
- **Depths:** Top 1%, 5%, 10% of book

**Visualizations:**
- **Time series:** Imbalance over time
- **Heatmap:** Imbalance by depth level
- **Correlation:** Imbalance vs price changes

**Research Value:**
- Does imbalance predict price movements?
- Lead-lag relationship with trades
- Early warning signals for moves

**Academic Reference:**
- Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order Book Events"

---

### 4. Cumulative Depth Profile ⭐⭐⭐⭐

**What:** Show total liquidity at various distances from mid-price.

**Visualization:**
- **Stepped area chart:** Cumulative volume by price level
- **Both sides:** Bids (left) vs Asks (right)
- **Animation:** Watch depth evolve every 1-10 seconds

**Metrics:**
- Depth at 0.1%, 0.5%, 1%, 5% from mid
- Total depth (sum of all levels)
- Depth asymmetry (bid vs ask)

**Research Value:**
- Market depth measurement
- Liquidity cost for large orders
- Compare liquidity across coins

---

### 5. Spread Dynamics Analysis ⭐⭐⭐⭐

**What:** Detailed analysis of bid-ask spread behavior.

**Visualizations:**
- **Spread Time Series:** Absolute and percentage spread
- **Spread Distribution:** Histogram of spread values
- **Spread vs Volume:** Does volume tighten spread?
- **Spread vs Volatility:** Wide spreads = volatile periods?

**Metrics:**
- Mean/median spread
- Spread volatility (std dev)
- % time at minimum tick size
- Quoted spread vs effective spread

**Research Value:**
- Trading cost estimation
- Identify low-cost execution windows
- Market maker profitability proxies

---

## 🔮 Phase 2: Advanced Order Lifecycle (Next Session)

### 6. Order Lifetime Analysis ⭐⭐⭐⭐

**What:** Track orders from placement to fill/cancel.

**Metrics:**
- Time-to-fill distribution
- Time-to-cancel distribution
- Fill probability by queue position
- Survival curves (Kaplan-Meier)

**Visualizations:**
- **Lifetime histograms** - How long do orders survive?
- **Fill rate by price level** - Distance from mid-price
- **Decay curves** - Queue position evolution

**Research Value:**
- Optimal order placement strategy
- Market maker adverse selection
- Execution probability modeling

---

### 7. Market Maker Identification ⭐⭐⭐⭐⭐

**What:** Identify and rank liquidity providers.

**Identification Criteria:**
- **Two-sided quoting:** Simultaneously bid + ask
- **High cancel rate:** Frequent adjustments
- **Size concentration:** Large orders
- **Time on book:** Patient liquidity provision

**Metrics per Wallet:**
- Total liquidity provided (size × time)
- Quote update frequency
- Inventory management (net position)
- Adverse selection ratio

**Visualizations:**
- **Top 20 market makers** - Ranked by metrics
- **Quote intensity heatmap** - Who quotes when?
- **Spread capture** - Profitability estimates

**Research Value:**
- Who are the professionals?
- Market maker strategies
- Competitive dynamics

---

### 8. Spoofing & Manipulation Detection ⭐⭐⭐⭐

**What:** Identify suspicious order book behavior.

**Detection Methods:**

**A. Iceberg Orders:**
- Large order placed → quick cancel
- No fill occurred
- Same wallet repeats pattern

**B. Layering:**
- Multiple orders stacked on one side
- Intent to create false depth
- Canceled after opposite-side fill

**C. Quote Stuffing:**
- Rapid placement/cancellation
- >10 orders per second
- Minimal fill rate

**D. Wash Trading:**
- Same wallet on both sides
- Self-trading detection

**Visualizations:**
- **Timeline of suspicious events**
- **Wallet behavior flags**
- **Order lifetime scatter** (size vs duration)

**Research Value:**
- Market integrity monitoring
- Regulatory research
- Fair market assessment

---

### 9. Liquidity Provision Quality ⭐⭐⭐

**What:** Measure effectiveness of liquidity providers.

**Metrics:**
- **Fill rate:** % orders filled vs canceled
- **Adverse selection:** Filled before unfavorable move
- **Inventory risk:** Net position exposure
- **Spread capture:** Estimated profit per trade

**Visualizations:**
- **Quality matrix:** Fill rate vs adverse selection
- **Wallet rankings** - Best to worst providers
- **ROI estimates** - Profitability proxies

**Research Value:**
- Skilled vs unskilled market makers
- Optimal MM strategies
- Inventory risk management

---

### 10. Order Flow Toxicity (VPIN) ⭐⭐⭐⭐

**What:** Measure informed trading pressure.

**VPIN Formula:**
```
VPIN = |buy_volume - sell_volume| / total_volume
```

**Interpretation:**
- High VPIN = informed traders active
- Low VPIN = balanced flow, uninformed
- Spikes = adverse selection risk

**Visualizations:**
- **VPIN time series** - Rolling window
- **VPIN vs spread** - Widen during toxicity?
- **VPIN vs volatility** - Predictive?

**Research Value:**
- Market maker risk management
- Execution timing (avoid toxic periods)
- Flash crash precursors

**Academic Reference:**
- Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World"

---

## 🛠️ Technical Implementation

### File Structure

```
/Users/thebunnymac/Desktop/hyperorderbook/
├── orderbook_loader.py          # NEW: Order book data parser
├── orderbook_visualizations.py  # NEW: Order book charts
├── data_loader.py               # EXISTING: Trade data
├── visualizations.py            # EXISTING: Trade visualizations
├── dashboard.py                 # UPDATE: Add new tab
└── Hyperliquid Data Expanded/
    ├── Hyperliquid_orderbooks.json.gz  # 50M events
    ├── node_fills_20251027_1700-1800.json
    └── misc_events_20251027_1700-1800.json
```

### Key Algorithms

**1. Order Book State Reconstruction:**
```python
order_book = {
    'BTC': {
        'bids': {},  # {price: [(oid, size, wallet, timestamp), ...]}
        'asks': {}
    }
}

# Process events sequentially
for event in events:
    if status == 'open':
        order_book[coin][side][price].append(order)
    elif status == 'filled':
        order_book[coin][side][price].remove(order)
    elif status == 'canceled':
        order_book[coin][side][price].remove(order)
```

**2. Liquidity Aggregation:**
```python
def get_depth(order_book, coin, side, num_levels=10):
    prices = sorted(order_book[coin][side].keys(), reverse=(side=='bids'))
    depth = []
    for price in prices[:num_levels]:
        total_size = sum(order[1] for order in order_book[coin][side][price])
        depth.append((price, total_size))
    return depth
```

**3. Imbalance Calculation:**
```python
def calculate_imbalance(bid_depth, ask_depth):
    return (bid_depth - ask_depth) / (bid_depth + ask_depth)
```

### Performance Optimization

**Challenges:**
- 50M events = memory intensive
- Need efficient order tracking

**Solutions:**
1. **Sampling:** Process every Nth block for visualization
2. **Filtering:** Focus on top 10-20 liquid coins
3. **Caching:** Save processed states to Parquet
4. **Incremental:** Build state incrementally, not all at once
5. **Indexing:** Use `oid` and `timestamp` indices

**Expected Performance:**
- Initial load: 2-5 minutes (one-time)
- Cached load: 5-10 seconds
- Visualization render: <1 second

---

## 📈 Visualization Gallery (Examples)

### Liquidity Heatmap Example
```
       17:00  17:10  17:20  17:30  17:40  17:50  18:00
+2.0%  [🟩🟩🟩🟨🟨🟦🟦]  ASKS
+1.0%  [🟩🟩🟩🟩🟨🟨🟦]
+0.5%  [🟩🟩🟩🟩🟩🟨🟨]
 Mid   [━━━━━━━━━━━━━━━]
-0.5%  [🟥🟥🟥🟥🟥🟧🟧]
-1.0%  [🟥🟥🟥🟥🟧🟧🟦]
-2.0%  [🟥🟥🟥🟧🟧🟦🟦]  BIDS

Color: 🟩/🟥 = High liquidity, 🟦 = Low liquidity
```

### Order Book Imbalance Example
```
Imbalance Ratio Over Time

+1.0 ┤     ╭─╮           (Heavy bid pressure)
     │    ╭╯ ╰╮
 0.5 ┤   ╭╯   ╰╮
     │  ╭╯     ╰─╮
 0.0 ┼──╯         ╰──────
     │
-0.5 ┤              ╭─╮  (Heavy ask pressure)
     │             ╭╯ ╰╮
-1.0 ┤            ╭╯   ╰─
     └────────────────────
     17:00              18:00
```

---

## 🎓 Academic & Research Applications

### 1. Market Microstructure Studies
- **Liquidity provision dynamics**
- **Price discovery mechanisms**
- **Order flow information content**
- **Adverse selection costs**

### 2. Trading Strategy Development
- **Optimal execution algorithms**
- **Market making strategies**
- **Liquidity taking vs providing**
- **Timing strategies (when to trade)**

### 3. Market Quality Measurement
- **Spread analysis**
- **Depth resilience**
- **Price impact minimization**
- **Execution cost benchmarking**

### 4. Behavioral Analysis
- **Trader type classification**
- **Informed vs uninformed flow**
- **Herding behavior**
- **Strategic order placement**

### 5. Regulatory & Compliance
- **Manipulation detection**
- **Market abuse monitoring**
- **Fair market assessment**
- **Systemic risk indicators**

---

## 📚 Key Academic References

1. **Hasbrouck, J. (2007).** *Empirical Market Microstructure*
   - Foundational text on order book analysis

2. **Gould, M. D., et al. (2013).** "Limit order books"
   - Comprehensive review of LOB dynamics

3. **Cont, R., Kukanov, A., & Stoikov, S. (2014).** "The Price Impact of Order Book Events"
   - How order book changes affect prices

4. **Easley, D., López de Prado, M., & O'Hara, M. (2012).** "Flow Toxicity and Liquidity in a High-Frequency World"
   - VPIN metric for informed trading

5. **Kyle, A. S. (1985).** "Continuous Auctions and Insider Trading"
   - Lambda (price impact) measurement

6. **Biais, B., Hillion, P., & Spatt, C. (1995).** "An Empirical Analysis of the Limit Order Book"
   - Classic LOB empirical study

---

## 🎯 Success Metrics

**Phase 1 Complete When:**
- ✅ Order book state fully reconstructed
- ✅ 5 core visualizations working
- ✅ Processing <5 minutes for full dataset
- ✅ Interactive dashboard tab added
- ✅ Interpretation guides included

**Phase 2 Complete When:**
- ✅ Order lifecycle tracking working
- ✅ Market maker identification automated
- ✅ Manipulation detection algorithms deployed
- ✅ All 10 visualization families complete

---

## ⚡ Quick Start (Phase 1)

**Files to Create:**
1. `orderbook_loader.py` - Data parsing & state reconstruction
2. `orderbook_visualizations.py` - 5 visualization classes
3. Update `dashboard.py` - Add "📖 Order Book Dynamics" tab

**Expected Build Time:** 2-3 hours

**Let's build the most advanced order book analysis tool in crypto!** 🚀

---

*Document created: October 29, 2025*
*Dataset: Hyperliquid Order Book Events (Oct 27, 2025, 17:00-18:00)*
*Events: ~50,000,000 | Size: 430MB compressed*

