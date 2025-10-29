"""
Hyperliquid Market Microstructure Dashboard
Interactive Streamlit application for Phase 1 visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from data_loader import HyperliquidDataLoader
from visualizations import (
    OrderBookHeatmap,
    MakerTakerFlow,
    SpreadAnalysis,
    VolumeProfile,
    TraderAnalytics,
    OrderbookImpactAnalysis,
    LiquidationAnalysis,
    OrderFlowImbalance,
    CorrelationMatrix,
    MarketImpactAnalysis
)
from orderbook_loader import OrderBookLoader
from orderbook_visualizations import (
    OrderBookDepthChart,
    LiquidityHeatmap,
    ImbalanceAnalysis,
    SpreadAnalysisOrderBook,
    OrderLifecycleAnalysis,
    OrderBookLadderVisualization
)
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Hyperliquid Market Microstructure",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(data_path: str, max_lines: int = None):
    """Load and cache data"""
    loader = HyperliquidDataLoader(misc_events_path=data_path)
    
    # Create unique cache file per data source
    import hashlib
    path_hash = hashlib.md5(data_path.encode()).hexdigest()[:8]
    processed_path = f'processed_data_{path_hash}.parquet'
    
    if os.path.exists(processed_path):
        st.info("Loading from cached processed data...")
        df = loader.load_processed_data(processed_path)
    else:
        with st.spinner(f"Loading data from {Path(data_path).name}..."):
            df = loader.load_misc_events(max_lines=max_lines)
            loader.save_processed_data(processed_path)
    
    return df, loader


@st.cache_data(show_spinner=False)
def get_trader_analytics(_loader):
    """Get trader analytics (cached)"""
    return _loader.get_trader_analytics(min_trades=5)


@st.cache_data(show_spinner=False)
def get_coin_summary(_loader):
    """Get coin summary (cached)"""
    return _loader.get_coin_summary()


def main():
    # Header
    st.markdown('<p class="main-header">📊 Hyperliquid Market Microstructure Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("### Phase 1: Core Market Structure Analysis")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Data loading
    st.sidebar.header("📁 Data Source")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["📊 Use sample data (demo)", "📤 Upload your own file", "📂 Use local file path"],
        help="Sample data: 1000 blocks for demo. Upload: Use your own Hyperliquid data."
    )
    
    data_path = None
    
    if data_source == "📊 Use sample data (demo)":
        # Use included sample data
        data_path = "sample_data.json"
        st.sidebar.info("Using sample dataset (1000 blocks, ~3MB)")
        
    elif data_source == "📤 Upload your own file":
        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Upload node_fills JSON file",
            type=['json'],
            help="Upload your Hyperliquid historical data file"
        )
        
        if uploaded_file:
            # Save temporarily
            temp_path = 'uploaded_data.json'
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            data_path = temp_path
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")
        else:
            st.sidebar.warning("⬆️ Please upload a data file to continue")
            st.info("👈 Upload your Hyperliquid data file using the sidebar")
            st.stop()
            
    else:  # Use local file path
        data_path = st.sidebar.text_input(
            "Local File Path",
            value="Hyperliquid Data Expanded/node_fills_20251027_1700-1800.json",
            help="Path to your local node_fills JSON file"
        )
    
    use_limit = st.sidebar.checkbox("Limit data for testing", value=False)
    max_lines = st.sidebar.number_input(
        "Max lines (0 = all)",
        min_value=0,
        max_value=50000,
        value=10000 if use_limit else 0,
        step=1000
    ) if use_limit else None
    
    # Load data
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {data_path}")
        st.stop()
    
    try:
        df, loader = load_data(data_path, max_lines)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Get analytics
    df_traders = get_trader_analytics(loader)
    df_coins = get_coin_summary(loader)
    
    # Overview metrics
    st.sidebar.header("📈 Data Overview")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Trades", f"{len(df):,}")
        st.metric("Unique Coins", df['coin'].nunique())
    with col2:
        st.metric("Unique Traders", df['address'].nunique())
        st.metric("Total Volume", f"${df['notional'].sum()/1e6:.2f}M")
    
    # Time range
    st.sidebar.info(f"**Time Range:**  \n{df['timestamp'].min().strftime('%Y-%m-%d %H:%M')}  \nto  \n{df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "🔥 Order Book Heatmap",
        "🔄 Maker vs Taker Flow",
        "📏 Spread Analysis",
        "📊 Volume Profile",
        "👥 Trader Analytics",
        "🎯 Wallet Impact",
        "💥 Liquidations",
        "📈 Order Flow Imbalance",
        "🔗 Asset Correlation",
        "💎 Market Impact",
        "📖 Order Book Dynamics",
        "📋 Data Explorer"
    ])
    
    # ========== TAB 1: Order Book Heatmap ==========
    with tab1:
        st.markdown('<p class="sub-header">Dynamic Order Book Heatmap</p>', 
                    unsafe_allow_html=True)
        st.markdown("Visualize market depth evolution over time across different price levels")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Get top coins by volume
            top_coins = df_coins['coin'].head(20).tolist()
            selected_coin_heatmap = st.selectbox(
                "Select Coin",
                options=top_coins,
                index=0,
                key='coin_heatmap'
            )
        
        with col2:
            time_res_heatmap = st.selectbox(
                "Time Resolution",
                options=['5S', '10S', '30S', '1T', '5T'],
                index=1,
                key='time_res_heatmap'
            )
        
        with col3:
            price_bins_heatmap = st.slider(
                "Price Bins",
                min_value=20,
                max_value=100,
                value=50,
                key='price_bins_heatmap'
            )
        
        if selected_coin_heatmap:
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center;'><b>Order Book Analysis - {selected_coin_heatmap}</b></h2>", 
                       unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.spinner("Generating order book heatmap..."):
                heatmap = OrderBookHeatmap(df)
                fig = heatmap.create_heatmap(
                    selected_coin_heatmap,
                    time_resolution=time_res_heatmap,
                    price_bins=price_bins_heatmap
                )
                
                st.markdown("### 🔥 Dynamic Depth Heatmap")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            # Interpretation guide
            with st.expander("📖 How to Read the Order Book Heatmap", expanded=False):
                st.markdown("""
                **🔥 Main Heatmap (Time × Price × Depth)**
                - **X-axis:** Time progression (left to right)
                - **Y-axis:** Price levels
                - **Color intensity:** Brighter = more trading volume at that price/time
                - **What to look for:**
                    - **Horizontal bands** = Support/resistance levels (lots of trading activity)
                    - **Gaps (dark areas)** = Liquidity gaps (price moves quickly through these)
                    - **Vertical bands** = High activity periods (news, volatility)
                    - **Shifting patterns** = Market sentiment changes
                
                **📊 Bid/Ask Imbalance (Bottom Chart)**
                - **Above zero (positive):** More buy pressure (bullish)
                - **Below zero (negative):** More sell pressure (bearish)
                - **Magnitude:** How strong the directional pressure is
                - **Use:** Leading indicator for short-term price movements
                """)
            
            st.markdown("---")
            
            # Show coin stats
            coin_stats = df_coins[df_coins['coin'] == selected_coin_heatmap].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", f"{coin_stats['num_trades']:,}")
            with col2:
                st.metric("Unique Traders", f"{coin_stats['num_traders']:,}")
            with col3:
                st.metric("Total Volume", f"${coin_stats['total_notional']/1e6:.2f}M")
            with col4:
                st.metric("Price Range", f"{coin_stats['price_range_pct']:.2f}%")
    
    # ========== TAB 2: Maker vs Taker Flow ==========
    with tab2:
        st.markdown('<p class="sub-header">Maker vs Taker Flow Analysis</p>', 
                    unsafe_allow_html=True)
        st.markdown("Analyze liquidity provision (makers) vs consumption (takers)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            flow_coin_options = ['All Coins'] + df_coins['coin'].head(20).tolist()
            selected_coin_flow = st.selectbox(
                "Select Coin",
                options=flow_coin_options,
                index=0,
                key='coin_flow'
            )
        
        with col2:
            time_res_flow = st.selectbox(
                "Time Resolution",
                options=['30S', '1T', '5T', '15T'],
                index=1,
                key='time_res_flow'
            )
        
        coin_filter = None if selected_coin_flow == 'All Coins' else selected_coin_flow
        
        st.markdown("---")
        title_coin = coin_filter if coin_filter else "All Coins"
        st.markdown(f"<h2 style='text-align: center;'><b>Maker vs Taker Flow - {title_coin}</b></h2>", 
                   unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.spinner("Generating flow analysis..."):
            flow = MakerTakerFlow(df)
            fig = flow.create_flow_analysis(coin=coin_filter, time_resolution=time_res_flow)
            
            st.markdown("### 🔄 Liquidity Flow Analysis")
            with st.expander("🔍 Expand for Full Screen View", expanded=True):
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        
        # Interpretation guide
        with st.expander("📖 Understanding Maker vs Taker Flow", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🔄 Maker vs Taker Volume (Stacked Area)**
                - **Blue (Maker):** Passive orders providing liquidity
                - **Purple (Taker):** Aggressive orders consuming liquidity
                - **Healthy market:** 40-60% maker ratio
                - **High maker ratio:** Strong liquidity provision
                - **High taker ratio:** Aggressive trading, possible volatility
                
                **📈 Maker Ratio (Line Chart)**
                - **Above 50%:** More liquidity provision (calm market)
                - **Below 50%:** More aggressive trading (active market)
                - **Sudden drops:** Potential price movement incoming
                """)
            
            with col2:
                st.markdown("""
                **💰 Cumulative Fees**
                - **Positive:** Net fees paid to exchange (mostly takers)
                - **Negative:** Net rebates received (mostly makers)
                - **Slope:** Rate of fee accumulation
                
                **🥧 Trade Distribution (Pie Chart)**
                - **Overall balance:** Market health indicator
                - **Mostly maker:** Market making dominant
                - **Mostly taker:** Active trading/volatility
                """)
        
        st.markdown("---")
        
        # Summary stats
        if coin_filter:
            df_filtered = df[df['coin'] == coin_filter]
        else:
            df_filtered = df
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            maker_ratio = df_filtered['is_maker'].sum() / len(df_filtered)
            st.metric("Maker Ratio", f"{maker_ratio*100:.1f}%")
        with col2:
            maker_vol = df_filtered[df_filtered['is_maker']]['notional'].sum()
            st.metric("Maker Volume", f"${maker_vol/1e6:.2f}M")
        with col3:
            taker_vol = df_filtered[df_filtered['is_taker']]['notional'].sum()
            st.metric("Taker Volume", f"${taker_vol/1e6:.2f}M")
        with col4:
            total_fees = df_filtered['fee'].sum()
            st.metric("Net Fees", f"${total_fees:,.0f}")
    
    # ========== TAB 3: Spread Analysis ==========
    with tab3:
        st.markdown('<p class="sub-header">Spread Analysis Dashboard</p>', 
                    unsafe_allow_html=True)
        st.markdown("Compare bid-ask spreads across different assets")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            top_20_coins = df_coins['coin'].head(20).tolist()
            selected_coins_spread = st.multiselect(
                "Select Coins (up to 5)",
                options=top_20_coins,
                default=top_20_coins[:3],
                max_selections=5,
                key='coins_spread'
            )
        
        with col2:
            time_res_spread = st.selectbox(
                "Time Resolution",
                options=['30S', '1T', '5T'],
                index=1,
                key='time_res_spread'
            )
        
        if selected_coins_spread:
            st.markdown("---")
            coins_str = ", ".join(selected_coins_spread[:3]) + ("..." if len(selected_coins_spread) > 3 else "")
            st.markdown(f"<h2 style='text-align: center;'><b>Spread Analysis - {coins_str}</b></h2>", 
                       unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.spinner("Generating spread analysis..."):
                spread_viz = SpreadAnalysis(df)
                fig = spread_viz.create_spread_dashboard(
                    selected_coins_spread,
                    time_resolution=time_res_spread
                )
                
                st.markdown("### 📏 Multi-Asset Spread Comparison")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            # Interpretation guide
            with st.expander("📖 Understanding Spread Analysis", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **📏 Spread Over Time (Top Left)**
                    - **Y-axis:** Bid-ask spread in basis points (bps)
                    - **What's good:** Lower spread = more efficient market
                    - **Benchmarks:**
                        - **<5 bps:** Excellent (major pairs like BTC/ETH)
                        - **5-20 bps:** Good (established altcoins)
                        - **>20 bps:** Wide (low liquidity or volatile assets)
                    - **Patterns:**
                        - Widening spreads = volatility or liquidity drain
                        - Tight spreads = calm, liquid market
                    
                    **📊 Spread vs Volume (Top Right)**
                    - **Expected:** Negative correlation (higher volume = tighter spread)
                    - **Outliers:** High spread + high volume = volatile event
                    """)
                
                with col2:
                    st.markdown("""
                    **📈 Spread Distribution (Bottom Left)**
                    - Shows how often each spread level occurs
                    - **Tight distribution:** Stable market
                    - **Wide distribution:** Variable liquidity conditions
                    - **Multiple peaks:** Different market regimes
                    
                    **📊 Cross-Asset Comparison (Bottom Right)**
                    - **Average spread** for each selected asset
                    - **Direct comparison** of market efficiency
                    - **Use for:** Choosing most liquid markets to trade
                    - **Lower bars** = better for execution
                    """)
            
            st.markdown("---")
        else:
            st.warning("Please select at least one coin")
    
    # ========== TAB 4: Volume Profile ==========
    with tab4:
        st.markdown('<p class="sub-header">Volume Profile by Asset</p>', 
                    unsafe_allow_html=True)
        st.markdown("Distribution of trading volume across price levels")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_coin_profile = st.selectbox(
                "Select Coin",
                options=df_coins['coin'].head(20).tolist(),
                index=0,
                key='coin_profile'
            )
        
        with col2:
            price_bins_profile = st.slider(
                "Price Bins",
                min_value=20,
                max_value=100,
                value=50,
                key='price_bins_profile'
            )
        
        if selected_coin_profile:
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center;'><b>Volume Profile - {selected_coin_profile}</b></h2>", 
                       unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.spinner("Generating volume profile..."):
                profile = VolumeProfile(df)
                fig = profile.create_volume_profile(
                    selected_coin_profile,
                    price_bins=price_bins_profile
                )
                
                st.markdown("### 📊 Volume at Price Distribution")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            # Interpretation guide
            with st.expander("📖 How to Read the Volume Profile", expanded=False):
                st.markdown("""
                **📊 Main Volume Profile (Left - Horizontal Bars)**
                - **Green bars:** Buyer-initiated volume (market buy orders)
                - **Red bars:** Seller-initiated volume (market sell orders)
                - **Length of bar:** Total volume traded at that price level
                
                **Key Concepts:**
                - **POC (Yellow line):** Point of Control - price with highest volume = "fair value"
                - **Value Area:** Price range containing 70% of volume (most accepted prices)
                - **High Volume Nodes (HVN):** Long bars = support/resistance levels
                - **Low Volume Nodes (LVN):** Short bars = price likely to move quickly through
                
                **Trading Applications:**
                - Price above POC = bullish bias
                - Price below POC = bearish bias
                - POC acts as magnet (price often returns to it)
                - LVNs = potential breakout areas
                
                **📈 Price Distribution (Right)**
                - Shows how often each price was visited
                - Complements volume profile with frequency data
                """)
            
            st.markdown("---")
            
            # Additional insights
            df_coin = df[df['coin'] == selected_coin_profile]
            
            st.markdown("#### Volume Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                buy_vol = df_coin[df_coin['aggressor_side'] == 'buy']['notional'].sum()
                st.metric("Buyer Initiated", f"${buy_vol/1e3:.0f}K")
            with col2:
                sell_vol = df_coin[df_coin['aggressor_side'] == 'sell']['notional'].sum()
                st.metric("Seller Initiated", f"${sell_vol/1e3:.0f}K")
            with col3:
                buy_sell_ratio = buy_vol / (sell_vol + 1e-10)
                st.metric("Buy/Sell Ratio", f"{buy_sell_ratio:.2f}")
            with col4:
                avg_trade = df_coin['notional'].mean()
                st.metric("Avg Trade Size", f"${avg_trade:,.0f}")
    
    # ========== TAB 5: Trader Analytics ==========
    with tab5:
        st.markdown('<p class="sub-header">Account-Level Trader Analytics</p>', 
                    unsafe_allow_html=True)
        st.markdown("Identify top traders and market movers")
        
        subtab1, subtab2 = st.tabs(["Top Traders Overview", "Individual Trader Detail"])
        
        with subtab1:
            top_n = st.slider("Number of top traders to show", 10, 50, 20)
            
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center;'><b>Top {top_n} Trader Analytics</b></h2>", 
                       unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.spinner("Generating trader analytics..."):
                trader_viz = TraderAnalytics(df_traders, df)
                fig = trader_viz.create_trader_dashboard(top_n=top_n)
                
                st.markdown("### 👥 Trader Performance Dashboard")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            # Interpretation guides for each chart
            with st.expander("📖 How to Interpret These Charts", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **📊 Top Traders by Volume (Bar Chart)**
                    - **What it shows:** The biggest players ranked by total trading volume
                    - **Color gradient:** Darker = higher volume
                    - **How to use it:** 
                        - Identify whales and major market participants
                        - Top 20 traders often represent 40-50% of total volume
                        - Focus on addresses with >$1M volume for whale analysis
                    
                    **📈 Trade Count vs Volume (Scatter Plot)**
                    - **What it shows:** Relationship between activity and size
                    - **Color coding:** Green = high maker ratio (market makers), Red = low maker ratio (aggressive traders)
                    - **How to use it:**
                        - Top-right = high-volume, high-frequency (likely algo/MM)
                        - Top-left = high-volume, low-frequency (whales doing big trades)
                        - Bottom = low activity traders
                    """)
                
                with col2:
                    st.markdown("""
                    **📉 Maker Ratio Distribution (Histogram)**
                    - **What it shows:** How traders are distributed by strategy type
                    - **Maker ratio ranges:**
                        - **>80%** = Pure market makers (provide liquidity)
                        - **40-80%** = Mixed strategies
                        - **<40%** = Aggressive takers (consume liquidity)
                    - **How to use it:** Understand market participant composition
                    
                    **📦 P&L Distribution (Box Plots)**
                    - **What it shows:** Profit/loss range for top 10 traders
                    - **Box elements:**
                        - Center line = median P&L per trade
                        - Box = 25th to 75th percentile
                        - Whiskers = outliers
                    - **How to use it:**
                        - Wide boxes = inconsistent results
                        - Box above zero = mostly profitable
                        - Outliers = occasional big wins/losses
                    """)
            
            st.markdown("---")
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            # Top traders table
            st.markdown("#### 📋 Top Traders Detailed Table")
            
            with st.expander("💡 Table Guide", expanded=False):
                st.markdown("""
                **How to identify different trader types:**
                - 🐋 **Whales:** High volume, low maker ratio (<30%), few trades
                - 🤖 **Market Makers:** High volume, high maker ratio (>70%), many trades
                - 📈 **Active Traders:** Medium volume, balanced maker ratio (40-70%)
                - 💰 **Profitable Whales:** High volume + positive P&L
                """)
            
            display_cols = ['address', 'num_trades', 'total_notional', 'num_coins', 
                          'maker_ratio', 'total_pnl', 'abs_fees']
            
            df_display = df_traders.head(top_n)[display_cols].copy()
            df_display['address'] = df_display['address'].str[:16] + '...'
            df_display['total_notional'] = df_display['total_notional'].apply(lambda x: f"${x:,.0f}")
            df_display['total_pnl'] = df_display['total_pnl'].apply(lambda x: f"${x:,.2f}")
            df_display['abs_fees'] = df_display['abs_fees'].apply(lambda x: f"${x:,.2f}")
            df_display['maker_ratio'] = df_display['maker_ratio'].apply(lambda x: f"{x*100:.1f}%")
            
            df_display.columns = ['Address', 'Trades', 'Volume', 'Coins', 'Maker %', 'P&L', 'Fees']
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        with subtab2:
            # Select trader
            trader_options = df_traders['address'].head(50).tolist()
            selected_trader = st.selectbox(
                "Select Trader Address",
                options=trader_options,
                format_func=lambda x: f"{x[:16]}... ({df_traders[df_traders['address']==x]['num_trades'].iloc[0]} trades)"
            )
            
            if selected_trader:
                st.markdown("---")
                st.markdown(f"<h2 style='text-align: center;'><b>Trader Profile: {selected_trader[:16]}...</b></h2>", 
                           unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                with st.spinner("Loading trader detail..."):
                    trader_viz = TraderAnalytics(df_traders, df)
                    fig = trader_viz.create_trader_detail(selected_trader)
                    
                    st.markdown("### 📊 Individual Trader Analysis")
                    with st.expander("🔍 Expand for Full Screen View", expanded=True):
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                
                # Interpretation guide for individual trader charts
                with st.expander("📖 How to Read This Trader's Profile", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **💰 Cumulative P&L Over Time**
                        - **What it shows:** Running total of profit/loss
                        - **Trend interpretation:**
                            - Upward slope = profitable period
                            - Downward slope = losing period
                            - Flat = break-even trading
                        - **Volatility:** Sharp jumps = large individual trades
                        
                        **📊 Trade Size Distribution**
                        - **What it shows:** How consistent are their trade sizes?
                        - **Patterns:**
                            - Single peak = consistent sizing (algorithmic)
                            - Multiple peaks = different strategies
                            - Long tail = occasional large trades
                        """)
                    
                    with col2:
                        st.markdown("""
                        **🎯 Coins Traded**
                        - **What it shows:** Trading focus/specialization
                        - **Strategy indicators:**
                            - 1-2 coins = Specialist (deep knowledge)
                            - 3-5 coins = Focused diversification
                            - 5+ coins = Generalist/market maker
                        
                        **🔄 Maker vs Taker Pie**
                        - **What it shows:** Trading style classification
                        - **Interpretation:**
                            - **>70% Maker** = Market maker/liquidity provider
                            - **50-50** = Balanced/opportunistic
                            - **>70% Taker** = Aggressive/informed trader
                        """)
                
                st.markdown("---")
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                
                # Trader stats
                trader_info = df_traders[df_traders['address'] == selected_trader].iloc[0]
                
                st.markdown("#### 📊 Trader Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Trades", f"{trader_info['num_trades']:,}")
                with col2:
                    st.metric("Total Volume", f"${trader_info['total_notional']/1e6:.2f}M")
                with col3:
                    st.metric("Unique Coins", trader_info['num_coins'])
                with col4:
                    st.metric("Maker Ratio", f"{trader_info['maker_ratio']*100:.1f}%")
                with col5:
                    st.metric("Total P&L", f"${trader_info['total_pnl']:,.2f}")
                
                # Add trader type classification
                st.markdown("<br>", unsafe_allow_html=True)
                
                maker_ratio = trader_info['maker_ratio']
                volume = trader_info['total_notional']
                num_trades = trader_info['num_trades']
                
                # Classify trader
                if maker_ratio > 0.7:
                    trader_type = "🤖 Market Maker"
                    description = "This trader primarily provides liquidity to the market"
                elif maker_ratio < 0.3 and volume > 1e6:
                    trader_type = "🐋 Whale (Aggressive)"
                    description = "High-volume trader who aggressively takes liquidity"
                elif maker_ratio < 0.3:
                    trader_type = "⚡ Aggressive Trader"
                    description = "Primarily consumes liquidity (market taker)"
                else:
                    trader_type = "📈 Balanced Trader"
                    description = "Uses both market making and taking strategies"
                
                st.info(f"**Trader Type:** {trader_type}  \n{description}")
    
    # ========== TAB 6: Wallet Impact Analysis ==========
    with tab6:
        st.markdown('<p class="sub-header">🎯 Orderbook Impact Analysis</p>', 
                    unsafe_allow_html=True)
        st.markdown("**Research Question:** *Which wallets move each orderbook the most?*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_coin_impact = st.selectbox(
                "Select Coin",
                options=df_coins['coin'].head(20).tolist(),
                index=0,
                key='coin_impact'
            )
        
        with col2:
            top_n_impact = st.slider(
                "Top N Wallets",
                min_value=5,
                max_value=30,
                value=15,
                key='top_n_impact'
            )
        
        if selected_coin_impact:
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center;'><b>Orderbook Impact Analysis - {selected_coin_impact}</b></h2>", 
                       unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.spinner("Analyzing wallet impact on orderbook..."):
                impact_viz = OrderbookImpactAnalysis(df)
                
                # Get individual charts
                fig1 = impact_viz.create_volume_chart(selected_coin_impact, top_n=top_n_impact)
                fig2 = impact_viz.create_price_impact_chart(selected_coin_impact, top_n=top_n_impact)
                fig3 = impact_viz.create_aggressive_volume_chart(selected_coin_impact, top_n=top_n_impact)
                fig4 = impact_viz.create_trade_size_distribution(selected_coin_impact, top_n=top_n_impact)
                
                # Display each chart individually with expander for full screen
                st.markdown("### 📊 Top Market Movers by Volume")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 📈 Average Price Impact per Trade")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 💥 Aggressive Trading Volume (Orderbook Consumption)")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig3, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 📦 Trade Size Distribution (Top 5 Wallets)")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig4, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interpretation guide
            with st.expander("📖 Understanding Orderbook Impact Analysis", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **🎯 Top Market Movers (Top Left)**
                    - **What it shows:** Wallets ranked by total volume
                    - **Color coding:** 
                        - **Red bars:** High taker ratio (aggressive, consumes orderbook)
                        - **Yellow/Green bars:** High maker ratio (provides liquidity)
                    - **Key insight:** Red bars = wallets that MOVE the market most
                    
                    **📈 Price Impact Analysis (Top Right)**
                    - **X-axis:** Number of trades
                    - **Y-axis:** Average price impact percentage
                    - **Bubble size:** Average trade size
                    - **What to look for:**
                        - Large bubbles high on Y-axis = whales with big impact
                        - Many trades + high impact = active market mover
                    """)
                
                with col2:
                    st.markdown("""
                    **💥 Aggressive Trading Volume (Bottom Left)**
                    - **Red bars:** Volume from orderbook-consuming trades
                    - **Higher = more aggressive** market taking
                    - **Use case:** Identify wallets that execute large market orders
                    
                    **📊 Trade Size Distribution (Bottom Right)**
                    - **Box plots** for top 5 wallets
                    - **Wide boxes:** Variable trade sizes (flexible strategy)
                    - **Narrow boxes:** Consistent sizing (algorithmic)
                    - **High medians:** Large average trades
                    
                    **Research Insight:** Wallets with high aggressive volume + large trade sizes = primary market movers
                    """)
            
            st.markdown("---")
            
            # Summary insights
            st.markdown("#### 💡 Key Insights")
            
            df_coin_data = df[df['coin'] == selected_coin_impact]
            total_volume = df_coin_data['notional'].sum()
            
            # Calculate top wallet impact
            top_wallets_volume = df_coin_data.groupby('address')['notional'].sum().nlargest(top_n_impact).sum()
            impact_pct = (top_wallets_volume / total_volume) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"Top {top_n_impact} Wallets Control",
                    f"{impact_pct:.1f}%",
                    "of total volume"
                )
            with col2:
                aggressive_trades = df_coin_data[df_coin_data['crossed'] == True]
                aggressive_pct = (len(aggressive_trades) / len(df_coin_data)) * 100
                st.metric(
                    "Market Taking",
                    f"{aggressive_pct:.1f}%",
                    "of all trades"
                )
            with col3:
                top_trader_vol = df_coin_data.groupby('address')['notional'].sum().max()
                st.metric(
                    "Top Single Wallet",
                    f"${top_trader_vol/1e6:.2f}M",
                    "trading volume"
                )
    
    # ========== TAB 7: Liquidation Analysis ==========
    with tab7:
        st.markdown('<p class="sub-header">💥 Liquidation Analysis</p>', 
                    unsafe_allow_html=True)
        st.markdown("**Research Question:** *At what price points do liquidations occur?*")
        
        selected_coin_liq = st.selectbox(
            "Select Coin",
            options=df_coins['coin'].head(20).tolist(),
            index=0,
            key='coin_liq'
        )
        
        if selected_coin_liq:
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center;'><b>Liquidation Analysis - {selected_coin_liq}</b></h2>", 
                       unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.spinner("Analyzing liquidation patterns..."):
                liq_viz = LiquidationAnalysis(df)
                
                # Get individual charts
                fig1 = liq_viz.create_liquidation_scatter(selected_coin_liq)
                fig2 = liq_viz.create_price_histogram(selected_coin_liq)
                fig3 = liq_viz.create_cumulative_volume(selected_coin_liq)
                fig4 = liq_viz.create_close_distribution(selected_coin_liq)
                
                # Display each chart individually with expander
                st.markdown("### 💥 Liquidation Hotspot Heatmap")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 📊 Large Closes by Price Level")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 📈 Position Close Volume Over Time")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig3, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 🎯 Close Direction Distribution")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    st.plotly_chart(fig4, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interpretation guide
            with st.expander("📖 Understanding Liquidation Analysis", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **💥 Liquidation Hotspot Heatmap (Top Left)**
                    - **Dots:** Large position closures (potential liquidations)
                    - **Bubble size:** Trade size
                    - **Color:** P&L (Red = loss, Green = profit)
                    - **What it reveals:**
                        - Clusters at specific prices = liquidation zones
                        - Red clusters = forced liquidations (losses)
                        - Horizontal patterns = support/resistance where liq happens
                    
                    **📊 Large Closes by Price (Top Right)**
                    - **Histogram** showing which prices had most closures
                    - **Peaks** = liquidation-prone price levels
                    - **Use:** Identify key price zones where positions get closed
                    """)
                
                with col2:
                    st.markdown("""
                    **📈 Position Close Volume Over Time (Bottom Left)**
                    - **Cumulative** volume of closed positions
                    - **Steep slopes:** Periods of heavy liquidations
                    - **Flat sections:** Calm periods
                    
                    **🎯 Close Direction Distribution (Bottom Right)**
                    - **Types of closes:** "Close Long", "Close Short", etc.
                    - **Height:** Frequency of each type
                    - **Insight:** Understand liquidation direction bias
                    
                    **Research Application:**
                    - Identify liquidation cascades (rapid increases in cumulative volume)
                    - Find price levels with concentrated risk
                    - Predict potential support/resistance based on liq zones
                    """)
            
            st.markdown("---")
            
            # Summary stats
            df_coin_liq = df[df['coin'] == selected_coin_liq]
            closes = df_coin_liq[df_coin_liq['dir'].str.contains('Close', na=False)]
            
            if not closes.empty:
                st.markdown("#### 📊 Liquidation Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_close_vol = closes['notional'].sum()
                    st.metric("Total Close Volume", f"${total_close_vol/1e6:.2f}M")
                
                with col2:
                    close_count = len(closes)
                    st.metric("Number of Closes", f"{close_count:,}")
                
                with col3:
                    avg_close_size = closes['notional'].mean()
                    st.metric("Avg Close Size", f"${avg_close_size:,.0f}")
                
                with col4:
                    # Calculate largest liquidation cluster
                    price_bins = pd.cut(closes['px'], bins=20)
                    cluster_vol = closes.groupby(price_bins)['notional'].sum().max()
                    st.metric("Largest Price Cluster", f"${cluster_vol/1e6:.2f}M")
                
                # Most dangerous price levels
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### ⚠️ High-Risk Price Levels")
                
                price_risk = closes.groupby(pd.cut(closes['px'], bins=10)).agg({
                    'notional': 'sum',
                    'sz': 'count'
                }).reset_index()
                price_risk.columns = ['Price Range', 'Total Volume', 'Close Count']
                price_risk = price_risk.sort_values('Total Volume', ascending=False).head(5)
                price_risk['Price Range'] = price_risk['Price Range'].astype(str)
                price_risk['Total Volume'] = price_risk['Total Volume'].apply(lambda x: f"${x/1e6:.2f}M")
                
                st.dataframe(price_risk, use_container_width=True, hide_index=True)
                
                st.info("💡 **Interpretation:** These price ranges saw the most position closures. They represent key liquidation zones where leverage is concentrated.")
    
    # ========== TAB 8: Order Flow Imbalance ==========
    with tab8:
        st.markdown('<p class="sub-header">Order Flow Imbalance (OFI) Analysis</p>', 
                    unsafe_allow_html=True)
        st.markdown("Predictive metric: Net buying/selling pressure predicts short-term price movements")
        
        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_coin_ofi = st.selectbox(
                "Select Coin",
                options=df_coins['coin'].head(20).tolist(),
                index=0,
                key='coin_ofi'
            )
        
        with col2:
            time_window_ofi = st.selectbox(
                "Time Window",
                options=['1S', '5S', '10S', '30S', '1T', '5T'],
                index=1,
                key='time_window_ofi'
            )
        
        with col3:
            lag_periods = st.slider(
                "Prediction Lag",
                min_value=1,
                max_value=20,
                value=5,
                help="Periods ahead to predict price change",
                key='lag_periods_ofi'
            )
        
        if selected_coin_ofi:
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center;'><b>OFI Analysis - {selected_coin_ofi}</b></h2>", 
                       unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.spinner("Calculating Order Flow Imbalance..."):
                # Calculate OFI
                df_ofi = loader.calculate_order_flow_imbalance(
                    coin=selected_coin_ofi,
                    time_window=time_window_ofi
                )
                
                ofi_viz = OrderFlowImbalance(df_ofi, df)
                
                # Chart 1: OFI Time Series with Price
                st.markdown("### 📈 OFI Time Series with Price")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    fig1 = ofi_viz.create_ofi_timeseries(selected_coin_ofi)
                    st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Chart 2: OFI vs Future Price Change (Predictive Power)
                st.markdown("### 🎯 OFI Predictive Power")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    fig2 = ofi_viz.create_ofi_price_correlation(selected_coin_ofi, lag_periods)
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Chart 3: Cumulative OFI
                st.markdown("### 📊 Cumulative Flow Pressure")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    fig3 = ofi_viz.create_cumulative_ofi(selected_coin_ofi)
                    st.plotly_chart(fig3, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Chart 4: OFI Distribution
                st.markdown("### 📉 OFI Distribution")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    fig4 = ofi_viz.create_ofi_distribution(selected_coin_ofi)
                    st.plotly_chart(fig4, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interpretation guide
            with st.expander("📖 How to Interpret Order Flow Imbalance", expanded=False):
                st.markdown("""
                ### 🧠 What is OFI?
                
                **Order Flow Imbalance (OFI)** measures net buying/selling pressure by aggregating signed trade volumes:
                - **Positive OFI** = Net buying pressure (more/larger buy trades)
                - **Negative OFI** = Net selling pressure (more/larger sell trades)
                - **Formula:** `OFI = Σ(buy_volume) - Σ(sell_volume)` over time window
                
                ---
                
                ### 📊 Reading the Charts
                
                **1. OFI Time Series (Top Chart)**
                - **Green bars** = Buying pressure (positive OFI)
                - **Red bars** = Selling pressure (negative OFI)
                - **Height** = Magnitude of imbalance
                - **Compare with price**: Does OFI lead price?
                
                **2. OFI Predictive Power (Scatter)**
                - **X-axis**: Current OFI value
                - **Y-axis**: Future price change (N periods ahead)
                - **Regression line**: Shows relationship strength
                - **R²**: Predictive power (higher = better)
                - **Positive slope**: OFI predicts direction
                
                **Key Insight:** If R² > 0.1, OFI has predictive power!
                
                **3. Cumulative OFI**
                - **Rising trend**: Sustained buying pressure
                - **Falling trend**: Sustained selling pressure
                - **Divergence from price**: Potential reversal signal
                
                **4. OFI Distribution**
                - **Skew > 0**: More buying pressure overall
                - **Skew < 0**: More selling pressure overall
                - **Wide spread**: High volatility in flow
                
                ---
                
                ### 🎯 Research Applications
                
                1. **Price Prediction**: Use OFI as leading indicator
                2. **Liquidity Analysis**: Identify when smart money moves
                3. **Market Impact**: Compare OFI to actual price changes
                4. **Regime Detection**: OFI patterns change during different market conditions
                5. **Execution**: Trade when OFI suggests directional momentum
                
                ---
                
                ### 📚 Academic Foundation
                
                OFI is widely studied in market microstructure:
                - Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
                - Used by HFTs for short-term prediction
                - Correlation with future returns typically 0.1-0.3
                """)
    
    # ========== TAB 9: Asset Correlation ==========
    with tab9:
        st.markdown('<p class="sub-header">Multi-Asset Correlation Analysis</p>', 
                    unsafe_allow_html=True)
        st.markdown("Understand how different assets move together")
        
        # Controls
        col1, col2 = st.columns([1, 1])
        
        with col1:
            top_n_corr = st.slider(
                "Number of top coins to analyze",
                min_value=5,
                max_value=30,
                value=15,
                key='top_n_corr'
            )
        
        with col2:
            time_window_corr = st.selectbox(
                "Time Window for Returns",
                options=['30S', '1T', '5T', '15T', '1H'],
                index=1,
                key='time_window_corr'
            )
        
        st.markdown("---")
        st.markdown(f"<h2 style='text-align: center;'><b>Correlation Analysis - Top {top_n_corr} Assets</b></h2>", 
                   unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.spinner("Calculating correlation matrix..."):
            # Calculate correlation
            corr_matrix, returns, prices = loader.calculate_correlation_matrix(
                top_n=top_n_corr,
                time_window=time_window_corr
            )
            
            corr_viz = CorrelationMatrix(corr_matrix, returns, prices)
            
            # Chart 1: Correlation Heatmap
            st.markdown("### 🔥 Correlation Heatmap")
            with st.expander("🔍 Expand for Full Screen View", expanded=True):
                fig1 = corr_viz.create_correlation_heatmap()
                st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Chart 2: Correlation Network
            threshold = st.slider(
                "Network Edge Threshold (min correlation)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key='network_threshold'
            )
            
            st.markdown("### 🕸️ Correlation Network")
            with st.expander("🔍 Expand for Full Screen View", expanded=True):
                fig2 = corr_viz.create_correlation_network(threshold=threshold)
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Chart 3: Lead-Lag Analysis
            st.markdown("### ⏱️ Lead-Lag Analysis")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                coin1_leadlag = st.selectbox(
                    "Coin 1 (Potential Leader)",
                    options=list(corr_matrix.columns),
                    index=0,
                    key='coin1_leadlag'
                )
            
            with col2:
                coin2_leadlag = st.selectbox(
                    "Coin 2 (Potential Follower)",
                    options=list(corr_matrix.columns),
                    index=1 if len(corr_matrix.columns) > 1 else 0,
                    key='coin2_leadlag'
                )
            
            with col3:
                max_lag = st.slider(
                    "Max Lag Periods",
                    min_value=5,
                    max_value=50,
                    value=10,
                    key='max_lag_corr'
                )
            
            if coin1_leadlag != coin2_leadlag:
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    with st.spinner("Calculating lead-lag..."):
                        leadlag_df = loader.calculate_lead_lag_correlation(
                            coin1_leadlag,
                            coin2_leadlag,
                            max_lag=max_lag,
                            time_window='30S'
                        )
                        fig3 = corr_viz.create_lead_lag_chart(leadlag_df, coin1_leadlag, coin2_leadlag)
                        st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Please select two different coins for lead-lag analysis")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Chart 4: Rolling Correlation
            st.markdown("### 📊 Rolling Correlation Over Time")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                coin1_rolling = st.selectbox(
                    "Coin 1",
                    options=list(corr_matrix.columns),
                    index=0,
                    key='coin1_rolling'
                )
            
            with col2:
                coin2_rolling = st.selectbox(
                    "Coin 2",
                    options=list(corr_matrix.columns),
                    index=1 if len(corr_matrix.columns) > 1 else 0,
                    key='coin2_rolling'
                )
            
            with col3:
                rolling_window = st.slider(
                    "Rolling Window",
                    min_value=10,
                    max_value=100,
                    value=20,
                    key='rolling_window'
                )
            
            if coin1_rolling != coin2_rolling:
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    fig4 = corr_viz.create_rolling_correlation(coin1_rolling, coin2_rolling, window=rolling_window)
                    st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("Please select two different coins for rolling correlation")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Chart 5: Diversification Metrics
            st.markdown("### 🎯 Diversification Analysis")
            with st.expander("🔍 Expand for Full Screen View", expanded=True):
                fig5 = corr_viz.create_diversification_metrics()
                st.plotly_chart(fig5, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interpretation guide
        with st.expander("📖 How to Interpret Correlation Analysis", expanded=False):
            st.markdown("""
            ### 🧠 What is Correlation?
            
            **Correlation** measures how two assets move together:
            - **+1.0** = Perfect positive correlation (move in same direction)
            - **0.0** = No correlation (move independently)
            - **-1.0** = Perfect negative correlation (move in opposite directions)
            
            ---
            
            ### 📊 Reading the Charts
            
            **1. Correlation Heatmap**
            - **Red cells**: Strong positive correlation (move together)
            - **Blue cells**: Strong negative correlation (move opposite)
            - **White cells**: No correlation (independent)
            - **Diagonal**: Always 1.0 (asset vs itself)
            
            **2. Correlation Network**
            - **Connected nodes**: Correlated assets (above threshold)
            - **Green edges**: Positive correlation
            - **Red edges**: Negative correlation (rare in crypto)
            - **Larger nodes**: More connections (market leaders)
            - **Clusters**: Assets that move together as a group
            
            **3. Lead-Lag Analysis**
            - **Peak at negative lag**: Coin 2 leads Coin 1
            - **Peak at positive lag**: Coin 1 leads Coin 2
            - **Peak at zero**: They move simultaneously
            - **Use**: Identify which assets are market leaders
            
            **4. Rolling Correlation**
            - **Rising line**: Correlation increasing (coupling)
            - **Falling line**: Correlation decreasing (decoupling)
            - **Spikes**: Temporary high correlation (market events)
            - **Use**: Identify regime changes
            
            **5. Diversification Metrics**
            - **High avg correlation**: Asset moves with the market
            - **Low avg correlation**: Good for diversification
            - **Use**: Build uncorrelated portfolios
            
            ---
            
            ### 🎯 Research Applications
            
            1. **Portfolio Construction**: Choose uncorrelated assets
            2. **Risk Management**: Understand systemic risk
            3. **Market Leadership**: Identify which coins lead vs follow
            4. **Regime Detection**: Spot when correlations break down
            5. **Pair Trading**: Find temporarily decoupled pairs
            
            ---
            
            ### 📚 Key Insights
            
            **High Correlation (>0.7)**:
            - Assets move together
            - No diversification benefit
            - Often linked by fundamentals
            
            **Medium Correlation (0.3-0.7)**:
            - Some co-movement
            - Moderate diversification
            - Market-dependent relationship
            
            **Low Correlation (<0.3)**:
            - Independent movement
            - Good diversification
            - Different market drivers
            
            **Negative Correlation**:
            - Rare in crypto (most correlations positive)
            - Excellent for hedging
            - Often temporary
            
            ---
            
            ### ⚠️ Important Notes
            
            - Correlation ≠ Causation
            - Correlations change over time (use rolling window)
            - High volatility periods often see increased correlations
            - Market-wide events cause temporary spikes
            """)
    
    # ========== TAB 10: Market Impact ==========
    with tab10:
        st.markdown('<p class="sub-header">Market Impact Analysis (Kyle&#39;s Lambda)</p>', 
                    unsafe_allow_html=True)
        st.markdown("Measure execution costs: How much do trades move prices?")
        
        # Controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_coin_impact_analysis = st.selectbox(
                "Select Coin for Detailed Analysis",
                options=['All Coins'] + df_coins['coin'].head(20).tolist(),
                index=0,
                key='coin_impact_analysis'
            )
        
        with col2:
            min_trade_size = st.number_input(
                "Min Trade Size (USD)",
                min_value=10,
                max_value=10000,
                value=100,
                step=50,
                key='min_trade_size'
            )
        
        st.markdown("---")
        st.markdown(f"<h2 style='text-align: center;'><b>Market Impact Analysis - {selected_coin_impact_analysis}</b></h2>", 
                   unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.spinner("Calculating market impact..."):
            # Calculate impact
            coin_filter = None if selected_coin_impact_analysis == 'All Coins' else selected_coin_impact_analysis
            df_impact = loader.calculate_market_impact(coin=coin_filter, min_trade_size=min_trade_size)
            df_lambda = loader.calculate_lambda_by_asset(top_n=20)
            
            impact_viz = MarketImpactAnalysis(df_impact, df_lambda)
            
            # Chart 1: Impact Scatter
            st.markdown("### 📈 Trade Size vs Price Impact")
            with st.expander("🔍 Expand for Full Screen View", expanded=True):
                fig1 = impact_viz.create_impact_scatter(coin=coin_filter)
                st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Chart 2: Lambda Comparison (only for 'All Coins')
            if selected_coin_impact_analysis == 'All Coins':
                st.markdown("### 💎 Kyle's Lambda by Asset")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    fig2 = impact_viz.create_lambda_comparison()
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Chart 3: Impact by Size Bucket
            st.markdown("### 📊 Impact by Trade Size Bucket")
            with st.expander("🔍 Expand for Full Screen View", expanded=True):
                fig3 = impact_viz.create_impact_by_size_bucket(coin=coin_filter)
                st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Chart 4: Temporal Impact
            rolling_window_impact = st.slider(
                "Rolling Window for Temporal Analysis",
                min_value=10,
                max_value=200,
                value=50,
                key='rolling_window_impact'
            )
            
            st.markdown("### ⏱️ Market Impact Over Time")
            with st.expander("🔍 Expand for Full Screen View", expanded=True):
                fig4 = impact_viz.create_temporal_impact(coin=coin_filter, window=rolling_window_impact)
                st.plotly_chart(fig4, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Chart 5: Wallet Execution Efficiency
            if selected_coin_impact_analysis != 'All Coins':
                st.markdown("### 🎯 Wallet Execution Efficiency")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    df_wallet_impact = loader.calculate_impact_by_wallet(coin=coin_filter, top_n=30)
                    fig5 = impact_viz.create_wallet_impact_efficiency(df_wallet_impact)
                    st.plotly_chart(fig5, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Chart 6: Liquidity Quality Matrix (only for 'All Coins')
            if selected_coin_impact_analysis == 'All Coins':
                st.markdown("### 🌟 Liquidity Quality Matrix")
                with st.expander("🔍 Expand for Full Screen View", expanded=True):
                    fig6 = impact_viz.create_liquidity_quality_matrix()
                    st.plotly_chart(fig6, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interpretation guide
        with st.expander("📖 How to Interpret Market Impact Analysis", expanded=False):
            st.markdown("""
            ### 🧠 What is Market Impact?
            
            **Market Impact** measures how much a trade moves the price:
            - **Kyle's Lambda (λ)**: Price change per unit traded
            - **Formula:** `Impact = λ × Trade Size`
            - **Lower impact** = More liquid market, better execution
            
            ---
            
            ### 📊 Reading the Charts
            
            **1. Trade Size vs Price Impact (Scatter)**
            - **X-axis**: Trade size in USD
            - **Y-axis**: How much price changed (%)
            - **Regression line**: Kyle's Lambda estimate
            - **R²**: How well size predicts impact
            - **Color**: Time between trades (liquidity indicator)
            
            **Key Insight:** Steeper slope = Higher impact = Less liquid
            
            **2. Kyle's Lambda by Asset**
            - **Green bars**: Low impact (good liquidity)
            - **Orange bars**: High impact (poor liquidity)
            - **Median line**: Market average
            - **Use**: Choose low-impact assets for large trades
            
            **3. Impact by Trade Size Bucket**
            - Shows how impact changes with trade size
            - **Non-linear**: Larger trades often have disproportionate impact
            - **Error bars**: Impact variability
            - **Use**: Optimal trade sizing
            
            **4. Temporal Impact Analysis**
            - **Top chart**: Rolling average impact over time
            - **Bottom chart**: Trading volume
            - **Rising impact**: Market getting less liquid
            - **Spikes**: Temporary liquidity crunches
            
            **5. Wallet Execution Efficiency**
            - **Lower = Better**: Lower impact per $1M
            - **Bubble size**: Average trade size
            - **Color**: Taker ratio (red = aggressive, green = patient)
            - **Use**: Identify skilled traders
            
            **6. Liquidity Quality Matrix**
            - **Best quadrant**: High volume, low impact (top right to bottom right)
            - **Bubble size**: Number of trades (more = more reliable)
            - **Color**: R² (darker = more predictable impact)
            
            ---
            
            ### 🎯 Research Applications
            
            1. **Execution Cost Analysis**: Estimate slippage before trading
            2. **Liquidity Comparison**: Which assets have deepest markets?
            3. **Optimal Execution**: Size trades to minimize impact
            4. **Market Quality**: Measure market efficiency
            5. **Manipulation Detection**: Abnormal impact = potential manipulation
            
            ---
            
            ### 📚 Key Metrics
            
            **Kyle's Lambda Interpretation:**
            - **<0.1% per $1M**: Excellent liquidity (BTC, ETH typically)
            - **0.1-0.5%**: Good liquidity (major altcoins)
            - **0.5-2.0%**: Moderate liquidity (mid-caps)
            - **>2.0%**: Poor liquidity (small caps, be careful!)
            
            **Execution Efficiency:**
            - **Low impact traders**: Patient, use limit orders, good timing
            - **High impact traders**: Aggressive, market orders, poor timing
            - **Whales with low impact**: Sophisticated execution algorithms
            
            ---
            
            ### ⚠️ Important Notes
            
            - Impact varies by time of day (liquidity cycles)
            - Volatile periods have higher impact
            - Our estimate uses trade-to-trade price changes (simplified)
            - True impact includes temporary vs permanent components
            - Multiple small trades often better than one large trade
            
            ---
            
            ### 💡 Trading Implications
            
            **For Large Orders:**
            1. Check Kyle's Lambda first
            2. Split into smaller pieces (TWAP/VWAP)
            3. Trade during high liquidity periods
            4. Consider impact cost in strategy P&L
            
            **For Market Making:**
            1. Low lambda = tight spreads possible
            2. High lambda = need wider spreads
            3. Monitor temporal changes
            
            **For Research:**
            1. Market efficiency metric
            2. Compare to other exchanges
            3. Test order execution algorithms
            """)
    
    # ========== TAB 11: Data Explorer ==========
    with tab11:
        st.markdown('<p class="sub-header">Data Explorer</p>', 
                    unsafe_allow_html=True)
        
        st.markdown("#### Coin Summary")
        st.dataframe(
            df_coins.head(20)[['coin', 'num_trades', 'num_traders', 'total_notional', 
                              'price_range_pct', 'maker_ratio']].style.format({
                'total_notional': '${:,.0f}',
                'price_range_pct': '{:.2f}%',
                'maker_ratio': '{:.2%}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("#### Raw Trade Data (Sample)")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            filter_coin = st.selectbox(
                "Filter by Coin",
                options=['All'] + df['coin'].unique().tolist()[:20],
                key='filter_coin_explorer'
            )
        with col2:
            filter_side = st.selectbox(
                "Filter by Side",
                options=['All', 'Buy', 'Sell'],
                key='filter_side_explorer'
            )
        
        df_filtered = df.copy()
        if filter_coin != 'All':
            df_filtered = df_filtered[df_filtered['coin'] == filter_coin]
        if filter_side == 'Buy':
            df_filtered = df_filtered[df_filtered['is_buy']]
        elif filter_side == 'Sell':
            df_filtered = df_filtered[df_filtered['is_sell']]
        
        display_cols_raw = ['timestamp', 'coin', 'address', 'px', 'sz', 'side', 
                           'crossed', 'fee', 'closedPnl', 'notional']
        
        df_display_raw = df_filtered[display_cols_raw].head(100).copy()
        df_display_raw['address'] = df_display_raw['address'].str[:12] + '...'
        df_display_raw['timestamp'] = df_display_raw['timestamp'].dt.strftime('%H:%M:%S')
        
        st.dataframe(df_display_raw, use_container_width=True, hide_index=True)
        
        st.info(f"Showing 100 of {len(df_filtered):,} filtered rows")
    
    # ========== TAB 12: Order Book Dynamics ==========
    with tab12:
        st.markdown('<p class="sub-header">📖 Order Book Dynamics (Level 2)</p>', 
                    unsafe_allow_html=True)
        st.markdown("Deep dive into order book microstructure using full L2 data")
        
        # Check if orderbook file exists
        orderbook_path = "Hyperliquid Data Expanded/Hyperliquid_orderbooks.json.gz"
        
        if not os.path.exists(orderbook_path):
            st.warning("⚠️ Order book data file not found!")
            st.info(f"Please ensure `{orderbook_path}` exists in your directory.")
            st.markdown("""
            **This tab requires the orderbook data file:**
            - File: `Hyperliquid_orderbooks.json.gz`
            - Size: ~430MB compressed
            - Events: ~50 million order book events
            """)
        else:
            # Controls
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_coin_ob = st.selectbox(
                    "Select Coin for Order Book Analysis",
                    options=['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'MATIC', 'ARB'],
                    index=0,
                    key='coin_orderbook'
                )
            
            with col2:
                max_events_ob = st.number_input(
                    "Max Events (0=all)",
                    min_value=0,
                    max_value=10000000,
                    value=500000,
                    step=100000,
                    key='max_events_ob',
                    help="Limit events for faster processing (0 = load all)"
                )
            
            with col3:
                sample_interval = st.selectbox(
                    "Sample Interval",
                    options=['1S', '5S', '10S', '30S', '1Min'],
                    index=2,
                    key='sample_interval_ob'
                )
            
            st.markdown("---")
            
            # Load order book data with caching
            @st.cache_data(show_spinner=False)
            def load_orderbook_data(coin, max_events):
                """Load and process orderbook data"""
                ob_loader = OrderBookLoader(orderbook_path)
                
                # Try to load from cache first
                cache_path = f'orderbook_{coin}_{max_events}.parquet'
                if os.path.exists(cache_path):
                    ob_loader.load_processed_data(cache_path)
                else:
                    # Load fresh data
                    ob_loader.load_events(
                        max_events=max_events if max_events > 0 else None,
                        coins_filter=[coin]
                    )
                    # Save cache
                    ob_loader.save_processed_data(cache_path)
                
                return ob_loader
            
            with st.spinner(f"Loading order book data for {selected_coin_ob}..."):
                try:
                    ob_loader = load_orderbook_data(selected_coin_ob, max_events_ob)
                    
                    if ob_loader.df_events is None or len(ob_loader.df_events) == 0:
                        st.warning(f"No order book data found for {selected_coin_ob}")
                    else:
                        st.success(f"✅ Loaded {len(ob_loader.df_events):,} order book events for {selected_coin_ob}")
                        
                        # Reconstruct order book state
                        with st.spinner("Reconstructing order book state..."):
                            df_snapshots = ob_loader.reconstruct_order_book(
                                selected_coin_ob,
                                sample_interval=sample_interval
                            )
                        
                        if len(df_snapshots) == 0:
                            st.warning("No valid order book snapshots generated")
                        else:
                            st.success(f"✅ Generated {len(df_snapshots)} order book snapshots")
                            
                            # Create OrderBookLadderVisualization instance
                            ladder_viz = OrderBookLadderVisualization(ob_loader)
                            
                            # ===== FEATURED: Animated Order Book Ladder =====
                            st.markdown("## 🎬 Animated Order Book Ladder")
                            st.markdown("Watch the order book evolve in real-time with play/pause controls")
                            
                            # Animation controls
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                num_frames = st.slider(
                                    "Animation Frames",
                                    min_value=10,
                                    max_value=100,
                                    value=50,
                                    help="More frames = smoother but slower",
                                    key='animation_frames'
                                )
                            with col2:
                                num_ladder_levels = st.slider(
                                    "Price Levels",
                                    min_value=5,
                                    max_value=30,
                                    value=15,
                                    help="Number of price levels to show",
                                    key='ladder_levels'
                                )
                            
                            with st.expander("🎬 Animated Order Book (Play to Start)", expanded=True):
                                with st.spinner("Creating animation frames..."):
                                    # Sample timestamps for animation
                                    all_timestamps = df_snapshots['timestamp'].tolist()
                                    step = max(1, len(all_timestamps) // num_frames)
                                    animation_timestamps = all_timestamps[::step]
                                    
                                    if len(animation_timestamps) > 0:
                                        fig_animated = ladder_viz.create_animated_order_book(
                                            selected_coin_ob,
                                            animation_timestamps,
                                            num_levels=num_ladder_levels
                                        )
                                        st.plotly_chart(fig_animated, use_container_width=True)
                                        
                                        st.info(f"🎬 {len(animation_timestamps)} frames | Click ▶ Play to start animation")
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # ===== FEATURED: Static Order Book Ladder (Current State) =====
                            st.markdown("## 📊 Order Book Ladder (Current State)")
                            st.markdown("Traditional order book view showing bid/ask depth at each price level")
                            
                            with st.expander("📊 Current Order Book Ladder", expanded=True):
                                with st.spinner("Rendering order book ladder..."):
                                    # Use the last timestamp for "current" state
                                    current_timestamp = df_snapshots['timestamp'].iloc[-1]
                                    
                                    fig_ladder = ladder_viz.create_order_book_ladder(
                                        selected_coin_ob,
                                        current_timestamp,
                                        num_levels=25
                                    )
                                    st.plotly_chart(fig_ladder, use_container_width=True)
                            
                            st.markdown("---")
                            st.markdown("## 📈 Additional Microstructure Analysis")
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # ===== Chart 1: Depth Evolution =====
                            st.markdown("### 📈 Order Book Depth Evolution")
                            with st.expander("🔍 Expand for Full Screen View", expanded=False):
                                depth_chart = OrderBookDepthChart(df_snapshots)
                                fig1 = depth_chart.create_depth_evolution_chart()
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # ===== Chart 2: Depth Distribution =====
                            st.markdown("### 📊 Bid/Ask Depth & Imbalance")
                            with st.expander("🔍 Expand for Full Screen View", expanded=True):
                                fig2 = depth_chart.create_depth_distribution_chart()
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # ===== Chart 3: Imbalance Analysis =====
                            st.markdown("### ⚖️ Order Book Imbalance Analysis")
                            with st.expander("🔍 Expand for Full Screen View", expanded=True):
                                imbalance_viz = ImbalanceAnalysis(df_snapshots)
                                fig3 = imbalance_viz.create_imbalance_timeseries()
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # ===== Chart 4: Spread Analysis =====
                            st.markdown("### 📏 Spread Dynamics")
                            with st.expander("🔍 Expand for Full Screen View", expanded=False):
                                spread_viz = SpreadAnalysisOrderBook(df_snapshots)
                                fig4 = spread_viz.create_spread_analysis_chart()
                                st.plotly_chart(fig4, use_container_width=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # ===== Chart 5: Liquidity Heatmap =====
                            st.markdown("### 🔥 Liquidity Heatmap (Price × Time)")
                            with st.expander("🔍 Expand for Full Screen View", expanded=False):
                                with st.spinner("Generating liquidity heatmap..."):
                                    df_heatmap = ob_loader.get_liquidity_heatmap_data(
                                        selected_coin_ob,
                                        price_range_pct=2.0,
                                        num_bins=50
                                    )
                                    
                                    if len(df_heatmap) > 0:
                                        heatmap_viz = LiquidityHeatmap(df_heatmap)
                                        fig5 = heatmap_viz.create_heatmap()
                                        st.plotly_chart(fig5, use_container_width=True)
                                    else:
                                        st.warning("Not enough data for heatmap visualization")
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # ===== Chart 6: Order Lifecycle =====
                            st.markdown("### ⏱️ Order Lifecycle Analysis")
                            with st.expander("🔍 Expand for Full Screen View", expanded=False):
                                with st.spinner("Analyzing order lifecycles..."):
                                    df_lifecycle = ob_loader.get_order_lifecycle_stats(selected_coin_ob)
                                    
                                    if len(df_lifecycle) > 0:
                                        lifecycle_viz = OrderLifecycleAnalysis(df_lifecycle)
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            fig6a = lifecycle_viz.create_outcome_breakdown()
                                            st.plotly_chart(fig6a, use_container_width=True)
                                        with col2:
                                            fig6b = lifecycle_viz.create_fill_rate_by_side()
                                            st.plotly_chart(fig6b, use_container_width=True)
                                        
                                        fig6c = lifecycle_viz.create_lifetime_distribution()
                                        st.plotly_chart(fig6c, use_container_width=True)
                                    else:
                                        st.warning("No lifecycle data available")
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Interpretation Guide
                            with st.expander("📖 How to Interpret Order Book Dynamics", expanded=False):
                                st.markdown("""
                                ### 🧠 Understanding Order Book Microstructure
                                
                                **What is Level 2 Order Book Data?**
                                - Shows ALL limit orders at different price levels
                                - Updates in real-time as orders are placed/filled/canceled
                                - More detailed than just trade data (Level 1)
                                
                                ---
                                
                                ### 📊 Reading the Charts
                                
                                **1. Depth Evolution Chart**
                                - **Best Bid/Ask**: Tightest prices where orders exist
                                - **Mid-Price**: Average of best bid and ask
                                - **Spread**: Cost of immediate execution
                                - **Narrow spread** = Liquid market, low trading cost
                                
                                **2. Depth & Imbalance**
                                - **Bid Depth**: Total buy orders within 1% of mid
                                - **Ask Depth**: Total sell orders within 1% of mid
                                - **Imbalance**: `(Bid - Ask) / (Bid + Ask)`
                                  - Positive = More bids (bullish pressure)
                                  - Negative = More asks (bearish pressure)
                                
                                **3. Imbalance Analysis**
                                - **Green bars**: More buying pressure
                                - **Red bars**: More selling pressure
                                - **Extreme values** (>0.5 or <-0.5): Strong directional pressure
                                - Often predicts short-term price moves
                                
                                **4. Spread Dynamics**
                                - **Absolute spread**: Dollar cost of crossing
                                - **% spread**: Relative cost (better for comparison)
                                - **Tighter spreads**: More market makers active
                                - **Wider spreads**: Volatile or thin market
                                
                                **5. Liquidity Heatmap**
                                - **Hot colors** (red/yellow): High liquidity concentration
                                - **Cold colors** (blue): Sparse liquidity
                                - **Horizontal bands**: Price levels with persistent orders
                                - **Gaps**: Price levels avoiding (resistance/support)
                                
                                **6. Order Lifecycle**
                                - **Filled orders**: Successful execution
                                - **Canceled orders**: Changed mind or adjusted
                                - **Rejected orders**: Invalid (price too aggressive)
                                - **Short lifetime**: Aggressive orders or fast market
                                - **Long lifetime**: Patient makers or slow market
                                
                                ---
                                
                                ### 🎯 Research Applications
                                
                                1. **Market Making Strategy**
                                   - Where do successful orders get placed?
                                   - How long before they get filled?
                                   - Optimal queue position?
                                
                                2. **Liquidity Analysis**
                                   - When is the market most/least liquid?
                                   - Where are the "walls" of liquidity?
                                   - Support/resistance levels?
                                
                                3. **Price Prediction**
                                   - Does imbalance predict price moves?
                                   - Lead-lag relationship?
                                   - Early warning signals?
                                
                                4. **Execution Quality**
                                   - Best time to execute large orders?
                                   - Expected slippage?
                                   - Optimal order size?
                                
                                5. **Manipulation Detection**
                                   - Spoofing (place then cancel)?
                                   - Layering (false depth)?
                                   - Quote stuffing (rapid place/cancel)?
                                
                                ---
                                
                                ### 💡 Key Insights
                                
                                **Order Book Imbalance:**
                                - Strong predictor of short-term (1-10 second) price moves
                                - More reliable when combined with trade flow
                                - Reset after large trades
                                
                                **Spread Behavior:**
                                - Widens before large price moves (uncertainty)
                                - Tightens in stable markets (competition)
                                - Spikes during liquidity crunches
                                
                                **Order Placement:**
                                - Most orders placed near best bid/ask
                                - Patient makers place deeper in book
                                - Aggressive orders cross spread immediately
                                
                                **Lifecycle Patterns:**
                                - ~60-80% of limit orders get canceled (typical)
                                - Median lifetime: 5-30 seconds (varies by coin)
                                - Bids have different fill rates than asks
                                
                                ---
                                
                                ### 📚 Academic References
                                
                                1. **Cont, R., Kukanov, A., & Stoikov, S. (2014)**: "The Price Impact of Order Book Events"
                                2. **Gould, M. D., et al. (2013)**: "Limit order books"
                                3. **Hasbrouck, J. (2007)**: "Empirical Market Microstructure"
                                
                                ---
                                
                                ### ⚠️ Important Notes
                                
                                - This data is historical (research purposes)
                                - Real-time requires live data feed
                                - Patterns vary by time of day and market conditions
                                - Always validate findings statistically
                                """)
                
                except Exception as e:
                    st.error(f"Error loading order book data: {str(e)}")
                    st.info("Try reducing the max events parameter or selecting a different coin")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Hyperliquid Market Microstructure Dashboard | Phase 1 Prototype</p>
        <p>Data Source: Hyperliquid L1 Blockchain | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

