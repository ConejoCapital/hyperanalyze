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
    LiquidationAnalysis
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
    
    # Check if processed data exists
    processed_path = 'processed_data.parquet'
    
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
    st.sidebar.header("Data Settings")
    
    data_path = st.sidebar.text_input(
        "Data Path",
        value="/Users/thebunnymac/Desktop/hyperorderbook/Hyperliquid Data Expanded/node_fills_20251027_1700-1800.json",
        help="Use node_fills for trade data, misc_events for other event types"
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔥 Order Book Heatmap",
        "🔄 Maker vs Taker Flow",
        "📏 Spread Analysis",
        "📊 Volume Profile",
        "👥 Trader Analytics",
        "🎯 Wallet Impact",
        "💥 Liquidations",
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
    
    # ========== TAB 8: Data Explorer ==========
    with tab8:
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

