"""
Phase 1 Visualizations for Hyperliquid Market Microstructure Analysis
- Dynamic Order Book Heatmap
- Maker vs Taker Flow Analysis  
- Spread Analysis Dashboard
- Volume Profile by Asset
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class OrderBookHeatmap:
    """Dynamic Order Book Heatmap Visualization"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def create_heatmap(self, coin: str, time_resolution: str = '10S', 
                      price_bins: int = 50) -> go.Figure:
        """
        Create time x price x depth heatmap
        
        Args:
            coin: Trading pair to visualize
            time_resolution: Time binning (e.g., '10S', '1T')
            price_bins: Number of price levels
            
        Returns:
            Plotly figure
        """
        # Filter for coin
        df_coin = self.df[self.df['coin'] == coin].copy()
        
        if df_coin.empty:
            return self._empty_figure(f"No data for {coin}")
        
        # Set timestamp as index
        df_coin = df_coin.set_index('timestamp')
        
        # Get price range
        price_min = df_coin['px'].quantile(0.01)
        price_max = df_coin['px'].quantile(0.99)
        price_range = np.linspace(price_min, price_max, price_bins)
        
        # Time range
        time_range = pd.date_range(
            start=df_coin.index.min(),
            end=df_coin.index.max(),
            freq=time_resolution
        )
        
        # Build heatmap matrix
        heatmap_matrix = np.zeros((len(price_range), len(time_range)))
        bid_matrix = np.zeros((len(price_range), len(time_range)))
        ask_matrix = np.zeros((len(price_range), len(time_range)))
        
        for time_idx, time_point in enumerate(time_range[:-1]):
            # Get trades in this time window
            window = df_coin.loc[time_point:time_range[time_idx + 1]]
            
            if len(window) == 0:
                continue
            
            # Accumulate volume at each price level
            for price_idx, price_level in enumerate(price_range[:-1]):
                # Trades in this price range
                price_mask = (window['px'] >= price_level) & (window['px'] < price_range[price_idx + 1])
                trades_at_level = window[price_mask]
                
                if len(trades_at_level) > 0:
                    total_volume = trades_at_level['sz'].sum()
                    bid_volume = trades_at_level[trades_at_level['is_buy']]['sz'].sum()
                    ask_volume = trades_at_level[trades_at_level['is_sell']]['sz'].sum()
                    
                    heatmap_matrix[price_idx, time_idx] = total_volume
                    bid_matrix[price_idx, time_idx] = bid_volume
                    ask_matrix[price_idx, time_idx] = ask_volume
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f'{coin} Order Book Depth Evolution',
                'Bid/Ask Imbalance'
            ),
            vertical_spacing=0.12
        )
        
        # Main heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_matrix,
                x=time_range,
                y=price_range,
                colorscale='Viridis',
                name='Depth',
                colorbar=dict(title='Volume', x=1.02)
            ),
            row=1, col=1
        )
        
        # Calculate imbalance
        total_vol = bid_matrix + ask_matrix
        imbalance = np.divide(
            bid_matrix - ask_matrix,
            total_vol,
            where=total_vol > 0,
            out=np.zeros_like(total_vol)
        )
        
        # Imbalance over time (aggregate across price levels)
        imbalance_time_series = np.average(imbalance, axis=0, weights=total_vol + 1e-10)
        
        fig.add_trace(
            go.Scatter(
                x=time_range,
                y=imbalance_time_series,
                mode='lines',
                name='Imbalance',
                line=dict(color='orange', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Imbalance", row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f"{coin} Market Depth Heatmap",
            title_x=0.5,
            hovermode='closest'
        )
        
        return fig
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig


class MakerTakerFlow:
    """Maker vs Taker Flow Analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_flow_analysis(self, coin: Optional[str] = None, 
                           time_resolution: str = '1T') -> go.Figure:
        """
        Create maker vs taker flow visualization
        
        Args:
            coin: Optional coin filter (None = all coins)
            time_resolution: Time binning
            
        Returns:
            Plotly figure
        """
        df_filtered = self.df if coin is None else self.df[self.df['coin'] == coin]
        
        if df_filtered.empty:
            return self._empty_figure("No data")
        
        # Resample by time
        df_time = df_filtered.set_index('timestamp')
        
        # Aggregate maker/taker stats
        resampled = df_time.resample(time_resolution).agg({
            'sz': 'sum',
            'notional': 'sum',
            'is_maker': 'sum',
            'is_taker': 'sum',
            'fee': 'sum'
        })
        
        # Calculate volumes
        maker_vol = []
        taker_vol = []
        
        for timestamp, group in df_time.resample(time_resolution):
            maker_trades = group[group['is_maker']]
            taker_trades = group[group['is_taker']]
            
            maker_vol.append(maker_trades['notional'].sum())
            taker_vol.append(taker_trades['notional'].sum())
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Maker vs Taker Volume Over Time',
                'Maker Ratio',
                'Cumulative Fee Flow',
                'Trade Count Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        timestamps = resampled.index
        
        # 1. Stacked area chart
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=maker_vol,
                name='Maker Volume',
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(0, 176, 246, 0.6)',
                line=dict(width=0.5, color='rgb(0, 176, 246)')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=taker_vol,
                name='Taker Volume',
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(231, 107, 243, 0.6)',
                line=dict(width=0.5, color='rgb(231, 107, 243)')
            ),
            row=1, col=1
        )
        
        # 2. Maker ratio
        total_vol_arr = np.array(maker_vol) + np.array(taker_vol)
        maker_ratio = np.divide(
            maker_vol,
            total_vol_arr,
            where=total_vol_arr > 0,
            out=np.zeros_like(maker_vol, dtype=float)
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=maker_ratio,
                name='Maker Ratio',
                mode='lines',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=2)
        
        # 3. Cumulative fees
        cumulative_fees = resampled['fee'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cumulative_fees,
                name='Cumulative Fees',
                mode='lines',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # 4. Pie chart
        total_maker = df_filtered['is_maker'].sum()
        total_taker = df_filtered['is_taker'].sum()
        
        fig.add_trace(
            go.Pie(
                labels=['Maker', 'Taker'],
                values=[total_maker, total_taker],
                marker=dict(colors=['rgb(0, 176, 246)', 'rgb(231, 107, 243)']),
                hole=0.4
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        fig.update_yaxes(title_text="Volume (USDC)", row=1, col=1)
        fig.update_yaxes(title_text="Ratio", row=1, col=2, range=[0, 1])
        fig.update_yaxes(title_text="Cumulative Fees (USDC)", row=2, col=1)
        
        title = f"Maker vs Taker Flow Analysis{' - ' + coin if coin else ''}"
        fig.update_layout(
            height=800,
            title_text=title,
            title_x=0.5,
            showlegend=True
        )
        
        return fig
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig


class SpreadAnalysis:
    """Spread Analysis Dashboard"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_spread_dashboard(self, coins: List[str], 
                               time_resolution: str = '30S') -> go.Figure:
        """
        Create comprehensive spread analysis dashboard
        
        Args:
            coins: List of coins to analyze
            time_resolution: Time binning
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Spread Over Time (bps)',
                'Spread vs Volume',
                'Spread Distribution',
                'Cross-Asset Spread Comparison'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "histogram"}, {"type": "bar"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        colors = px.colors.qualitative.Set2
        spread_data = {}
        
        for idx, coin in enumerate(coins):
            df_coin = self.df[self.df['coin'] == coin].copy()
            
            if df_coin.empty:
                continue
            
            # Calculate spread metrics
            df_coin = df_coin.set_index('timestamp')
            
            spread_series = []
            volume_series = []
            timestamps = []
            
            for timestamp, group in df_coin.resample(time_resolution):
                if len(group) == 0:
                    continue
                
                buys = group[group['is_buy']]
                sells = group[group['is_sell']]
                
                if len(buys) > 0 and len(sells) > 0:
                    best_bid = buys['px'].max()
                    best_ask = sells['px'].min()
                    mid = (best_bid + best_ask) / 2
                    spread = best_ask - best_bid
                    spread_bps = (spread / mid) * 10000
                    
                    spread_series.append(spread_bps)
                    volume_series.append(group['notional'].sum())
                    timestamps.append(timestamp)
            
            if len(spread_series) == 0:
                continue
            
            spread_data[coin] = {
                'timestamps': timestamps,
                'spreads': spread_series,
                'volumes': volume_series
            }
            
            color = colors[idx % len(colors)]
            
            # 1. Spread time series
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=spread_series,
                    name=coin,
                    mode='lines',
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )
            
            # 2. Spread vs Volume scatter
            fig.add_trace(
                go.Scatter(
                    x=volume_series,
                    y=spread_series,
                    name=coin,
                    mode='markers',
                    marker=dict(color=color, size=6, opacity=0.6)
                ),
                row=1, col=2
            )
            
            # 3. Spread distribution
            fig.add_trace(
                go.Histogram(
                    x=spread_series,
                    name=coin,
                    marker=dict(color=color),
                    opacity=0.6,
                    nbinsx=30
                ),
                row=2, col=1
            )
        
        # 4. Average spread comparison bar chart
        avg_spreads = {coin: np.mean(data['spreads']) 
                      for coin, data in spread_data.items()}
        
        fig.add_trace(
            go.Bar(
                x=list(avg_spreads.keys()),
                y=list(avg_spreads.values()),
                marker=dict(color=colors[:len(avg_spreads)]),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Volume (USDC)", row=1, col=2)
        fig.update_xaxes(title_text="Spread (bps)", row=2, col=1)
        fig.update_xaxes(title_text="Coin", row=2, col=2)
        
        fig.update_yaxes(title_text="Spread (bps)", row=1, col=1)
        fig.update_yaxes(title_text="Spread (bps)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Avg Spread (bps)", row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text="Spread Analysis Dashboard",
            title_x=0.5,
            showlegend=True,
            barmode='overlay'
        )
        
        return fig


class VolumeProfile:
    """Volume Profile Visualization"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_volume_profile(self, coin: str, price_bins: int = 50) -> go.Figure:
        """
        Create volume profile (volume at price) visualization
        
        Args:
            coin: Trading pair
            price_bins: Number of price levels
            
        Returns:
            Plotly figure
        """
        df_coin = self.df[self.df['coin'] == coin].copy()
        
        if df_coin.empty:
            return self._empty_figure(f"No data for {coin}")
        
        # Create price bins
        price_min = df_coin['px'].quantile(0.01)
        price_max = df_coin['px'].quantile(0.99)
        
        df_coin['price_bin'] = pd.cut(df_coin['px'], bins=price_bins)
        
        # Calculate volume at each price level
        volume_profile = df_coin.groupby('price_bin').agg({
            'sz': 'sum',
            'notional': 'sum'
        }).reset_index()
        
        # Separate buyer vs seller initiated
        buyer_initiated = df_coin[df_coin['aggressor_side'] == 'buy'].groupby('price_bin')['notional'].sum()
        seller_initiated = df_coin[df_coin['aggressor_side'] == 'sell'].groupby('price_bin')['notional'].sum()
        
        # Get price midpoints for plotting
        price_midpoints = volume_profile['price_bin'].apply(lambda x: x.mid)
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            subplot_titles=(f'{coin} Volume Profile', 'Price Distribution'),
            horizontal_spacing=0.1
        )
        
        # Main volume profile (horizontal bars)
        fig.add_trace(
            go.Bar(
                y=price_midpoints,
                x=buyer_initiated.reindex(volume_profile['price_bin']).fillna(0),
                name='Buyer Initiated',
                orientation='h',
                marker=dict(color='green', opacity=0.7)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                y=price_midpoints,
                x=seller_initiated.reindex(volume_profile['price_bin']).fillna(0),
                name='Seller Initiated',
                orientation='h',
                marker=dict(color='red', opacity=0.7)
            ),
            row=1, col=1
        )
        
        # Point of Control (POC) - price level with most volume
        poc_idx = volume_profile['notional'].idxmax()
        poc_price = price_midpoints.iloc[poc_idx]
        
        fig.add_hline(
            y=poc_price,
            line_dash="dash",
            line_color="yellow",
            line_width=3,
            annotation_text="POC",
            row=1, col=1
        )
        
        # Value Area (70% of volume)
        volume_profile_sorted = volume_profile.sort_values('notional', ascending=False)
        cumsum = volume_profile_sorted['notional'].cumsum()
        total_volume = volume_profile_sorted['notional'].sum()
        value_area = volume_profile_sorted[cumsum <= 0.7 * total_volume]
        
        # Price distribution
        fig.add_trace(
            go.Histogram(
                x=df_coin['px'],
                nbinsx=30,
                marker=dict(color='blue', opacity=0.6),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Volume (USDC)", row=1, col=1)
        fig.update_xaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=2)
        
        fig.update_layout(
            height=700,
            title_text=f"{coin} Volume Profile",
            title_x=0.5,
            showlegend=True,
            barmode='overlay'
        )
        
        # Add statistics annotation
        stats_text = f"POC: ${poc_price:.4f}<br>Total Volume: {total_volume:,.0f} USDC<br>Num Trades: {len(df_coin)}"
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        return fig
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig


class OrderbookImpactAnalysis:
    """Analyze which wallets move the orderbook the most"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def _calculate_wallet_impact(self, coin: str, top_n: int = 15) -> pd.DataFrame:
        """
        Calculate wallet impact metrics
        
        Args:
            coin: Trading pair to analyze
            top_n: Number of top impactful wallets to show
            
        Returns:
            DataFrame with wallet impact metrics
        """
        df_coin = self.df[self.df['coin'] == coin].copy()
        
        if df_coin.empty:
            return pd.DataFrame()
        
        # Calculate impact metrics per wallet
        wallet_impact = []
        
        for address, trades in df_coin.groupby('address'):
            # Sort by time
            trades = trades.sort_values('timestamp')
            
            # Calculate price impact (price change magnitude)
            trades['price_change'] = trades['px'].diff().abs()
            trades['price_change_pct'] = (trades['px'].pct_change().abs() * 100)
            
            # Calculate orderbook movement (large trades that cross spread)
            aggressive_trades = trades[trades['crossed'] == True]
            
            impact = {
                'address': address,
                'total_volume': trades['notional'].sum(),
                'num_trades': len(trades),
                'avg_trade_size': trades['notional'].mean(),
                'max_trade_size': trades['notional'].max(),
                
                # Impact metrics
                'total_price_movement': trades['price_change'].sum(),
                'avg_price_impact': trades['price_change_pct'].mean(),
                'max_price_impact': trades['price_change_pct'].max(),
                
                # Aggressive trading (orderbook consumption)
                'aggressive_trades': len(aggressive_trades),
                'aggressive_volume': aggressive_trades['notional'].sum(),
                'aggressive_ratio': len(aggressive_trades) / len(trades) if len(trades) > 0 else 0,
                
                # Market maker or taker
                'taker_ratio': trades['crossed'].sum() / len(trades),
            }
            
            wallet_impact.append(impact)
        
        df_impact = pd.DataFrame(wallet_impact)
        df_impact = df_impact.sort_values('total_volume', ascending=False).head(top_n)
        df_impact['short_address'] = df_impact['address'].str[:10] + '...'
        
        return df_impact
    
    def create_volume_chart(self, coin: str, top_n: int = 15) -> go.Figure:
        """Create top wallets by volume chart"""
        df_impact = self._calculate_wallet_impact(coin, top_n)
        
        if df_impact.empty:
            return self._empty_figure(f"No data for {coin}")
        
        # Top wallets by volume (colored by taker ratio)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_impact['short_address'],
                y=df_impact['total_volume'],
                marker=dict(
                    color=df_impact['taker_ratio'],
                    colorscale='RdYlGn_r',  # Red = high taker (aggressive), Green = low taker (maker)
                    showscale=True,
                    colorbar=dict(title="Taker Ratio"),
                    cmin=0,
                    cmax=1
                ),
                text=df_impact['total_volume'].apply(lambda x: f'${x/1e6:.2f}M'),
                textposition='auto',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Volume: $%{y:,.0f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Top {top_n} Market Movers by Volume - {coin}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Wallet",
            yaxis_title="Total Volume (USDC)",
            yaxis_type='log',
            height=600,
            xaxis_tickangle=45,
            template='plotly_dark'
        )
        
        return fig
    
    def create_price_impact_chart(self, coin: str, top_n: int = 15) -> go.Figure:
        """Create price impact scatter chart"""
        df_impact = self._calculate_wallet_impact(coin, top_n)
        
        if df_impact.empty:
            return self._empty_figure(f"No data for {coin}")
        
        # Average price impact vs trade count
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_impact['num_trades'],
                y=df_impact['avg_price_impact'],
                mode='markers',
                marker=dict(
                    size=df_impact['avg_trade_size'] / 1000,  # Size by avg trade
                    color=df_impact['taker_ratio'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Taker Ratio"),
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=df_impact['short_address'],
                hovertemplate='<b>%{text}</b><br>Trades: %{x}<br>Avg Impact: %{y:.3f}%<extra></extra>',
                showlegend=False
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Average Price Impact per Trade - {coin}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Number of Trades",
            yaxis_title="Avg Price Impact (%)",
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def create_aggressive_volume_chart(self, coin: str, top_n: int = 15) -> go.Figure:
        """Create aggressive volume bar chart"""
        df_impact = self._calculate_wallet_impact(coin, top_n)
        
        if df_impact.empty:
            return self._empty_figure(f"No data for {coin}")
        
        # Aggressive volume (who consumes the most orderbook)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_impact['short_address'],
                y=df_impact['aggressive_volume'],
                marker=dict(color='rgba(255, 99, 71, 0.7)'),
                text=df_impact['aggressive_volume'].apply(lambda x: f'${x/1e6:.2f}M' if x > 1e6 else f'${x/1e3:.0f}K'),
                textposition='auto',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Aggressive Vol: $%{y:,.0f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Aggressive Trading Volume (Orderbook Consumption) - {coin}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Wallet",
            yaxis_title="Aggressive Volume (USDC)",
            yaxis_type='log',
            height=600,
            xaxis_tickangle=45,
            template='plotly_dark'
        )
        
        return fig
    
    def create_trade_size_distribution(self, coin: str, top_n: int = 15) -> go.Figure:
        """Create trade size distribution box plots"""
        df_impact = self._calculate_wallet_impact(coin, top_n)
        df_coin = self.df[self.df['coin'] == coin].copy()
        
        if df_impact.empty or df_coin.empty:
            return self._empty_figure(f"No data for {coin}")
        
        # Trade size distribution (box plots for top 5)
        fig = go.Figure()
        
        for i, row in df_impact.head(5).iterrows():
            trader_trades = df_coin[df_coin['address'] == row['address']]
            fig.add_trace(
                go.Box(
                    y=trader_trades['notional'],
                    name=row['short_address'],
                    marker=dict(opacity=0.7)
                )
            )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Trade Size Distribution (Top 5 Wallets) - {coin}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Wallet",
            yaxis_title="Trade Size (USDC)",
            yaxis_type='log',
            height=600,
            xaxis_tickangle=45,
            template='plotly_dark',
            showlegend=True
        )
        
        return fig
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig


class LiquidationAnalysis:
    """Analyze liquidation events and price points"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def _get_liquidation_data(self, coin: str):
        """
        Get liquidation data for a coin
        
        Args:
            coin: Trading pair to analyze
            
        Returns:
            Tuple of (df_coin, large_closes, closes)
        """
        df_coin = self.df[self.df['coin'] == coin].copy()
        
        if df_coin.empty:
            return None, None, None
        
        # Identify potential liquidations
        # Liquidations are characterized by:
        # 1. Large position changes
        # 2. Direction indicating forced closure
        # 3. Often at specific price levels
        
        df_coin['position_size_change'] = df_coin['sz'].abs()
        df_coin['is_large_close'] = (
            (df_coin['dir'].str.contains('Close', na=False)) & 
            (df_coin['notional'] > df_coin['notional'].quantile(0.75))
        )
        
        # Identify potential liquidation-prone price levels
        # These are areas with lots of closing trades
        closes = df_coin[df_coin['dir'].str.contains('Close', na=False)].copy()
        large_closes = df_coin[df_coin['is_large_close']].copy()
        
        return df_coin, large_closes, closes
    
    def create_liquidation_scatter(self, coin: str) -> go.Figure:
        """Create liquidation hotspot scatter plot"""
        df_coin, large_closes, closes = self._get_liquidation_data(coin)
        
        if df_coin is None:
            return self._empty_figure(f"No data for {coin}")
        
        fig = go.Figure()
        
        if not large_closes.empty:
            fig.add_trace(
                go.Scatter(
                    x=large_closes['timestamp'],
                    y=large_closes['px'],
                    mode='markers',
                    marker=dict(
                        size=large_closes['notional'] / 10000,  # Scale for visibility
                        color=large_closes['closedPnl'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="P&L (USDC)"),
                        opacity=0.6,
                        line=dict(width=1, color='white')
                    ),
                    text=large_closes.apply(
                        lambda x: f"${x['notional']:,.0f}<br>{x['dir']}<br>P&L: ${x['closedPnl']:.2f}",
                        axis=1
                    ),
                    hovertemplate='<b>%{text}</b><br>Price: $%{y:.4f}<br>%{x}<extra></extra>',
                    showlegend=False
                )
            )
            
            # Add summary statistics annotation
            total_liq_volume = large_closes['notional'].sum()
            avg_liq_size = large_closes['notional'].mean()
            liq_count = len(large_closes)
            
            fig.add_annotation(
                text=f"<b>Stats:</b><br>" + 
                     f"Total: ${total_liq_volume/1e6:.2f}M<br>" +
                     f"Count: {liq_count}<br>" +
                     f"Avg: ${avg_liq_size:,.0f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=2,
                font=dict(size=10),
                align='left'
            )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Liquidation Hotspot Heatmap - {coin}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Time",
            yaxis_title="Price",
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def create_price_histogram(self, coin: str) -> go.Figure:
        """Create histogram of close prices"""
        df_coin, large_closes, closes = self._get_liquidation_data(coin)
        
        if df_coin is None or closes.empty:
            return self._empty_figure(f"No data for {coin}")
        
        fig = go.Figure()
        
        # Histogram of close prices
        fig.add_trace(
            go.Histogram(
                x=closes['px'],
                nbinsx=50,
                marker=dict(color='rgba(255, 107, 107, 0.7)'),
                showlegend=False,
                hovertemplate='Price: $%{x:.4f}<br>Count: %{y}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Large Closes by Price Level - {coin}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Price",
            yaxis_title="Count",
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def create_cumulative_volume(self, coin: str) -> go.Figure:
        """Create cumulative close volume chart"""
        df_coin, large_closes, closes = self._get_liquidation_data(coin)
        
        if df_coin is None or closes.empty:
            return self._empty_figure(f"No data for {coin}")
        
        fig = go.Figure()
        
        # Close volume over time
        closes_sorted = closes.sort_values('timestamp')
        closes_sorted['cumulative_close_volume'] = closes_sorted['notional'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=closes_sorted['timestamp'],
                y=closes_sorted['cumulative_close_volume'],
                mode='lines',
                line=dict(color='red', width=2),
                fill='tozeroy',
                showlegend=False,
                hovertemplate='Time: %{x}<br>Cumulative: $%{y:,.0f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Position Close Volume Over Time - {coin}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Time",
            yaxis_title="Cumulative Volume (USDC)",
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def create_close_distribution(self, coin: str) -> go.Figure:
        """Create close direction distribution bar chart"""
        df_coin, large_closes, closes = self._get_liquidation_data(coin)
        
        if df_coin is None or closes.empty:
            return self._empty_figure(f"No data for {coin}")
        
        fig = go.Figure()
        
        # Close direction breakdown
        close_dirs = closes['dir'].value_counts().head(10)
        
        fig.add_trace(
            go.Bar(
                x=close_dirs.index,
                y=close_dirs.values,
                marker=dict(
                    color=close_dirs.values,
                    colorscale='Reds',
                    showscale=False
                ),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Close Direction Distribution - {coin}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Close Type",
            yaxis_title="Count",
            xaxis_tickangle=45,
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig


class TraderAnalytics:
    """Account-level trader analytics visualization"""
    
    def __init__(self, df_traders: pd.DataFrame, df_trades: pd.DataFrame):
        self.df_traders = df_traders
        self.df_trades = df_trades
    
    def create_trader_dashboard(self, top_n: int = 20) -> go.Figure:
        """
        Create trader analytics dashboard
        
        Args:
            top_n: Number of top traders to show
            
        Returns:
            Plotly figure
        """
        top_traders = self.df_traders.head(top_n)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Top {top_n} Traders by Volume',
                'Maker Ratio Distribution',
                'Trade Count vs Volume',
                'P&L Distribution'
            ),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "box"}]],
            vertical_spacing=0.25,  # Increased from 0.15 for more space between rows
            horizontal_spacing=0.15  # Slightly increased horizontal spacing too
        )
        
        # Shorten addresses for display
        top_traders['short_address'] = top_traders['address'].str[:8] + '...'
        
        # 1. Top traders by volume
        fig.add_trace(
            go.Bar(
                x=top_traders['short_address'],
                y=top_traders['total_notional'],
                marker=dict(color=top_traders['total_notional'], 
                          colorscale='Viridis',
                          showscale=True,
                          colorbar=dict(title="Volume", x=1.15)),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Maker ratio distribution
        fig.add_trace(
            go.Histogram(
                x=self.df_traders['maker_ratio'],
                nbinsx=30,
                marker=dict(color='lightblue'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Trade count vs volume scatter
        fig.add_trace(
            go.Scatter(
                x=self.df_traders['num_trades'],
                y=self.df_traders['total_notional'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.df_traders['maker_ratio'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Maker Ratio", x=1.15, y=0.25),
                    opacity=0.6
                ),
                text=self.df_traders['address'].str[:10],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. P&L box plot (top traders)
        pnl_data = []
        trader_labels = []
        
        for _, trader in top_traders.head(10).iterrows():
            trader_trades = self.df_trades[self.df_trades['address'] == trader['address']]
            pnl_data.append(trader_trades['closedPnl'].values)
            trader_labels.append(trader['short_address'])
        
        for i, (pnl, label) in enumerate(zip(pnl_data, trader_labels)):
            fig.add_trace(
                go.Box(
                    y=pnl,
                    name=label,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update axes
        fig.update_xaxes(title_text="Trader", row=1, col=1, tickangle=45)
        fig.update_xaxes(title_text="Maker Ratio", row=1, col=2)
        fig.update_xaxes(title_text="Number of Trades", row=2, col=1, type="log")
        fig.update_xaxes(title_text="Trader", row=2, col=2, tickangle=45)
        
        fig.update_yaxes(title_text="Total Volume (USDC)", row=1, col=1, type="log")
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Total Volume (USDC)", row=2, col=1, type="log")
        fig.update_yaxes(title_text="P&L (USDC)", row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text="Top Trader Analytics",
            title_x=0.5
        )
        
        return fig
    
    def create_trader_detail(self, address: str) -> go.Figure:
        """
        Create detailed view for specific trader
        
        Args:
            address: Trader address
            
        Returns:
            Plotly figure
        """
        trader_trades = self.df_trades[self.df_trades['address'] == address].copy()
        
        if trader_trades.empty:
            return self._empty_figure("No trades for this address")
        
        trader_trades = trader_trades.sort_values('timestamp')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cumulative P&L Over Time',
                'Trade Size Distribution',
                'Coins Traded',
                'Maker vs Taker'
            ),
            specs=[[{"secondary_y": False}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "pie"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Cumulative P&L
        trader_trades['cumulative_pnl'] = trader_trades['closedPnl'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=trader_trades['timestamp'],
                y=trader_trades['cumulative_pnl'],
                mode='lines',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Trade size distribution
        fig.add_trace(
            go.Histogram(
                x=trader_trades['notional'],
                nbinsx=30,
                marker=dict(color='green'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Coins traded
        coin_counts = trader_trades['coin'].value_counts().head(10)
        
        fig.add_trace(
            go.Bar(
                x=coin_counts.index,
                y=coin_counts.values,
                marker=dict(color='orange'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Maker vs Taker pie
        maker_count = trader_trades['is_maker'].sum()
        taker_count = trader_trades['is_taker'].sum()
        
        fig.add_trace(
            go.Pie(
                labels=['Maker', 'Taker'],
                values=[maker_count, taker_count],
                hole=0.4,
                marker=dict(colors=['lightblue', 'lightcoral'])
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Trade Size (USDC)", row=1, col=2)
        fig.update_xaxes(title_text="Coin", row=2, col=1, tickangle=45)
        
        fig.update_yaxes(title_text="Cumulative P&L (USDC)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Trade Count", row=2, col=1)
        
        fig.update_layout(
            height=800,
            title_text=f"Trader Detail: {address[:16]}...",
            title_x=0.5
        )
        
        return fig
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

"""
Order Flow Imbalance Visualization Class
Add this to visualizations.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats


class OrderFlowImbalance:
    """Visualizations for Order Flow Imbalance (OFI) analysis"""
    
    def __init__(self, df_ofi: pd.DataFrame, df_trades: pd.DataFrame = None):
        """
        Initialize with OFI data
        
        Args:
            df_ofi: DataFrame from calculate_order_flow_imbalance()
            df_trades: Optional raw trades DataFrame for additional analysis
        """
        self.df_ofi = df_ofi
        self.df_trades = df_trades
    
    def create_ofi_timeseries(self, coin: str = None) -> go.Figure:
        """
        Create OFI time series with price overlay
        
        Args:
            coin: Specific coin to visualize (if None, uses all data)
            
        Returns:
            Plotly figure
        """
        df = self.df_ofi.copy()
        if coin:
            df = df[df['coin'] == coin]
        
        # Create subplots: OFI, Price, Volume
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Order Flow Imbalance (OFI)',
                'Price',
                'Trading Volume'
            ),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 1. OFI with color gradient
        colors = ['red' if x < 0 else 'green' for x in df['OFI_volume']]
        
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['OFI_volume'],
                name='OFI Volume',
                marker_color=colors,
                opacity=0.7,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # 2. Price with candlestick
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open_price'],
                high=df['high_price'],
                low=df['low_price'],
                close=df['close_price'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=2, col=1
        )
        
        # 3. Volume bars
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['total_volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.5
            ),
            row=3, col=1
        )
        
        # Update layout
        title_text = f"<b>Order Flow Imbalance Analysis - {coin if coin else 'All Assets'}</b>"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            height=900,
            template='plotly_dark',
            hovermode='x unified',
            xaxis3_title="Time",
            yaxis_title="OFI (Net Buy Volume)",
            yaxis2_title="Price",
            yaxis3_title="Volume",
            showlegend=True
        )
        
        return fig
    
    def create_ofi_price_correlation(self, coin: str = None, lag_periods: int = 5) -> go.Figure:
        """
        Scatter plot showing OFI vs subsequent price changes
        
        Args:
            coin: Specific coin to analyze
            lag_periods: Number of periods to look ahead for price change
            
        Returns:
            Plotly figure
        """
        df = self.df_ofi.copy()
        if coin:
            df = df[df['coin'] == coin]
        
        # Calculate forward price changes
        if 'coin' in df.columns:
            df['future_price_change'] = df.groupby('coin')['close_price'].shift(-lag_periods)
            df['future_pct_change'] = ((df['future_price_change'] - df['close_price']) / df['close_price']) * 100
        else:
            df['future_price_change'] = df['close_price'].shift(-lag_periods)
            df['future_pct_change'] = ((df['future_price_change'] - df['close_price']) / df['close_price']) * 100
        
        # Remove NaN
        df = df.dropna(subset=['OFI_volume', 'future_pct_change'])
        
        # Calculate correlation
        if len(df) > 0:
            correlation = df['OFI_volume'].corr(df['future_pct_change'])
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df['OFI_volume'], df['future_pct_change']
            )
        else:
            correlation = 0
            slope, intercept, r_value = 0, 0, 0
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['OFI_volume'],
                y=df['future_pct_change'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df['total_volume'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Volume"),
                    opacity=0.6
                ),
                text=[f"Time: {t}<br>Volume: {v:.2f}" 
                      for t, v in zip(df['timestamp'], df['total_volume'])],
                hovertemplate='<b>OFI:</b> %{x:.2f}<br>' +
                              '<b>Future Δ%:</b> %{y:.2f}<br>' +
                              '%{text}<extra></extra>',
                name='Data Points'
            )
        )
        
        # Add regression line
        if len(df) > 0:
            x_range = np.array([df['OFI_volume'].min(), df['OFI_volume'].max()])
            y_pred = slope * x_range + intercept
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name=f'Regression (R²={r_value**2:.3f})'
                )
            )
        
        # Update layout
        title_text = f"<b>OFI vs Future Price Change ({lag_periods} periods ahead) - {coin if coin else 'All'}</b>"
        annotation_text = f"Correlation: {correlation:.3f} | R²: {r_value**2:.3f}"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Order Flow Imbalance (OFI)",
            yaxis_title=f"Price Change % ({lag_periods} periods ahead)",
            height=600,
            template='plotly_dark',
            annotations=[
                dict(
                    text=annotation_text,
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(size=14, color="yellow"),
                    xanchor='center'
                )
            ]
        )
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def create_cumulative_ofi(self, coin: str = None) -> go.Figure:
        """
        Cumulative OFI showing net buying/selling pressure over time
        
        Args:
            coin: Specific coin to analyze
            
        Returns:
            Plotly figure
        """
        df = self.df_ofi.copy()
        if coin:
            df = df[df['coin'] == coin]
        
        # Create dual-axis chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                'Cumulative Order Flow Imbalance',
                'Price'
            )
        )
        
        # Cumulative OFI
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_OFI_volume'],
                mode='lines',
                name='Cumulative OFI',
                line=dict(color='cyan', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,255,255,0.1)'
            ),
            row=1, col=1
        )
        
        # Price
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['close_price'],
                mode='lines',
                name='Price',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        title_text = f"<b>Cumulative OFI vs Price - {coin if coin else 'All'}</b>"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            height=700,
            template='plotly_dark',
            hovermode='x unified',
            yaxis_title="Cumulative Net Volume",
            yaxis2_title="Price"
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        return fig
    
    def create_ofi_heatmap(self, top_n: int = 10) -> go.Figure:
        """
        Heatmap of OFI across multiple assets over time
        
        Args:
            top_n: Number of top assets to include
            
        Returns:
            Plotly figure
        """
        if 'coin' not in self.df_ofi.columns:
            return go.Figure().add_annotation(
                text="Multi-asset heatmap requires data with multiple coins",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        df = self.df_ofi.copy()
        
        # Get top coins by total volume
        top_coins = df.groupby('coin')['total_volume'].sum().nlargest(top_n).index
        df = df[df['coin'].isin(top_coins)]
        
        # Pivot for heatmap
        heatmap_data = df.pivot_table(
            index='coin',
            columns='timestamp',
            values='OFI_intensity',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title="OFI Intensity"),
            hovertemplate='<b>%{y}</b><br>' +
                          'Time: %{x}<br>' +
                          'OFI Intensity: %{z:.3f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>Order Flow Imbalance Heatmap - Top {top_n} Assets</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Time",
            yaxis_title="Asset",
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def create_ofi_distribution(self, coin: str = None) -> go.Figure:
        """
        Distribution of OFI values with statistics
        
        Args:
            coin: Specific coin to analyze
            
        Returns:
            Plotly figure
        """
        df = self.df_ofi.copy()
        if coin:
            df = df[df['coin'] == coin]
        
        ofi_values = df['OFI_volume'].dropna()
        
        # Calculate statistics
        mean_ofi = ofi_values.mean()
        median_ofi = ofi_values.median()
        std_ofi = ofi_values.std()
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=ofi_values,
                nbinsx=50,
                name='OFI Distribution',
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate='OFI Range: %{x}<br>Count: %{y}<extra></extra>'
            )
        )
        
        # Add mean line
        fig.add_vline(
            x=mean_ofi,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_ofi:.2f}",
            annotation_position="top"
        )
        
        # Add median line
        fig.add_vline(
            x=median_ofi,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_ofi:.2f}",
            annotation_position="bottom"
        )
        
        # Update layout
        title_text = f"<b>OFI Distribution - {coin if coin else 'All Assets'}</b>"
        stats_text = f"μ={mean_ofi:.2f} | σ={std_ofi:.2f} | Skew={(ofi_values.skew()):.2f}"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Order Flow Imbalance",
            yaxis_title="Frequency",
            height=500,
            template='plotly_dark',
            annotations=[
                dict(
                    text=stats_text,
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(size=12, color="yellow"),
                    xanchor='center'
                )
            ]
        )
        
        return fig

"""
Multi-Asset Correlation Matrix Visualization Class
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff


class CorrelationMatrix:
    """Visualizations for multi-asset correlation analysis"""
    
    def __init__(self, corr_matrix: pd.DataFrame, returns: pd.DataFrame = None, prices: pd.DataFrame = None):
        """
        Initialize with correlation data
        
        Args:
            corr_matrix: Correlation matrix (coins × coins)
            returns: Returns DataFrame (time × coins)
            prices: Price DataFrame (time × coins)
        """
        self.corr_matrix = corr_matrix
        self.returns = returns
        self.prices = prices
    
    def create_correlation_heatmap(self, title_suffix: str = "") -> go.Figure:
        """
        Interactive correlation heatmap
        
        Returns:
            Plotly figure
        """
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=self.corr_matrix.values,
            x=self.corr_matrix.columns,
            y=self.corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
            ),
            hovertemplate='<b>%{x} vs %{y}</b><br>' +
                          'Correlation: %{z:.3f}<extra></extra>',
            text=np.round(self.corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            showscale=True
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>Multi-Asset Correlation Matrix{title_suffix}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Asset",
            yaxis_title="Asset",
            height=700,
            width=800,
            template='plotly_dark',
            xaxis=dict(tickangle=45),
            yaxis=dict(autorange='reversed')  # Top to bottom
        )
        
        return fig
    
    def create_correlation_network(self, threshold: float = 0.5) -> go.Figure:
        """
        Network graph showing correlation clusters
        
        Args:
            threshold: Minimum correlation to show edge
            
        Returns:
            Plotly figure
        """
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        coins = list(self.corr_matrix.columns)
        for coin in coins:
            G.add_node(coin)
        
        # Add edges for correlations above threshold
        for i, coin1 in enumerate(coins):
            for j, coin2 in enumerate(coins):
                if i < j:  # Avoid duplicates
                    corr = self.corr_matrix.loc[coin1, coin2]
                    if abs(corr) >= threshold:
                        G.add_edge(coin1, coin2, weight=abs(corr), sign=np.sign(corr))
        
        # Layout using spring layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            sign = edge[2]['sign']
            
            # Color by correlation sign
            color = 'rgba(0,255,0,0.5)' if sign > 0 else 'rgba(255,0,0,0.5)'
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*3, color=color),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            # Size by degree (number of connections)
            node_size.append(G.degree(node) * 5 + 10)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{text}</b><br>Connections: %{marker.size}<extra></extra>',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=dict(
                text=f"<b>Correlation Network (|r| ≥ {threshold})</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            showlegend=False,
            hovermode='closest',
            height=700,
            template='plotly_dark',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="Green = Positive Correlation | Red = Negative Correlation",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    showarrow=False,
                    font=dict(size=12, color="gray")
                )
            ]
        )
        
        return fig
    
    def create_lead_lag_chart(self, lead_lag_df: pd.DataFrame, coin1: str, coin2: str) -> go.Figure:
        """
        Lead-lag correlation analysis
        
        Args:
            lead_lag_df: DataFrame from calculate_lead_lag_correlation()
            coin1: First coin name
            coin2: Second coin name
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add correlation line
        fig.add_trace(go.Scatter(
            x=lead_lag_df['lag'],
            y=lead_lag_df['correlation'],
            mode='lines+markers',
            name='Correlation',
            line=dict(color='cyan', width=2),
            marker=dict(size=8),
            hovertemplate='<b>Lag:</b> %{x}<br>' +
                          '<b>Correlation:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Find max correlation
        max_idx = lead_lag_df['correlation'].abs().idxmax()
        max_lag = lead_lag_df.loc[max_idx, 'lag']
        max_corr = lead_lag_df.loc[max_idx, 'correlation']
        
        # Add marker for max
        fig.add_trace(go.Scatter(
            x=[max_lag],
            y=[max_corr],
            mode='markers',
            name='Peak Correlation',
            marker=dict(size=15, color='red', symbol='star'),
            hovertemplate=f'<b>Peak at lag {max_lag}</b><br>Correlation: {max_corr:.3f}<extra></extra>'
        ))
        
        # Determine leader
        if max_lag < 0:
            leader_text = f"{coin2} leads {coin1} by {abs(max_lag)} periods"
        elif max_lag > 0:
            leader_text = f"{coin1} leads {coin2} by {max_lag} periods"
        else:
            leader_text = f"{coin1} and {coin2} move simultaneously"
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>Lead-Lag Analysis: {coin1} vs {coin2}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Lag (periods)",
            yaxis_title="Correlation",
            height=600,
            template='plotly_dark',
            annotations=[
                dict(
                    text=leader_text,
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(size=14, color="yellow"),
                    xanchor='center'
                ),
                dict(
                    text="Negative lag = coin2 leads | Positive lag = coin1 leads",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15,
                    showarrow=False,
                    font=dict(size=11, color="gray")
                )
            ]
        )
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def create_rolling_correlation(self, coin1: str, coin2: str, window: int = 20) -> go.Figure:
        """
        Rolling correlation over time
        
        Args:
            coin1: First coin
            coin2: Second coin
            window: Rolling window size
            
        Returns:
            Plotly figure
        """
        if self.returns is None:
            return go.Figure().add_annotation(
                text="Returns data required for rolling correlation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Calculate rolling correlation
        rolling_corr = self.returns[coin1].rolling(window=window).corr(self.returns[coin2])
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f'Rolling Correlation ({window}-period window)',
                'Price Movements'
            ),
            row_heights=[0.6, 0.4]
        )
        
        # Rolling correlation
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode='lines',
                name='Rolling Correlation',
                line=dict(color='cyan', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,255,255,0.1)'
            ),
            row=1, col=1
        )
        
        # Add horizontal lines for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.5, row=1, col=1)
        fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)
        
        # Prices (normalized)
        if self.prices is not None:
            price1_norm = self.prices[coin1] / self.prices[coin1].iloc[0] * 100
            price2_norm = self.prices[coin2] / self.prices[coin2].iloc[0] * 100
            
            fig.add_trace(
                go.Scatter(
                    x=price1_norm.index,
                    y=price1_norm.values,
                    mode='lines',
                    name=coin1,
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=price2_norm.index,
                    y=price2_norm.values,
                    mode='lines',
                    name=coin2,
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>Rolling Correlation: {coin1} vs {coin2}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            height=800,
            template='plotly_dark',
            hovermode='x unified',
            yaxis_title="Correlation",
            yaxis2_title="Normalized Price (Base=100)",
            showlegend=True
        )
        
        return fig
    
    def create_diversification_metrics(self) -> go.Figure:
        """
        Diversification and clustering analysis
        
        Returns:
            Plotly figure with metrics
        """
        # Calculate metrics
        avg_corr = self.corr_matrix.values[np.triu_indices_from(self.corr_matrix.values, k=1)].mean()
        max_corr_pair = self.corr_matrix.stack().sort_values(ascending=False).iloc[1]  # Skip 1.0 diagonal
        min_corr_pair = self.corr_matrix.stack().sort_values().iloc[0]
        
        # Identify most/least correlated pairs
        corr_stack = self.corr_matrix.stack()
        corr_stack = corr_stack[corr_stack < 0.9999]  # Remove self-correlations
        
        most_corr_idx = corr_stack.idxmax()
        least_corr_idx = corr_stack.idxmin()
        
        # Create bar chart of average correlations per coin
        avg_corr_per_coin = self.corr_matrix.mean().sort_values(ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=avg_corr_per_coin.index,
            y=avg_corr_per_coin.values,
            marker_color='lightblue',
            hovertemplate='<b>%{x}</b><br>Avg Correlation: %{y:.3f}<extra></extra>'
        ))
        
        # Add average line
        fig.add_hline(
            y=avg_corr,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Market Avg: {avg_corr:.3f}",
            annotation_position="right"
        )
        
        # Update layout
        title_text = f"<b>Diversification Analysis</b><br>" + \
                     f"<sub>Avg Market Correlation: {avg_corr:.3f} | " + \
                     f"Most Correlated: {most_corr_idx[0]}-{most_corr_idx[1]} ({max_corr_pair:.2f}) | " + \
                     f"Least: {least_corr_idx[0]}-{least_corr_idx[1]} ({min_corr_pair:.2f})</sub>"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=18, family='Arial Black')
            ),
            xaxis_title="Asset",
            yaxis_title="Average Correlation with Other Assets",
            height=600,
            template='plotly_dark',
            xaxis=dict(tickangle=45)
        )
        
        return fig

"""
Market Impact (Kyle's Lambda) Visualization Class
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats


class MarketImpactAnalysis:
    """Visualizations for market impact and Kyle's Lambda analysis"""
    
    def __init__(self, df_impact: pd.DataFrame, df_lambda: pd.DataFrame = None):
        """
        Initialize with market impact data
        
        Args:
            df_impact: DataFrame from calculate_market_impact()
            df_lambda: DataFrame from calculate_lambda_by_asset()
        """
        self.df_impact = df_impact
        self.df_lambda = df_lambda
    
    def create_impact_scatter(self, coin: str = None) -> go.Figure:
        """
        Scatter plot: Trade Size vs Price Impact
        
        Args:
            coin: Specific coin to visualize
            
        Returns:
            Plotly figure
        """
        df = self.df_impact.copy()
        if coin:
            df = df[df['coin'] == coin]
        
        # Calculate regression
        if len(df) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df['notional'], 
                df['abs_price_change_pct']
            )
            lambda_estimate = slope * 1_000_000  # Per $1M
        else:
            slope, intercept, r_value = 0, 0, 0
            lambda_estimate = 0
        
        # Create scatter
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['notional'],
                y=df['abs_price_change_pct'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df['time_delta'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time Delta (s)"),
                    opacity=0.6
                ),
                text=[f"Coin: {c}<br>Size: ${n:,.0f}<br>Impact: {i:.3f}%" 
                      for c, n, i in zip(df['coin'], df['notional'], df['abs_price_change_pct'])],
                hovertemplate='%{text}<extra></extra>',
                name='Trades'
            )
        )
        
        # Add regression line
        if len(df) > 10:
            x_range = np.array([df['notional'].min(), df['notional'].max()])
            y_pred = slope * x_range + intercept
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name=f'λ={lambda_estimate:.2f}% per $1M (R²={r_value**2:.3f})'
                )
            )
        
        # Update layout
        title = f"<b>Market Impact: Trade Size vs Price Movement - {coin if coin else 'All Assets'}</b>"
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Trade Size (USD)",
            yaxis_title="Absolute Price Change (%)",
            height=600,
            template='plotly_dark',
            annotations=[
                dict(
                    text=f"Kyle's Lambda: {lambda_estimate:.2f}% per $1M | R²: {r_value**2:.3f}",
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(size=14, color="yellow"),
                    xanchor='center'
                )
            ]
        )
        
        return fig
    
    def create_lambda_comparison(self) -> go.Figure:
        """
        Compare Kyle's Lambda across different assets
        
        Returns:
            Plotly figure
        """
        if self.df_lambda is None:
            return go.Figure().add_annotation(
                text="Lambda by asset data required",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        df = self.df_lambda.sort_values('avg_impact_per_1m')
        
        # Create bar chart
        fig = go.Figure()
        
        # Color by impact level (green = low impact, red = high impact)
        colors = ['green' if x < df['avg_impact_per_1m'].median() else 'orange' 
                  for x in df['avg_impact_per_1m']]
        
        fig.add_trace(
            go.Bar(
                x=df['coin'],
                y=df['avg_impact_per_1m'],
                marker_color=colors,
                text=df['avg_impact_per_1m'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                              'Impact: %{y:.2f}% per $1M<br>' +
                              '<extra></extra>'
            )
        )
        
        # Add median line
        median_impact = df['avg_impact_per_1m'].median()
        fig.add_hline(
            y=median_impact,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {median_impact:.2f}%",
            annotation_position="right"
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="<b>Market Impact Comparison (Kyle's Lambda by Asset)</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Asset",
            yaxis_title="Price Impact per $1M Traded (%)",
            height=600,
            template='plotly_dark',
            xaxis=dict(tickangle=45),
            showlegend=False
        )
        
        return fig
    
    def create_impact_by_size_bucket(self, coin: str = None) -> go.Figure:
        """
        Impact analysis by trade size buckets
        
        Args:
            coin: Specific coin to analyze
            
        Returns:
            Plotly figure
        """
        df = self.df_impact.copy()
        if coin:
            df = df[df['coin'] == coin]
        
        # Group by size bucket
        bucket_stats = df.groupby('size_bucket').agg({
            'impact_per_1m_usd': ['mean', 'median', 'std'],
            'notional': ['count', 'sum']
        }).reset_index()
        
        # Flatten columns
        bucket_stats.columns = ['size_bucket', 'mean_impact', 'median_impact', 'std_impact',
                               'num_trades', 'total_volume']
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=bucket_stats['size_bucket'],
                y=bucket_stats['mean_impact'],
                name='Mean Impact',
                marker_color='lightblue',
                error_y=dict(type='data', array=bucket_stats['std_impact'], visible=True),
                text=bucket_stats['mean_impact'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside'
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=bucket_stats['size_bucket'],
                y=bucket_stats['median_impact'],
                name='Median Impact',
                marker_color='orange',
                text=bucket_stats['median_impact'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside'
            )
        )
        
        # Update layout
        title = f"<b>Market Impact by Trade Size - {coin if coin else 'All Assets'}</b>"
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Trade Size Bucket",
            yaxis_title="Price Impact per $1M (%)",
            height=600,
            template='plotly_dark',
            barmode='group',
            showlegend=True
        )
        
        return fig
    
    def create_temporal_impact(self, coin: str = None, window: int = 50) -> go.Figure:
        """
        Rolling average market impact over time
        
        Args:
            coin: Specific coin to analyze
            window: Rolling window size
            
        Returns:
            Plotly figure
        """
        df = self.df_impact.copy()
        if coin:
            df = df[df['coin'] == coin]
        
        df = df.sort_values('timestamp')
        
        # Calculate rolling metrics
        df['rolling_impact'] = df['impact_per_1m_usd'].rolling(window=window, min_periods=10).mean()
        df['rolling_volume'] = df['notional'].rolling(window=window).sum()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f'Rolling Market Impact ({window}-trade window)',
                'Rolling Trade Volume'
            ),
            row_heights=[0.6, 0.4]
        )
        
        # Rolling impact
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rolling_impact'],
                mode='lines',
                name='Rolling Impact',
                line=dict(color='cyan', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,255,255,0.1)'
            ),
            row=1, col=1
        )
        
        # Add individual points
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['impact_per_1m_usd'],
                mode='markers',
                name='Individual Trades',
                marker=dict(size=3, color='gray', opacity=0.3),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Rolling volume
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['rolling_volume'],
                name='Rolling Volume',
                marker_color='lightgreen',
                opacity=0.6
            ),
            row=2, col=1
        )
        
        # Update layout
        title = f"<b>Temporal Market Impact Analysis - {coin if coin else 'All Assets'}</b>"
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            height=800,
            template='plotly_dark',
            hovermode='x unified',
            yaxis_title="Impact per $1M (%)",
            yaxis2_title="Volume (USD)",
            showlegend=True
        )
        
        return fig
    
    def create_wallet_impact_efficiency(self, df_wallet_impact: pd.DataFrame) -> go.Figure:
        """
        Wallet execution efficiency (lower impact = better)
        
        Args:
            df_wallet_impact: DataFrame from calculate_impact_by_wallet()
            
        Returns:
            Plotly figure
        """
        df = df_wallet_impact.copy().head(30)
        
        # Create scatter: Volume vs Impact
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['total_volume'],
                y=df['avg_impact'],
                mode='markers',
                marker=dict(
                    size=df['avg_trade_size'] / 100,  # Scale for visibility
                    color=df['taker_ratio'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Taker Ratio"),
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                text=[f"Wallet: {a[:16]}...<br>Volume: ${v:,.0f}<br>Impact: {i:.3f}%<br>Avg Size: ${s:,.0f}" 
                      for a, v, i, s in zip(df['address'], df['total_volume'], 
                                           df['avg_impact'], df['avg_trade_size'])],
                hovertemplate='%{text}<extra></extra>',
                name='Wallets'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="<b>Wallet Execution Efficiency</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Total Trading Volume (USD)",
            yaxis_title="Average Market Impact per $1M (%)",
            xaxis_type='log',
            height=600,
            template='plotly_dark',
            annotations=[
                dict(
                    text="Lower impact = Better execution | Size = Avg trade size | Color = Taker ratio",
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(size=12, color="gray"),
                    xanchor='center'
                )
            ]
        )
        
        return fig
    
    def create_liquidity_quality_matrix(self) -> go.Figure:
        """
        Matrix showing volume, impact, and efficiency for top assets
        
        Returns:
            Plotly figure
        """
        if self.df_lambda is None:
            return go.Figure().add_annotation(
                text="Lambda by asset data required",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        df = self.df_lambda.head(15)
        
        # Create bubble chart: Volume vs Impact
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['total_volume_usd'],
                y=df['avg_impact_per_1m'],
                mode='markers+text',
                marker=dict(
                    size=df['num_trades'] / 10,  # Scale by number of trades
                    color=df['r_squared'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="R²"),
                    opacity=0.7,
                    line=dict(color='white', width=2)
                ),
                text=df['coin'],
                textposition='top center',
                textfont=dict(size=12, color='white'),
                hovertemplate='<b>%{text}</b><br>' +
                              'Volume: $%{x:,.0f}<br>' +
                              'Impact: %{y:.2f}% per $1M<br>' +
                              '<extra></extra>'
            )
        )
        
        # Add quadrant lines
        median_volume = df['total_volume_usd'].median()
        median_impact = df['avg_impact_per_1m'].median()
        
        fig.add_vline(x=median_volume, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=median_impact, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(
            text="High Volume<br>Low Impact<br>(BEST)",
            x=df['total_volume_usd'].max() * 0.8,
            y=df['avg_impact_per_1m'].min() * 1.2,
            showarrow=False,
            font=dict(size=10, color="green")
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="<b>Liquidity Quality Matrix</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Total Trading Volume (USD)",
            yaxis_title="Average Market Impact per $1M (%)",
            xaxis_type='log',
            height=700,
            template='plotly_dark',
            showlegend=False
        )
        
        return fig

