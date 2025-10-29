"""
Order Book Visualizations for Hyperliquid Market Microstructure Analysis
Phase 1: Core Order Book Reconstruction Visualizations
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


class OrderBookDepthChart:
    """Animated depth chart showing bid/ask evolution"""
    
    def __init__(self, df_snapshots: pd.DataFrame):
        """
        Initialize with order book snapshots
        
        Args:
            df_snapshots: DataFrame with columns [timestamp, best_bid, best_ask, mid_price, etc.]
        """
        self.df = df_snapshots
    
    def create_depth_evolution_chart(self) -> go.Figure:
        """
        Create depth evolution chart showing best bid/ask and spread over time
        
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                '<b>Best Bid/Ask & Mid-Price</b>',
                '<b>Spread (%)</b>'
            ),
            row_heights=[0.65, 0.35]
        )
        
        # Top: Best bid/ask/mid
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['best_bid'],
                name='Best Bid',
                line=dict(color='#00CC66', width=2),
                fill='tonexty',
                fillcolor='rgba(0, 204, 102, 0.2)',
                hovertemplate='<b>Best Bid</b><br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['best_ask'],
                name='Best Ask',
                line=dict(color='#FF4444', width=2),
                fill='tonexty',
                fillcolor='rgba(255, 68, 68, 0.2)',
                hovertemplate='<b>Best Ask</b><br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['mid_price'],
                name='Mid-Price',
                line=dict(color='#FFA500', width=2, dash='dash'),
                hovertemplate='<b>Mid-Price</b><br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Bottom: Spread
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['spread_pct'],
                name='Spread %',
                line=dict(color='#9B59B6', width=2),
                fill='tozeroy',
                fillcolor='rgba(155, 89, 182, 0.3)',
                hovertemplate='<b>Spread</b><br>Time: %{x}<br>%{y:.4f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Spread (%)", row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        fig.update_layout(
            template='plotly_dark',
            height=700,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_depth_distribution_chart(self) -> go.Figure:
        """
        Create chart showing distribution of depth on both sides
        
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                '<b>Order Book Depth Over Time</b>',
                '<b>Depth Imbalance</b>'
            ),
            row_heights=[0.6, 0.4]
        )
        
        # Top: Bid vs Ask depth
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['bid_depth_1pct'],
                name='Bid Depth (1%)',
                line=dict(color='#00CC66', width=2),
                stackgroup='depth',
                hovertemplate='<b>Bid Depth</b><br>%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['ask_depth_1pct'],
                name='Ask Depth (1%)',
                line=dict(color='#FF4444', width=2),
                stackgroup='depth',
                hovertemplate='<b>Ask Depth</b><br>%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Bottom: Imbalance
        colors = ['#00CC66' if x > 0 else '#FF4444' for x in self.df['imbalance']]
        
        fig.add_trace(
            go.Bar(
                x=self.df['timestamp'],
                y=self.df['imbalance'],
                name='Imbalance',
                marker=dict(color=colors),
                hovertemplate='<b>Imbalance</b><br>%{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_xaxes(title_text="Time", row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Depth", row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Imbalance", row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        fig.update_layout(
            template='plotly_dark',
            height=700,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig


class LiquidityHeatmap:
    """Liquidity heatmap showing price × time"""
    
    def __init__(self, df_heatmap: pd.DataFrame):
        """
        Initialize with heatmap data
        
        Args:
            df_heatmap: DataFrame with columns [time, price, side, total_size]
        """
        self.df = df_heatmap
    
    def create_heatmap(self) -> go.Figure:
        """
        Create 2D heatmap of liquidity (price × time)
        
        Returns:
            Plotly Figure
        """
        # Split bid and ask
        df_bid = self.df[self.df['side'] == 'B'].copy()
        df_ask = self.df[self.df['side'] == 'A'].copy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('<b>ASK Liquidity</b>', '<b>BID Liquidity</b>'),
            row_heights=[0.5, 0.5]
        )
        
        # Pivot for heatmap
        if len(df_ask) > 0:
            pivot_ask = df_ask.pivot_table(
                index='price', 
                columns='time', 
                values='total_size', 
                fill_value=0
            )
            
            fig.add_trace(
                go.Heatmap(
                    z=np.log1p(pivot_ask.values),  # Log scale for better visualization
                    x=pivot_ask.columns,
                    y=pivot_ask.index,
                    colorscale='Reds',
                    name='Ask Liquidity',
                    hovertemplate='Time: %{x}<br>Price: $%{y:,.2f}<br>Size: %{z:.2f}<extra></extra>',
                    colorbar=dict(title='Log(Size)', len=0.45, y=0.75)
                ),
                row=1, col=1
            )
        
        if len(df_bid) > 0:
            pivot_bid = df_bid.pivot_table(
                index='price', 
                columns='time', 
                values='total_size', 
                fill_value=0
            )
            
            fig.add_trace(
                go.Heatmap(
                    z=np.log1p(pivot_bid.values),
                    x=pivot_bid.columns,
                    y=pivot_bid.index,
                    colorscale='Greens',
                    name='Bid Liquidity',
                    hovertemplate='Time: %{x}<br>Price: $%{y:,.2f}<br>Size: %{z:.2f}<extra></extra>',
                    colorbar=dict(title='Log(Size)', len=0.45, y=0.25)
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=800,
            title_text="<b>Liquidity Heatmap: Price × Time</b>",
            title_x=0.5,
            title_font=dict(size=20, family='Arial Black')
        )
        
        return fig


class ImbalanceAnalysis:
    """Order book imbalance analysis"""
    
    def __init__(self, df_snapshots: pd.DataFrame):
        """
        Initialize with order book snapshots
        
        Args:
            df_snapshots: DataFrame with imbalance metrics
        """
        self.df = df_snapshots
    
    def create_imbalance_timeseries(self) -> go.Figure:
        """
        Create time series chart of order book imbalance
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Imbalance ratio
        colors = ['#00CC66' if x > 0 else '#FF4444' for x in self.df['imbalance']]
        
        fig.add_trace(
            go.Bar(
                x=self.df['timestamp'],
                y=self.df['imbalance'],
                name='Imbalance Ratio',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.3)', width=0.5)
                ),
                hovertemplate='<b>Imbalance</b><br>Time: %{x}<br>Ratio: %{y:.3f}<extra></extra>'
            )
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Balanced")
        fig.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Strong Bid")
        fig.add_hline(y=-0.5, line_dash="dot", line_color="red", annotation_text="Strong Ask")
        
        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text="<b>Order Book Imbalance Over Time</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Time",
            yaxis_title="Imbalance Ratio",
            height=500,
            hovermode='x unified',
            yaxis=dict(range=[-1, 1])
        )
        
        return fig
    
    def create_imbalance_distribution(self) -> go.Figure:
        """
        Create histogram of imbalance distribution
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=self.df['imbalance'],
                nbinsx=50,
                name='Imbalance Distribution',
                marker=dict(
                    color=self.df['imbalance'],
                    colorscale='RdYlGn',
                    line=dict(color='white', width=0.5)
                ),
                hovertemplate='Imbalance: %{x:.3f}<br>Count: %{y}<extra></extra>'
            )
        )
        
        # Add mean line
        mean_imbalance = self.df['imbalance'].mean()
        fig.add_vline(
            x=mean_imbalance, 
            line_dash="dash", 
            line_color="yellow",
            annotation_text=f"Mean: {mean_imbalance:.3f}"
        )
        
        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text="<b>Imbalance Distribution</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Imbalance Ratio",
            yaxis_title="Frequency",
            height=500
        )
        
        return fig


class SpreadAnalysisOrderBook:
    """Spread analysis for order book"""
    
    def __init__(self, df_snapshots: pd.DataFrame):
        """
        Initialize with order book snapshots
        
        Args:
            df_snapshots: DataFrame with spread metrics
        """
        self.df = df_snapshots
    
    def create_spread_analysis_chart(self) -> go.Figure:
        """
        Create comprehensive spread analysis chart
        
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                '<b>Absolute Spread ($)</b>',
                '<b>Percentage Spread (%)</b>',
                '<b>Mid-Price</b>'
            ),
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # Absolute spread
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['spread'],
                name='Spread ($)',
                line=dict(color='#9B59B6', width=2),
                fill='tozeroy',
                fillcolor='rgba(155, 89, 182, 0.3)',
                hovertemplate='<b>Spread</b><br>$%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Percentage spread
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['spread_pct'],
                name='Spread %',
                line=dict(color='#E74C3C', width=2),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.3)',
                hovertemplate='<b>Spread</b><br>%{y:.4f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Mid-price
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['mid_price'],
                name='Mid-Price',
                line=dict(color='#3498DB', width=2),
                hovertemplate='<b>Mid-Price</b><br>$%{y:,.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add mean lines
        fig.add_hline(y=self.df['spread'].mean(), line_dash="dash", line_color="yellow", row=1, col=1)
        fig.add_hline(y=self.df['spread_pct'].mean(), line_dash="dash", line_color="yellow", row=2, col=1)
        
        fig.update_xaxes(title_text="Time", row=3, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Spread ($)", row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Spread (%)", row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Price ($)", row=3, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        fig.update_layout(
            template='plotly_dark',
            height=900,
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
    
    def create_spread_distribution(self) -> go.Figure:
        """
        Create spread distribution histogram
        
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('<b>Absolute Spread</b>', '<b>Percentage Spread</b>')
        )
        
        fig.add_trace(
            go.Histogram(
                x=self.df['spread'],
                nbinsx=50,
                name='Spread ($)',
                marker=dict(color='#9B59B6', line=dict(color='white', width=0.5)),
                hovertemplate='Spread: $%{x:.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=self.df['spread_pct'],
                nbinsx=50,
                name='Spread %',
                marker=dict(color='#E74C3C', line=dict(color='white', width=0.5)),
                hovertemplate='Spread: %{x:.4f}%<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Spread ($)", row=1, col=1)
        fig.update_xaxes(title_text="Spread (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            showlegend=False
        )
        
        return fig


class OrderLifecycleAnalysis:
    """Order lifecycle and behavior analysis"""
    
    def __init__(self, df_lifecycle: pd.DataFrame):
        """
        Initialize with order lifecycle data
        
        Args:
            df_lifecycle: DataFrame with order lifetime metrics
        """
        self.df = df_lifecycle
    
    def create_lifetime_distribution(self) -> go.Figure:
        """
        Create histogram of order lifetimes
        
        Returns:
            Plotly Figure
        """
        # Filter outliers for better visualization
        df_clean = self.df[self.df['lifetime_seconds'] < self.df['lifetime_seconds'].quantile(0.95)]
        
        fig = go.Figure()
        
        # Separate by outcome
        for outcome, color, name in [
            ('was_filled', '#00CC66', 'Filled'),
            ('was_canceled', '#FF4444', 'Canceled'),
            ('was_rejected', '#FFA500', 'Rejected')
        ]:
            df_subset = df_clean[df_clean[outcome]]
            if len(df_subset) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=df_subset['lifetime_seconds'],
                        name=name,
                        marker=dict(color=color, opacity=0.7, line=dict(color='white', width=0.5)),
                        nbinsx=50,
                        hovertemplate=f'<b>{name}</b><br>Lifetime: %{{x:.1f}}s<br>Count: %{{y}}<extra></extra>'
                    )
                )
        
        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text="<b>Order Lifetime Distribution</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Lifetime (seconds)",
            yaxis_title="Frequency",
            height=500,
            barmode='overlay'
        )
        
        return fig
    
    def create_outcome_breakdown(self) -> go.Figure:
        """
        Create pie chart of order outcomes
        
        Returns:
            Plotly Figure
        """
        outcomes = {
            'Filled': self.df['was_filled'].sum(),
            'Canceled': self.df['was_canceled'].sum(),
            'Rejected': self.df['was_rejected'].sum()
        }
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(outcomes.keys()),
                values=list(outcomes.values()),
                marker=dict(colors=['#00CC66', '#FF4444', '#FFA500']),
                textinfo='label+percent',
                textfont=dict(size=14),
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text="<b>Order Outcome Distribution</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            height=500
        )
        
        return fig
    
    def create_fill_rate_by_side(self) -> go.Figure:
        """
        Create bar chart of fill rates by bid/ask
        
        Returns:
            Plotly Figure
        """
        fill_rates = self.df.groupby('side')['was_filled'].agg(['sum', 'count'])
        fill_rates['rate'] = fill_rates['sum'] / fill_rates['count']
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=['Bids', 'Asks'],
                y=[
                    fill_rates.loc['B', 'rate'] if 'B' in fill_rates.index else 0,
                    fill_rates.loc['A', 'rate'] if 'A' in fill_rates.index else 0
                ],
                marker=dict(color=['#00CC66', '#FF4444']),
                text=[f"{x:.1%}" for x in [
                    fill_rates.loc['B', 'rate'] if 'B' in fill_rates.index else 0,
                    fill_rates.loc['A', 'rate'] if 'A' in fill_rates.index else 0
                ]],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Fill Rate: %{y:.1%}<extra></extra>'
            )
        )
        
        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text="<b>Fill Rate by Side</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title="Side",
            yaxis_title="Fill Rate",
            height=500,
            yaxis=dict(tickformat='.0%')
        )
        
        return fig

