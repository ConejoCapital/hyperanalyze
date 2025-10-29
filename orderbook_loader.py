"""
Hyperliquid Order Book Data Loader and State Reconstruction
Handles 50M+ order events for Level 2 order book analysis
"""

import json
import gzip
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class OrderBookLoader:
    """Load and reconstruct order book state from Hyperliquid order events"""
    
    def __init__(self, orderbook_path: str):
        """
        Initialize order book loader
        
        Args:
            orderbook_path: Path to Hyperliquid_orderbooks.json.gz file
        """
        self.orderbook_path = orderbook_path
        self.df_events = None
        self.order_book_states = {}  # Cached snapshots
        
    def load_events(self, max_events: Optional[int] = None, coins_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load order book events from compressed JSON
        
        Args:
            max_events: Limit number of events to load (None = all)
            coins_filter: Only load events for specific coins (None = all)
            
        Returns:
            DataFrame with parsed order events
        """
        print(f"Loading order book events from {self.orderbook_path}...")
        
        events = []
        event_count = 0
        
        with gzip.open(self.orderbook_path, 'rt') as f:
            # Skip opening bracket
            next(f)
            
            for line in f:
                try:
                    # Remove trailing comma and whitespace
                    line = line.strip().rstrip(',')
                    if not line or line == ']':
                        continue
                    
                    event_data = json.loads(line)
                    
                    # Parse nested VALUE field
                    value = json.loads(event_data['VALUE'])
                    order = value['order']
                    
                    # Filter by coin if specified
                    if coins_filter and order['coin'] not in coins_filter:
                        continue
                    
                    # Extract key fields
                    event = {
                        'block_number': int(event_data['BLOCK_NUMBER']),
                        'datetime': pd.to_datetime(event_data['DATETIME']),
                        'timestamp': order['timestamp'],
                        'coin': order['coin'],
                        'side': order['side'],  # B = bid, A = ask
                        'price': float(order['limitPx']),
                        'size': float(order['sz']),
                        'orig_size': float(order['origSz']),
                        'oid': int(order['oid']),
                        'order_type': order['orderType'],
                        'tif': order['tif'],
                        'status': value['status'],
                        'user': value['user'],
                        'hash': value.get('hash'),
                        'cloid': order.get('cloid'),
                        'is_trigger': order.get('isTrigger', False),
                        'reduce_only': order.get('reduceOnly', False)
                    }
                    
                    events.append(event)
                    event_count += 1
                    
                    if event_count % 100000 == 0:
                        print(f"  Loaded {event_count:,} events...")
                    
                    if max_events and event_count >= max_events:
                        break
                        
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    # Skip malformed lines
                    continue
        
        print(f"Loaded {len(events):,} order book events")
        
        self.df_events = pd.DataFrame(events)
        
        # Add computed fields
        self.df_events['is_bid'] = self.df_events['side'] == 'B'
        self.df_events['is_ask'] = self.df_events['side'] == 'A'
        self.df_events['is_open'] = self.df_events['status'] == 'open'
        self.df_events['is_filled'] = self.df_events['status'].str.contains('filled', case=False, na=False)
        self.df_events['is_canceled'] = self.df_events['status'].str.contains('cancel', case=False, na=False)
        self.df_events['is_rejected'] = self.df_events['status'].str.contains('rejected', case=False, na=False)
        
        # Partial fill detection
        self.df_events['is_partial_fill'] = (
            self.df_events['is_filled'] & 
            (self.df_events['size'] < self.df_events['orig_size'])
        )
        
        print(f"Date range: {self.df_events['datetime'].min()} to {self.df_events['datetime'].max()}")
        print(f"Unique coins: {self.df_events['coin'].nunique()}")
        print(f"Unique orders: {self.df_events['oid'].nunique():,}")
        print(f"Unique wallets: {self.df_events['user'].nunique():,}")
        
        return self.df_events
    
    def reconstruct_order_book(self, coin: str, sample_interval: str = '1S') -> pd.DataFrame:
        """
        Reconstruct order book state over time for a specific coin
        
        Args:
            coin: Coin to reconstruct (e.g., 'BTC', 'ETH')
            sample_interval: Time interval for snapshots (e.g., '1S', '5S', '10S')
            
        Returns:
            DataFrame with order book snapshots
        """
        if self.df_events is None:
            raise ValueError("Must load events first with load_events()")
        
        print(f"Reconstructing order book for {coin} (sample interval: {sample_interval})...")
        
        # Filter to specific coin
        df_coin = self.df_events[self.df_events['coin'] == coin].copy()
        df_coin = df_coin.sort_values('datetime').reset_index(drop=True)
        
        print(f"  Processing {len(df_coin):,} events for {coin}")
        
        # Track active orders
        active_orders = {
            'bids': {},  # {oid: {'price': x, 'size': y, 'user': z, 'time': t}}
            'asks': {}
        }
        
        snapshots = []
        
        for idx, event in df_coin.iterrows():
            side_key = 'bids' if event['is_bid'] else 'asks'
            oid = event['oid']
            
            # Update order book state
            if event['is_open']:
                # New order placed
                active_orders[side_key][oid] = {
                    'price': event['price'],
                    'size': event['size'],
                    'user': event['user'],
                    'time': event['datetime']
                }
            elif event['is_filled'] or event['is_canceled']:
                # Order filled or canceled - remove from book
                if oid in active_orders[side_key]:
                    del active_orders[side_key][oid]
            
            # Take snapshot at regular intervals
            if idx % 100 == 0 or idx == len(df_coin) - 1:
                snapshot = self._create_snapshot(event['datetime'], active_orders)
                if snapshot:
                    snapshots.append(snapshot)
        
        df_snapshots = pd.DataFrame(snapshots)
        
        # Resample to desired interval
        if len(df_snapshots) > 0:
            df_snapshots = df_snapshots.set_index('timestamp')
            df_snapshots = df_snapshots.resample(sample_interval).last().ffill()
            df_snapshots = df_snapshots.reset_index()
        
        print(f"  Created {len(df_snapshots)} snapshots")
        
        return df_snapshots
    
    def _create_snapshot(self, timestamp, active_orders):
        """Create a snapshot of current order book state"""
        bids = active_orders['bids']
        asks = active_orders['asks']
        
        if not bids or not asks:
            return None
        
        # Aggregate by price level
        bid_levels = defaultdict(float)
        ask_levels = defaultdict(float)
        
        for oid, order in bids.items():
            bid_levels[order['price']] += order['size']
        
        for oid, order in asks.items():
            ask_levels[order['price']] += order['size']
        
        # Get best bid/ask
        best_bid = max(bid_levels.keys()) if bid_levels else 0
        best_ask = min(ask_levels.keys()) if ask_levels else 0
        
        if best_bid == 0 or best_ask == 0:
            return None
        
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
        
        # Calculate depth at various levels
        bid_depth_1pct = sum(size for price, size in bid_levels.items() if price >= mid_price * 0.99)
        ask_depth_1pct = sum(size for price, size in ask_levels.items() if price <= mid_price * 1.01)
        
        bid_depth_total = sum(bid_levels.values())
        ask_depth_total = sum(ask_levels.values())
        
        imbalance = (bid_depth_1pct - ask_depth_1pct) / (bid_depth_1pct + ask_depth_1pct) if (bid_depth_1pct + ask_depth_1pct) > 0 else 0
        
        return {
            'timestamp': timestamp,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'spread': spread,
            'spread_pct': spread_pct,
            'bid_depth_1pct': bid_depth_1pct,
            'ask_depth_1pct': ask_depth_1pct,
            'bid_depth_total': bid_depth_total,
            'ask_depth_total': ask_depth_total,
            'imbalance': imbalance,
            'num_bid_levels': len(bid_levels),
            'num_ask_levels': len(ask_levels),
            'num_bid_orders': len(active_orders['bids']),
            'num_ask_orders': len(active_orders['asks'])
        }
    
    def get_liquidity_heatmap_data(self, coin: str, price_range_pct: float = 2.0, num_bins: int = 50) -> pd.DataFrame:
        """
        Get data for liquidity heatmap (price × time)
        
        Args:
            coin: Coin to analyze
            price_range_pct: Price range around mid (+/- %)
            num_bins: Number of price bins
            
        Returns:
            DataFrame with liquidity by price level and time
        """
        if self.df_events is None:
            raise ValueError("Must load events first with load_events()")
        
        print(f"Calculating liquidity heatmap for {coin}...")
        
        df_coin = self.df_events[self.df_events['coin'] == coin].copy()
        df_coin = df_coin.sort_values('datetime')
        
        # Get approximate mid-price (median of all order prices)
        mid_price = df_coin['price'].median()
        
        # Define price bins relative to mid
        price_min = mid_price * (1 - price_range_pct / 100)
        price_max = mid_price * (1 + price_range_pct / 100)
        price_bins = np.linspace(price_min, price_max, num_bins)
        
        df_coin['price_bin'] = pd.cut(df_coin['price'], bins=price_bins, labels=price_bins[:-1])
        df_coin['time_bin'] = df_coin['datetime'].dt.floor('1Min')
        
        # Aggregate liquidity by time and price bins
        df_open = df_coin[df_coin['is_open']]
        
        heatmap_data = df_open.groupby(['time_bin', 'price_bin', 'side']).agg({
            'size': 'sum',
            'oid': 'count'
        }).reset_index()
        
        heatmap_data.columns = ['time', 'price', 'side', 'total_size', 'num_orders']
        
        print(f"  Generated heatmap with {len(heatmap_data)} data points")
        
        return heatmap_data
    
    def get_order_lifecycle_stats(self, coin: str = None) -> pd.DataFrame:
        """
        Calculate order lifecycle statistics
        
        Args:
            coin: Specific coin (None = all coins)
            
        Returns:
            DataFrame with lifecycle metrics per order
        """
        if self.df_events is None:
            raise ValueError("Must load events first with load_events()")
        
        print(f"Calculating order lifecycle statistics...")
        
        df = self.df_events.copy()
        if coin:
            df = df[df['coin'] == coin]
        
        # Group by order ID to track lifecycle
        df_lifecycle = df.groupby('oid').agg({
            'datetime': ['first', 'last'],
            'status': lambda x: x.iloc[-1],  # Final status
            'coin': 'first',
            'side': 'first',
            'price': 'first',
            'size': ['first', 'last'],
            'user': 'first'
        })
        
        df_lifecycle.columns = ['_'.join(col).strip('_') for col in df_lifecycle.columns.values]
        df_lifecycle = df_lifecycle.reset_index()
        
        # Calculate metrics
        df_lifecycle['lifetime_seconds'] = (
            df_lifecycle['datetime_last'] - df_lifecycle['datetime_first']
        ).dt.total_seconds()
        
        df_lifecycle['was_filled'] = df_lifecycle['status'].str.contains('filled', case=False, na=False)
        df_lifecycle['was_canceled'] = df_lifecycle['status'].str.contains('cancel', case=False, na=False)
        df_lifecycle['was_rejected'] = df_lifecycle['status'].str.contains('reject', case=False, na=False)
        df_lifecycle['was_partial'] = df_lifecycle['size_last'] < df_lifecycle['size_first']
        df_lifecycle['fill_rate'] = df_lifecycle['size_last'] / df_lifecycle['size_first']
        
        print(f"  Analyzed {len(df_lifecycle):,} orders")
        print(f"  Fill rate: {df_lifecycle['was_filled'].mean():.1%}")
        print(f"  Cancel rate: {df_lifecycle['was_canceled'].mean():.1%}")
        print(f"  Reject rate: {df_lifecycle['was_rejected'].mean():.1%}")
        print(f"  Median lifetime: {df_lifecycle['lifetime_seconds'].median():.1f}s")
        
        return df_lifecycle
    
    def save_processed_data(self, filepath: str):
        """Save processed events to Parquet for faster loading"""
        if self.df_events is not None:
            self.df_events.to_parquet(filepath, compression='snappy', index=False)
            print(f"Saved {len(self.df_events):,} events to {filepath}")
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """Load previously processed events from Parquet"""
        print(f"Loading processed order book data from {filepath}...")
        self.df_events = pd.read_parquet(filepath)
        print(f"Loaded {len(self.df_events):,} events")
        print(f"Date range: {self.df_events['datetime'].min()} to {self.df_events['datetime'].max()}")
        return self.df_events

