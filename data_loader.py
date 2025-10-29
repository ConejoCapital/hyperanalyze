"""
Hyperliquid Data Loader and Preprocessing Pipeline
Handles loading and processing of misc_events and node_fills data
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HyperliquidDataLoader:
    """Load and preprocess Hyperliquid trading data"""
    
    def __init__(self, misc_events_path: str, node_fills_path: Optional[str] = None):
        """
        Initialize data loader
        
        Args:
            misc_events_path: Path to misc_events JSON file
            node_fills_path: Optional path to node_fills JSON file
        """
        self.misc_events_path = misc_events_path
        self.node_fills_path = node_fills_path
        self.df_trades = None
        self.df_fills = None
        
    def load_misc_events(self, max_lines: Optional[int] = None) -> pd.DataFrame:
        """
        Load misc_events data from JSON lines file
        
        Args:
            max_lines: Optional limit on number of lines to read
            
        Returns:
            DataFrame with all trade events
        """
        print(f"Loading misc_events from {self.misc_events_path}...")
        
        events = []
        line_count = 0
        
        with open(self.misc_events_path, 'r') as f:
            for line in f:
                if max_lines and line_count >= max_lines:
                    break
                    
                try:
                    data = json.loads(line)
                    block_time = data.get('block_time')
                    block_number = data.get('block_number')
                    local_time = data.get('local_time')
                    
                    # Extract events
                    for event in data.get('events', []):
                        if isinstance(event, list) and len(event) == 2:
                            # Standard trade event [address, trade_details]
                            address, trade = event
                            trade['address'] = address
                            trade['block_time'] = block_time
                            trade['block_number'] = block_number
                            trade['local_time'] = local_time
                            events.append(trade)
                        elif isinstance(event, dict):
                            # Other event types (LedgerUpdate, etc.)
                            # Skip for now or handle separately
                            pass
                            
                    line_count += 1
                    
                    if line_count % 5000 == 0:
                        print(f"  Processed {line_count} blocks, {len(events)} events so far...")
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line {line_count}: {e}")
                    continue
        
        print(f"Loaded {len(events)} trade events from {line_count} blocks")
        
        df = pd.DataFrame(events)
        self.df_trades = self._preprocess_trades(df)
        
        return self.df_trades
    
    def _preprocess_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess trade data with type conversions and computed fields
        
        Args:
            df: Raw trade DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            return df
            
        print("Preprocessing trade data...")
        
        # Convert numeric fields
        numeric_fields = ['px', 'sz', 'fee', 'closedPnl', 'startPosition', 
                         'time', 'oid', 'tid', 'block_number']
        
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        df['block_time'] = pd.to_datetime(df['block_time'], errors='coerce')
        df['local_time'] = pd.to_datetime(df['local_time'], errors='coerce')
        
        # Add computed fields
        df['is_maker'] = ~df['crossed'].fillna(False)
        df['is_taker'] = df['crossed'].fillna(False)
        df['is_buy'] = df['side'] == 'B'
        df['is_sell'] = df['side'] == 'A'
        
        # Notional value in USDC
        df['notional'] = df['px'] * df['sz']
        
        # Absolute fee
        df['abs_fee'] = df['fee'].abs()
        
        # Determine aggressor side (who crossed the spread)
        df['aggressor_side'] = df.apply(
            lambda x: 'buy' if x['crossed'] and x['side'] == 'B' 
            else ('sell' if x['crossed'] and x['side'] == 'A' else 'maker'),
            axis=1
        )
        
        # Position change
        df['position_change'] = df.apply(
            lambda x: x['sz'] if x['side'] == 'B' else -x['sz'],
            axis=1
        )
        
        # Clean up coin names (remove @ prefix for some tickers)
        df['coin_clean'] = df['coin'].astype(str)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Preprocessing complete. Shape: {df.shape}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Unique coins: {df['coin'].nunique()}")
        print(f"Unique traders: {df['address'].nunique()}")
        
        return df
    
    def get_orderbook_snapshots(self, coin: str, interval: str = '1S') -> pd.DataFrame:
        """
        Reconstruct order book snapshots for a specific coin
        
        Args:
            coin: Trading pair (e.g., 'ETH', 'BTC')
            interval: Time interval for snapshots (e.g., '1S', '5S', '1T')
            
        Returns:
            DataFrame with order book snapshots
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        # Filter for specific coin
        df_coin = self.df_trades[self.df_trades['coin'] == coin].copy()
        
        if df_coin.empty:
            print(f"No data found for coin: {coin}")
            return pd.DataFrame()
        
        # Set timestamp as index
        df_coin = df_coin.set_index('timestamp')
        
        # Resample to get snapshots
        snapshots = []
        
        for timestamp, group in df_coin.resample(interval):
            if len(group) > 0:
                # Get best bid/ask from this interval
                buys = group[group['is_buy']]
                sells = group[group['is_sell']]
                
                snapshot = {
                    'timestamp': timestamp,
                    'coin': coin,
                    'num_trades': len(group),
                    'volume': group['sz'].sum(),
                    'notional': group['notional'].sum(),
                    'best_bid': buys['px'].max() if len(buys) > 0 else np.nan,
                    'best_ask': sells['px'].min() if len(sells) > 0 else np.nan,
                    'bid_volume': buys['sz'].sum() if len(buys) > 0 else 0,
                    'ask_volume': sells['sz'].sum() if len(sells) > 0 else 0,
                }
                
                # Calculate mid, spread
                if not np.isnan(snapshot['best_bid']) and not np.isnan(snapshot['best_ask']):
                    snapshot['mid'] = (snapshot['best_bid'] + snapshot['best_ask']) / 2
                    snapshot['spread'] = snapshot['best_ask'] - snapshot['best_bid']
                    snapshot['spread_bps'] = (snapshot['spread'] / snapshot['mid']) * 10000
                else:
                    snapshot['mid'] = group['px'].mean()  # fallback
                    snapshot['spread'] = np.nan
                    snapshot['spread_bps'] = np.nan
                
                # Order book imbalance
                total_vol = snapshot['bid_volume'] + snapshot['ask_volume']
                if total_vol > 0:
                    snapshot['imbalance'] = (snapshot['bid_volume'] - snapshot['ask_volume']) / total_vol
                else:
                    snapshot['imbalance'] = 0
                
                snapshots.append(snapshot)
        
        return pd.DataFrame(snapshots)
    
    def get_trader_analytics(self, min_trades: int = 10) -> pd.DataFrame:
        """
        Calculate trader-level analytics
        
        Args:
            min_trades: Minimum number of trades for inclusion
            
        Returns:
            DataFrame with trader statistics
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        print(f"Calculating trader analytics (min {min_trades} trades)...")
        
        trader_stats = []
        
        for address, group in self.df_trades.groupby('address'):
            if len(group) < min_trades:
                continue
            
            stats = {
                'address': address,
                'num_trades': len(group),
                'total_volume': group['sz'].sum(),
                'total_notional': group['notional'].sum(),
                'num_coins': group['coin'].nunique(),
                'total_fees_paid': group['fee'].sum(),
                'abs_fees': group['abs_fee'].sum(),
                
                # Maker/Taker breakdown
                'num_maker_trades': group['is_maker'].sum(),
                'num_taker_trades': group['is_taker'].sum(),
                'maker_ratio': group['is_maker'].sum() / len(group),
                'taker_ratio': group['is_taker'].sum() / len(group),
                
                # Directional stats
                'num_buys': group['is_buy'].sum(),
                'num_sells': group['is_sell'].sum(),
                'buy_volume': group[group['is_buy']]['sz'].sum(),
                'sell_volume': group[group['is_sell']]['sz'].sum(),
                
                # P&L
                'total_pnl': group['closedPnl'].sum(),
                'avg_trade_pnl': group['closedPnl'].mean(),
                
                # Size stats
                'avg_trade_size': group['sz'].mean(),
                'median_trade_size': group['sz'].median(),
                'max_trade_size': group['sz'].max(),
                'avg_trade_notional': group['notional'].mean(),
                
                # Time stats
                'first_trade': group['timestamp'].min(),
                'last_trade': group['timestamp'].max(),
            }
            
            # Net position change per coin
            position_changes = group.groupby('coin')['position_change'].sum().to_dict()
            stats['position_changes'] = position_changes
            
            # Most traded coins
            top_coins = group.groupby('coin').size().sort_values(ascending=False).head(3)
            stats['top_coins'] = top_coins.to_dict()
            
            trader_stats.append(stats)
        
        df_traders = pd.DataFrame(trader_stats)
        
        # Sort by total notional
        df_traders = df_traders.sort_values('total_notional', ascending=False).reset_index(drop=True)
        
        print(f"Calculated stats for {len(df_traders)} traders")
        
        return df_traders
    
    def get_coin_summary(self) -> pd.DataFrame:
        """
        Get summary statistics by coin
        
        Returns:
            DataFrame with coin-level statistics
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        print("Calculating coin summary statistics...")
        
        coin_stats = []
        
        for coin, group in self.df_trades.groupby('coin'):
            stats = {
                'coin': coin,
                'num_trades': len(group),
                'num_traders': group['address'].nunique(),
                'total_volume': group['sz'].sum(),
                'total_notional': group['notional'].sum(),
                'avg_price': group['px'].mean(),
                'min_price': group['px'].min(),
                'max_price': group['px'].max(),
                'price_range_pct': ((group['px'].max() - group['px'].min()) / group['px'].mean()) * 100,
                
                # Maker/Taker
                'maker_trades': group['is_maker'].sum(),
                'taker_trades': group['is_taker'].sum(),
                'maker_ratio': group['is_maker'].sum() / len(group),
                
                # Fees
                'total_fees': group['fee'].sum(),
                'avg_fee_per_trade': group['fee'].mean(),
                
                # Timing
                'first_trade': group['timestamp'].min(),
                'last_trade': group['timestamp'].max(),
            }
            
            coin_stats.append(stats)
        
        df_coins = pd.DataFrame(coin_stats)
        df_coins = df_coins.sort_values('total_notional', ascending=False).reset_index(drop=True)
        
        return df_coins
    
    def get_market_impact_data(self, coin: str) -> pd.DataFrame:
        """
        Calculate market impact metrics for a specific coin
        
        Args:
            coin: Trading pair
            
        Returns:
            DataFrame with market impact analysis
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        df_coin = self.df_trades[self.df_trades['coin'] == coin].copy()
        df_coin = df_coin.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate mid price at each point
        df_coin['mid'] = df_coin['px']  # Simplified - would need order book for true mid
        
        # Calculate price change from previous trade
        df_coin['price_change'] = df_coin['px'].diff()
        df_coin['price_change_pct'] = df_coin['px'].pct_change() * 100
        
        # Impact per unit size (Kyle's lambda proxy)
        df_coin['impact_per_size'] = df_coin['price_change_pct'] / df_coin['sz']
        
        return df_coin
    
    def save_processed_data(self, output_path: str = 'processed_data.parquet'):
        """
        Save processed data to Parquet for faster loading
        
        Args:
            output_path: Path for output file
        """
        if self.df_trades is None:
            raise ValueError("No data to save. Load data first.")
        
        print(f"Saving processed data to {output_path}...")
        self.df_trades.to_parquet(output_path, compression='gzip', index=False)
        print(f"Saved {len(self.df_trades)} rows")
    
    def load_processed_data(self, input_path: str = 'processed_data.parquet'):
        """
        Load previously processed data from Parquet
        
        Args:
            input_path: Path to parquet file
        """
        print(f"Loading processed data from {input_path}...")
        self.df_trades = pd.read_parquet(input_path)
        print(f"Loaded {len(self.df_trades)} rows")
        print(f"Date range: {self.df_trades['timestamp'].min()} to {self.df_trades['timestamp'].max()}")
        return self.df_trades


def main():
    """Example usage"""
    # NOTE: Use node_fills for trade data, misc_events for other event types
    loader = HyperliquidDataLoader(
        misc_events_path='/Users/thebunnymac/Desktop/hyperorderbook/Hyperliquid Data Expanded/node_fills_20251027_1700-1800.json'
    )
    
    # Load data (limit to first 1000 blocks for testing)
    df = loader.load_misc_events(max_lines=1000)
    
    print("\n=== Data Summary ===")
    print(df.info())
    print("\n=== Sample Trades ===")
    print(df.head())
    
    # Get coin summary
    coin_summary = loader.get_coin_summary()
    print("\n=== Top 10 Coins by Volume ===")
    print(coin_summary.head(10)[['coin', 'num_trades', 'total_notional', 'num_traders']])
    
    # Get trader analytics
    trader_analytics = loader.get_trader_analytics(min_trades=5)
    print("\n=== Top 10 Traders by Notional ===")
    print(trader_analytics.head(10)[['address', 'num_trades', 'total_notional', 'maker_ratio']])
    
    # Save processed data
    loader.save_processed_data()


if __name__ == "__main__":
    main()

