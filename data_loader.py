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
    
    def calculate_order_flow_imbalance(self, coin: Optional[str] = None, 
                                       time_window: str = '5S') -> pd.DataFrame:
        """
        Calculate Order Flow Imbalance (OFI) - a predictive metric for price movements
        
        OFI measures net buying/selling pressure by aggregating signed trade sizes.
        Positive OFI = net buying pressure, Negative OFI = net selling pressure
        
        Args:
            coin: Specific coin to analyze (None = all coins)
            time_window: Time aggregation window ('1S', '5S', '30S', '1T', '5T')
            
        Returns:
            DataFrame with OFI metrics per time window
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        print(f"Calculating Order Flow Imbalance (window={time_window})...")
        
        # Filter by coin if specified
        df = self.df_trades.copy()
        if coin:
            df = df[df['coin'] == coin].copy()
        
        # Create signed volume (buy = +, sell = -)
        df['signed_volume'] = np.where(df['is_buy'], df['sz'], -df['sz'])
        df['signed_notional'] = np.where(df['is_buy'], df['notional'], -df['notional'])
        
        # Also track aggressive (taker) flow specifically
        df['aggressive_signed_volume'] = np.where(
            df['is_taker'], 
            df['signed_volume'], 
            0
        )
        
        # Group by time window
        df_grouped = df.groupby([
            pd.Grouper(key='timestamp', freq=time_window),
            'coin' if not coin else pd.Grouper(key='coin')
        ]).agg({
            'signed_volume': 'sum',  # OFI (volume-based)
            'signed_notional': 'sum',  # OFI (dollar-based)
            'aggressive_signed_volume': 'sum',  # Aggressive OFI
            'px': ['mean', 'first', 'last', 'min', 'max'],  # Price stats
            'sz': ['sum', 'count'],  # Total volume and trade count
            'is_buy': 'sum',  # Number of buys
            'is_sell': 'sum',  # Number of sells
            'is_taker': 'sum',  # Number of aggressive trades
            'notional': 'sum',  # Total notional
        }).reset_index()
        
        # Flatten column names
        df_grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in df_grouped.columns.values]
        
        # Rename for clarity
        df_grouped = df_grouped.rename(columns={
            'signed_volume_sum': 'OFI_volume',
            'signed_notional_sum': 'OFI_notional',
            'aggressive_signed_volume_sum': 'OFI_aggressive',
            'px_mean': 'avg_price',
            'px_first': 'open_price',
            'px_last': 'close_price',
            'px_min': 'low_price',
            'px_max': 'high_price',
            'sz_sum': 'total_volume',
            'sz_count': 'num_trades',
            'is_buy_sum': 'num_buys',
            'is_sell_sum': 'num_sells',
            'is_taker_sum': 'num_taker_trades',
            'notional_sum': 'total_notional'
        })
        
        # Calculate price change
        if not coin:
            df_grouped['price_change'] = df_grouped.groupby('coin')['close_price'].diff()
            df_grouped['price_change_pct'] = df_grouped.groupby('coin')['close_price'].pct_change() * 100
            
            # Calculate cumulative OFI per coin
            df_grouped['cumulative_OFI_volume'] = df_grouped.groupby('coin')['OFI_volume'].cumsum()
            df_grouped['cumulative_OFI_notional'] = df_grouped.groupby('coin')['OFI_notional'].cumsum()
        else:
            df_grouped['price_change'] = df_grouped['close_price'].diff()
            df_grouped['price_change_pct'] = df_grouped['close_price'].pct_change() * 100
            
            # Calculate cumulative OFI
            df_grouped['cumulative_OFI_volume'] = df_grouped['OFI_volume'].cumsum()
            df_grouped['cumulative_OFI_notional'] = df_grouped['OFI_notional'].cumsum()
        
        # Calculate OFI intensity (normalized by volume)
        df_grouped['OFI_intensity'] = np.where(
            df_grouped['total_volume'] > 0,
            df_grouped['OFI_volume'] / df_grouped['total_volume'],
            0
        )
        
        # Buy/sell ratio
        df_grouped['buy_sell_ratio'] = np.where(
            df_grouped['num_sells'] > 0,
            df_grouped['num_buys'] / df_grouped['num_sells'],
            df_grouped['num_buys']
        )
        
        print(f"Calculated OFI for {len(df_grouped)} time windows")
        
        return df_grouped
    
    def calculate_correlation_matrix(self, top_n: int = 20, time_window: str = '1T') -> tuple:
        """
        Calculate correlation matrix for price movements across assets
        
        Args:
            top_n: Number of top coins by volume to include
            time_window: Time aggregation window for returns
            
        Returns:
            Tuple of (correlation_matrix, returns_df, price_df)
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        print(f"Calculating correlation matrix (top {top_n} coins, window={time_window})...")
        
        # Get top coins by volume
        top_coins = self.df_trades.groupby('coin')['notional'].sum().nlargest(top_n).index.tolist()
        
        df = self.df_trades[self.df_trades['coin'].isin(top_coins)].copy()
        
        # Aggregate prices by time window
        price_data = df.groupby([
            pd.Grouper(key='timestamp', freq=time_window),
            'coin'
        ]).agg({
            'px': 'last',  # Close price
            'notional': 'sum'  # Volume
        }).reset_index()
        
        # Pivot to wide format (time × coins)
        price_pivot = price_data.pivot_table(
            index='timestamp',
            columns='coin',
            values='px',
            aggfunc='last'
        )
        
        # Forward fill missing values (if a coin has no trades in a window)
        price_pivot = price_pivot.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate returns
        returns = price_pivot.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        print(f"Correlation matrix calculated for {len(top_coins)} coins")
        print(f"Time range: {price_pivot.index.min()} to {price_pivot.index.max()}")
        print(f"Data points per coin: {len(price_pivot)}")
        
        return corr_matrix, returns, price_pivot
    
    def calculate_lead_lag_correlation(self, coin1: str, coin2: str, 
                                       max_lag: int = 10, time_window: str = '30S') -> pd.DataFrame:
        """
        Calculate lead-lag correlation between two coins
        
        Args:
            coin1: First coin (potential leader)
            coin2: Second coin (potential follower)
            max_lag: Maximum lag periods to test
            time_window: Time aggregation window
            
        Returns:
            DataFrame with lag periods and correlations
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        print(f"Calculating lead-lag correlation: {coin1} vs {coin2}...")
        
        # Get price data for both coins
        df1 = self.df_trades[self.df_trades['coin'] == coin1].copy()
        df2 = self.df_trades[self.df_trades['coin'] == coin2].copy()
        
        # Aggregate by time
        prices1 = df1.groupby(pd.Grouper(key='timestamp', freq=time_window))['px'].last()
        prices2 = df2.groupby(pd.Grouper(key='timestamp', freq=time_window))['px'].last()
        
        # Align timestamps
        df_combined = pd.DataFrame({
            'price1': prices1,
            'price2': prices2
        }).fillna(method='ffill').dropna()
        
        # Calculate returns
        df_combined['return1'] = df_combined['price1'].pct_change()
        df_combined['return2'] = df_combined['price2'].pct_change()
        
        # Calculate correlation at different lags
        lag_results = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # coin2 leads coin1
                corr = df_combined['return1'].corr(df_combined['return2'].shift(-lag))
                leader = coin2
                follower = coin1
            elif lag > 0:
                # coin1 leads coin2
                corr = df_combined['return1'].shift(lag).corr(df_combined['return2'])
                leader = coin1
                follower = coin2
            else:
                # Contemporaneous
                corr = df_combined['return1'].corr(df_combined['return2'])
                leader = 'Simultaneous'
                follower = 'Simultaneous'
            
            lag_results.append({
                'lag': lag,
                'correlation': corr,
                'leader': leader if lag != 0 else 'Simultaneous',
                'follower': follower if lag != 0 else 'Simultaneous',
                'lag_seconds': lag * pd.Timedelta(time_window).total_seconds()
            })
        
        return pd.DataFrame(lag_results)
    
    def calculate_market_impact(self, coin: str = None, min_trade_size: float = 100) -> pd.DataFrame:
        """
        Calculate market impact metrics (Kyle's Lambda)
        
        Market impact = how much a trade moves the price
        Kyle's Lambda = dP / dQ (price change per unit volume)
        
        Args:
            coin: Specific coin to analyze (None = all coins)
            min_trade_size: Minimum trade size to include (notional USD)
            
        Returns:
            DataFrame with impact metrics
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        print(f"Calculating market impact (Kyle's Lambda)...")
        
        df = self.df_trades.copy()
        if coin:
            df = df[df['coin'] == coin].copy()
        
        # Filter by minimum trade size
        df = df[df['notional'] >= min_trade_size].copy()
        
        # Sort by timestamp and coin
        df = df.sort_values(['coin', 'timestamp']).reset_index(drop=True)
        
        # Calculate price changes
        df['prev_price'] = df.groupby('coin')['px'].shift(1)
        df['price_change'] = df['px'] - df['prev_price']
        df['price_change_pct'] = (df['price_change'] / df['prev_price']) * 100
        df['abs_price_change_pct'] = df['price_change_pct'].abs()
        
        # Calculate time between trades (in seconds)
        df['prev_timestamp'] = df.groupby('coin')['timestamp'].shift(1)
        df['time_delta'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()
        
        # Sign the price change by trade direction
        # If it's a buy (taker buy), positive price change is expected
        df['signed_price_change'] = np.where(
            df['is_buy'],
            df['price_change'],
            -df['price_change']
        )
        df['signed_price_change_pct'] = np.where(
            df['is_buy'],
            df['price_change_pct'],
            -df['price_change_pct']
        )
        
        # Calculate impact per unit (Kyle's Lambda proxy)
        # Impact = price_change / trade_size
        df['impact_per_unit'] = df['abs_price_change_pct'] / df['sz']
        df['impact_per_1k_usd'] = (df['abs_price_change_pct'] / df['notional']) * 1000
        df['impact_per_1m_usd'] = (df['abs_price_change_pct'] / df['notional']) * 1_000_000
        
        # Market impact score (considering direction)
        df['directional_impact'] = df['signed_price_change_pct'] / df['notional'] * 1_000_000
        
        # Classify trade sizes
        df['size_bucket'] = pd.cut(
            df['notional'],
            bins=[0, 1000, 5000, 10000, 50000, 100000, float('inf')],
            labels=['<$1K', '$1K-5K', '$5K-10K', '$10K-50K', '$50K-100K', '>$100K']
        )
        
        # Remove NaN (first trade per coin)
        df = df.dropna(subset=['price_change', 'impact_per_unit'])
        
        # Remove outliers (>99th percentile)
        df = df[df['abs_price_change_pct'] < df['abs_price_change_pct'].quantile(0.99)]
        
        print(f"Calculated impact for {len(df)} trades across {df['coin'].nunique()} coins")
        
        return df
    
    def calculate_lambda_by_asset(self, top_n: int = 20, min_trades: int = 50) -> pd.DataFrame:
        """
        Estimate Kyle's Lambda for each asset
        
        Args:
            top_n: Number of top coins to analyze
            min_trades: Minimum trades required per coin
            
        Returns:
            DataFrame with lambda estimates per coin
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        print(f"Estimating Kyle's Lambda by asset...")
        
        # Get impact data
        df_impact = self.calculate_market_impact()
        
        # Calculate lambda per coin
        lambda_results = []
        
        for coin, group in df_impact.groupby('coin'):
            if len(group) < min_trades:
                continue
            
            # Simple average impact
            avg_impact = group['impact_per_1m_usd'].mean()
            median_impact = group['impact_per_1m_usd'].median()
            
            # Regression-based lambda (more robust)
            from scipy import stats
            if len(group) > 10:
                # Regress abs(price_change_pct) on notional
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    group['notional'], 
                    group['abs_price_change_pct']
                )
                lambda_regression = slope * 1_000_000  # Per $1M
            else:
                lambda_regression = avg_impact
                r_value = 0
            
            lambda_results.append({
                'coin': coin,
                'avg_impact_per_1m': avg_impact,
                'median_impact_per_1m': median_impact,
                'lambda_regression': lambda_regression,
                'r_squared': r_value**2,
                'num_trades': len(group),
                'total_volume_usd': group['notional'].sum(),
                'avg_trade_size': group['notional'].mean(),
                'std_impact': group['impact_per_1m_usd'].std()
            })
        
        df_lambda = pd.DataFrame(lambda_results)
        df_lambda = df_lambda.sort_values('total_volume_usd', ascending=False).head(top_n)
        
        print(f"Calculated lambda for {len(df_lambda)} coins")
        
        return df_lambda
    
    def calculate_impact_by_wallet(self, coin: str = None, top_n: int = 50) -> pd.DataFrame:
        """
        Calculate average market impact by wallet/trader
        
        Args:
            coin: Specific coin to analyze
            top_n: Number of top traders to return
            
        Returns:
            DataFrame with impact metrics per wallet
        """
        if self.df_trades is None:
            raise ValueError("Must load data first with load_misc_events()")
        
        print(f"Calculating market impact by wallet...")
        
        # Get impact data
        df_impact = self.calculate_market_impact(coin=coin)
        
        # Group by wallet
        wallet_impact = df_impact.groupby('address').agg({
            'impact_per_1m_usd': ['mean', 'median'],
            'notional': ['sum', 'mean', 'count'],
            'abs_price_change_pct': 'mean',
            'is_taker': 'sum'
        }).reset_index()
        
        # Flatten columns
        wallet_impact.columns = ['address', 'avg_impact', 'median_impact', 
                                'total_volume', 'avg_trade_size', 'num_trades',
                                'avg_price_move', 'num_taker_trades']
        
        # Calculate efficiency score (lower impact = better)
        wallet_impact['impact_score'] = wallet_impact['avg_impact']
        wallet_impact['taker_ratio'] = wallet_impact['num_taker_trades'] / wallet_impact['num_trades']
        
        # Filter and sort
        wallet_impact = wallet_impact[wallet_impact['num_trades'] >= 10]
        wallet_impact = wallet_impact.sort_values('total_volume', ascending=False).head(top_n)
        
        print(f"Calculated impact for {len(wallet_impact)} wallets")
        
        return wallet_impact
    
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

