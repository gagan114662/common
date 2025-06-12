"""
Dark Pool Liquidity Analysis Engine
Advanced institutional order flow detection and dark liquidity modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import scipy.stats as stats
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
import asyncio
import json

class InstitutionalOrderType(Enum):
    """Types of institutional orders"""
    ICEBERG = "iceberg"
    BLOCK = "block"
    TWAP = "twap"
    VWAP = "vwap"
    STEALTH = "stealth"
    MOMENTUM_IGNITION = "momentum_ignition"
    STOP_HUNT = "stop_hunt"

class DarkPoolType(Enum):
    """Types of dark pools"""
    BROKER_DEALER = "broker_dealer"
    ELECTRONIC_CROSSING = "electronic_crossing"
    EXCHANGE_SPONSORED = "exchange_sponsored"
    CONSORTIUM = "consortium"

@dataclass
class DarkPoolSignal:
    """Dark pool activity signal"""
    timestamp: datetime
    symbol: str
    signal_type: InstitutionalOrderType
    confidence: float
    estimated_size: float
    price_level: float
    impact_prediction: float
    time_horizon: int  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LiquidityProfile:
    """Market liquidity profile"""
    symbol: str
    timestamp: datetime
    visible_liquidity: float
    hidden_liquidity_estimate: float
    institutional_flow_ratio: float
    dark_pool_participation: float
    fragmentation_index: float
    price_impact_coefficient: float
    optimal_execution_size: float

@dataclass
class OrderFlowAnalysis:
    """Order flow analysis results"""
    buy_pressure: float
    sell_pressure: float
    net_flow: float
    flow_persistence: float
    size_weighted_flow: float
    institutional_signature: float
    retail_flow_estimate: float

class DarkPoolLiquidityAnalyzer:
    """
    Advanced dark pool and institutional order flow analyzer
    
    Features:
    - Iceberg order detection
    - Hidden liquidity estimation
    - Institutional footprint analysis
    - Dark pool participation modeling
    - Optimal execution timing
    - Market microstructure analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.lookback_periods = {
            'micro': 60,     # 1 minute
            'short': 300,    # 5 minutes
            'medium': 1800,  # 30 minutes
            'long': 7200     # 2 hours
        }
        
        # Detection thresholds
        self.iceberg_threshold = 3.0        # Standard deviations
        self.block_size_threshold = 10000   # Minimum block size
        self.institutional_confidence = 0.7
        
        # Data storage
        self.market_data: Dict[str, List[Dict]] = {}
        self.order_flow_history: Dict[str, List[OrderFlowAnalysis]] = {}
        self.dark_signals: List[DarkPoolSignal] = []
        
        # ML models (simplified)
        self.size_classifier = None
        self.flow_regressor = None
        
        # Market maker signatures
        self.mm_signatures = self._initialize_mm_signatures()
        
    def _initialize_mm_signatures(self) -> Dict[str, Dict[str, float]]:
        """Initialize market maker behavioral signatures"""
        return {
            'citadel': {
                'latency_ms': 0.1,
                'size_preference': 0.85,  # Prefers smaller sizes
                'time_preference': 'pre_market',
                'venue_preference': 'dark_pools'
            },
            'jane_street': {
                'latency_ms': 0.05,
                'size_preference': 0.75,
                'time_preference': 'market_open',
                'venue_preference': 'lit_markets'
            },
            'two_sigma': {
                'latency_ms': 0.2,
                'size_preference': 0.95,
                'time_preference': 'continuous',
                'venue_preference': 'crossing_networks'
            }
        }
    
    async def analyze_dark_pool_activity(
        self, 
        symbol: str, 
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        trade_data: Optional[pd.DataFrame] = None
    ) -> Tuple[List[DarkPoolSignal], LiquidityProfile]:
        """
        Analyze dark pool activity and institutional order flow
        
        Args:
            symbol: Trading symbol
            price_data: OHLCV price data
            volume_data: Detailed volume data
            trade_data: Optional tick-by-tick trade data
            
        Returns:
            Tuple of dark pool signals and liquidity profile
        """
        
        # Prepare data
        market_data = self._prepare_market_data(symbol, price_data, volume_data, trade_data)
        
        # Detect institutional order patterns
        signals = []
        
        # 1. Iceberg order detection
        iceberg_signals = await self._detect_iceberg_orders(market_data)
        signals.extend(iceberg_signals)
        
        # 2. Block trade detection
        block_signals = await self._detect_block_trades(market_data)
        signals.extend(block_signals)
        
        # 3. TWAP/VWAP algorithm detection
        algo_signals = await self._detect_algo_trading(market_data)
        signals.extend(algo_signals)
        
        # 4. Stealth trading detection
        stealth_signals = await self._detect_stealth_trading(market_data)
        signals.extend(stealth_signals)
        
        # 5. Market manipulation detection
        manipulation_signals = await self._detect_market_manipulation(market_data)
        signals.extend(manipulation_signals)
        
        # Create liquidity profile
        liquidity_profile = await self._create_liquidity_profile(market_data, signals)
        
        # Store signals
        self.dark_signals.extend(signals)
        
        self.logger.info(f"Detected {len(signals)} dark pool signals for {symbol}")
        
        return signals, liquidity_profile
    
    def _prepare_market_data(
        self, 
        symbol: str, 
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        trade_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Prepare and structure market data for analysis"""
        
        # Ensure datetime index
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index)
        
        # Calculate derived metrics
        price_data['returns'] = price_data['close'].pct_change()
        price_data['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
        price_data['volatility'] = price_data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Volume analysis
        price_data['volume_ma'] = price_data['volume'].rolling(window=20).mean()
        price_data['volume_ratio'] = price_data['volume'] / price_data['volume_ma']
        price_data['dollar_volume'] = price_data['close'] * price_data['volume']
        
        # Price impact metrics
        price_data['price_impact'] = np.abs(price_data['returns']) / np.log(price_data['volume'] + 1)
        price_data['bid_ask_spread'] = (price_data['high'] - price_data['low']) / price_data['close']
        
        # Microstructure indicators
        price_data['tick_direction'] = np.sign(price_data['close'].diff())
        price_data['price_acceleration'] = price_data['close'].diff().diff()
        
        return {
            'symbol': symbol,
            'price_data': price_data,
            'volume_data': volume_data,
            'trade_data': trade_data,
            'timestamp': datetime.now()
        }
    
    async def _detect_iceberg_orders(self, market_data: Dict[str, Any]) -> List[DarkPoolSignal]:
        """Detect iceberg orders through volume clustering analysis"""
        
        signals = []
        price_data = market_data['price_data']
        
        if len(price_data) < 50:
            return signals
        
        # Look for repeated volume patterns at similar price levels
        recent_data = price_data.tail(100)
        
        # Volume clustering analysis
        volumes = recent_data['volume'].values.reshape(-1, 1)
        prices = recent_data['close'].values.reshape(-1, 1)
        
        # Standardize features
        scaler = StandardScaler()
        features = scaler.fit_transform(np.hstack([volumes, prices]))
        
        # Cluster analysis
        clustering = DBSCAN(eps=0.5, min_samples=5)
        clusters = clustering.fit_predict(features)
        
        # Analyze clusters for iceberg patterns
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = clusters == cluster_id
            cluster_data = recent_data[cluster_mask]
            
            if len(cluster_data) < 3:
                continue
            
            # Check for iceberg characteristics
            volume_consistency = cluster_data['volume'].std() / cluster_data['volume'].mean()
            price_stability = cluster_data['close'].std() / cluster_data['close'].mean()
            time_persistence = len(cluster_data)
            
            # Iceberg detection criteria
            if (volume_consistency < 0.2 and    # Consistent volume sizes
                price_stability < 0.01 and      # Stable price level
                time_persistence >= 5):         # Persistent over time
                
                confidence = min(0.95, 1.0 - volume_consistency - price_stability)
                estimated_total_size = cluster_data['volume'].sum() * 3  # Estimate hidden size
                
                signal = DarkPoolSignal(
                    timestamp=cluster_data.index[-1],
                    symbol=market_data['symbol'],
                    signal_type=InstitutionalOrderType.ICEBERG,
                    confidence=confidence,
                    estimated_size=estimated_total_size,
                    price_level=cluster_data['close'].mean(),
                    impact_prediction=self._predict_price_impact(estimated_total_size, market_data),
                    time_horizon=300,  # 5 minutes
                    metadata={
                        'volume_consistency': volume_consistency,
                        'price_stability': price_stability,
                        'execution_count': len(cluster_data),
                        'cluster_id': cluster_id
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    async def _detect_block_trades(self, market_data: Dict[str, Any]) -> List[DarkPoolSignal]:
        """Detect large block trades and institutional activity"""
        
        signals = []
        price_data = market_data['price_data']
        
        # Calculate volume percentiles
        volume_95th = price_data['volume'].quantile(0.95)
        volume_99th = price_data['volume'].quantile(0.99)
        
        # Look for unusually large volumes
        large_volume_mask = price_data['volume'] > volume_95th
        large_volumes = price_data[large_volume_mask]
        
        for idx, row in large_volumes.iterrows():
            volume_z_score = (row['volume'] - price_data['volume'].mean()) / price_data['volume'].std()
            
            if volume_z_score > 3.0:  # 3+ standard deviations
                # Check for block trade characteristics
                price_impact = abs(row['returns']) if not pd.isna(row['returns']) else 0
                volume_ratio = row['volume'] / row['volume_ma'] if row['volume_ma'] > 0 else 1
                
                # Block trade criteria
                if (row['volume'] > self.block_size_threshold and
                    volume_ratio > 2.0 and
                    price_impact < 0.005):  # Low impact suggests hidden execution
                    
                    confidence = min(0.9, volume_z_score / 10)
                    
                    signal = DarkPoolSignal(
                        timestamp=idx,
                        symbol=market_data['symbol'],
                        signal_type=InstitutionalOrderType.BLOCK,
                        confidence=confidence,
                        estimated_size=row['volume'],
                        price_level=row['close'],
                        impact_prediction=price_impact,
                        time_horizon=60,  # 1 minute
                        metadata={
                            'volume_z_score': volume_z_score,
                            'volume_ratio': volume_ratio,
                            'dollar_volume': row['dollar_volume']
                        }
                    )
                    
                    signals.append(signal)
        
        return signals
    
    async def _detect_algo_trading(self, market_data: Dict[str, Any]) -> List[DarkPoolSignal]:
        """Detect TWAP/VWAP and other algorithmic trading patterns"""
        
        signals = []
        price_data = market_data['price_data']
        
        if len(price_data) < 20:
            return signals
        
        # Calculate VWAP
        price_data['vwap'] = (price_data['close'] * price_data['volume']).cumsum() / price_data['volume'].cumsum()
        
        # Look for systematic trading around VWAP
        price_data['vwap_deviation'] = (price_data['close'] - price_data['vwap']) / price_data['vwap']
        
        # Detect VWAP algorithm patterns
        recent_data = price_data.tail(30)
        
        # Check for mean-reverting behavior around VWAP
        vwap_deviations = recent_data['vwap_deviation'].dropna()
        
        if len(vwap_deviations) > 10:
            # Test for mean reversion
            deviations_lagged = vwap_deviations.shift(1).dropna()
            correlation = vwap_deviations[1:].corr(deviations_lagged[1:])
            
            if correlation < -0.3:  # Negative correlation suggests mean reversion
                # Look for systematic volume patterns
                volume_regularity = 1.0 - (recent_data['volume'].std() / recent_data['volume'].mean())
                
                if volume_regularity > 0.6:  # Regular volume suggests algo trading
                    confidence = min(0.8, volume_regularity + abs(correlation) * 0.5)
                    
                    signal = DarkPoolSignal(
                        timestamp=recent_data.index[-1],
                        symbol=market_data['symbol'],
                        signal_type=InstitutionalOrderType.VWAP,
                        confidence=confidence,
                        estimated_size=recent_data['volume'].sum(),
                        price_level=recent_data['vwap'].iloc[-1],
                        impact_prediction=recent_data['vwap_deviation'].abs().mean(),
                        time_horizon=900,  # 15 minutes
                        metadata={
                            'vwap_correlation': correlation,
                            'volume_regularity': volume_regularity,
                            'mean_deviation': vwap_deviations.mean()
                        }
                    )
                    
                    signals.append(signal)
        
        # Detect TWAP patterns (time-weighted average price)
        time_intervals = pd.cut(range(len(recent_data)), bins=5, labels=False)
        interval_volumes = []
        
        for interval in range(5):
            interval_mask = time_intervals == interval
            if interval_mask.sum() > 0:
                interval_volume = recent_data[interval_mask]['volume'].mean()
                interval_volumes.append(interval_volume)
        
        if len(interval_volumes) == 5:
            volume_variance = np.var(interval_volumes) / np.mean(interval_volumes)
            
            if volume_variance < 0.1:  # Very regular volume suggests TWAP
                confidence = 1.0 - volume_variance * 5
                
                signal = DarkPoolSignal(
                    timestamp=recent_data.index[-1],
                    symbol=market_data['symbol'],
                    signal_type=InstitutionalOrderType.TWAP,
                    confidence=confidence,
                    estimated_size=sum(interval_volumes) * 2,  # Estimate remaining size
                    price_level=recent_data['close'].iloc[-1],
                    impact_prediction=volume_variance,
                    time_horizon=1800,  # 30 minutes
                    metadata={
                        'volume_variance': volume_variance,
                        'interval_volumes': interval_volumes
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    async def _detect_stealth_trading(self, market_data: Dict[str, Any]) -> List[DarkPoolSignal]:
        """Detect stealth trading through hidden order flow analysis"""
        
        signals = []
        price_data = market_data['price_data']
        
        if len(price_data) < 50:
            return signals
        
        # Calculate order flow imbalance
        price_data['order_flow'] = np.where(
            price_data['tick_direction'] > 0,
            price_data['volume'],
            -price_data['volume']
        )
        
        # Smooth order flow to detect persistent patterns
        window_size = min(21, len(price_data) // 3)
        if window_size < 5:
            return signals
        
        smoothed_flow = savgol_filter(price_data['order_flow'], window_size, 3)
        price_data['smoothed_flow'] = smoothed_flow
        
        # Detect stealth patterns: persistent flow with minimal price impact
        recent_data = price_data.tail(30)
        
        # Calculate flow persistence
        flow_persistence = abs(recent_data['smoothed_flow'].mean()) / recent_data['smoothed_flow'].std()
        
        # Calculate average price impact per unit flow
        nonzero_returns = recent_data[recent_data['returns'] != 0]['returns']
        nonzero_flow = recent_data[recent_data['returns'] != 0]['order_flow']
        
        if len(nonzero_returns) > 5:
            impact_ratio = nonzero_returns.abs().mean() / nonzero_flow.abs().mean()
            
            # Stealth criteria: persistent flow with low impact
            if (flow_persistence > 2.0 and          # Persistent directional flow
                impact_ratio < 0.001 and            # Low price impact
                recent_data['volume'].sum() > recent_data['volume_ma'].mean() * 5):  # Significant volume
                
                confidence = min(0.85, flow_persistence / 5 + (0.001 - impact_ratio) * 1000)
                
                signal = DarkPoolSignal(
                    timestamp=recent_data.index[-1],
                    symbol=market_data['symbol'],
                    signal_type=InstitutionalOrderType.STEALTH,
                    confidence=confidence,
                    estimated_size=abs(recent_data['order_flow'].sum()) * 2,  # Estimate hidden portion
                    price_level=recent_data['close'].iloc[-1],
                    impact_prediction=impact_ratio,
                    time_horizon=3600,  # 1 hour
                    metadata={
                        'flow_persistence': flow_persistence,
                        'impact_ratio': impact_ratio,
                        'net_flow': recent_data['order_flow'].sum()
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    async def _detect_market_manipulation(self, market_data: Dict[str, Any]) -> List[DarkPoolSignal]:
        """Detect potential market manipulation patterns"""
        
        signals = []
        price_data = market_data['price_data']
        
        if len(price_data) < 30:
            return signals
        
        recent_data = price_data.tail(20)
        
        # Detect momentum ignition patterns
        # Look for small trades followed by large price movements
        for i in range(2, len(recent_data) - 2):
            window = recent_data.iloc[i-2:i+3]
            
            # Check for small volume followed by large price move
            initial_volume = window['volume'].iloc[:2].mean()
            subsequent_volume = window['volume'].iloc[2:].mean()
            
            price_move = abs(window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
            volume_ratio = subsequent_volume / (initial_volume + 1)
            
            # Momentum ignition criteria
            if (initial_volume < recent_data['volume_ma'].iloc[i] * 0.5 and  # Small initial volume
                price_move > 0.01 and                                        # Significant price move
                volume_ratio > 3.0):                                         # Volume surge after move
                
                confidence = min(0.7, price_move * 50 + volume_ratio / 10)
                
                signal = DarkPoolSignal(
                    timestamp=window.index[-1],
                    symbol=market_data['symbol'],
                    signal_type=InstitutionalOrderType.MOMENTUM_IGNITION,
                    confidence=confidence,
                    estimated_size=subsequent_volume,
                    price_level=window['close'].iloc[-1],
                    impact_prediction=price_move,
                    time_horizon=120,  # 2 minutes
                    metadata={
                        'initial_volume': initial_volume,
                        'subsequent_volume': subsequent_volume,
                        'price_move': price_move,
                        'volume_ratio': volume_ratio
                    }
                )
                
                signals.append(signal)
        
        # Detect stop hunting patterns
        # Look for price spikes that reverse quickly
        price_changes = recent_data['close'].pct_change()
        
        for i in range(1, len(price_changes) - 1):
            current_change = price_changes.iloc[i]
            next_change = price_changes.iloc[i + 1]
            
            # Stop hunt criteria: large move followed by immediate reversal
            if (abs(current_change) > 0.005 and          # Significant move
                np.sign(current_change) != np.sign(next_change) and  # Reversal
                abs(next_change) > abs(current_change) * 0.5):       # Substantial reversal
                
                confidence = min(0.8, abs(current_change) * 100)
                
                signal = DarkPoolSignal(
                    timestamp=recent_data.index[i],
                    symbol=market_data['symbol'],
                    signal_type=InstitutionalOrderType.STOP_HUNT,
                    confidence=confidence,
                    estimated_size=recent_data['volume'].iloc[i],
                    price_level=recent_data['close'].iloc[i],
                    impact_prediction=abs(current_change),
                    time_horizon=60,  # 1 minute
                    metadata={
                        'initial_move': current_change,
                        'reversal_move': next_change,
                        'reversal_ratio': abs(next_change) / abs(current_change)
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _predict_price_impact(self, order_size: float, market_data: Dict[str, Any]) -> float:
        """Predict price impact of order execution"""
        
        price_data = market_data['price_data']
        
        # Simple linear price impact model
        # Impact = sqrt(order_size / average_volume) * volatility
        avg_volume = price_data['volume'].mean()
        current_volatility = price_data['volatility'].iloc[-1] if not pd.isna(price_data['volatility'].iloc[-1]) else 0.01
        
        size_ratio = order_size / avg_volume
        predicted_impact = np.sqrt(size_ratio) * current_volatility * 0.1  # Scaling factor
        
        return min(0.05, predicted_impact)  # Cap at 5%
    
    async def _create_liquidity_profile(
        self, 
        market_data: Dict[str, Any], 
        signals: List[DarkPoolSignal]
    ) -> LiquidityProfile:
        """Create comprehensive liquidity profile"""
        
        price_data = market_data['price_data']
        symbol = market_data['symbol']
        
        # Calculate visible liquidity metrics
        avg_volume = price_data['volume'].mean()
        avg_dollar_volume = price_data['dollar_volume'].mean()
        
        # Estimate hidden liquidity based on detected signals
        total_hidden_size = sum(signal.estimated_size for signal in signals)
        hidden_liquidity_ratio = total_hidden_size / (avg_volume * len(signals)) if signals else 0
        
        # Calculate institutional flow ratio
        institutional_signals = [s for s in signals if s.confidence > self.institutional_confidence]
        institutional_volume = sum(s.estimated_size for s in institutional_signals)
        institutional_flow_ratio = institutional_volume / avg_volume if avg_volume > 0 else 0
        
        # Estimate dark pool participation
        dark_pool_volume = sum(s.estimated_size for s in signals 
                              if s.signal_type in [InstitutionalOrderType.ICEBERG, 
                                                   InstitutionalOrderType.STEALTH,
                                                   InstitutionalOrderType.BLOCK])
        dark_pool_participation = dark_pool_volume / avg_volume if avg_volume > 0 else 0
        
        # Calculate market fragmentation index
        signal_count_by_type = {}
        for signal in signals:
            signal_count_by_type[signal.signal_type] = signal_count_by_type.get(signal.signal_type, 0) + 1
        
        fragmentation_index = len(signal_count_by_type) / 7  # Normalize by total signal types
        
        # Calculate price impact coefficient
        recent_data = price_data.tail(20)
        volume_impact_corr = recent_data['volume'].corr(recent_data['returns'].abs())
        price_impact_coefficient = max(0, volume_impact_corr) if not pd.isna(volume_impact_corr) else 0.1
        
        # Estimate optimal execution size
        avg_spread = recent_data['bid_ask_spread'].mean()
        optimal_size = avg_volume * 0.1 / (1 + avg_spread * 100)  # Heuristic calculation
        
        return LiquidityProfile(
            symbol=symbol,
            timestamp=datetime.now(),
            visible_liquidity=avg_volume,
            hidden_liquidity_estimate=total_hidden_size,
            institutional_flow_ratio=min(1.0, institutional_flow_ratio),
            dark_pool_participation=min(1.0, dark_pool_participation),
            fragmentation_index=min(1.0, fragmentation_index),
            price_impact_coefficient=price_impact_coefficient,
            optimal_execution_size=optimal_size
        )
    
    def get_execution_recommendation(
        self, 
        symbol: str, 
        order_size: float, 
        direction: str
    ) -> Dict[str, Any]:
        """Get optimal execution recommendation based on dark pool analysis"""
        
        # Find recent signals for symbol
        recent_signals = [
            s for s in self.dark_signals 
            if s.symbol == symbol and 
            (datetime.now() - s.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        if not recent_signals:
            return {
                'recommendation': 'standard',
                'confidence': 0.5,
                'explanation': 'No recent dark pool activity detected'
            }
        
        # Analyze current market structure
        institutional_activity = sum(1 for s in recent_signals if s.confidence > 0.7)
        stealth_activity = sum(1 for s in recent_signals if s.signal_type == InstitutionalOrderType.STEALTH)
        
        # Generate recommendation
        if institutional_activity >= 3:
            if stealth_activity >= 1:
                recommendation = 'use_dark_pools'
                explanation = 'High institutional activity detected. Recommend dark pool execution to minimize impact.'
                confidence = 0.8
            else:
                recommendation = 'split_order'
                explanation = 'Institutional activity present. Recommend order splitting over time.'
                confidence = 0.7
        elif len(recent_signals) >= 2:
            recommendation = 'twap_algorithm'
            explanation = 'Moderate activity detected. TWAP execution recommended.'
            confidence = 0.6
        else:
            recommendation = 'standard'
            explanation = 'Low institutional activity. Standard execution acceptable.'
            confidence = 0.5
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'explanation': explanation,
            'detected_signals': len(recent_signals),
            'institutional_signals': institutional_activity,
            'optimal_chunk_size': order_size / max(1, institutional_activity * 2),
            'estimated_completion_time': max(300, institutional_activity * 600)  # seconds
        }
    
    def get_dark_pool_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get summary of dark pool analysis"""
        
        # Filter signals by symbol if specified
        if symbol:
            signals = [s for s in self.dark_signals if s.symbol == symbol]
        else:
            signals = self.dark_signals
        
        # Calculate summary statistics
        signal_counts = {}
        confidence_scores = []
        estimated_sizes = []
        
        for signal in signals:
            signal_type = signal.signal_type.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            confidence_scores.append(signal.confidence)
            estimated_sizes.append(signal.estimated_size)
        
        return {
            'total_signals': len(signals),
            'signal_types': signal_counts,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'total_estimated_volume': sum(estimated_sizes),
            'high_confidence_signals': sum(1 for c in confidence_scores if c > 0.8),
            'recent_signals_1h': sum(
                1 for s in signals 
                if (datetime.now() - s.timestamp).total_seconds() < 3600
            ),
            'symbols_analyzed': len(set(s.symbol for s in signals)) if signals else 0
        }