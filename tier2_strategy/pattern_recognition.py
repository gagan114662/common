"""
Advanced Technical Analysis & Pattern Recognition Module
Implements professional-grade chart pattern detection and microstructure analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import ta
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PatternSignal:
    """Pattern recognition signal"""
    pattern_type: str
    confidence: float  # 0-1
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "Daily"
    pattern_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: pd.Timestamp = None

@dataclass
class MarketMicrostructure:
    """Market microstructure analysis results"""
    order_flow_imbalance: float
    volume_profile_poc: float  # Point of Control
    value_area_high: float
    value_area_low: float
    market_maker_bias: str  # 'bullish', 'bearish', 'neutral'
    liquidity_zones: List[Tuple[float, float]]
    smart_money_flow: float

class PatternDetector(ABC):
    """Base class for pattern detection algorithms"""
    
    @abstractmethod
    def detect(self, data: pd.DataFrame) -> List[PatternSignal]:
        pass
    
    @abstractmethod
    def get_pattern_name(self) -> str:
        pass

class WyckoffPatternDetector(PatternDetector):
    """Wyckoff accumulation/distribution pattern detection"""
    
    def __init__(self, volume_threshold: float = 1.5):
        self.volume_threshold = volume_threshold
    
    def detect(self, data: pd.DataFrame) -> List[PatternSignal]:
        signals = []
        
        # Ensure we have required columns
        if not all(col in data.columns for col in ['close', 'volume', 'high', 'low']):
            return signals
        
        # Calculate volume moving average
        data['volume_ma'] = data['volume'].rolling(20).mean()
        
        # Identify potential accumulation zones
        for i in range(50, len(data) - 10):
            window = data.iloc[i-20:i+10]
            
            # Check for sideways price action with increasing volume
            price_range = (window['high'].max() - window['low'].min()) / window['close'].iloc[-1]
            volume_increase = window['volume'].iloc[-10:].mean() / window['volume'].iloc[:10].mean()
            
            if price_range < 0.05 and volume_increase > self.volume_threshold:
                # Potential accumulation
                confidence = min(0.9, volume_increase / 3.0)
                
                signals.append(PatternSignal(
                    pattern_type="wyckoff_accumulation",
                    confidence=confidence,
                    entry_price=data['close'].iloc[i],
                    stop_loss=window['low'].min() * 0.98,
                    take_profit=data['close'].iloc[i] * 1.15,
                    pattern_data={
                        'volume_increase': volume_increase,
                        'price_range': price_range,
                        'poc_price': window['close'].median()
                    },
                    timestamp=data.index[i]
                ))
        
        return signals
    
    def get_pattern_name(self) -> str:
        return "Wyckoff Accumulation/Distribution"

class SmartMoneyConceptDetector(PatternDetector):
    """Smart Money Concept (SMC) pattern detection"""
    
    def detect(self, data: pd.DataFrame) -> List[PatternSignal]:
        signals = []
        
        # Calculate swing highs and lows
        swing_highs = self._find_swing_points(data['high'], order=5, find_highs=True)
        swing_lows = self._find_swing_points(data['low'], order=5, find_highs=False)
        
        # Identify break of structure (BOS)
        bos_signals = self._detect_break_of_structure(data, swing_highs, swing_lows)
        signals.extend(bos_signals)
        
        # Identify fair value gaps (FVG)
        fvg_signals = self._detect_fair_value_gaps(data)
        signals.extend(fvg_signals)
        
        # Order blocks
        ob_signals = self._detect_order_blocks(data, swing_highs, swing_lows)
        signals.extend(ob_signals)
        
        return signals
    
    def _find_swing_points(self, series: pd.Series, order: int = 5, find_highs: bool = True) -> List[int]:
        """Find swing highs or lows"""
        if find_highs:
            peaks, _ = find_peaks(series, distance=order)
            return peaks.tolist()
        else:
            troughs, _ = find_peaks(-series, distance=order)
            return troughs.tolist()
    
    def _detect_break_of_structure(self, data: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> List[PatternSignal]:
        """Detect break of structure patterns"""
        signals = []
        
        # Bullish BOS - price breaks above previous swing high
        for i, high_idx in enumerate(swing_highs[1:], 1):
            prev_high = data['high'].iloc[swing_highs[i-1]]
            current_high = data['high'].iloc[high_idx]
            
            if current_high > prev_high * 1.001:  # 0.1% buffer
                signals.append(PatternSignal(
                    pattern_type="smc_bullish_bos",
                    confidence=0.75,
                    entry_price=current_high,
                    stop_loss=data['low'].iloc[max(0, high_idx-10):high_idx].min(),
                    take_profit=current_high * 1.08,
                    pattern_data={'previous_high': prev_high},
                    timestamp=data.index[high_idx]
                ))
        
        return signals
    
    def _detect_fair_value_gaps(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect fair value gaps (FVG)"""
        signals = []
        
        for i in range(2, len(data)):
            # Bullish FVG: gap between candle 1 high and candle 3 low
            candle1_high = data['high'].iloc[i-2]
            candle3_low = data['low'].iloc[i]
            
            if candle3_low > candle1_high:
                gap_size = (candle3_low - candle1_high) / data['close'].iloc[i]
                
                if gap_size > 0.002:  # Minimum 0.2% gap
                    signals.append(PatternSignal(
                        pattern_type="smc_bullish_fvg",
                        confidence=min(0.8, gap_size * 100),
                        entry_price=candle1_high,
                        take_profit=candle3_low,
                        pattern_data={
                            'gap_size': gap_size,
                            'fvg_high': candle3_low,
                            'fvg_low': candle1_high
                        },
                        timestamp=data.index[i]
                    ))
        
        return signals
    
    def _detect_order_blocks(self, data: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> List[PatternSignal]:
        """Detect institutional order blocks"""
        signals = []
        
        # Bullish order block: last bearish candle before swing low
        for low_idx in swing_lows:
            if low_idx < 5:
                continue
                
            # Find last bearish candle before swing
            for j in range(low_idx-1, max(0, low_idx-10), -1):
                if data['close'].iloc[j] < data['open'].iloc[j]:
                    # This is our order block
                    ob_high = data['high'].iloc[j]
                    ob_low = data['low'].iloc[j]
                    
                    signals.append(PatternSignal(
                        pattern_type="smc_bullish_ob",
                        confidence=0.7,
                        entry_price=ob_low,
                        stop_loss=ob_low * 0.995,
                        take_profit=ob_high * 1.05,
                        pattern_data={
                            'ob_high': ob_high,
                            'ob_low': ob_low,
                            'swing_low': data['low'].iloc[low_idx]
                        },
                        timestamp=data.index[j]
                    ))
                    break
        
        return signals
    
    def get_pattern_name(self) -> str:
        return "Smart Money Concepts"

class FibonacciClusterDetector(PatternDetector):
    """Fibonacci retracement and extension cluster analysis"""
    
    def __init__(self, confluence_threshold: int = 3):
        self.confluence_threshold = confluence_threshold
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
    
    def detect(self, data: pd.DataFrame) -> List[PatternSignal]:
        signals = []
        
        # Find significant swings
        swing_highs = self._find_swing_points(data['high'], order=10, find_highs=True)
        swing_lows = self._find_swing_points(data['low'], order=10, find_highs=False)
        
        # Calculate fibonacci clusters
        clusters = self._calculate_fib_clusters(data, swing_highs, swing_lows)
        
        # Generate signals from clusters
        for cluster in clusters:
            if cluster['confluence'] >= self.confluence_threshold:
                signals.append(PatternSignal(
                    pattern_type="fibonacci_cluster",
                    confidence=min(0.95, cluster['confluence'] / 10),
                    entry_price=cluster['price'],
                    stop_loss=cluster['price'] * 0.985 if cluster['bias'] == 'bullish' else cluster['price'] * 1.015,
                    take_profit=cluster['price'] * 1.03 if cluster['bias'] == 'bullish' else cluster['price'] * 0.97,
                    pattern_data=cluster,
                    timestamp=cluster['timestamp']
                ))
        
        return signals
    
    def _find_swing_points(self, series: pd.Series, order: int = 10, find_highs: bool = True) -> List[int]:
        """Find swing highs or lows"""
        if find_highs:
            peaks, _ = find_peaks(series, distance=order)
            return peaks.tolist()
        else:
            troughs, _ = find_peaks(-series, distance=order)
            return troughs.tolist()
    
    def _calculate_fib_clusters(self, data: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> List[Dict]:
        """Calculate fibonacci cluster zones"""
        clusters = []
        current_price = data['close'].iloc[-1]
        
        # Calculate all fibonacci levels from recent swings
        fib_levels = []
        
        # From swing highs to lows (retracements)
        for high_idx in swing_highs[-5:]:  # Last 5 swings
            for low_idx in swing_lows[-5:]:
                if abs(high_idx - low_idx) < 5:  # Skip if too close
                    continue
                    
                high = data['high'].iloc[high_idx]
                low = data['low'].iloc[low_idx]
                
                for level in self.fib_levels:
                    if high > low:  # Downtrend retracement
                        fib_price = low + (high - low) * level
                    else:  # Uptrend retracement
                        fib_price = high + (low - high) * level
                    
                    fib_levels.append({
                        'price': fib_price,
                        'level': level,
                        'swing_high': high,
                        'swing_low': low
                    })
        
        # Find clusters
        price_tolerance = current_price * 0.005  # 0.5% tolerance
        
        for i, fib1 in enumerate(fib_levels):
            cluster_fibs = [fib1]
            cluster_price = fib1['price']
            
            # Find nearby levels
            for j, fib2 in enumerate(fib_levels[i+1:], i+1):
                if abs(fib2['price'] - cluster_price) <= price_tolerance:
                    cluster_fibs.append(fib2)
            
            if len(cluster_fibs) >= self.confluence_threshold:
                avg_price = np.mean([f['price'] for f in cluster_fibs])
                
                # Determine bias based on recent price action
                if current_price < avg_price:
                    bias = 'bullish'
                else:
                    bias = 'bearish'
                
                clusters.append({
                    'price': avg_price,
                    'confluence': len(cluster_fibs),
                    'bias': bias,
                    'levels': cluster_fibs,
                    'timestamp': data.index[-1]
                })
        
        return clusters
    
    def get_pattern_name(self) -> str:
        return "Fibonacci Clusters"

class VolumeProfileAnalyzer:
    """Volume Profile and Market Microstructure Analysis"""
    
    def __init__(self, profile_periods: int = 20):
        self.profile_periods = profile_periods
    
    def analyze_volume_profile(self, data: pd.DataFrame) -> MarketMicrostructure:
        """Analyze volume profile and market microstructure"""
        
        # Create price-volume profile
        price_range = data['high'].max() - data['low'].min()
        price_levels = np.linspace(data['low'].min(), data['high'].max(), 100)
        volume_at_price = np.zeros(100)
        
        # Distribute volume across price levels
        for i, row in data.iterrows():
            candle_range = row['high'] - row['low']
            if candle_range == 0:
                continue
                
            # Find price level indices for this candle
            start_idx = int((row['low'] - data['low'].min()) / price_range * 99)
            end_idx = int((row['high'] - data['low'].min()) / price_range * 99)
            
            # Distribute volume proportionally
            volume_per_level = row['volume'] / max(1, end_idx - start_idx + 1)
            volume_at_price[start_idx:end_idx+1] += volume_per_level
        
        # Find Point of Control (highest volume)
        poc_idx = np.argmax(volume_at_price)
        poc_price = price_levels[poc_idx]
        
        # Calculate Value Area (70% of volume)
        total_volume = np.sum(volume_at_price)
        target_volume = total_volume * 0.7
        
        # Expand from POC until we reach 70% volume
        va_low_idx = va_high_idx = poc_idx
        current_volume = volume_at_price[poc_idx]
        
        while current_volume < target_volume and (va_low_idx > 0 or va_high_idx < len(volume_at_price)-1):
            # Expand to side with more volume
            low_volume = volume_at_price[va_low_idx-1] if va_low_idx > 0 else 0
            high_volume = volume_at_price[va_high_idx+1] if va_high_idx < len(volume_at_price)-1 else 0
            
            if low_volume >= high_volume and va_low_idx > 0:
                va_low_idx -= 1
                current_volume += volume_at_price[va_low_idx]
            elif va_high_idx < len(volume_at_price)-1:
                va_high_idx += 1
                current_volume += volume_at_price[va_high_idx]
            else:
                break
        
        va_low = price_levels[va_low_idx]
        va_high = price_levels[va_high_idx]
        
        # Calculate order flow imbalance
        order_flow_imbalance = self._calculate_order_flow_imbalance(data)
        
        # Identify liquidity zones (high volume areas)
        liquidity_zones = self._identify_liquidity_zones(price_levels, volume_at_price)
        
        # Determine market maker bias
        mm_bias = self._determine_market_maker_bias(data, poc_price)
        
        # Calculate smart money flow
        smart_money_flow = self._calculate_smart_money_flow(data)
        
        return MarketMicrostructure(
            order_flow_imbalance=order_flow_imbalance,
            volume_profile_poc=poc_price,
            value_area_high=va_high,
            value_area_low=va_low,
            market_maker_bias=mm_bias,
            liquidity_zones=liquidity_zones,
            smart_money_flow=smart_money_flow
        )
    
    def _calculate_order_flow_imbalance(self, data: pd.DataFrame) -> float:
        """Calculate order flow imbalance"""
        # Approximate buying/selling pressure
        buying_pressure = 0
        selling_pressure = 0
        
        for i, row in data.iterrows():
            close_position = (row['close'] - row['low']) / max(row['high'] - row['low'], 0.001)
            
            if close_position > 0.5:
                buying_pressure += row['volume'] * close_position
            else:
                selling_pressure += row['volume'] * (1 - close_position)
        
        total_flow = buying_pressure + selling_pressure
        if total_flow == 0:
            return 0
        
        return (buying_pressure - selling_pressure) / total_flow
    
    def _identify_liquidity_zones(self, price_levels: np.ndarray, volume_at_price: np.ndarray) -> List[Tuple[float, float]]:
        """Identify high liquidity zones"""
        volume_threshold = np.percentile(volume_at_price, 80)
        liquidity_zones = []
        
        in_zone = False
        zone_start = 0
        
        for i, volume in enumerate(volume_at_price):
            if volume >= volume_threshold and not in_zone:
                in_zone = True
                zone_start = i
            elif volume < volume_threshold and in_zone:
                in_zone = False
                if i - zone_start > 2:  # Minimum zone size
                    liquidity_zones.append((price_levels[zone_start], price_levels[i-1]))
        
        # Close final zone if needed
        if in_zone:
            liquidity_zones.append((price_levels[zone_start], price_levels[-1]))
        
        return liquidity_zones
    
    def _determine_market_maker_bias(self, data: pd.DataFrame, poc_price: float) -> str:
        """Determine market maker bias"""
        current_price = data['close'].iloc[-1]
        
        # Compare current price to POC
        if current_price > poc_price * 1.01:
            return 'bullish'
        elif current_price < poc_price * 0.99:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_smart_money_flow(self, data: pd.DataFrame) -> float:
        """Calculate smart money flow indicator"""
        # Money Flow Index calculation
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).sum()
        
        if positive_flow + negative_flow == 0:
            return 0
        
        return (positive_flow - negative_flow) / (positive_flow + negative_flow)

class AdvancedPatternRecognizer:
    """
    Advanced pattern recognition system combining multiple detection methods
    """
    
    def __init__(self):
        self.detectors = [
            WyckoffPatternDetector(),
            SmartMoneyConceptDetector(),
            FibonacciClusterDetector()
        ]
        self.volume_analyzer = VolumeProfileAnalyzer()
        
    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive market analysis"""
        
        # Run all pattern detectors
        all_signals = []
        for detector in self.detectors:
            try:
                signals = detector.detect(data)
                all_signals.extend(signals)
            except Exception as e:
                print(f"Error in {detector.get_pattern_name()}: {e}")
        
        # Analyze market microstructure
        try:
            microstructure = self.volume_analyzer.analyze_volume_profile(data)
        except Exception as e:
            print(f"Error in volume analysis: {e}")
            microstructure = None
        
        # Filter and rank signals
        high_confidence_signals = [s for s in all_signals if s.confidence > 0.7]
        
        return {
            'all_signals': all_signals,
            'high_confidence_signals': high_confidence_signals,
            'microstructure': microstructure,
            'signal_count': len(all_signals),
            'pattern_types': list(set(s.pattern_type for s in all_signals))
        }
    
    def get_trading_signal(self, data: pd.DataFrame) -> Optional[PatternSignal]:
        """Get the best trading signal from current analysis"""
        analysis = self.analyze_market(data)
        
        if not analysis['high_confidence_signals']:
            return None
        
        # Return highest confidence signal
        best_signal = max(analysis['high_confidence_signals'], key=lambda x: x.confidence)
        
        # Enhance with microstructure confirmation
        if analysis['microstructure']:
            ms = analysis['microstructure']
            
            # Confirm bullish signals
            if 'bullish' in best_signal.pattern_type:
                if ms.order_flow_imbalance > 0.2 and ms.smart_money_flow > 0.1:
                    best_signal.confidence = min(0.95, best_signal.confidence * 1.2)
            
            # Confirm bearish signals  
            elif 'bearish' in best_signal.pattern_type:
                if ms.order_flow_imbalance < -0.2 and ms.smart_money_flow < -0.1:
                    best_signal.confidence = min(0.95, best_signal.confidence * 1.2)
        
        return best_signal if best_signal.confidence > 0.75 else None