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
        
        # Identify potential accumulation/distribution phases and springs/upthrusts
        # This is a simplified conceptual implementation. A full Wyckoff detector is complex.

        rolling_window_size = 50 # For establishing trading range
        spring_upthrust_check_window = 5 # Candles to check for recovery after break
        volume_confirmation_period = 5 # Candles for volume check during spring/upthrust

        for i in range(rolling_window_size, len(data) - spring_upthrust_check_window):
            # Define the potential trading range (TR) window
            tr_window = data.iloc[i - rolling_window_size : i]

            tr_low = tr_window['low'].min()
            tr_high = tr_window['high'].max()
            tr_median_close = tr_window['close'].median()
            
            # Sideways price action check
            price_range_metric = (tr_high - tr_low) / tr_median_close if tr_median_close > 0 else 0
            
            if not (0.02 < price_range_metric < 0.10): # Reasonable sideways range (2% to 10%)
                continue

            # Volume characteristics for the TR (e.g., diminishing or average)
            avg_volume_in_tr = tr_window['volume'].mean()
            if avg_volume_in_tr < data['volume_ma'].iloc[i-rolling_window_size:i].mean() * 0.7: # Volume drying up is a good sign for TR
                pass # Or some other TR volume characteristic

            # Check for Spring (potential accumulation)
            # Price dips below TR low then recovers
            current_candle = data.iloc[i]
            if current_candle['low'] < tr_low:
                # Check for recovery in the next few candles
                recovered = False
                increased_volume_on_spring = False
                for j in range(1, spring_upthrust_check_window + 1):
                    if i + j >= len(data): break
                    recovery_candle = data.iloc[i+j]
                    if recovery_candle['close'] > tr_low: # Recovered into the range
                        recovered = True
                        # Check volume on the spring/recovery candles
                        spring_volume_window = data.iloc[i : i+j+1]
                        if spring_volume_window['volume'].mean() > data['volume_ma'].iloc[i] * self.volume_threshold:
                            increased_volume_on_spring = True
                        break
                
                if recovered and increased_volume_on_spring:
                    confidence = 0.75 # Base confidence for spring
                    signals.append(PatternSignal(
                        pattern_type="wyckoff_spring_accumulation",
                        confidence=confidence,
                        entry_price=current_candle['close'], # Entry on confirmation of recovery
                        stop_loss=current_candle['low'] * 0.99, # Below the spring low
                        take_profit=current_candle['close'] * (1 + price_range_metric * 2), # Target based on TR height
                        pattern_data={
                            'trading_range_low': tr_low,
                            'trading_range_high': tr_high,
                            'spring_low': current_candle['low'],
                            'volume_increase_ratio': data.iloc[i:i+spring_upthrust_check_window]['volume'].mean() / avg_volume_in_tr if avg_volume_in_tr else 0,
                        },
                        timestamp=data.index[i+j] if 'j' in locals() and i+j < len(data) else data.index[i]
                    ))

            # Check for Upthrust (potential distribution)
            # Price pushes above TR high then rejects
            if current_candle['high'] > tr_high:
                rejected = False
                increased_volume_on_upthrust = False
                for j in range(1, spring_upthrust_check_window + 1):
                    if i + j >= len(data): break
                    rejection_candle = data.iloc[i+j]
                    if rejection_candle['close'] < tr_high: # Rejected back into the range
                        rejected = True
                         # Check volume on the upthrust/rejection candles
                        upthrust_volume_window = data.iloc[i : i+j+1]
                        if upthrust_volume_window['volume'].mean() > data['volume_ma'].iloc[i] * self.volume_threshold:
                            increased_volume_on_upthrust = True
                        break

                if rejected and increased_volume_on_upthrust:
                    confidence = 0.75 # Base confidence for upthrust
                    signals.append(PatternSignal(
                        pattern_type="wyckoff_upthrust_distribution",
                        confidence=confidence,
                        entry_price=current_candle['close'], # Entry on confirmation of rejection
                        stop_loss=current_candle['high'] * 1.01, # Above the upthrust high
                        take_profit=current_candle['close'] * (1 - price_range_metric * 2),
                        pattern_data={
                            'trading_range_low': tr_low,
                            'trading_range_high': tr_high,
                            'upthrust_high': current_candle['high'],
                            'volume_increase_ratio': data.iloc[i:i+spring_upthrust_check_window]['volume'].mean() / avg_volume_in_tr if avg_volume_in_tr else 0,
                        },
                        timestamp=data.index[i+j] if 'j' in locals() and i+j < len(data) else data.index[i]
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
        avg_volume_period = 20 # Look back period for average volume calculation
        if 'volume' not in data.columns: # Ensure volume data is present
            return signals
        data['volume_ma_short'] = data['volume'].rolling(window=avg_volume_period).mean()

        for i, high_idx in enumerate(swing_highs[1:], 1):
            prev_high_idx = swing_highs[i-1]
            prev_high_val = data['high'].iloc[prev_high_idx]
            
            # Ensure high_idx is a valid index for current_candle related operations
            if high_idx >= len(data): continue
            current_candle = data.iloc[high_idx]
            current_high_val = current_candle['high']

            if current_high_val > prev_high_val * 1.001:  # 0.1% buffer for BOS
                # Volume confirmation for BOS
                # Check volume of the candle that broke the structure
                bos_candle_volume = current_candle['volume']
                avg_volume_before_bos = data['volume_ma_short'].iloc[max(0, high_idx-1)] # Avg volume just before BOS

                volume_confirmed = False
                if avg_volume_before_bos > 0 and bos_candle_volume > avg_volume_before_bos * 1.5: # e.g., 50% higher
                    volume_confirmed = True

                if volume_confirmed:
                    signals.append(PatternSignal(
                        pattern_type="smc_bullish_bos_vol_confirmed",
                        confidence=0.85, # Higher confidence due to volume confirmation
                        entry_price=current_high_val,
                        stop_loss=data['low'].iloc[max(0, high_idx-10):high_idx].min(), # Or related swing low
                        take_profit=current_high_val * 1.08, # Example take profit
                        pattern_data={
                            'previous_high': prev_high_val,
                            'bos_candle_volume': bos_candle_volume,
                            'avg_volume_before_bos': avg_volume_before_bos
                        },
                        timestamp=data.index[high_idx]
                    ))
                else: # BOS without strong volume confirmation (lower confidence)
                    signals.append(PatternSignal(
                        pattern_type="smc_bullish_bos",
                        confidence=0.65,
                        entry_price=current_high_val,
                        stop_loss=data['low'].iloc[max(0, high_idx-10):high_idx].min(),
                        take_profit=current_high_val * 1.08,
                        pattern_data={'previous_high': prev_high_val, 'volume_confirmed': False},
                        timestamp=data.index[high_idx]
                    ))

        # Bearish BOS would be similar, checking breaks of swing_lows downwards. (Not implemented here for brevity)
        return signals
    
    def _detect_fair_value_gaps(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect fair value gaps (FVG)"""
        signals = []
        
        for i in range(2, len(data)):
            # Bullish FVG: gap between candle 1 high and candle 3 low
            candle1 = data.iloc[i-2]
            candle2 = data.iloc[i-1] # The middle candle, potentially high momentum
            candle3 = data.iloc[i]
            
            # Bullish FVG: gap between candle 1 high and candle 3 low
            if candle3['low'] > candle1['high']:
                gap_size_abs = candle3['low'] - candle1['high']
                gap_size_rel = gap_size_abs / candle2['close'] if candle2['close'] > 0 else 0

                if gap_size_rel > 0.002:  # Minimum 0.2% relative gap size

                    # Check middle candle (candle2) momentum and volume
                    candle2_body = abs(candle2['close'] - candle2['open'])
                    candle2_range = candle2['high'] - candle2['low']
                    is_candle2_high_momentum = (candle2_body / candle2_range > 0.7) if candle2_range > 0 else False

                    # Check volume of candle2 (assuming 'volume_ma_short' is available or calculate it)
                    if 'volume_ma_short' not in data.columns:
                         data['volume_ma_short'] = data['volume'].rolling(window=20).mean() # Ensure it's present

                    is_candle2_volume_notable = candle2['volume'] > data['volume_ma_short'].iloc[i-1] * 1.2 # e.g. 20% above avg

                    confidence = min(0.7, gap_size_rel * 100) # Base confidence
                    pattern_type = "smc_bullish_fvg"

                    if is_candle2_high_momentum and is_candle2_volume_notable:
                        confidence = min(0.85, confidence * 1.2) # Boost confidence
                        pattern_type = "smc_bullish_fvg_strong_impulse"

                    signals.append(PatternSignal(
                        pattern_type=pattern_type,
                        confidence=confidence,
                        entry_price=candle1['high'], # Entry at the start of the gap
                        stop_loss=candle1['low'] * 0.995, # Example SL below candle1 low
                        take_profit=candle3['low'], # Target filling the gap
                        pattern_data={
                            'gap_size_abs': gap_size_abs,
                            'gap_size_rel': gap_size_rel,
                            'fvg_high': candle3['low'],
                            'fvg_low': candle1['high'],
                            'middle_candle_momentum': is_candle2_high_momentum,
                            'middle_candle_volume_notable': is_candle2_volume_notable,
                        },
                        timestamp=data.index[i]
                    ))
        
        # Bearish FVG: gap between candle 1 low and candle 3 high (not implemented for brevity)
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