"""
Pattern Recognition Strategy Templates
Advanced templates for technical analysis patterns
"""

def get_pattern_recognition_template() -> str:
    """Advanced Pattern Recognition template with multiple techniques"""
    return '''
from tier2_strategy.pattern_recognition import AdvancedPatternRecognizer, VolumeProfileAnalyzer

class AdvancedPatternRecognitionAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.symbol = self.AddEquity("{symbol}", Resolution.Daily).Symbol
        
        # Pattern recognition parameters
        self.pattern_confidence_threshold = {pattern_confidence_threshold}
        self.wyckoff_volume_threshold = {wyckoff_volume_threshold}
        self.fibonacci_confluence_threshold = {fibonacci_confluence_threshold}
        self.smc_enabled = {smc_enabled}
        self.volume_profile_enabled = {volume_profile_enabled}
        self.position_size = {position_size}
        self.risk_per_trade = {risk_per_trade}
        
        # Initialize pattern recognizer
        self.pattern_recognizer = AdvancedPatternRecognizer()
        self.volume_analyzer = VolumeProfileAnalyzer()
        
        # Data management
        self.lookback_period = 100
        self.price_data = []
        
        # Position management
        self.current_signal = None
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        
    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return
            
        # Update price data
        bar = data[self.symbol]
        self.price_data.append({
            'timestamp': self.Time,
            'open': float(bar.Open),
            'high': float(bar.High),
            'low': float(bar.Low),
            'close': float(bar.Close),
            'volume': float(bar.Volume)
        })
        
        # Keep only recent data
        if len(self.price_data) > self.lookback_period:
            self.price_data = self.price_data[-self.lookback_period:]
        
        # Need sufficient data for analysis
        if len(self.price_data) < 50:
            return
            
        # Convert to DataFrame for analysis
        import pandas as pd
        df = pd.DataFrame(self.price_data)
        df.set_index('timestamp', inplace=True)
        
        # Get current position
        current_holdings = self.Portfolio[self.symbol].Quantity
        
        if current_holdings == 0:
            # Look for entry signals
            signal = self.pattern_recognizer.get_trading_signal(df)
            
            if signal and signal.confidence >= self.pattern_confidence_threshold:
                # Additional volume confirmation if enabled
                if self.volume_profile_enabled:
                    microstructure = self.volume_analyzer.analyze_volume_profile(df)
                    
                    # Confirm signal with volume profile
                    if self._confirm_with_volume_profile(signal, microstructure):
                        signal.confidence = min(0.95, signal.confidence * 1.1)
                    else:
                        return  # Skip this signal
                
                # Execute trade
                self._execute_entry(signal)
                
        else:
            # Manage existing position
            self._manage_position(data[self.symbol])
    
    def _confirm_with_volume_profile(self, signal, microstructure):
        """Confirm signal with volume profile analysis"""
        if not microstructure:
            return False
            
        current_price = self.price_data[-1]['close']
        
        # Check if price is near value area for mean reversion setups
        if 'reversion' in signal.pattern_type:
            near_va = (microstructure.value_area_low <= current_price <= microstructure.value_area_high)
            return not near_va  # Want to be outside value area for reversion
        
        # Check volume confirmation for breakout setups
        elif 'breakout' in signal.pattern_type or 'bos' in signal.pattern_type:
            return microstructure.smart_money_flow > 0.1
        
        # Check order flow for trend continuation
        elif 'trend' in signal.pattern_type:
            if 'bullish' in signal.pattern_type:
                return microstructure.order_flow_imbalance > 0.2
            else:
                return microstructure.order_flow_imbalance < -0.2
        
        return True  # Default to allowing the signal
    
    def _execute_entry(self, signal):
        """Execute entry based on pattern signal"""
        self.current_signal = signal
        self.entry_price = signal.entry_price
        
        # Calculate position size based on risk
        if signal.stop_loss:
            risk_per_share = abs(signal.entry_price - signal.stop_loss)
            max_shares = (self.Portfolio.TotalPortfolioValue * self.risk_per_trade) / risk_per_share
            target_value = max_shares * signal.entry_price
            position_pct = min(self.position_size, target_value / self.Portfolio.TotalPortfolioValue)
        else:
            position_pct = self.position_size
        
        # Determine direction
        if 'bullish' in signal.pattern_type or 'long' in signal.pattern_type:
            self.SetHoldings(self.symbol, position_pct)
        elif 'bearish' in signal.pattern_type or 'short' in signal.pattern_type:
            self.SetHoldings(self.symbol, -position_pct)
        
        # Set stop loss and take profit
        if signal.stop_loss:
            self.stop_loss_price = signal.stop_loss
        if signal.take_profit:
            self.take_profit_price = signal.take_profit
            
        self.Debug(f"Entered {signal.pattern_type} position with confidence {signal.confidence:.2f}")
    
    def _manage_position(self, current_bar):
        """Manage existing position"""
        current_price = float(current_bar.Close)
        
        # Check stop loss
        if self.stop_loss_price:
            if (self.Portfolio[self.symbol].IsLong and current_price <= self.stop_loss_price) or \
               (self.Portfolio[self.symbol].IsShort and current_price >= self.stop_loss_price):
                self.Liquidate(self.symbol)
                self.Debug(f"Stop loss hit at {current_price}")
                self._reset_position_data()
                return
        
        # Check take profit
        if self.take_profit_price:
            if (self.Portfolio[self.symbol].IsLong and current_price >= self.take_profit_price) or \
               (self.Portfolio[self.symbol].IsShort and current_price <= self.take_profit_price):
                self.Liquidate(self.symbol)
                self.Debug(f"Take profit hit at {current_price}")
                self._reset_position_data()
                return
        
        # Pattern-specific exit rules could be added here
        
    def _reset_position_data(self):
        """Reset position tracking data"""
        self.current_signal = None
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
'''


def get_smc_template() -> str:
    """Smart Money Concepts template"""
    return '''
from tier2_strategy.pattern_recognition import SmartMoneyConceptDetector

class SmartMoneyConceptsAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.symbol = self.AddEquity("{symbol}", Resolution.Daily).Symbol
        
        # SMC parameters
        self.swing_detection_period = {swing_detection_period}
        self.fvg_min_gap_size = {fvg_min_gap_size}
        self.ob_lookback_period = {ob_lookback_period}
        self.bos_confirmation_buffer = {bos_confirmation_buffer}
        self.position_size = {position_size}
        
        # Initialize SMC detector
        self.smc_detector = SmartMoneyConceptDetector()
        
        # Data management
        self.lookback_period = 200
        self.price_data = []
        
        # SMC state tracking
        self.market_structure = "neutral"  # bullish, bearish, neutral
        self.key_levels = []
        self.active_fvgs = []
        self.order_blocks = []
        
    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return
            
        # Update price data
        bar = data[self.symbol]
        self.price_data.append({
            'timestamp': self.Time,
            'open': float(bar.Open),
            'high': float(bar.High),
            'low': float(bar.Low),
            'close': float(bar.Close),
            'volume': float(bar.Volume)
        })
        
        # Keep only recent data
        if len(self.price_data) > self.lookback_period:
            self.price_data = self.price_data[-self.lookback_period:]
        
        if len(self.price_data) < 50:
            return
            
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(self.price_data)
        df.set_index('timestamp', inplace=True)
        
        # Update market structure analysis
        self._update_market_structure(df)
        
        # Get SMC signals
        smc_signals = self.smc_detector.detect(df)
        
        current_holdings = self.Portfolio[self.symbol].Quantity
        
        if current_holdings == 0:
            # Look for entry opportunities
            for signal in smc_signals:
                if self._validate_smc_entry(signal, df):
                    self._execute_smc_entry(signal)
                    break
        else:
            # Manage existing position
            self._manage_smc_position(data[self.symbol], df)
    
    def _update_market_structure(self, df):
        """Update market structure based on break of structure"""
        if len(df) < 20:
            return
            
        # Simple trend determination based on recent highs and lows
        recent_data = df.tail(20)
        
        if recent_data['high'].iloc[-1] > recent_data['high'].iloc[-10]:
            if recent_data['low'].iloc[-1] > recent_data['low'].iloc[-10]:
                self.market_structure = "bullish"
        elif recent_data['low'].iloc[-1] < recent_data['low'].iloc[-10]:
            if recent_data['high'].iloc[-1] < recent_data['high'].iloc[-10]:
                self.market_structure = "bearish"
        else:
            self.market_structure = "neutral"
    
    def _validate_smc_entry(self, signal, df):
        """Validate SMC entry signal"""
        current_price = df['close'].iloc[-1]
        
        # Only trade in direction of market structure
        if signal.pattern_type == "smc_bullish_bos" and self.market_structure != "bullish":
            return False
        if signal.pattern_type == "smc_bearish_bos" and self.market_structure != "bearish":
            return False
        
        # Validate price levels
        if signal.pattern_type == "smc_bullish_fvg":
            # Price should be approaching the FVG from below
            fvg_low = signal.pattern_data.get('fvg_low', signal.entry_price)
            return current_price <= fvg_low * 1.005  # 0.5% buffer
        
        if signal.pattern_type == "smc_bullish_ob":
            # Price should be at or near the order block
            ob_low = signal.pattern_data.get('ob_low', signal.entry_price)
            ob_high = signal.pattern_data.get('ob_high', signal.entry_price)
            return ob_low <= current_price <= ob_high * 1.01
        
        return signal.confidence > 0.7
    
    def _execute_smc_entry(self, signal):
        """Execute SMC-based entry"""
        direction = 1 if 'bullish' in signal.pattern_type else -1
        
        self.SetHoldings(self.symbol, direction * self.position_size)
        
        # Set stop loss and take profit based on SMC principles
        if signal.stop_loss:
            self.stop_loss = signal.stop_loss
        else:
            # Default stop loss below/above the structure
            current_price = self.Securities[self.symbol].Price
            if direction == 1:
                self.stop_loss = current_price * 0.98
            else:
                self.stop_loss = current_price * 1.02
        
        if signal.take_profit:
            self.take_profit = signal.take_profit
        else:
            # Default take profit at 2:1 risk-reward
            risk = abs(current_price - self.stop_loss)
            if direction == 1:
                self.take_profit = current_price + (risk * 2)
            else:
                self.take_profit = current_price - (risk * 2)
        
        self.entry_time = self.Time
        self.Debug(f"SMC Entry: {signal.pattern_type} at {current_price}")
    
    def _manage_smc_position(self, current_bar, df):
        """Manage SMC position"""
        current_price = float(current_bar.Close)
        
        # Check stop loss
        if hasattr(self, 'stop_loss'):
            if (self.Portfolio[self.symbol].IsLong and current_price <= self.stop_loss) or \
               (self.Portfolio[self.symbol].IsShort and current_price >= self.stop_loss):
                self.Liquidate(self.symbol)
                self.Debug(f"SMC Stop loss at {current_price}")
                return
        
        # Check take profit
        if hasattr(self, 'take_profit'):
            if (self.Portfolio[self.symbol].IsLong and current_price >= self.take_profit) or \
               (self.Portfolio[self.symbol].IsShort and current_price <= self.take_profit):
                self.Liquidate(self.symbol)
                self.Debug(f"SMC Take profit at {current_price}")
                return
        
        # Check for structure break against position
        if hasattr(self, 'entry_time'):
            if (self.Time - self.entry_time).days > 10:  # Max holding period
                self.Liquidate(self.symbol)
                self.Debug(f"SMC Max holding period reached")
'''