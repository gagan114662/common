"""
TIER 2: Template-Based Strategy Generator
High-performance strategy generation system capable of 100+ strategies per hour
Based on Gemini's recommendations for template-based generation
"""

import asyncio
import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np
from itertools import product
import ast
import re

from tier1_core.logger import get_logger, PERF_LOGGER
from tier1_core.quantconnect_client import QuantConnectClient
from tier1_core.real_time_dashboard import DASHBOARD
from tier2_strategy.pattern_templates import get_pattern_recognition_template, get_smc_template

@dataclass
class ParameterRange:
    """Parameter value range for optimization"""
    name: str
    type: str  # 'int', 'float', 'choice', 'bool'
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    step: Optional[Union[int, float]] = None
    
    def generate_value(self) -> Any:
        """Generate a random value within the parameter range"""
        if self.type == 'bool':
            return random.choice([True, False])
        elif self.type == 'choice':
            return random.choice(self.choices)
        elif self.type == 'int':
            if self.step:
                values = list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
                return random.choice(values)
            else:
                return random.randint(int(self.min_value), int(self.max_value))
        elif self.type == 'float':
            if self.step:
                num_steps = int((self.max_value - self.min_value) / self.step)
                step_num = random.randint(0, num_steps)
                return self.min_value + step_num * self.step
            else:
                return random.uniform(self.min_value, self.max_value)
        else:
            return self.default

@dataclass
class StrategyTemplate:
    """Strategy template with parameterized code"""
    name: str
    description: str
    category: str  # 'trend_following', 'mean_reversion', 'momentum', 'arbitrage', 'market_neutral'
    code_template: str
    parameters: List[ParameterRange]
    asset_classes: List[str] = field(default_factory=lambda: ["Equity"])
    timeframes: List[str] = field(default_factory=lambda: ["Daily"])
    minimum_history: int = 252  # Trading days
    complexity_score: int = 1  # 1-10 scale
    expected_turnover: float = 0.1  # Annual turnover estimate
    
    def generate_code(self, parameter_values: Dict[str, Any]) -> str:
        """Generate strategy code with specific parameter values"""
        code = self.code_template
        
        # Replace parameter placeholders
        for param_name, value in parameter_values.items():
            if isinstance(value, str):
                code = code.replace(f"{{{param_name}}}", f'"{value}"')
            else:
                code = code.replace(f"{{{param_name}}}", str(value))
        
        return code
    
    def generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameter values within ranges"""
        return {param.name: param.generate_value() for param in self.parameters}
    
    def validate_parameters(self, parameter_values: Dict[str, Any]) -> bool:
        """Validate parameter values against ranges"""
        for param in self.parameters:
            if param.name not in parameter_values:
                return False
            
            value = parameter_values[param.name]
            
            if param.type == 'int' and not isinstance(value, int):
                return False
            elif param.type == 'float' and not isinstance(value, (int, float)):
                return False
            elif param.type == 'choice' and value not in param.choices:
                return False
            elif param.type == 'bool' and not isinstance(value, bool):
                return False
            elif param.type in ['int', 'float']:
                if param.min_value is not None and value < param.min_value:
                    return False
                if param.max_value is not None and value > param.max_value:
                    return False
        
        return True

@dataclass
class GeneratedStrategy:
    """Generated strategy with metadata"""
    strategy_id: str
    template_name: str
    category: str
    parameters: Dict[str, Any]
    code: str
    generation_time: datetime
    hash_signature: str
    asset_class: str
    timeframe: str
    complexity_score: int
    expected_performance: Optional[Dict[str, float]] = None
    
    @classmethod
    def create(cls, template: StrategyTemplate, parameters: Dict[str, Any], asset_class: str = "Equity", timeframe: str = "Daily") -> "GeneratedStrategy":
        """Create a new generated strategy"""
        code = template.generate_code(parameters)
        strategy_id = cls._generate_strategy_id(template.name, parameters, asset_class, timeframe)
        hash_signature = hashlib.md5(code.encode()).hexdigest()
        
        return cls(
            strategy_id=strategy_id,
            template_name=template.name,
            category=template.category,
            parameters=parameters,
            code=code,
            generation_time=datetime.now(),
            hash_signature=hash_signature,
            asset_class=asset_class,
            timeframe=timeframe,
            complexity_score=template.complexity_score
        )
    
    @staticmethod
    def _generate_strategy_id(template_name: str, parameters: Dict[str, Any], asset_class: str, timeframe: str) -> str:
        """Generate unique strategy ID"""
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(parameters.items()))
        full_str = f"{template_name}_{asset_class}_{timeframe}_{param_str}"
        return hashlib.sha256(full_str.encode()).hexdigest()[:16]

class StrategyTemplateLibrary:
    """Library of strategy templates for different trading approaches"""
    
    def __init__(self):
        self.templates: Dict[str, StrategyTemplate] = {}
        self.logger = get_logger(__name__)
        self._initialize_templates()
    
    def _initialize_templates(self) -> None:
        """Initialize built-in strategy templates"""
        # Advanced Pattern Recognition Strategy
        self.templates["advanced_pattern_recognition"] = StrategyTemplate(
            name="advanced_pattern_recognition",
            description="Advanced Technical Analysis with Pattern Recognition",
            category="pattern_recognition",
            code_template=get_pattern_recognition_template(),
            parameters=[
                ParameterRange("symbol", "choice", choices=["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]),
                ParameterRange("pattern_confidence_threshold", "float", 0.6, 0.9, step=0.05),
                ParameterRange("wyckoff_volume_threshold", "float", 1.2, 2.0, step=0.1),
                ParameterRange("fibonacci_confluence_threshold", "int", 2, 5, step=1),
                ParameterRange("smc_enabled", "bool"),
                ParameterRange("volume_profile_enabled", "bool"),
                ParameterRange("position_size", "float", 0.1, 1.0, step=0.1),
                ParameterRange("risk_per_trade", "float", 0.01, 0.05, step=0.005)
            ],
            complexity_score=8
        )
        
        # Smart Money Concepts Strategy
        self.templates["smart_money_concepts"] = StrategyTemplate(
            name="smart_money_concepts",
            description="Smart Money Concepts Trading Strategy",
            category="smart_money",
            code_template=get_smc_template(),
            parameters=[
                ParameterRange("symbol", "choice", choices=["SPY", "QQQ", "IWM", "XLF", "XLE", "XLI", "XLK"]),
                ParameterRange("swing_detection_period", "int", 3, 10, step=1),
                ParameterRange("fvg_min_gap_size", "float", 0.001, 0.005, step=0.0005),
                ParameterRange("ob_lookback_period", "int", 5, 15, step=2),
                ParameterRange("bos_confirmation_buffer", "float", 0.001, 0.003, step=0.0005),
                ParameterRange("position_size", "float", 0.2, 1.0, step=0.1)
            ],
            complexity_score=7
        )
        
        # Moving Average Crossover
        self.templates["ma_crossover"] = StrategyTemplate(
            name="ma_crossover",
            description="Moving Average Crossover Strategy",
            category="trend_following",
            code_template=self._get_ma_crossover_template(),
            parameters=[
                ParameterRange("fast_period", "int", 5, 50, step=5),
                ParameterRange("slow_period", "int", 20, 200, step=10),
                ParameterRange("symbol", "choice", choices=["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]),
                ParameterRange("stop_loss", "float", 0.02, 0.10, step=0.01),
                ParameterRange("take_profit", "float", 0.05, 0.20, step=0.01)
            ]
        )
        
        # RSI Mean Reversion
        self.templates["rsi_mean_reversion"] = StrategyTemplate(
            name="rsi_mean_reversion",
            description="RSI-based Mean Reversion Strategy",
            category="mean_reversion",
            code_template=self._get_rsi_template(),
            parameters=[
                ParameterRange("rsi_period", "int", 10, 30, step=2),
                ParameterRange("oversold_threshold", "int", 20, 35, step=5),
                ParameterRange("overbought_threshold", "int", 65, 80, step=5),
                ParameterRange("symbol", "choice", choices=["SPY", "QQQ", "IWM", "XLF", "XLE", "XLI", "XLK"]),
                ParameterRange("position_size", "float", 0.1, 1.0, step=0.1),
                ParameterRange("holding_period", "int", 1, 10, step=1)
            ]
        )
        
        # Bollinger Bands
        self.templates["bollinger_bands"] = StrategyTemplate(
            name="bollinger_bands",
            description="Bollinger Bands Mean Reversion",
            category="mean_reversion",
            code_template=self._get_bollinger_template(),
            parameters=[
                ParameterRange("period", "int", 15, 30, step=5),
                ParameterRange("std_dev", "float", 1.5, 2.5, step=0.25),
                ParameterRange("symbol", "choice", choices=["SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "VXX"]),
                ParameterRange("reversion_threshold", "float", 0.8, 1.0, step=0.05),
                ParameterRange("exit_threshold", "float", 0.3, 0.7, step=0.1)
            ]
        )
        
        # Momentum Strategy
        self.templates["momentum"] = StrategyTemplate(
            name="momentum",
            description="Price Momentum Strategy",
            category="momentum",
            code_template=self._get_momentum_template(),
            parameters=[
                ParameterRange("lookback_period", "int", 20, 100, step=10),
                ParameterRange("holding_period", "int", 5, 30, step=5),
                ParameterRange("top_quantile", "float", 0.1, 0.3, step=0.05),
                ParameterRange("universe_size", "int", 50, 200, step=50),
                ParameterRange("rebalance_frequency", "choice", choices=["Weekly", "Monthly", "Quarterly"]),
                ParameterRange("sector_neutral", "bool")
            ],
            complexity_score=3
        )
        
        # Pairs Trading
        self.templates["pairs_trading"] = StrategyTemplate(
            name="pairs_trading",
            description="Statistical Arbitrage Pairs Trading",
            category="arbitrage",
            code_template=self._get_pairs_template(),
            parameters=[
                ParameterRange("pair_symbol1", "choice", choices=["SPY", "QQQ", "IWM", "XLF", "XLE"]),
                ParameterRange("pair_symbol2", "choice", choices=["EFA", "EEM", "VGK", "EWJ", "FXI"]),
                ParameterRange("lookback_period", "int", 60, 252, step=30),
                ParameterRange("entry_threshold", "float", 1.5, 3.0, step=0.25),
                ParameterRange("exit_threshold", "float", 0.0, 1.0, step=0.25),
                ParameterRange("stop_loss", "float", 3.0, 5.0, step=0.5)
            ],
            complexity_score=4
        )
        
        # Market Neutral Long-Short
        self.templates["long_short_equity"] = StrategyTemplate(
            name="long_short_equity",
            description="Market Neutral Long-Short Equity",
            category="market_neutral",
            code_template=self._get_long_short_template(),
            parameters=[
                ParameterRange("alpha_factor", "choice", choices=["ROE", "ROA", "PE", "PB", "EV_EBITDA"]),
                ParameterRange("universe_size", "int", 100, 500, step=100),
                ParameterRange("long_quantile", "float", 0.1, 0.3, step=0.05),
                ParameterRange("short_quantile", "float", 0.1, 0.3, step=0.05),
                ParameterRange("leverage", "float", 1.0, 2.0, step=0.2),
                ParameterRange("rebalance_frequency", "choice", choices=["Monthly", "Quarterly"])
            ],
            complexity_score=5
        )
        
        self.logger.info(f"Initialized {len(self.templates)} strategy templates")
    
    def _get_ma_crossover_template(self) -> str:
        """Moving Average Crossover template"""
        return '''
class MovingAverageCrossover(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.symbol = self.AddEquity("{symbol}", Resolution.Daily).Symbol
        self.fast_ema = self.EMA(self.symbol, {fast_period}, Resolution.Daily)
        self.slow_ema = self.EMA(self.symbol, {slow_period}, Resolution.Daily)
        
        self.stop_loss_pct = {stop_loss}
        self.take_profit_pct = {take_profit}
        
    def OnData(self, data):
        if not (self.fast_ema.IsReady and self.slow_ema.IsReady):
            return
            
        if not self.Portfolio[self.symbol].Invested:
            if self.fast_ema.Current.Value > self.slow_ema.Current.Value:
                self.SetHoldings(self.symbol, 1.0)
                self.entry_price = data[self.symbol].Close
                
        elif self.Portfolio[self.symbol].IsLong:
            current_price = data[self.symbol].Close
            
            # Stop loss
            if current_price < self.entry_price * (1 - self.stop_loss_pct):
                self.Liquidate(self.symbol)
                
            # Take profit
            elif current_price > self.entry_price * (1 + self.take_profit_pct):
                self.Liquidate(self.symbol)
                
            # Exit signal
            elif self.fast_ema.Current.Value < self.slow_ema.Current.Value:
                self.Liquidate(self.symbol)
'''
    
    def _get_rsi_template(self) -> str:
        """RSI Mean Reversion template"""
        return '''
class RSIMeanReversion(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.symbol = self.AddEquity("{symbol}", Resolution.Daily).Symbol
        self.rsi = self.RSI(self.symbol, {rsi_period}, Resolution.Daily)
        
        self.oversold = {oversold_threshold}
        self.overbought = {overbought_threshold}
        self.position_size = {position_size}
        self.holding_period = {holding_period}
        self.entry_time = None
        
    def OnData(self, data):
        if not self.rsi.IsReady:
            return
            
        rsi_value = self.rsi.Current.Value
        
        if not self.Portfolio[self.symbol].Invested:
            if rsi_value < self.oversold:
                self.SetHoldings(self.symbol, self.position_size)
                self.entry_time = self.Time
            elif rsi_value > self.overbought:
                self.SetHoldings(self.symbol, -self.position_size)
                self.entry_time = self.Time
                
        else:
            # Exit after holding period
            if self.entry_time and (self.Time - self.entry_time).days >= self.holding_period:
                self.Liquidate(self.symbol)
                self.entry_time = None
                
            # Exit on mean reversion
            elif (self.Portfolio[self.symbol].IsLong and rsi_value > 50) or \
                 (self.Portfolio[self.symbol].IsShort and rsi_value < 50):
                self.Liquidate(self.symbol)
                self.entry_time = None
'''
    
    def _get_bollinger_template(self) -> str:
        """Bollinger Bands template"""
        return '''
class BollingerBands(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.symbol = self.AddEquity("{symbol}", Resolution.Daily).Symbol
        self.bb = self.BB(self.symbol, {period}, {std_dev}, Resolution.Daily)
        
        self.reversion_threshold = {reversion_threshold}
        self.exit_threshold = {exit_threshold}
        
    def OnData(self, data):
        if not self.bb.IsReady:
            return
            
        price = data[self.symbol].Close
        upper_band = self.bb.UpperBand.Current.Value
        lower_band = self.bb.LowerBand.Current.Value
        middle_band = self.bb.MiddleBand.Current.Value
        
        band_width = upper_band - lower_band
        
        if not self.Portfolio[self.symbol].Invested:
            # Buy when price touches lower band
            if price <= lower_band + band_width * (1 - self.reversion_threshold):
                self.SetHoldings(self.symbol, 1.0)
                
            # Sell when price touches upper band
            elif price >= upper_band - band_width * (1 - self.reversion_threshold):
                self.SetHoldings(self.symbol, -1.0)
                
        else:
            # Exit when price moves toward middle
            price_from_middle = abs(price - middle_band) / (band_width / 2)
            
            if price_from_middle < self.exit_threshold:
                self.Liquidate(self.symbol)
'''
    
    def _get_momentum_template(self) -> str:
        """Momentum Strategy template"""
        return '''
class MomentumStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.lookback_period = {lookback_period}
        self.holding_period = {holding_period}
        self.top_quantile = {top_quantile}
        self.universe_size = {universe_size}
        self.sector_neutral = {sector_neutral}
        
        self.symbols = []
        self.rebalance_time = None
        
        # Add universe
        self.AddUniverse(self.CoarseSelectionFunction)
        
    def CoarseSelectionFunction(self, coarse):
        # Select liquid stocks
        filtered = [x for x in coarse if x.HasFundamentalData and x.Price > 5]
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:self.universe_size]]
        
    def OnData(self, data):
        if self.rebalance_time is None or self.Time >= self.rebalance_time:
            self.Rebalance(data)
            self.rebalance_time = self.Time + timedelta(days=self.holding_period)
            
    def Rebalance(self, data):
        # Calculate momentum scores
        momentum_scores = {}
        for symbol in self.symbols:
            if symbol in data and data[symbol] is not None:
                history = self.History(symbol, self.lookback_period, Resolution.Daily)
                if not history.empty:
                    returns = (history['close'][-1] / history['close'][0]) - 1
                    momentum_scores[symbol] = returns
                    
        # Select top performers
        sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        num_positions = int(len(sorted_symbols) * self.top_quantile)
        selected_symbols = [x[0] for x in sorted_symbols[:num_positions]]
        
        # Equal weight positions
        if selected_symbols:
            weight = 1.0 / len(selected_symbols)
            for symbol in selected_symbols:
                self.SetHoldings(symbol, weight)
                
        # Liquidate non-selected positions
        for symbol in self.Portfolio.Keys:
            if symbol not in selected_symbols:
                self.Liquidate(symbol)
'''
    
    def _get_pairs_template(self) -> str:
        """Pairs Trading template"""
        return '''
class PairsTrading(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.symbol1 = self.AddEquity("{pair_symbol1}", Resolution.Daily).Symbol
        self.symbol2 = self.AddEquity("{pair_symbol2}", Resolution.Daily).Symbol
        
        self.lookback_period = {lookback_period}
        self.entry_threshold = {entry_threshold}
        self.exit_threshold = {exit_threshold}
        self.stop_loss = {stop_loss}
        
        self.spread_history = []
        
    def OnData(self, data):
        if not (data.ContainsKey(self.symbol1) and data.ContainsKey(self.symbol2)):
            return
            
        price1 = data[self.symbol1].Close
        price2 = data[self.symbol2].Close
        
        # Calculate spread
        spread = price1 - price2
        self.spread_history.append(spread)
        
        if len(self.spread_history) < self.lookback_period:
            return
            
        # Keep only recent history
        self.spread_history = self.spread_history[-self.lookback_period:]
        
        # Calculate z-score
        mean_spread = sum(self.spread_history) / len(self.spread_history)
        std_spread = (sum([(x - mean_spread)**2 for x in self.spread_history]) / len(self.spread_history))**0.5
        
        if std_spread == 0:
            return
            
        z_score = (spread - mean_spread) / std_spread
        
        if not self.Portfolio[self.symbol1].Invested:
            if z_score > self.entry_threshold:
                # Spread too high, short symbol1, long symbol2
                self.SetHoldings(self.symbol1, -0.5)
                self.SetHoldings(self.symbol2, 0.5)
                
            elif z_score < -self.entry_threshold:
                # Spread too low, long symbol1, short symbol2
                self.SetHoldings(self.symbol1, 0.5)
                self.SetHoldings(self.symbol2, -0.5)
                
        else:
            # Exit conditions
            if abs(z_score) < self.exit_threshold or abs(z_score) > self.stop_loss:
                self.Liquidate()
'''
    
    def _get_long_short_template(self) -> str:
        """Long-Short Equity template"""
        return '''
class LongShortEquity(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.alpha_factor = "{alpha_factor}"
        self.universe_size = {universe_size}
        self.long_quantile = {long_quantile}
        self.short_quantile = {short_quantile}
        self.leverage = {leverage}
        
        self.rebalance_frequency = "{rebalance_frequency}"
        self.last_rebalance = None
        
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        
    def CoarseSelectionFunction(self, coarse):
        filtered = [x for x in coarse if x.HasFundamentalData and x.Price > 5]
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:self.universe_size]]
        
    def FineSelectionFunction(self, fine):
        return [f.Symbol for f in fine if f.ValuationRatios.PERatio > 0]
        
    def OnData(self, data):
        if self.ShouldRebalance():
            self.Rebalance()
            
    def ShouldRebalance(self):
        if self.last_rebalance is None:
            return True
            
        if self.rebalance_frequency == "Monthly":
            return (self.Time - self.last_rebalance).days >= 30
        elif self.rebalance_frequency == "Quarterly":
            return (self.Time - self.last_rebalance).days >= 90
            
        return False
        
    def Rebalance(self):
        # Get fundamental data and calculate alpha scores
        fine_data = self.Universe.SelectMany(lambda x: x.Fine)
        alpha_scores = {}
        
        for fine in fine_data:
            if hasattr(fine.ValuationRatios, self.alpha_factor):
                score = getattr(fine.ValuationRatios, self.alpha_factor)
                if score and score > 0:
                    alpha_scores[fine.Symbol] = 1.0 / score  # Lower ratios = higher scores
                    
        # Sort by alpha score
        sorted_symbols = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)
        
        num_long = int(len(sorted_symbols) * self.long_quantile)
        num_short = int(len(sorted_symbols) * self.short_quantile)
        
        long_symbols = [x[0] for x in sorted_symbols[:num_long]]
        short_symbols = [x[0] for x in sorted_symbols[-num_short:]]
        
        # Calculate position sizes
        if long_symbols and short_symbols:
            long_weight = self.leverage / (2 * len(long_symbols))
            short_weight = -self.leverage / (2 * len(short_symbols))
            
            # Set long positions
            for symbol in long_symbols:
                self.SetHoldings(symbol, long_weight)
                
            # Set short positions
            for symbol in short_symbols:
                self.SetHoldings(symbol, short_weight)
                
        # Liquidate other positions
        for symbol in self.Portfolio.Keys:
            if symbol not in long_symbols and symbol not in short_symbols:
                self.Liquidate(symbol)
                
        self.last_rebalance = self.Time
'''
    
    def get_template(self, name: str) -> Optional[StrategyTemplate]:
        """Get template by name"""
        return self.templates.get(name)
    
    def get_templates_by_category(self, category: str) -> List[StrategyTemplate]:
        """Get all templates in a category"""
        return [t for t in self.templates.values() if t.category == category]
    
    def get_all_templates(self) -> List[StrategyTemplate]:
        """Get all available templates"""
        return list(self.templates.values())
    
    def add_template(self, template: StrategyTemplate) -> None:
        """Add a new template to the library"""
        self.templates[template.name] = template
        self.logger.info(f"Added template: {template.name}")

class StrategyGenerator:
    """
    High-performance strategy generator capable of 100+ strategies per hour
    
    Features:
    - Template-based generation for speed and reliability
    - Parameter space exploration with constraints
    - Parallel generation across multiple templates
    - Strategy deduplication and validation
    - Performance estimation and filtering
    """
    
    def __init__(self, quantconnect_client: QuantConnectClient, config: Any):
        self.qc_client = quantconnect_client
        self.config = config
        self.logger = get_logger(__name__)
        
        # Strategy library
        self.template_library = StrategyTemplateLibrary()
        
        # Memory system for learning
        self.memory = get_strategy_memory()
        
        # Generation tracking
        self.generated_strategies: Dict[str, GeneratedStrategy] = {}
        self.generation_stats = {
            "total_generated": 0,
            "unique_strategies": 0,
            "duplicates_filtered": 0,
            "invalid_strategies": 0,
            "generation_rate_per_hour": 0.0,
            "memory_guided": 0,
            "successful_patterns_used": 0
        }
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.last_generation_time = 0.0
        
        # Filtering and constraints
        self.max_complexity_score = 5
        self.min_expected_sharpe = 0.5
        self.preferred_asset_classes = ["Equity", "ETF"]
        self.preferred_timeframes = ["Daily", "Hour"]
        
    async def initialize(self) -> None:
        """Initialize the strategy generator"""
        self.start_time = datetime.now()
        self.logger.info(f"Strategy generator initialized with {len(self.template_library.templates)} templates")
    
    async def generate_strategies_batch(
        self, 
        count: int, 
        categories: Optional[List[str]] = None,
        max_concurrent: int = 10
    ) -> List[GeneratedStrategy]:
        """Generate a batch of strategies concurrently"""
        start_time = time.time()
        
        # Determine templates to use
        if categories:
            templates = []
            for category in categories:
                templates.extend(self.template_library.get_templates_by_category(category))
        else:
            templates = self.template_library.get_all_templates()
        
        if not templates:
            return []
        
        # Create generation tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        
        for i in range(count):
            template = random.choice(templates)
            asset_class = random.choice(self.preferred_asset_classes)
            timeframe = random.choice(self.preferred_timeframes)
            
            task = self._generate_single_strategy(semaphore, template, asset_class, timeframe)
            tasks.append(task)
        
        # Execute generation tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        strategies = []
        for result in results:
            if isinstance(result, GeneratedStrategy):
                strategies.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Strategy generation failed: {str(result)}")
        
        # Update statistics
        generation_time = time.time() - start_time
        self.last_generation_time = generation_time
        
        strategies_per_second = len(strategies) / generation_time if generation_time > 0 else 0
        strategies_per_hour = strategies_per_second * 3600
        
        self.generation_stats.update({
            "total_generated": self.generation_stats["total_generated"] + len(strategies),
            "generation_rate_per_hour": strategies_per_hour
        })
        
        self.logger.info(
            f"Generated {len(strategies)} strategies in {generation_time:.2f}s "
            f"({strategies_per_hour:.1f} strategies/hour)"
        )
        
        # Log performance metrics
        PERF_LOGGER.log_system_performance({
            "strategies_generated": len(strategies),
            "generation_time_seconds": generation_time,
            "generation_rate_per_hour": strategies_per_hour,
            "unique_strategies": len(set(s.hash_signature for s in strategies))
        })
        
        return strategies
    
    async def _generate_single_strategy(
        self, 
        semaphore: asyncio.Semaphore, 
        template: StrategyTemplate, 
        asset_class: str, 
        timeframe: str
    ) -> GeneratedStrategy:
        """Generate a single strategy with concurrency control"""
        async with semaphore:
            return await self._create_strategy_from_template(template, asset_class, timeframe)
    
    async def _create_strategy_from_template(
        self, 
        template: StrategyTemplate, 
        asset_class: str, 
        timeframe: str
    ) -> GeneratedStrategy:
        """Create a strategy from a template"""
        try:
            # Generate random parameters
            parameters = template.generate_random_parameters()
            
            # Validate parameters
            if not template.validate_parameters(parameters):
                raise ValueError(f"Invalid parameters generated for template {template.name}")
            
            # Create strategy
            strategy = GeneratedStrategy.create(template, parameters, asset_class, timeframe)
            
            # Log strategy generation start to dashboard
            DASHBOARD.log_strategy_generation_start(
                strategy_id=strategy.strategy_id,
                template_name=template.name,
                agent_name="strategy_generator",
                parameters=parameters
            )
            
            # Check for duplicates
            if strategy.hash_signature in [s.hash_signature for s in self.generated_strategies.values()]:
                self.generation_stats["duplicates_filtered"] += 1
                # Log completion with duplicate status
                DASHBOARD.log_strategy_generation_complete(
                    strategy_id=strategy.strategy_id,
                    code_length=len(strategy.code),
                    complexity_score=strategy.complexity_score,
                    validation_status="duplicate"
                )
                raise ValueError("Duplicate strategy detected")
            
            # Validate strategy code
            if not self._validate_strategy_code(strategy.code):
                self.generation_stats["invalid_strategies"] += 1
                # Log completion with invalid status
                DASHBOARD.log_strategy_generation_complete(
                    strategy_id=strategy.strategy_id,
                    code_length=len(strategy.code),
                    complexity_score=strategy.complexity_score,
                    validation_status="invalid"
                )
                raise ValueError("Invalid strategy code generated")
            
            # Store strategy
            self.generated_strategies[strategy.strategy_id] = strategy
            self.generation_stats["unique_strategies"] += 1
            
            # Log successful completion to dashboard
            DASHBOARD.log_strategy_generation_complete(
                strategy_id=strategy.strategy_id,
                code_length=len(strategy.code),
                complexity_score=strategy.complexity_score,
                validation_status="valid"
            )
            
            return strategy
            
        except Exception as e:
            self.logger.debug(f"Strategy generation failed for {template.name}: {str(e)}")
            raise
    
    def _validate_strategy_code(self, code: str) -> bool:
        """Validate generated strategy code"""
        try:
            # Basic syntax check
            ast.parse(code)
            
            # Check for required elements
            required_patterns = [
                r'class\s+\w+\(QCAlgorithm\)',  # Class definition
                r'def\s+Initialize\(',           # Initialize method
                r'def\s+OnData\(',              # OnData method
                r'SetStartDate\(',              # Start date
                r'SetEndDate\(',                # End date
                r'SetCash\('                    # Initial cash
            ]
            
            for pattern in required_patterns:
                if not re.search(pattern, code):
                    return False
            
            return True
            
        except SyntaxError:
            return False
    
    async def generate_strategies_for_category(
        self, 
        category: str, 
        count: int
    ) -> List[GeneratedStrategy]:
        """Generate strategies for a specific category"""
        return await self.generate_strategies_batch(count, categories=[category])
    
    async def generate_optimized_strategies(
        self, 
        template_name: str, 
        optimization_target: str = "sharpe",
        iterations: int = 50
    ) -> List[GeneratedStrategy]:
        """Generate optimized strategies using parameter space exploration"""
        template = self.template_library.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        strategies = []
        
        # Generate multiple variations
        for _ in range(iterations):
            # Use different parameter generation strategies
            if random.random() < 0.5:
                # Random parameters
                parameters = template.generate_random_parameters()
            else:
                # Grid search variation
                parameters = self._generate_grid_parameters(template)
            
            try:
                strategy = await self._create_strategy_from_template(
                    template, 
                    random.choice(self.preferred_asset_classes),
                    random.choice(self.preferred_timeframes)
                )
                strategies.append(strategy)
            except Exception:
                continue
        
        return strategies
    
    def _generate_grid_parameters(self, template: StrategyTemplate) -> Dict[str, Any]:
        """Generate parameters using grid search approach"""
        parameters = {}
        
        for param in template.parameters:
            if param.type == 'int' and param.step:
                # Use grid points
                values = list(range(int(param.min_value), int(param.max_value) + 1, int(param.step)))
                parameters[param.name] = random.choice(values)
            elif param.type == 'float' and param.step:
                # Use grid points
                num_steps = int((param.max_value - param.min_value) / param.step)
                step_num = random.randint(0, num_steps)
                parameters[param.name] = param.min_value + step_num * param.step
            else:
                # Use random generation
                parameters[param.name] = param.generate_value()
        
        return parameters
    
    async def generate_memory_guided_strategies(
        self, 
        count: int, 
        market_regime: str = "bull",
        min_similarity_threshold: float = 0.6
    ) -> List[GeneratedStrategy]:
        """Generate strategies guided by successful patterns in memory"""
        self.logger.info(f"Generating {count} memory-guided strategies for {market_regime} market")
        
        strategies = []
        memory_recommendations = self.memory.get_market_recommendations(market_regime)
        
        # Use memory recommendations if available
        if memory_recommendations.get('confidence_level', 0) >= 0.5:
            recommended_types = memory_recommendations.get('recommended_strategy_types', [])
            optimal_params = memory_recommendations.get('optimal_parameters', {})
            
            self.logger.info(f"Using memory recommendations with {memory_recommendations['confidence_level']:.2f} confidence")
            
            for i in range(count):
                # Choose strategy type based on recommendations
                if recommended_types:
                    rec = random.choice(recommended_types)
                    strategy_type = rec['strategy_type']
                    expected_performance = rec.get('expected_performance', 0.5)
                else:
                    # Fallback to random selection
                    strategy_type = random.choice(list(self.template_library.templates.keys()))
                    expected_performance = 0.5
                
                template = self.template_library.get_template(strategy_type)
                if not template:
                    continue
                
                # Generate parameters using memory guidance
                parameters = self._generate_memory_guided_parameters(template, optimal_params)
                
                # Create strategy
                strategy = GeneratedStrategy.create(template, parameters, "Equity", "Daily")
                
                # Find similar successful strategies for validation
                similar_strategies = self.memory.find_similar_successful_strategies(
                    strategy.code, parameters, k=3, min_performance=min_similarity_threshold
                )
                
                if similar_strategies:
                    # Adjust confidence based on similarity to successful strategies
                    avg_similarity = np.mean([s['similarity_score'] for s in similar_strategies])
                    avg_performance = np.mean([s['performance']['success_score'] for s in similar_strategies])
                    
                    strategy.estimated_performance = avg_performance * avg_similarity
                    strategy.metadata['memory_guided'] = True
                    strategy.metadata['similar_count'] = len(similar_strategies)
                    strategy.metadata['avg_similarity'] = avg_similarity
                    strategy.metadata['expected_performance'] = expected_performance
                    
                    self.generation_stats["memory_guided"] += 1
                    self.generation_stats["successful_patterns_used"] += len(similar_strategies)
                else:
                    strategy.metadata['memory_guided'] = False
                
                strategies.append(strategy)
                self.generated_strategies[strategy.strategy_id] = strategy
                
                # Log to dashboard
                DASHBOARD.log_strategy_generation_complete(
                    strategy_id=strategy.strategy_id,
                    template_name=template.name,
                    agent_name="memory_guided_generator",
                    success=True,
                    estimated_performance=strategy.estimated_performance,
                    metadata=strategy.metadata
                )
        
        else:
            # Fallback to regular generation if no memory available
            self.logger.info("Insufficient memory data, using regular generation")
            strategies = await self.generate_strategies_batch(count)
        
        self.generation_stats["total_generated"] += len(strategies)
        self.generation_stats["unique_strategies"] += len(strategies)
        
        return strategies
    
    def _generate_memory_guided_parameters(self, template: 'StrategyTemplate', optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters guided by memory insights"""
        parameters = {}
        
        for param in template.parameters:
            param_name = param.name
            
            # Check if we have optimal guidance for this parameter
            if param_name in optimal_params:
                optimal_info = optimal_params[param_name]
                recommended_value = optimal_info['recommended_value']
                confidence = optimal_info['confidence']
                
                # Use recommended value with some variation based on confidence
                if param.type == 'int':
                    # Add variation inversely proportional to confidence
                    variation = int((1.0 - confidence) * (param.max_value - param.min_value) * 0.1)
                    variation = max(1, variation)
                    
                    min_val = max(param.min_value, int(recommended_value) - variation)
                    max_val = min(param.max_value, int(recommended_value) + variation)
                    parameters[param_name] = random.randint(min_val, max_val)
                    
                elif param.type == 'float':
                    # Add variation inversely proportional to confidence
                    variation = (1.0 - confidence) * (param.max_value - param.min_value) * 0.1
                    
                    min_val = max(param.min_value, recommended_value - variation)
                    max_val = min(param.max_value, recommended_value + variation)
                    parameters[param_name] = random.uniform(min_val, max_val)
                    
                else:
                    # For non-numeric parameters, use default generation
                    parameters[param_name] = param.generate_value()
            else:
                # No memory guidance, use regular generation
                parameters[param_name] = param.generate_value()
        
        return parameters
    
    async def learn_from_backtest_result(self, strategy: 'GeneratedStrategy', backtest_result: Dict[str, Any], market_data: Dict[str, Any]):
        """Learn from backtest results and update memory"""
        try:
            # Extract strategy parameters including template info
            strategy_params = {
                'template_name': strategy.template_name,
                'category': strategy.category,
                'asset_class': strategy.asset_class,
                'timeframe': strategy.timeframe,
                **strategy.parameters
            }
            
            # Store in memory for learning
            strategy_id = self.memory.remember_strategy(
                strategy_code=strategy.code,
                strategy_params=strategy_params,
                performance_result=backtest_result,
                market_data=market_data
            )
            
            self.logger.info(f"Learned from strategy {strategy.strategy_id} -> memory ID {strategy_id}")
            
            # Update strategy metadata with memory info
            strategy.metadata['memory_id'] = strategy_id
            strategy.metadata['learned'] = True
            
        except Exception as e:
            self.logger.error(f"Error learning from backtest result: {str(e)}")
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights from the strategy memory system"""
        try:
            memory_stats = self.memory.get_memory_stats()
            
            # Add generator-specific insights
            insights = {
                'memory_stats': memory_stats,
                'generation_stats': self.get_generation_stats(),
                'memory_utilization': {
                    'strategies_in_memory': memory_stats['total_strategies'],
                    'success_rate': memory_stats['success_rate'],
                    'learned_patterns': memory_stats['learned_patterns'],
                    'memory_guided_generations': self.generation_stats.get('memory_guided', 0),
                    'successful_patterns_used': self.generation_stats.get('successful_patterns_used', 0)
                }
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting memory insights: {str(e)}")
            return {'error': str(e)}
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get strategy generation statistics"""
        if self.start_time:
            runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            avg_rate = (
                self.generation_stats["total_generated"] / runtime_hours 
                if runtime_hours > 0 else 0
            )
        else:
            avg_rate = 0
        
        return {
            **self.generation_stats,
            "average_rate_per_hour": avg_rate,
            "runtime_hours": runtime_hours if self.start_time else 0,
            "last_generation_time": self.last_generation_time,
            "templates_available": len(self.template_library.templates),
            "strategies_in_memory": len(self.generated_strategies)
        }
    
    def get_strategy(self, strategy_id: str) -> Optional[GeneratedStrategy]:
        """Get strategy by ID"""
        return self.generated_strategies.get(strategy_id)
    
    def get_strategies_by_category(self, category: str) -> List[GeneratedStrategy]:
        """Get all generated strategies in a category"""
        return [s for s in self.generated_strategies.values() if s.category == category]
    
    def clear_strategies(self) -> None:
        """Clear all generated strategies"""
        self.generated_strategies.clear()
        self.logger.info("Cleared all generated strategies")