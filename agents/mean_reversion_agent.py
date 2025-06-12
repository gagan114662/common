"""
Mean Reversion Agent - Specializes in mean reversion trading strategies
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent, AgentConfig
from agents.knowledge_base import MarketInsight, SharedKnowledgeBase
from tier2_strategy.strategy_generator import StrategyGenerator, GeneratedStrategy

class MeanReversionAgent(BaseAgent):
    """
    Specialized agent for mean reversion strategies
    
    Focuses on:
    - RSI oversold/overbought conditions
    - Bollinger Bands reversions
    - Statistical arbitrage
    - Range-bound trading
    """
    
    def __init__(
        self,
        strategy_generator: StrategyGenerator,
        strategy_tester: Any,
        knowledge_base: SharedKnowledgeBase
    ):
        config = AgentConfig(
            name="mean_reversion",
            category="mean_reversion",
            max_concurrent_tasks=4,
            generation_batch_size=12,
            min_sharpe_threshold=0.7,
            min_cagr_threshold=0.06,
            risk_tolerance=0.4,  # Lower risk for mean reversion
            exploration_rate=0.15
        )
        
        super().__init__(config, strategy_generator, strategy_tester, knowledge_base)
        
        # Mean reversion specific parameters
        self.preferred_templates = ["rsi_mean_reversion", "bollinger_bands", "pairs_trading"]
        self.range_detection_threshold = 0.70
        self.minimum_range_duration = 30  # days
        
    async def _initialize_agent(self) -> None:
        """Initialize mean reversion agent"""
        self.logger.info("Mean Reversion Agent initialized")
        
        # Share initial insight
        insight = MarketInsight(
            insight_id=f"{self.config.name}_init_{datetime.now().timestamp()}",
            agent_name=self.config.name,
            category="mean_reversion",
            asset_class="Equity",
            symbols=["SPY", "QQQ", "IWM", "XLF", "XLE"],
            timeframe="Daily",
            description="Mean reversion agent online, monitoring for range-bound conditions and oversold/overbought levels",
            confidence=0.8,
            validity_period=timedelta(hours=24),
            supporting_data={"agent_type": "mean_reversion"},
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.add_market_insight(insight)
    
    async def _generate_strategies(self, insights: List[MarketInsight]) -> List[GeneratedStrategy]:
        """Generate mean reversion strategies based on market insights"""
        strategies = []
        
        # Filter insights for range-bound or volatility signals
        range_insights = [
            i for i in insights 
            if i.category in ["range", "volatility", "mean_reversion"] 
            and i.confidence >= self.range_detection_threshold
        ]
        
        # Check market regime from supervisor
        market_regime = await self._get_market_regime()
        
        if range_insights or market_regime == "ranging":
            # Range-bound market detected - ideal for mean reversion
            self.logger.info(f"Range-bound conditions detected, generating mean reversion strategies")
            
            # Increase generation for favorable conditions
            batch_size = min(self.config.generation_batch_size * 2, 25)
            
            # Generate mean reversion strategies
            new_strategies = await self.generator.generate_strategies_batch(
                count=batch_size,
                categories=["mean_reversion", "arbitrage"]
            )
            
            strategies.extend(new_strategies)
            
            # Optimize for current volatility
            for strategy in strategies[-5:]:
                await self._optimize_for_volatility(strategy, range_insights)
                
        else:
            # Normal generation
            new_strategies = await self.generator.generate_strategies_batch(
                count=self.config.generation_batch_size,
                categories=["mean_reversion"]
            )
            
            strategies.extend(new_strategies)
        
        # Update state
        self.state.strategies_generated += len(strategies)
        
        # Share discovery if significant
        if len(strategies) > 15 and range_insights:
            await self._share_range_discovery(strategies, range_insights)
        
        return strategies
    
    async def _get_market_regime(self) -> str:
        """Get current market regime from knowledge base"""
        # Check for supervisor messages about market regime
        messages = await self.knowledge_base.get_agent_messages(
            self.config.name,
            since=datetime.now() - timedelta(hours=1)
        )
        
        for message in messages:
            if message.sender == "supervisor" and "market_regime" in message.content:
                return message.content["market_regime"]
        
        return "normal"
    
    async def _optimize_for_volatility(self, strategy: GeneratedStrategy, insights: List[MarketInsight]) -> None:
        """Optimize strategy parameters based on volatility insights"""
        # This would adjust parameters like thresholds and holding periods
        # based on current volatility conditions
        pass
    
    async def _share_range_discovery(self, strategies: List[GeneratedStrategy], insights: List[MarketInsight]) -> None:
        """Share range-bound market discovery"""
        # Aggregate symbols in range
        ranging_symbols = set()
        avg_volatility = 0.0
        
        for insight in insights:
            ranging_symbols.update(insight.symbols)
            if "volatility" in insight.supporting_data:
                avg_volatility += insight.supporting_data["volatility"]
        
        if len(insights) > 0:
            avg_volatility /= len(insights)
        
        # Create insight
        discovery_insight = MarketInsight(
            insight_id=f"{self.config.name}_range_{datetime.now().timestamp()}",
            agent_name=self.config.name,
            category="mean_reversion",
            asset_class="Equity",
            symbols=list(ranging_symbols)[:10],
            timeframe="Daily",
            description=f"Range-bound conditions detected across {len(ranging_symbols)} symbols. "
                       f"Average volatility: {avg_volatility:.2%}. Generated {len(strategies)} mean reversion strategies.",
            confidence=0.80,
            validity_period=timedelta(hours=48),
            supporting_data={
                "strategy_count": len(strategies),
                "market_condition": "range_bound",
                "avg_volatility": avg_volatility,
                "recommended_approach": "mean_reversion"
            },
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.add_market_insight(discovery_insight)