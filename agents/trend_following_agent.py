"""
Trend Following Agent - Specializes in trend-based trading strategies
"""

import asyncio
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent, AgentConfig
from agents.knowledge_base import MarketInsight, SharedKnowledgeBase
from tier2_strategy.strategy_generator import StrategyGenerator, GeneratedStrategy

class TrendFollowingAgent(BaseAgent):
    """
    Specialized agent for trend following strategies
    
    Focuses on:
    - Moving average crossovers
    - Momentum breakouts
    - Trend strength indicators
    - Directional movement strategies
    """
    
    def __init__(
        self,
        strategy_generator: StrategyGenerator,
        strategy_tester: Any,
        knowledge_base: SharedKnowledgeBase
    ):
        config = AgentConfig(
            name="trend_following",
            category="trend_following",
            max_concurrent_tasks=5,
            generation_batch_size=15,
            min_sharpe_threshold=0.6,
            min_cagr_threshold=0.08,
            risk_tolerance=0.7,  # Higher risk tolerance for trends
            exploration_rate=0.25
        )
        
        super().__init__(config, strategy_generator, strategy_tester, knowledge_base)
        
        # Trend-specific parameters
        self.preferred_templates = ["ma_crossover", "momentum"]
        self.trend_detection_threshold = 0.65
        self.minimum_trend_duration = 20  # days
        
    async def _initialize_agent(self) -> None:
        """Initialize trend following agent"""
        self.logger.info("Trend Following Agent initialized")
        
        # Share initial market insight
        insight = MarketInsight(
            insight_id=f"{self.config.name}_init_{datetime.now().timestamp()}",
            agent_name=self.config.name,
            category="trend",
            asset_class="Equity",
            symbols=["SPY", "QQQ", "IWM"],
            timeframe="Daily",
            description="Trend following agent online, monitoring for sustained directional movements",
            confidence=0.8,
            validity_period=timedelta(hours=24),
            supporting_data={"agent_type": "trend_following"},
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.add_market_insight(insight)
    
    async def _generate_strategies(self, insights: List[MarketInsight]) -> List[GeneratedStrategy]:
        """Generate trend following strategies based on market insights"""
        strategies = []
        
        # Filter insights for trend signals
        trend_insights = [
            i for i in insights 
            if i.category in ["trend", "momentum"] and i.confidence >= self.trend_detection_threshold
        ]
        
        # Determine generation approach based on insights
        if trend_insights:
            # Strong trend detected - generate more aggressive strategies
            self.logger.info(f"Strong trends detected in {len(trend_insights)} insights")
            
            # Generate trend-following strategies
            batch_size = min(self.config.generation_batch_size * 2, 30)  # Double generation in strong trends
            
            new_strategies = await self.generator.generate_strategies_batch(
                count=batch_size,
                categories=["trend_following", "momentum"]
            )
            
            strategies.extend(new_strategies)
            
            # Adjust parameters for strong trends
            for strategy in strategies[-5:]:  # Enhance last 5 strategies
                await self._enhance_strategy_for_trends(strategy, trend_insights)
        
        else:
            # No strong trends - generate exploratory strategies
            self.logger.debug("No strong trends detected, generating exploratory strategies")
            
            new_strategies = await self.generator.generate_strategies_batch(
                count=self.config.generation_batch_size,
                categories=["trend_following"]
            )
            
            strategies.extend(new_strategies)
        
        # Update state
        self.state.strategies_generated += len(strategies)
        
        # Share trend discovery if significant
        if len(strategies) > 20:
            await self._share_trend_discovery(strategies, trend_insights)
        
        return strategies
    
    async def _enhance_strategy_for_trends(self, strategy: GeneratedStrategy, insights: List[MarketInsight]) -> None:
        """Enhance strategy parameters based on trend insights"""
        # This would modify strategy parameters based on detected trends
        # For example, extending holding periods in strong trends
        pass
    
    async def _share_trend_discovery(self, strategies: List[GeneratedStrategy], insights: List[MarketInsight]) -> None:
        """Share trend discovery with other agents"""
        # Aggregate trend information
        trending_symbols = set()
        for insight in insights:
            trending_symbols.update(insight.symbols)
        
        # Create market insight
        discovery_insight = MarketInsight(
            insight_id=f"{self.config.name}_trend_{datetime.now().timestamp()}",
            agent_name=self.config.name,
            category="trend",
            asset_class="Equity",
            symbols=list(trending_symbols)[:10],  # Top 10 symbols
            timeframe="Daily",
            description=f"Strong trending conditions detected across {len(trending_symbols)} symbols. "
                       f"Generated {len(strategies)} trend-following strategies.",
            confidence=0.75,
            validity_period=timedelta(hours=48),
            supporting_data={
                "strategy_count": len(strategies),
                "trend_strength": "strong",
                "recommended_approach": "trend_following"
            },
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.add_market_insight(discovery_insight)
    
    async def _handle_message(self, message: Any) -> None:
        """Handle messages with trend-specific logic"""
        await super()._handle_message(message)
        
        # Handle trend-specific messages
        if message.message_type == 'notification' and 'market_regime' in message.content:
            new_regime = message.content['market_regime']
            
            if new_regime == 'trending':
                # Increase generation rate in trending markets
                self.config.generation_batch_size = min(self.config.generation_batch_size * 2, 30)
                self.logger.info("Increased generation rate for trending market")
            else:
                # Reset to normal
                self.config.generation_batch_size = 15