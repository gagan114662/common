"""
Supervisor Agent - Orchestrates the multi-agent system
Coordinates activities, allocates resources, and optimizes overall performance
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from agents.base_agent import BaseAgent, AgentConfig, AgentState
from agents.knowledge_base import SharedKnowledgeBase, MarketInsight, AgentMessage
from tier1_core.logger import get_logger
from tier1_core.performance_monitor import PerformanceMonitor
from tier2_strategy.strategy_generator import StrategyGenerator, GeneratedStrategy
from tier2_strategy.strategy_tester import StrategyTester, StrategyPerformance
from tier3_evolution.evolution_engine import EvolutionEngine
from config.settings import AgentConfigs as SystemAgentConfig, PerformanceTargets # Changed AgentConfig to AgentConfigs

@dataclass
class AgentPerformanceProfile:
    """Performance profile for an agent"""
    agent_name: str
    total_strategies: int = 0
    successful_strategies: int = 0
    average_sharpe: float = 0.0
    average_cagr: float = 0.0
    generation_rate: float = 0.0
    success_rate: float = 0.0
    specialization_score: float = 0.0
    collaboration_score: float = 0.0
    resource_efficiency: float = 0.0
    overall_score: float = 0.0

@dataclass
class ResourceAllocation:
    """Resource allocation for agents"""
    agent_name: str
    cpu_allocation: float  # 0.0 to 1.0
    memory_allocation: float  # 0.0 to 1.0
    strategy_quota: int  # Strategies per hour
    priority: int  # 1-5, 1 being highest

class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent - The orchestrator of the multi-agent system
    
    Responsibilities:
    - Coordinate activities across all agents
    - Allocate computational resources
    - Monitor overall system performance
    - Identify and resolve conflicts
    - Optimize agent collaboration
    - Ensure performance targets are met
    """
    
    def __init__(
        self,
        strategy_generator: StrategyGenerator,
        strategy_tester: StrategyTester,
        evolution_engine: EvolutionEngine,
        performance_monitor: PerformanceMonitor,
        knowledge_base: SharedKnowledgeBase,
        agent_config: SystemAgentConfig
    ):
        # Initialize supervisor config
        config = AgentConfig(
            name="supervisor",
            category="coordination",
            max_concurrent_tasks=10,
            generation_batch_size=5,
            min_sharpe_threshold=0.7,
            min_cagr_threshold=0.10,
            risk_tolerance=0.5,
            exploration_rate=0.3
        )
        
        super().__init__(config, strategy_generator, strategy_tester, knowledge_base)

        self.strategy_generator = strategy_generator # Added assignment
        self.strategy_tester = strategy_tester # Added assignment
        
        self.evolution_engine = evolution_engine
        self.performance_monitor = performance_monitor
        self.system_agent_config = agent_config
        self.performance_targets = PerformanceTargets()
        
        # Agent management
        self.managed_agents: Dict[str, BaseAgent] = {}
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        
        # System state
        self.system_start_time: Optional[datetime] = None
        self.targets_achieved = False
        self.best_system_performance = {
            'cagr': 0.0,
            'sharpe': 0.0,
            'drawdown': 1.0,
            'average_profit_per_trade': 0.0
        }
        
        # Coordination state
        self.current_market_regime = "normal"  # normal, volatile, trending, ranging
        self.agent_task_queue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.collaboration_matrix: Dict[Tuple[str, str], float] = {}  # Agent pair -> synergy score
        
        # Optimization parameters
        self.rebalance_interval = timedelta(hours=1)
        self.last_rebalance = datetime.now()
        self.performance_window = timedelta(hours=24)
        
    async def _initialize_agent(self) -> None:
        """Initialize the supervisor agent"""
        self.system_start_time = datetime.now()
        self.logger.info("Supervisor agent initializing...")
        
        # Initialize agent profiles
        # self.system_agent_config is now an instance of AgentConfigs
        # We can get agent names by iterating through its fields (e.g., trend_following, mean_reversion)

        # Get agent setting objects from AgentConfigs
        agent_setting_objects = [getattr(self.system_agent_config, field_name) for field_name in self.system_agent_config.__annotations__ if hasattr(self.system_agent_config, field_name)]
        configured_agent_settings = [setting for setting in agent_setting_objects if hasattr(setting, 'name')] # Filter out non-agent settings if any

        num_agents = len(configured_agent_settings)

        for agent_setting in configured_agent_settings:
            agent_name = agent_setting.name # Get name from the specific agent's settings
            self.agent_profiles[agent_name] = AgentPerformanceProfile(agent_name=agent_name)
            self.resource_allocations[agent_name] = ResourceAllocation(
                agent_name=agent_name,
                cpu_allocation=1.0 / num_agents if num_agents > 0 else 1.0,
                memory_allocation=1.0 / num_agents if num_agents > 0 else 1.0,
                strategy_quota=100 // num_agents if num_agents > 0 else 100,
                priority=3
            )
        
        self.logger.info(f"Supervisor initialized with {len(self.agent_profiles)} agents to manage")
    
    def add_managed_agent(self, agent: BaseAgent) -> None:
        """Add an agent to be managed by the supervisor"""
        self.managed_agents[agent.config.name] = agent
        self.logger.info(f"Added managed agent: {agent.config.name}")
    
    async def _generate_strategies(self, insights: List[MarketInsight]) -> List[GeneratedStrategy]:
        """Supervisor generates coordination strategies, not trading strategies"""
        # The supervisor doesn't generate trading strategies directly
        # Instead, it coordinates other agents to generate strategies
        return []
    
    async def _main_loop(self) -> None:
        """Main supervisor loop - orchestrate the system"""
        while not self.shutdown_event.is_set():
            try:
                # Monitor system performance
                await self._monitor_system_performance()
                
                # Analyze agent performance
                await self._analyze_agent_performance()
                
                # Rebalance resources if needed
                if datetime.now() - self.last_rebalance > self.rebalance_interval:
                    await self._rebalance_resources()
                    self.last_rebalance = datetime.now()
                
                # Coordinate agent activities
                await self._coordinate_agents()
                
                # Check for target achievement
                await self._check_performance_targets()
                
                # Identify collaboration opportunities
                await self._identify_collaborations()
                
                # Handle system-level optimizations
                await self._optimize_system()
                
                await asyncio.sleep(10)  # Supervisor checks every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in supervisor main loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def _monitor_system_performance(self) -> None:
        """Monitor overall system performance"""
        # Get current metrics from performance monitor
        metrics = self.performance_monitor.get_current_metrics()
        
        # Update best system performance
        if metrics and 'trading' in metrics:
            trading_metrics = metrics['trading']
            
            if trading_metrics.get('best_cagr', 0) > self.best_system_performance['cagr']:
                self.best_system_performance['cagr'] = trading_metrics['best_cagr']
                
            if trading_metrics.get('best_sharpe', 0) > self.best_system_performance['sharpe']:
                self.best_system_performance['sharpe'] = trading_metrics['best_sharpe']
                
            if trading_metrics.get('best_drawdown', 1) < self.best_system_performance['drawdown']:
                self.best_system_performance['drawdown'] = trading_metrics['best_drawdown']

            if trading_metrics.get('best_average_profit_per_trade', 0.0) > self.best_system_performance.get('average_profit_per_trade', 0.0):
                self.best_system_performance['average_profit_per_trade'] = trading_metrics['best_average_profit_per_trade']
        
        # Detect market regime
        await self._detect_market_regime()
        
        # Log system status
        if self.state.strategies_tested % 100 == 0:  # Log every 100 strategies
            self.logger.info(
                f"System Performance - CAGR: {self.best_system_performance['cagr']:.2%}, "
                f"Sharpe: {self.best_system_performance['sharpe']:.2f}, "
                f"Drawdown: {self.best_system_performance['drawdown']:.2%}, "
                f"AvgProfit/Trade: {self.best_system_performance.get('average_profit_per_trade', 0.0):.4f}"
            )
    
    async def _analyze_agent_performance(self) -> None:
        """Analyze individual agent performance"""
        # Get agent rankings from knowledge base
        rankings = await self.knowledge_base.get_agent_rankings()
        
        for agent_name, agent in self.managed_agents.items():
            # Get agent status
            status = agent.get_status()
            
            # Update performance profile
            profile = self.agent_profiles[agent_name]
            profile.total_strategies = status['strategies_generated']
            profile.successful_strategies = status['successful_strategies']
            profile.success_rate = status['success_rate'] / 100.0
            
            # Calculate additional metrics
            if agent_name in rankings:
                agent_metrics = rankings[agent_name]['metrics']
                profile.average_sharpe = agent_metrics.get('average_sharpe', 0.0)
                profile.average_cagr = agent_metrics.get('average_cagr', 0.0)
            
            # Calculate specialization score
            profile.specialization_score = self._calculate_specialization_score(agent)
            
            # Calculate collaboration score
            profile.collaboration_score = self._calculate_collaboration_score(agent_name)
            
            # Calculate resource efficiency
            allocation = self.resource_allocations[agent_name]
            if allocation.strategy_quota > 0:
                profile.resource_efficiency = (
                    profile.successful_strategies / 
                    (allocation.strategy_quota * 
                     (datetime.now() - self.system_start_time).total_seconds() / 3600)
                )
            
            # Calculate overall score
            profile.overall_score = (
                profile.success_rate * 30 +
                profile.average_sharpe * 30 +
                profile.specialization_score * 20 +
                profile.collaboration_score * 10 +
                profile.resource_efficiency * 10
            )
    
    async def _rebalance_resources(self) -> None:
        """Rebalance resource allocation based on agent performance"""
        self.logger.info("Rebalancing resource allocations...")
        
        # Calculate total scores
        total_score = sum(p.overall_score for p in self.agent_profiles.values())
        
        if total_score == 0:
            return
        
        # Reallocate based on performance
        for agent_name, profile in self.agent_profiles.items():
            # Calculate new allocation proportional to performance
            performance_ratio = profile.overall_score / total_score
            
            # Apply smoothing to prevent drastic changes
            current_allocation = self.resource_allocations[agent_name]
            smoothing_factor = 0.7
            
            new_cpu = (
                smoothing_factor * current_allocation.cpu_allocation +
                (1 - smoothing_factor) * performance_ratio
            )
            new_memory = (
                smoothing_factor * current_allocation.memory_allocation +
                (1 - smoothing_factor) * performance_ratio
            )
            
            # Update allocation
            current_allocation.cpu_allocation = max(0.1, min(0.5, new_cpu))  # Min 10%, Max 50%
            current_allocation.memory_allocation = max(0.1, min(0.5, new_memory))
            current_allocation.strategy_quota = int(100 * performance_ratio)
            
            # Update priority based on performance
            if profile.overall_score > 80:
                current_allocation.priority = 1
            elif profile.overall_score > 60:
                current_allocation.priority = 2
            elif profile.overall_score > 40:
                current_allocation.priority = 3
            elif profile.overall_score > 20:
                current_allocation.priority = 4
            else:
                current_allocation.priority = 5
        
        # Notify agents of new allocations
        await self._notify_resource_changes()
    
    async def _coordinate_agents(self) -> None:
        """Coordinate agent activities based on system needs"""
        # Determine system needs
        needs = await self._assess_system_needs()
        
        # Assign tasks to agents based on their specialization and performance
        for need in needs:
            best_agent = await self._select_best_agent_for_task(need)
            
            if best_agent:
                # Create task assignment
                task = {
                    'type': need['type'],
                    'parameters': need['parameters'],
                    'priority': need['priority'],
                    'deadline': datetime.now() + timedelta(hours=1)
                }
                
                # Add to agent's task queue
                self.agent_task_queue[best_agent].append(task)
                
                # Send task assignment message
                await self._send_task_assignment(best_agent, task)
    
    async def _check_performance_targets(self) -> None:
        """Check if performance targets have been achieved"""
        current_performance = self.best_system_performance
        
        targets_met = (
            current_performance['cagr'] >= self.performance_targets.target_cagr and
            current_performance['sharpe'] >= self.performance_targets.target_sharpe and
            current_performance['drawdown'] <= self.performance_targets.max_drawdown and
            current_performance.get('average_profit_per_trade', 0.0) >= self.performance_targets.target_average_profit_per_trade
        )
        
        if targets_met and not self.targets_achieved:
            self.targets_achieved = True
            self.logger.info("🎉 PERFORMANCE TARGETS ACHIEVED!")
            self.logger.info(
                f"CAGR: {current_performance['cagr']:.2%} (target: {self.performance_targets.target_cagr:.2%})"
            )
            self.logger.info(
                f"Sharpe: {current_performance['sharpe']:.2f} (target: {self.performance_targets.target_sharpe:.2f})"
            )
            self.logger.info(
                f"Drawdown: {current_performance['drawdown']:.2%} (target: {self.performance_targets.max_drawdown:.2%})"
            )
            self.logger.info(
                f"AvgProfit/Trade: {current_performance.get('average_profit_per_trade', 0.0):.4f} (target: {self.performance_targets.target_average_profit_per_trade:.4f})"
            )
            
            # Notify all agents
            await self._broadcast_achievement()
    
    async def _identify_collaborations(self) -> None:
        """Identify opportunities for agent collaboration"""
        # Analyze strategy correlations from knowledge base
        summary = self.knowledge_base.get_knowledge_summary()
        
        # Look for complementary strategies across agents
        for agent1_name, agent1 in self.managed_agents.items():
            for agent2_name, agent2 in self.managed_agents.items():
                if agent1_name >= agent2_name:  # Avoid duplicates
                    continue
                
                # Calculate synergy score
                synergy = await self._calculate_synergy_score(agent1, agent2)
                
                # Update collaboration matrix
                self.collaboration_matrix[(agent1_name, agent2_name)] = synergy
                
                # If high synergy, suggest collaboration
                if synergy > 0.7:
                    await self._suggest_collaboration(agent1_name, agent2_name)
    
    async def _optimize_system(self) -> None:
        """Perform system-level optimizations"""
        # Trigger evolution engine optimization if enough strategies
        knowledge_summary = self.knowledge_base.get_knowledge_summary()
        
        if knowledge_summary['total_discoveries'] >= 50:
            # Run evolution cycle
            self.logger.info("Triggering evolution engine optimization...")
            # This would integrate with the evolution engine
    
    async def _detect_market_regime(self) -> None:
        """Detect current market regime based on insights"""
        insights = await self.knowledge_base.get_relevant_insights(
            agent_name=self.config.name,
            min_confidence=0.7
        )
        
        # Analyze insights to determine regime
        regime_votes = defaultdict(int)
        
        for insight in insights:
            if 'volatility' in insight.description.lower():
                regime_votes['volatile'] += insight.confidence
            elif 'trend' in insight.description.lower():
                regime_votes['trending'] += insight.confidence
            elif 'range' in insight.description.lower():
                regime_votes['ranging'] += insight.confidence
            else:
                regime_votes['normal'] += insight.confidence
        
        # Select regime with highest votes
        if regime_votes:
            new_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
            
            if new_regime != self.current_market_regime:
                self.logger.info(f"Market regime changed: {self.current_market_regime} -> {new_regime}")
                self.current_market_regime = new_regime
                
                # Notify agents of regime change
                await self._notify_regime_change(new_regime)
    
    def _calculate_specialization_score(self, agent: BaseAgent) -> float:
        """Calculate how well an agent specializes in its category"""
        # This would analyze the agent's success rate within its category
        # vs other categories
        status = agent.get_status()
        
        # Simple implementation - would be more sophisticated in practice
        if status['success_rate'] > 50:
            return 0.8
        elif status['success_rate'] > 30:
            return 0.6
        elif status['success_rate'] > 10:
            return 0.4
        else:
            return 0.2
    
    def _calculate_collaboration_score(self, agent_name: str) -> float:
        """Calculate agent's collaboration effectiveness"""
        # Count successful collaborations
        collaboration_count = 0
        total_synergy = 0.0
        
        for (agent1, agent2), synergy in self.collaboration_matrix.items():
            if agent1 == agent_name or agent2 == agent_name:
                collaboration_count += 1
                total_synergy += synergy
        
        if collaboration_count == 0:
            return 0.5  # Neutral score if no collaborations
        
        return min(1.0, total_synergy / collaboration_count)
    
    async def _assess_system_needs(self) -> List[Dict[str, Any]]:
        """Assess what the system needs to achieve targets"""
        needs = []
        
        # Check performance gaps
        cagr_gap = self.performance_targets.target_cagr - self.best_system_performance['cagr']
        sharpe_gap = self.performance_targets.target_sharpe - self.best_system_performance['sharpe']
        
        # Need more high-CAGR strategies
        if cagr_gap > 0.05:
            needs.append({
                'type': 'high_return_strategies',
                'parameters': {'min_cagr': self.performance_targets.target_cagr},
                'priority': 1
            })
        
        # Need better risk-adjusted strategies
        if sharpe_gap > 0.2:
            needs.append({
                'type': 'risk_adjusted_strategies',
                'parameters': {'min_sharpe': self.performance_targets.target_sharpe},
                'priority': 1
            })
        
        # Need diversification based on market regime
        if self.current_market_regime == 'volatile':
            needs.append({
                'type': 'defensive_strategies',
                'parameters': {'max_volatility': 0.15},
                'priority': 2
            })
        
        return needs
    
    async def _select_best_agent_for_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Select the best agent for a specific task"""
        best_agent = None
        best_score = 0.0
        
        for agent_name, profile in self.agent_profiles.items():
            # Calculate task fitness score
            score = 0.0
            
            if task['type'] == 'high_return_strategies':
                score = profile.average_cagr * 50 + profile.success_rate * 30 + profile.overall_score * 20
            elif task['type'] == 'risk_adjusted_strategies':
                score = profile.average_sharpe * 50 + profile.success_rate * 30 + profile.overall_score * 20
            elif task['type'] == 'defensive_strategies':
                # Prefer agents with lower volatility strategies
                score = (1 - profile.resource_efficiency) * 30 + profile.success_rate * 40 + profile.overall_score * 30
            
            if score > best_score:
                best_score = score
                best_agent = agent_name
        
        return best_agent
    
    async def _send_task_assignment(self, agent_name: str, task: Dict[str, Any]) -> None:
        """Send task assignment to an agent"""
        message = AgentMessage(
            message_id=f"supervisor_task_{datetime.now().timestamp()}",
            sender=self.config.name,
            recipient=agent_name,
            message_type='request',
            content={
                'task': task,
                'supervisor_directive': True
            },
            priority=task['priority'],
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.send_agent_message(message)
    
    async def _notify_resource_changes(self) -> None:
        """Notify agents of resource allocation changes"""
        for agent_name, allocation in self.resource_allocations.items():
            message = AgentMessage(
                message_id=f"supervisor_resources_{datetime.now().timestamp()}",
                sender=self.config.name,
                recipient=agent_name,
                message_type='notification',
                content={
                    'resource_allocation': asdict(allocation)
                },
                priority=3,
                timestamp=datetime.now()
            )
            
            await self.knowledge_base.send_agent_message(message)
    
    async def _notify_regime_change(self, new_regime: str) -> None:
        """Notify all agents of market regime change"""
        message = AgentMessage(
            message_id=f"supervisor_regime_{datetime.now().timestamp()}",
            sender=self.config.name,
            recipient='all',
            message_type='notification',
            content={
                'market_regime': new_regime,
                'timestamp': datetime.now().isoformat()
            },
            priority=2,
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.send_agent_message(message)
    
    async def _calculate_synergy_score(self, agent1: BaseAgent, agent2: BaseAgent) -> float:
        """Calculate synergy score between two agents"""
        # Simple implementation - would analyze actual strategy correlations
        if agent1.config.category != agent2.config.category:
            # Different categories often have good synergy
            return 0.8
        else:
            # Same category - less synergy
            return 0.3
    
    async def _suggest_collaboration(self, agent1_name: str, agent2_name: str) -> None:
        """Suggest collaboration between two agents"""
        message = AgentMessage(
            message_id=f"supervisor_collab_{datetime.now().timestamp()}",
            sender=self.config.name,
            recipient='all',
            message_type='request',
            content={
                'collaboration_suggestion': {
                    'agents': [agent1_name, agent2_name],
                    'reason': 'high_synergy',
                    'synergy_score': self.collaboration_matrix.get((agent1_name, agent2_name), 0.0)
                }
            },
            priority=3,
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.send_agent_message(message)
    
    async def _broadcast_achievement(self) -> None:
        """Broadcast target achievement to all agents"""
        message = AgentMessage(
            message_id=f"supervisor_achievement_{datetime.now().timestamp()}",
            sender=self.config.name,
            recipient='all',
            message_type='notification',
            content={
                'achievement': 'performance_targets_met',
                'performance': self.best_system_performance,
                'timestamp': datetime.now().isoformat()
            },
            priority=1,
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.send_agent_message(message)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        return {
            'supervisor_status': self.get_status(),
            'system_performance': self.best_system_performance,
            'targets_achieved': self.targets_achieved,
            'market_regime': self.current_market_regime,
            'agent_profiles': {
                name: {
                    'success_rate': profile.success_rate,
                    'average_sharpe': profile.average_sharpe,
                    'overall_score': profile.overall_score,
                    'resource_allocation': asdict(self.resource_allocations[name])
                }
                for name, profile in self.agent_profiles.items()
            },
            'knowledge_base_summary': self.knowledge_base.get_knowledge_summary(),
            'runtime_hours': (
                (datetime.now() - self.system_start_time).total_seconds() / 3600
                if self.system_start_time else 0
            )
        }
    
    def get_active_agents(self) -> List[str]:
        """Get list of active agents"""
        return list(self.managed_agents.keys())
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all managed agents"""
        self.logger.info("Supervisor initiating system shutdown...")
        
        # Notify all agents
        message = AgentMessage(
            message_id=f"supervisor_shutdown_{datetime.now().timestamp()}",
            sender=self.config.name,
            recipient='all',
            message_type='notification',
            content={'shutdown': True},
            priority=1,
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.send_agent_message(message)
        
        # Stop all managed agents
        for agent_name, agent in self.managed_agents.items():
            self.logger.info(f"Stopping agent: {agent_name}")
            await agent.stop()
        
        # Stop supervisor
        await super().stop()
        
        self.logger.info("Supervisor shutdown complete")