"""
Base Agent Class for Multi-Agent System
Provides common functionality for all specialized trading agents
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from tier1_core.logger import get_logger, PERF_LOGGER
from tier2_strategy.strategy_generator import StrategyGenerator, GeneratedStrategy
from tier2_strategy.strategy_tester import StrategyTester, StrategyPerformance
from agents.knowledge_base import SharedKnowledgeBase, MarketInsight, StrategyDiscovery, AgentMessage
from tier1_core.real_time_dashboard import DASHBOARD # Moved import to top

from typing import Dict, Any, Optional # Ensure Any, Dict are imported if not already
from dataclasses import field # Ensure field is imported if not already for default_factory

@dataclass
class AgentState:
    """Agent operational state"""
    is_active: bool = False
    status: str = "stopped"  # Added: e.g., stopped, initializing, running, error, idle
    details: Dict[str, Any] = field(default_factory=dict)  # Added
    last_error: str = "" # Added from original prompt's suggestion for state

    # Existing fields from the file
    strategies_generated: int = 0
    strategies_tested: int = 0
    successful_strategies: int = 0
    best_sharpe: float = 0.0
    best_cagr: float = 0.0
    last_activity: Optional[datetime] = None
    current_task: Optional[str] = None

@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str
    category: str  # Strategy category focus
    max_concurrent_tasks: int = 3
    generation_batch_size: int = 10
    min_sharpe_threshold: float = 0.5
    min_cagr_threshold: float = 0.05
    risk_tolerance: float = 0.5  # 0=conservative, 1=aggressive
    exploration_rate: float = 0.2  # Exploration vs exploitation
    communication_frequency: int = 60  # seconds

class BaseAgent(ABC):
    """
    Base class for all trading agents in the multi-agent system
    
    Features:
    - Autonomous operation with task management
    - Knowledge base integration for collaboration
    - Strategy generation and testing capabilities
    - Performance tracking and reporting
    - Inter-agent communication
    """
    
    def __init__(
        self,
        config: AgentConfig,
        strategy_generator: StrategyGenerator,
        strategy_tester: StrategyTester,
        knowledge_base: SharedKnowledgeBase
    ):
        self.config = config
        self.generator = strategy_generator
        self.tester = strategy_tester
        self.knowledge_base = knowledge_base
        try:
            self.logger = get_logger(f"agent.{config.name if hasattr(config, 'name') else 'Unknown'}")
        except Exception as e:
            print(f"!!! Logger initialization failed for agent.{config.name if hasattr(config, 'name') else 'Unknown'}: {e}") # Will show in test output
            import logging
            # Provide a fallback logger instance
            self.logger = logging.getLogger(f"FallbackAgent.{config.name if hasattr(config, 'name') else 'Unknown'}")
            self.logger.warning("Using fallback logger due to initialization error.")
        
        # Agent state
        self.state = AgentState()
        self.agent_id = str(uuid.uuid4())[:8]
        
        # Task management
        self.tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.discovered_strategies: List[StrategyDiscovery] = []
        
        # Communication
        self.last_message_check = datetime.now()
        self.pending_responses: Dict[str, Any] = {}
        
    async def initialize(self) -> None:
        """Initialize the agent"""
        await self.knowledge_base.register_agent(self.config.name)
        # Removed misplaced import from here

        self.state.is_active = True
        self.state.last_activity = datetime.now()
        self.state.status = "initializing" # Update status
        
        self.logger.info(f"Agent '{self.config.name}' initialized with ID {self.agent_id}")
        DASHBOARD.log_agent_activity(self.config.name, "Agent initialized", {"agent_id": self.agent_id})
        
        # Agent-specific initialization
        await self._initialize_agent()
        self.state.status = "idle" # After specific init, move to idle
    
    @abstractmethod
    async def _initialize_agent(self) -> None:
        """Agent-specific initialization"""
        pass
    
    async def start(self) -> None:
        """Start the agent's autonomous operation"""
        if not self.state.is_active:
            await self.initialize()
        
        # Start background tasks
        self.tasks.append(asyncio.create_task(self._main_loop()))
        self.tasks.append(asyncio.create_task(self._communication_loop()))
        self.tasks.append(asyncio.create_task(self._monitoring_loop())) # Corrected typo
        
        self.state.status = "running" # Update status
        self.logger.info(f"Agent '{self.config.name}' started")
        DASHBOARD.log_agent_activity(self.config.name, "Agent started", {}) # Log start
    
    async def stop(self) -> None:
        """Stop the agent"""
        self.logger.info(f"Stopping agent '{self.config.name}'...")
        DASHBOARD.log_agent_activity(self.config.name, "Agent stopping", {}) # Log stopping
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Unregister from knowledge base
        await self.knowledge_base.unregister_agent(self.config.name)
        
        self.state.is_active = False
        self.state.status = "stopped" # Update status
        self.logger.info(f"Agent '{self.config.name}' stopped")
        DASHBOARD.log_agent_activity(self.config.name, "Agent stopped", {}) # Log stopped
    
    async def _main_loop(self) -> None:
        """Main agent operation loop"""
        while not self.shutdown_event.is_set():
            try:
                # Update current task
                self.state.current_task = "strategy_generation"
                
                # Get market insights from knowledge base
                insights = await self._get_relevant_insights()
                
                # Generate strategies based on insights and agent specialization
                strategies = await self._generate_strategies(insights)
                
                if strategies:
                    # Test strategies
                    self.state.current_task = "strategy_testing"
                    performances = await self._test_strategies(strategies)
                    
                    # Process results
                    await self._process_results(performances)
                    
                    # Share discoveries
                    await self._share_discoveries(performances)
                
                # Update state
                self.state.last_activity = datetime.now()
                self.state.current_task = None
                
                # Brief pause
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(10)
    
    async def _communication_loop(self) -> None:
        """Handle inter-agent communication"""
        while not self.shutdown_event.is_set():
            try:
                # Check for messages
                messages = await self.knowledge_base.get_agent_messages(
                    self.config.name,
                    since=self.last_message_check
                )
                
                self.last_message_check = datetime.now()
                
                # Process messages
                for message in messages:
                    await self._handle_message(message)
                
                await asyncio.sleep(self.config.communication_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in communication loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def _monitoring_loop(self) -> None:
        """Monitor agent performance and health"""
        while not self.shutdown_event.is_set():
            try:
                # Calculate performance metrics
                metrics = self._calculate_performance_metrics()
                
                # Record to knowledge base
                await self.knowledge_base.record_agent_performance(
                    self.config.name,
                    metrics
                )
                
                # Log performance
                PERF_LOGGER.log_agent_performance(self.config.name, metrics)
                
                # Check for performance issues
                if metrics.get('success_rate', 0) < 0.1:
                    self.logger.warning(f"Low success rate: {metrics['success_rate']:.2%}")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(600)
    
    @abstractmethod
    async def _generate_strategies(self, insights: List[MarketInsight]) -> List[GeneratedStrategy]:
        """Generate strategies based on agent specialization"""
        pass
    
    async def _test_strategies(self, strategies: List[GeneratedStrategy]) -> List[StrategyPerformance]:
        """Test generated strategies"""
        if not strategies:
            return []
        
        self.logger.info(f"Testing {len(strategies)} strategies")
        
        # Test strategies
        performances = await self.tester.test_strategies(strategies)
        
        # Update state
        self.state.strategies_tested += len(strategies)
        
        # Track successful strategies
        successful = [p for p in performances if self._is_successful_strategy(p)]
        self.state.successful_strategies += len(successful)
        
        # Update best performance
        for perf in successful:
            if perf.sharpe_ratio > self.state.best_sharpe:
                self.state.best_sharpe = perf.sharpe_ratio
            if perf.cagr > self.state.best_cagr:
                self.state.best_cagr = perf.cagr
        
        return performances
    
    async def _process_results(self, performances: List[StrategyPerformance]) -> None:
        """Process strategy test results"""
        # Store performance history
        for perf in performances:
            self.performance_history.append({
                'timestamp': datetime.now(),
                'strategy_id': perf.strategy_id,
                'sharpe': perf.sharpe_ratio,
                'cagr': perf.cagr,
                'drawdown': perf.max_drawdown
            })
        
        # Identify successful strategies
        successful = [p for p in performances if self._is_successful_strategy(p)]
        
        if successful:
            self.logger.info(
                f"Found {len(successful)} successful strategies. "
                f"Best Sharpe: {max(p.sharpe_ratio for p in successful):.2f}"
            )
    
    async def _share_discoveries(self, performances: List[StrategyPerformance]) -> None:
        """Share successful strategies with other agents"""
        for perf in performances:
            if not self._is_successful_strategy(perf):
                continue
            
            # Create discovery
            discovery = StrategyDiscovery(
                discovery_id=f"{self.config.name}_{perf.strategy_id}",
                agent_name=self.config.name,
                template_name=perf.template_name,
                category=perf.category,
                parameters={},  # Would need to get from strategy
                performance_metrics={
                    'cagr': perf.cagr,
                    'sharpe': perf.sharpe_ratio,
                    'drawdown': perf.max_drawdown,
                    'sortino': perf.sortino_ratio,
                    'calmar': perf.calmar_ratio
                },
                market_conditions={
                    'timeframe': 'Daily',
                    'asset_class': 'Equity'
                },
                complementary_strategies=[],
                risk_profile={
                    'volatility': perf.volatility,
                    'beta': perf.beta,
                    'var_95': perf.var_95
                },
                timestamp=datetime.now()
            )
            
            # Add to knowledge base
            await self.knowledge_base.add_strategy_discovery(discovery)
            
            # Track discovery
            self.discovered_strategies.append(discovery)
    
    async def _get_relevant_insights(self) -> List[MarketInsight]:
        """Get relevant market insights from knowledge base"""
        return await self.knowledge_base.get_relevant_insights(
            agent_name=self.config.name,
            category=None,  # Get all categories
            asset_class='Equity',
            min_confidence=0.6
        )
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle incoming message from another agent"""
        self.logger.debug(f"Received message from {message.sender}: {message.message_type}")
        
        if message.message_type == 'request':
            # Handle strategy request
            await self._handle_strategy_request(message)
        elif message.message_type == 'discovery':
            # Process shared discovery
            await self._handle_discovery_notification(message)
        elif message.message_type == 'warning':
            # Process warning
            self.logger.warning(f"Warning from {message.sender}: {message.content}")
    
    async def _handle_strategy_request(self, message: AgentMessage) -> None:
        """Handle strategy generation request from another agent"""
        # This would be implemented based on specific request types
        pass
    
    async def _handle_discovery_notification(self, message: AgentMessage) -> None:
        """Handle discovery notification from another agent"""
        # Process and potentially act on the discovery
        pass
    
    def _is_successful_strategy(self, performance: StrategyPerformance) -> bool:
        """Determine if a strategy is successful based on agent criteria"""
        return (
            performance.sharpe_ratio >= self.config.min_sharpe_threshold and
            performance.cagr >= self.config.min_cagr_threshold and
            performance.max_drawdown <= 0.20  # Max 20% drawdown
        )
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate agent performance metrics"""
        success_rate = (
            self.state.successful_strategies / self.state.strategies_tested
            if self.state.strategies_tested > 0 else 0.0
        )
        
        generation_rate = (
            self.state.strategies_generated / 
            ((datetime.now() - self.state.last_activity).total_seconds() / 3600)
            if self.state.last_activity else 0.0
        )
        
        return {
            'strategies_generated': self.state.strategies_generated,
            'strategies_tested': self.state.strategies_tested,
            'successful_strategies': self.state.successful_strategies,
            'success_rate': success_rate,
            'generation_rate_per_hour': generation_rate,
            'best_sharpe': self.state.best_sharpe,
            'best_cagr': self.state.best_cagr,
            'discoveries': len(self.discovered_strategies)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'name': self.config.name,
            'category': self.config.category,
            'is_active': self.state.is_active,
            'current_task': self.state.current_task,
            'strategies_generated': self.state.strategies_generated,
            'strategies_tested': self.state.strategies_tested,
            'successful_strategies': self.state.successful_strategies,
            'success_rate': (
                self.state.successful_strategies / self.state.strategies_tested * 100
                if self.state.strategies_tested > 0 else 0.0
            ),
            'best_performance': {
                'sharpe': self.state.best_sharpe,
                'cagr': self.state.best_cagr
            },
            'last_activity': self.state.last_activity.isoformat() if self.state.last_activity else None
        }
    
    async def request_collaboration(
        self, 
        agent_names: List[str], 
        task: str, 
        parameters: Dict[str, Any]
    ) -> None:
        """Request collaboration from other agents"""
        message = AgentMessage(
            message_id=f"{self.config.name}_collab_{uuid.uuid4().hex[:8]}",
            sender=self.config.name,
            recipient='all' if len(agent_names) > 1 else agent_names[0],
            message_type='request',
            content={
                'task': task,
                'parameters': parameters,
                'requesting_agent': self.config.name
            },
            priority=2,
            timestamp=datetime.now()
        )
        
        await self.knowledge_base.send_agent_message(message)