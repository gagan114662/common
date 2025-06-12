import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock #Removed patch

from agents.supervisor_agent import SupervisorAgent
from agents.knowledge_base import SharedKnowledgeBase
#from tier1_core.controller import SystemController # Controller not used directly by SupervisorAgent
from config.settings import SYSTEM_CONFIG # Used for agent_config
from tier2_strategy.strategy_generator import StrategyGenerator
from tier2_strategy.strategy_tester import StrategyTester
from tier3_evolution.evolution_engine import EvolutionEngine
from tier1_core.performance_monitor import PerformanceMonitor
from agents.base_agent import AgentConfig # For mock agents

class TestSupervisorAgent(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Mock dependencies for SupervisorAgent constructor
        self.mock_strategy_generator = MagicMock(spec=StrategyGenerator)
        self.mock_strategy_tester = MagicMock(spec=StrategyTester)
        self.mock_evolution_engine = MagicMock(spec=EvolutionEngine)
        self.mock_performance_monitor = MagicMock(spec=PerformanceMonitor)
        self.mock_knowledge_base = MagicMock(spec=SharedKnowledgeBase)

        # Mock agents - these will be added via add_managed_agent
        # Create specific configs for mock agents to give them names
        self.mock_trend_agent_config = AgentConfig(name="TrendFollowingAgent", category="trend", max_concurrent_tasks=1)
        self.mock_trend_agent = AsyncMock(spec=SupervisorAgent) # Using SupervisorAgent spec for now for base methods
        self.mock_trend_agent.config = self.mock_trend_agent_config
        self.mock_trend_agent.get_status.return_value = { # Mocking get_status for this agent
            "agent_name": "TrendFollowingAgent", "is_running": True, "state": "running",
            "strategies_generated": 0, "strategies_tested": 0,
            "successful_strategies": 0, "success_rate": 0.0, "last_error": None
        }


        self.mock_mean_reversion_agent_config = AgentConfig(name="MeanReversionAgent", category="reversion", max_concurrent_tasks=1)
        self.mock_mean_reversion_agent = AsyncMock(spec=SupervisorAgent)
        self.mock_mean_reversion_agent.config = self.mock_mean_reversion_agent_config
        self.mock_mean_reversion_agent.get_status.return_value = {
            "agent_name": "MeanReversionAgent", "is_running": True, "state": "running",
            "strategies_generated": 0, "strategies_tested": 0,
            "successful_strategies": 0, "success_rate": 0.0, "last_error": None
        }

        self.mock_research_agent_config = AgentConfig(name="ResearchHypothesisAgent", category="research", max_concurrent_tasks=1)
        self.mock_research_agent = AsyncMock(spec=SupervisorAgent)
        self.mock_research_agent.config = self.mock_research_agent_config
        self.mock_research_agent.get_status.return_value = {
            "agent_name": "ResearchHypothesisAgent", "is_running": True, "state": "running",
            "strategies_generated": 0, "strategies_tested": 0,
            "successful_strategies": 0, "success_rate": 0.0, "last_error": None
        }

        # Actual SYSTEM_CONFIG will be passed as agent_config to SupervisorAgent
        # The supervisor's own config is hardcoded internally.
        self.supervisor = SupervisorAgent(
            strategy_generator=self.mock_strategy_generator,
            strategy_tester=self.mock_strategy_tester,
            evolution_engine=self.mock_evolution_engine,
            performance_monitor=self.mock_performance_monitor,
            knowledge_base=self.mock_knowledge_base,
            agent_config=SYSTEM_CONFIG
        )
        # The supervisor's internal config (self.config) is distinct from agent_config (self.system_agent_config)
        # self.supervisor.state.is_active is False after BaseAgent init

    async def asyncTearDown(self):
        # No patchers to stop
        # Ensure supervisor is stopped if it was started and state is initialized
        if hasattr(self.supervisor, 'state') and self.supervisor.state.is_active:
             await self.supervisor.stop()
        pass

    async def test_supervisor_initialization(self):
        self.assertEqual(self.supervisor.strategy_generator, self.mock_strategy_generator)
        self.assertEqual(self.supervisor.strategy_tester, self.mock_strategy_tester)
        self.assertEqual(self.supervisor.evolution_engine, self.mock_evolution_engine)
        self.assertEqual(self.supervisor.performance_monitor, self.mock_performance_monitor)
        self.assertEqual(self.supervisor.knowledge_base, self.mock_knowledge_base)
        self.assertEqual(self.supervisor.system_agent_config, SYSTEM_CONFIG)
        self.assertIsNotNone(self.supervisor.logger)
        self.assertFalse(self.supervisor.state.is_active) # is_active is False after BaseAgent init

        # Supervisor no longer auto-instantiates these agents. They are added via add_managed_agent.
        # So, these checks are removed:
        # self.MockTrendAgent.assert_called_once_with(self.mock_knowledge_base, SYSTEM_CONFIG.agents.trend_following)
        # self.MockMeanReversionAgent.assert_called_once_with(self.mock_knowledge_base, SYSTEM_CONFIG.agents.mean_reversion)
        # self.MockResearchAgent.assert_called_once_with(self.mock_knowledge_base, SYSTEM_CONFIG.agents.research_hypothesis)
        self.assertEqual(len(self.supervisor.managed_agents), 0) # Starts with no managed agents

    async def test_start_supervisor_starts_itself(self): # Renamed and modified
        # Supervisor.start() starts itself, not its managed agents directly.
        await self.supervisor.start()
        self.assertTrue(self.supervisor.state.is_active) # start() calls initialize() which sets is_active = True
        self.assertTrue(len(self.supervisor.tasks) > 0) # BaseAgent.start() creates tasks
        # Stop the supervisor's loop to clean up
        await self.supervisor.stop()
        self.assertFalse(self.supervisor.state.is_active)


    async def test_stop_supervisor_stops_all_managed_agents(self): # Renamed and modified
        # Add mock agents to the supervisor
        self.supervisor.add_managed_agent(self.mock_trend_agent)
        self.supervisor.add_managed_agent(self.mock_mean_reversion_agent)
        self.supervisor.add_managed_agent(self.mock_research_agent)

        # Start the supervisor (its start doesn't start managed agents, but shutdown should stop them)
        # Simulate managed agents are active for stop() to be meaningful on them.
        # The mock agents need a 'state' attribute with 'is_active'
        self.mock_trend_agent.state = MagicMock(is_active=True)
        self.mock_mean_reversion_agent.state = MagicMock(is_active=True)
        self.mock_research_agent.state = MagicMock(is_active=True)

        # The actual SupervisorAgent.shutdown() calls super().stop() which stops its own tasks.
        # It also iterates self.managed_agents and calls stop() on them.
        # For this test, ensure supervisor itself is started so shutdown can proceed.
        await self.supervisor.start() # Start supervisor so it can be shut down

        await self.supervisor.shutdown() # This is the method that stops managed agents

        self.mock_trend_agent.stop.assert_called_once()
        self.mock_mean_reversion_agent.stop.assert_called_once()
        self.mock_research_agent.stop.assert_called_once()
        self.assertFalse(self.supervisor.state.is_active) # Supervisor's own state.is_active is set to False

    async def test_supervisor_main_loop_cycle_runs_monitoring_and_analysis(self):
        # This test reinterprets "run_cycle" for the supervisor.
        # It checks if the supervisor's _main_loop performs its core duties.
        # We'll mock parts of _main_loop's functionality.

        self.supervisor._monitor_system_performance = AsyncMock()
        self.supervisor._analyze_agent_performance = AsyncMock()
        self.supervisor._rebalance_resources = AsyncMock()
        self.supervisor._coordinate_agents = AsyncMock()
        self.supervisor._check_performance_targets = AsyncMock()
        self.supervisor._identify_collaborations = AsyncMock()
        self.supervisor._optimize_system = AsyncMock()

        # To ensure the loop runs once and then exits for the test
        # The supervisor's _main_loop sleeps for 10 seconds.
        # We will mock asyncio.sleep to control the loop.

        original_main_loop_sleep_duration = 10 # Default sleep in Supervisor's _main_loop
        real_asyncio_sleep = asyncio.sleep # Capture real sleep

        # This mock_sleep will be for the supervisor's own _main_loop
        async def supervisor_main_loop_mock_sleep(duration):
            if duration == original_main_loop_sleep_duration: # Target only the main loop's sleep
                # Schedule stop to run soon, decouple from this await chain
                asyncio.create_task(self.supervisor.stop())
            await real_asyncio_sleep(0.001) # Call real sleep

        with unittest.mock.patch('agents.supervisor_agent.asyncio.sleep', supervisor_main_loop_mock_sleep):
            await self.supervisor.start() # This will start all three loops in BaseAgent

            # Wait for all tasks to complete. BaseAgent.stop() handles this.
            # The start() method is not awaited fully itself, it launches tasks.
            # We need to wait for the agent to effectively stop.
            # A simple way is to poll state.is_active or use a timeout.
            for _ in range(100): # Timeout after ~1 second
                if not self.supervisor.state.is_active:
                    break
                await asyncio.sleep(0.01)

        self.supervisor._monitor_system_performance.assert_called()
        self.supervisor._analyze_agent_performance.assert_called()
        # _rebalance_resources is conditional, so might not be called every cycle
        # self.supervisor._rebalance_resources.assert_called()
        self.supervisor._coordinate_agents.assert_called()
        self.supervisor._check_performance_targets.assert_called()
        self.supervisor._identify_collaborations.assert_called()
        self.supervisor._optimize_system.assert_called()

        # Ensure it stopped
        self.assertFalse(self.supervisor.state.is_active)


    async def test_main_loop_handles_exception_gracefully(self):
        # Test that if a part of the main loop logic raises an exception,
        # the supervisor logs it and continues (or stops gracefully).
        self.supervisor._monitor_system_performance = AsyncMock(side_effect=Exception("Test Monitoring Error"))

        real_asyncio_sleep = asyncio.sleep # Capture real sleep
        stop_triggered_event = asyncio.Event()

        async def mock_sleep_after_error(duration):
            # Call stop only once, the first time supervisor tries to sleep after the error.
            if not stop_triggered_event.is_set():
                # Schedule stop to run soon
                asyncio.create_task(self.supervisor.stop())
                stop_triggered_event.set()
            await real_asyncio_sleep(0.001) # Allow other sleeps to be short

        with self.assertLogs(self.supervisor.logger, level='ERROR') as cm:
            # Patching sleep within supervisor_agent module where _main_loop is defined
            with unittest.mock.patch('agents.supervisor_agent.asyncio.sleep', mock_sleep_after_error):
                await self.supervisor.start()
                # Wait for agent to stop
                for _ in range(100): # Timeout
                    if not self.supervisor.state.is_active:
                        break
                    await asyncio.sleep(0.01) # Use real asyncio.sleep for polling

        self.assertTrue(any("Error in supervisor main loop: Test Monitoring Error" in message for message in cm.output))
        self.assertFalse(self.supervisor.state.is_active) # Supervisor should have stopped

    def test_get_status_returns_supervisor_own_status(self):
        # This test now checks the actual get_status from BaseAgent for the supervisor itself.

        # Initialize agent state for status check, as start() is not called in this sync test
        # Normally, initialize() is called by start() which sets is_active.
        # For a synchronous test of get_status, we manually set state if needed or ensure defaults.
        # BaseAgent.__init__ sets up self.state = AgentState() (is_active=False by default)
        # and self.config.

        status = self.supervisor.get_status() # This is BaseAgent.get_status()

        self.assertEqual(status['name'], "supervisor") # Changed from 'agent_name'
        self.assertFalse(status['is_active']) # is_active is false by default after init
        self.assertIn('category', status)
        # self.assertIn('state', status) # 'state' key is not in BaseAgent.get_status()
                                        # It returns discrete state values.

if __name__ == '__main__':
    unittest.main()
