import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

from agents.trend_following_agent import TrendFollowingAgent
from agents.knowledge_base import SharedKnowledgeBase
from config.settings import SYSTEM_CONFIG
# Needed for TrendFollowingAgent __init__ due to BaseAgent requirements
from tier2_strategy.strategy_generator import StrategyGenerator
from tier2_strategy.strategy_tester import StrategyTester

class TestTrendFollowingAgent(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.mock_knowledge_base = AsyncMock(spec=SharedKnowledgeBase)
        # Assuming SYSTEM_CONFIG.agents.trend_following is a valid config object
        # that matches what TrendFollowingAgent expects for its 'config' parameter.
        # This config object should have attributes like initial_symbols_watchlist, name, category, etc.
        self.agent_specific_config = SYSTEM_CONFIG.agents.trend_following # This is an AgentConfig from settings

        # Mocks for BaseAgent constructor requirements, TrendFollowingAgent will pass these to super()
        self.mock_strategy_generator = AsyncMock(spec=StrategyGenerator)
        self.mock_strategy_tester = AsyncMock(spec=StrategyTester)

        # Mock the dashboard for logging
        # Patching where DASHBOARD is imported and used, assuming base_agent or trend_following_agent
        self.dashboard_patcher = patch('agents.base_agent.DASHBOARD', new_callable=MagicMock)
        self.MockDashboard = self.dashboard_patcher.start()

        # Also patch DASHBOARD if TrendFollowingAgent imports it directly
        # (The provided agent code for TFA now also imports DASHBOARD)
        self.tfa_dashboard_patcher = patch('agents.trend_following_agent.DASHBOARD', new_callable=MagicMock)
        self.MockTFADashboard = self.tfa_dashboard_patcher.start()


        self.agent = TrendFollowingAgent(
            knowledge_base=self.mock_knowledge_base,
            config=self.agent_specific_config, # Pass the specific config part
            strategy_generator=self.mock_strategy_generator,
            strategy_tester=self.mock_strategy_tester
        )

        # BaseAgent.__init__ initializes self.state to an AgentState object.
        # The test assumes it can directly set attributes on this state for some tests,
        # or that start/stop methods correctly update it.
        # The test uses self.agent.state.is_active, self.agent.state.status.
        # AgentState dataclass in base_agent.py now has these.
        # No need to mock self.state if BaseAgent.__init__ does its job.
        # self.agent.state = MagicMock(is_active=True, status="idle", details={}) # Removed this line

    async def asyncTearDown(self):
        self.dashboard_patcher.stop()
        self.tfa_dashboard_patcher.stop()
        # Ensure agent is stopped if it was started and has a state
        if hasattr(self.agent, 'state') and self.agent.state.is_active:
            await self.agent.stop()

    async def test_agent_initialization(self):
        self.assertEqual(self.agent.knowledge_base, self.mock_knowledge_base)
        # TrendFollowingAgent stores its detailed config in self.config
        self.assertEqual(self.agent.config, self.agent_specific_config)
        # The name used by BaseAgent (and thus by self.logger, self.state, etc.)
        # is derived from the name field of the config object passed to TrendFollowingAgent's __init__
        self.assertEqual(self.agent.config.name, self.agent_specific_config.name)
        self.assertIsNotNone(self.agent.logger)
        self.assertIsNotNone(self.agent.state) # Should be an AgentState instance
        self.assertFalse(self.agent.state.is_active) # is_active is False after init, before start
        self.assertEqual(self.agent.state.status, "stopped") # Default status from AgentState

    async def test_start_agent(self):
        # The actual log messages and details depend on BaseAgent and TFA._initialize_agent
        # BaseAgent.initialize() logs "Agent initialized"
        # BaseAgent.start() logs "Agent started"
        # The test expects "Agent started" with "initial_symbols".
        # This specific detail is now logged by TFA._initialize_agent via DASHBOARD if needed,
        # but the primary "Agent started" log is from BaseAgent.start().
        # Let's adjust the test to reflect the actual log from BaseAgent.start().

        await self.agent.start()
        self.assertTrue(self.agent.state.is_active)
        self.assertEqual(self.agent.state.status, "running") # Set by BaseAgent.start()

        # Check the log made by BaseAgent.start()
        # The dashboard mock in base_agent is self.MockDashboard
        self.MockDashboard.log_agent_activity.assert_any_call(
            self.agent.config.name, "Agent started", {}
        )
        # The test originally expected: {"initial_symbols": self.config.initial_symbols_watchlist}
        # This detail might be logged by TFA._initialize_agent if we add it there,
        # or the test needs to be less specific about the details for the "Agent started" log.
        # For now, I'll check the generic one from BaseAgent.
        # The TFA._initialize_agent now logs "Agent specific initialization complete"
        self.MockTFADashboard.log_agent_activity.assert_any_call(
            self.agent.config.name, "Agent specific initialization complete",
            {"initial_symbols": self.agent.symbols_watchlist}
        )


    async def test_stop_agent(self):
        await self.agent.start() # Start first
        await self.agent.stop()
        self.assertFalse(self.agent.state.is_active)
        self.assertEqual(self.agent.state.status, "stopped")
        # BaseAgent.stop() logs "Agent stopping" then "Agent stopped"
        self.MockDashboard.log_agent_activity.assert_any_call(
            self.agent.config.name, "Agent stopped", {}
        )

    @patch('agents.trend_following_agent.yf.Ticker')
    async def test_fetch_market_data_success(self, MockTicker):
        mock_ticker_instance = MockTicker.return_value
        mock_history = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Volume': [1000] * 11
        })
        mock_ticker_instance.history.return_value = mock_history

        # Use the agent's actual config for data_fetch_period
        data = await self.agent._fetch_market_data("TEST.SYMBOL")

        self.assertIsNotNone(data)
        self.assertEqual(len(data), 11)
        MockTicker.assert_called_with("TEST.SYMBOL")
        mock_ticker_instance.history.assert_called_with(period=self.agent.data_fetch_period)

    @patch('agents.trend_following_agent.yf.Ticker')
    async def test_fetch_market_data_failure(self, MockTicker):
        expected_error_message = "API Error"
        MockTicker.return_value.history.side_effect = Exception(expected_error_message)

        data = await self.agent._fetch_market_data("FAIL.SYMBOL")

        self.assertIsNone(data)
        # Check the log made by TrendFollowingAgent._fetch_market_data
        self.MockTFADashboard.log_agent_activity.assert_called_with(
            self.agent.config.name, f"Error fetching data for FAIL.SYMBOL: {expected_error_message}", {"symbol": "FAIL.SYMBOL"}
        )

    async def test_identify_trend_strong_uptrend(self):
        # Prices clearly going up
        data = pd.DataFrame({
            'Close': [i for i in range(100, 100 + self.agent.min_data_points_for_trend * 2, 2)],
            'Volume': [1000] * (self.agent.min_data_points_for_trend) # Ensure enough data
        })
        trend_signal = self.agent._identify_trend(data, "TEST.SYMBOL")
        self.assertIsNotNone(trend_signal)
        self.assertEqual(trend_signal['symbol'], "TEST.SYMBOL")
        self.assertEqual(trend_signal['trend_type'], "uptrend")
        self.assertGreater(trend_signal['strength_score'], 0.7)

    async def test_identify_trend_strong_downtrend(self):
        data = pd.DataFrame({
            'Close': [i for i in range(100 + self.agent.min_data_points_for_trend * 2, 100, -2)],
            'Volume': [1000] * (self.agent.min_data_points_for_trend)
        })
        trend_signal = self.agent._identify_trend(data, "TEST.SYMBOL")
        self.assertIsNotNone(trend_signal)
        self.assertEqual(trend_signal['symbol'], "TEST.SYMBOL")
        self.assertEqual(trend_signal['trend_type'], "downtrend")
        self.assertGreater(trend_signal['strength_score'], 0.7)

    async def test_identify_trend_no_clear_trend(self):
        # Sideways data, ensure enough points
        prices = [100, 101] * (self.agent.min_data_points_for_trend // 2)
        if len(prices) < self.agent.min_data_points_for_trend: # Ensure enough data
            prices.extend([100] * (self.agent.min_data_points_for_trend - len(prices)))

        data = pd.DataFrame({'Close': prices, 'Volume': [1000]*len(prices)})
        trend_signal = self.agent._identify_trend(data, "TEST.SYMBOL")
        self.assertIsNotNone(trend_signal)
        self.assertEqual(trend_signal['symbol'], "TEST.SYMBOL")
        self.assertEqual(trend_signal['trend_type'], "no_clear_trend")
        # For "no_clear_trend", strength_score is abs(normalized_slope) which should be low
        self.assertLess(trend_signal['strength_score'], self.agent.trend_strength_threshold)


    async def test_identify_trend_insufficient_data(self):
        data = pd.DataFrame({'Close': [100, 101, 102]}) # Clearly not enough
        trend_signal = self.agent._identify_trend(data, "TEST.SYMBOL")
        self.assertIsNotNone(trend_signal)
        self.assertEqual(trend_signal['symbol'], "TEST.SYMBOL")
        self.assertEqual(trend_signal['trend_type'], "insufficient_data")
        self.assertEqual(trend_signal['strength_score'], 0)

    @patch('agents.trend_following_agent.TrendFollowingAgent._fetch_market_data')
    @patch('agents.trend_following_agent.TrendFollowingAgent._identify_trend')
    async def test_run_cycle_processes_symbols_and_updates_kb(self, mock_identify_trend, mock_fetch_market_data):
        # Do not start the agent's main loop for this test, as we call run_cycle directly.
        # await self.agent.start()
        self.agent.symbols_watchlist = ["SYM1", "SYM2"]

        # Define an async side_effect function for mock_fetch_market_data
        async def fetch_side_effect(symbol):
            if symbol == "SYM1":
                return pd.DataFrame({'Close': [100]*self.agent.min_data_points_for_trend, 'Volume': [100]*self.agent.min_data_points_for_trend})
            elif symbol == "SYM2":
                return pd.DataFrame({'Close': [200]*self.agent.min_data_points_for_trend, 'Volume': [100]*self.agent.min_data_points_for_trend})
            return None
        mock_fetch_market_data.side_effect = fetch_side_effect

        mock_identify_trend.side_effect = [
            {"symbol": "SYM1", "trend_type": "uptrend", "strength_score": 0.8, "details": {}},
            {"symbol": "SYM2", "trend_type": "downtrend", "strength_score": 0.9, "details": {}}
        ]

        await self.agent.run_cycle()

        self.assertEqual(mock_fetch_market_data.call_count, 2)
        mock_fetch_market_data.assert_any_call("SYM1")
        mock_fetch_market_data.assert_any_call("SYM2")
        self.assertEqual(mock_identify_trend.call_count, 2)

        # Check that add_market_insight was called for each symbol
        self.assertEqual(self.mock_knowledge_base.add_market_insight.call_count, 2)

        # Inspect the calls to add_market_insight
        call_args_list = self.mock_knowledge_base.add_market_insight.call_args_list

        # Check SYM1 insight
        sym1_insight_call = next(c for c in call_args_list if c[0][0].symbols == ["SYM1"])
        self.assertIsNotNone(sym1_insight_call)
        sym1_insight_obj = sym1_insight_call[0][0]
        self.assertEqual(sym1_insight_obj.category, "trend_following_uptrend") # Corrected expected category
        self.assertEqual(sym1_insight_obj.confidence, 0.8) # strength_score used as confidence
        self.assertEqual(sym1_insight_obj.supporting_data, {}) # details from mock_identify_trend

        # Check SYM2 insight
        sym2_insight_call = next(c for c in call_args_list if c[0][0].symbols == ["SYM2"])
        self.assertIsNotNone(sym2_insight_call)
        sym2_insight_obj = sym2_insight_call[0][0]
        self.assertEqual(sym2_insight_obj.category, "trend_following_downtrend") # Corrected expected category
        self.assertEqual(sym2_insight_obj.confidence, 0.9)
        self.assertEqual(sym2_insight_obj.supporting_data, {})

        # This log is from TrendFollowingAgent.run_cycle
        self.MockTFADashboard.log_agent_activity.assert_any_call(
            self.agent.config.name, "Trend analysis complete", {"symbols_analyzed": 2, "trends_found": 2}
        )
        # No need to call self.agent.stop() if agent was not started.

    async def test_run_cycle_handles_fetch_failure(self):
        # Do not start the agent's main loop for this test.
        # await self.agent.start()
        self.agent.symbols_watchlist = ["FAIL_SYM"]

        with patch.object(self.agent, '_fetch_market_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None
            await self.agent.run_cycle()
            mock_fetch.assert_called_once_with("FAIL_SYM")
            # add_market_insight should not have been called for FAIL_SYM
            self.mock_knowledge_base.add_market_insight.assert_not_called()
        # No need to call self.agent.stop() if agent was not started.


if __name__ == '__main__':
    unittest.main()
