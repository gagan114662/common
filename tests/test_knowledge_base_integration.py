import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import time # Keep for now, though uuid is used for db name
import pandas as pd
import numpy as np
import uuid # For unique DB names
import os # For DB cleanup
from pathlib import Path # For DB cleanup


from agents.knowledge_base import SharedKnowledgeBase, MarketInsight
from agents.trend_following_agent import TrendFollowingAgent
from agents.mean_reversion_agent import MeanReversionAgent
from config.settings import SYSTEM_CONFIG, TrendFollowingAgentSettings, MeanReversionAgentSettings
from tier2_strategy.strategy_generator import StrategyGenerator # For BaseAgent
from tier2_strategy.strategy_tester import StrategyTester     # For BaseAgent

class TestKnowledgeBaseIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Use a temporary, unique DB path for each test run to ensure isolation
        self.db_file_name = f"test_kb_integration_{uuid.uuid4().hex}.db" # Use uuid
        self.knowledge_base = SharedKnowledgeBase(db_path=Path(self.db_file_name)) # Use Path

        self.mock_strategy_generator = MagicMock(spec=StrategyGenerator)
        self.mock_strategy_tester = MagicMock(spec=StrategyTester)

        self.trend_agent_config = TrendFollowingAgentSettings(
            name="TestTrendAgentKB",
            category="trend_following", # Added category
            initial_symbols_watchlist=["SYM_A"],
            data_fetch_period="60d",
            sma_short_window=10,    # Ensure long > short
            sma_long_window=20,
            trend_strength_threshold=0.01,
            min_data_points_for_trend=20 # Ensure >= sma_long_window
        )

        self.mr_agent_config = MeanReversionAgentSettings(
            name="TestMRAgentKB",
            category="mean_reversion", # Added category
            initial_symbols_watchlist=["SYM_A"],
            data_fetch_period="60d",
            bollinger_window=20,
            bollinger_std_dev=2.0,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            min_volume_threshold=1000,
            opportunity_score_threshold=0.5
        )

        # Common patch for DASHBOARD used by BaseAgent and potentially agents
        self.dashboard_patcher = patch('agents.base_agent.DASHBOARD', new_callable=MagicMock)
        self.MockDashboard = self.dashboard_patcher.start()

        # Patch DASHBOARD in agent modules if they import it directly
        self.trend_dashboard_patcher = patch('agents.trend_following_agent.DASHBOARD', new=self.MockDashboard)
        self.trend_dashboard_patcher.start()

        self.mr_dashboard_patcher = patch('agents.mean_reversion_agent.DASHBOARD', new=self.MockDashboard)
        self.mr_dashboard_patcher.start()


        self.trend_agent = TrendFollowingAgent(
            knowledge_base=self.knowledge_base,
            config=self.trend_agent_config,
            strategy_generator=self.mock_strategy_generator,
            strategy_tester=self.mock_strategy_tester
        )

        self.mr_agent = MeanReversionAgent(
            knowledge_base=self.knowledge_base,
            config=self.mr_agent_config,
            strategy_generator=self.mock_strategy_generator,
            strategy_tester=self.mock_strategy_tester
        )

        # Mock agents' internal loops to prevent them from running automatically
        self.trend_agent._main_loop = AsyncMock()
        self.mr_agent._main_loop = AsyncMock()

        # Set agents to active state for run_cycle logic as if they were started
        # BaseAgent.initialize() sets status to "initializing", then "idle"
        # BaseAgent.start() sets status to "running"
        # For these tests, we are not calling start() but directly run_cycle()
        # So, manually setting a state that allows run_cycle to proceed if it checks.
        # The current run_cycle implementations don't strictly check self.state.is_active.
        self.trend_agent.state.is_active = True # Mimic agent being started
        self.trend_agent.state.status = "idle"
        self.mr_agent.state.is_active = True   # Mimic agent being started
        self.mr_agent.state.status = "idle"


    async def asyncTearDown(self):
        self.dashboard_patcher.stop()
        self.trend_dashboard_patcher.stop()
        self.mr_dashboard_patcher.stop()

        # Attempt to clean up the database file
        if hasattr(self, 'knowledge_base') and self.knowledge_base and hasattr(self.knowledge_base, 'engine'):
            self.knowledge_base.engine.dispose()

        if hasattr(self, 'db_file_name') and os.path.exists(self.db_file_name):
            try:
                os.remove(self.db_file_name)
                # print(f"Cleaned up test database: {self.db_file_name}") # For debugging
            except Exception as e:
                print(f"Error cleaning up test database {self.db_file_name}: {e}")


    def _generate_data(self, length=100, base_price=100.0, trend_slope=0.0, vol=1000000.0): # Use floats
        dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq='B')
        data = pd.DataFrame(index=dates)
        noise = np.random.randn(length) * 0.5
        data['Close'] = base_price + trend_slope * np.arange(length) + noise.cumsum()
        data['Open'] = data['Close'] - np.random.rand(length) * 0.1 + noise # Open related to close
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.rand(length) * 0.2
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.rand(length) * 0.2
        data['Volume'] = vol + np.random.randint(-vol*0.1, vol*0.1, size=length)
        data['Volume'] = data['Volume'].clip(lower=0)
        return data


    async def test_trend_agent_publishes_insight_retrievable_by_type(self):
        min_len = self.trend_agent.config.min_data_points_for_trend
        mock_trend_data = self._generate_data(length=min_len + 20, base_price=100, trend_slope=0.5) # Strong uptrend data

        with patch.object(self.trend_agent, '_fetch_market_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_trend_data
            await self.trend_agent.run_cycle()

        insights = self.knowledge_base.get_insights_by_type("trend_following_uptrend", max_age_seconds=60)
        self.assertGreaterEqual(len(insights), 1, "Should find at least one uptrend insight")
        self.assertEqual(insights[0].symbols[0], "SYM_A") # Trend agent uses symbols from its watchlist
        self.assertEqual(insights[0].category, "trend_following_uptrend")
        self.assertGreater(insights[0].confidence, 0.5) # Expect decent score for clear trend


    async def test_mr_agent_publishes_insight_retrievable_by_symbol_type(self):
        mr_data_len = self.mr_agent.config.bollinger_window + self.mr_agent.config.rsi_period + 20
        # Create data that first goes up (to set up bands) then drops sharply to trigger buy
        prices = list(np.linspace(100, 110, mr_data_len - 10)) + list(np.linspace(110, 80, 10))
        mock_mr_data = self._generate_data(length=mr_data_len, base_price=100) # Base data
        mock_mr_data['Close'] = prices # Override close to force signal
        mock_mr_data['Volume'] = self.mr_agent.config.min_volume_threshold * 2 # Ensure volume

        with patch.object(self.mr_agent, '_fetch_market_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_mr_data
            await self.mr_agent.run_cycle()

        latest_insight = self.knowledge_base.get_latest_insight_by_symbol_type("SYM_A", "mean_reversion_buy", max_age_seconds=60)
        self.assertIsNotNone(latest_insight, "Mean reversion buy insight should be found")
        if latest_insight: # To satisfy mypy
            self.assertEqual(latest_insight.symbols[0], "SYM_A")
            self.assertEqual(latest_insight.category, "mean_reversion_buy")
            self.assertGreaterEqual(latest_insight.confidence, self.mr_agent.config.opportunity_score_threshold)
            self.assertEqual(latest_insight.supporting_data['opportunity_type'], "buy_reversion")


    async def test_insight_timestamp_and_expiry(self):
        test_symbol = "SYM_B"
        self.trend_agent.symbols_watchlist = [test_symbol]

        min_len = self.trend_agent.config.min_data_points_for_trend
        mock_trend_data = self._generate_data(length=min_len + 20, base_price=100, trend_slope=0.5)

        with patch.object(self.trend_agent, '_fetch_market_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_trend_data
            await self.trend_agent.run_cycle()

        insights_cat = f"trend_following_uptrend" # Based on data and agent logic
        insights = self.knowledge_base.get_insights_by_type(insights_cat, max_age_seconds=60)
        self.assertGreaterEqual(len(insights), 1)

        await asyncio.sleep(0.1)
        stale_insights = self.knowledge_base.get_insights_by_type(insights_cat, max_age_seconds=0.05)
        self.assertEqual(len(stale_insights), 0, "Insight should be stale with very short max_age_seconds")

        fresh_insights = self.knowledge_base.get_insights_by_type(insights_cat, max_age_seconds=120)
        self.assertGreaterEqual(len(fresh_insights), 1, "Insight should be fresh with longer max_age_seconds")


    async def test_multiple_insights_for_symbol_retrieval(self):
        symbol = "SYM_MULTI"
        self.trend_agent.symbols_watchlist = [symbol]
        self.mr_agent.symbols_watchlist = [symbol]

        # Trend Agent publishes first (uptrend)
        min_len_trend = self.trend_agent.config.min_data_points_for_trend
        mock_trend_data = self._generate_data(length=min_len_trend + 20, base_price=100, trend_slope=0.5)
        with patch.object(self.trend_agent, '_fetch_market_data', new_callable=AsyncMock) as mock_fetch_trend:
            mock_fetch_trend.return_value = mock_trend_data
            await self.trend_agent.run_cycle()

        await asyncio.sleep(0.01)

        # MR Agent publishes next (buy_reversion)
        mr_data_len = self.mr_agent.config.bollinger_window + self.mr_agent.config.rsi_period + 20
        prices_mr = list(np.linspace(120, 130, mr_data_len - 10)) + list(np.linspace(130, 100, 10)) # Goes up then drops
        mock_mr_data = self._generate_data(length=mr_data_len, base_price=120)
        mock_mr_data['Close'] = prices_mr
        mock_mr_data['Volume'] = self.mr_agent.config.min_volume_threshold * 2

        with patch.object(self.mr_agent, '_fetch_market_data', new_callable=AsyncMock) as mock_fetch_mr:
            mock_fetch_mr.return_value = mock_mr_data
            await self.mr_agent.run_cycle()

        all_insights_for_symbol = self.knowledge_base.get_all_insights_for_symbol(symbol, max_age_seconds=60)
        self.assertEqual(len(all_insights_for_symbol), 2, f"Expected 2 insights, got {len(all_insights_for_symbol)}")

        categories_found = {insight.category for insight in all_insights_for_symbol}
        self.assertIn("trend_following_uptrend", categories_found)
        self.assertIn("mean_reversion_buy", categories_found)

        latest_trend = self.knowledge_base.get_latest_insight_by_symbol_type(symbol, "trend_following_uptrend", 60)
        latest_mr = self.knowledge_base.get_latest_insight_by_symbol_type(symbol, "mean_reversion_buy", 60)

        self.assertIsNotNone(latest_trend)
        self.assertIsNotNone(latest_mr)
        self.assertTrue(hasattr(latest_mr, 'timestamp') and hasattr(latest_trend, 'timestamp'), "Insights missing timestamp")
        if latest_mr and latest_trend: # for type checker
             self.assertGreater(latest_mr.timestamp, latest_trend.timestamp)


if __name__ == '__main__':
    unittest.main()
