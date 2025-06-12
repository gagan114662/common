import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

from agents.mean_reversion_agent import MeanReversionAgent
from agents.knowledge_base import SharedKnowledgeBase, MarketInsight
from config.settings import SYSTEM_CONFIG, AgentConfigs, MeanReversionAgentSettings
from tier2_strategy.strategy_generator import StrategyGenerator # For BaseAgent
from tier2_strategy.strategy_tester import StrategyTester     # For BaseAgent

class TestMeanReversionAgent(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.mock_knowledge_base = AsyncMock(spec=SharedKnowledgeBase)

        # Use a concrete instance of the settings dataclass
        self.agent_config = MeanReversionAgentSettings(
            name="TestMeanReversionAgent",
            # version="1.0", # Not used by agent __init__ directly, but part of settings
            # description="Test MR Agent", # Not used by agent __init__ directly
            initial_symbols_watchlist=["TEST.SYM"],
            data_fetch_period="60d",
            bollinger_window=20,
            bollinger_std_dev=2.0,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            min_volume_threshold=100000,
            opportunity_score_threshold=0.6,
            # category will be set by agent's default or can be added here if MeanReversionAgentSettings has it
            category="mean_reversion" # Ensure this matches agent's internal base_config
        )

        # Mock base agent dependencies
        self.mock_strategy_generator = MagicMock(spec=StrategyGenerator)
        self.mock_strategy_tester = MagicMock(spec=StrategyTester)

        # Patch Dashboard - target where it's imported by the modules under test
        self.base_dashboard_patcher = patch('agents.base_agent.DASHBOARD', new_callable=MagicMock)
        self.MockBaseDashboard = self.base_dashboard_patcher.start()

        self.mr_dashboard_patcher = patch('agents.mean_reversion_agent.DASHBOARD', new_callable=MagicMock)
        self.MockMRDashboard = self.mr_dashboard_patcher.start()

        # Patch yfinance Ticker
        self.ticker_patcher = patch('agents.mean_reversion_agent.yf.Ticker')
        self.MockTicker = self.ticker_patcher.start()

        self.agent = MeanReversionAgent(
            knowledge_base=self.mock_knowledge_base,
            config=self.agent_config,
            strategy_generator=self.mock_strategy_generator,
            strategy_tester=self.mock_strategy_tester
        )
        # Agent state is managed by BaseAgent. Agent starts in "stopped", "is_active=False".
        # For tests that don't call start(), manually activate if needed for cycle methods.
        # self.agent.state.is_active = True
        # self.agent.state.status = "idle" # or running if cycle implies agent is fully active


    async def asyncTearDown(self):
        self.base_dashboard_patcher.stop()
        self.mr_dashboard_patcher.stop()
        self.ticker_patcher.stop()
        if hasattr(self.agent, 'state') and self.agent.state.is_active:
            await self.agent.stop()


    def _generate_test_data(self, length=60, base_price=100.0, vol=1000000.0, date_start_offset_days=0):
        # Adding float to base_price and vol for more realistic calculations
        dates = pd.date_range(end=pd.Timestamp.now() - pd.Timedelta(days=date_start_offset_days), periods=length, freq='B') # Business days
        data = pd.DataFrame(index=dates)
        data['Open'] = base_price + np.random.randn(length).cumsum()
        data['High'] = data['Open'] + np.random.rand(length) * 2 # Slightly larger range
        data['Low'] = data['Open'] - np.random.rand(length) * 2
        data['Close'] = data['Open'] + np.random.randn(length) * 0.5
        data['Volume'] = vol + np.random.randint(-int(vol*0.1), int(vol*0.1), size=length)
        data.loc[data['Volume'] < 0, 'Volume'] = 0 # Ensure volume is not negative
        return data

    async def test_agent_initialization(self):
        self.assertEqual(self.agent.knowledge_base, self.mock_knowledge_base)
        self.assertEqual(self.agent.config, self.agent_config) # Agent's self.config is MeanReversionAgentSettings
        # BaseAgent methods use self.config.name. Since MeanReversionAgent sets self.config
        # to MeanReversionAgentSettings, BaseAgent methods will use MeanReversionAgentSettings.name.
        self.assertEqual(self.agent.config.name, self.agent_config.name)
        self.assertFalse(self.agent.state.is_active) # is_active is False by default after BaseAgent init
        self.assertEqual(self.agent.state.status, "stopped") # Default status from AgentState


    async def test_fetch_market_data_success(self):
        mock_ticker_instance = self.MockTicker.return_value
        test_df = self._generate_test_data()
        mock_ticker_instance.history.return_value = test_df

        data = await self.agent._fetch_market_data("TEST.SYM")

        self.assertIsNotNone(data)
        pd.testing.assert_frame_equal(data, test_df)
        self.MockTicker.assert_called_with("TEST.SYM")
        mock_ticker_instance.history.assert_called_with(period=self.agent.config.data_fetch_period)

    async def test_fetch_market_data_failure(self):
        self.MockTicker.return_value.history.side_effect = Exception("API Error")
        data = await self.agent._fetch_market_data("FAIL.SYM")
        self.assertIsNone(data)
        self.MockMRDashboard.log_agent_activity.assert_any_call( # Log from MR Agent's _fetch_market_data
            self.agent.config.name, "Error fetching data for FAIL.SYM: API Error", {"symbol": "FAIL.SYM"}
        )

    async def test_calculate_indicators_sufficient_data(self):
        # Test data length should be enough for all indicators to have non-NaN values at the end
        length = max(self.agent.config.bollinger_window, self.agent.config.rsi_period) + 5
        data = self._generate_test_data(length=length)
        data_with_indicators = self.agent._calculate_indicators(data.copy(), "TEST.SYM")

        self.assertIsNotNone(data_with_indicators)
        self.assertIn('SMA', data_with_indicators)
        self.assertIn('UpperBand', data_with_indicators)
        self.assertIn('LowerBand', data_with_indicators)
        self.assertIn('RSI', data_with_indicators)

        # Last row should have valid indicator values
        self.assertFalse(data_with_indicators[['SMA', 'UpperBand', 'LowerBand', 'RSI']].iloc[-1].isnull().any())


    async def test_calculate_indicators_insufficient_data(self):
        data = self._generate_test_data(length=self.agent.config.bollinger_window - 5)
        data_with_indicators = self.agent._calculate_indicators(data.copy(), "TEST.SYM")
        self.assertIsNone(data_with_indicators)

    async def test_identify_mean_reversion_opportunity_buy(self):
        # Generate data and calculate indicators first
        data_orig = self._generate_test_data(length=100, base_price=100)
        data_with_indicators = self.agent._calculate_indicators(data_orig.copy(), "TEST.SYM")
        self.assertIsNotNone(data_with_indicators) # Ensure indicators were calculated

        # Use the last row for modification
        last_row = data_with_indicators.iloc[-1].copy()
        if pd.isna(last_row['LowerBand']) or pd.isna(last_row['RSI']):
            self.skipTest("Indicators are NaN, cannot force buy signal reliably. Increase data length or check calculations.")

        last_row['Close'] = last_row['LowerBand'] * 0.95 # Made price further from band
        last_row['RSI'] = self.agent.config.rsi_oversold * 0.5 # Made RSI more extreme
        last_row['Volume'] = self.agent.config.min_volume_threshold * 2

        opportunity = self.agent._identify_opportunity(last_row, "TEST.SYM")

        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity['symbol'], "TEST.SYM")
        self.assertEqual(opportunity['opportunity_type'], "buy_reversion")
        self.assertGreaterEqual(opportunity['score'], self.agent.config.opportunity_score_threshold)

    async def test_identify_mean_reversion_opportunity_sell(self):
        data_orig = self._generate_test_data(length=100, base_price=100)
        data_with_indicators = self.agent._calculate_indicators(data_orig.copy(), "TEST.SYM")
        self.assertIsNotNone(data_with_indicators)

        last_row = data_with_indicators.iloc[-1].copy()
        if pd.isna(last_row['UpperBand']) or pd.isna(last_row['RSI']):
             self.skipTest("Indicators are NaN, cannot force sell signal reliably. Increase data length or check calculations.")

        last_row['Close'] = last_row['UpperBand'] * 1.05 # Made price further from band
        last_row['RSI'] = self.agent.config.rsi_overbought + (100.0 - self.agent.config.rsi_overbought) * 0.5 # Made RSI more extreme
        last_row['Volume'] = self.agent.config.min_volume_threshold * 2

        opportunity = self.agent._identify_opportunity(last_row, "TEST.SYM")

        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity['symbol'], "TEST.SYM")
        self.assertEqual(opportunity['opportunity_type'], "sell_reversion")
        self.assertGreaterEqual(opportunity['score'], self.agent.config.opportunity_score_threshold)

    async def test_identify_mean_reversion_opportunity_no_signal(self):
        data_orig = self._generate_test_data(length=100)
        data_with_indicators = self.agent._calculate_indicators(data_orig.copy(), "TEST.SYM")
        self.assertIsNotNone(data_with_indicators)

        last_row = data_with_indicators.iloc[-1].copy()
        if pd.isna(last_row['SMA']) or pd.isna(last_row['RSI']):
            self.skipTest("Indicators are NaN, cannot force no_signal reliably.")

        last_row['Close'] = last_row['SMA']
        last_row['RSI'] = 50
        last_row['Volume'] = self.agent.config.min_volume_threshold * 2

        opportunity = self.agent._identify_opportunity(last_row, "TEST.SYM")
        self.assertIsNone(opportunity)

    async def test_identify_mean_reversion_opportunity_low_volume(self):
        data_orig = self._generate_test_data(length=100)
        data_with_indicators = self.agent._calculate_indicators(data_orig.copy(), "TEST.SYM")
        self.assertIsNotNone(data_with_indicators)

        last_row = data_with_indicators.iloc[-1].copy()
        if pd.isna(last_row['LowerBand']) or pd.isna(last_row['RSI']):
            self.skipTest("Indicators are NaN, cannot force low_volume check reliably.")

        last_row['Close'] = last_row['LowerBand'] * 0.98
        last_row['RSI'] = self.agent.config.rsi_oversold - 5
        last_row['Volume'] = self.agent.config.min_volume_threshold / 2

        opportunity = self.agent._identify_opportunity(last_row, "TEST.SYM")
        self.assertIsNone(opportunity)

    @patch('agents.mean_reversion_agent.MeanReversionAgent._fetch_market_data', new_callable=AsyncMock)
    @patch('agents.mean_reversion_agent.MeanReversionAgent._calculate_indicators')
    @patch('agents.mean_reversion_agent.MeanReversionAgent._identify_opportunity')
    async def test_run_cycle_full_flow(self, mock_identify_opportunity, mock_calculate_indicators, mock_fetch_market_data):
        # Agent should not be running its own loop for this direct test of run_cycle
        # self.agent.state.is_active = True # Set in setup or ensure not needed if not starting agent

        self.agent.symbols_watchlist = ["SYM1", "SYM2"]

        df1 = self._generate_test_data(length=100) # Ensure enough data for indicators
        df2 = self._generate_test_data(length=100)
        mock_fetch_market_data.side_effect = [df1, df2]

        # Ensure mocked calculate_indicators returns a DataFrame with a valid last row
        # and necessary indicator columns.
        df1_indicators = df1.copy()
        df1_indicators['SMA'] = 100.0
        df1_indicators['UpperBand'] = 105.0
        df1_indicators['LowerBand'] = 95.0
        df1_indicators['RSI'] = 25.0
        # Ensure Volume is present as _identify_opportunity checks it
        if 'Volume' not in df1_indicators: df1_indicators['Volume'] = self.agent_config.min_volume_threshold * 2
        if 'Close' not in df1_indicators: df1_indicators['Close'] = 100.0 # Needed by _identify_opportunity

        df2_indicators = df2.copy()
        df2_indicators['SMA'] = 200.0
        df2_indicators['UpperBand'] = 210.0
        df2_indicators['LowerBand'] = 190.0
        df2_indicators['RSI'] = 75.0
        if 'Volume' not in df2_indicators: df2_indicators['Volume'] = self.agent_config.min_volume_threshold * 2
        if 'Close' not in df2_indicators: df2_indicators['Close'] = 200.0


        mock_calculate_indicators.side_effect = [df1_indicators, df2_indicators]

        opportunity1 = {"symbol": "SYM1", "opportunity_type": "buy_reversion", "score": 0.8, "price": 95.0,
                        "indicators": {"sma": 100.0, 'close': 95.0, 'lower_band': 95.0, 'upper_band': 105.0, 'rsi': 25.0, 'volume': 200000}}
        opportunity2 = {"symbol": "SYM2", "opportunity_type": "sell_reversion", "score": 0.7, "price": 210.0, "indicators": {"sma": 200.0}}
        mock_identify_opportunity.side_effect = [opportunity1, opportunity2]

        await self.agent.run_cycle()

        self.assertEqual(mock_fetch_market_data.call_count, 2)
        self.assertEqual(mock_calculate_indicators.call_count, 2)
        self.assertEqual(mock_identify_opportunity.call_count, 2)

        self.assertEqual(self.mock_knowledge_base.add_market_insight.call_count, 2)

        call_args_list = self.mock_knowledge_base.add_market_insight.call_args_list

        insight1_obj = next(c[0][0] for c in call_args_list if c[0][0].symbols == ["SYM1"])
        insight2_obj = next(c[0][0] for c in call_args_list if c[0][0].symbols == ["SYM2"])

        self.assertEqual(insight1_obj.supporting_data['opportunity_type'], "buy_reversion")
        self.assertEqual(insight1_obj.category, "mean_reversion_buy") # Check category used in insight
        self.assertEqual(insight2_obj.supporting_data['opportunity_type'], "sell_reversion")
        self.assertEqual(insight2_obj.category, "mean_reversion_sell") # Check category used in insight


if __name__ == '__main__':
    unittest.main()
