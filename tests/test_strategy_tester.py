import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from tier2_strategy.strategy_tester import StrategyPerformance
from tier2_strategy.strategy_generator import GeneratedStrategy, StrategyTemplate, ParameterRange
from tier1_core.quantconnect_client import BacktestResult
from config.settings import SYSTEM_CONFIG, PerformanceTargets

class TestStrategyPerformance(unittest.TestCase):

    def create_mock_generated_strategy(self):
        mock_template = StrategyTemplate(
            template_id="test_template",
            name="Test Template",
            description="A test template",
            category="test",
            code_template="class TestStrategy(QCAlgorithm): pass",
            parameters=[
                ParameterRange(name="param1", param_type="int", min_value=1, max_value=10, step=1, default=5)
            ],
            asset_class="Equity",
            timeframe="Daily"
        )
        return GeneratedStrategy.create(
            template=mock_template,
            parameters={"param1": 7},
            asset_class="Equity",
            timeframe="Daily"
        )

    @patch('tier2_strategy.strategy_tester.SYSTEM_CONFIG')
    def test_strategy_performance_calculation(self, mock_system_config):
        # Mock SYSTEM_CONFIG.performance
        mock_performance_targets = PerformanceTargets(
            target_cagr=0.25,
            target_sharpe=1.0,
            max_drawdown=0.20,
            min_win_rate=0.55,
            max_volatility=0.20,
            target_average_profit_per_trade=0.0075,
            risk_free_rate=0.05
        )
        mock_system_config.performance = mock_performance_targets

        mock_strategy = self.create_mock_generated_strategy()

        mock_qc_result = BacktestResult(
            is_successful=True,
            backtest_id="test_backtest_123",
            statistics={
                "TotalPerformance": {
                    "CompoundAnnualReturn": 0.30,  # 30% CAGR
                    "StandardDeviation": 0.15,     # Volatility
                    "SharpeRatio": 1.5,            # Original Sharpe, will be recalculated
                    "TotalTrades": 100,
                    "PortfolioStatistics": {
                        "TotalNetProfit": 1500.0
                    }
                }
            },
            charts={},
            orders=[],
            logs="",
            error=None
        )

        performance = StrategyPerformance.from_backtest_result(mock_strategy, mock_qc_result)

        # Test average_profit_per_trade
        expected_avg_profit = 1500.0 / 100
        self.assertAlmostEqual(performance.average_profit_per_trade, expected_avg_profit)

        # Test Sharpe Ratio calculation
        # sharpe = (cagr - risk_free_rate) / volatility
        cagr = 0.30
        risk_free_rate = 0.05
        volatility = 0.15
        expected_sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0.0
        self.assertAlmostEqual(performance.sharpe_ratio, expected_sharpe)

        # Test other fields are populated
        self.assertEqual(performance.strategy_id, mock_strategy.strategy_id)
        self.assertEqual(performance.cagr, cagr)
        self.assertEqual(performance.volatility, volatility)
        self.assertEqual(performance.total_trades, 100)

    @patch('tier2_strategy.strategy_tester.SYSTEM_CONFIG')
    def test_strategy_performance_meets_targets(self, mock_system_config):
        # Mock SYSTEM_CONFIG.performance
        mock_targets = PerformanceTargets(
            target_cagr=0.20,
            target_sharpe=0.8,
            max_drawdown=0.25,
            min_win_rate=0.50, # This is also part of PerformanceTargets
            max_volatility=0.30, # This is also part of PerformanceTargets
            target_average_profit_per_trade=0.0050,
            risk_free_rate=0.02
        )
        mock_system_config.performance = mock_targets

        # Scenario 1: All targets met
        perf_meets = StrategyPerformance(
            strategy_id="s1", template_name="t1", category="c1",
            cagr=0.25, sharpe_ratio=1.0, max_drawdown=0.15,
            volatility=0.18, win_rate=0.60, average_profit_per_trade=0.0060,
            # Other fields can be default/zero for this test
            sortino_ratio=0, var_95=0, cvar_95=0, beta=0, alpha=0, total_trades=10,
            avg_win=0, avg_loss=0, profit_factor=0, calmar_ratio=0, information_ratio=0,
            treynor_ratio=0, downside_deviation=0, excess_return=0, tracking_error=0,
            risk_adjusted_score=0, fitness_score=0, backtest_duration=0, test_timestamp=datetime.now()
        )
        self.assertTrue(perf_meets.meets_targets(mock_system_config.performance))

        # Scenario 2: Average profit per trade not met
        perf_fail_avg_profit = StrategyPerformance(
            strategy_id="s2", template_name="t1", category="c1",
            cagr=0.25, sharpe_ratio=1.0, max_drawdown=0.15,
            volatility=0.18, win_rate=0.60, average_profit_per_trade=0.0040, # Fails here
            sortino_ratio=0, var_95=0, cvar_95=0, beta=0, alpha=0, total_trades=10,
            avg_win=0, avg_loss=0, profit_factor=0, calmar_ratio=0, information_ratio=0,
            treynor_ratio=0, downside_deviation=0, excess_return=0, tracking_error=0,
            risk_adjusted_score=0, fitness_score=0, backtest_duration=0, test_timestamp=datetime.now()
        )
        self.assertFalse(perf_fail_avg_profit.meets_targets(mock_system_config.performance))

        # Scenario 3: CAGR not met
        perf_fail_cagr = StrategyPerformance(
            strategy_id="s3", template_name="t1", category="c1",
            cagr=0.15, sharpe_ratio=1.0, max_drawdown=0.15, # Fails here
            volatility=0.18, win_rate=0.60, average_profit_per_trade=0.0060,
            sortino_ratio=0, var_95=0, cvar_95=0, beta=0, alpha=0, total_trades=10,
            avg_win=0, avg_loss=0, profit_factor=0, calmar_ratio=0, information_ratio=0,
            treynor_ratio=0, downside_deviation=0, excess_return=0, tracking_error=0,
            risk_adjusted_score=0, fitness_score=0, backtest_duration=0, test_timestamp=datetime.now()
        )
        self.assertFalse(perf_fail_cagr.meets_targets(mock_system_config.performance))

        # Scenario 4: Max drawdown exceeded
        perf_fail_drawdown = StrategyPerformance(
            strategy_id="s4", template_name="t1", category="c1",
            cagr=0.25, sharpe_ratio=1.0, max_drawdown=0.30, # Fails here
            volatility=0.18, win_rate=0.60, average_profit_per_trade=0.0060,
            sortino_ratio=0, var_95=0, cvar_95=0, beta=0, alpha=0, total_trades=10,
            avg_win=0, avg_loss=0, profit_factor=0, calmar_ratio=0, information_ratio=0,
            treynor_ratio=0, downside_deviation=0, excess_return=0, tracking_error=0,
            risk_adjusted_score=0, fitness_score=0, backtest_duration=0, test_timestamp=datetime.now()
        )
        self.assertFalse(perf_fail_drawdown.meets_targets(mock_system_config.performance))

        # Scenario 5: Sharpe ratio not met
        perf_fail_sharpe = StrategyPerformance(
            strategy_id="s5", template_name="t1", category="c1",
            cagr=0.25, sharpe_ratio=0.7, max_drawdown=0.15, # Fails here
            volatility=0.18, win_rate=0.60, average_profit_per_trade=0.0060,
            sortino_ratio=0, var_95=0, cvar_95=0, beta=0, alpha=0, total_trades=10,
            avg_win=0, avg_loss=0, profit_factor=0, calmar_ratio=0, information_ratio=0,
            treynor_ratio=0, downside_deviation=0, excess_return=0, tracking_error=0,
            risk_adjusted_score=0, fitness_score=0, backtest_duration=0, test_timestamp=datetime.now()
        )
        self.assertFalse(perf_fail_sharpe.meets_targets(mock_system_config.performance))

        # Scenario 6: Win rate not met
        perf_fail_win_rate = StrategyPerformance(
            strategy_id="s6", template_name="t1", category="c1",
            cagr=0.25, sharpe_ratio=1.0, max_drawdown=0.15,
            volatility=0.18, win_rate=0.40, average_profit_per_trade=0.0060, # Fails here
            sortino_ratio=0, var_95=0, cvar_95=0, beta=0, alpha=0, total_trades=10,
            avg_win=0, avg_loss=0, profit_factor=0, calmar_ratio=0, information_ratio=0,
            treynor_ratio=0, downside_deviation=0, excess_return=0, tracking_error=0,
            risk_adjusted_score=0, fitness_score=0, backtest_duration=0, test_timestamp=datetime.now()
        )
        self.assertFalse(perf_fail_win_rate.meets_targets(mock_system_config.performance))


if __name__ == '__main__':
    unittest.main()
