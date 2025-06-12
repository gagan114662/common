import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from tier3_evolution.evolution_engine import EvolutionEngine
from tier2_strategy.strategy_tester import StrategyPerformance # Assuming StrategyPerformance is in strategy_tester
from config.settings import SYSTEM_CONFIG, EvolutionConfig, PerformanceTargets
# Mock dependencies for EvolutionEngine constructor
from tier2_strategy.strategy_generator import StrategyGenerator
from tier2_strategy.strategy_tester import StrategyTester


class TestEvolutionEngine(unittest.TestCase):

    def setUp(self):
        # Mock dependencies for EvolutionEngine
        self.mock_strategy_generator = MagicMock(spec=StrategyGenerator)
        self.mock_strategy_tester = MagicMock(spec=StrategyTester)

        # Default EvolutionConfig
        self.evolution_config = EvolutionConfig(
            population_size=50,
            generations=10, # Reduced for testing
            mutation_rate=0.1,
            crossover_rate=0.8,
            selection_pressure=2.0,
            elitism_rate=0.1
        )

        self.engine = EvolutionEngine(
            strategy_generator=self.mock_strategy_generator,
            strategy_tester=self.mock_strategy_tester,
            evolution_config=self.evolution_config
        )

    def create_mock_performance(self, cagr, sharpe, drawdown, avg_profit_trade, win_rate=0.55):
        # Helper to create StrategyPerformance instances
        return StrategyPerformance(
            strategy_id="s_test", template_name="t_test", category="c_test",
            cagr=cagr, sharpe_ratio=sharpe, max_drawdown=drawdown,
            average_profit_per_trade=avg_profit_trade,
            win_rate=win_rate, # Ensure this doesn't cause meets_targets to fail if we use it
            # Default other values
            volatility=0.1, sortino_ratio=0, var_95=0, cvar_95=0, beta=0, alpha=0,
            total_trades=10, avg_win=0, avg_loss=0, profit_factor=0, calmar_ratio=0,
            information_ratio=0, treynor_ratio=0, downside_deviation=0, excess_return=0,
            tracking_error=0, risk_adjusted_score=0, fitness_score=0,
            backtest_duration=0, test_timestamp=datetime.now()
        )

    @patch('tier3_evolution.evolution_engine.SYSTEM_CONFIG')
    def test_calculate_fitness(self, mock_system_config):
        # Define mocked performance targets
        mock_targets = PerformanceTargets(
            target_cagr=0.20,
            target_sharpe=1.0,
            max_drawdown=0.20, # Target is 20%
            target_average_profit_per_trade=0.0050, # 0.5%
            risk_free_rate=0.02 # Not directly used in _calculate_fitness, but good to set
        )
        mock_system_config.performance = mock_targets

        # Scenario 1: Baseline performance (below targets)
        perf_baseline = self.create_mock_performance(
            cagr=0.10, sharpe=0.5, drawdown=0.10, avg_profit_trade=0.0025
        )
        fitness_baseline = self.engine._calculate_fitness(perf_baseline)

        # Expected scores for baseline:
        # cagr_score = min(0.10 / 0.20, 1.0) = 0.5
        # sharpe_score = min(0.5 / 1.0, 1.0) = 0.5
        # avg_profit_score = min(0.0025 / 0.0050, 1.0) = 0.5
        # drawdown_score = max(0.0, (0.20 - 0.10) / 0.20) = 0.5
        # fitness = (0.5*0.2) + (0.5*0.3) + (0.5*0.3) + (0.5*0.2) = 0.1 + 0.15 + 0.15 + 0.1 = 0.5
        self.assertAlmostEqual(fitness_baseline, 0.5)


        # Scenario 2: All targets met - bonus should apply
        perf_all_met = self.create_mock_performance(
            cagr=0.25, sharpe=1.2, drawdown=0.15, avg_profit_trade=0.0060
        )
        # cagr_score = min(0.25 / 0.20, 1.0) = 1.0
        # sharpe_score = min(1.2 / 1.0, 1.0) = 1.0
        # avg_profit_score = min(0.0060 / 0.0050, 1.0) = 1.0
        # drawdown_score = max(0.0, (0.20 - 0.15) / 0.20) = 0.25
        # base_fitness = (1.0*0.2) + (1.0*0.3) + (1.0*0.3) + (0.25*0.2)
        #              = 0.2 + 0.3 + 0.3 + 0.05 = 0.85
        # Bonus: 0.85 * 1.2 = 1.02
        expected_fitness_bonus = 0.85 * 1.2
        fitness_all_met = self.engine._calculate_fitness(perf_all_met)
        self.assertAlmostEqual(fitness_all_met, expected_fitness_bonus)

        # Scenario 3: High drawdown (> 0.30) - penalty should apply
        perf_high_drawdown = self.create_mock_performance(
            cagr=0.25, sharpe=1.2, drawdown=0.35, avg_profit_trade=0.0060
        )
        # Base fitness calculation (drawdown_score will be 0 due to max_drawdown > target_max_drawdown)
        # cagr_score = 1.0
        # sharpe_score = 1.0
        # avg_profit_score = 1.0
        # drawdown_score = max(0.0, (0.20 - 0.35) / 0.20) = 0.0
        # base_fitness = (1.0*0.2) + (1.0*0.3) + (1.0*0.3) + (0.0*0.2) = 0.2 + 0.3 + 0.3 + 0.0 = 0.8
        # Targets are met (assuming win_rate is ok), so bonus applies: 0.8 * 1.2 = 0.96
        # Then penalty: 0.96 * 0.5 = 0.48
        expected_fitness_penalty = (0.8 * 1.2) * 0.5
        # Let's re-evaluate the bonus condition:
        # cagr (0.25) >= target_cagr (0.20) -> True
        # sharpe (1.2) >= target_sharpe (1.0) -> True
        # max_drawdown (0.35) <= target_max_drawdown (0.20) -> False
        # So bonus does NOT apply.
        # base_fitness = 0.8
        # Penalty: 0.8 * 0.5 = 0.4
        expected_fitness_penalty_no_bonus = 0.8 * 0.5
        fitness_high_drawdown = self.engine._calculate_fitness(perf_high_drawdown)
        self.assertAlmostEqual(fitness_high_drawdown, expected_fitness_penalty_no_bonus)


        # Scenario 4: Vary one metric (e.g., average_profit_per_trade)
        # Baseline was 0.5 fitness with avg_profit_trade = 0.0025 (score 0.5)
        # Let's increase avg_profit_trade to meet target: 0.0050 (score 1.0)
        # cagr_score = 0.5 (0.10 / 0.20)
        # sharpe_score = 0.5 (0.5 / 1.0)
        # avg_profit_score = 1.0 (0.0050 / 0.0050)
        # drawdown_score = 0.5 ((0.20 - 0.10) / 0.20)
        # fitness_varied = (0.5*0.2) + (0.5*0.3) + (1.0*0.3) + (0.5*0.2)
        #                = 0.1 + 0.15 + 0.30 + 0.1 = 0.65
        perf_varied_profit = self.create_mock_performance(
            cagr=0.10, sharpe=0.5, drawdown=0.10, avg_profit_trade=0.0050
        )
        fitness_varied = self.engine._calculate_fitness(perf_varied_profit)
        self.assertAlmostEqual(fitness_varied, 0.65)
        # This shows increase from 0.5 to 0.65 due to avg_profit_score change from 0.5 to 1.0
        # Change in fitness = (1.0 - 0.5) * 0.30 (weight of avg_profit_score) = 0.5 * 0.30 = 0.15
        # Original fitness_baseline was 0.5. So 0.5 + 0.15 = 0.65. Correct.

        # Scenario 5: Zero target_avg_profit_trade (should not cause division by zero)
        mock_targets_zero_profit_target = PerformanceTargets(
            target_cagr=0.20, target_sharpe=1.0, max_drawdown=0.20,
            target_average_profit_per_trade=0.0, # Zero target
            risk_free_rate=0.02
        )
        mock_system_config.performance = mock_targets_zero_profit_target
        perf_zero_target = self.create_mock_performance(
            cagr=0.10, sharpe=0.5, drawdown=0.10, avg_profit_trade=0.0025
        )
        # cagr_score = 0.5
        # sharpe_score = 0.5
        # avg_profit_score = 0.0 (because target is 0)
        # drawdown_score = 0.5
        # fitness = (0.5*0.2) + (0.5*0.3) + (0.0*0.3) + (0.5*0.2) = 0.1 + 0.15 + 0.0 + 0.1 = 0.35
        fitness_zero_target = self.engine._calculate_fitness(perf_zero_target)
        self.assertAlmostEqual(fitness_zero_target, 0.35)


if __name__ == '__main__':
    unittest.main()
