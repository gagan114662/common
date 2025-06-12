"""
TIER 2: Strategy Testing System
High-performance backtesting system with parallel execution and performance optimization
Based on Gemini's recommendations for 30-minute backtesting over 15 years
"""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import json
import statistics

from tier1_core.logger import get_logger, PERF_LOGGER
from tier1_core.quantconnect_client import QuantConnectClient, BacktestResult
from tier2_strategy.strategy_generator import GeneratedStrategy
from config.settings import BacktestConfig, PerformanceTargets, SYSTEM_CONFIG
from tier1_core.real_time_dashboard import DASHBOARD
from tier1_core.strategy_memory import get_strategy_memory

@dataclass
class StrategyPerformance:
    """Comprehensive strategy performance metrics"""
    strategy_id: str
    template_name: str
    category: str
    
    # Core performance metrics
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    beta: float
    alpha: float
    
    # Trade statistics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Additional metrics
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    downside_deviation: float
    
    # Benchmark comparison
    excess_return: float
    tracking_error: float
    
    # Risk-adjusted scores
    risk_adjusted_score: float
    fitness_score: float
    
    # Execution details
    backtest_duration: float
    test_timestamp: datetime
    average_profit_per_trade: float
    
    @classmethod
    def from_backtest_result(cls, strategy: GeneratedStrategy, result: BacktestResult, benchmark_return: float = 0.10) -> "StrategyPerformance":
        """Create performance metrics from backtest result"""
        try:
            stats = result.statistics.get("TotalPerformance", {}) if result.statistics else {}
            risk_free_rate = SYSTEM_CONFIG.performance.risk_free_rate
            
            # Extract basic metrics
            cagr = stats.get("CompoundAnnualReturn", 0.0)
            volatility = stats.get("TotalPerformance", {}).get("StandardDeviation", 0.0) # Ensure volatility is defined before sharpe
            sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0.0
            drawdown = abs(stats.get("Drawdown", 1.0))
            
            # Calculate additional metrics
            total_trades = stats.get("TotalTrades", 0)
            portfolio_stats = stats.get("PortfolioStatistics", {})
            total_net_profit = portfolio_stats.get("TotalNetProfit", 0.0)
            avg_profit_trade = total_net_profit / total_trades if total_trades > 0 else 0.0
            win_rate = stats.get("WinRate", 0.0) / 100.0 if stats.get("WinRate") else 0.0
            
            # Risk-adjusted scoring
            risk_adjusted_score = cls._calculate_risk_adjusted_score(cagr, sharpe, drawdown)
            fitness_score = cls._calculate_fitness_score(cagr, sharpe, drawdown, volatility)
            
            return cls(
                strategy_id=strategy.strategy_id,
                template_name=strategy.template_name,
                category=strategy.category,
                cagr=cagr,
                sharpe_ratio=sharpe,
                sortino_ratio=stats.get("SortinoRatio", 0.0),
                max_drawdown=drawdown,
                volatility=volatility,
                var_95=0.0,  # Placeholder - would need returns data
                cvar_95=0.0,  # Placeholder
                beta=stats.get("Beta", 1.0),
                alpha=stats.get("Alpha", 0.0),
                total_trades=total_trades,
                win_rate=win_rate,
                avg_win=stats.get("AverageWin", 0.0),
                avg_loss=stats.get("AverageLoss", 0.0),
                profit_factor=stats.get("ProfitLossRatio", 0.0),
                calmar_ratio=cagr / drawdown if drawdown > 0 else 0.0,
                information_ratio=stats.get("InformationRatio", 0.0),
                treynor_ratio=stats.get("TreynorRatio", 0.0),
                downside_deviation=0.0,  # Placeholder
                excess_return=cagr - benchmark_return,
                tracking_error=0.0,  # Placeholder
                risk_adjusted_score=risk_adjusted_score,
                fitness_score=fitness_score,
                backtest_duration=0.0,  # Set by caller
                test_timestamp=datetime.now(),
                average_profit_per_trade=avg_profit_trade
            )
            
        except Exception as e:
            # Return default performance with error indicators
            return cls(
                strategy_id=strategy.strategy_id,
                template_name=strategy.template_name,
                category=strategy.category,
                cagr=0.0, sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=1.0,
                volatility=0.0, var_95=0.0, cvar_95=0.0, beta=1.0, alpha=0.0,
                total_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                profit_factor=0.0, calmar_ratio=0.0, information_ratio=0.0,
                treynor_ratio=0.0, downside_deviation=0.0, excess_return=0.0,
                tracking_error=0.0, risk_adjusted_score=0.0, fitness_score=0.0,
                backtest_duration=0.0, test_timestamp=datetime.now(),
                average_profit_per_trade=0.0
            )
    
    @staticmethod
    def _calculate_risk_adjusted_score(cagr: float, sharpe: float, drawdown: float) -> float:
        """Calculate risk-adjusted performance score"""
        # Multi-factor scoring emphasizing risk-adjusted returns
        cagr_score = min(cagr / 0.25, 1.0) * 40  # 40% weight, target 25% CAGR
        sharpe_score = min(sharpe / 1.0, 1.0) * 30  # 30% weight, target 1.0 Sharpe
        drawdown_score = max(0, (0.15 - drawdown) / 0.15) * 30  # 30% weight, target <15% drawdown
        
        return cagr_score + sharpe_score + drawdown_score
    
    @staticmethod
    def _calculate_fitness_score(cagr: float, sharpe: float, drawdown: float, volatility: float) -> float:
        """Calculate evolutionary fitness score"""
        # Genetic algorithm fitness function
        if drawdown >= 0.5:  # Extreme drawdown penalty
            return 0.0
        
        base_score = cagr * sharpe / max(drawdown, 0.01)
        volatility_penalty = max(0, volatility - 0.20) * 10  # Penalty for >20% volatility
        
        return max(0, base_score - volatility_penalty)
    
    def meets_targets(self, targets: PerformanceTargets) -> bool:
        """Check if performance meets target criteria"""
        return (
            self.cagr >= targets.target_cagr and
            self.sharpe_ratio >= targets.target_sharpe and
            self.max_drawdown <= targets.max_drawdown and
            self.win_rate >= targets.min_win_rate and
            self.average_profit_per_trade >= targets.target_average_profit_per_trade
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["test_timestamp"] = self.test_timestamp.isoformat()
        return data

@dataclass
class BacktestJob:
    """Backtest job for queue processing"""
    strategy: GeneratedStrategy
    project_id: int
    priority: int = 1  # 1=high, 5=low
    timeout: int = 1800  # 30 minutes
    retry_count: int = 0
    max_retries: int = 2

class StrategyTester:
    """
    High-performance strategy testing system
    
    Features:
    - Parallel backtesting with configurable concurrency
    - Optimized for 30-minute backtesting over 15 years
    - Comprehensive performance analysis
    - Automatic retry logic for failed tests
    - Performance caching and deduplication
    - Real-time progress monitoring
    """
    
    def __init__(self, quantconnect_client: QuantConnectClient, backtest_config: BacktestConfig):
        self.qc_client = quantconnect_client
        self.config = backtest_config
        self.logger = get_logger(__name__)
        
        # Memory system for learning
        self.memory = get_strategy_memory()
        
        # Performance tracking
        self.total_tests = 0
        self.successful_tests = 0
        self.failed_tests = 0
        self.total_test_time = 0.0
        self.average_test_time = 0.0
        self.learned_tests = 0  # Tests that contributed to memory
        
        # Results cache
        self.performance_cache: Dict[str, StrategyPerformance] = {}
        self.backtest_queue: List[BacktestJob] = []
        
        # Concurrency settings
        self.max_concurrent_backtests = 5  # Conservative for QuantConnect
        self.batch_size = 10
        
        # Project management
        self.test_project_id: Optional[int] = None
        
        # Performance thresholds for early filtering
        self.min_acceptable_sharpe = 0.5
        self.max_acceptable_drawdown = 0.30
        
    async def initialize(self) -> None:
        """Initialize the strategy tester"""
        # Create or get testing project
        try:
            self.test_project_id = await self.qc_client.create_project(
                name=f"EvolutionSystem_Testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                language="Py"
            )
            self.logger.info(f"Created testing project ID: {self.test_project_id}")
        except Exception as e:
            self.logger.error(f"Failed to create testing project: {str(e)}")
            raise
    
    async def test_strategies(self, strategies: List[GeneratedStrategy]) -> List[StrategyPerformance]:
        """Test multiple strategies with parallel execution"""
        start_time = time.time()
        
        # Filter out already tested strategies
        new_strategies = [
            s for s in strategies 
            if s.hash_signature not in self.performance_cache
        ]
        
        if not new_strategies:
            self.logger.info("All strategies already tested, returning cached results")
            return [self.performance_cache[s.hash_signature] for s in strategies if s.hash_signature in self.performance_cache]
        
        self.logger.info(f"Testing {len(new_strategies)} new strategies (cached: {len(strategies) - len(new_strategies)})")
        
        # Create backtest jobs
        jobs = [
            BacktestJob(strategy=strategy, project_id=self.test_project_id)
            for strategy in new_strategies
        ]
        
        # Execute backtests in batches
        all_results = []
        
        for i in range(0, len(jobs), self.batch_size):
            batch = jobs[i:i + self.batch_size]
            batch_results = await self._execute_backtest_batch(batch)
            all_results.extend(batch_results)
            
            # Log progress
            progress = min(i + len(batch), len(jobs))
            self.logger.info(f"Completed {progress}/{len(jobs)} backtests")
        
        # Update statistics
        test_duration = time.time() - start_time
        self.total_test_time += test_duration
        self.total_tests += len(new_strategies)
        self.successful_tests += len([r for r in all_results if r is not None])
        self.failed_tests += len([r for r in all_results if r is None])
        
        if self.total_tests > 0:
            self.average_test_time = self.total_test_time / self.total_tests
        
        # Filter out None results
        valid_results = [r for r in all_results if r is not None]
        
        # Add cached results
        cached_results = [
            self.performance_cache[s.hash_signature] 
            for s in strategies 
            if s.hash_signature in self.performance_cache
        ]
        
        final_results = valid_results + cached_results
        
        # Log performance summary
        if valid_results:
            avg_cagr = statistics.mean([r.cagr for r in valid_results])
            avg_sharpe = statistics.mean([r.sharpe_ratio for r in valid_results])
            avg_drawdown = statistics.mean([r.max_drawdown for r in valid_results])
            
            self.logger.info(
                f"Backtest batch completed: {len(valid_results)} strategies tested in {test_duration:.1f}s "
                f"(avg: {test_duration/len(new_strategies):.1f}s per strategy)"
            )
            self.logger.info(
                f"Performance summary - CAGR: {avg_cagr:.2%}, Sharpe: {avg_sharpe:.2f}, "
                f"Drawdown: {avg_drawdown:.2%}"
            )
        
        # Log performance metrics
        PERF_LOGGER.log_system_performance({
            "strategies_tested": len(new_strategies),
            "test_duration_seconds": test_duration,
            "average_test_time": test_duration / len(new_strategies) if new_strategies else 0,
            "success_rate": len(valid_results) / len(new_strategies) if new_strategies else 0,
            "cache_hit_rate": (len(strategies) - len(new_strategies)) / len(strategies) if strategies else 0
        })
        
        return final_results
    
    async def _execute_backtest_batch(self, jobs: List[BacktestJob]) -> List[Optional[StrategyPerformance]]:
        """Execute a batch of backtests concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent_backtests)
        
        async def run_single_backtest(job: BacktestJob) -> Optional[StrategyPerformance]:
            async with semaphore:
                return await self._run_backtest_with_retry(job)
        
        tasks = [run_single_backtest(job) for job in jobs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Backtest failed for {jobs[i].strategy.strategy_id}: {str(result)}")
                final_results.append(None)
            else:
                final_results.append(result)
        
        return final_results
    
    async def _run_backtest_with_retry(self, job: BacktestJob) -> Optional[StrategyPerformance]:
        """Run backtest with retry logic"""
        for attempt in range(job.max_retries + 1):
            try:
                return await self._run_single_backtest(job)
            except Exception as e:
                job.retry_count += 1
                if attempt < job.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Backtest attempt {attempt + 1} failed for {job.strategy.strategy_id}: {str(e)}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Backtest failed after {job.max_retries + 1} attempts for {job.strategy.strategy_id}: {str(e)}"
                    )
                    return None
        
        return None
    
    async def _run_single_backtest(self, job: BacktestJob) -> StrategyPerformance:
        """Run a single backtest"""
        start_time = time.time()
        strategy = job.strategy
        backtest_id = f"bt_{strategy.strategy_id[:8]}_{int(time.time())}"
        
        try:
            # Check cache first
            if strategy.hash_signature in self.performance_cache:
                return self.performance_cache[strategy.hash_signature]
            
            # Log backtest start to dashboard
            DASHBOARD.log_backtest_start(
                strategy_id=strategy.strategy_id,
                backtest_id=backtest_id,
                agent_name="strategy_tester"
            )
            
            self.logger.debug(f"Running backtest for strategy {strategy.strategy_id}")
            
            # Log compilation progress
            DASHBOARD.log_backtest_progress(
                backtest_id=backtest_id,
                progress=10.0,
                status="compiling"
            )
            
            # Run backtest workflow
            result = await asyncio.wait_for(
                self.qc_client.run_backtest_workflow(
                    project_id=job.project_id,
                    strategy_code=strategy.code,
                    strategy_name=f"{strategy.template_name}_{strategy.strategy_id[:8]}",
                    parameters=strategy.parameters
                ),
                timeout=job.timeout
            )
            
            # Log running progress
            DASHBOARD.log_backtest_progress(
                backtest_id=backtest_id,
                progress=75.0,
                status="running"
            )
            
            if not result.is_successful:
                # Log failure
                DASHBOARD.log_backtest_complete(
                    backtest_id=backtest_id,
                    final_results={},
                    error_message=result.error
                )
                raise Exception(f"Backtest failed: {result.error}")
            
            # Calculate performance metrics
            backtest_duration = time.time() - start_time
            performance = StrategyPerformance.from_backtest_result(strategy, result)
            performance.backtest_duration = backtest_duration
            
            # Early filtering for poor performance
            if not self._passes_minimum_thresholds(performance):
                self.logger.debug(f"Strategy {strategy.strategy_id} filtered out for poor performance")
                performance.fitness_score = 0.0
            
            # Cache result
            self.performance_cache[strategy.hash_signature] = performance
            
            # Learn from this backtest result
            await self._learn_from_backtest(strategy, performance, result)
            
            # Log completion with results to dashboard
            final_results = {
                "cagr": performance.cagr,
                "sharpe": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "fitness_score": performance.fitness_score
            }
            
            DASHBOARD.log_backtest_complete(
                backtest_id=backtest_id,
                final_results=final_results
            )
            
            self.logger.debug(
                f"Backtest completed for {strategy.strategy_id} in {backtest_duration:.1f}s - "
                f"CAGR: {performance.cagr:.2%}, Sharpe: {performance.sharpe_ratio:.2f}, "
                f"Drawdown: {performance.max_drawdown:.2%}"
            )
            
            return performance
            
        except asyncio.TimeoutError:
            DASHBOARD.log_backtest_complete(
                backtest_id=backtest_id,
                final_results={},
                error_message=f"Timeout after {job.timeout}s"
            )
            raise Exception(f"Backtest timeout after {job.timeout}s")
        except Exception as e:
            DASHBOARD.log_backtest_complete(
                backtest_id=backtest_id,
                final_results={},
                error_message=str(e)
            )
            self.logger.error(f"Backtest error for {strategy.strategy_id}: {str(e)}")
            raise
    
    def _passes_minimum_thresholds(self, performance: StrategyPerformance) -> bool:
        """Check if performance meets minimum acceptable thresholds"""
        return (
            performance.sharpe_ratio >= self.min_acceptable_sharpe and
            performance.max_drawdown <= self.max_acceptable_drawdown and
            performance.cagr > 0.0
        )
    
    async def test_single_strategy(self, strategy: GeneratedStrategy) -> Optional[StrategyPerformance]:
        """Test a single strategy"""
        results = await self.test_strategies([strategy])
        return results[0] if results else None
    
    def get_best_performers(self, count: int = 10, metric: str = "fitness_score") -> List[StrategyPerformance]:
        """Get top performing strategies"""
        all_performances = list(self.performance_cache.values())
        
        if not all_performances:
            return []
        
        # Sort by specified metric
        if metric == "fitness_score":
            sorted_performances = sorted(all_performances, key=lambda x: x.fitness_score, reverse=True)
        elif metric == "sharpe_ratio":
            sorted_performances = sorted(all_performances, key=lambda x: x.sharpe_ratio, reverse=True)
        elif metric == "cagr":
            sorted_performances = sorted(all_performances, key=lambda x: x.cagr, reverse=True)
        elif metric == "risk_adjusted_score":
            sorted_performances = sorted(all_performances, key=lambda x: x.risk_adjusted_score, reverse=True)
        else:
            sorted_performances = sorted(all_performances, key=lambda x: x.fitness_score, reverse=True)
        
        return sorted_performances[:count]
    
    def get_strategies_meeting_targets(self, targets: PerformanceTargets) -> List[StrategyPerformance]:
        """Get strategies that meet performance targets"""
        return [
            perf for perf in self.performance_cache.values()
            if perf.meets_targets(targets)
        ]
    
    def get_performance_by_category(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by strategy category"""
        category_performances = {}
        
        for performance in self.performance_cache.values():
            category = performance.category
            if category not in category_performances:
                category_performances[category] = []
            category_performances[category].append(performance)
        
        # Calculate statistics for each category
        category_stats = {}
        for category, performances in category_performances.items():
            if performances:
                category_stats[category] = {
                    "count": len(performances),
                    "avg_cagr": statistics.mean([p.cagr for p in performances]),
                    "avg_sharpe": statistics.mean([p.sharpe_ratio for p in performances]),
                    "avg_drawdown": statistics.mean([p.max_drawdown for p in performances]),
                    "best_fitness": max([p.fitness_score for p in performances]),
                    "targets_met": len([p for p in performances if p.meets_targets(PerformanceTargets())])
                }
        
        return category_stats
    
    def get_testing_stats(self) -> Dict[str, Any]:
        """Get comprehensive testing statistics"""
        success_rate = (
            self.successful_tests / self.total_tests * 100 
            if self.total_tests > 0 else 0
        )
        
        cache_size = len(self.performance_cache)
        
        # Performance distribution
        if self.performance_cache:
            all_performances = list(self.performance_cache.values())
            cagr_values = [p.cagr for p in all_performances]
            sharpe_values = [p.sharpe_ratio for p in all_performances]
            drawdown_values = [p.max_drawdown for p in all_performances]
            
            performance_stats = {
                "cagr": {
                    "mean": statistics.mean(cagr_values),
                    "median": statistics.median(cagr_values),
                    "max": max(cagr_values),
                    "min": min(cagr_values)
                },
                "sharpe": {
                    "mean": statistics.mean(sharpe_values),
                    "median": statistics.median(sharpe_values),
                    "max": max(sharpe_values),
                    "min": min(sharpe_values)
                },
                "drawdown": {
                    "mean": statistics.mean(drawdown_values),
                    "median": statistics.median(drawdown_values),
                    "max": max(drawdown_values),
                    "min": min(drawdown_values)
                }
            }
        else:
            performance_stats = {}
        
        return {
            "total_tests": self.total_tests,
            "successful_tests": self.successful_tests,
            "failed_tests": self.failed_tests,
            "success_rate_percent": success_rate,
            "average_test_time_seconds": self.average_test_time,
            "total_test_time_hours": self.total_test_time / 3600,
            "cached_results": cache_size,
            "performance_distribution": performance_stats,
            "category_breakdown": self.get_performance_by_category()
        }
    
    def export_results(self, filepath: Path) -> None:
        """Export all performance results to JSON"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "testing_stats": self.get_testing_stats(),
            "performances": [perf.to_dict() for perf in self.performance_cache.values()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(self.performance_cache)} results to {filepath}")
    
    def clear_cache(self) -> None:
        """Clear performance cache"""
        cache_size = len(self.performance_cache)
        self.performance_cache.clear()
        self.logger.info(f"Cleared performance cache ({cache_size} entries)")
    
    async def _learn_from_backtest(self, strategy: GeneratedStrategy, performance: StrategyPerformance, backtest_result: BacktestResult):
        """Learn from backtest result and update memory"""
        try:
            # Create performance result dictionary for memory system
            performance_result = {
                "backtest_id": performance.backtest_id,
                "cagr": performance.cagr,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "win_rate": performance.win_rate,
                "profit_factor": performance.profit_factor,
                "total_trades": performance.total_trades,
                "avg_trade_duration": getattr(performance, 'avg_trade_duration', 0.0),
                "best_month": getattr(performance, 'best_month', 0.0),
                "worst_month": getattr(performance, 'worst_month', 0.0),
                "market_correlation": getattr(performance, 'market_correlation', 0.0),
                "volatility": getattr(performance, 'volatility', 0.0),
                "calmar_ratio": getattr(performance, 'calmar_ratio', 0.0),
                "sortino_ratio": getattr(performance, 'sortino_ratio', 0.0),
                "execution_time": performance.backtest_duration,
                "fitness_score": performance.fitness_score,
                "risk_adjusted_score": performance.risk_adjusted_score
            }
            
            # Extract market conditions from current state
            market_data = self._extract_market_conditions()
            
            # Store in memory
            strategy_params = {
                'template_name': strategy.template_name,
                'category': strategy.category,
                'asset_class': strategy.asset_class,
                'timeframe': strategy.timeframe,
                **strategy.parameters
            }
            
            memory_id = self.memory.remember_strategy(
                strategy_code=strategy.code,
                strategy_params=strategy_params,
                performance_result=performance_result,
                market_data=market_data
            )
            
            self.learned_tests += 1
            
            # Add memory metadata to strategy
            strategy.metadata['memory_id'] = memory_id
            strategy.metadata['learned'] = True
            strategy.metadata['performance_recorded'] = True
            
            self.logger.debug(f"Learned from strategy {strategy.strategy_id} -> memory ID {memory_id}")
            
        except Exception as e:
            self.logger.error(f"Error learning from backtest result: {str(e)}")
    
    def _extract_market_conditions(self) -> Dict[str, Any]:
        """Extract current market conditions for learning"""
        # This would ideally fetch real market data
        # For now, provide reasonable defaults
        current_time = datetime.now()
        
        return {
            'market_regime': 'bull',  # Would be determined by market analysis
            'volatility': 0.2,  # Would be calculated from recent market data
            'trend_strength': 0.6,  # Would be calculated from trend indicators
            'sector_performance': {
                'technology': 0.15,
                'healthcare': 0.12,
                'finance': 0.08,
                'energy': 0.05,
                'utilities': 0.03
            },
            'economic_indicators': {
                'unemployment_rate': 4.0,
                'inflation_rate': 3.2,
                'gdp_growth': 2.8,
                'interest_rate': 5.25
            },
            'timestamp': current_time.isoformat(),
            'period_start': (current_time - timedelta(days=365)).isoformat(),
            'period_end': current_time.isoformat()
        }
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights from memory system related to testing"""
        try:
            memory_stats = self.memory.get_memory_stats()
            
            return {
                'memory_stats': memory_stats,
                'testing_stats': {
                    'total_tests': self.total_tests,
                    'successful_tests': self.successful_tests,
                    'failed_tests': self.failed_tests,
                    'learned_tests': self.learned_tests,
                    'learning_rate': self.learned_tests / max(self.total_tests, 1),
                    'average_test_time': self.average_test_time,
                    'cached_results': len(self.performance_cache)
                },
                'insights': {
                    'tests_contributing_to_learning': self.learned_tests,
                    'success_rate': self.successful_tests / max(self.total_tests, 1),
                    'memory_utilization': memory_stats.get('total_strategies', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory insights: {str(e)}")
            return {'error': str(e)}