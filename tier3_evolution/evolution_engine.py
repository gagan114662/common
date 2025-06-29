"""
TIER 3: Evolution Engine with Genetic Algorithms
Advanced strategy evolution using DEAP framework as recommended by Gemini
"""

import asyncio
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from collections import defaultdict

from deap import base, creator, tools, algorithms
import multiprocessing as mp

from tier1_core.logger import get_logger, PERF_LOGGER
from tier2_strategy.strategy_generator import (
    StrategyGenerator, 
    GeneratedStrategy, 
    StrategyTemplate,
    ParameterRange
)
from tier2_strategy.strategy_tester import StrategyTester, StrategyPerformance
from config.settings import EvolutionConfig, SYSTEM_CONFIG

@dataclass
class Individual:
    """Individual in the genetic population"""
    strategy_id: str
    template_name: str
    parameters: Dict[str, Any]
    fitness: Optional[float] = None
    generation: int = 0
    parents: List[str] = None
    
    def __post_init__(self):
        if self.parents is None:
            self.parents = []

@dataclass
class Population:
    """Population of strategies"""
    generation: int
    individuals: List[Individual]
    best_fitness: float
    average_fitness: float
    diversity_score: float
    
    def get_elite(self, count: int) -> List[Individual]:
        """Get top performing individuals"""
        sorted_individuals = sorted(
            self.individuals, 
            key=lambda x: x.fitness if x.fitness else 0, 
            reverse=True
        )
        return sorted_individuals[:count]

@dataclass
class EvolutionStats:
    """Evolution statistics tracking"""
    total_generations: int = 0
    total_evaluations: int = 0
    best_fitness_history: List[float] = None
    average_fitness_history: List[float] = None
    diversity_history: List[float] = None
    convergence_rate: float = 0.0
    stagnation_counter: int = 0
    
    def __post_init__(self):
        if self.best_fitness_history is None:
            self.best_fitness_history = []
        if self.average_fitness_history is None:
            self.average_fitness_history = []
        if self.diversity_history is None:
            self.diversity_history = []

class EvolutionEngine:
    """
    Genetic Algorithm-based strategy evolution engine
    
    Features:
    - Multi-objective optimization (CAGR, Sharpe, Drawdown)
    - Adaptive mutation rates
    - Elitism with diversity preservation
    - Parallel fitness evaluation
    - Convergence detection and restart
    - Template-aware crossover and mutation
    """
    
    def __init__(
        self,
        strategy_generator: StrategyGenerator,
        strategy_tester: StrategyTester,
        evolution_config: EvolutionConfig
    ):
        self.generator = strategy_generator
        self.tester = strategy_tester
        self.config = evolution_config
        self.logger = get_logger(__name__)
        
        # Evolution state
        self.current_population: Optional[Population] = None
        self.evolution_stats = EvolutionStats()
        self.is_running = False
        
        # Performance cache
        self.fitness_cache: Dict[str, float] = {}
        self.performance_cache: Dict[str, StrategyPerformance] = {}
        
        # Evolution parameters (adaptive)
        self.current_mutation_rate = self.config.mutation_rate
        self.current_crossover_rate = self.config.crossover_rate
        
        # Convergence detection
        self.fitness_plateau_threshold = 5  # generations
        self.diversity_threshold = 0.1
        self.restart_threshold = 10  # generations without improvement
        
        # Parallel processing
        self.max_workers = mp.cpu_count() - 1
        
        # Initialize DEAP
        self._setup_deap()
        
    def _setup_deap(self) -> None:
        """Setup DEAP framework for genetic algorithms"""
        # Create fitness class (maximize)
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self._template_aware_crossover)
        self.toolbox.register("mutate", self._template_aware_mutation)
        
    async def initialize(self) -> None:
        """Initialize the evolution engine"""
        self.logger.info("Evolution engine initializing...")
        
        # Create initial population
        await self._create_initial_population()
        
        self.is_running = True
        self.logger.info(f"Evolution engine initialized with population size {self.config.population_size}")
    
    async def _create_initial_population(self) -> None:
        """Create the initial population of strategies"""
        self.logger.info("Creating initial population...")
        
        # Generate diverse initial strategies
        initial_strategies = await self.generator.generate_strategies_batch(
            count=self.config.population_size,
            categories=None  # All categories for diversity
        )
        
        # Convert to individuals
        individuals = []
        for strategy in initial_strategies:
            individual = Individual(
                strategy_id=strategy.strategy_id,
                template_name=strategy.template_name,
                parameters=strategy.parameters,
                generation=0
            )
            individuals.append(individual)
        
        # Evaluate initial population
        await self._evaluate_population(individuals)
        
        # Calculate population statistics
        fitness_values = [ind.fitness for ind in individuals if ind.fitness]
        best_fitness = max(fitness_values) if fitness_values else 0
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0
        diversity = self._calculate_diversity(individuals)
        
        self.current_population = Population(
            generation=0,
            individuals=individuals,
            best_fitness=best_fitness,
            average_fitness=avg_fitness,
            diversity_score=diversity
        )
        
        # Update stats
        self.evolution_stats.best_fitness_history.append(best_fitness)
        self.evolution_stats.average_fitness_history.append(avg_fitness)
        self.evolution_stats.diversity_history.append(diversity)
        
        self.logger.info(
            f"Initial population created - Best fitness: {best_fitness:.3f}, "
            f"Avg fitness: {avg_fitness:.3f}, Diversity: {diversity:.3f}"
        )
    
    async def evolve_generation(self) -> Population:
        """Evolve one generation"""
        if not self.current_population:
            await self._create_initial_population()
        
        generation = self.current_population.generation + 1
        self.logger.info(f"Evolving generation {generation}...")
        
        # Select parents
        parents = self._select_parents()
        
        # Create offspring through crossover and mutation
        offspring = await self._create_offspring(parents)
        
        # Evaluate offspring
        await self._evaluate_population(offspring)
        
        # Select next generation (elitism + offspring)
        next_generation = self._select_next_generation(
            self.current_population.individuals + offspring
        )
        
        # Calculate statistics
        fitness_values = [ind.fitness for ind in next_generation if ind.fitness]
        best_fitness = max(fitness_values) if fitness_values else 0
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0
        diversity = self._calculate_diversity(next_generation)
        
        # Create new population
        new_population = Population(
            generation=generation,
            individuals=next_generation,
            best_fitness=best_fitness,
            average_fitness=avg_fitness,
            diversity_score=diversity
        )
        
        # Update evolution stats
        self.evolution_stats.total_generations += 1
        self.evolution_stats.best_fitness_history.append(best_fitness)
        self.evolution_stats.average_fitness_history.append(avg_fitness)
        self.evolution_stats.diversity_history.append(diversity)
        
        # Check for stagnation
        if self._is_stagnant():
            self.evolution_stats.stagnation_counter += 1
            await self._handle_stagnation()
        else:
            self.evolution_stats.stagnation_counter = 0
        
        # Adapt evolution parameters
        self._adapt_parameters()
        
        # Log progress
        improvement = best_fitness - self.current_population.best_fitness
        self.logger.info(
            f"Generation {generation} complete - Best: {best_fitness:.3f} "
            f"(+{improvement:.3f}), Avg: {avg_fitness:.3f}, Diversity: {diversity:.3f}"
        )
        
        # Log to performance tracker
        PERF_LOGGER.log_system_performance({
            "evolution_generation": generation,
            "best_fitness": best_fitness,
            "average_fitness": avg_fitness,
            "diversity": diversity,
            "population_size": len(next_generation),
            "mutation_rate": self.current_mutation_rate,
            "crossover_rate": self.current_crossover_rate
        })
        
        self.current_population = new_population
        return new_population
    
    async def evolve_population(self, performance_results: List[StrategyPerformance]) -> None:
        """Evolve population based on performance results"""
        # Update fitness cache with new results
        for perf in performance_results:
            fitness = self._calculate_fitness(perf)
            self.fitness_cache[perf.strategy_id] = fitness
            self.performance_cache[perf.strategy_id] = perf
        
        # Evolve one generation
        await self.evolve_generation()
    
    def _select_parents(self) -> List[Individual]:
        """Select parents for reproduction using tournament selection"""
        # Convert to DEAP individuals for selection
        deap_pop = []
        for ind in self.current_population.individuals:
            deap_ind = creator.Individual(ind.parameters)
            deap_ind.fitness.values = (ind.fitness if ind.fitness else 0,)
            deap_pop.append(deap_ind)
        
        # Tournament selection
        parents_deap = self.toolbox.select(
            deap_pop, 
            k=int(self.config.population_size * self.current_crossover_rate)
        )
        
        # Convert back to our Individual format
        parents = []
        for i, deap_ind in enumerate(parents_deap):
            # Find matching individual
            for ind in self.current_population.individuals:
                if ind.parameters == dict(deap_ind):
                    parents.append(ind)
                    break
        
        return parents
    
    async def _create_offspring(self, parents: List[Individual]) -> List[Individual]:
        """Create offspring through crossover and mutation"""
        offspring = []
        
        # Pair up parents
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            
            # Crossover
            if random.random() < self.current_crossover_rate:
                child1_params, child2_params = self._template_aware_crossover(
                    parent1.parameters, 
                    parent2.parameters
                )
                
                # Create children
                for params in [child1_params, child2_params]:
                    # Mutation
                    if random.random() < self.current_mutation_rate:
                        params = self._template_aware_mutation(
                            params, 
                            parent1.template_name
                        )
                    
                    # Create new individual
                    child = Individual(
                        strategy_id=f"evolved_{len(offspring)}_{datetime.now().timestamp()}",
                        template_name=parent1.template_name,  # Inherit template
                        parameters=params,
                        generation=self.current_population.generation + 1,
                        parents=[parent1.strategy_id, parent2.strategy_id]
                    )
                    offspring.append(child)
        
        # Fill remaining population slots with mutations
        while len(offspring) < self.config.population_size - int(self.config.population_size * self.config.elitism_rate):
            # Select random parent
            parent = random.choice(parents)
            
            # Mutate
            mutated_params = self._template_aware_mutation(
                parent.parameters.copy(), 
                parent.template_name
            )
            
            # Create mutant
            mutant = Individual(
                strategy_id=f"mutant_{len(offspring)}_{datetime.now().timestamp()}",
                template_name=parent.template_name,
                parameters=mutated_params,
                generation=self.current_population.generation + 1,
                parents=[parent.strategy_id]
            )
            offspring.append(mutant)
        
        return offspring
    
    def _template_aware_crossover(
        self, 
        parent1_params: Dict[str, Any], 
        parent2_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover that respects template parameter constraints"""
        child1_params = {}
        child2_params = {}
        
        # Get common parameters
        common_keys = set(parent1_params.keys()) & set(parent2_params.keys())
        
        for key in common_keys:
            # Randomly swap parameters
            if random.random() < 0.5:
                child1_params[key] = parent1_params[key]
                child2_params[key] = parent2_params[key]
            else:
                child1_params[key] = parent2_params[key]
                child2_params[key] = parent1_params[key]
        
        # Handle non-common parameters
        for key in parent1_params:
            if key not in common_keys:
                child1_params[key] = parent1_params[key]
                child2_params[key] = parent1_params[key]
        
        for key in parent2_params:
            if key not in common_keys and key not in child1_params:
                child1_params[key] = parent2_params[key]
                child2_params[key] = parent2_params[key]
        
        return child1_params, child2_params
    
    def _template_aware_mutation(
        self, 
        params: Dict[str, Any], 
        template_name: str
    ) -> Dict[str, Any]:
        """Perform mutation that respects template parameter constraints"""
        mutated = params.copy()
        
        # Get template to understand parameter constraints
        template = self.generator.template_library.get_template(template_name)
        if not template:
            return mutated
        
        # Mutate each parameter with probability
        for param in template.parameters:
            if param.name in mutated and random.random() < 0.3:  # 30% chance per parameter
                # Generate new value within constraints
                mutated[param.name] = param.generate_value()
        
        return mutated
    
    async def _evaluate_population(self, individuals: List[Individual]) -> None:
        """Evaluate fitness of individuals"""
        # Filter individuals that need evaluation
        to_evaluate = [
            ind for ind in individuals 
            if ind.fitness is None and ind.strategy_id not in self.fitness_cache
        ]
        
        if not to_evaluate:
            # Use cached fitness
            for ind in individuals:
                if ind.strategy_id in self.fitness_cache:
                    ind.fitness = self.fitness_cache[ind.strategy_id]
            return
        
        self.logger.info(f"Evaluating {len(to_evaluate)} individuals...")
        
        # Create strategies for evaluation
        strategies = []
        for ind in to_evaluate:
            # Get template
            template = self.generator.template_library.get_template(ind.template_name)
            if template:
                # Generate strategy code
                strategy = GeneratedStrategy.create(
                    template=template,
                    parameters=ind.parameters,
                    asset_class="Equity",
                    timeframe="Daily"
                )
                strategy.strategy_id = ind.strategy_id  # Override ID
                strategies.append(strategy)
        
        # Test strategies in parallel
        if strategies:
            performances = await self.tester.test_strategies(strategies)
            
            # Calculate fitness and update individuals
            for perf in performances:
                fitness = self._calculate_fitness(perf)
                
                # Update individual
                for ind in to_evaluate:
                    if ind.strategy_id == perf.strategy_id:
                        ind.fitness = fitness
                        break
                
                # Cache results
                self.fitness_cache[perf.strategy_id] = fitness
                self.performance_cache[perf.strategy_id] = perf
            
            self.evolution_stats.total_evaluations += len(performances)
    
    def _calculate_fitness(self, performance: StrategyPerformance) -> float:
        """Calculate multi-objective fitness score"""
        targets = SYSTEM_CONFIG.performance
        target_cagr = targets.target_cagr
        target_sharpe = targets.target_sharpe
        target_max_drawdown = targets.max_drawdown  # This will be 0.20
        target_avg_profit_trade = targets.target_average_profit_per_trade # This will be 0.0075
        # performance.sharpe_ratio is already calculated with the correct risk_free_rate

        # Normalize metrics
        cagr_score = min(performance.cagr / target_cagr, 1.0) if target_cagr > 0 else 0.0
        sharpe_score = min(performance.sharpe_ratio / target_sharpe, 1.0) if target_sharpe > 0 else 0.0
        # The StrategyPerformance object now has average_profit_per_trade
        avg_profit_score = min(performance.average_profit_per_trade / target_avg_profit_trade, 1.0) if target_avg_profit_trade > 0 else 0.0
        # Drawdown score: 1.0 for zero drawdown, 0.0 if drawdown hits target_max_drawdown or worse.
        drawdown_score = max(0.0, (target_max_drawdown - performance.max_drawdown) / target_max_drawdown) if target_max_drawdown > 0 else 0.0

        # Calculate weighted fitness
        fitness = (
            cagr_score * 0.20 +
            sharpe_score * 0.30 +
            avg_profit_score * 0.30 +
            drawdown_score * 0.20
        )

        # Bonus for exceeding targets
        if (performance.cagr >= target_cagr and
            performance.sharpe_ratio >= target_sharpe and
            performance.max_drawdown <= target_max_drawdown and
            performance.average_profit_per_trade >= target_avg_profit_trade):
            fitness *= 1.2  # 20% bonus

        # Penalty for very high drawdown
        if performance.max_drawdown > 0.30: # This is an absolute threshold, more severe than target_max_drawdown
            fitness *= 0.5  # 50% penalty

        return max(0.0, fitness)

    def _select_next_generation(self, all_individuals: List[Individual]) -> List[Individual]:
        """Select next generation with elitism"""
        # Sort by fitness
        sorted_individuals = sorted(
            all_individuals,
            key=lambda x: x.fitness if x.fitness else 0,
            reverse=True
        )
        
        next_generation = []
        
        # Elitism - keep top performers
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        next_generation.extend(sorted_individuals[:elite_count])
        
        # Fill remaining slots with diversity preservation
        remaining_slots = self.config.population_size - elite_count
        
        # Add diverse individuals
        candidates = sorted_individuals[elite_count:]
        while len(next_generation) < self.config.population_size and candidates:
            # Select individual that maximizes diversity
            best_candidate = None
            best_diversity = 0
            
            for candidate in candidates[:10]:  # Check top 10 candidates
                temp_population = next_generation + [candidate]
                diversity = self._calculate_diversity(temp_population)
                
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate
            
            if best_candidate:
                next_generation.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                # Just add next best
                next_generation.append(candidates.pop(0))
        
        return next_generation[:self.config.population_size]
    
    def _calculate_diversity(self, individuals: List[Individual]) -> float:
        """Calculate population diversity"""
        if len(individuals) < 2:
            return 1.0
        
        # Calculate parameter diversity
        all_params = defaultdict(list)
        
        for ind in individuals:
            for param_name, value in ind.parameters.items():
                if isinstance(value, (int, float)):
                    all_params[param_name].append(value)
        
        # Calculate variance for each parameter
        diversities = []
        for param_name, values in all_params.items():
            if len(values) > 1:
                # Normalize variance by range
                if max(values) != min(values):
                    normalized_var = np.var(values) / (max(values) - min(values))**2
                    diversities.append(normalized_var)
        
        # Average diversity across all parameters
        return np.mean(diversities) if diversities else 0.5
    
    def _is_stagnant(self) -> bool:
        """Check if evolution is stagnant"""
        if len(self.evolution_stats.best_fitness_history) < self.fitness_plateau_threshold:
            return False
        
        # Check if best fitness hasn't improved
        recent_best = self.evolution_stats.best_fitness_history[-self.fitness_plateau_threshold:]
        improvement = max(recent_best) - min(recent_best)
        
        return improvement < 0.01  # Less than 1% improvement
    
    async def _handle_stagnation(self) -> None:
        """Handle evolutionary stagnation"""
        self.logger.warning(f"Evolution stagnant for {self.evolution_stats.stagnation_counter} generations")
        
        if self.evolution_stats.stagnation_counter >= self.restart_threshold:
            # Restart with new population
            self.logger.info("Restarting evolution with new population")
            await self._create_initial_population()
            self.evolution_stats.stagnation_counter = 0
        else:
            # Increase mutation rate temporarily
            self.current_mutation_rate = min(0.5, self.current_mutation_rate * 1.5)
            self.logger.info(f"Increased mutation rate to {self.current_mutation_rate:.3f}")
    
    def _adapt_parameters(self) -> None:
        """Adapt evolution parameters based on progress"""
        # Adjust mutation rate based on diversity
        if self.current_population.diversity_score < self.diversity_threshold:
            # Low diversity - increase mutation
            self.current_mutation_rate = min(0.3, self.current_mutation_rate * 1.1)
        else:
            # Good diversity - decay mutation rate
            self.current_mutation_rate = max(
                self.config.mutation_rate, 
                self.current_mutation_rate * 0.95
            )
        
        # Adjust selection pressure based on convergence
        if len(self.evolution_stats.average_fitness_history) > 10:
            recent_avg = self.evolution_stats.average_fitness_history[-10:]
            convergence_rate = (recent_avg[-1] - recent_avg[0]) / 10
            self.evolution_stats.convergence_rate = convergence_rate
    
    def get_best_strategies(self, count: int = 10) -> List[Tuple[Individual, StrategyPerformance]]:
        """Get best performing strategies"""
        if not self.current_population:
            return []
        
        elite = self.current_population.get_elite(count)
        
        results = []
        for ind in elite:
            if ind.strategy_id in self.performance_cache:
                results.append((ind, self.performance_cache[ind.strategy_id]))
        
        return results
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary"""
        if not self.current_population:
            return {"status": "not_initialized"}
        
        best_strategies = self.get_best_strategies(5)
        
        return {
            "current_generation": self.current_population.generation,
            "population_size": len(self.current_population.individuals),
            "best_fitness": self.current_population.best_fitness,
            "average_fitness": self.current_population.average_fitness,
            "diversity_score": self.current_population.diversity_score,
            "total_evaluations": self.evolution_stats.total_evaluations,
            "convergence_rate": self.evolution_stats.convergence_rate,
            "stagnation_counter": self.evolution_stats.stagnation_counter,
            "current_mutation_rate": self.current_mutation_rate,
            "current_crossover_rate": self.current_crossover_rate,
            "best_strategies": [
                {
                    "strategy_id": ind.strategy_id,
                    "template": ind.template_name,
                    "fitness": ind.fitness,
                    "performance": {
                        "cagr": perf.cagr,
                        "sharpe": perf.sharpe_ratio,
                        "drawdown": perf.max_drawdown
                    } if perf else None
                }
                for ind, perf in best_strategies
            ],
            "fitness_history": {
                "best": self.evolution_stats.best_fitness_history[-20:],
                "average": self.evolution_stats.average_fitness_history[-20:],
                "diversity": self.evolution_stats.diversity_history[-20:]
            }
        }
    
    async def export_best_strategies(self, filepath: Path, count: int = 20) -> None:
        """Export best strategies to file"""
        best_strategies = self.get_best_strategies(count)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "evolution_summary": self.get_evolution_summary(),
            "strategies": []
        }
        
        for ind, perf in best_strategies:
            strategy_data = {
                "strategy_id": ind.strategy_id,
                "template_name": ind.template_name,
                "parameters": ind.parameters,
                "fitness": ind.fitness,
                "generation": ind.generation,
                "performance": perf.to_dict() if perf else None
            }
            export_data["strategies"].append(strategy_data)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(best_strategies)} best strategies to {filepath}")