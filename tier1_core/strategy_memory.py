"""
TIER 1: Intelligent Strategy Memory System
Advanced learning and memory system for strategy optimization and market pattern recognition
"""

import asyncio
import hashlib
import json
import numpy as np
import sqlite3
import pickle
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import threading

# Vector similarity and embeddings
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Knowledge graph (simplified in-memory version)
import networkx as nx

from tier1_core.logger import get_logger

@dataclass
class StrategyFingerprint:
    """Unique fingerprint of a strategy for similarity matching"""
    strategy_id: str
    strategy_type: str
    parameters: Dict[str, Any]
    indicators: List[str]
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_parameters: Dict[str, float]
    code_hash: str
    created_at: datetime
    
    def to_text(self) -> str:
        """Convert strategy to searchable text representation"""
        return f"""
        Strategy Type: {self.strategy_type}
        Parameters: {json.dumps(self.parameters, sort_keys=True)}
        Indicators: {', '.join(self.indicators)}
        Entry: {', '.join(self.entry_conditions)}
        Exit: {', '.join(self.exit_conditions)}
        Risk: {json.dumps(self.risk_parameters, sort_keys=True)}
        """

@dataclass
class MarketConditions:
    """Market conditions during strategy testing"""
    period_start: datetime
    period_end: datetime
    market_regime: str  # "bull", "bear", "sideways", "volatile"
    volatility_level: float
    trend_strength: float
    sector_performance: Dict[str, float]
    economic_indicators: Dict[str, float]
    
    def to_vector(self) -> np.ndarray:
        """Convert market conditions to numeric vector"""
        regime_encoding = {"bull": 1.0, "bear": -1.0, "sideways": 0.0, "volatile": 0.5}
        
        vector = [
            regime_encoding.get(self.market_regime, 0.0),
            self.volatility_level,
            self.trend_strength,
            np.mean(list(self.sector_performance.values())),
            np.mean(list(self.economic_indicators.values()))
        ]
        
        return np.array(vector, dtype=np.float32)

@dataclass
class StrategyPerformance:
    """Comprehensive strategy performance metrics"""
    strategy_id: str
    backtest_id: str
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    best_month: float
    worst_month: float
    market_correlation: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    execution_time: float
    success_score: float  # Composite score
    
    @classmethod
    def from_backtest_result(cls, strategy_id: str, backtest_result: Dict[str, Any]) -> 'StrategyPerformance':
        """Create performance object from backtest result"""
        return cls(
            strategy_id=strategy_id,
            backtest_id=backtest_result.get("backtest_id", ""),
            cagr=backtest_result.get("cagr", 0.0),
            sharpe_ratio=backtest_result.get("sharpe_ratio", 0.0),
            max_drawdown=backtest_result.get("max_drawdown", 1.0),
            win_rate=backtest_result.get("win_rate", 0.0),
            profit_factor=backtest_result.get("profit_factor", 0.0),
            total_trades=backtest_result.get("total_trades", 0),
            avg_trade_duration=backtest_result.get("avg_trade_duration", 0.0),
            best_month=backtest_result.get("best_month", 0.0),
            worst_month=backtest_result.get("worst_month", 0.0),
            market_correlation=backtest_result.get("market_correlation", 0.0),
            volatility=backtest_result.get("volatility", 0.0),
            calmar_ratio=backtest_result.get("calmar_ratio", 0.0),
            sortino_ratio=backtest_result.get("sortino_ratio", 0.0),
            execution_time=backtest_result.get("execution_time", 0.0),
            success_score=cls._calculate_success_score(backtest_result)
        )
    
    @staticmethod
    def _calculate_success_score(result: Dict[str, Any]) -> float:
        """Calculate composite success score"""
        cagr = result.get("cagr", 0.0)
        sharpe = result.get("sharpe_ratio", 0.0)
        drawdown = result.get("max_drawdown", 1.0)
        win_rate = result.get("win_rate", 0.0)
        
        # Weighted composite score
        score = (
            cagr * 0.3 +
            sharpe * 0.3 +
            (1.0 - drawdown) * 0.2 +
            win_rate * 0.2
        )
        
        return max(0.0, min(1.0, score))

class VectorDatabase:
    """FAISS-based vector database for strategy similarity search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None
            self.logger.warning("SentenceTransformers not available, using fallback embeddings")
        
        # Initialize FAISS index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.logger.warning("FAISS not available, using in-memory similarity search")
            self._vectors = []
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding vector from text"""
        if self.model:
            embedding = self.model.encode([text])[0]
            # Pad or truncate to fixed dimension
            if len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            elif len(embedding) < self.dimension:
                padding = np.zeros(self.dimension - len(embedding))
                embedding = np.concatenate([embedding, padding])
            return embedding.astype(np.float32)
        else:
            # Fallback: simple hash-based embedding
            hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_value % (2**32))
            return np.random.normal(0, 1, self.dimension).astype(np.float32)
    
    def add_strategy(self, fingerprint: StrategyFingerprint, performance: StrategyPerformance, market_conditions: MarketConditions):
        """Add strategy to vector database"""
        with self.lock:
            # Create embedding
            text_repr = fingerprint.to_text()
            embedding = self.create_embedding(text_repr)
            
            # Add to index
            if FAISS_AVAILABLE and self.index:
                self.index.add(embedding.reshape(1, -1))
            else:
                self._vectors.append(embedding)
            
            # Store metadata
            metadata = {
                'fingerprint': asdict(fingerprint),
                'performance': asdict(performance), 
                'market_conditions': asdict(market_conditions),
                'timestamp': datetime.now().isoformat()
            }
            self.metadata.append(metadata)
            
            self.logger.debug(f"Added strategy {fingerprint.strategy_id} to vector database")
    
    def find_similar_strategies(self, fingerprint: StrategyFingerprint, k: int = 5, min_performance: float = 0.5) -> List[Tuple[Dict[str, Any], float]]:
        """Find similar high-performing strategies"""
        with self.lock:
            if len(self.metadata) == 0:
                return []
            
            # Create query embedding
            text_repr = fingerprint.to_text()
            query_embedding = self.create_embedding(text_repr)
            
            # Search for similar strategies
            if FAISS_AVAILABLE and self.index:
                distances, indices = self.index.search(query_embedding.reshape(1, -1), min(k * 2, len(self.metadata)))
                results = []
                
                for distance, idx in zip(distances[0], indices[0]):
                    if idx < len(self.metadata):
                        metadata = self.metadata[idx]
                        performance = metadata['performance']
                        
                        # Filter by performance
                        if performance['success_score'] >= min_performance:
                            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                            results.append((metadata, similarity))
                
                # Sort by similarity and return top k
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:k]
            
            else:
                # Fallback: compute similarities manually
                similarities = []
                
                for i, vector in enumerate(self._vectors):
                    distance = np.linalg.norm(query_embedding - vector)
                    similarity = 1.0 / (1.0 + distance)
                    
                    metadata = self.metadata[i]
                    performance = metadata['performance']
                    
                    if performance['success_score'] >= min_performance:
                        similarities.append((metadata, similarity))
                
                # Sort and return top k
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:k]

class KnowledgeGraph:
    """NetworkX-based knowledge graph for strategy relationships"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def add_strategy_node(self, fingerprint: StrategyFingerprint, performance: StrategyPerformance):
        """Add strategy as a node in the knowledge graph"""
        with self.lock:
            node_id = fingerprint.strategy_id
            
            self.graph.add_node(node_id, 
                strategy_type=fingerprint.strategy_type,
                parameters=fingerprint.parameters,
                performance=asdict(performance),
                created_at=fingerprint.created_at.isoformat()
            )
            
            self.logger.debug(f"Added strategy node {node_id} to knowledge graph")
    
    def add_performance_relationship(self, strategy_id1: str, strategy_id2: str, relationship_type: str, strength: float):
        """Add relationship between strategies based on performance patterns"""
        with self.lock:
            if self.graph.has_node(strategy_id1) and self.graph.has_node(strategy_id2):
                self.graph.add_edge(strategy_id1, strategy_id2,
                    relationship=relationship_type,
                    strength=strength,
                    created_at=datetime.now().isoformat()
                )
    
    def find_successful_patterns(self, min_performance: float = 0.7) -> List[Dict[str, Any]]:
        """Identify patterns in successful strategies"""
        with self.lock:
            patterns = []
            
            # Find high-performing nodes
            high_performers = []
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                performance = node_data.get('performance', {})
                if performance.get('success_score', 0.0) >= min_performance:
                    high_performers.append(node_id)
            
            # Analyze common characteristics
            if len(high_performers) >= 2:
                strategy_types = defaultdict(int)
                parameter_patterns = defaultdict(list)
                
                for node_id in high_performers:
                    node_data = self.graph.nodes[node_id]
                    strategy_types[node_data.get('strategy_type')] += 1
                    
                    parameters = node_data.get('parameters', {})
                    for param_name, param_value in parameters.items():
                        parameter_patterns[param_name].append(param_value)
                
                # Extract patterns
                for strategy_type, count in strategy_types.items():
                    if count >= 2:
                        patterns.append({
                            'type': 'strategy_type_pattern',
                            'strategy_type': strategy_type,
                            'occurrence_count': count,
                            'success_rate': count / len(high_performers)
                        })
                
                for param_name, values in parameter_patterns.items():
                    if len(values) >= 2:
                        patterns.append({
                            'type': 'parameter_pattern',
                            'parameter_name': param_name,
                            'mean_value': np.mean(values),
                            'std_value': np.std(values),
                            'occurrence_count': len(values)
                        })
            
            return patterns

class PerformanceWarehouse:
    """SQLite-based structured storage for performance data"""
    
    def __init__(self, db_path: str = "data/strategy_performance.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Strategies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    strategy_type TEXT,
                    parameters TEXT,
                    indicators TEXT,
                    code_hash TEXT,
                    created_at TEXT
                )
            ''')
            
            # Performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT,
                    backtest_id TEXT,
                    cagr REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    success_score REAL,
                    market_regime TEXT,
                    volatility_level REAL,
                    recorded_at TEXT,
                    FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
                )
            ''')
            
            # Market conditions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT,
                    period_start TEXT,
                    period_end TEXT,
                    market_regime TEXT,
                    volatility_level REAL,
                    trend_strength REAL,
                    sector_data TEXT,
                    economic_data TEXT,
                    FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
                )
            ''')
            
            # Learning insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type TEXT,
                    insight_data TEXT,
                    confidence_score REAL,
                    strategies_count INTEGER,
                    created_at TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def store_strategy_result(self, fingerprint: StrategyFingerprint, performance: StrategyPerformance, market_conditions: MarketConditions):
        """Store complete strategy result"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert strategy
            cursor.execute('''
                INSERT OR REPLACE INTO strategies 
                (strategy_id, strategy_type, parameters, indicators, code_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                fingerprint.strategy_id,
                fingerprint.strategy_type,
                json.dumps(fingerprint.parameters),
                json.dumps(fingerprint.indicators),
                fingerprint.code_hash,
                fingerprint.created_at.isoformat()
            ))
            
            # Insert performance
            cursor.execute('''
                INSERT INTO performance 
                (strategy_id, backtest_id, cagr, sharpe_ratio, max_drawdown, win_rate, 
                 profit_factor, total_trades, success_score, market_regime, volatility_level, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                performance.strategy_id,
                performance.backtest_id,
                performance.cagr,
                performance.sharpe_ratio,
                performance.max_drawdown,
                performance.win_rate,
                performance.profit_factor,
                performance.total_trades,
                performance.success_score,
                market_conditions.market_regime,
                market_conditions.volatility_level,
                datetime.now().isoformat()
            ))
            
            # Insert market conditions
            cursor.execute('''
                INSERT INTO market_conditions 
                (strategy_id, period_start, period_end, market_regime, volatility_level, 
                 trend_strength, sector_data, economic_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fingerprint.strategy_id,
                market_conditions.period_start.isoformat(),
                market_conditions.period_end.isoformat(),
                market_conditions.market_regime,
                market_conditions.volatility_level,
                market_conditions.trend_strength,
                json.dumps(market_conditions.sector_performance),
                json.dumps(market_conditions.economic_indicators)
            ))
            
            conn.commit()
            conn.close()
    
    def get_top_performers(self, limit: int = 100, min_success_score: float = 0.6) -> List[Dict[str, Any]]:
        """Get top performing strategies"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.*, p.* FROM strategies s
                JOIN performance p ON s.strategy_id = p.strategy_id
                WHERE p.success_score >= ?
                ORDER BY p.success_score DESC
                LIMIT ?
            ''', (min_success_score, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in results]
    
    def analyze_market_performance(self, market_regime: str) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    AVG(cagr) as avg_cagr,
                    AVG(sharpe_ratio) as avg_sharpe,
                    AVG(max_drawdown) as avg_drawdown,
                    AVG(success_score) as avg_success,
                    COUNT(*) as count
                FROM performance 
                WHERE market_regime = ?
            ''', (market_regime,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[4] > 0:  # count > 0
                return {
                    'market_regime': market_regime,
                    'avg_cagr': result[0],
                    'avg_sharpe': result[1],
                    'avg_drawdown': result[2],
                    'avg_success_score': result[3],
                    'strategy_count': result[4]
                }
            
            return {'market_regime': market_regime, 'strategy_count': 0}

class LearningEngine:
    """Reinforcement learning and pattern analysis engine"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.logger = get_logger(__name__)
        
        # Learning parameters
        self.success_threshold = 0.7
        self.learning_rate = 0.1
        self.pattern_confidence_threshold = 0.6
        
        # Learned patterns
        self.successful_patterns = []
        self.failure_patterns = []
        self.market_adaptations = {}
    
    def learn_from_outcome(self, fingerprint: StrategyFingerprint, performance: StrategyPerformance, market_conditions: MarketConditions):
        """Learn from strategy outcome using reinforcement learning principles"""
        success_score = performance.success_score
        
        # Identify if this is a success or failure
        if success_score >= self.success_threshold:
            self._reinforce_successful_pattern(fingerprint, performance, market_conditions)
        else:
            self._learn_from_failure(fingerprint, performance, market_conditions)
        
        # Update market regime adaptations
        self._update_market_adaptations(fingerprint, performance, market_conditions)
    
    def _reinforce_successful_pattern(self, fingerprint: StrategyFingerprint, performance: StrategyPerformance, market_conditions: MarketConditions):
        """Reinforce patterns that lead to success"""
        pattern = {
            'strategy_type': fingerprint.strategy_type,
            'key_parameters': self._extract_key_parameters(fingerprint.parameters),
            'market_regime': market_conditions.market_regime,
            'success_count': 1,
            'avg_performance': performance.success_score,
            'confidence': performance.success_score
        }
        
        # Check if pattern already exists
        existing_pattern = self._find_similar_pattern(pattern, self.successful_patterns)
        if existing_pattern:
            # Update existing pattern
            existing_pattern['success_count'] += 1
            existing_pattern['avg_performance'] = (
                existing_pattern['avg_performance'] * (existing_pattern['success_count'] - 1) + 
                performance.success_score
            ) / existing_pattern['success_count']
            existing_pattern['confidence'] = min(1.0, existing_pattern['confidence'] + self.learning_rate)
        else:
            # Add new pattern
            self.successful_patterns.append(pattern)
        
        self.logger.info(f"Reinforced successful pattern: {fingerprint.strategy_type} in {market_conditions.market_regime} market")
    
    def _learn_from_failure(self, fingerprint: StrategyFingerprint, performance: StrategyPerformance, market_conditions: MarketConditions):
        """Learn from strategy failures"""
        failure_pattern = {
            'strategy_type': fingerprint.strategy_type,
            'key_parameters': self._extract_key_parameters(fingerprint.parameters),
            'market_regime': market_conditions.market_regime,
            'failure_count': 1,
            'avg_poor_performance': performance.success_score,
            'confidence': 1.0 - performance.success_score
        }
        
        # Track failure patterns to avoid
        existing_failure = self._find_similar_pattern(failure_pattern, self.failure_patterns)
        if existing_failure:
            existing_failure['failure_count'] += 1
            existing_failure['confidence'] = min(1.0, existing_failure['confidence'] + self.learning_rate)
        else:
            self.failure_patterns.append(failure_pattern)
    
    def _extract_key_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key parameters that impact performance"""
        # Focus on numeric parameters that likely impact performance
        key_params = {}
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                # Discretize numeric values for pattern matching
                if 'period' in key.lower() or 'window' in key.lower():
                    key_params[key] = self._discretize_period(value)
                elif 'threshold' in key.lower() or 'level' in key.lower():
                    key_params[key] = self._discretize_threshold(value)
                else:
                    key_params[key] = value
            elif isinstance(value, str):
                key_params[key] = value
        
        return key_params
    
    def _discretize_period(self, value: float) -> str:
        """Discretize period values into categories"""
        if value <= 5:
            return "very_short"
        elif value <= 20:
            return "short"
        elif value <= 50:
            return "medium"
        else:
            return "long"
    
    def _discretize_threshold(self, value: float) -> str:
        """Discretize threshold values"""
        if value <= 0.1:
            return "low"
        elif value <= 0.3:
            return "medium"
        else:
            return "high"
    
    def _find_similar_pattern(self, pattern: Dict[str, Any], pattern_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find similar pattern in list"""
        for existing_pattern in pattern_list:
            if (existing_pattern['strategy_type'] == pattern['strategy_type'] and
                existing_pattern['market_regime'] == pattern['market_regime']):
                
                # Check parameter similarity
                param_similarity = self._calculate_parameter_similarity(
                    pattern['key_parameters'], 
                    existing_pattern['key_parameters']
                )
                
                if param_similarity > 0.8:  # 80% similarity threshold
                    return existing_pattern
        
        return None
    
    def _calculate_parameter_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate similarity between parameter sets"""
        if not params1 or not params2:
            return 0.0
        
        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if params1[key] == params2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def _update_market_adaptations(self, fingerprint: StrategyFingerprint, performance: StrategyPerformance, market_conditions: MarketConditions):
        """Update market-specific adaptations"""
        regime = market_conditions.market_regime
        
        if regime not in self.market_adaptations:
            self.market_adaptations[regime] = {
                'preferred_strategy_types': defaultdict(list),
                'optimal_parameters': defaultdict(list),
                'avg_performance': 0.0,
                'sample_count': 0
            }
        
        adaptation = self.market_adaptations[regime]
        
        # Update preferred strategy types
        adaptation['preferred_strategy_types'][fingerprint.strategy_type].append(performance.success_score)
        
        # Update optimal parameters
        for param_name, param_value in fingerprint.parameters.items():
            if isinstance(param_value, (int, float)):
                adaptation['optimal_parameters'][param_name].append((param_value, performance.success_score))
        
        # Update average performance
        adaptation['sample_count'] += 1
        adaptation['avg_performance'] = (
            adaptation['avg_performance'] * (adaptation['sample_count'] - 1) + 
            performance.success_score
        ) / adaptation['sample_count']
    
    def get_recommendations(self, target_market_regime: str) -> Dict[str, Any]:
        """Get strategy recommendations based on learned patterns"""
        recommendations = {
            'recommended_strategy_types': [],
            'optimal_parameters': {},
            'patterns_to_avoid': [],
            'confidence_level': 0.0
        }
        
        # Successful patterns for target market
        relevant_patterns = [p for p in self.successful_patterns 
                           if p['market_regime'] == target_market_regime and 
                           p['confidence'] >= self.pattern_confidence_threshold]
        
        if relevant_patterns:
            # Sort by confidence and success
            relevant_patterns.sort(key=lambda x: x['confidence'] * x['avg_performance'], reverse=True)
            
            # Extract recommendations
            strategy_scores = defaultdict(list)
            for pattern in relevant_patterns:
                strategy_scores[pattern['strategy_type']].append(pattern['avg_performance'])
            
            # Recommend top strategy types
            for strategy_type, scores in strategy_scores.items():
                avg_score = np.mean(scores)
                recommendations['recommended_strategy_types'].append({
                    'strategy_type': strategy_type,
                    'expected_performance': avg_score,
                    'sample_count': len(scores)
                })
            
            # Sort recommendations
            recommendations['recommended_strategy_types'].sort(
                key=lambda x: x['expected_performance'], reverse=True
            )
            
            # Extract optimal parameters
            if self.market_adaptations.get(target_market_regime):
                adaptation = self.market_adaptations[target_market_regime]
                for param_name, param_data in adaptation['optimal_parameters'].items():
                    if param_data:
                        # Find parameter values that correlate with high performance
                        values, scores = zip(*param_data)
                        if len(values) >= 3:  # Need sufficient data
                            # Weight by performance
                            weighted_avg = np.average(values, weights=scores)
                            recommendations['optimal_parameters'][param_name] = {
                                'recommended_value': weighted_avg,
                                'confidence': np.mean(scores)
                            }
            
            recommendations['confidence_level'] = np.mean([p['confidence'] for p in relevant_patterns])
        
        # Add patterns to avoid
        failure_patterns = [p for p in self.failure_patterns 
                          if p['market_regime'] == target_market_regime and 
                          p['confidence'] >= self.pattern_confidence_threshold]
        
        for pattern in failure_patterns:
            recommendations['patterns_to_avoid'].append({
                'strategy_type': pattern['strategy_type'],
                'parameters': pattern['key_parameters'],
                'failure_rate': pattern['confidence']
            })
        
        return recommendations

class StrategyMemory:
    """
    Intelligent Strategy Memory System
    
    Combines vector database, knowledge graph, and performance warehouse
    for comprehensive strategy learning and optimization
    """
    
    def __init__(self, memory_dir: str = "data/strategy_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.vector_db = VectorDatabase()
        self.knowledge_graph = KnowledgeGraph()
        self.performance_warehouse = PerformanceWarehouse(
            str(self.memory_dir / "performance.db")
        )
        self.learning_engine = LearningEngine(self)
        
        # Memory statistics
        self.total_strategies = 0
        self.successful_strategies = 0
        self.last_update = datetime.now()
        
        self.logger.info("Strategy Memory System initialized")
    
    def remember_strategy(self, strategy_code: str, strategy_params: Dict[str, Any], 
                         performance_result: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """
        Remember a strategy and its performance
        
        Args:
            strategy_code: The strategy source code
            strategy_params: Strategy parameters and configuration
            performance_result: Backtest performance results
            market_data: Market conditions during the test period
        
        Returns:
            strategy_id: Unique identifier for the stored strategy
        """
        try:
            # Create strategy fingerprint
            strategy_id = self._generate_strategy_id(strategy_code, strategy_params)
            fingerprint = self._create_fingerprint(strategy_id, strategy_code, strategy_params)
            
            # Create performance object
            performance = StrategyPerformance.from_backtest_result(strategy_id, performance_result)
            
            # Create market conditions
            market_conditions = self._extract_market_conditions(market_data)
            
            # Store in all components
            self.vector_db.add_strategy(fingerprint, performance, market_conditions)
            self.knowledge_graph.add_strategy_node(fingerprint, performance)
            self.performance_warehouse.store_strategy_result(fingerprint, performance, market_conditions)
            
            # Learn from this outcome
            self.learning_engine.learn_from_outcome(fingerprint, performance, market_conditions)
            
            # Update statistics
            self.total_strategies += 1
            if performance.success_score >= 0.7:
                self.successful_strategies += 1
            self.last_update = datetime.now()
            
            self.logger.info(f"Remembered strategy {strategy_id} with success score {performance.success_score:.3f}")
            
            return strategy_id
            
        except Exception as e:
            self.logger.error(f"Error remembering strategy: {str(e)}")
            raise
    
    def find_similar_successful_strategies(self, strategy_code: str, strategy_params: Dict[str, Any], 
                                         k: int = 5, min_performance: float = 0.6) -> List[Dict[str, Any]]:
        """Find similar strategies that performed well"""
        try:
            # Create temporary fingerprint for search
            temp_id = "search_query"
            fingerprint = self._create_fingerprint(temp_id, strategy_code, strategy_params)
            
            # Search vector database
            similar_strategies = self.vector_db.find_similar_strategies(
                fingerprint, k=k, min_performance=min_performance
            )
            
            results = []
            for metadata, similarity in similar_strategies:
                results.append({
                    'strategy_data': metadata,
                    'similarity_score': similarity,
                    'performance': metadata['performance'],
                    'market_conditions': metadata['market_conditions']
                })
            
            self.logger.debug(f"Found {len(results)} similar successful strategies")
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar strategies: {str(e)}")
            return []
    
    def get_market_recommendations(self, current_market_regime: str) -> Dict[str, Any]:
        """Get recommendations based on current market conditions"""
        try:
            # Get learning engine recommendations
            recommendations = self.learning_engine.get_recommendations(current_market_regime)
            
            # Enhance with warehouse analysis
            market_analysis = self.performance_warehouse.analyze_market_performance(current_market_regime)
            recommendations['market_analysis'] = market_analysis
            
            # Add knowledge graph patterns
            successful_patterns = self.knowledge_graph.find_successful_patterns(min_performance=0.7)
            recommendations['knowledge_patterns'] = successful_patterns
            
            self.logger.info(f"Generated recommendations for {current_market_regime} market with {recommendations['confidence_level']:.2f} confidence")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating market recommendations: {str(e)}")
            return {'confidence_level': 0.0, 'recommended_strategy_types': []}
    
    def analyze_failure_modes(self, strategy_type: str = None) -> Dict[str, Any]:
        """Analyze common failure modes to avoid"""
        try:
            analysis = {
                'failure_patterns': self.learning_engine.failure_patterns,
                'common_issues': [],
                'recommendations': []
            }
            
            # Filter by strategy type if specified
            if strategy_type:
                analysis['failure_patterns'] = [
                    p for p in analysis['failure_patterns'] 
                    if p['strategy_type'] == strategy_type
                ]
            
            # Analyze common failure modes
            if analysis['failure_patterns']:
                # Group by common characteristics
                parameter_failures = defaultdict(list)
                market_failures = defaultdict(int)
                
                for pattern in analysis['failure_patterns']:
                    market_failures[pattern['market_regime']] += pattern['failure_count']
                    for param, value in pattern['key_parameters'].items():
                        parameter_failures[param].append(value)
                
                # Generate insights
                analysis['common_issues'] = [
                    f"Strategies tend to fail in {regime} markets ({count} failures)"
                    for regime, count in market_failures.items()
                ]
                
                analysis['recommendations'] = [
                    f"Avoid {param} values around {set(values)}"
                    for param, values in parameter_failures.items()
                    if len(values) >= 2
                ]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing failure modes: {str(e)}")
            return {'failure_patterns': [], 'common_issues': [], 'recommendations': []}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            'total_strategies': self.total_strategies,
            'successful_strategies': self.successful_strategies,
            'success_rate': self.successful_strategies / max(self.total_strategies, 1),
            'vector_db_size': len(self.vector_db.metadata),
            'knowledge_graph_nodes': self.knowledge_graph.graph.number_of_nodes(),
            'knowledge_graph_edges': self.knowledge_graph.graph.number_of_edges(),
            'learned_patterns': len(self.learning_engine.successful_patterns),
            'failure_patterns': len(self.learning_engine.failure_patterns),
            'market_adaptations': len(self.learning_engine.market_adaptations),
            'last_update': self.last_update.isoformat(),
            'memory_dir': str(self.memory_dir)
        }
    
    def _generate_strategy_id(self, strategy_code: str, strategy_params: Dict[str, Any]) -> str:
        """Generate unique strategy ID"""
        content = f"{strategy_code}_{json.dumps(strategy_params, sort_keys=True)}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()
        return f"strategy_{hash_value[:12]}"
    
    def _create_fingerprint(self, strategy_id: str, strategy_code: str, strategy_params: Dict[str, Any]) -> StrategyFingerprint:
        """Create strategy fingerprint from code and parameters"""
        # Extract strategy characteristics from code and parameters
        strategy_type = strategy_params.get('template_name', 'unknown')
        indicators = self._extract_indicators(strategy_code)
        entry_conditions = self._extract_entry_conditions(strategy_code)
        exit_conditions = self._extract_exit_conditions(strategy_code)
        risk_parameters = self._extract_risk_parameters(strategy_params)
        
        code_hash = hashlib.md5(strategy_code.encode()).hexdigest()
        
        return StrategyFingerprint(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            parameters=strategy_params,
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_parameters=risk_parameters,
            code_hash=code_hash,
            created_at=datetime.now()
        )
    
    def _extract_indicators(self, code: str) -> List[str]:
        """Extract indicators used in strategy code"""
        indicators = []
        common_indicators = [
            'sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stochastic',
            'williams', 'cci', 'momentum', 'roc', 'adx', 'obv', 'vwap'
        ]
        
        code_lower = code.lower()
        for indicator in common_indicators:
            if indicator in code_lower:
                indicators.append(indicator.upper())
        
        return indicators
    
    def _extract_entry_conditions(self, code: str) -> List[str]:
        """Extract entry conditions from strategy code"""
        conditions = []
        
        # Look for common entry patterns
        if 'cross above' in code.lower() or 'crossabove' in code.lower():
            conditions.append('cross_above')
        if 'cross below' in code.lower() or 'crossbelow' in code.lower():
            conditions.append('cross_below')
        if 'breakout' in code.lower():
            conditions.append('breakout')
        if 'oversold' in code.lower():
            conditions.append('oversold')
        if 'overbought' in code.lower():
            conditions.append('overbought')
        
        return conditions
    
    def _extract_exit_conditions(self, code: str) -> List[str]:
        """Extract exit conditions from strategy code"""
        conditions = []
        
        # Look for common exit patterns
        if 'stop loss' in code.lower() or 'stoploss' in code.lower():
            conditions.append('stop_loss')
        if 'take profit' in code.lower() or 'takeprofit' in code.lower():
            conditions.append('take_profit')
        if 'trailing stop' in code.lower():
            conditions.append('trailing_stop')
        if 'time exit' in code.lower():
            conditions.append('time_exit')
        
        return conditions
    
    def _extract_risk_parameters(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Extract risk-related parameters"""
        risk_params = {}
        
        risk_keys = ['stop_loss', 'take_profit', 'position_size', 'risk_per_trade', 'max_positions']
        for key in risk_keys:
            if key in params and isinstance(params[key], (int, float)):
                risk_params[key] = float(params[key])
        
        return risk_params
    
    def _extract_market_conditions(self, market_data: Dict[str, Any]) -> MarketConditions:
        """Extract market conditions from market data"""
        # Default values
        period_start = datetime.now() - timedelta(days=365)
        period_end = datetime.now()
        market_regime = market_data.get('market_regime', 'unknown')
        volatility_level = market_data.get('volatility', 0.2)
        trend_strength = market_data.get('trend_strength', 0.5)
        sector_performance = market_data.get('sector_performance', {})
        economic_indicators = market_data.get('economic_indicators', {})
        
        return MarketConditions(
            period_start=period_start,
            period_end=period_end,
            market_regime=market_regime,
            volatility_level=volatility_level,
            trend_strength=trend_strength,
            sector_performance=sector_performance,
            economic_indicators=economic_indicators
        )
    
    async def cleanup_old_memories(self, days_old: int = 365):
        """Clean up old, low-performing memories"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Remove old strategies with poor performance
            # This would involve database cleanup operations
            self.logger.info(f"Cleaned up memories older than {days_old} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up memories: {str(e)}")

# Global instance
_strategy_memory = None

def get_strategy_memory() -> StrategyMemory:
    """Get global strategy memory instance"""
    global _strategy_memory
    if _strategy_memory is None:
        _strategy_memory = StrategyMemory()
    return _strategy_memory