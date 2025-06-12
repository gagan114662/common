"""
Intelligent Memory System for Strategy Learning and Adaptation
Implements vector embeddings, knowledge graphs, and performance attribution
"""

import numpy as np
import pandas as pd
import sqlite3
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging

@dataclass
class StrategyFingerprint:
    """Unique fingerprint for strategy identification and similarity"""
    strategy_id: str
    template_name: str
    parameter_hash: str
    code_hash: str
    semantic_features: np.ndarray  # Vector embedding
    complexity_score: int
    asset_class: str
    timeframe: str
    creation_timestamp: datetime
    
@dataclass 
class PerformanceMemory:
    """Performance results with context"""
    strategy_id: str
    backtest_results: Dict[str, float]
    market_conditions: Dict[str, Any]
    attribution_factors: Dict[str, float]
    failure_modes: List[str]
    success_patterns: List[str]
    regime_context: str
    timestamp: datetime

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_id: str
    volatility_level: str  # 'low', 'medium', 'high'
    trend_direction: str   # 'bull', 'bear', 'sideways'
    correlation_environment: str  # 'low', 'medium', 'high'
    macro_context: Dict[str, Any]
    start_date: datetime
    end_date: Optional[datetime] = None

@dataclass
class LearningInsight:
    """Learned insights from strategy performance"""
    insight_id: str
    insight_type: str  # 'parameter_optimization', 'regime_sensitivity', 'failure_pattern'
    description: str
    confidence: float
    supporting_evidence: List[str]
    applicable_templates: List[str]
    discovered_timestamp: datetime

class VectorDatabase:
    """FAISS-based vector database for strategy embeddings"""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.strategy_map: Dict[int, str] = {}  # Maps FAISS index to strategy_id
        self.strategy_metadata: Dict[str, StrategyFingerprint] = {}
        
    def add_strategy(self, fingerprint: StrategyFingerprint) -> None:
        """Add strategy fingerprint to vector database"""
        # Normalize vector for cosine similarity
        vector = fingerprint.semantic_features.reshape(1, -1)
        vector = vector / np.linalg.norm(vector)
        
        # Add to FAISS index
        self.index.add(vector.astype('float32'))
        
        # Update mappings
        faiss_id = self.index.ntotal - 1
        self.strategy_map[faiss_id] = fingerprint.strategy_id
        self.strategy_metadata[fingerprint.strategy_id] = fingerprint
        
    def find_similar_strategies(self, query_fingerprint: StrategyFingerprint, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar strategies"""
        if self.index.ntotal == 0:
            return []
            
        # Normalize query vector
        query_vector = query_fingerprint.semantic_features.reshape(1, -1)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Search FAISS index
        similarities, indices = self.index.search(query_vector.astype('float32'), min(k, self.index.ntotal))
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx in self.strategy_map:
                strategy_id = self.strategy_map[idx]
                results.append((strategy_id, float(sim)))
                
        return results
    
    def get_strategy_fingerprint(self, strategy_id: str) -> Optional[StrategyFingerprint]:
        """Get strategy fingerprint by ID"""
        return self.strategy_metadata.get(strategy_id)

class KnowledgeGraph:
    """Graph-based knowledge representation for strategy relationships"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
    def _create_tables(self) -> None:
        """Create knowledge graph tables"""
        cursor = self.conn.cursor()
        
        # Strategies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                template_name TEXT,
                parameters TEXT,
                creation_time TIMESTAMP,
                complexity_score INTEGER
            )
        ''')
        
        # Relationships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_strategy TEXT,
                target_strategy TEXT,
                relationship_type TEXT,
                strength REAL,
                context TEXT,
                created_time TIMESTAMP,
                FOREIGN KEY (source_strategy) REFERENCES strategies (strategy_id),
                FOREIGN KEY (target_strategy) REFERENCES strategies (strategy_id)
            )
        ''')
        
        # Market regimes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_regimes (
                regime_id TEXT PRIMARY KEY,
                volatility_level TEXT,
                trend_direction TEXT,
                correlation_env TEXT,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                macro_context TEXT
            )
        ''')
        
        # Performance attribution table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT,
                regime_id TEXT,
                performance_metrics TEXT,
                attribution_factors TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id),
                FOREIGN KEY (regime_id) REFERENCES market_regimes (regime_id)
            )
        ''')
        
        self.conn.commit()
    
    def add_strategy_relationship(self, source_id: str, target_id: str, relationship_type: str, strength: float, context: str) -> None:
        """Add relationship between strategies"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO relationships 
            (source_strategy, target_strategy, relationship_type, strength, context, created_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (source_id, target_id, relationship_type, strength, context, datetime.now()))
        self.conn.commit()
    
    def get_related_strategies(self, strategy_id: str, relationship_types: List[str] = None) -> List[Dict[str, Any]]:
        """Get strategies related to given strategy"""
        cursor = self.conn.cursor()
        
        if relationship_types:
            placeholders = ','.join(['?' for _ in relationship_types])
            query = f'''
                SELECT target_strategy, relationship_type, strength, context
                FROM relationships 
                WHERE source_strategy = ? AND relationship_type IN ({placeholders})
                ORDER BY strength DESC
            '''
            cursor.execute(query, [strategy_id] + relationship_types)
        else:
            cursor.execute('''
                SELECT target_strategy, relationship_type, strength, context
                FROM relationships 
                WHERE source_strategy = ?
                ORDER BY strength DESC
            ''', (strategy_id,))
        
        return [
            {
                'strategy_id': row[0],
                'relationship_type': row[1], 
                'strength': row[2],
                'context': row[3]
            }
            for row in cursor.fetchall()
        ]

class PerformanceWarehouse:
    """Structured storage and analysis of backtest results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
    def _create_tables(self) -> None:
        """Create performance warehouse tables"""
        cursor = self.conn.cursor()
        
        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT,
                total_return REAL,
                cagr REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                volatility REAL,
                var_95 REAL,
                beta REAL,
                alpha REAL,
                win_rate REAL,
                profit_factor REAL,
                avg_win REAL,
                avg_loss REAL,
                trades_count INTEGER,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                backtest_timestamp TIMESTAMP
            )
        ''')
        
        # Detailed performance attribution
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT,
                attribution_factor TEXT,
                contribution REAL,
                confidence REAL,
                regime_context TEXT,
                timestamp TIMESTAMP
            )
        ''')
        
        # Failure mode analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failure_modes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT,
                failure_type TEXT,
                description TEXT,
                frequency REAL,
                impact REAL,
                market_conditions TEXT,
                timestamp TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def store_backtest_result(self, strategy_id: str, results: Dict[str, Any]) -> None:
        """Store backtest results"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtest_results (
                strategy_id, total_return, cagr, sharpe_ratio, sortino_ratio,
                max_drawdown, volatility, var_95, beta, alpha, win_rate,
                profit_factor, avg_win, avg_loss, trades_count,
                start_date, end_date, backtest_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_id,
            results.get('total_return', 0),
            results.get('cagr', 0),
            results.get('sharpe_ratio', 0),
            results.get('sortino_ratio', 0),
            results.get('max_drawdown', 0),
            results.get('volatility', 0),
            results.get('var_95', 0),
            results.get('beta', 0),
            results.get('alpha', 0),
            results.get('win_rate', 0),
            results.get('profit_factor', 0),
            results.get('avg_win', 0),
            results.get('avg_loss', 0),
            results.get('trades_count', 0),
            results.get('start_date'),
            results.get('end_date'),
            datetime.now()
        ))
        
        self.conn.commit()
    
    def get_performance_distribution(self, metric: str, template_name: str = None) -> Dict[str, Any]:
        """Get performance distribution for a metric"""
        cursor = self.conn.cursor()
        
        if template_name:
            # Need to join with strategies table to filter by template
            query = f'''
                SELECT {metric} FROM backtest_results br
                JOIN strategies s ON br.strategy_id = s.strategy_id
                WHERE s.template_name = ? AND {metric} IS NOT NULL
            '''
            cursor.execute(query, (template_name,))
        else:
            cursor.execute(f'SELECT {metric} FROM backtest_results WHERE {metric} IS NOT NULL')
        
        values = [row[0] for row in cursor.fetchall()]
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'percentiles': {
                '25': np.percentile(values, 25),
                '75': np.percentile(values, 75),
                '90': np.percentile(values, 90),
                '95': np.percentile(values, 95)
            },
            'count': len(values)
        }

class IntelligentMemorySystem:
    """
    Main memory system orchestrating all components
    """
    
    def __init__(self, data_dir: str = "data/memory"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.vector_db = VectorDatabase()
        self.knowledge_graph = KnowledgeGraph(str(self.data_dir / "knowledge_graph.db"))
        self.performance_warehouse = PerformanceWarehouse(str(self.data_dir / "performance_warehouse.db"))
        
        # Learning state
        self.learning_insights: List[LearningInsight] = []
        self.market_regimes: Dict[str, MarketRegime] = {}
        
        # Feature extraction
        self.tfidf_vectorizer = TfidfVectorizer(max_features=128, stop_words='english')
        self._is_fitted = False
        
        self.logger = logging.getLogger(__name__)
    
    def remember_strategy(self, strategy_code: str, template_name: str, parameters: Dict[str, Any], 
                         performance: Dict[str, Any], market_conditions: Dict[str, Any]) -> str:
        """Remember a strategy and its performance"""
        
        # Create strategy fingerprint
        strategy_id = self._generate_strategy_id(strategy_code, parameters)
        fingerprint = self._create_strategy_fingerprint(
            strategy_id, strategy_code, template_name, parameters
        )
        
        # Store in vector database
        self.vector_db.add_strategy(fingerprint)
        
        # Store in knowledge graph
        self._update_knowledge_graph(strategy_id, template_name, parameters)
        
        # Store performance
        self.performance_warehouse.store_backtest_result(strategy_id, performance)
        
        # Learn from this strategy
        self._extract_learning_insights(strategy_id, performance, market_conditions)
        
        # Find and store relationships
        self._discover_strategy_relationships(fingerprint, performance)
        
        return strategy_id
    
    def recall_similar_strategies(self, strategy_code: str, template_name: str, 
                                parameters: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Recall similar strategies with their performance"""
        
        # Create fingerprint for query
        query_fingerprint = self._create_strategy_fingerprint(
            "query", strategy_code, template_name, parameters
        )
        
        # Find similar strategies
        similar_strategies = self.vector_db.find_similar_strategies(query_fingerprint, k)
        
        results = []
        for strategy_id, similarity in similar_strategies:
            # Get performance data
            performance = self._get_strategy_performance(strategy_id)
            if performance:
                results.append({
                    'strategy_id': strategy_id,
                    'similarity': similarity,
                    'performance': performance,
                    'fingerprint': self.vector_db.get_strategy_fingerprint(strategy_id)
                })
        
        return results
    
    def get_optimization_suggestions(self, template_name: str, current_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get parameter optimization suggestions based on learned patterns"""
        
        suggestions = []
        
        # Find high-performing strategies of same template
        high_performers = self._find_high_performing_strategies(template_name)
        
        for strategy_data in high_performers:
            fingerprint = strategy_data['fingerprint']
            performance = strategy_data['performance']
            
            # Extract parameter suggestions
            param_diff = self._compare_parameters(current_parameters, fingerprint.strategy_id)
            
            if param_diff:
                suggestions.append({
                    'parameter_changes': param_diff,
                    'expected_improvement': self._estimate_improvement(param_diff, performance),
                    'confidence': self._calculate_suggestion_confidence(strategy_data),
                    'source_strategy': fingerprint.strategy_id
                })
        
        # Sort by expected improvement
        suggestions.sort(key=lambda x: x['expected_improvement'], reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
    
    def predict_strategy_performance(self, strategy_code: str, template_name: str, 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Predict strategy performance based on memory"""
        
        # Find similar strategies
        similar_strategies = self.recall_similar_strategies(strategy_code, template_name, parameters, k=10)
        
        if not similar_strategies:
            return {'confidence': 0, 'prediction': None}
        
        # Weight predictions by similarity
        weighted_metrics = defaultdict(float)
        total_weight = 0
        
        for strategy in similar_strategies:
            weight = strategy['similarity']
            performance = strategy['performance']
            
            for metric, value in performance.items():
                if isinstance(value, (int, float)):
                    weighted_metrics[metric] += value * weight
            
            total_weight += weight
        
        # Normalize predictions
        if total_weight > 0:
            predictions = {metric: value / total_weight for metric, value in weighted_metrics.items()}
        else:
            predictions = {}
        
        # Calculate confidence based on similarity and sample size
        confidence = min(1.0, total_weight / len(similar_strategies) * len(similar_strategies) / 10)
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'similar_strategies_count': len(similar_strategies),
            'basis_strategies': [s['strategy_id'] for s in similar_strategies[:3]]
        }
    
    def _generate_strategy_id(self, strategy_code: str, parameters: Dict[str, Any]) -> str:
        """Generate unique strategy ID"""
        code_hash = hashlib.md5(strategy_code.encode()).hexdigest()
        param_hash = hashlib.md5(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        return f"{code_hash[:8]}_{param_hash[:8]}"
    
    def _create_strategy_fingerprint(self, strategy_id: str, strategy_code: str, 
                                   template_name: str, parameters: Dict[str, Any]) -> StrategyFingerprint:
        """Create strategy fingerprint with semantic features"""
        
        # Create semantic features from code
        semantic_features = self._extract_semantic_features(strategy_code)
        
        # Generate hashes
        parameter_hash = hashlib.md5(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        code_hash = hashlib.md5(strategy_code.encode()).hexdigest()
        
        return StrategyFingerprint(
            strategy_id=strategy_id,
            template_name=template_name,
            parameter_hash=parameter_hash,
            code_hash=code_hash,
            semantic_features=semantic_features,
            complexity_score=self._calculate_complexity_score(strategy_code),
            asset_class="Equity",  # Default
            timeframe="Daily",     # Default
            creation_timestamp=datetime.now()
        )
    
    def _extract_semantic_features(self, strategy_code: str) -> np.ndarray:
        """Extract semantic features from strategy code using TF-IDF"""
        
        # Prepare code for feature extraction
        cleaned_code = self._clean_code_for_features(strategy_code)
        
        if not self._is_fitted:
            # First strategy - fit the vectorizer
            features = self.tfidf_vectorizer.fit_transform([cleaned_code])
            self._is_fitted = True
        else:
            # Transform using fitted vectorizer
            features = self.tfidf_vectorizer.transform([cleaned_code])
        
        # Ensure we have the right dimension
        feature_vector = features.toarray()[0]
        
        # Pad or truncate to match vector database dimension
        if len(feature_vector) < self.vector_db.dimension:
            feature_vector = np.pad(feature_vector, (0, self.vector_db.dimension - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.vector_db.dimension]
        
        return feature_vector
    
    def _clean_code_for_features(self, code: str) -> str:
        """Clean strategy code for feature extraction"""
        # Remove comments and extra whitespace
        lines = []
        for line in code.split('\n'):
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if line:
                lines.append(line)
        
        return ' '.join(lines)
    
    def _calculate_complexity_score(self, strategy_code: str) -> int:
        """Calculate strategy complexity score"""
        lines = len([l for l in strategy_code.split('\n') if l.strip()])
        functions = strategy_code.count('def ')
        conditionals = strategy_code.count('if ') + strategy_code.count('elif ') + strategy_code.count('else:')
        loops = strategy_code.count('for ') + strategy_code.count('while ')
        
        return min(10, 1 + lines // 10 + functions + conditionals + loops)
    
    def _update_knowledge_graph(self, strategy_id: str, template_name: str, parameters: Dict[str, Any]) -> None:
        """Update knowledge graph with new strategy"""
        cursor = self.knowledge_graph.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO strategies 
            (strategy_id, template_name, parameters, creation_time, complexity_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            strategy_id,
            template_name, 
            json.dumps(parameters),
            datetime.now(),
            self._calculate_complexity_score("")  # Would need actual code
        ))
        
        self.knowledge_graph.conn.commit()
    
    def _discover_strategy_relationships(self, fingerprint: StrategyFingerprint, performance: Dict[str, Any]) -> None:
        """Discover and store relationships with existing strategies"""
        
        # Find similar strategies
        similar_strategies = self.vector_db.find_similar_strategies(fingerprint, k=5)
        
        for similar_id, similarity in similar_strategies:
            if similar_id == fingerprint.strategy_id:
                continue
                
            # Add similarity relationship
            self.knowledge_graph.add_strategy_relationship(
                fingerprint.strategy_id,
                similar_id,
                "similarity",
                similarity,
                f"Code similarity: {similarity:.3f}"
            )
            
            # Add performance relationship if both performed well
            similar_perf = self._get_strategy_performance(similar_id)
            if similar_perf and performance.get('sharpe_ratio', 0) > 1.0 and similar_perf.get('sharpe_ratio', 0) > 1.0:
                self.knowledge_graph.add_strategy_relationship(
                    fingerprint.strategy_id,
                    similar_id,
                    "performance_correlation",
                    0.8,  # High correlation for both being successful
                    "Both strategies show high Sharpe ratios"
                )
    
    def _get_strategy_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get latest performance results for strategy"""
        cursor = self.performance_warehouse.conn.cursor()
        
        cursor.execute('''
            SELECT * FROM backtest_results 
            WHERE strategy_id = ?
            ORDER BY backtest_timestamp DESC
            LIMIT 1
        ''', (strategy_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # Map to dictionary (assuming column order)
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    
    def _find_high_performing_strategies(self, template_name: str, min_sharpe: float = 1.0) -> List[Dict[str, Any]]:
        """Find high-performing strategies for a template"""
        cursor = self.performance_warehouse.conn.cursor()
        
        cursor.execute('''
            SELECT br.strategy_id, br.sharpe_ratio, br.cagr, br.max_drawdown
            FROM backtest_results br
            JOIN strategies s ON br.strategy_id = s.strategy_id
            WHERE s.template_name = ? AND br.sharpe_ratio >= ?
            ORDER BY br.sharpe_ratio DESC
            LIMIT 10
        ''', (template_name, min_sharpe))
        
        results = []
        for row in cursor.fetchall():
            strategy_id = row[0]
            fingerprint = self.vector_db.get_strategy_fingerprint(strategy_id)
            
            if fingerprint:
                results.append({
                    'strategy_id': strategy_id,
                    'performance': {
                        'sharpe_ratio': row[1],
                        'cagr': row[2], 
                        'max_drawdown': row[3]
                    },
                    'fingerprint': fingerprint
                })
        
        return results
    
    def _compare_parameters(self, current_params: Dict[str, Any], strategy_id: str) -> Dict[str, Any]:
        """Compare parameters and suggest changes"""
        # This would get stored parameters for the strategy and compare
        # Simplified for now
        return {}
    
    def _estimate_improvement(self, param_changes: Dict[str, Any], reference_performance: Dict[str, Any]) -> float:
        """Estimate performance improvement from parameter changes"""
        # Simplified estimation - would use more sophisticated modeling
        return reference_performance.get('sharpe_ratio', 0) * 0.1
    
    def _calculate_suggestion_confidence(self, strategy_data: Dict[str, Any]) -> float:
        """Calculate confidence in parameter suggestion"""
        similarity = strategy_data.get('similarity', 0)
        performance = strategy_data.get('performance', {})
        sharpe = performance.get('sharpe_ratio', 0)
        
        return min(1.0, similarity * 0.5 + (sharpe / 3.0) * 0.5)
    
    def _extract_learning_insights(self, strategy_id: str, performance: Dict[str, Any], 
                                 market_conditions: Dict[str, Any]) -> None:
        """Extract learning insights from strategy performance"""
        
        # Identify high-performing strategies
        if performance.get('sharpe_ratio', 0) > 2.0:
            insight = LearningInsight(
                insight_id=f"high_perf_{strategy_id}",
                insight_type="high_performance_pattern",
                description=f"Strategy {strategy_id} achieved exceptional Sharpe ratio of {performance['sharpe_ratio']:.2f}",
                confidence=0.9,
                supporting_evidence=[strategy_id],
                applicable_templates=[self.vector_db.get_strategy_fingerprint(strategy_id).template_name],
                discovered_timestamp=datetime.now()
            )
            self.learning_insights.append(insight)
        
        # Identify failure patterns
        if performance.get('max_drawdown', 0) > 0.3:
            insight = LearningInsight(
                insight_id=f"high_dd_{strategy_id}",
                insight_type="failure_pattern",
                description=f"Strategy {strategy_id} experienced excessive drawdown of {performance['max_drawdown']:.1%}",
                confidence=0.8,
                supporting_evidence=[strategy_id],
                applicable_templates=[self.vector_db.get_strategy_fingerprint(strategy_id).template_name],
                discovered_timestamp=datetime.now()
            )
            self.learning_insights.append(insight)
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learned insights"""
        
        insight_types = defaultdict(int)
        recent_insights = []
        
        for insight in self.learning_insights:
            insight_types[insight.insight_type] += 1
            
            # Include recent insights
            if (datetime.now() - insight.discovered_timestamp).days <= 7:
                recent_insights.append({
                    'type': insight.insight_type,
                    'description': insight.description,
                    'confidence': insight.confidence
                })
        
        return {
            'total_insights': len(self.learning_insights),
            'insight_types': dict(insight_types),
            'recent_insights': recent_insights[:10],
            'strategies_remembered': self.vector_db.index.ntotal
        }