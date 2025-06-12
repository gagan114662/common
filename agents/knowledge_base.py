"""
Multi-Agent Shared Knowledge Base
Central repository for agents to share discoveries, market insights, and performance data
Based on Gemini's recommendation for collaborative agent coordination
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict, deque
import numpy as np
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from tier1_core.logger import get_logger

Base = declarative_base()

@dataclass
class MarketInsight:
    """Market insight discovered by an agent"""
    insight_id: str
    agent_name: str
    category: str  # 'trend', 'anomaly', 'correlation', 'regime_change', 'opportunity'
    asset_class: str
    symbols: List[str]
    timeframe: str
    description: str
    confidence: float  # 0.0 to 1.0
    validity_period: timedelta
    supporting_data: Dict[str, Any]
    timestamp: datetime
    
    def is_valid(self) -> bool:
        """Check if insight is still valid"""
        return datetime.now() < self.timestamp + self.validity_period

@dataclass
class StrategyDiscovery:
    """Successful strategy discovered by an agent"""
    discovery_id: str
    agent_name: str
    template_name: str
    category: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]  # cagr, sharpe, drawdown, etc.
    market_conditions: Dict[str, Any]  # when it works best
    complementary_strategies: List[str]  # IDs of strategies that work well together
    risk_profile: Dict[str, float]
    timestamp: datetime

@dataclass
class AgentMessage:
    """Message between agents"""
    message_id: str
    sender: str
    recipient: str  # 'all' for broadcast
    message_type: str  # 'discovery', 'warning', 'request', 'response'
    content: Dict[str, Any]
    priority: int  # 1-5, 1 being highest
    timestamp: datetime

# SQLAlchemy Models
class MarketInsightDB(Base):
    __tablename__ = 'market_insights'
    
    insight_id = Column(String, primary_key=True)
    agent_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    asset_class = Column(String, nullable=False)
    symbols = Column(Text, nullable=False)  # JSON
    timeframe = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    validity_hours = Column(Integer, nullable=False)
    supporting_data = Column(Text, nullable=False)  # JSON
    timestamp = Column(DateTime, nullable=False)

class StrategyDiscoveryDB(Base):
    __tablename__ = 'strategy_discoveries'
    
    discovery_id = Column(String, primary_key=True)
    agent_name = Column(String, nullable=False)
    template_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    parameters = Column(Text, nullable=False)  # JSON
    performance_metrics = Column(Text, nullable=False)  # JSON
    market_conditions = Column(Text, nullable=False)  # JSON
    complementary_strategies = Column(Text, nullable=False)  # JSON
    risk_profile = Column(Text, nullable=False)  # JSON
    timestamp = Column(DateTime, nullable=False)

class AgentPerformanceDB(Base):
    __tablename__ = 'agent_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)

class SharedKnowledgeBase:
    """
    Central knowledge repository for multi-agent collaboration
    
    Features:
    - Market insight sharing and validation
    - Strategy discovery tracking
    - Agent communication system
    - Performance metric aggregation
    - Collaborative learning support
    - Real-time data synchronization
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.logger = get_logger(__name__)
        
        # Database setup
        if db_path is None:
            db_path = Path("data/knowledge_base.db")
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database engine with thread safety
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # In-memory caches for fast access
        self.insights_cache: Dict[str, MarketInsight] = {}
        self.discoveries_cache: Dict[str, StrategyDiscovery] = {}
        self.message_queue: deque[AgentMessage] = deque(maxlen=1000)
        
        # Agent tracking
        self.active_agents: Set[str] = set()
        self.agent_contributions: defaultdict[str, int] = defaultdict(int)
        
        # Synchronization
        self.lock = threading.Lock()
        
        # Performance tracking
        self.total_insights = 0
        self.total_discoveries = 0
        self.total_messages = 0
        
        self.logger.info("Shared knowledge base initialized")
    
    async def register_agent(self, agent_name: str) -> None:
        """Register an agent with the knowledge base"""
        with self.lock:
            self.active_agents.add(agent_name)
        self.logger.info(f"Agent '{agent_name}' registered with knowledge base")
    
    async def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent from the knowledge base"""
        with self.lock:
            self.active_agents.discard(agent_name)
        self.logger.info(f"Agent '{agent_name}' unregistered from knowledge base")
    
    async def add_market_insight(self, insight: MarketInsight) -> bool:
        """Add a new market insight to the knowledge base"""
        try:
            # Validate insight
            if not self._validate_insight(insight):
                return False
            
            # Add to cache
            with self.lock:
                self.insights_cache[insight.insight_id] = insight
                self.agent_contributions[insight.agent_name] += 1
                self.total_insights += 1
            
            # Persist to database
            with self.SessionLocal() as session:
                db_insight = MarketInsightDB(
                    insight_id=insight.insight_id,
                    agent_name=insight.agent_name,
                    category=insight.category,
                    asset_class=insight.asset_class,
                    symbols=json.dumps(insight.symbols),
                    timeframe=insight.timeframe,
                    description=insight.description,
                    confidence=insight.confidence,
                    validity_hours=int(insight.validity_period.total_seconds() / 3600),
                    supporting_data=json.dumps(insight.supporting_data),
                    timestamp=insight.timestamp
                )
                session.add(db_insight)
                session.commit()
            
            # Broadcast to other agents
            await self._broadcast_insight(insight)
            
            self.logger.info(
                f"Added market insight '{insight.insight_id}' from {insight.agent_name}: "
                f"{insight.description[:50]}..."
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add market insight: {str(e)}")
            return False
    
    async def add_strategy_discovery(self, discovery: StrategyDiscovery) -> bool:
        """Add a new strategy discovery to the knowledge base"""
        try:
            # Validate discovery
            if not self._validate_discovery(discovery):
                return False
            
            # Add to cache
            with self.lock:
                self.discoveries_cache[discovery.discovery_id] = discovery
                self.agent_contributions[discovery.agent_name] += 1
                self.total_discoveries += 1
            
            # Persist to database
            with self.SessionLocal() as session:
                db_discovery = StrategyDiscoveryDB(
                    discovery_id=discovery.discovery_id,
                    agent_name=discovery.agent_name,
                    template_name=discovery.template_name,
                    category=discovery.category,
                    parameters=json.dumps(discovery.parameters),
                    performance_metrics=json.dumps(discovery.performance_metrics),
                    market_conditions=json.dumps(discovery.market_conditions),
                    complementary_strategies=json.dumps(discovery.complementary_strategies),
                    risk_profile=json.dumps(discovery.risk_profile),
                    timestamp=discovery.timestamp
                )
                session.add(db_discovery)
                session.commit()
            
            # Analyze for complementary strategies
            await self._analyze_strategy_synergies(discovery)
            
            self.logger.info(
                f"Added strategy discovery '{discovery.discovery_id}' from {discovery.agent_name}: "
                f"{discovery.template_name} with Sharpe {discovery.performance_metrics.get('sharpe', 0):.2f}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add strategy discovery: {str(e)}")
            return False
    
    async def get_relevant_insights(
        self, 
        agent_name: str, 
        category: Optional[str] = None,
        asset_class: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> List[MarketInsight]:
        """Get relevant market insights for an agent"""
        insights = []
        
        with self.lock:
            for insight in self.insights_cache.values():
                # Skip own insights
                if insight.agent_name == agent_name:
                    continue
                
                # Check validity
                if not insight.is_valid():
                    continue
                
                # Apply filters
                if category and insight.category != category:
                    continue
                
                if asset_class and insight.asset_class != asset_class:
                    continue
                
                if insight.confidence < min_confidence:
                    continue
                
                insights.append(insight)
        
        # Sort by confidence and recency
        insights.sort(key=lambda x: (x.confidence, x.timestamp), reverse=True)
        
        return insights[:20]  # Return top 20 insights
    
    async def get_complementary_strategies(
        self, 
        strategy_id: str, 
        max_correlation: float = 0.7
    ) -> List[StrategyDiscovery]:
        """Get strategies that complement the given strategy"""
        complementary = []
        
        if strategy_id not in self.discoveries_cache:
            return []
        
        base_strategy = self.discoveries_cache[strategy_id]
        
        with self.lock:
            for discovery in self.discoveries_cache.values():
                if discovery.discovery_id == strategy_id:
                    continue
                
                # Check if explicitly marked as complementary
                if strategy_id in discovery.complementary_strategies:
                    complementary.append(discovery)
                    continue
                
                # Check for low correlation based on category
                if self._calculate_strategy_correlation(base_strategy, discovery) < max_correlation:
                    complementary.append(discovery)
        
        # Sort by performance
        complementary.sort(
            key=lambda x: x.performance_metrics.get('sharpe', 0), 
            reverse=True
        )
        
        return complementary[:10]
    
    async def send_agent_message(self, message: AgentMessage) -> None:
        """Send a message between agents"""
        with self.lock:
            self.message_queue.append(message)
            self.total_messages += 1
        
        # Wake up recipient agents
        if message.recipient != 'all':
            await self._notify_agent(message.recipient, message)
        else:
            for agent in self.active_agents:
                if agent != message.sender:
                    await self._notify_agent(agent, message)
    
    async def get_agent_messages(self, agent_name: str, since: Optional[datetime] = None) -> List[AgentMessage]:
        """Get messages for a specific agent"""
        messages = []
        
        with self.lock:
            for message in self.message_queue:
                # Check recipient
                if message.recipient != 'all' and message.recipient != agent_name:
                    continue
                
                # Check timestamp
                if since and message.timestamp < since:
                    continue
                
                # Skip own messages
                if message.sender == agent_name:
                    continue
                
                messages.append(message)
        
        # Sort by priority and timestamp
        messages.sort(key=lambda x: (-x.priority, x.timestamp))
        
        return messages
    
    async def record_agent_performance(
        self, 
        agent_name: str, 
        metrics: Dict[str, float]
    ) -> None:
        """Record agent performance metrics"""
        try:
            with self.SessionLocal() as session:
                for metric_name, value in metrics.items():
                    perf_record = AgentPerformanceDB(
                        agent_name=agent_name,
                        metric_name=metric_name,
                        value=value,
                        timestamp=datetime.now()
                    )
                    session.add(perf_record)
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to record agent performance: {str(e)}")
    
    async def get_agent_rankings(self) -> Dict[str, Dict[str, Any]]:
        """Get agent performance rankings"""
        rankings = {}
        
        try:
            with self.SessionLocal() as session:
                # Get latest performance metrics for each agent
                for agent in self.active_agents:
                    # Get contributions
                    contributions = self.agent_contributions[agent]
                    
                    # Get average performance metrics from database
                    metrics = {}
                    for metric_name in ['strategies_discovered', 'average_sharpe', 'success_rate']:
                        result = session.query(
                            AgentPerformanceDB.value
                        ).filter(
                            AgentPerformanceDB.agent_name == agent,
                            AgentPerformanceDB.metric_name == metric_name
                        ).order_by(
                            AgentPerformanceDB.timestamp.desc()
                        ).first()
                        
                        if result:
                            metrics[metric_name] = result[0]
                        else:
                            metrics[metric_name] = 0.0
                    
                    rankings[agent] = {
                        'contributions': contributions,
                        'metrics': metrics,
                        'score': self._calculate_agent_score(contributions, metrics)
                    }
            
            # Sort by score
            sorted_rankings = dict(sorted(
                rankings.items(), 
                key=lambda x: x[1]['score'], 
                reverse=True
            ))
            
            return sorted_rankings
            
        except Exception as e:
            self.logger.error(f"Failed to get agent rankings: {str(e)}")
            return {}
    
    def _validate_insight(self, insight: MarketInsight) -> bool:
        """Validate market insight"""
        if not insight.insight_id or not insight.agent_name:
            return False
        
        if insight.confidence < 0 or insight.confidence > 1:
            return False
        
        if not insight.symbols or not insight.description:
            return False
        
        return True
    
    def _validate_discovery(self, discovery: StrategyDiscovery) -> bool:
        """Validate strategy discovery"""
        if not discovery.discovery_id or not discovery.agent_name:
            return False
        
        if not discovery.parameters or not discovery.performance_metrics:
            return False
        
        # Check for required performance metrics
        required_metrics = ['cagr', 'sharpe', 'drawdown']
        for metric in required_metrics:
            if metric not in discovery.performance_metrics:
                return False
        
        return True
    
    async def _broadcast_insight(self, insight: MarketInsight) -> None:
        """Broadcast new insight to all agents"""
        message = AgentMessage(
            message_id=f"broadcast_{insight.insight_id}",
            sender="knowledge_base",
            recipient="all",
            message_type="discovery",
            content={
                "type": "market_insight",
                "insight": asdict(insight)
            },
            priority=2 if insight.confidence > 0.8 else 3,
            timestamp=datetime.now()
        )
        
        await self.send_agent_message(message)
    
    async def _analyze_strategy_synergies(self, discovery: StrategyDiscovery) -> None:
        """Analyze synergies between strategies"""
        synergistic_strategies = []
        
        with self.lock:
            for other_discovery in self.discoveries_cache.values():
                if other_discovery.discovery_id == discovery.discovery_id:
                    continue
                
                # Check for low correlation
                correlation = self._calculate_strategy_correlation(discovery, other_discovery)
                
                if correlation < 0.5:  # Low correlation = good diversification
                    synergistic_strategies.append(other_discovery.discovery_id)
        
        # Update complementary strategies
        discovery.complementary_strategies.extend(synergistic_strategies[:5])
    
    def _calculate_strategy_correlation(
        self, 
        strategy1: StrategyDiscovery, 
        strategy2: StrategyDiscovery
    ) -> float:
        """Calculate correlation between strategies based on characteristics"""
        # Simplified correlation based on category and parameters
        if strategy1.category == strategy2.category:
            # Same category - check parameter similarity
            param_similarity = self._calculate_parameter_similarity(
                strategy1.parameters, 
                strategy2.parameters
            )
            return 0.5 + param_similarity * 0.5
        else:
            # Different categories - generally lower correlation
            return 0.2
    
    def _calculate_parameter_similarity(
        self, 
        params1: Dict[str, Any], 
        params2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between parameter sets"""
        common_keys = set(params1.keys()) & set(params2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = params1[key], params2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                diff = abs(val1 - val2)
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarities.append(1 - diff / max_val)
            elif val1 == val2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return sum(similarities) / len(similarities)
    
    def _calculate_agent_score(
        self, 
        contributions: int, 
        metrics: Dict[str, float]
    ) -> float:
        """Calculate overall agent score"""
        # Weighted scoring
        contribution_score = min(contributions / 100, 1.0) * 20
        discovery_score = metrics.get('strategies_discovered', 0) * 2
        sharpe_score = metrics.get('average_sharpe', 0) * 30
        success_score = metrics.get('success_rate', 0) * 20
        
        return contribution_score + discovery_score + sharpe_score + success_score
    
    async def _notify_agent(self, agent_name: str, message: AgentMessage) -> None:
        """Notify an agent about a new message"""
        # This would integrate with the agent's message processing system
        # For now, just log it
        self.logger.debug(f"Notifying agent '{agent_name}' about message from '{message.sender}'")
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base contents"""
        # Clean expired insights
        valid_insights = [i for i in self.insights_cache.values() if i.is_valid()]
        
        # Category breakdowns
        insight_categories = defaultdict(int)
        for insight in valid_insights:
            insight_categories[insight.category] += 1
        
        strategy_categories = defaultdict(int)
        for discovery in self.discoveries_cache.values():
            strategy_categories[discovery.category] += 1
        
        # Performance summary
        if self.discoveries_cache:
            avg_sharpe = np.mean([
                d.performance_metrics.get('sharpe', 0) 
                for d in self.discoveries_cache.values()
            ])
            avg_cagr = np.mean([
                d.performance_metrics.get('cagr', 0) 
                for d in self.discoveries_cache.values()
            ])
            best_sharpe = max([
                d.performance_metrics.get('sharpe', 0) 
                for d in self.discoveries_cache.values()
            ])
        else:
            avg_sharpe = avg_cagr = best_sharpe = 0.0
        
        return {
            "total_insights": self.total_insights,
            "valid_insights": len(valid_insights),
            "total_discoveries": self.total_discoveries,
            "total_messages": self.total_messages,
            "active_agents": len(self.active_agents),
            "insight_categories": dict(insight_categories),
            "strategy_categories": dict(strategy_categories),
            "performance_summary": {
                "average_sharpe": avg_sharpe,
                "average_cagr": avg_cagr,
                "best_sharpe": best_sharpe
            },
            "top_contributors": dict(sorted(
                self.agent_contributions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])
        }
    
    async def cleanup(self) -> None:
        """Clean up expired data"""
        # Remove expired insights
        with self.lock:
            expired_insights = [
                insight_id 
                for insight_id, insight in self.insights_cache.items() 
                if not insight.is_valid()
            ]
            
            for insight_id in expired_insights:
                del self.insights_cache[insight_id]
        
        self.logger.info(f"Cleaned up {len(expired_insights)} expired insights")

    # Methods required by test_knowledge_base_integration.py
    def get_insights_by_type(self, insight_category: str, max_age_seconds: Optional[int] = None) -> List[MarketInsight]:
        """
        Retrieves insights of a specific category, optionally filtered by maximum age.
        The test uses 'insight_type', which we map to 'category'.
        """
        results = []
        with self.lock:
            for insight in self.insights_cache.values():
                if insight.category == insight_category:
                    if max_age_seconds is not None:
                        if datetime.now() - insight.timestamp > timedelta(seconds=max_age_seconds):
                            continue # Skip if older than max_age_seconds
                    if insight.is_valid(): # Also respect built-in validity period
                        results.append(insight)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results

    def get_latest_insight_by_symbol_type(self, symbol: str, insight_category: str, max_age_seconds: Optional[int] = None) -> Optional[MarketInsight]:
        """
        Retrieves the latest insight for a specific symbol and category, optionally filtered by maximum age.
        The test uses 'insight_type', which we map to 'category'.
        """
        latest_insight: Optional[MarketInsight] = None
        with self.lock:
            for insight in self.insights_cache.values():
                if insight.category == insight_category and symbol in insight.symbols:
                    if max_age_seconds is not None:
                        if datetime.now() - insight.timestamp > timedelta(seconds=max_age_seconds):
                            continue
                    if insight.is_valid():
                        if latest_insight is None or insight.timestamp > latest_insight.timestamp:
                            latest_insight = insight
        return latest_insight

    def get_all_insights_for_symbol(self, symbol: str, max_age_seconds: Optional[int] = None) -> List[MarketInsight]:
        """Retrieves all valid insights for a specific symbol, optionally filtered by maximum age."""
        results = []
        with self.lock:
            for insight in self.insights_cache.values():
                if symbol in insight.symbols:
                    if max_age_seconds is not None:
                        if datetime.now() - insight.timestamp > timedelta(seconds=max_age_seconds):
                            continue
                    if insight.is_valid():
                         results.append(insight)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results