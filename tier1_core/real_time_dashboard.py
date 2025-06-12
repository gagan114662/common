"""
Real-Time System Dashboard - Complete Visibility
Provides comprehensive monitoring of all system activities including strategy creation and backtesting
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import threading
from pathlib import Path
import websockets
import logging

from tier1_core.logger import get_logger

@dataclass
class ActivityEvent:
    """Real-time activity event"""
    timestamp: datetime
    event_id: str
    event_type: str  # 'strategy_generation', 'backtesting', 'evolution', 'agent_action'
    source: str  # Agent or component name
    action: str  # Specific action being performed
    details: Dict[str, Any]
    status: str  # 'started', 'progress', 'completed', 'failed'
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class StrategyCreationEvent:
    """Detailed strategy creation tracking"""
    strategy_id: str
    template_name: str
    agent_name: str
    creation_start: datetime
    creation_end: Optional[datetime] = None
    parameters: Dict[str, Any] = None
    validation_status: str = 'pending'  # 'pending', 'valid', 'invalid'
    code_length: int = 0
    complexity_score: int = 0
    
    @property
    def creation_duration(self) -> float:
        if self.creation_end:
            return (self.creation_end - self.creation_start).total_seconds()
        return (datetime.now() - self.creation_start).total_seconds()

@dataclass
class BacktestEvent:
    """Detailed backtesting tracking"""
    strategy_id: str
    backtest_id: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    progress: float = 0.0
    status: str = 'queued'  # 'queued', 'compiling', 'running', 'completed', 'failed'
    error_message: Optional[str] = None
    preliminary_results: Optional[Dict[str, float]] = None
    final_results: Optional[Dict[str, float]] = None
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

@dataclass
class AgentActivitySnapshot:
    """Real-time agent activity snapshot"""
    agent_name: str
    current_task: Optional[str]
    task_start_time: Optional[datetime]
    strategies_in_queue: int
    active_backtests: int
    recent_completions: int
    success_rate: float
    average_generation_time: float
    average_backtest_time: float

class RealTimeDashboard:
    """
    Complete real-time visibility system for strategy creation and backtesting
    
    Features:
    - Live activity streaming
    - Detailed progress tracking
    - Performance metrics
    - Error monitoring
    - Agent coordination visibility
    - Resource usage tracking
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Event storage (last 24 hours)
        self.activity_events: deque = deque(maxlen=10000)
        self.strategy_events: Dict[str, StrategyCreationEvent] = {}
        self.backtest_events: Dict[str, BacktestEvent] = {}
        
        # Real-time tracking
        self.active_agents: Dict[str, AgentActivitySnapshot] = {}
        self.system_metrics: Dict[str, Any] = {}
        
        # Performance tracking
        self.generation_stats = {
            'total_generated': 0,
            'total_tested': 0,
            'success_rate': 0.0,
            'average_generation_time': 0.0,
            'average_backtest_time': 0.0,
            'strategies_per_hour': 0.0
        }
        
        # Live subscribers (WebSocket connections)
        self.subscribers: List[Any] = []
        
        # Threading for real-time updates
        self.update_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
    def start(self) -> None:
        """Start the real-time dashboard"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        self.logger.info("Real-time dashboard started")
    
    def stop(self) -> None:
        """Stop the dashboard"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.logger.info("Real-time dashboard stopped")
    
    def _update_loop(self) -> None:
        """Background update loop"""
        while self.running:
            try:
                self._calculate_metrics()
                self._broadcast_updates()
                time.sleep(1)  # Update every second
            except Exception as e:
                self.logger.error(f"Dashboard update error: {str(e)}")
                time.sleep(5)
    
    # Event Recording Methods
    
    def log_strategy_generation_start(self, strategy_id: str, template_name: str, agent_name: str, parameters: Dict[str, Any]) -> None:
        """Log start of strategy generation"""
        event = StrategyCreationEvent(
            strategy_id=strategy_id,
            template_name=template_name,
            agent_name=agent_name,
            creation_start=datetime.now(),
            parameters=parameters
        )
        
        self.strategy_events[strategy_id] = event
        
        activity_event = ActivityEvent(
            timestamp=datetime.now(),
            event_id=f"gen_{strategy_id}",
            event_type='strategy_generation',
            source=agent_name,
            action='generate_strategy',
            details={
                'strategy_id': strategy_id,
                'template': template_name,
                'parameters': parameters
            },
            status='started'
        )
        
        self._record_event(activity_event)
        self.logger.info(f"🔄 [{agent_name}] Started generating strategy {strategy_id[:8]} using {template_name}")
    
    def log_strategy_generation_complete(self, strategy_id: str, code_length: int, complexity_score: int, validation_status: str) -> None:
        """Log completion of strategy generation"""
        if strategy_id in self.strategy_events:
            event = self.strategy_events[strategy_id]
            event.creation_end = datetime.now()
            event.code_length = code_length
            event.complexity_score = complexity_score
            event.validation_status = validation_status
            
            activity_event = ActivityEvent(
                timestamp=datetime.now(),
                event_id=f"gen_complete_{strategy_id}",
                event_type='strategy_generation',
                source=event.agent_name,
                action='generation_complete',
                details={
                    'strategy_id': strategy_id,
                    'duration_seconds': event.creation_duration,
                    'code_length': code_length,
                    'complexity_score': complexity_score,
                    'validation_status': validation_status
                },
                status='completed' if validation_status == 'valid' else 'failed',
                duration_ms=event.creation_duration * 1000
            )
            
            self._record_event(activity_event)
            
            status_emoji = "✅" if validation_status == 'valid' else "❌"
            self.logger.info(
                f"{status_emoji} [{event.agent_name}] Generated strategy {strategy_id[:8]} "
                f"in {event.creation_duration:.2f}s (code: {code_length} chars, complexity: {complexity_score})"
            )
    
    def log_backtest_start(self, strategy_id: str, backtest_id: str, agent_name: str) -> None:
        """Log start of backtesting"""
        event = BacktestEvent(
            strategy_id=strategy_id,
            backtest_id=backtest_id,
            agent_name=agent_name,
            start_time=datetime.now(),
            status='queued'
        )
        
        self.backtest_events[backtest_id] = event
        
        activity_event = ActivityEvent(
            timestamp=datetime.now(),
            event_id=f"backtest_{backtest_id}",
            event_type='backtesting',
            source=agent_name,
            action='start_backtest',
            details={
                'strategy_id': strategy_id,
                'backtest_id': backtest_id
            },
            status='started'
        )
        
        self._record_event(activity_event)
        self.logger.info(f"🧪 [{agent_name}] Started backtest {backtest_id[:8]} for strategy {strategy_id[:8]}")
    
    def log_backtest_progress(self, backtest_id: str, progress: float, status: str, preliminary_results: Optional[Dict[str, float]] = None) -> None:
        """Log backtesting progress"""
        if backtest_id in self.backtest_events:
            event = self.backtest_events[backtest_id]
            event.progress = progress
            event.status = status
            if preliminary_results:
                event.preliminary_results = preliminary_results
            
            activity_event = ActivityEvent(
                timestamp=datetime.now(),
                event_id=f"backtest_progress_{backtest_id}_{int(time.time())}",
                event_type='backtesting',
                source=event.agent_name,
                action='backtest_progress',
                details={
                    'backtest_id': backtest_id,
                    'progress': progress,
                    'status': status,
                    'duration_seconds': event.duration,
                    'preliminary_results': preliminary_results
                },
                status='progress'
            )
            
            self._record_event(activity_event)
            
            if progress % 25 == 0 or preliminary_results:  # Log every 25% or when results available
                self.logger.info(
                    f"📊 [{event.agent_name}] Backtest {backtest_id[:8]} progress: {progress:.1f}% "
                    f"({status}) - {event.duration:.1f}s elapsed"
                )
    
    def log_backtest_complete(self, backtest_id: str, final_results: Dict[str, float], error_message: Optional[str] = None) -> None:
        """Log completion of backtesting"""
        if backtest_id in self.backtest_events:
            event = self.backtest_events[backtest_id]
            event.end_time = datetime.now()
            event.final_results = final_results
            event.error_message = error_message
            event.status = 'completed' if not error_message else 'failed'
            
            activity_event = ActivityEvent(
                timestamp=datetime.now(),
                event_id=f"backtest_complete_{backtest_id}",
                event_type='backtesting',
                source=event.agent_name,
                action='backtest_complete',
                details={
                    'backtest_id': backtest_id,
                    'strategy_id': event.strategy_id,
                    'duration_seconds': event.duration,
                    'final_results': final_results,
                    'error_message': error_message
                },
                status='completed' if not error_message else 'failed',
                duration_ms=event.duration * 1000
            )
            
            self._record_event(activity_event)
            
            if error_message:
                self.logger.error(f"❌ [{event.agent_name}] Backtest {backtest_id[:8]} failed: {error_message}")
            else:
                cagr = final_results.get('cagr', 0) * 100
                sharpe = final_results.get('sharpe', 0)
                drawdown = final_results.get('max_drawdown', 0) * 100
                
                self.logger.info(
                    f"✅ [{event.agent_name}] Backtest {backtest_id[:8]} completed in {event.duration:.1f}s - "
                    f"CAGR: {cagr:.1f}%, Sharpe: {sharpe:.2f}, Drawdown: {drawdown:.1f}%"
                )
    
    def log_agent_activity(self, agent_name: str, activity: str, details: Dict[str, Any]) -> None:
        """Log general agent activity"""
        activity_event = ActivityEvent(
            timestamp=datetime.now(),
            event_id=f"agent_{agent_name}_{int(time.time())}",
            event_type='agent_action',
            source=agent_name,
            action=activity,
            details=details,
            status='completed'
        )
        
        self._record_event(activity_event)
        self.logger.debug(f"🤖 [{agent_name}] {activity}: {details}")
    
    def log_evolution_event(self, generation: int, best_fitness: float, avg_fitness: float, diversity: float) -> None:
        """Log evolution engine events"""
        activity_event = ActivityEvent(
            timestamp=datetime.now(),
            event_id=f"evolution_gen_{generation}",
            event_type='evolution',
            source='evolution_engine',
            action='generation_complete',
            details={
                'generation': generation,
                'best_fitness': best_fitness,
                'average_fitness': avg_fitness,
                'diversity': diversity
            },
            status='completed'
        )
        
        self._record_event(activity_event)
        self.logger.info(
            f"🧬 Evolution Generation {generation} complete - "
            f"Best: {best_fitness:.3f}, Avg: {avg_fitness:.3f}, Diversity: {diversity:.3f}"
        )
    
    def _record_event(self, event: ActivityEvent) -> None:
        """Record an activity event"""
        self.activity_events.append(event)
        
        # Trigger callbacks
        for callback in self.event_callbacks[event.event_type]:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Event callback error: {str(e)}")
    
    def _calculate_metrics(self) -> None:
        """Calculate real-time metrics"""
        now = datetime.now()
        
        # Calculate generation stats
        recent_generations = [
            e for e in self.strategy_events.values()
            if e.creation_end and (now - e.creation_end).total_seconds() < 3600  # Last hour
        ]
        
        recent_backtests = [
            e for e in self.backtest_events.values()
            if e.end_time and (now - e.end_time).total_seconds() < 3600  # Last hour
        ]
        
        if recent_generations:
            self.generation_stats['average_generation_time'] = sum(
                e.creation_duration for e in recent_generations
            ) / len(recent_generations)
            
            self.generation_stats['strategies_per_hour'] = len(recent_generations)
        
        if recent_backtests:
            self.generation_stats['average_backtest_time'] = sum(
                e.duration for e in recent_backtests
            ) / len(recent_backtests)
            
            successful_backtests = [e for e in recent_backtests if e.status == 'completed']
            self.generation_stats['success_rate'] = len(successful_backtests) / len(recent_backtests) * 100
        
        # Update agent snapshots
        self._update_agent_snapshots()
    
    def _update_agent_snapshots(self) -> None:
        """Update real-time agent activity snapshots"""
        now = datetime.now()
        
        # Get unique agents
        agents = set()
        for event in self.activity_events:
            if event.source != 'evolution_engine':
                agents.add(event.source)
        
        for agent_name in agents:
            # Get recent agent events
            agent_events = [
                e for e in self.activity_events
                if e.source == agent_name and (now - e.timestamp).total_seconds() < 3600
            ]
            
            # Calculate metrics
            strategy_events = [e for e in agent_events if e.event_type == 'strategy_generation']
            backtest_events = [e for e in agent_events if e.event_type == 'backtesting']
            
            current_task = None
            task_start_time = None
            
            # Find current task
            for event in reversed(agent_events):
                if event.status == 'started' and event.action in ['generate_strategy', 'start_backtest']:
                    current_task = event.action
                    task_start_time = event.timestamp
                    break
            
            strategies_in_queue = len([e for e in strategy_events if e.status == 'started'])
            active_backtests = len([e for e in backtest_events if e.status in ['started', 'progress']])
            recent_completions = len([e for e in agent_events if e.status == 'completed'])
            
            # Calculate success rate
            completed_events = [e for e in agent_events if e.status in ['completed', 'failed']]
            success_rate = (
                len([e for e in completed_events if e.status == 'completed']) / len(completed_events) * 100
                if completed_events else 0
            )
            
            # Average times
            completed_generations = [e for e in strategy_events if e.status == 'completed' and e.duration_ms]
            avg_generation_time = (
                sum(e.duration_ms for e in completed_generations) / len(completed_generations) / 1000
                if completed_generations else 0
            )
            
            completed_backtests = [e for e in backtest_events if e.status == 'completed' and e.duration_ms]
            avg_backtest_time = (
                sum(e.duration_ms for e in completed_backtests) / len(completed_backtests) / 1000
                if completed_backtests else 0
            )
            
            self.active_agents[agent_name] = AgentActivitySnapshot(
                agent_name=agent_name,
                current_task=current_task,
                task_start_time=task_start_time,
                strategies_in_queue=strategies_in_queue,
                active_backtests=active_backtests,
                recent_completions=recent_completions,
                success_rate=success_rate,
                average_generation_time=avg_generation_time,
                average_backtest_time=avg_backtest_time
            )
    
    def _broadcast_updates(self) -> None:
        """Broadcast updates to all subscribers"""
        if not self.subscribers:
            return
        
        update_data = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self.generation_stats,
            'active_agents': {
                name: asdict(snapshot) for name, snapshot in self.active_agents.items()
            },
            'recent_events': [
                event.to_dict() for event in list(self.activity_events)[-10:]  # Last 10 events
            ]
        }
        
        # Would broadcast to WebSocket subscribers
        # For now, just update internal state
    
    # Query Methods for Live Data
    
    def get_live_dashboard_data(self) -> Dict[str, Any]:
        """Get complete live dashboard data"""
        now = datetime.now()
        
        # Recent activity (last 5 minutes)
        recent_events = [
            event.to_dict() for event in self.activity_events
            if (now - event.timestamp).total_seconds() < 300
        ]
        
        # Active operations
        active_generations = [
            asdict(event) for event in self.strategy_events.values()
            if not event.creation_end
        ]
        
        active_backtests = [
            asdict(event) for event in self.backtest_events.values()
            if event.status in ['queued', 'compiling', 'running']
        ]
        
        return {
            'timestamp': now.isoformat(),
            'system_status': 'running',
            'generation_stats': self.generation_stats,
            'active_agents': {
                name: asdict(snapshot) for name, snapshot in self.active_agents.items()
            },
            'recent_events': recent_events,
            'active_generations': active_generations,
            'active_backtests': active_backtests,
            'performance_summary': self._get_performance_summary()
        }
    
    def get_strategy_timeline(self, strategy_id: str) -> Dict[str, Any]:
        """Get complete timeline for a specific strategy"""
        timeline = []
        
        # Add all events related to this strategy
        for event in self.activity_events:
            if event.details.get('strategy_id') == strategy_id:
                timeline.append(event.to_dict())
        
        # Add creation event if exists
        if strategy_id in self.strategy_events:
            creation_event = self.strategy_events[strategy_id]
            timeline.append({
                'type': 'creation',
                'data': asdict(creation_event)
            })
        
        # Add backtest events
        for backtest_id, backtest_event in self.backtest_events.items():
            if backtest_event.strategy_id == strategy_id:
                timeline.append({
                    'type': 'backtest',
                    'data': asdict(backtest_event)
                })
        
        # Sort by timestamp (handle both string and datetime objects)
        def get_sort_key(x):
            timestamp = x.get('timestamp')
            if timestamp is None:
                # Try to get from data
                data_time = x.get('data', {}).get('creation_start')
                if isinstance(data_time, datetime):
                    return data_time.isoformat()
                return data_time or ''
            if isinstance(timestamp, datetime):
                return timestamp.isoformat()
            return timestamp or ''
        
        timeline.sort(key=get_sort_key)
        
        return {
            'strategy_id': strategy_id,
            'timeline': timeline
        }
    
    def get_agent_details(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific agent"""
        if agent_name not in self.active_agents:
            return {'error': 'Agent not found'}
        
        snapshot = self.active_agents[agent_name]
        
        # Get agent's recent events
        now = datetime.now()
        agent_events = [
            event.to_dict() for event in self.activity_events
            if event.source == agent_name and (now - event.timestamp).total_seconds() < 3600
        ]
        
        return {
            'agent_name': agent_name,
            'current_snapshot': asdict(snapshot),
            'recent_events': agent_events[-20:],  # Last 20 events
            'performance_metrics': self._calculate_agent_metrics(agent_name)
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        now = datetime.now()
        
        # Last hour metrics
        hour_ago = now - timedelta(hours=1)
        recent_events = [e for e in self.activity_events if e.timestamp > hour_ago]
        
        strategies_generated = len([e for e in recent_events if e.event_type == 'strategy_generation' and e.status == 'completed'])
        strategies_tested = len([e for e in recent_events if e.event_type == 'backtesting' and e.status == 'completed'])
        
        return {
            'last_hour': {
                'strategies_generated': strategies_generated,
                'strategies_tested': strategies_tested,
                'generation_rate': strategies_generated,  # Per hour
                'test_rate': strategies_tested
            },
            'current_active': {
                'total_agents': len(self.active_agents),
                'active_generations': len([e for e in self.strategy_events.values() if not e.creation_end]),
                'active_backtests': len([e for e in self.backtest_events.values() if e.status in ['queued', 'running']])
            }
        }
    
    def _calculate_agent_metrics(self, agent_name: str) -> Dict[str, Any]:
        """Calculate detailed metrics for a specific agent"""
        now = datetime.now()
        agent_events = [
            e for e in self.activity_events
            if e.source == agent_name and (now - e.timestamp).total_seconds() < 3600
        ]
        
        return {
            'events_last_hour': len(agent_events),
            'strategies_generated': len([e for e in agent_events if e.event_type == 'strategy_generation']),
            'backtests_run': len([e for e in agent_events if e.event_type == 'backtesting']),
            'success_rate': self.active_agents[agent_name].success_rate,
            'avg_generation_time': self.active_agents[agent_name].average_generation_time,
            'avg_backtest_time': self.active_agents[agent_name].average_backtest_time
        }
    
    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        """Add callback for specific event types"""
        self.event_callbacks[event_type].append(callback)
    
    def export_activity_log(self, filepath: Path, hours: int = 24) -> None:
        """Export activity log to file"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'hours_covered': hours,
            'events': [
                event.to_dict() for event in self.activity_events
                if event.timestamp > cutoff
            ],
            'strategy_events': {
                sid: asdict(event) for sid, event in self.strategy_events.items()
                if event.creation_start > cutoff
            },
            'backtest_events': {
                bid: asdict(event) for bid, event in self.backtest_events.items()
                if event.start_time > cutoff
            },
            'performance_stats': self.generation_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(export_data['events'])} events to {filepath}")

# Global dashboard instance
DASHBOARD = RealTimeDashboard()