"""
TIER 1: Performance Monitoring System
Real-time monitoring with 1-second update frequency as recommended by Gemini
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import statistics
from pathlib import Path

from config.settings import PerformanceTargets, SystemRequirements
from tier1_core.logger import get_logger, PERF_LOGGER

@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_threads: int
    open_files: int
    
@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    timestamp: datetime
    strategies_generated: int
    strategies_tested: int
    strategies_per_hour: float
    best_cagr: float
    best_sharpe: float
    best_drawdown: float
    average_backtest_time: float
    api_response_time: float
    success_rate: float

@dataclass
class AlertConfig:
    """Alert configuration"""
    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'eq'
    enabled: bool = True
    callback: Optional[Callable] = None

@dataclass
class Alert:
    """Performance alert"""
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # 'warning', 'critical'
    message: str

class PerformanceMonitor:
    """
    Real-time performance monitoring system
    
    Features:
    - 1-second update frequency
    - System resource monitoring
    - Trading performance tracking
    - Automatic alerting
    - Historical data retention
    - Performance trend analysis
    """
    
    def __init__(self, targets: PerformanceTargets, requirements: SystemRequirements):
        self.targets = targets
        self.requirements = requirements
        self.logger = get_logger(__name__)
        
        # Monitoring state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Metrics storage (last 24 hours with 1-second resolution)
        self.max_history_points = 24 * 60 * 60  # 24 hours
        self.system_metrics: deque = deque(maxlen=self.max_history_points)
        self.trading_metrics: deque = deque(maxlen=self.max_history_points)
        
        # Alert system
        self.alerts: List[Alert] = []
        self.alert_configs: List[AlertConfig] = []
        self.alert_cooldown: Dict[str, datetime] = {}
        self.alert_cooldown_seconds = 300  # 5 minutes
        
        # Performance tracking
        self.strategies_generated_total = 0
        self.strategies_tested_total = 0
        self.backtest_times: deque = deque(maxlen=1000)  # Last 1000 backtests
        self.api_response_times: deque = deque(maxlen=1000)
        
        # System baseline
        self.baseline_metrics: Optional[SystemMetrics] = None
        self.performance_degradation_threshold = 0.20  # 20% degradation
        
        # Initialize default alerts
        self._setup_default_alerts()
        
    def _setup_default_alerts(self) -> None:
        """Setup default performance alerts"""
        # System resource alerts
        self.alert_configs.extend([
            AlertConfig("cpu_percent", 80.0, "gt"),
            AlertConfig("memory_percent", 85.0, "gt"),
            AlertConfig("memory_used_gb", self.requirements.memory_limit_gb * 0.9, "gt"),
            AlertConfig("disk_usage_percent", 90.0, "gt"),
            
            # Performance alerts
            AlertConfig("strategies_per_hour", self.requirements.strategy_generation_rate * 0.8, "lt"),
            AlertConfig("average_backtest_time", self.requirements.backtest_duration_minutes * 60, "gt"),
            AlertConfig("api_response_time", self.requirements.api_response_time_ms / 1000, "gt"),
            
            # Trading performance alerts
            AlertConfig("best_cagr", self.targets.target_cagr, "gt"),  # Positive alert
            AlertConfig("best_sharpe", self.targets.target_sharpe, "gt"),  # Positive alert
            AlertConfig("best_drawdown", self.targets.max_drawdown, "gt"),  # Negative alert
        ])
    
    async def start(self) -> None:
        """Start the performance monitoring system"""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_time = datetime.now()
        
        # Capture baseline metrics
        await self._capture_baseline()
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Performance monitoring started with 1-second frequency")
    
    async def stop(self) -> None:
        """Stop the performance monitoring system"""
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance monitoring stopped")
    
    async def _capture_baseline(self) -> None:
        """Capture baseline system metrics"""
        self.baseline_metrics = await self._collect_system_metrics()
        self.logger.info("Baseline system metrics captured")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop with 1-second frequency"""
        while self.is_running:
            try:
                # Collect metrics
                system_metrics = await self._collect_system_metrics()
                trading_metrics = await self._collect_trading_metrics()
                
                # Store metrics
                self.system_metrics.append(system_metrics)
                self.trading_metrics.append(trading_metrics)
                
                # Check alerts
                await self._check_alerts(system_metrics, trading_metrics)
                
                # Log performance data
                await self._log_performance_data(system_metrics, trading_metrics)
                
                # Wait for next second
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # Get system info
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Get process info
        process = psutil.Process()
        process_info = process.as_dict(['num_threads', 'num_fds'])
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            active_threads=process_info.get('num_threads', 0),
            open_files=process_info.get('num_fds', 0)
        )
    
    async def _collect_trading_metrics(self) -> TradingMetrics:
        """Collect current trading performance metrics"""
        now = datetime.now()
        
        # Calculate strategies per hour
        if self.start_time:
            hours_running = (now - self.start_time).total_seconds() / 3600
            strategies_per_hour = (
                self.strategies_generated_total / hours_running 
                if hours_running > 0 else 0
            )
        else:
            strategies_per_hour = 0
        
        # Calculate average backtest time
        average_backtest_time = (
            statistics.mean(self.backtest_times) 
            if self.backtest_times else 0
        )
        
        # Calculate average API response time
        api_response_time = (
            statistics.mean(self.api_response_times) 
            if self.api_response_times else 0
        )
        
        # Calculate success rate (placeholder - to be updated by controller)
        success_rate = (
            self.strategies_tested_total / self.strategies_generated_total * 100
            if self.strategies_generated_total > 0 else 0
        )
        
        # Get best performance (placeholder - to be updated by controller)
        best_cagr = 0.0
        best_sharpe = 0.0
        best_drawdown = 1.0
        
        return TradingMetrics(
            timestamp=now,
            strategies_generated=self.strategies_generated_total,
            strategies_tested=self.strategies_tested_total,
            strategies_per_hour=strategies_per_hour,
            best_cagr=best_cagr,
            best_sharpe=best_sharpe,
            best_drawdown=best_drawdown,
            average_backtest_time=average_backtest_time,
            api_response_time=api_response_time,
            success_rate=success_rate
        )
    
    async def _check_alerts(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics) -> None:
        """Check for alert conditions"""
        for config in self.alert_configs:
            if not config.enabled:
                continue
                
            # Check cooldown
            if config.metric_name in self.alert_cooldown:
                if datetime.now() - self.alert_cooldown[config.metric_name] < timedelta(seconds=self.alert_cooldown_seconds):
                    continue
            
            # Get current value
            current_value = self._get_metric_value(config.metric_name, system_metrics, trading_metrics)
            if current_value is None:
                continue
            
            # Check condition
            alert_triggered = False
            if config.comparison == "gt" and current_value > config.threshold_value:
                alert_triggered = True
            elif config.comparison == "lt" and current_value < config.threshold_value:
                alert_triggered = True
            elif config.comparison == "eq" and abs(current_value - config.threshold_value) < 0.001:
                alert_triggered = True
            
            if alert_triggered:
                await self._trigger_alert(config, current_value, system_metrics, trading_metrics)
    
    def _get_metric_value(self, metric_name: str, system_metrics: SystemMetrics, trading_metrics: TradingMetrics) -> Optional[float]:
        """Get metric value by name"""
        # System metrics
        if hasattr(system_metrics, metric_name):
            return getattr(system_metrics, metric_name)
        
        # Trading metrics
        if hasattr(trading_metrics, metric_name):
            return getattr(trading_metrics, metric_name)
        
        return None
    
    async def _trigger_alert(
        self, 
        config: AlertConfig, 
        current_value: float, 
        system_metrics: SystemMetrics, 
        trading_metrics: TradingMetrics
    ) -> None:
        """Trigger a performance alert"""
        # Determine severity
        if config.metric_name in ["cpu_percent", "memory_percent", "disk_usage_percent"]:
            severity = "critical" if current_value > 90 else "warning"
        elif config.metric_name in ["best_cagr", "best_sharpe"]:
            severity = "info"  # Positive alerts
        else:
            severity = "warning"
        
        # Create alert
        alert = Alert(
            timestamp=datetime.now(),
            metric_name=config.metric_name,
            current_value=current_value,
            threshold_value=config.threshold_value,
            severity=severity,
            message=self._generate_alert_message(config, current_value)
        )
        
        # Store alert
        self.alerts.append(alert)
        
        # Set cooldown
        self.alert_cooldown[config.metric_name] = datetime.now()
        
        # Log alert
        log_level = "warning" if severity in ["warning", "critical"] else "info"
        getattr(self.logger, log_level)(alert.message)
        
        # Execute callback if provided
        if config.callback:
            try:
                await config.callback(alert, system_metrics, trading_metrics)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {str(e)}")
    
    def _generate_alert_message(self, config: AlertConfig, current_value: float) -> str:
        """Generate human-readable alert message"""
        if config.metric_name == "cpu_percent":
            return f"High CPU usage: {current_value:.1f}% (threshold: {config.threshold_value:.1f}%)"
        elif config.metric_name == "memory_percent":
            return f"High memory usage: {current_value:.1f}% (threshold: {config.threshold_value:.1f}%)"
        elif config.metric_name == "memory_used_gb":
            return f"Memory usage: {current_value:.1f}GB (limit: {config.threshold_value:.1f}GB)"
        elif config.metric_name == "strategies_per_hour":
            return f"Low strategy generation rate: {current_value:.1f}/hour (target: {config.threshold_value:.1f}/hour)"
        elif config.metric_name == "average_backtest_time":
            return f"Slow backtesting: {current_value:.1f}s (target: <{config.threshold_value:.1f}s)"
        elif config.metric_name == "api_response_time":
            return f"Slow API responses: {current_value*1000:.1f}ms (target: <{config.threshold_value*1000:.1f}ms)"
        elif config.metric_name == "best_cagr":
            return f"🎉 CAGR target achieved: {current_value:.2%} (target: {config.threshold_value:.2%})"
        elif config.metric_name == "best_sharpe":
            return f"🎉 Sharpe target achieved: {current_value:.2f} (target: {config.threshold_value:.2f})"
        elif config.metric_name == "best_drawdown":
            return f"⚠️ High drawdown: {current_value:.2%} (limit: {config.threshold_value:.2%})"
        else:
            return f"Alert: {config.metric_name} = {current_value:.2f} (threshold: {config.threshold_value:.2f})"
    
    async def _log_performance_data(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics) -> None:
        """Log performance data for analysis"""
        # Log every 60 seconds to avoid overwhelming logs
        if len(self.system_metrics) % 60 == 0:
            PERF_LOGGER.log_system_performance({
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "memory_used_gb": system_metrics.memory_used_gb,
                "disk_usage_percent": system_metrics.disk_usage_percent,
                "active_threads": system_metrics.active_threads,
                "strategies_generated": trading_metrics.strategies_generated,
                "strategies_tested": trading_metrics.strategies_tested,
                "strategies_per_hour": trading_metrics.strategies_per_hour,
                "average_backtest_time": trading_metrics.average_backtest_time,
                "api_response_time": trading_metrics.api_response_time,
                "success_rate": trading_metrics.success_rate
            })
    
    async def update_metrics(self, system_status: Any) -> None:
        """Update metrics from system status"""
        self.strategies_generated_total = system_status.strategies_generated
        self.strategies_tested_total = system_status.strategies_tested
    
    def add_backtest_time(self, duration_seconds: float) -> None:
        """Record a backtest duration"""
        self.backtest_times.append(duration_seconds)
    
    def add_api_response_time(self, duration_seconds: float) -> None:
        """Record an API response time"""
        self.api_response_times.append(duration_seconds)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.system_metrics or not self.trading_metrics:
            return {}
        
        latest_system = self.system_metrics[-1]
        latest_trading = self.trading_metrics[-1]
        
        return {
            "timestamp": latest_system.timestamp.isoformat(),
            "system": {
                "cpu_percent": latest_system.cpu_percent,
                "memory_percent": latest_system.memory_percent,
                "memory_used_gb": latest_system.memory_used_gb,
                "disk_usage_percent": latest_system.disk_usage_percent,
                "active_threads": latest_system.active_threads,
                "open_files": latest_system.open_files
            },
            "trading": {
                "strategies_generated": latest_trading.strategies_generated,
                "strategies_tested": latest_trading.strategies_tested,
                "strategies_per_hour": latest_trading.strategies_per_hour,
                "best_cagr": latest_trading.best_cagr,
                "best_sharpe": latest_trading.best_sharpe,
                "best_drawdown": latest_trading.best_drawdown,
                "average_backtest_time": latest_trading.average_backtest_time,
                "api_response_time": latest_trading.api_response_time,
                "success_rate": latest_trading.success_rate
            },
            "targets_status": {
                "cagr_achieved": latest_trading.best_cagr >= self.targets.target_cagr,
                "sharpe_achieved": latest_trading.best_sharpe >= self.targets.target_sharpe,
                "drawdown_within_limit": latest_trading.best_drawdown <= self.targets.max_drawdown,
                "generation_rate_met": latest_trading.strategies_per_hour >= self.requirements.strategy_generation_rate
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_metrics = self.get_current_metrics()
        
        # Calculate uptime
        uptime_seconds = (
            (datetime.now() - self.start_time).total_seconds()
            if self.start_time else 0
        )
        
        # Recent alerts
        recent_alerts = [
            {
                "timestamp": alert.timestamp.isoformat(),
                "metric": alert.metric_name,
                "severity": alert.severity,
                "message": alert.message
            }
            for alert in self.alerts[-10:]  # Last 10 alerts
        ]
        
        return {
            "uptime_seconds": uptime_seconds,
            "current_metrics": current_metrics,
            "recent_alerts": recent_alerts,
            "total_alerts": len(self.alerts),
            "monitoring_points": len(self.system_metrics),
            "performance_degradation": self._calculate_performance_degradation()
        }
    
    def _calculate_performance_degradation(self) -> Dict[str, float]:
        """Calculate performance degradation from baseline"""
        if not self.baseline_metrics or not self.system_metrics:
            return {}
        
        current = self.system_metrics[-1]
        baseline = self.baseline_metrics
        
        return {
            "cpu_degradation": (current.cpu_percent - baseline.cpu_percent) / baseline.cpu_percent,
            "memory_degradation": (current.memory_percent - baseline.memory_percent) / baseline.memory_percent,
            "overall_health": min(100.0, max(0.0, 100.0 - current.cpu_percent - current.memory_percent))
        }