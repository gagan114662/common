"""
Catastrophic Event Circuit Breakers for 100% Target Guarantee
Emergency protection systems that halt trading during extreme market events
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

from tier1_core.logger import get_logger

class ThreatLevel(Enum):
    """System threat levels"""
    GREEN = "green"       # Normal operation
    YELLOW = "yellow"     # Elevated monitoring
    ORANGE = "orange"     # Heightened alert
    RED = "red"          # Emergency - halt trading
    BLACK = "black"      # System shutdown required

class CircuitBreakerType(Enum):
    """Types of circuit breakers"""
    MARKET_VOLATILITY = "market_volatility"
    PORTFOLIO_LOSS = "portfolio_loss"
    SYSTEM_OVERLOAD = "system_overload"
    API_FAILURE = "api_failure"
    STRATEGY_DIVERGENCE = "strategy_divergence"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    NETWORK_FAILURE = "network_failure"
    DATA_ANOMALY = "data_anomaly"

@dataclass
class CircuitBreakerRule:
    """Configuration for a circuit breaker"""
    breaker_type: CircuitBreakerType
    threshold_value: float
    time_window_seconds: int
    action: str  # "halt", "reduce", "alert", "shutdown"
    cooldown_seconds: int
    enabled: bool = True

@dataclass
class CircuitBreakerEvent:
    """Record of a circuit breaker activation"""
    timestamp: datetime
    breaker_type: CircuitBreakerType
    trigger_value: float
    threshold_value: float
    threat_level: ThreatLevel
    action_taken: str
    recovery_time: Optional[datetime] = None

class CatastrophicEventProtector:
    """
    Advanced circuit breaker system for protecting against catastrophic events
    
    Features:
    - Real-time market monitoring
    - Multi-level threat assessment
    - Automatic trading halts
    - Portfolio protection
    - System resource monitoring
    - Gradual recovery mechanisms
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.is_active = False
        self.current_threat_level = ThreatLevel.GREEN
        
        # Circuit breaker rules
        self.breaker_rules: Dict[CircuitBreakerType, CircuitBreakerRule] = {}
        self.active_breakers: Dict[CircuitBreakerType, datetime] = {}
        self.breaker_history: List[CircuitBreakerEvent] = []
        
        # Monitoring data
        self.market_data_buffer: Dict[str, List[Tuple[datetime, float]]] = {}
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        self.system_metrics_history: List[Tuple[datetime, Dict[str, float]]] = []
        
        # Emergency callbacks
        self.emergency_callbacks: List[Callable] = []
        self.trading_halt_callback: Optional[Callable] = None
        self.recovery_callback: Optional[Callable] = None
        
        # State tracking
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_timestamp: Optional[datetime] = None
        
        # Initialize default rules
        self._setup_default_circuit_breakers()
    
    def _setup_default_circuit_breakers(self) -> None:
        """Setup default circuit breaker rules"""
        
        # Market volatility breaker - VIX-style
        self.breaker_rules[CircuitBreakerType.MARKET_VOLATILITY] = CircuitBreakerRule(
            breaker_type=CircuitBreakerType.MARKET_VOLATILITY,
            threshold_value=50.0,  # VIX equivalent > 50
            time_window_seconds=300,  # 5 minutes
            action="halt",
            cooldown_seconds=1800  # 30 minutes
        )
        
        # Portfolio loss breaker
        self.breaker_rules[CircuitBreakerType.PORTFOLIO_LOSS] = CircuitBreakerRule(
            breaker_type=CircuitBreakerType.PORTFOLIO_LOSS,
            threshold_value=0.05,  # 5% portfolio loss
            time_window_seconds=3600,  # 1 hour
            action="halt",
            cooldown_seconds=3600  # 1 hour
        )
        
        # System overload breaker
        self.breaker_rules[CircuitBreakerType.SYSTEM_OVERLOAD] = CircuitBreakerRule(
            breaker_type=CircuitBreakerType.SYSTEM_OVERLOAD,
            threshold_value=90.0,  # 90% CPU/Memory
            time_window_seconds=600,  # 10 minutes
            action="reduce",
            cooldown_seconds=1200  # 20 minutes
        )
        
        # API failure breaker
        self.breaker_rules[CircuitBreakerType.API_FAILURE] = CircuitBreakerRule(
            breaker_type=CircuitBreakerType.API_FAILURE,
            threshold_value=0.5,  # 50% failure rate
            time_window_seconds=300,  # 5 minutes
            action="halt",
            cooldown_seconds=600  # 10 minutes
        )
        
        # Strategy divergence breaker
        self.breaker_rules[CircuitBreakerType.STRATEGY_DIVERGENCE] = CircuitBreakerRule(
            breaker_type=CircuitBreakerType.STRATEGY_DIVERGENCE,
            threshold_value=3.0,  # 3 standard deviations
            time_window_seconds=1800,  # 30 minutes
            action="alert",
            cooldown_seconds=900  # 15 minutes
        )
        
        # Memory exhaustion breaker
        self.breaker_rules[CircuitBreakerType.MEMORY_EXHAUSTION] = CircuitBreakerRule(
            breaker_type=CircuitBreakerType.MEMORY_EXHAUSTION,
            threshold_value=95.0,  # 95% memory usage
            time_window_seconds=60,  # 1 minute
            action="shutdown",
            cooldown_seconds=300  # 5 minutes
        )
        
        # Data anomaly breaker
        self.breaker_rules[CircuitBreakerType.DATA_ANOMALY] = CircuitBreakerRule(
            breaker_type=CircuitBreakerType.DATA_ANOMALY,
            threshold_value=5.0,  # 5 sigma event
            time_window_seconds=60,  # 1 minute
            action="halt",
            cooldown_seconds=1800  # 30 minutes
        )
    
    async def start_monitoring(self) -> None:
        """Start the circuit breaker monitoring system"""
        if self.is_active:
            return
        
        self.is_active = True
        self.logger.info("🛡️ Catastrophic event protection system activated")
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop the circuit breaker monitoring system"""
        self.is_active = False
        self.logger.info("Circuit breaker monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_active:
            try:
                # Check all circuit breakers
                await self._check_all_breakers()
                
                # Update threat level
                self._update_threat_level()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep for next check
                await asyncio.sleep(1.0)  # 1-second monitoring frequency
                
            except Exception as e:
                self.logger.error(f"Error in circuit breaker monitoring: {str(e)}")
                await asyncio.sleep(5.0)
    
    async def _check_all_breakers(self) -> None:
        """Check all circuit breaker conditions"""
        
        for breaker_type, rule in self.breaker_rules.items():
            if not rule.enabled:
                continue
            
            # Skip if in cooldown
            if breaker_type in self.active_breakers:
                cooldown_end = self.active_breakers[breaker_type] + timedelta(seconds=rule.cooldown_seconds)
                if datetime.now() < cooldown_end:
                    continue
                else:
                    # Remove from active breakers
                    del self.active_breakers[breaker_type]
            
            # Check specific breaker condition
            trigger_value = await self._check_breaker_condition(breaker_type, rule)
            
            if trigger_value is not None and trigger_value > rule.threshold_value:
                await self._trigger_circuit_breaker(breaker_type, rule, trigger_value)
    
    async def _check_breaker_condition(
        self, 
        breaker_type: CircuitBreakerType, 
        rule: CircuitBreakerRule
    ) -> Optional[float]:
        """Check specific circuit breaker condition"""
        
        if breaker_type == CircuitBreakerType.MARKET_VOLATILITY:
            return await self._check_market_volatility(rule)
        elif breaker_type == CircuitBreakerType.PORTFOLIO_LOSS:
            return await self._check_portfolio_loss(rule)
        elif breaker_type == CircuitBreakerType.SYSTEM_OVERLOAD:
            return await self._check_system_overload(rule)
        elif breaker_type == CircuitBreakerType.API_FAILURE:
            return await self._check_api_failure(rule)
        elif breaker_type == CircuitBreakerType.STRATEGY_DIVERGENCE:
            return await self._check_strategy_divergence(rule)
        elif breaker_type == CircuitBreakerType.MEMORY_EXHAUSTION:
            return await self._check_memory_exhaustion(rule)
        elif breaker_type == CircuitBreakerType.DATA_ANOMALY:
            return await self._check_data_anomaly(rule)
        
        return None
    
    async def _check_market_volatility(self, rule: CircuitBreakerRule) -> Optional[float]:
        """Check market volatility levels"""
        
        # Get recent market data
        cutoff_time = datetime.now() - timedelta(seconds=rule.time_window_seconds)
        
        # Calculate implied volatility from price movements
        all_returns = []
        for symbol, data_points in self.market_data_buffer.items():
            recent_data = [(ts, price) for ts, price in data_points if ts >= cutoff_time]
            
            if len(recent_data) >= 10:
                prices = [price for _, price in recent_data]
                returns = np.diff(prices) / np.array(prices[:-1])
                all_returns.extend(returns)
        
        if len(all_returns) >= 10:
            # Calculate annualized volatility (VIX-style)
            volatility = np.std(all_returns) * np.sqrt(252 * 24 * 60)  # Annualized
            return volatility * 100  # Convert to percentage
        
        return None
    
    async def _check_portfolio_loss(self, rule: CircuitBreakerRule) -> Optional[float]:
        """Check portfolio loss levels"""
        
        if len(self.portfolio_value_history) < 2:
            return None
        
        cutoff_time = datetime.now() - timedelta(seconds=rule.time_window_seconds)
        recent_values = [(ts, value) for ts, value in self.portfolio_value_history if ts >= cutoff_time]
        
        if len(recent_values) >= 2:
            start_value = recent_values[0][1]
            current_value = recent_values[-1][1]
            
            loss_percentage = (start_value - current_value) / start_value
            return loss_percentage if loss_percentage > 0 else 0.0
        
        return None
    
    async def _check_system_overload(self, rule: CircuitBreakerRule) -> Optional[float]:
        """Check system resource overload"""
        
        cutoff_time = datetime.now() - timedelta(seconds=rule.time_window_seconds)
        recent_metrics = [(ts, metrics) for ts, metrics in self.system_metrics_history if ts >= cutoff_time]
        
        if len(recent_metrics) >= 5:
            # Calculate average resource usage
            cpu_usage = [metrics.get('cpu_percent', 0) for _, metrics in recent_metrics]
            memory_usage = [metrics.get('memory_percent', 0) for _, metrics in recent_metrics]
            
            avg_cpu = np.mean(cpu_usage)
            avg_memory = np.mean(memory_usage)
            
            # Return the higher of CPU or memory usage
            return max(avg_cpu, avg_memory)
        
        return None
    
    async def _check_api_failure(self, rule: CircuitBreakerRule) -> Optional[float]:
        """Check API failure rates"""
        
        # This would be populated by the QuantConnect client
        # For now, return a placeholder
        return None
    
    async def _check_strategy_divergence(self, rule: CircuitBreakerRule) -> Optional[float]:
        """Check for unusual strategy performance divergence"""
        
        # This would analyze strategy performance vs expectations
        # For now, return a placeholder
        return None
    
    async def _check_memory_exhaustion(self, rule: CircuitBreakerRule) -> Optional[float]:
        """Check memory exhaustion levels"""
        
        if len(self.system_metrics_history) > 0:
            latest_metrics = self.system_metrics_history[-1][1]
            return latest_metrics.get('memory_percent', 0)
        
        return None
    
    async def _check_data_anomaly(self, rule: CircuitBreakerRule) -> Optional[float]:
        """Check for data anomalies (outliers, missing data, etc.)"""
        
        # Analyze recent market data for anomalies
        anomaly_scores = []
        
        for symbol, data_points in self.market_data_buffer.items():
            if len(data_points) >= 20:
                prices = [price for _, price in data_points[-20:]]
                returns = np.diff(prices) / np.array(prices[:-1])
                
                # Calculate z-score of latest return
                if len(returns) > 1:
                    latest_return = returns[-1]
                    mean_return = np.mean(returns[:-1])
                    std_return = np.std(returns[:-1])
                    
                    if std_return > 0:
                        z_score = abs((latest_return - mean_return) / std_return)
                        anomaly_scores.append(z_score)
        
        if anomaly_scores:
            return max(anomaly_scores)
        
        return None
    
    async def _trigger_circuit_breaker(
        self, 
        breaker_type: CircuitBreakerType, 
        rule: CircuitBreakerRule, 
        trigger_value: float
    ) -> None:
        """Trigger a circuit breaker"""
        
        # Determine threat level
        if rule.action == "shutdown":
            threat_level = ThreatLevel.BLACK
        elif rule.action == "halt":
            threat_level = ThreatLevel.RED
        elif rule.action == "reduce":
            threat_level = ThreatLevel.ORANGE
        else:
            threat_level = ThreatLevel.YELLOW
        
        # Create event record
        event = CircuitBreakerEvent(
            timestamp=datetime.now(),
            breaker_type=breaker_type,
            trigger_value=trigger_value,
            threshold_value=rule.threshold_value,
            threat_level=threat_level,
            action_taken=rule.action
        )
        
        self.breaker_history.append(event)
        self.active_breakers[breaker_type] = datetime.now()
        
        # Execute action
        await self._execute_emergency_action(rule.action, breaker_type, event)
        
        # Log event
        self.logger.critical(
            f"🚨 CIRCUIT BREAKER TRIGGERED: {breaker_type.value} | "
            f"Value: {trigger_value:.4f} | Threshold: {rule.threshold_value:.4f} | "
            f"Action: {rule.action} | Threat Level: {threat_level.value}"
        )
    
    async def _execute_emergency_action(
        self, 
        action: str, 
        breaker_type: CircuitBreakerType, 
        event: CircuitBreakerEvent
    ) -> None:
        """Execute emergency action"""
        
        if action == "halt":
            await self._halt_trading(f"Circuit breaker: {breaker_type.value}")
            
        elif action == "reduce":
            await self._reduce_system_load()
            
        elif action == "shutdown":
            await self._emergency_shutdown(f"Critical event: {breaker_type.value}")
            
        elif action == "alert":
            await self._send_alert(event)
        
        # Execute registered emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                await callback(action, breaker_type, event)
            except Exception as e:
                self.logger.error(f"Emergency callback failed: {str(e)}")
    
    async def _halt_trading(self, reason: str) -> None:
        """Halt all trading activities"""
        
        if self.trading_halted:
            return
        
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_timestamp = datetime.now()
        
        self.logger.critical(f"🛑 TRADING HALTED: {reason}")
        
        # Execute halt callback
        if self.trading_halt_callback:
            try:
                await self.trading_halt_callback(reason)
            except Exception as e:
                self.logger.error(f"Trading halt callback failed: {str(e)}")
    
    async def _reduce_system_load(self) -> None:
        """Reduce system computational load"""
        
        self.logger.warning("⚡ Reducing system load to prevent overload")
        
        # This would implement load reduction strategies:
        # - Reduce strategy generation rate
        # - Pause non-critical background tasks
        # - Limit concurrent operations
        # Implementation depends on system architecture
    
    async def _emergency_shutdown(self, reason: str) -> None:
        """Emergency system shutdown"""
        
        self.logger.critical(f"💀 EMERGENCY SHUTDOWN: {reason}")
        
        # This would trigger a controlled system shutdown
        # - Close all positions
        # - Save state
        # - Notify administrators
        # - Shut down system components
    
    async def _send_alert(self, event: CircuitBreakerEvent) -> None:
        """Send alert notification"""
        
        self.logger.warning(
            f"⚠️ ALERT: {event.breaker_type.value} - "
            f"Value: {event.trigger_value:.4f}, Threshold: {event.threshold_value:.4f}"
        )
    
    def _update_threat_level(self) -> None:
        """Update overall system threat level"""
        
        # Count active breakers by severity
        active_events = [
            event for event in self.breaker_history[-10:]  # Last 10 events
            if event.timestamp > datetime.now() - timedelta(hours=1)  # Last hour
        ]
        
        if any(event.threat_level == ThreatLevel.BLACK for event in active_events):
            new_level = ThreatLevel.BLACK
        elif any(event.threat_level == ThreatLevel.RED for event in active_events):
            new_level = ThreatLevel.RED
        elif any(event.threat_level == ThreatLevel.ORANGE for event in active_events):
            new_level = ThreatLevel.ORANGE
        elif any(event.threat_level == ThreatLevel.YELLOW for event in active_events):
            new_level = ThreatLevel.YELLOW
        else:
            new_level = ThreatLevel.GREEN
        
        if new_level != self.current_threat_level:
            self.logger.info(f"Threat level changed: {self.current_threat_level.value} → {new_level.value}")
            self.current_threat_level = new_level
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data"""
        
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours
        
        # Clean market data
        for symbol in list(self.market_data_buffer.keys()):
            self.market_data_buffer[symbol] = [
                (ts, price) for ts, price in self.market_data_buffer[symbol]
                if ts >= cutoff_time
            ]
        
        # Clean portfolio history
        self.portfolio_value_history = [
            (ts, value) for ts, value in self.portfolio_value_history
            if ts >= cutoff_time
        ]
        
        # Clean system metrics
        self.system_metrics_history = [
            (ts, metrics) for ts, metrics in self.system_metrics_history
            if ts >= cutoff_time
        ]
        
        # Clean old events (keep 1000 most recent)
        if len(self.breaker_history) > 1000:
            self.breaker_history = self.breaker_history[-1000:]
    
    # Public interface methods
    
    def add_market_data(self, symbol: str, price: float) -> None:
        """Add market data point for monitoring"""
        
        if symbol not in self.market_data_buffer:
            self.market_data_buffer[symbol] = []
        
        self.market_data_buffer[symbol].append((datetime.now(), price))
        
        # Keep only recent data
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.market_data_buffer[symbol] = [
            (ts, p) for ts, p in self.market_data_buffer[symbol] 
            if ts >= cutoff_time
        ]
    
    def add_portfolio_value(self, value: float) -> None:
        """Add portfolio value for monitoring"""
        
        self.portfolio_value_history.append((datetime.now(), value))
    
    def add_system_metrics(self, metrics: Dict[str, float]) -> None:
        """Add system metrics for monitoring"""
        
        self.system_metrics_history.append((datetime.now(), metrics))
    
    def register_emergency_callback(self, callback: Callable) -> None:
        """Register emergency callback function"""
        
        self.emergency_callbacks.append(callback)
    
    def set_trading_halt_callback(self, callback: Callable) -> None:
        """Set trading halt callback"""
        
        self.trading_halt_callback = callback
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker system status"""
        
        return {
            "is_active": self.is_active,
            "current_threat_level": self.current_threat_level.value,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "halt_timestamp": self.halt_timestamp.isoformat() if self.halt_timestamp else None,
            "active_breakers": len(self.active_breakers),
            "total_events": len(self.breaker_history),
            "recent_events": len([
                e for e in self.breaker_history
                if e.timestamp > datetime.now() - timedelta(hours=1)
            ])
        }
    
    async def reset_breaker(self, breaker_type: CircuitBreakerType) -> bool:
        """Manually reset a circuit breaker"""
        
        if breaker_type in self.active_breakers:
            del self.active_breakers[breaker_type]
            self.logger.info(f"Circuit breaker reset: {breaker_type.value}")
            return True
        
        return False
    
    async def resume_trading(self, authorization_code: str) -> bool:
        """Resume trading after manual authorization"""
        
        # In production, this would require proper authorization
        if self.trading_halted and authorization_code == "EMERGENCY_OVERRIDE":
            self.trading_halted = False
            self.halt_reason = ""
            self.halt_timestamp = None
            
            self.logger.warning("🟢 Trading resumed after manual authorization")
            
            if self.recovery_callback:
                try:
                    await self.recovery_callback()
                except Exception as e:
                    self.logger.error(f"Recovery callback failed: {str(e)}")
            
            return True
        
        return False