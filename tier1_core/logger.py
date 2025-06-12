"""
TIER 1: Logging System
Comprehensive logging system with QuantConnect integration
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

from config.settings import SYSTEM_CONFIG

def setup_logging() -> logging.Logger:
    """Setup comprehensive logging system"""
    config = SYSTEM_CONFIG.logging
    
    # Create logs directory
    logs_dir = Path(SYSTEM_CONFIG.project_root) / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Main logger
    logger = logging.getLogger("evolution_system")
    logger.setLevel(getattr(logging, config.level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "evolution_system.log",
        maxBytes=config.max_file_size_mb * 1024 * 1024,
        backupCount=config.backup_count
    )
    file_handler.setLevel(getattr(logging, config.level))
    file_formatter = logging.Formatter(config.format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "errors.log",
        maxBytes=config.max_file_size_mb * 1024 * 1024,
        backupCount=config.backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    # Performance log handler
    perf_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "performance.log",
        maxBytes=config.max_file_size_mb * 1024 * 1024,
        backupCount=config.backup_count
    )
    perf_handler.setLevel(logging.INFO)
    perf_formatter = logging.Formatter(
        "%(asctime)s - PERFORMANCE - %(message)s"
    )
    perf_handler.setFormatter(perf_formatter)
    
    # Create performance logger
    perf_logger = logging.getLogger("evolution_system.performance")
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    logger.info("Logging system initialized")
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger for the specified module"""
    return logging.getLogger(f"evolution_system.{name}")

class QuantConnectLogger:
    """Logger that respects QuantConnect's 3MB daily limit"""
    
    def __init__(self, daily_limit_mb: float = 3.0):
        self.daily_limit_bytes = daily_limit_mb * 1024 * 1024
        self.daily_usage = 0
        self.last_reset = datetime.now().date()
        self.logger = get_logger("quantconnect")
        
    def log(self, message: str, level: str = "INFO") -> bool:
        """Log message if within daily limit"""
        # Reset daily usage if new day
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_usage = 0
            self.last_reset = today
            
        # Check if we can log this message
        message_size = len(message.encode('utf-8'))
        if self.daily_usage + message_size > self.daily_limit_bytes:
            return False
            
        # Log the message
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        self.daily_usage += message_size
        
        return True
        
    def get_usage_stats(self) -> dict:
        """Get current usage statistics"""
        return {
            "daily_usage_mb": self.daily_usage / (1024 * 1024),
            "daily_limit_mb": self.daily_limit_bytes / (1024 * 1024),
            "usage_percentage": (self.daily_usage / self.daily_limit_bytes) * 100,
            "remaining_mb": (self.daily_limit_bytes - self.daily_usage) / (1024 * 1024)
        }

class PerformanceLogger:
    """Specialized logger for performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger("evolution_system.performance")
        
    def log_strategy_performance(self, strategy_id: str, metrics: dict):
        """Log strategy performance metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "metrics": metrics
        }
        self.logger.info(f"STRATEGY_PERFORMANCE: {json.dumps(log_data)}")
        
    def log_system_performance(self, metrics: dict):
        """Log system performance metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "system_performance",
            "metrics": metrics
        }
        self.logger.info(f"SYSTEM_PERFORMANCE: {json.dumps(log_data)}")
        
    def log_agent_performance(self, agent_name: str, metrics: dict):
        """Log agent performance metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "metrics": metrics
        }
        self.logger.info(f"AGENT_PERFORMANCE: {json.dumps(log_data)}")

class SecurityLogger:
    """Security event logger"""
    
    def __init__(self):
        self.logger = get_logger("security")
        
    def log_auth_attempt(self, user_id: str, success: bool, details: str = ""):
        """Log authentication attempt"""
        self.logger.warning(f"AUTH_ATTEMPT: user={user_id}, success={success}, details={details}")
        
    def log_api_call(self, endpoint: str, user_id: str, success: bool):
        """Log API call"""
        self.logger.info(f"API_CALL: endpoint={endpoint}, user={user_id}, success={success}")
        
    def log_security_event(self, event_type: str, details: dict):
        """Log security event"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self.logger.warning(f"SECURITY_EVENT: {json.dumps(log_data)}")

# Global logger instances
QC_LOGGER = QuantConnectLogger()
PERF_LOGGER = PerformanceLogger()
SECURITY_LOGGER = SecurityLogger()