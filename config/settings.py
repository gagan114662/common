"""
Configuration settings for the 3-Tier Evolutionary Trading System
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
TEMPLATES_DIR = PROJECT_ROOT / "data" / "strategy_templates"

# QuantConnect Configuration
@dataclass
class QuantConnectConfig:
    """QuantConnect API configuration"""
    user_id: str = "357130"
    token: str = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    api_url: str = "https://www.quantconnect.com/api/v2"
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 60

# Performance targets
@dataclass
class PerformanceTargets:
    """Target performance metrics"""
    target_cagr: float = 0.25  # 25% CAGR
    target_sharpe: float = 1.0  # 1.0+ Sharpe ratio
    max_drawdown: float = 0.15  # <15% maximum drawdown
    min_win_rate: float = 0.55  # 55% minimum win rate
    max_volatility: float = 0.20  # 20% maximum volatility

# System performance requirements  
@dataclass
class SystemRequirements:
    """Technical performance requirements"""
    strategy_generation_rate: int = 100  # strategies per hour
    backtest_duration_minutes: int = 30  # max backtest time
    api_response_time_ms: int = 100  # max API response time
    memory_limit_gb: int = 16  # max memory usage
    initialization_time_seconds: int = 60  # max startup time
    
# Evolution parameters
@dataclass  
class EvolutionConfig:
    """Genetic algorithm configuration"""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 2.0
    elitism_rate: float = 0.1
    
# Multi-agent configuration
@dataclass
class AgentConfig:
    """Multi-agent system configuration"""
    agents: List[str] = field(default_factory=lambda: [
        "supervisor",
        "trend_following", 
        "mean_reversion",
        "momentum",
        "arbitrage",
        "market_neutral"
    ])
    max_concurrent_agents: int = 6
    communication_timeout: int = 10
    
# Backtesting configuration
@dataclass
class BacktestConfig:
    """Backtesting parameters"""
    start_date: str = "2009-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100000.0
    benchmark: str = "SPY"
    resolution: str = "Daily"
    data_normalization: str = "Adjusted"
    
# Data configuration
@dataclass
class DataConfig:
    """Data sources and management"""
    supported_assets: List[str] = field(default_factory=lambda: [
        "Equities", "Options", "Futures", "CFD", "Forex", "Crypto"
    ])
    data_providers: List[str] = field(default_factory=lambda: [
        "QuantConnect", "IEX", "Alpha Vantage", "Polygon", "Alpaca"
    ])
    cache_duration_hours: int = 24
    max_cache_size_gb: int = 10
    
# Logging configuration
@dataclass
class LoggingConfig:
    """Logging settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size_mb: int = 10
    backup_count: int = 5
    log_to_quantconnect: bool = True
    quantconnect_log_limit_mb: int = 3  # QC daily limit
    
# Security configuration
@dataclass
class SecurityConfig:
    """Security and encryption settings"""
    encrypt_credentials: bool = True
    api_key_rotation_days: int = 30
    max_failed_auth_attempts: int = 3
    session_timeout_minutes: int = 60
    audit_log_retention_days: int = 90

# Main system configuration
@dataclass
class SystemConfig:
    """Main system configuration"""
    quantconnect: QuantConnectConfig = field(default_factory=QuantConnectConfig)
    performance: PerformanceTargets = field(default_factory=PerformanceTargets)
    requirements: SystemRequirements = field(default_factory=SystemRequirements)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Environment variables override
    def __post_init__(self):
        self.project_root = Path(__file__).parent.parent
        """Override with environment variables if available"""
        if os.getenv("QC_USER_ID"):
            self.quantconnect.user_id = os.getenv("QC_USER_ID")
        if os.getenv("QC_TOKEN"):
            self.quantconnect.token = os.getenv("QC_TOKEN")
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL")

# Global configuration instance
SYSTEM_CONFIG = SystemConfig()

# Validation
def validate_config() -> bool:
    """Validate configuration settings"""
    config = SYSTEM_CONFIG
    
    # Required fields
    if not config.quantconnect.user_id or not config.quantconnect.token:
        raise ValueError("QuantConnect credentials are required")
        
    # Performance targets validation
    if config.performance.target_cagr <= 0:
        raise ValueError("Target CAGR must be positive")
        
    if config.performance.target_sharpe <= 0:
        raise ValueError("Target Sharpe ratio must be positive")
        
    # System requirements validation
    if config.requirements.strategy_generation_rate <= 0:
        raise ValueError("Strategy generation rate must be positive")
        
    return True

# Initialize configuration
try:
    validate_config()
    print("✅ Configuration validation successful")
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
    raise