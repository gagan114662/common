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
    max_drawdown: float = 0.20  # <15% maximum drawdown
    min_win_rate: float = 0.55  # 55% minimum win rate
    max_volatility: float = 0.20  # 20% maximum volatility
    target_average_profit_per_trade: float = 0.0075
    risk_free_rate: float = 0.05

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
    
# Agent-specific settings

@dataclass
class TrendFollowingAgentSettings:
    name: str = "TrendFollowingAgent"
    category: str = "trend_following"
    initial_symbols_watchlist: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL"])
    data_fetch_period: str = "1y"
    sma_short_window: int = 20
    sma_long_window: int = 50
    trend_strength_threshold: float = 0.05
    min_data_points_for_trend: int = 60
    run_cycle_interval_seconds: int = 3600
    max_concurrent_tasks: int = 3 # Copied from base_agent.AgentConfig default for now
    generation_batch_size: int = 0 # Default as TFA is not generating strategies in this version
    min_sharpe_threshold: float = 0.0
    min_cagr_threshold: float = 0.0
    risk_tolerance: float = 0.5
    exploration_rate: float = 0.0
    communication_frequency: int = 60

@dataclass
class MeanReversionAgentSettings:
    name: str = "MeanReversionAgent"
    category: str = "mean_reversion" # Added category for consistency with BaseAgent.config
    version: str = "1.0" # From test
    description: str = "Mean Reversion Agent" # From test
    initial_symbols_watchlist: List[str] = field(default_factory=lambda: ["MSFT", "GOOG"]) # Default example
    data_fetch_period: str = "60d"
    bollinger_window: int = 20
    bollinger_std_dev: float = 2.0
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    min_volume_threshold: int = 100000
    opportunity_score_threshold: float = 0.6
    run_cycle_interval_seconds: int = 3600 # Added for consistency
    # BaseAgent config fields that might be part of its specific config section
    max_concurrent_tasks: int = 3
    generation_batch_size: int = 0
    min_sharpe_threshold: float = 0.0
    min_cagr_threshold: float = 0.0
    risk_tolerance: float = 0.5
    exploration_rate: float = 0.0
    communication_frequency: int = 60


# Container for all agent-specific configurations
@dataclass
class AgentConfigs:
    trend_following: TrendFollowingAgentSettings = field(default_factory=TrendFollowingAgentSettings)
    mean_reversion: MeanReversionAgentSettings = field(default_factory=MeanReversionAgentSettings)
    research_hypothesis: "ResearchAgentSettings" = field(default_factory=lambda: ResearchAgentSettings()) # Added
    # Add other agents as needed:
    # supervisor: SupervisorAgentSettings = field(default_factory=SupervisorAgentSettings) # Example
    # momentum: MomentumAgentSettings = field(default_factory=MomentumAgentSettings)
    # arbitrage: ArbitrageAgentSettings = field(default_factory=ArbitrageAgentSettings)
    # market_neutral: MarketNeutralAgentSettings = field(default_factory=MarketNeutralAgentSettings)

@dataclass
class ResearchAgentSettings:
    name: str = "ResearchHypothesisAgent"
    category: str = "research" # For BaseAgent config
    version: str = "1.0"
    description: str = "Generates and validates research hypotheses."
    research_interval_hours: int = 6
    max_active_hypotheses: int = 10
    min_confidence_threshold: float = 0.65
    firecrawl_api_key: Optional[str] = os.getenv("FIRECRAWL_API_KEY") # For actual client
    firecrawl_max_pages: int = 5 # For scraping depth
    reasoning_model_provider: str = "openai" # e.g., "openai", "gemini"
    reasoning_model_name: str = "gpt-4-turbo"
    # BaseAgent config fields
    max_concurrent_tasks: int = 2
    generation_batch_size: int = 0
    min_sharpe_threshold: float = 0.0
    min_cagr_threshold: float = 0.0
    risk_tolerance: float = 0.5
    exploration_rate: float = 0.0
    communication_frequency: int = 300


# Global agent system settings (if any are left that are not per-agent type)
@dataclass
class GlobalAgentSystemSettings:
    """Global settings for the multi-agent system if needed"""
    # Example: maybe max_total_concurrent_agents_across_system: int = 20
    # For now, the old AgentConfig fields are moved into individual agent settings or are general enough.
    # The list of agent *names* that was in the old AgentConfig might be derived from AgentConfigs keys.
    max_concurrent_agents: int = 6 # This was in old AgentConfig, maybe it's a global cap?
    communication_timeout: int = 10 # This was in old AgentConfig

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
    agents: AgentConfigs = field(default_factory=AgentConfigs) # Changed here
    global_agent_settings: GlobalAgentSystemSettings = field(default_factory=GlobalAgentSystemSettings) # Added
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