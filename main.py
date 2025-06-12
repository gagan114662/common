#!/usr/bin/env python3
"""
3-Tier Evolutionary Trading System - Main Entry Point
Version: 1.0.0
Author: Multi-AI Collaboration Team

This is the main entry point for the sophisticated algorithmic trading system
that leverages QuantConnect's cloud platform for strategy generation, optimization,
and deployment.

PERFORMANCE TARGETS:
- 25% CAGR (Compound Annual Growth Rate)
- 1.0+ Sharpe Ratio (Risk-adjusted return) 
- <15% Maximum Drawdown (Risk management)

TECHNICAL REQUIREMENTS:
- Strategy Generation Rate: 100+ strategies per hour
- 15-year historical validation (2009-2024) in <30 minutes
- API Response Time: <100ms average
- Memory Usage: <16GB peak
- System Initialization: <1 minute
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tier1_core.controller import SystemController
from config.settings import SYSTEM_CONFIG
from tier1_core.logger import setup_logging

__version__ = "1.0.0"

def banner():
    """Display system banner"""
    print("=" * 80)
    print(f"🚀 3-TIER EVOLUTIONARY TRADING SYSTEM v{__version__}")
    print("=" * 80)
    print("Target Performance: 25% CAGR | 1.0+ Sharpe | <15% Drawdown")
    print("QuantConnect Integration | Multi-Agent Architecture")
    print("User ID: 357130")
    print("=" * 80)
    print()

async def main():
    """Main system entry point"""
    controller = None
    logger = None
    
    try:
        # Display banner
        banner()
        
        # Setup logging
        logger = setup_logging()
        logger.info(f"Starting 3-Tier Evolutionary Trading System v{__version__}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"User ID: {SYSTEM_CONFIG.quantconnect.user_id}")
        
        # Initialize system controller
        logger.info("Creating system controller...")
        controller = SystemController()
        
        # Start system initialization
        logger.info("🔧 Initializing system components...")
        start_time = datetime.now()
        
        await controller.initialize()
        
        init_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ System initialization complete in {init_duration:.2f} seconds")
        
        # Display startup summary
        status = controller.get_status()
        logger.info(f"🤖 Active agents: {len(status.active_agents)}")
        logger.info(f"📊 Performance targets: CAGR {SYSTEM_CONFIG.performance.target_cagr:.1%}, Sharpe {SYSTEM_CONFIG.performance.target_sharpe:.2f}")
        logger.info(f"⚡ Ready to generate {SYSTEM_CONFIG.requirements.strategy_generation_rate} strategies/hour")
        
        logger.info("🚀 Starting main execution loop...")
        
        # Run main system loop
        await controller.run()
        
    except KeyboardInterrupt:
        logger.info("⏹️  Received shutdown signal (Ctrl+C)")
        print("\n⏹️  Shutting down gracefully...")
    except Exception as e:
        if logger:
            logger.error(f"💥 Fatal system error: {str(e)}", exc_info=True)
        else:
            print(f"💥 Fatal error during startup: {str(e)}")
        sys.exit(1)
    finally:
        # Graceful shutdown
        if controller:
            try:
                print("🔄 Shutting down system components...")
                await controller.shutdown()
                print("✅ System shutdown complete")
            except Exception as e:
                print(f"❌ Error during shutdown: {str(e)}")
        
        if logger:
            logger.info("System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())