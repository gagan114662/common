#!/usr/bin/env python3
"""
Start the 3-Tier Evolutionary Trading System with Real-Time Monitoring
This script starts the complete system and provides live monitoring options
"""

import asyncio
import time
import threading
from datetime import datetime
from pathlib import Path
import sys
import signal

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tier1_core.controller import SystemController
from tier1_core.real_time_dashboard import DASHBOARD
from tier1_core.logger import get_logger
from dashboard_viewer import start_dashboard_server, console_monitor

logger = get_logger(__name__)

class SystemWithMonitoring:
    """Combined system controller with monitoring"""
    
    def __init__(self):
        self.controller = SystemController()
        self.dashboard_server_thread = None
        self.console_monitor_thread = None
        self.running = False
        
    async def start_system(self, enable_web_dashboard=True, enable_console=False):
        """Start the complete system with monitoring"""
        print("🚀 Starting 3-Tier Evolutionary Trading System")
        print("=" * 60)
        
        try:
            # Initialize system controller
            print("⚙️ Initializing system components...")
            await self.controller.initialize()
            
            # Start monitoring options
            if enable_web_dashboard:
                print("🌐 Starting web dashboard...")
                self.dashboard_server_thread = threading.Thread(
                    target=start_dashboard_server, 
                    args=(8080,), 
                    daemon=True
                )
                self.dashboard_server_thread.start()
                time.sleep(2)  # Give server time to start
                
            if enable_console:
                print("📺 Starting console monitor...")
                self.console_monitor_thread = threading.Thread(
                    target=console_monitor, 
                    daemon=True
                )
                self.console_monitor_thread.start()
            
            print("✅ System startup complete!")
            print("\n🎯 Performance Targets:")
            print(f"   - CAGR: 25%")
            print(f"   - Sharpe Ratio: 1.0+")
            print(f"   - Max Drawdown: <15%")
            print(f"   - Generation Rate: 100+ strategies/hour")
            
            if enable_web_dashboard:
                print(f"\n🌐 Monitor at: http://localhost:8080")
            
            print("\n🔄 System is now running...")
            print("💡 Press Ctrl+C to shutdown gracefully")
            
            # Run the main system
            self.running = True
            await self.controller.run()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received...")
            await self.shutdown()
        except Exception as e:
            logger.error(f"System error: {e}")
            await self.shutdown()
            
    async def shutdown(self):
        """Graceful system shutdown"""
        if self.running:
            print("🔒 Shutting down system...")
            self.running = False
            await self.controller.shutdown()
            print("✅ System shutdown complete")

def print_banner():
    """Print system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    3-TIER EVOLUTIONARY TRADING SYSTEM                       ║
║                          Real-Time Monitoring Enabled                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎯 TARGETS: 25% CAGR | 1.0+ Sharpe | <15% Drawdown | 100+ Strategies/Hour  ║
║  🔧 ARCHITECTURE: 3-Tier (Core + Strategy + Evolution)                      ║
║  🤖 AGENTS: Multi-AI Collaboration System                                   ║
║  🌐 MONITORING: Web Dashboard + Console + Real-time Logs                    ║
║  📊 PLATFORM: QuantConnect Cloud (User: 357130)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def main():
    """Main function"""
    print_banner()
    
    print("🚀 System Startup Options:")
    print("1. 🌐 Full System + Web Dashboard (recommended)")
    print("2. 📺 Full System + Console Monitor")
    print("3. 🎛️ Full System + Both Monitors")
    print("4. 🧪 Dashboard Test Only (no trading system)")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    if choice == "4":
        # Run dashboard test
        from test_dashboard_integration import test_dashboard_integration
        print("\n🧪 Running Dashboard Integration Test...")
        success = await test_dashboard_integration()
        return 0 if success else 1
    
    # Configure monitoring
    enable_web = choice in ["1", "3"]
    enable_console = choice in ["2", "3"]
    
    if choice not in ["1", "2", "3"]:
        print("Invalid choice, using default (web dashboard)...")
        enable_web = True
        enable_console = False
    
    # Start the system
    system = SystemWithMonitoring()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        asyncio.create_task(system.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await system.start_system(enable_web, enable_console)
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)