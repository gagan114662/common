#!/usr/bin/env python3
"""
Start Real 3-Tier Evolutionary Trading System with QuantConnect Integration
This starts the complete system with REAL API calls and data
"""

import asyncio
import time
import threading
import sys
import signal
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tier1_core.controller import SystemController
from tier1_core.real_time_dashboard import DASHBOARD
from tier1_core.logger import setup_logging, get_logger
from config.settings import SYSTEM_CONFIG
from dashboard_viewer import start_dashboard_server

class RealTradingSystemWithMonitoring:
    """Real trading system with QuantConnect integration and monitoring"""
    
    def __init__(self):
        self.controller = SystemController()
        self.dashboard_server_thread = None
        self.running = False
        self.logger = get_logger(__name__)
        
    async def start_real_system(self):
        """Start the complete real trading system"""
        print("🚀 STARTING REAL 3-TIER EVOLUTIONARY TRADING SYSTEM")
        print("=" * 70)
        print("⚠️  WARNING: This will make REAL API calls to QuantConnect")
        print("📊 User ID: 357130")
        print("🎯 Targets: 25% CAGR | 1.0+ Sharpe | <15% Drawdown")
        print("⚡ Generation Rate: 100+ strategies/hour")
        print("=" * 70)
        
        # Confirm with user
        confirm = input("\n🔥 Start REAL system with QuantConnect API calls? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("❌ System startup cancelled")
            return
        
        try:
            # Start web dashboard in background
            print("\n🌐 Starting real-time web dashboard...")
            self.dashboard_server_thread = threading.Thread(
                target=start_dashboard_server, 
                args=(8080,), 
                daemon=True
            )
            self.dashboard_server_thread.start()
            time.sleep(2)
            
            print("✅ Web dashboard started at http://localhost:8080")
            
            # Initialize system with REAL QuantConnect credentials
            print("\n⚙️ Initializing system with QuantConnect credentials...")
            print(f"   User ID: {SYSTEM_CONFIG.quantconnect.user_id}")
            print(f"   API URL: {SYSTEM_CONFIG.quantconnect.api_url}")
            
            await self.controller.initialize()
            
            print("\n✅ System initialization complete!")
            print("\n🎯 PERFORMANCE TARGETS:")
            print(f"   📈 CAGR: {SYSTEM_CONFIG.performance.target_cagr:.1%}")
            print(f"   📊 Sharpe: {SYSTEM_CONFIG.performance.target_sharpe:.1f}+")
            print(f"   🛡️ Max Drawdown: {SYSTEM_CONFIG.performance.max_drawdown:.1%}")
            print(f"   ⚡ Generation Rate: {SYSTEM_CONFIG.requirements.strategy_generation_rate}+ strategies/hour")
            
            print(f"\n🌐 REAL-TIME MONITORING:")
            print(f"   Web Dashboard: http://localhost:8080")
            print(f"   Live API Data: http://localhost:8080/api/dashboard")
            
            print(f"\n🔄 SYSTEM STATUS: RUNNING")
            print("💡 Press Ctrl+C for graceful shutdown")
            print("📊 Monitor the web dashboard to see REAL activity!")
            
            # Start the REAL system
            self.running = True
            await self.controller.run()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received...")
            await self.shutdown()
        except Exception as e:
            self.logger.error(f"System error: {e}")
            print(f"❌ System error: {e}")
            await self.shutdown()
            
    async def shutdown(self):
        """Graceful system shutdown"""
        if self.running:
            print("\n🔒 Shutting down real system...")
            self.running = False
            await self.controller.shutdown()
            print("✅ Real system shutdown complete")

def print_real_system_banner():
    """Print banner for real system"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 REAL 3-TIER EVOLUTIONARY TRADING SYSTEM               ║
║                           QuantConnect Integration Active                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  👤 USER: {SYSTEM_CONFIG.quantconnect.user_id:<20} 🔐 AUTHENTICATED                     ║
║  🎯 TARGETS: 25% CAGR | 1.0+ Sharpe | <15% Drawdown | 100+ Strategies/Hour  ║
║  🏗️ ARCHITECTURE: 3-Tier (Core + Strategy + Evolution)                     ║
║  🤖 AGENTS: Multi-AI Collaboration System                                   ║
║  🌐 MONITORING: Real-time Web Dashboard + Live API Data                     ║
║  📊 PLATFORM: QuantConnect Cloud Platform (400TB+ Data)                     ║
║  ⚡ PERFORMANCE: Real backtesting over 15 years (2009-2024)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

🔥 REAL SYSTEM FEATURES:
   ✅ Real QuantConnect API authentication & calls
   ✅ Real strategy generation with 6 templates  
   ✅ Real backtesting over 15-year historical data
   ✅ Real performance metrics (CAGR, Sharpe, Drawdown)
   ✅ Real agent coordination and evolution
   ✅ Real-time dashboard with live data
   ✅ Real project creation and management

⚠️  WARNING: This makes REAL API calls to QuantConnect
📊 Monitor progress at: http://localhost:8080
    """
    print(banner)

async def quick_system_test():
    """Quick test of QuantConnect connectivity"""
    print("🧪 Testing QuantConnect connectivity...")
    
    try:
        from tier1_core.quantconnect_client import QuantConnectClient
        
        # Test authentication
        client = QuantConnectClient(
            user_id=SYSTEM_CONFIG.quantconnect.user_id,
            token=SYSTEM_CONFIG.quantconnect.token,
            api_url=SYSTEM_CONFIG.quantconnect.api_url
        )
        
        async with client:
            auth_success = await client.authenticate()
            if auth_success:
                print("✅ QuantConnect authentication successful")
                
                # Get projects
                projects = await client.get_projects()
                print(f"✅ Found {len(projects)} existing projects")
                
                return True
            else:
                print("❌ QuantConnect authentication failed")
                return False
                
    except Exception as e:
        print(f"❌ QuantConnect test failed: {e}")
        return False

async def main():
    """Main function"""
    print_real_system_banner()
    
    print("🚀 Real System Startup Options:")
    print("1. 🔥 Start REAL Trading System (with QuantConnect API)")
    print("2. 🧪 Test QuantConnect Connectivity Only")
    print("3. 📊 Dashboard Only (no trading system)")
    print("4. ❌ Cancel")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    if choice == "2":
        # Test connectivity only
        print("\n🧪 Testing QuantConnect connectivity...")
        success = await quick_system_test()
        return 0 if success else 1
        
    elif choice == "3":
        # Dashboard only
        print("\n📊 Starting dashboard only...")
        DASHBOARD.start()
        start_dashboard_server(8080)
        
    elif choice == "4":
        print("❌ Cancelled")
        return 0
        
    elif choice == "1":
        # Start real system
        print("\n🔥 Starting REAL trading system...")
        
        # Setup logging
        setup_logging()
        
        # Start the real system
        system = RealTradingSystemWithMonitoring()
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}")
            asyncio.create_task(system.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            await system.start_real_system()
        except Exception as e:
            print(f"❌ System startup failed: {e}")
            return 1
    else:
        print("❌ Invalid choice")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)