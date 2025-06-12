#!/usr/bin/env python3
"""
Demo Mode - 3-Tier Evolutionary Trading System
Runs the system with simulated data for demonstration purposes
"""

import asyncio
import sys
import signal
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tier1_core.real_time_dashboard import DASHBOARD
from tier2_strategy.strategy_generator import StrategyTemplateLibrary
from tier1_core.logger import setup_logging

__version__ = "1.0.0-demo"

def banner():
    """Display demo banner"""
    print("=" * 80)
    print(f"🚀 3-TIER EVOLUTIONARY TRADING SYSTEM v{__version__}")
    print("=" * 80)
    print("🧪 DEMO MODE - No QuantConnect API Required")
    print("Simulated Strategy Generation & Performance Monitoring")
    print("Real-time Dashboard and Multi-Agent Architecture Demo")
    print("=" * 80)
    print()

class DemoSystem:
    """Demo system that simulates trading operations"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.running = False
        self.strategy_count = 0
        
    async def start(self):
        """Start demo system"""
        banner()
        
        self.logger.info(f"Starting Demo System v{__version__}")
        
        # Start dashboard
        DASHBOARD.start()
        self.logger.info("✅ Real-time dashboard started")
        
        # Show available strategy templates
        template_lib = StrategyTemplateLibrary()
        templates = template_lib.get_all_templates()
        self.logger.info(f"✅ {len(templates)} strategy templates loaded:")
        for template in templates:
            self.logger.info(f"   - {template.name} ({template.category})")
        
        print("\n🎮 Demo Commands:")
        print("  's' - Simulate strategy generation")
        print("  'b' - Simulate backtest")
        print("  'e' - Simulate evolution")
        print("  'm' - Show system metrics")
        print("  'q' - Quit")
        print("\nPress Enter after each command...")
        
        self.running = True
        
        # Start demo loop
        await self._demo_loop()
    
    async def _demo_loop(self):
        """Main demo interaction loop"""
        while self.running:
            try:
                # Get user input (in real deployment this would be automated)
                print("\n> Enter command (s/b/e/m/q): ", end="", flush=True)
                
                # Simulate some automatic activity
                await asyncio.sleep(2)
                
                # Simulate strategy generation
                await self._simulate_strategy_generation()
                
                await asyncio.sleep(3)
                
                # Simulate backtest
                await self._simulate_backtest()
                
                await asyncio.sleep(2)
                
                # Show metrics
                await self._show_metrics()
                
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                break
                
        await self.shutdown()
    
    async def _simulate_strategy_generation(self):
        """Simulate strategy generation"""
        self.strategy_count += 1
        strategy_id = f"demo_strategy_{self.strategy_count:04d}"
        
        templates = ["moving_average_crossover", "rsi_mean_reversion", "bollinger_bands"]
        template = templates[self.strategy_count % len(templates)]
        
        # Log to dashboard
        DASHBOARD.log_strategy_generation_start(
            strategy_id=strategy_id,
            template_name=template,
            agent_name="demo_agent",
            parameters={"param1": 14, "param2": 0.7}
        )
        
        await asyncio.sleep(1)  # Simulate generation time
        
        DASHBOARD.log_strategy_generation_complete(
            strategy_id=strategy_id,
            code_length=1250,
            complexity_score=3,
            validation_status="valid"
        )
        
        self.logger.info(f"✅ Generated strategy: {strategy_id}")
    
    async def _simulate_backtest(self):
        """Simulate backtesting"""
        strategy_id = f"demo_strategy_{self.strategy_count:04d}"
        backtest_id = f"backtest_{self.strategy_count:04d}"
        
        # Start backtest
        DASHBOARD.log_backtest_start(
            strategy_id=strategy_id,
            backtest_id=backtest_id,
            agent_name="demo_agent"
        )
        
        # Simulate progress updates
        for progress in [25, 50, 75, 100]:
            await asyncio.sleep(0.5)
            DASHBOARD.log_backtest_progress(
                backtest_id=backtest_id,
                progress=progress,
                status="running",
                preliminary_results={
                    "cagr": 0.18 + (progress / 1000),
                    "sharpe": 1.2 + (progress / 2000),
                    "max_drawdown": 0.12 - (progress / 10000)
                }
            )
        
        # Complete backtest
        final_results = {
            "cagr": 0.22,
            "sharpe": 1.35,
            "max_drawdown": 0.08,
            "sortino": 1.8,
            "calmar": 2.75
        }
        
        DASHBOARD.log_backtest_complete(
            backtest_id=backtest_id,
            final_results=final_results
        )
        
        self.logger.info(f"✅ Completed backtest: {backtest_id}")
    
    async def _show_metrics(self):
        """Show system metrics"""
        data = DASHBOARD.get_live_dashboard_data()
        stats = data['generation_stats']
        
        print("\n📊 CURRENT SYSTEM METRICS:")
        print("-" * 40)
        print(f"Strategies Generated: {stats['strategies_per_hour']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Avg Generation Time: {stats['average_generation_time']:.2f}s")
        print(f"Avg Backtest Time: {stats['average_backtest_time']:.2f}s")
        
        # Show recent activity
        recent = data['recent_events']
        if recent:
            print(f"\n📋 Recent Activity ({len(recent)} events):")
            for event in recent[-3:]:  # Last 3 events
                status_emoji = {'completed': '✅', 'started': '🔄', 'failed': '❌'}.get(event['status'], '🔄')
                print(f"  {status_emoji} {event['source']}: {event['action']}")
    
    async def shutdown(self):
        """Shutdown demo system"""
        self.running = False
        DASHBOARD.stop()
        self.logger.info("🛑 Demo system stopped")

async def main():
    """Main demo entry point"""
    demo = DemoSystem()
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        demo.running = False
        print("\n🛑 Shutting down demo...")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        await demo.start()
    except KeyboardInterrupt:
        await demo.shutdown()

if __name__ == "__main__":
    asyncio.run(main())