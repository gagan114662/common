#!/usr/bin/env python3
"""
Demo 3-Tier Evolutionary Trading System
Shows REAL system behavior with simulated QuantConnect responses
This demonstrates exactly what you'll see when you get real credentials
"""

import asyncio
import time
import threading
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tier1_core.real_time_dashboard import DASHBOARD
from tier1_core.logger import setup_logging, get_logger
from dashboard_viewer import start_dashboard_server

class DemoTradingSystem:
    """Demo trading system that shows real behavior patterns"""
    
    def __init__(self):
        self.running = False
        self.logger = get_logger(__name__)
        self.strategies_generated = 0
        self.backtests_completed = 0
        
    async def start_demo_system(self):
        """Start demo system with realistic behavior"""
        print("🎭 DEMO: 3-Tier Evolutionary Trading System")
        print("=" * 60)
        print("🎯 This shows EXACTLY what you'll see with real QuantConnect data")
        print("📊 Real system behavior with simulated API responses")
        print("🌐 Web dashboard shows live activity at http://localhost:8080")
        print("=" * 60)
        
        # Start dashboard
        DASHBOARD.start()
        print("✅ Real-time dashboard started")
        
        # Start web server
        dashboard_thread = threading.Thread(
            target=start_dashboard_server,
            args=(8080,),
            daemon=True
        )
        dashboard_thread.start()
        time.sleep(2)
        
        print("🌐 Web dashboard available at: http://localhost:8080")
        print("🔄 System will generate and test strategies automatically")
        print("💡 Press Ctrl+C to stop")
        print()
        
        # Start demo activities
        self.running = True
        
        try:
            # Run demo activities
            await asyncio.gather(
                self.demo_system_initialization(),
                self.demo_strategy_generation(),
                self.demo_backtesting(),
                self.demo_agent_coordination(),
                self.demo_evolution_progress()
            )
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        finally:
            self.running = False
            DASHBOARD.stop()
            print("✅ Demo system shutdown complete")
    
    async def demo_system_initialization(self):
        """Demo system initialization"""
        await asyncio.sleep(1)
        
        DASHBOARD.log_agent_activity(
            agent_name="system_controller",
            activity="initialization_start",
            details={"components": ["quantconnect", "strategy_system", "evolution", "agents"]}
        )
        
        await asyncio.sleep(2)
        
        DASHBOARD.log_agent_activity(
            agent_name="system_controller", 
            activity="quantconnect_authenticated",
            details={"user_id": "YOUR_ID", "status": "connected"}
        )
        
        DASHBOARD.log_agent_activity(
            agent_name="system_controller",
            activity="initialization_complete",
            details={"duration_seconds": 3.2, "status": "ready"}
        )
        
        print("✅ Demo: System initialized")
    
    async def demo_strategy_generation(self):
        """Demo strategy generation with realistic patterns"""
        await asyncio.sleep(3)
        
        templates = ["ma_crossover", "rsi_mean_reversion", "bollinger_bands", "momentum", "pairs_trading", "long_short_equity"]
        symbols = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLF", "XLE", "XLI"]
        agents = ["trend_following_agent", "mean_reversion_agent", "momentum_agent"]
        
        while self.running:
            # Generate strategy
            template = random.choice(templates)
            symbol = random.choice(symbols)
            agent = random.choice(agents)
            
            strategy_id = f"strat_{self.strategies_generated + 1:04d}"
            
            parameters = {
                "symbol": symbol,
                "fast_period": random.randint(5, 20),
                "slow_period": random.randint(25, 50),
                "threshold": round(random.uniform(0.02, 0.10), 3)
            }
            
            # Log generation start
            DASHBOARD.log_strategy_generation_start(
                strategy_id=strategy_id,
                template_name=template,
                agent_name=agent,
                parameters=parameters
            )
            
            # Simulate generation time
            generation_time = random.uniform(0.5, 2.0)
            await asyncio.sleep(generation_time)
            
            # Log completion
            code_length = random.randint(1000, 2500)
            complexity = random.randint(1, 5)
            validation = "valid" if random.random() > 0.1 else "invalid"
            
            DASHBOARD.log_strategy_generation_complete(
                strategy_id=strategy_id,
                code_length=code_length,
                complexity_score=complexity,
                validation_status=validation
            )
            
            if validation == "valid":
                self.strategies_generated += 1
                print(f"🔄 Generated strategy {self.strategies_generated}: {template} on {symbol}")
            
            # Wait before next generation (targeting 100+/hour = ~36 seconds)
            await asyncio.sleep(random.uniform(20, 45))
    
    async def demo_backtesting(self):
        """Demo backtesting with realistic progress"""
        await asyncio.sleep(5)
        
        while self.running:
            if self.strategies_generated > self.backtests_completed:
                strategy_id = f"strat_{self.backtests_completed + 1:04d}"
                backtest_id = f"bt_{strategy_id}_{int(time.time())}"
                
                # Log backtest start
                DASHBOARD.log_backtest_start(
                    strategy_id=strategy_id,
                    backtest_id=backtest_id,
                    agent_name="strategy_tester"
                )
                
                # Simulate backtest progress
                for progress in [10, 25, 50, 75, 90]:
                    await asyncio.sleep(random.uniform(15, 30))  # Realistic backtest timing
                    
                    preliminary_results = None
                    if progress >= 50:
                        preliminary_results = {
                            "partial_cagr": random.uniform(0.05, 0.35),
                            "partial_sharpe": random.uniform(0.3, 1.8)
                        }
                    
                    DASHBOARD.log_backtest_progress(
                        backtest_id=backtest_id,
                        progress=progress,
                        status="running",
                        preliminary_results=preliminary_results
                    )
                
                # Complete backtest
                await asyncio.sleep(random.uniform(10, 20))
                
                # Generate realistic results
                cagr = random.uniform(-0.1, 0.4)  # -10% to 40%
                sharpe = random.uniform(-0.5, 2.5)
                drawdown = random.uniform(0.02, 0.25)
                
                final_results = {
                    "cagr": cagr,
                    "sharpe": sharpe,
                    "max_drawdown": drawdown,
                    "total_trades": random.randint(50, 500),
                    "win_rate": random.uniform(0.4, 0.7)
                }
                
                error_msg = None
                if random.random() < 0.05:  # 5% failure rate
                    error_msg = "Runtime error: Invalid symbol data"
                
                DASHBOARD.log_backtest_complete(
                    backtest_id=backtest_id,
                    final_results=final_results if not error_msg else {},
                    error_message=error_msg
                )
                
                if not error_msg:
                    self.backtests_completed += 1
                    print(f"🧪 Completed backtest {self.backtests_completed}: CAGR {cagr:.1%}, Sharpe {sharpe:.2f}")
                    
                    # Check if targets achieved
                    if cagr >= 0.25 and sharpe >= 1.0 and drawdown <= 0.15:
                        print(f"🎉 TARGET ACHIEVED! Strategy {strategy_id} meets all criteria!")
            
            await asyncio.sleep(5)
    
    async def demo_agent_coordination(self):
        """Demo agent activities"""
        await asyncio.sleep(7)
        
        agents = [
            "supervisor_agent", "trend_following_agent", "mean_reversion_agent",
            "momentum_agent", "arbitrage_agent", "market_neutral_agent"
        ]
        
        activities = [
            "analyzing_market_conditions", "optimizing_parameters", "sharing_insights",
            "coordinating_strategy_selection", "monitoring_performance", "adapting_approach"
        ]
        
        while self.running:
            agent = random.choice(agents)
            activity = random.choice(activities)
            
            details = {
                "timestamp": datetime.now().isoformat(),
                "strategies_managed": random.randint(1, 10),
                "success_rate": random.uniform(0.7, 0.95)
            }
            
            DASHBOARD.log_agent_activity(
                agent_name=agent,
                activity=activity,
                details=details
            )
            
            await asyncio.sleep(random.uniform(30, 90))
    
    async def demo_evolution_progress(self):
        """Demo evolution engine progress"""
        await asyncio.sleep(10)
        
        generation = 1
        
        while self.running:
            if self.backtests_completed >= generation * 5:  # Evolution every 5 backtests
                best_fitness = 0.3 + generation * 0.05 + random.uniform(-0.1, 0.1)
                avg_fitness = best_fitness * 0.7 + random.uniform(-0.05, 0.05)
                diversity = 0.9 - generation * 0.02 + random.uniform(-0.1, 0.1)
                
                DASHBOARD.log_evolution_event(
                    generation=generation,
                    best_fitness=max(0, best_fitness),
                    avg_fitness=max(0, avg_fitness),
                    diversity=max(0.1, min(1.0, diversity))
                )
                
                print(f"🧬 Evolution generation {generation}: Best fitness {best_fitness:.3f}")
                generation += 1
            
            await asyncio.sleep(60)

async def main():
    """Main demo function"""
    print("🎭 3-Tier Evolution System - DEMO MODE")
    print("=" * 50)
    print("🎯 This demonstrates the REAL system behavior")
    print("📊 Shows exactly what you'll see with QuantConnect data")
    print("🔧 Get real credentials from: https://www.quantconnect.com/account")
    print("=" * 50)
    
    response = input("\n🚀 Start demo system? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        demo_system = DemoTradingSystem()
        await demo_system.start_demo_system()
    else:
        print("❌ Demo cancelled")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🔒 Demo stopped")
    except EOFError:
        print("\n🎭 Starting demo automatically...")
        demo_system = DemoTradingSystem()
        asyncio.run(demo_system.start_demo_system())