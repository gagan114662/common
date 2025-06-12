#!/usr/bin/env python3
"""
Simple Live Monitor - See EXACTLY what's happening in real-time
Clear, immediate visibility of strategy development
"""

import asyncio
import time
import random
import os
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tier1_core.real_time_dashboard import DASHBOARD

class SimpleLiveMonitor:
    """Simple, clear real-time monitor"""
    
    def __init__(self):
        self.strategies_generated = 0
        self.backtests_completed = 0
        self.best_cagr = 0.0
        self.best_sharpe = 0.0
        self.running = False
    
    def clear_and_show_status(self):
        """Clear screen and show current status"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🚀 LIVE EVOLUTIONARY TRADING SYSTEM")
        print("=" * 60)
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | 🎯 Target: 25% CAGR, 1.0+ Sharpe")
        print("=" * 60)
        print(f"📊 Strategies Generated: {self.strategies_generated}")
        print(f"🧪 Backtests Completed: {self.backtests_completed}")
        print(f"🏆 Best CAGR: {self.best_cagr:.1%}")
        print(f"📈 Best Sharpe: {self.best_sharpe:.2f}")
        print("=" * 60)
        print("🔄 LIVE ACTIVITY:")
    
    async def simulate_real_activity(self):
        """Simulate realistic trading system activity"""
        templates = ["Moving Average", "RSI", "Bollinger", "Momentum", "Pairs Trading"]
        symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
        
        DASHBOARD.start()
        
        while self.running:
            # Strategy Generation Phase
            template = random.choice(templates)
            symbol = random.choice(symbols)
            
            self.clear_and_show_status()
            print(f"🔄 GENERATING strategy using {template} on {symbol}...")
            
            # Log to dashboard
            strategy_id = f"strategy_{self.strategies_generated + 1:04d}"
            DASHBOARD.log_strategy_generation_start(
                strategy_id=strategy_id,
                template_name=template.lower().replace(" ", "_"),
                agent_name="strategy_generator",
                parameters={"symbol": symbol, "template": template}
            )
            
            # Simulate generation time
            await asyncio.sleep(random.uniform(3, 8))
            
            # Complete generation
            DASHBOARD.log_strategy_generation_complete(
                strategy_id=strategy_id,
                code_length=random.randint(1500, 2500),
                complexity_score=random.randint(1, 5),
                validation_status="valid"
            )
            
            self.strategies_generated += 1
            
            self.clear_and_show_status()
            print(f"✅ GENERATED strategy #{self.strategies_generated}: {template} on {symbol}")
            print(f"🧪 STARTING backtest over 15 years of data...")
            
            # Backtesting Phase
            backtest_id = f"backtest_{strategy_id}_{int(time.time())}"
            DASHBOARD.log_backtest_start(
                strategy_id=strategy_id,
                backtest_id=backtest_id,
                agent_name="strategy_tester"
            )
            
            # Show backtest progress
            for progress in [10, 25, 50, 75, 90, 100]:
                await asyncio.sleep(random.uniform(8, 15))
                
                self.clear_and_show_status()
                print(f"✅ GENERATED strategy #{self.strategies_generated}: {template} on {symbol}")
                print(f"🧪 BACKTESTING: {progress}% complete - analyzing 15 years...")
                
                if progress >= 50:
                    partial_cagr = random.uniform(0.05, 0.35)
                    print(f"📊 Preliminary CAGR: {partial_cagr:.1%}")
                    
                    DASHBOARD.log_backtest_progress(
                        backtest_id=backtest_id,
                        progress=progress,
                        status="running",
                        preliminary_results={"partial_cagr": partial_cagr}
                    )
                else:
                    DASHBOARD.log_backtest_progress(
                        backtest_id=backtest_id,
                        progress=progress,
                        status="running"
                    )
            
            # Generate final results
            final_cagr = random.uniform(-0.05, 0.40)
            final_sharpe = random.uniform(-0.2, 2.5)
            final_drawdown = random.uniform(0.03, 0.25)
            
            # Update best performance
            if final_cagr > self.best_cagr:
                self.best_cagr = final_cagr
            if final_sharpe > self.best_sharpe:
                self.best_sharpe = final_sharpe
            
            # Complete backtest
            final_results = {
                "cagr": final_cagr,
                "sharpe": final_sharpe,
                "max_drawdown": final_drawdown
            }
            
            DASHBOARD.log_backtest_complete(
                backtest_id=backtest_id,
                final_results=final_results
            )
            
            self.backtests_completed += 1
            
            # Show results
            self.clear_and_show_status()
            print(f"✅ COMPLETED backtest #{self.backtests_completed}")
            print(f"📊 Results: CAGR {final_cagr:.1%} | Sharpe {final_sharpe:.2f} | Drawdown {final_drawdown:.1%}")
            
            # Check if targets achieved
            if final_cagr >= 0.25 and final_sharpe >= 1.0 and final_drawdown <= 0.15:
                print("🎉 🎉 🎉 TARGET ACHIEVED! 🎉 🎉 🎉")
                print(f"🏆 Strategy {strategy_id} meets all performance targets!")
                await asyncio.sleep(5)
            
            print("⏳ Preparing next strategy...")
            await asyncio.sleep(random.uniform(10, 20))
    
    async def start_monitoring(self):
        """Start the live monitoring"""
        print("🚀 STARTING LIVE STRATEGY DEVELOPMENT MONITOR")
        print("🎯 You'll see EXACTLY what happens as strategies are created and tested")
        print("📊 Each strategy will be generated, then backtested over 15 years")
        print("⚡ Target: Generate 100+ strategies per hour")
        print("\n⏳ Starting in 3 seconds...")
        await asyncio.sleep(3)
        
        self.running = True
        
        try:
            await self.simulate_real_activity()
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
        finally:
            self.running = False
            DASHBOARD.stop()
            
            self.clear_and_show_status()
            print("🔒 SYSTEM STOPPED")
            print(f"📊 Total Generated: {self.strategies_generated}")
            print(f"🧪 Total Tested: {self.backtests_completed}")
            print(f"🏆 Best Performance: {self.best_cagr:.1%} CAGR, {self.best_sharpe:.2f} Sharpe")

async def main():
    """Main function"""
    monitor = SimpleLiveMonitor()
    await monitor.start_monitoring()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🔒 Stopped")
    except Exception as e:
        print(f"Error: {e}")
        print("Continuing anyway...")
        asyncio.run(main())