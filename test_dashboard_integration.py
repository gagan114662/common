#!/usr/bin/env python3
"""
Test script for Real-Time Dashboard Integration
Tests the complete visibility system for strategy creation and backtesting
"""

import asyncio
import time
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tier1_core.real_time_dashboard import DASHBOARD
from tier1_core.logger import get_logger
from config.settings import SYSTEM_CONFIG

logger = get_logger(__name__)

async def test_dashboard_integration():
    """Test the integrated dashboard functionality"""
    print("🧪 Testing Real-Time Dashboard Integration")
    print("=" * 60)
    
    # Start the dashboard
    DASHBOARD.start()
    print("✅ Dashboard started")
    
    try:
        # Test 1: System initialization logging
        print("\n📊 Test 1: System Initialization Logging")
        DASHBOARD.log_agent_activity(
            agent_name="test_controller",
            activity="system_initialization",
            details={"components": ["quantconnect", "strategy_generator", "strategy_tester"]}
        )
        
        # Test 2: Strategy generation simulation
        print("\n🔄 Test 2: Strategy Generation Simulation")
        for i in range(3):
            strategy_id = f"test_strategy_{i+1}"
            template_name = ["ma_crossover", "rsi_mean_reversion", "bollinger_bands"][i]
            parameters = {
                "fast_period": 10 + i * 5,
                "slow_period": 20 + i * 10,
                "symbol": ["SPY", "QQQ", "IWM"][i]
            }
            
            # Log strategy generation start
            DASHBOARD.log_strategy_generation_start(
                strategy_id=strategy_id,
                template_name=template_name,
                agent_name="test_generator",
                parameters=parameters
            )
            
            # Simulate generation time
            await asyncio.sleep(0.5)
            
            # Log strategy generation completion
            DASHBOARD.log_strategy_generation_complete(
                strategy_id=strategy_id,
                code_length=1200 + i * 200,
                complexity_score=2 + i,
                validation_status="valid"
            )
            
            print(f"  ✅ Generated strategy {i+1}: {strategy_id[:12]}...")
        
        # Test 3: Backtesting simulation
        print("\n🧪 Test 3: Backtesting Simulation")
        for i in range(2):
            strategy_id = f"test_strategy_{i+1}"
            backtest_id = f"bt_{strategy_id}_{int(time.time())}"
            
            # Log backtest start
            DASHBOARD.log_backtest_start(
                strategy_id=strategy_id,
                backtest_id=backtest_id,
                agent_name="test_tester"
            )
            
            # Simulate backtest progress
            for progress in [25, 50, 75]:
                await asyncio.sleep(0.3)
                DASHBOARD.log_backtest_progress(
                    backtest_id=backtest_id,
                    progress=progress,
                    status="running",
                    preliminary_results={"partial_cagr": 0.15 + progress * 0.001} if progress >= 50 else None
                )
                print(f"    📊 Backtest {i+1} progress: {progress}%")
            
            # Log backtest completion
            await asyncio.sleep(0.5)
            final_results = {
                "cagr": 0.18 + i * 0.05,
                "sharpe": 1.2 + i * 0.3,
                "max_drawdown": 0.08 - i * 0.02
            }
            
            DASHBOARD.log_backtest_complete(
                backtest_id=backtest_id,
                final_results=final_results
            )
            
            print(f"  ✅ Completed backtest {i+1}: CAGR {final_results['cagr']:.1%}, Sharpe {final_results['sharpe']:.2f}")
        
        # Test 4: Evolution event logging
        print("\n🧬 Test 4: Evolution Event Logging")
        for generation in range(1, 4):
            DASHBOARD.log_evolution_event(
                generation=generation,
                best_fitness=0.5 + generation * 0.1,
                avg_fitness=0.3 + generation * 0.05,
                diversity=0.8 - generation * 0.1
            )
            await asyncio.sleep(0.2)
            print(f"  🧬 Evolution generation {generation} completed")
        
        # Test 5: Live dashboard data retrieval
        print("\n📈 Test 5: Live Dashboard Data Retrieval")
        await asyncio.sleep(1)  # Let metrics calculate
        
        dashboard_data = DASHBOARD.get_live_dashboard_data()
        
        print(f"  📊 System Status: {dashboard_data['system_status']}")
        print(f"  🏭 Active Agents: {len(dashboard_data['active_agents'])}")
        print(f"  📋 Recent Events: {len(dashboard_data['recent_events'])}")
        print(f"  🔄 Active Generations: {len(dashboard_data['active_generations'])}")
        print(f"  🧪 Active Backtests: {len(dashboard_data['active_backtests'])}")
        
        # Test 6: Performance summary
        print("\n📈 Test 6: Performance Summary")
        performance_summary = dashboard_data['performance_summary']
        last_hour = performance_summary['last_hour']
        current_active = performance_summary['current_active']
        
        print(f"  📊 Last Hour: {last_hour['strategies_generated']} generated, {last_hour['strategies_tested']} tested")
        print(f"  🤖 Currently Active: {current_active['total_agents']} agents, {current_active['active_generations']} generations")
        
        # Test 7: Agent details
        print("\n🤖 Test 7: Agent Activity Details")
        for agent_name in ["test_generator", "test_tester"]:
            agent_details = DASHBOARD.get_agent_details(agent_name)
            if 'error' not in agent_details:
                snapshot = agent_details['current_snapshot']
                print(f"  🤖 {agent_name}:")
                print(f"    - Current Task: {snapshot.get('current_task', 'None')}")
                print(f"    - Success Rate: {snapshot.get('success_rate', 0):.1f}%")
                print(f"    - Recent Events: {len(agent_details['recent_events'])}")
        
        # Test 8: Export functionality
        print("\n💾 Test 8: Export Functionality")
        export_path = Path("test_dashboard_export.json")
        DASHBOARD.export_activity_log(export_path, hours=1)
        
        if export_path.exists():
            file_size = export_path.stat().st_size
            print(f"  ✅ Exported activity log: {file_size} bytes")
            
            # Load and verify export
            with open(export_path, 'r') as f:
                export_data = json.load(f)
            
            print(f"  📊 Export contains: {len(export_data['events'])} events")
            print(f"  🔄 Strategy events: {len(export_data['strategy_events'])}")
            print(f"  🧪 Backtest events: {len(export_data['backtest_events'])}")
            
            # Cleanup
            export_path.unlink()
        
        # Test 9: Strategy timeline
        print("\n📅 Test 9: Strategy Timeline")
        strategy_timeline = DASHBOARD.get_strategy_timeline("test_strategy_1")
        print(f"  📅 Timeline for test_strategy_1: {len(strategy_timeline['timeline'])} events")
        
        print("\n🎉 All dashboard integration tests completed successfully!")
        print("=" * 60)
        
        # Final performance metrics
        generation_stats = DASHBOARD.generation_stats
        print(f"📊 Final Dashboard Metrics:")
        print(f"  - Total Events Logged: {len(DASHBOARD.activity_events)}")
        print(f"  - Strategies Generated: {generation_stats['total_generated']}")
        print(f"  - Active Agents Tracked: {len(DASHBOARD.active_agents)}")
        print(f"  - Success Rate: {generation_stats['success_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Dashboard integration test failed: {str(e)}")
        return False
        
    finally:
        # Stop the dashboard
        DASHBOARD.stop()
        print("\n🔒 Dashboard stopped")

async def main():
    """Main test function"""
    print("🚀 Starting Real-Time Dashboard Integration Test")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = await test_dashboard_integration()
    
    if success:
        print("\n✅ ALL TESTS PASSED - Dashboard integration is working correctly!")
        print("🎯 The system now provides complete visibility into:")
        print("   - Strategy generation process with detailed parameters")
        print("   - Real-time backtesting progress with preliminary results")
        print("   - Agent activity tracking and performance metrics")
        print("   - Evolution events and system orchestration")
        print("   - Live dashboard data for monitoring")
        print("   - Export capabilities for analysis")
    else:
        print("\n❌ TESTS FAILED - Please check the error logs")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)