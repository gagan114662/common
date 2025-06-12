#!/usr/bin/env python3
"""
Real-Time System Monitor
Provides live monitoring of the 3-Tier Evolutionary Trading System
"""

import asyncio
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tier1_core.real_time_dashboard import DASHBOARD

def clear_screen():
    """Clear terminal screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def display_live_dashboard():
    """Display live system dashboard"""
    try:
        # Get live data
        data = DASHBOARD.get_live_dashboard_data()
        
        clear_screen()
        
        # Header
        print("=" * 100)
        print("🚀 3-TIER EVOLUTIONARY TRADING SYSTEM - LIVE MONITOR")
        print("=" * 100)
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System Stats
        stats = data['generation_stats']
        print("📊 SYSTEM PERFORMANCE")
        print("-" * 50)
        print(f"Strategies Generated (last hour): {stats['strategies_per_hour']}")
        print(f"Total Tested: {stats['total_tested']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Avg Generation Time: {stats['average_generation_time']:.2f}s")
        print(f"Avg Backtest Time: {stats['average_backtest_time']:.2f}s")
        print()
        
        # Active Agents
        print("🤖 ACTIVE AGENTS")
        print("-" * 50)
        agents = data['active_agents']
        if agents:
            for agent_name, agent_data in agents.items():
                current_task = agent_data['current_task'] or 'idle'
                task_duration = ""
                if agent_data['task_start_time']:
                    start_time = datetime.fromisoformat(agent_data['task_start_time'].replace('Z', '+00:00'))
                    duration = (datetime.now() - start_time.replace(tzinfo=None)).total_seconds()
                    task_duration = f" ({format_duration(duration)})"
                
                print(f"{agent_name:20} | {current_task:15}{task_duration}")
                print(f"{'':20} | Queue: {agent_data['strategies_in_queue']}, "
                      f"Active: {agent_data['active_backtests']}, "
                      f"Success: {agent_data['success_rate']:.1f}%")
        else:
            print("No active agents")
        print()
        
        # Recent Activity
        print("📋 RECENT ACTIVITY")
        print("-" * 50)
        recent_events = data['recent_events']
        if recent_events:
            for event in recent_events[-10:]:  # Last 10 events
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                time_ago = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds()
                
                status_emoji = {
                    'started': '🔄',
                    'progress': '⏳', 
                    'completed': '✅',
                    'failed': '❌'
                }.get(event['status'], '🔄')
                
                print(f"{status_emoji} [{event['source']:15}] {event['action']:20} ({format_duration(time_ago)} ago)")
        else:
            print("No recent activity")
        print()
        
        # Active Operations
        active_gens = data['active_generations']
        active_tests = data['active_backtests']
        
        if active_gens or active_tests:
            print("⚡ ACTIVE OPERATIONS")
            print("-" * 50)
            
            if active_gens:
                print(f"🔧 Strategy Generations: {len(active_gens)}")
                for gen in active_gens[:5]:  # Show first 5
                    duration = (datetime.now() - datetime.fromisoformat(gen['creation_start'])).total_seconds()
                    print(f"   {gen['strategy_id'][:8]} | {gen['template_name']:15} | {format_duration(duration)}")
            
            if active_tests:
                print(f"🧪 Active Backtests: {len(active_tests)}")
                for test in active_tests[:5]:  # Show first 5
                    duration = (datetime.now() - datetime.fromisoformat(test['start_time'])).total_seconds()
                    progress = test.get('progress', 0)
                    print(f"   {test['backtest_id'][:8]} | {test['status']:10} | {progress:.1f}% | {format_duration(duration)}")
            print()
        
        # Performance Summary
        perf_summary = data['performance_summary']
        print("🎯 PERFORMANCE SUMMARY")
        print("-" * 50)
        print(f"Last Hour - Generated: {perf_summary['last_hour']['strategies_generated']}, "
              f"Tested: {perf_summary['last_hour']['strategies_tested']}")
        print(f"Currently Active - Agents: {perf_summary['current_active']['total_agents']}, "
              f"Generations: {perf_summary['current_active']['active_generations']}, "
              f"Backtests: {perf_summary['current_active']['active_backtests']}")
        
        print("\nPress Ctrl+C to exit")
        
    except Exception as e:
        print(f"Error displaying dashboard: {str(e)}")
        import traceback
        traceback.print_exc()

async def main():
    """Main monitoring loop"""
    print("🚀 Starting 3-Tier Evolution System Monitor...")
    print("Connecting to live dashboard...")
    
    # Start dashboard if not running
    try:
        DASHBOARD.start()
        print("✅ Connected to system dashboard")
        print("Press Ctrl+C to exit\n")
        
        while True:
            display_live_dashboard()
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped")
    except Exception as e:
        print(f"Monitor error: {str(e)}")
    finally:
        # Stop dashboard
        DASHBOARD.stop()

if __name__ == "__main__":
    asyncio.run(main())