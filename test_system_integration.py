#!/usr/bin/env python3
"""
System Integration Test Suite
Validates all components of the 3-Tier Evolutionary Trading System
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    tests = [
        ("config.settings", "SYSTEM_CONFIG"),
        ("tier1_core.controller", "SystemController"),
        ("tier1_core.quantconnect_client", "QuantConnectClient"),
        ("tier1_core.performance_monitor", "PerformanceMonitor"),
        ("tier1_core.real_time_dashboard", "DASHBOARD"),
        ("tier2_strategy.strategy_generator", "StrategyGenerator"),
        ("tier2_strategy.strategy_tester", "StrategyTester"),
        ("tier3_evolution.evolution_engine", "EvolutionEngine"),
        ("agents.supervisor_agent", "SupervisorAgent"),
        ("agents.knowledge_base", "SharedKnowledgeBase"),
        ("agents.trend_following_agent", "TrendFollowingAgent"),
        ("agents.mean_reversion_agent", "MeanReversionAgent"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✅ {module_name}.{class_name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {module_name}.{class_name}: {str(e)}")
            failed += 1
    
    print(f"\nImport Results: {passed} passed, {failed} failed")
    return failed == 0

def test_configuration():
    """Test system configuration"""
    print("\n🔧 Testing configuration...")
    
    try:
        from config.settings import SYSTEM_CONFIG
        
        # Check essential configuration
        assert SYSTEM_CONFIG.quantconnect.user_id == "357130", "User ID not set correctly"
        assert SYSTEM_CONFIG.quantconnect.token, "API token not configured"
        assert SYSTEM_CONFIG.performance.target_cagr > 0, "Target CAGR not set"
        assert SYSTEM_CONFIG.requirements.strategy_generation_rate > 0, "Generation rate not set"
        
        print(f"  ✅ User ID: {SYSTEM_CONFIG.quantconnect.user_id}")
        print(f"  ✅ Target CAGR: {SYSTEM_CONFIG.performance.target_cagr:.1%}")
        print(f"  ✅ Generation Rate: {SYSTEM_CONFIG.requirements.strategy_generation_rate} strategies/hour")
        print(f"  ✅ Backtest Duration: {SYSTEM_CONFIG.requirements.backtest_duration_minutes} minutes")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {str(e)}")
        return False

async def test_system_controller():
    """Test system controller initialization"""
    print("\n🎮 Testing SystemController...")
    
    try:
        from tier1_core.controller import SystemController
        
        # Create controller
        controller = SystemController()
        print("  ✅ Controller created")
        
        # Test configuration access
        assert controller.config, "Configuration not loaded"
        print("  ✅ Configuration loaded")
        
        # Test status initialization
        status = controller.get_status()
        assert not status.is_running, "Should not be running initially"
        print("  ✅ Status accessible")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Controller test failed: {str(e)}")
        return False

async def test_dashboard():
    """Test dashboard functionality"""
    print("\n📊 Testing Dashboard...")
    
    try:
        from tier1_core.real_time_dashboard import DASHBOARD
        
        # Test basic dashboard methods
        data = DASHBOARD.get_live_dashboard_data()
        assert isinstance(data, dict), "Dashboard data should be a dictionary"
        print("  ✅ Dashboard data accessible")
        
        # Test event logging
        DASHBOARD.log_agent_activity("test_agent", "test_activity", {"test": "data"})
        print("  ✅ Event logging works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dashboard test failed: {str(e)}")
        return False

async def test_strategy_components():
    """Test strategy generation and testing components"""
    print("\n🧠 Testing Strategy Components...")
    
    try:
        from tier2_strategy.strategy_generator import StrategyGenerator, StrategyTemplateLibrary
        from tier2_strategy.strategy_tester import StrategyTester
        from tier1_core.quantconnect_client import QuantConnectClient
        from config.settings import SYSTEM_CONFIG
        
        # Test strategy template library first
        template_lib = StrategyTemplateLibrary()
        templates = template_lib.get_all_templates()
        assert len(templates) > 0, "No strategy templates available"
        print(f"  ✅ {len(templates)} strategy templates available")
        
        # Test client creation for strategy components
        qc_client = QuantConnectClient(
            user_id=SYSTEM_CONFIG.quantconnect.user_id,
            token=SYSTEM_CONFIG.quantconnect.token,
            api_url=SYSTEM_CONFIG.quantconnect.api_url
        )
        print("  ✅ QuantConnect client created for strategy components")
        
        # Test strategy generator (needs client and config)
        generator = StrategyGenerator(qc_client, SYSTEM_CONFIG)
        print("  ✅ StrategyGenerator created")
        
        # Test strategy tester (needs client and config)
        tester = StrategyTester(qc_client, SYSTEM_CONFIG)
        print("  ✅ StrategyTester created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy components test failed: {str(e)}")
        return False

async def test_agent_system():
    """Test multi-agent system components"""
    print("\n🤖 Testing Agent System...")
    
    try:
        from agents.knowledge_base import SharedKnowledgeBase
        from agents.supervisor_agent import SupervisorAgent
        from agents.trend_following_agent import TrendFollowingAgent
        from agents.mean_reversion_agent import MeanReversionAgent
        
        # Test knowledge base
        knowledge_base = SharedKnowledgeBase()
        print("  ✅ SharedKnowledgeBase created")
        
        # Test agent creation (without full initialization)
        print("  ✅ Agent classes importable")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent system test failed: {str(e)}")
        return False

async def test_evolution_engine():
    """Test evolution engine"""
    print("\n🧬 Testing Evolution Engine...")
    
    try:
        from tier3_evolution.evolution_engine import EvolutionEngine
        from tier2_strategy.strategy_generator import StrategyGenerator
        from tier2_strategy.strategy_tester import StrategyTester
        from tier1_core.quantconnect_client import QuantConnectClient
        from config.settings import SYSTEM_CONFIG
        
        # Create dependencies first
        qc_client = QuantConnectClient(
            user_id=SYSTEM_CONFIG.quantconnect.user_id,
            token=SYSTEM_CONFIG.quantconnect.token,
            api_url=SYSTEM_CONFIG.quantconnect.api_url
        )
        generator = StrategyGenerator(qc_client, SYSTEM_CONFIG)
        tester = StrategyTester(qc_client, SYSTEM_CONFIG)
        
        # Test engine creation (needs generator, tester, and evolution config)
        engine = EvolutionEngine(generator, tester, SYSTEM_CONFIG.evolution)
        print("  ✅ EvolutionEngine created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Evolution engine test failed: {str(e)}")
        return False

async def test_quantconnect_client():
    """Test QuantConnect API client (without making actual API calls)"""
    print("\n🔗 Testing QuantConnect Client...")
    
    try:
        from tier1_core.quantconnect_client import QuantConnectClient
        from config.settings import SYSTEM_CONFIG
        
        # Test client creation (needs user_id, token, api_url)
        client = QuantConnectClient(
            user_id=SYSTEM_CONFIG.quantconnect.user_id,
            token=SYSTEM_CONFIG.quantconnect.token,
            api_url=SYSTEM_CONFIG.quantconnect.api_url
        )
        print("  ✅ QuantConnectClient created")
        
        # Test API endpoint building
        assert client.api_url, "API URL should be set"
        assert client.user_id, "User ID should be set"
        assert client.token, "Token should be set"
        print("  ✅ API configuration valid")
        
        return True
        
    except Exception as e:
        print(f"  ❌ QuantConnect client test failed: {str(e)}")
        return False

async def run_all_tests():
    """Run all system integration tests"""
    print("🚀 3-TIER EVOLUTIONARY TRADING SYSTEM - INTEGRATION TESTS")
    print("=" * 70)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("System Controller Tests", test_system_controller),
        ("Dashboard Tests", test_dashboard),
        ("Strategy Components Tests", test_strategy_components),
        ("Agent System Tests", test_agent_system),
        ("Evolution Engine Tests", test_evolution_engine),
        ("QuantConnect Client Tests", test_quantconnect_client),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System ready for deployment.")
        print("\nTo start the system:")
        print("  python main.py")
        print("\nTo monitor the system:")
        print("  python monitor_system.py")
    else:
        print(f"\n⚠️  {failed} tests failed. Please resolve issues before deployment.")
    
    print(f"\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return failed == 0

async def main():
    """Main test execution"""
    try:
        success = await run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())