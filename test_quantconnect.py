#!/usr/bin/env python3
"""
Test QuantConnect API Connectivity
Simple test to verify the credentials and API are working
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tier1_core.quantconnect_client import QuantConnectClient
from config.settings import SYSTEM_CONFIG

async def test_quantconnect():
    """Test QuantConnect API connectivity"""
    print("🧪 Testing QuantConnect API Connectivity")
    print("=" * 50)
    print(f"User ID: {SYSTEM_CONFIG.quantconnect.user_id}")
    print(f"API URL: {SYSTEM_CONFIG.quantconnect.api_url}")
    print("=" * 50)
    
    try:
        # Create client
        client = QuantConnectClient(
            user_id=SYSTEM_CONFIG.quantconnect.user_id,
            token=SYSTEM_CONFIG.quantconnect.token,
            api_url=SYSTEM_CONFIG.quantconnect.api_url
        )
        
        print("🔌 Connecting to QuantConnect...")
        
        async with client:
            # Test authentication
            print("🔐 Testing authentication...")
            auth_success = await client.authenticate()
            
            if auth_success:
                print("✅ Authentication successful!")
                
                # Get projects
                print("📁 Fetching projects...")
                projects = await client.get_projects()
                print(f"✅ Found {len(projects)} existing projects")
                
                if projects:
                    print("\n📋 Your Projects:")
                    for i, project in enumerate(projects[:5]):  # Show first 5
                        print(f"   {i+1}. {project.name} (ID: {project.project_id})")
                        print(f"      Created: {project.created.strftime('%Y-%m-%d')}")
                        print(f"      Language: {project.language}")
                
                # Test creating a new project
                print("\n🔧 Testing project creation...")
                test_project_id = await client.create_project(
                    name=f"Test_Evolution_System_{int(asyncio.get_event_loop().time())}",
                    language="Python"
                )
                print(f"✅ Created test project: {test_project_id}")
                
                # Get performance stats
                stats = client.get_performance_stats()
                print(f"\n📊 Client Performance:")
                print(f"   Total Requests: {stats['total_requests']}")
                print(f"   Success Rate: {stats['success_rate_percent']:.1f}%")
                print(f"   Average Response Time: {stats['average_response_time_ms']:.1f}ms")
                
                print("\n🎉 QuantConnect API is working perfectly!")
                print("✅ Ready to start the real trading system")
                return True
                
            else:
                print("❌ Authentication failed!")
                print("🔍 Check your credentials in config/settings.py")
                return False
                
    except Exception as e:
        print(f"❌ QuantConnect test failed: {e}")
        print("🔍 Possible issues:")
        print("   - Internet connectivity")
        print("   - Invalid credentials")
        print("   - QuantConnect API issues")
        return False

if __name__ == "__main__":
    print("🚀 QuantConnect API Test")
    success = asyncio.run(test_quantconnect())
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Start real system: python3 -c \"import asyncio; from main import main; asyncio.run(main())\"")
        print("2. Start with monitoring: python3 start_real_system.py")
        print("3. Web dashboard only: python3 dashboard_viewer.py")
        sys.exit(0)
    else:
        print("\n❌ Fix the issues above before starting the real system")
        sys.exit(1)