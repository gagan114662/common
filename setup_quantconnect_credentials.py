#!/usr/bin/env python3
"""
QuantConnect Credentials Setup Helper
Helps you configure the correct QuantConnect API credentials
"""

import sys
from pathlib import Path

def print_credential_help():
    """Print help for getting QuantConnect credentials"""
    print("🔐 QuantConnect API Credentials Setup")
    print("=" * 60)
    print()
    print("📋 Steps to get your QuantConnect API credentials:")
    print()
    print("1. 🌐 Go to: https://www.quantconnect.com/account")
    print("2. 🔑 Log into your QuantConnect account")
    print("3. 📊 Navigate to 'Organization' or 'Account Settings'")
    print("4. 🔧 Find 'API Access' or 'API Keys' section")
    print("5. 📝 Copy your:")
    print("   - User ID (should be a number)")
    print("   - API Token (long string)")
    print()
    print("🔍 Current configuration issues:")
    print("   - User ID: 357130 (may be invalid)")
    print("   - Token: Set but authentication failing")
    print()
    print("💡 Common issues:")
    print("   ❌ User ID format (should be just numbers)")
    print("   ❌ Expired or invalid API token")
    print("   ❌ Account not verified/activated")
    print("   ❌ API access not enabled")
    print()
    print("🔧 To fix:")
    print("1. Get fresh credentials from QuantConnect website")
    print("2. Update config/settings.py with correct values")
    print("3. Or set environment variables:")
    print("   export QC_USER_ID='your_user_id'")
    print("   export QC_TOKEN='your_api_token'")
    print()

def test_credential_format():
    """Test if credentials are in the right format"""
    print("🧪 Testing credential format...")
    
    try:
        from config.settings import SYSTEM_CONFIG
        
        user_id = SYSTEM_CONFIG.quantconnect.user_id
        token = SYSTEM_CONFIG.quantconnect.token
        
        print(f"User ID: '{user_id}' (type: {type(user_id).__name__})")
        print(f"Token: '{token[:10]}...' (length: {len(token)})")
        
        # Check user ID format
        try:
            int(user_id)
            print("✅ User ID is numeric")
        except ValueError:
            print("❌ User ID should be numeric")
        
        # Check token format
        if len(token) < 30:
            print("❌ Token seems too short (should be 40+ characters)")
        elif len(token) > 100:
            print("❌ Token seems too long")
        else:
            print("✅ Token length seems reasonable")
        
        return user_id, token
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return None, None

def create_test_config():
    """Create a test configuration file"""
    print("\n🔧 Would you like to create a test configuration?")
    print("This will help you test different credential combinations.")
    
    response = input("Create test config? (y/n): ").strip().lower()
    if response == 'y':
        user_id = input("Enter your QuantConnect User ID: ").strip()
        token = input("Enter your QuantConnect API Token: ").strip()
        
        test_config = f'''# Test QuantConnect Configuration
# Update config/settings.py with these values if they work

USER_ID = "{user_id}"
TOKEN = "{token}"
API_URL = "https://www.quantconnect.com/api/v2"

# To test: 
# 1. Update config/settings.py 
# 2. Run: python3 test_quantconnect.py
'''
        
        with open("test_credentials.py", "w") as f:
            f.write(test_config)
        
        print("✅ Created test_credentials.py")
        print("📝 Update config/settings.py with these values and test again")

def main():
    """Main function"""
    print_credential_help()
    
    user_id, token = test_credential_format()
    
    if user_id and token:
        print("\n📊 Current Status:")
        print("✅ Configuration file readable")
        print("❌ Authentication failing (credentials likely invalid)")
        print()
        print("🎯 Recommended Actions:")
        print("1. 🌐 Visit https://www.quantconnect.com/account")
        print("2. 🔑 Get fresh API credentials")
        print("3. 📝 Update config/settings.py")
        print("4. 🧪 Run: python3 test_quantconnect.py")
    
    create_test_config()

if __name__ == "__main__":
    main()