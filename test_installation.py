#!/usr/bin/env python3
"""
Quick verification script to test installation and data loading
"""

import sys

def test_imports():
    """Test that all required packages are installed"""
    print("🔍 Testing imports...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'plotly': 'plotly',
        'streamlit': 'streamlit',
    }
    
    missing = []
    for name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("📦 Install with: pip3 install -r requirements.txt")
        return False
    
    print("✅ All packages installed!\n")
    return True


def test_data_loader():
    """Test data loader functionality"""
    print("🔍 Testing data loader...")
    
    try:
        from data_loader import HyperliquidDataLoader
        print("  ✅ Data loader imported")
        
        # Test with a small sample
        loader = HyperliquidDataLoader(
            misc_events_path='Hyperliquid Data Expanded/node_fills_20251027_1700-1800.json'
        )
        print("  ✅ Loader initialized")
        
        # Load just 100 blocks as a test
        df = loader.load_misc_events(max_lines=100)
        
        if len(df) > 0:
            print(f"  ✅ Loaded {len(df)} trades from 100 blocks")
            print(f"  ✅ Found {df['coin'].nunique()} unique coins")
            print(f"  ✅ Found {df['address'].nunique()} unique traders")
            return True
        else:
            print("  ❌ No trades loaded")
            return False
            
    except FileNotFoundError:
        print("  ❌ Data file not found")
        print("  📁 Make sure you have the data files in 'Hyperliquid Data Expanded/' folder")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_visualizations():
    """Test visualization imports"""
    print("🔍 Testing visualizations...")
    
    try:
        from visualizations import (
            OrderBookHeatmap,
            MakerTakerFlow,
            SpreadAnalysis,
            VolumeProfile,
            TraderAnalytics
        )
        print("  ✅ All visualization classes imported")
        return True
    except Exception as e:
        print(f"  ❌ Error importing visualizations: {e}")
        return False


def test_dashboard():
    """Test dashboard imports"""
    print("🔍 Testing dashboard...")
    
    try:
        # Just check if it can be imported
        with open('dashboard.py', 'r') as f:
            content = f.read()
            if 'streamlit' in content and 'def main' in content:
                print("  ✅ Dashboard file looks good")
                return True
            else:
                print("  ❌ Dashboard file incomplete")
                return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("🧪 Hyperliquid Dashboard Installation Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Visualizations", test_visualizations()))
    results.append(("Data Loader", test_data_loader()))
    results.append(("Dashboard", test_dashboard()))
    
    # Summary
    print()
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! You're ready to go!")
        print()
        print("🚀 Next steps:")
        print("   1. Run: streamlit run dashboard.py")
        print("   2. Open browser to: http://localhost:8501")
        print()
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print()
        print("💡 Common fixes:")
        print("   - Install packages: pip3 install -r requirements.txt")
        print("   - Check data files are in 'Hyperliquid Data Expanded/' folder")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

