#!/usr/bin/env python3
"""
Bitcoin Forecaster Dashboard Launcher
=====================================

Simple launcher script to start the Streamlit web dashboard for the Bitcoin forecasting system.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 'plotly', 
        'seaborn', 'scikit-learn', 'tensorflow', 'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n📦 Install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("🚀 Bitcoin Price Forecaster")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("streamlit_dashboard.py").exists():
        print("❌ Error: streamlit_dashboard.py not found in current directory")
        print("Please run this script from the project root directory.")
        return
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return
    
    print("✅ All dependencies found!")
    print("\n🌐 Starting web dashboard...")
    print("📱 The dashboard will open in your web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n⚠️  To stop the dashboard, press Ctrl+C in this terminal")
    print("=" * 40)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_dashboard.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. Thank you for using Bitcoin Forecaster!")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("Try running manually with: streamlit run streamlit_dashboard.py")

if __name__ == "__main__":
    launch_dashboard()