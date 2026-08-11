"""
Main Entry Point for Streamlit Application.
Launch with: streamlit run app.py
"""
import sys
import os
import warnings

# Suppress benign protobuf deprecation warnings triggered by MediaPipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.dashboard import run_dashboard

if __name__ == "__main__":
    run_dashboard()
