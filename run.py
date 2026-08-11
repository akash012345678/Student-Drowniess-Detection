"""
CLI Runner Script for Student Drowsiness & Attention System.
Validates dependencies and automatically starts the Streamlit server.
"""
import sys
import subprocess
import os
import warnings

# Suppress benign protobuf deprecation warnings triggered by MediaPipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

def check_dependencies():
    """Verify required Python packages are installed."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    required = ["cv2", "mediapipe", "numpy", "streamlit", "pandas", "scipy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    print("✅ All dependencies verified successfully!")

def main():
    """Launch Streamlit application."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("=" * 70)
    print("🎓 AI POWERED STUDENT DROWSINESS & ATTENTION MONITORING SYSTEM")
    print("=" * 70)
    check_dependencies()
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print(f"🚀 Launching Streamlit server: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Application terminated by user.")

if __name__ == "__main__":
    main()
