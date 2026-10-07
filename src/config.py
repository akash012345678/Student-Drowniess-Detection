"""
Configuration and Constants Module for Student Drowsiness & Attention Monitoring System.
"""
import json
import os
import numpy as np

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(BASE_DIR, "config", "settings.json")


def load_settings():
    """Load settings from JSON file or return defaults if missing."""
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r") as f:
            return json.load(f)
    return {
        "thresholds": {
            "default_ear_threshold": 0.20,
            "default_mar_threshold": 0.40,
            "blink_ear_threshold": 0.14,
            "pitch_up_threshold": -18.0,
            "pitch_down_threshold": 18.0,
            "yaw_left_threshold": -22.0,
            "yaw_right_threshold": 22.0,
            "roll_threshold": 20.0,
        },
        "temporal": {
            "buffer_size": 90,
            "drowsy_duration_sec": 2.0,
            "distraction_duration_sec": 2.0,
            "yawn_duration_sec": 0.8,
            "calibration_duration_sec": 10.0,
            "target_fps": 30,
        },
        "low_light": {
            "luminance_threshold": 45.0,
            "clahe_clip_limit": 2.5,
            "clahe_grid_size": [8, 8],
        },
        "paths": {
            "reports_dir": "reports",
            "screenshots_dir": "reports/screenshots",
            "logs_dir": "logs",
            "assets_dir": "assets",
        },
        "admin_password": "Password@123",
    }


SETTINGS = load_settings()

# Directories Setup
REPORTS_DIR = os.path.join(BASE_DIR, SETTINGS["paths"]["reports_dir"])
SCREENSHOTS_DIR = os.path.join(BASE_DIR, SETTINGS["paths"]["screenshots_dir"])
LOGS_DIR = os.path.join(BASE_DIR, SETTINGS["paths"]["logs_dir"])
ASSETS_DIR = os.path.join(BASE_DIR, SETTINGS["paths"]["assets_dir"])

for directory in [REPORTS_DIR, SCREENSHOTS_DIR, LOGS_DIR, ASSETS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Official MediaPipe Facial Landmark Indices
# Left Eye (P1=outer, P2=top-right, P3=top-left, P4=inner, P5=bottom-left, P6=bottom-right)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Right Eye (P1=inner, P2=top-left, P3=top-right, P4=outer, P5=bottom-right, P6=bottom-left)
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Mouth Indices for MAR (corners: 61, 291; vertical pairs: (81, 178), (13, 14), (311, 402))
MOUTH_INDICES = {
    "left": 61,
    "right": 291,
    "top_1": 81,
    "bottom_1": 178,
    "top_2": 13,
    "bottom_2": 14,
    "top_3": 311,
    "bottom_3": 402,
}

# 2D Landmarks for Head Pose Estimation (solvePnP)
POSE_2D_LANDMARKS = [
    1,    # Nose tip
    152,  # Chin
    33,   # Left eye left corner
    263,  # Right eye right corner
    61,   # Left mouth corner
    291   # Right mouth corner
]

# 3D Model Points for Canonical Face Model (in mm)
POSE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, 330.0, -65.0),      # Chin
    (-225.0, -170.0, -135.0), # Left eye left corner
    (225.0, -170.0, -135.0),  # Right eye right corner
    (-150.0, 150.0, -125.0),  # Left mouth corner
    (150.0, 150.0, -125.0)    # Right mouth corner
], dtype=np.float64)

# Color Palette for Visual Overlays (BGR Format)
COLOR_GREEN = (0, 230, 115)     # Alert
COLOR_YELLOW = (0, 215, 255)    # Distracted / Warning
COLOR_RED = (50, 50, 255)       # Drowsy / Alert
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GRAY = (40, 40, 40)
COLOR_CYAN = (255, 255, 0)
