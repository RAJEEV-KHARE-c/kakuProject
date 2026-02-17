#!/usr/bin/env python3
"""
ESP32 Dashboard Backend Server with AI Vision Analysis
Handles data from ESP32 sensors, stores in database, serves real-time dashboard
and automatically analyzes images with AI-generated captions
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room
import sqlite3
import json
import os
from datetime import datetime
from threading import Thread, Lock
import base64

# AI Vision imports - Google ViT Model
try:
    from PIL import Image
    import numpy as np
    from transformers import pipeline
    import torch
    AI_AVAILABLE = True
    AI_MODE = "FULL"
    print("✅ Google ViT AI Vision model enabled (FULL mode with transformers)")
except ImportError as e:
    print(f"⚠️ FullAI modules not available: {e}")
    try:
        from PIL import Image
        import numpy as np
        AI_AVAILABLE = True
        AI_MODE = "BASIC"
        print("⚠️ Fallback to Basic AI Vision mode (PIL + image analysis)")
    except ImportError as e2:
        print(f"❌ No AI modules available: {e2}")
        AI_AVAILABLE = False
        AI_MODE = "NONE"

from collections import deque
step_counter_lock = Lock()
step_count_global = 0  # Total steps counted
accel_history = deque(maxlen=20)  # Keep last 20 acceleration readings (enough for 2-second span @ 100ms intervals)
last_step_time = 0  # Prevent duplicate step detection

# 👟 STEP DETECTION PARAMETERS (Optimized for 100ms buffered readings)
# Now with 20 readings per 2 seconds = 100ms intervals = much better temporal resolution
# Can detect faster changes and acceleration peaks more accurately
STEP_DETECTION_THRESHOLD = 0.3  # m/s² - Increased to reduce false positives
STEP_MIN_INTERVAL = 1  # seconds - minimum time between steps (with 100ms reads, can detect ~2 steps/sec)

def detect_steps(accel_x, accel_y, accel_z, current_time):
    """Detect steps from accelerometer data using magnitude changes
    ...existing code...
