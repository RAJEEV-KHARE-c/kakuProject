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

# ================= STEP COUNTER STATE =================
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
    
    Optimized for 100ms buffered readings (20 readings per 2-second batch):
    - Much better temporal resolution = can detect local acceleration peaks
    - Walking produces clear acceleration peaks 0.4-0.8 seconds apart
    - Uses peak detection: when previous reading is higher than both neighbors
    - Can maintain ~2 steps/sec walking pace with 0.5s minimum interval
    """
    global step_count_global, last_step_time
    
    # --- STOSS/BARRIER METHOD (Arduino style) ---
    stoss = (accel_x ** 2) + (accel_y ** 2) + (accel_z ** 2)
    if not hasattr(detect_steps, 'last_step_time'):
        detect_steps.last_step_time = 0
    if not hasattr(detect_steps, 'treshold'):
        detect_steps.treshold = 1  # Default sensitivity level
    steps_detected = 0
    # Lowered barrier for more sensitive step detection
    barrier = 10000  # Set barrier to match sensor data scale
    min_interval = 0.2  # Lowered interval for more frequent step detection
    time_since_last_step = current_time - detect_steps.last_step_time
    if stoss > barrier and time_since_last_step > min_interval:
        steps_detected = 1
        detect_steps.last_step_time = current_time
        with step_counter_lock:
            step_count_global += 1
        print(f'     ✅👣 STEP #{step_count_global}! stoss: {stoss:.0f} > barrier: {barrier} | interval: {time_since_last_step:.2f}s')
    return steps_detected

# ================= CREATE FLASK APP =================
app = Flask(__name__, static_folder='.', static_url_path='')

# ================= BUFFER CONFIGURATION =================
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max request size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Add stability configurations
import logging
logging.basicConfig(level=logging.ERROR)  # Reduce logging noise
app.logger.setLevel(logging.ERROR)

CORS(app)
socketio = SocketIO(app, 
    cors_allowed_origins="*",
    max_http_buffer_size=10*1024*1024,  # 10MB WebSocket buffer
    ping_timeout=60,
    ping_interval=25,
    logger=False,  # Disable SocketIO logging
    engineio_logger=False  # Disable EngineIO logging
)

# Database configuration
DB_PATH = 'sensor_data.db'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

# AI Configuration
if AI_AVAILABLE:
    CACHE_DIR = r"E:\Rajeev\esp 32\esp 32\.cache\huggingface"
    os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Determine device: GPU if available, otherwise CPU
    try:
        AI_DEVICE = 0 if torch.cuda.is_available() else -1
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ GPU not available, using CPU for AI analysis")
    except:
        AI_DEVICE = -1
        print("⚠️ Using CPU for AI analysis")
    
    # Initialize AI model (lazy loading)
    ai_classifier = None

# ================= ORIENTATION DETECTION (Server-side) =================
def detect_device_orientation(ax, ay, az):
    """
    Detect device orientation from raw accelerometer data
    This computation is now done on the server (not on ESP32)
    
    Returns: (direction_string, confidence_percentage)
    """
    try:
        # Normalize accelerometer values (remove gravity bias)
        magnitude = (ax**2 + ay**2 + az**2) ** 0.5
        
        # Calculate confidence based on how close to 1g the total acceleration is
        # 1g = 9.81 m/s² (gravity only, device not accelerating)
        confidence = 100.0 - abs(magnitude - 9.81) * 10.0
        if confidence > 100.0:
            confidence = 100.0
        if confidence < 0.0:
            confidence = 0.0
        
        # Determine dominant axis and direction
        abs_ax = abs(ax)
        abs_ay = abs(ay)
        abs_az = abs(az)
        
        # Z-axis dominant (device flat or inverted)
        if abs_az > abs_ax and abs_az > abs_ay:
            if az > 7.0:
                return "NEUTRAL", confidence      # Device flat, Z pointing up
            if az < -7.0:
                return "INVERTED", confidence     # Device flipped, Z pointing down
        
        # X-axis dominant (device tilted left/right)
        if abs_ax > abs_ay and abs_ax > abs_az:
            if ax > 5.0:
                return "RIGHT", confidence        # Device tilted right
            if ax < -5.0:
                return "LEFT", confidence         # Device tilted left
        
        # Y-axis dominant (device tilted forward/back)
        if abs_ay > abs_ax and abs_ay > abs_az:
            if ay > 5.0:
                return "BACK", confidence         # Device tilted back
            if ay < -5.0:
                return "FORWARD", confidence      # Device tilted forward
        
        return "NEUTRAL", confidence              # Default fallback
    
    except Exception as e:
        print(f"❌ Error in orientation detection: {e}")
        return "UNKNOWN", 0.0

# AI Analysis Function with fallback
def analyze_image_with_ai(image_path):
    """Analyze image using AI models or basic image analysis as fallback"""
    global ai_classifier
    
    if not AI_AVAILABLE:
        return "AI analysis not available - missing dependencies"
    
    try:
        if AI_MODE == "FULL":
            # Full AI Model Analysis (Google ViT)
            if ai_classifier is None:
                print("Loading Google ViT vision model...")
                ai_classifier = pipeline(
                    "image-classification",
                    model="google/vit-base-patch16-224",
                    device=AI_DEVICE
                )
                print("Google ViT model loaded successfully")
            
            # Load and analyze image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get AI predictions
            results = ai_classifier(image, top_k=5)
            
            # Generate natural caption 
            top_result = results[0]
            confidence = top_result['score'] * 100
            main_label = top_result['label']
            
            # Check for people-related content
            people_keywords = ['people', 'person', 'group', 'crowd', 'team', 'family', 'human', 'face', 'portrait']
            people_detected = any(keyword in result['label'].lower() for result in results for keyword in people_keywords)
            
            # Generate natural, descriptive caption
            if people_detected:
                caption = f"This image shows a group of people (detected with {confidence:.1f}% confidence)"
            else:
                main_label_clean = main_label.replace('_', ' ').replace('-', ' ')
                caption = f"This image shows {main_label_clean} (detected with {confidence:.1f}% confidence)"
            
            return caption
            
        elif AI_MODE == "BASIC":
            # Basic Image Analysis (PIL + Visual Features)
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image properties
            width, height = image.size
            img_array = np.array(image)
            
            # Basic color analysis
            mean_colors = np.mean(img_array, axis=(0, 1))
            color_variance = np.var(img_array.reshape(-1, 3), axis=0)
            total_variance = np.sum(color_variance)
            brightness = np.mean(img_array)
            
            # Basic feature detection
            aspect_ratio = width / height
            
            # Generate descriptive caption based on visual features
            caption_parts = []
            
            # Resolution description
            if width * height > 100000:
                caption_parts.append("high-resolution")
            else:
                caption_parts.append("compact")
            
            # Color description
            if brightness > 200:
                caption_parts.append("bright")
            elif brightness < 80:
                caption_parts.append("dark")
            else:
                caption_parts.append("well-lit")
            
            # Orientation
            if aspect_ratio > 1.5:
                caption_parts.append("landscape-oriented")
            elif aspect_ratio < 0.7:
                caption_parts.append("portrait-oriented")
            else:
                caption_parts.append("square-oriented")
            
            # Color richness
            if total_variance > 8000:
                caption_parts.append("colorful scene")
            elif total_variance > 3000:
                caption_parts.append("moderately colorful image")
            else:
                caption_parts.append("simple colored image")
            
            # Dominant color
            r, g, b = mean_colors
            if r > g and r > b:
                caption_parts.append("with reddish tones")
            elif g > r and g > b:
                caption_parts.append("with greenish tones")
            elif b > r and b > g:
                caption_parts.append("with bluish tones")
            
            caption = f"This is a {' '.join(caption_parts[:4])} captured from ESP32 camera"
            
            # Add technical details
            caption += f" (Resolution: {width}×{height}, Brightness: {brightness:.0f}/255)"
            
            return caption
    
    except Exception as e:
        print(f"AI Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return f"Image analysis failed: {str(e)[:100]}..."

        return f"Image analysis failed: {str(e)[:100]}..."
    
    return "Image received but analysis unavailable"

# ================= BACKGROUND AI ANALYSIS (NON-BLOCKING) =================
def analyze_and_store_image(filepath, filename):
    """Background task: Run AI analysis and store result without blocking server"""
    try:
        print(f"🤖 [BACKGROUND] Starting AI analysis for {filename}...")
        ai_caption = analyze_image_with_ai(filepath)
        
        # Store in database
        with db_lock:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE sensor_readings 
                        SET image_filename = ?, ai_caption = ?
                        WHERE id = (SELECT MAX(id) FROM sensor_readings WHERE accel_x IS NOT NULL)
                    ''', (filename, ai_caption))
                    
                    if cursor.rowcount == 0:
                        cursor.execute('''
                            INSERT INTO sensor_readings (device_id, image_filename, ai_caption, timestamp)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        ''', ('ESP32_CAM', filename, ai_caption))
                    
                    conn.commit()
                    print(f"✅ [BG] Stored AI caption in database")
                except sqlite3.Error as e:
                    print(f"❌ [BG] Database error: {e}")
                finally:
                    conn.close()
        
        # Broadcast result to dashboard
        image_url = f'/uploads/images/{filename}'
        broadcast_camera_update(
            image_url=image_url,
            ai_caption=ai_caption,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"✅ [BG] AI complete: {ai_caption[:60]}...")
    
    except Exception as e:
        print(f"❌ [BG] AI analysis error: {e}")

# Helper function for broadcasting camera updates to all connected clients
def broadcast_camera_update(image_url, ai_caption, timestamp=None):
    """
    Broadcast camera update event to all connected WebSocket clients
    Uses a background task to ensure proper context for emission
    """
    def emit_update():
        with app.app_context():
            socketio.emit('camera_update', {
                'image_url': image_url,
                'ai_caption': ai_caption,
                'timestamp': timestamp or datetime.now().isoformat(),
                'device_id': 'ESP32_CAM'
            })
            print(f"📡 Broadcasted camera update to all connected clients")
    
    socketio.start_background_task(emit_update)

def broadcast_step_counter_update(total_steps, daily_steps=0):
    """
    Broadcast step counter update event to all connected WebSocket clients
    Uses a background task to ensure proper context for emission
    """
    def emit_update():
        with app.app_context():
            socketio.emit('step_counter_updated', {
                'total_steps': total_steps,
                'daily_steps': daily_steps,
                'timestamp': datetime.now().isoformat(),
                'device_id': 'ESP32_001'
            })
            print(f"👟 Broadcasted step update: {total_steps} steps to all connected clients")
    
    socketio.start_background_task(emit_update)

# Endpoint to fetch image from database by sensor_readings id
from flask import Response
@app.route('/api/image/<int:image_id>')
def get_image(image_id):
    with db_lock:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute('SELECT camera_image FROM sensor_readings WHERE id=?', (image_id,))
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                return Response(row[0], mimetype='image/jpeg')
    return jsonify({'error': 'Image not found'}), 404

# Thread-safe database helper
db_lock = Lock()

def get_db_connection():
    """Get thread-safe database connection with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=10000;")
        conn.execute("PRAGMA temp_store=memory;")
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

# Initialize database
def init_database():
    """Initialize database with proper error handling"""
    with db_lock:
        conn = get_db_connection()
        if not conn:
            print("❌ Failed to initialize database")
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT DEFAULT 'ESP32_001',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accel_x REAL,
                    accel_y REAL,
                    accel_z REAL,
                    gyro_x REAL,
                    gyro_y REAL,
                    gyro_z REAL,
                    mic_level REAL,
                    sound_data INTEGER,
                    camera_image BLOB,
                    audio_data BLOB,
                    image_filename TEXT,
                    ai_caption TEXT,
                    device_orientation TEXT,
                    orientation_confidence REAL,
                    calibrated_ax REAL,
                    calibrated_ay REAL,
                    calibrated_az REAL
                )
            ''')
            
            # Add new columns if they don't exist (for existing databases)
            cursor.execute("PRAGMA table_info(sensor_readings)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'image_filename' not in columns:
                cursor.execute("ALTER TABLE sensor_readings ADD COLUMN image_filename TEXT")
                print("✅ Added image_filename column")
                
            if 'ai_caption' not in columns:
                cursor.execute("ALTER TABLE sensor_readings ADD COLUMN ai_caption TEXT")
                print("✅ Added ai_caption column for AI analysis")
                
            if 'device_orientation' not in columns:
                cursor.execute("ALTER TABLE sensor_readings ADD COLUMN device_orientation TEXT")
                print("✅ Added device_orientation column")
                
            if 'orientation_confidence' not in columns:
                cursor.execute("ALTER TABLE sensor_readings ADD COLUMN orientation_confidence REAL")
                print("✅ Added orientation_confidence column")
                
            if 'calibrated_ax' not in columns:
                cursor.execute("ALTER TABLE sensor_readings ADD COLUMN calibrated_ax REAL")
                print("✅ Added calibrated_ax column")
                
            if 'calibrated_ay' not in columns:
                cursor.execute("ALTER TABLE sensor_readings ADD COLUMN calibrated_ay REAL")
                print("✅ Added calibrated_ay column")
                
            if 'calibrated_az' not in columns:
                cursor.execute("ALTER TABLE sensor_readings ADD COLUMN calibrated_az REAL")
                print("✅ Added calibrated_az column")
            
            # Add step_count column for step tracking
            cursor.execute("PRAGMA table_info(sensor_readings)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'step_count' not in columns:
                cursor.execute("ALTER TABLE sensor_readings ADD COLUMN step_count INTEGER DEFAULT 0")
                print("✅ Added step_count column")
            
            # Create step_statistics table for aggregated step data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS step_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT DEFAULT 'ESP32_001',
                    date_recorded DATE DEFAULT CURRENT_DATE,
                    total_steps INTEGER DEFAULT 0,
                    peak_steps INTEGER DEFAULT 0,
                    avg_step_interval REAL DEFAULT 0.0,
                    activity_level TEXT DEFAULT 'INACTIVE',
                    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add columns if they don't exist
            cursor.execute("PRAGMA table_info(step_statistics)")
            stat_columns = [column[1] for column in cursor.fetchall()]
            
            if 'peak_steps' not in stat_columns:
                cursor.execute("ALTER TABLE step_statistics ADD COLUMN peak_steps INTEGER DEFAULT 0")
                print("✅ Added peak_steps column to step_statistics")
            
            if 'avg_step_interval' not in stat_columns:
                cursor.execute("ALTER TABLE step_statistics ADD COLUMN avg_step_interval REAL DEFAULT 0.0")
                print("✅ Added avg_step_interval column")
            
            if 'activity_level' not in stat_columns:
                cursor.execute("ALTER TABLE step_statistics ADD COLUMN activity_level TEXT DEFAULT 'INACTIVE'")
                print("✅ Added activity_level column")
            
            if 'updated_at' not in stat_columns:
                cursor.execute("ALTER TABLE step_statistics ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP")
                print("✅ Added updated_at column")
            
            print("✅ Created step_statistics table")            
            
            # Add device_id column if it doesn't exist (for existing databases)
            cursor.execute("PRAGMA table_info(sensor_readings)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'device_id' not in columns:
                cursor.execute("ALTER TABLE sensor_readings ADD COLUMN device_id TEXT DEFAULT 'ESP32_001'")
                print("✅ Added device_id column to existing sensor_readings table")
            
            # Create important_events table for ESP32 event polling
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS important_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT DEFAULT 'ESP32_001',
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL, 
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_sent BOOLEAN DEFAULT 0
                )
            ''')
            
            # Create oled_display_state table for display commands
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS oled_display_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT DEFAULT 'ESP32_001',
                    animation_type TEXT DEFAULT 'pet',
                    animation_id INTEGER DEFAULT 1,
                    animation_name TEXT DEFAULT 'CHILD',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_by TEXT DEFAULT 'web_ui'
                )
            ''')
            
            # Initialize default OLED state if not exists
            cursor.execute('SELECT COUNT(*) FROM oled_display_state')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO oled_display_state 
                    (device_id, animation_type, animation_id, animation_name, updated_by)
                    VALUES (?, ?, ?, ?, ?)
                ''', ('ESP32_001', 'pet', 1, 'CHILD', 'system_init'))
                print("✅ Initialized default OLED display state in database")
            
            conn.commit()
            print("✅ Database initialized successfully")
            return True
        except sqlite3.Error as e:
            print(f"❌ Database initialization error: {e}")
            return False
        finally:
            conn.close()

# Initialize database on startup
init_database()

# ==================== IMAGE CLEANUP TASK ====================
import time
from threading import Lock

image_cleanup_lock = Lock()

def cleanup_old_images():
    """Delete all images except the latest one every 30 seconds"""
    while True:
        try:
            time.sleep(30)  # Wait 30 seconds before first cleanup (HIGH FREQUENCY)
            
            with image_cleanup_lock:
                uploads_dir = os.path.join(os.getcwd(), 'uploads', 'images')
                
                if not os.path.exists(uploads_dir):
                    continue
                
                # Get all image files with their modification times
                image_files = []
                for filename in os.listdir(uploads_dir):
                    filepath = os.path.join(uploads_dir, filename)
                    if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        mod_time = os.path.getmtime(filepath)
                        image_files.append((filepath, filename, mod_time))
                
                # Keep only the latest image, delete the rest
                if len(image_files) > 1:
                    image_files.sort(key=lambda x: x[2], reverse=True)
                    latest_file = image_files[0][1]
                    
                    deleted_count = 0
                    for filepath, filename, _ in image_files[1:]:
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                            print(f"Deleted old image: {filename}")
                            
                            # Keep AI captions in database (preserve knowledge/experience)
                        except Exception as e:
                            print(f"Error deleting image {filename}: {e}")
                    
                    if deleted_count > 0:
                        print(f"Cleanup: Deleted {deleted_count} old images, kept latest: {latest_file}")
        except Exception as e:
            print(f"Error in cleanup task: {e}")

# Start cleanup thread
cleanup_thread = Thread(target=cleanup_old_images, daemon=True)
cleanup_thread.start()
print("Image cleanup task started (runs every 30 seconds, keeps latest image + AI captions)")

# ==================== STEP STATISTICS UPDATE TASK ====================

def update_step_statistics():
    """Periodically aggregate step data and update statistics table"""
    while True:
        try:
            time.sleep(60)  # Update every 60 seconds
            
            with db_lock:
                conn = get_db_connection()
                if not conn:
                    continue
                
                cursor = conn.cursor()
                
                # Get today's date
                today = datetime.now().date()
                device_id = 'ESP32_001'
                
                # Calculate today's step statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as batch_count,
                        SUM(step_count) as total_today,
                        MAX(step_count) as peak_steps,
                        AVG(CASE WHEN step_count > 0 THEN step_count ELSE NULL END) as avg_steps_per_batch
                    FROM sensor_readings
                    WHERE device_id = ? AND DATE(timestamp) = ?
                ''', (device_id, today))
                
                result = cursor.fetchone()
                if result:
                    batch_count, total_today, peak_steps, avg_steps = result
                    total_today = total_today or 0
                    peak_steps = peak_steps or 0
                    avg_steps = avg_steps or 0.0
                    
                    # Determine activity level based on total steps
                    if total_today == 0:
                        activity = 'INACTIVE'
                    elif total_today < 500:
                        activity = 'LOW'
                    elif total_today < 2000:
                        activity = 'MODERATE'
                    elif total_today < 5000:
                        activity = 'HIGH'
                    else:
                        activity = 'VERY_HIGH'
                    
                    # Update or insert today's statistics
                    cursor.execute('''
                        UPDATE step_statistics
                        SET total_steps = ?, peak_steps = ?, avg_step_interval = ?, 
                            activity_level = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE device_id = ? AND date_recorded = ?
                    ''', (total_today, peak_steps, avg_steps, activity, device_id, today))
                    
                    if cursor.rowcount == 0:
                        # Insert new record if doesn't exist
                        cursor.execute('''
                            INSERT INTO step_statistics 
                            (device_id, date_recorded, total_steps, peak_steps, avg_step_interval, activity_level)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (device_id, today, total_today, peak_steps, avg_steps, activity))
                    
                    conn.commit()
                    print(f"📊 Step statistics updated: {total_today} total | {peak_steps} peak | Activity: {activity}")
                
                conn.close()
        
        except Exception as e:
            print(f"❌ Error in step statistics update: {e}")

# Start statistics update thread
stats_thread = Thread(target=update_step_statistics, daemon=True)
stats_thread.start()
print("Step statistics update task started (runs every 60 seconds)")

# Connected clients  
connected_clients = set()

# ==================== WebSocket Events ====================

@socketio.on('connect')
def handle_connect():
    try:
        print(f'Client connected: {request.sid}')
        connected_clients.add(request.sid)
        def emit_connection():
            with app.app_context():
                socketio.emit('connection_response', {'status': 'Connected to dashboard'})
        socketio.start_background_task(emit_connection)
    except Exception as e:
        print(f'Connection error: {e}')

@socketio.on('disconnect')
def handle_disconnect():
    try:
        print(f'Client disconnected: {request.sid}')
        connected_clients.discard(request.sid)
    except Exception as e:
        print(f'Disconnect error: {e}')

@socketio.on('sensor_data')
def handle_sensor_data(data):
    """Receive sensor data from ESP32 and broadcast to all connected clients"""
    try:
        print(f'Received sensor data: {data}')
        
        # Store in database
        store_sensor_data(data)
        
        # Broadcast to all connected clients
        def emit_sensor_data():
            with app.app_context():
                socketio.emit('sensor_update', data)
        socketio.start_background_task(emit_sensor_data)
        
        return {'status': 'success', 'message': 'Data received and stored'}
    except Exception as e:
        print(f'Error processing sensor data: {e}')
        return {'status': 'error', 'message': str(e)}

# ==================== REST API Endpoints ====================

@app.route('/')
def index():
    """Serve the main dashboard HTML"""
    return app.send_static_file('index.html')

@app.route('/api/sensor-data', methods=['POST'])
def receive_sensor_data():
    """Receive sensor data from ESP32 and compute orientation on server"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data received'}), 400
        
        # Extract accelerometer data
        accel_x = data.get('accel_x', 0)
        accel_y = data.get('accel_y', 0)
        accel_z = data.get('accel_z', 0)
        
        # 👣 COMPUTE STEP COUNT ON SERVER
        import time
        current_time = time.time()
        
        # NEW: Process sensor batch if available (multiple readings from ESP32)
        total_steps_batch = 0
        
        # DEBUG: Log if batch exists
        if data.get('sensor_batch'):
            print(f"\n🎯 SENSOR BATCH RECEIVED!")
            print(f"   ├─ reading_count: {data['sensor_batch'].get('reading_count', 0)}")
            print(f"   ├─ avg_mic_level: {data['sensor_batch'].get('avg_mic_level', 0):.1f}")
            print(f"   ├─ sound_data: {data['sensor_batch'].get('sound_data', 0)}")
            readings = data['sensor_batch'].get('readings', [])
            print(f"   └─ readings array length: {len(readings)}")
        else:
            print(f"\n⚠️  NO SENSOR BATCH! Server falling back to single reading detection")
        
        if data.get('sensor_batch') and data['sensor_batch'].get('readings'):
            readings = data['sensor_batch']['readings']
            print(f"\n📦 Processing {len(readings)} readings from batch...")
            print(f"   Threshold: {STEP_DETECTION_THRESHOLD} m/s² | Min Interval: {STEP_MIN_INTERVAL}s")
            print(f"   History size: {len(accel_history)} | Max size: {accel_history.maxlen}")
            
            for idx, reading in enumerate(readings):
                batch_accel_x = reading.get('accel_x', 0)
                batch_accel_y = reading.get('accel_y', 0)
                batch_accel_z = reading.get('accel_z', 0)
                batch_time = current_time + (idx * 0.1)  # Approximate timing based on index
                
                # Calculate magnitude before feeding to detect_steps
                batch_magnitude = (batch_accel_x**2 + batch_accel_y**2 + batch_accel_z**2) ** 0.5
                
                steps_in_reading = detect_steps(batch_accel_x, batch_accel_y, batch_accel_z, batch_time)
                total_steps_batch += steps_in_reading
                
                # Show every 5th reading + any with steps
                if idx % 5 == 0 or steps_in_reading > 0:
                    print(f"   ├─ R[{idx:2d}] ax={batch_accel_x:7.2f} ay={batch_accel_y:7.2f} az={batch_accel_z:7.2f} | mag={batch_magnitude:.3f} → {steps_in_reading} step{'s' if steps_in_reading != 1 else ''}")
            
            print(f"📊 Batch processing complete: {total_steps_batch} steps detected\n")
        else:
            # Fall back to single reading detection
            print(f"   ├─ Single reading: ax={accel_x:.2f} ay={accel_y:.2f} az={accel_z:.2f}")
            steps_in_reading = detect_steps(accel_x, accel_y, accel_z, current_time)
            total_steps_batch = steps_in_reading
            print(f"   └─ Steps detected: {total_steps_batch}\n")
        
        # 🧭 COMPUTE ORIENTATION ON SERVER (moved from ESP32)
        direction, confidence = detect_device_orientation(accel_x, accel_y, accel_z)
        
        # Add computed values to data
        data['device_orientation'] = direction
        data['orientation_confidence'] = confidence
        data['calibrated_ax'] = accel_x
        data['calibrated_ay'] = accel_y
        data['calibrated_az'] = accel_z
        data['step_count'] = total_steps_batch
        
        print(f'📊 Sensor data: accel=({accel_x:.2f}, {accel_y:.2f}, {accel_z:.2f}) '
              f'gyro=({data.get("gyro_x", 0):.2f}, {data.get("gyro_y", 0):.2f}, {data.get("gyro_z", 0):.2f}) '
              f'mic={data.get("mic_level", 0):.1f}dB steps_detected={total_steps_batch} total={step_count_global}')
        print(f'🧭 Direction computed on server: {direction} ({confidence:.1f}% confidence)')
        print(f'👣 Total step count: {step_count_global}')
        
        # Store safely in database (including computed orientation)
        success = store_sensor_data(data)
        if not success:
            return jsonify({'status': 'error', 'message': 'Database storage failed'}), 500
        
        # Broadcast to connected clients with orientation data
        try:
            def emit_sensor_update():
                with app.app_context():
                    socketio.emit('sensor_update', {
                        'timestamp': datetime.now().isoformat(),
                        'device_id': data.get('device_id', 'ESP32_001'),
                        'accel_x': accel_x,
                        'accel_y': accel_y, 
                        'accel_z': accel_z,
                        'gyro_x': data.get('gyro_x', 0),
                        'gyro_y': data.get('gyro_y', 0),
                        'gyro_z': data.get('gyro_z', 0),
                        'mic_level': data.get('mic_level', 0),
                        'sound_data': data.get('sound_data', 0)
                    })
            
            def emit_orientation():
                with app.app_context():
                    socketio.emit('orientation_update', {
                        'timestamp': datetime.now().isoformat(),
                        'device_id': data.get('device_id', 'ESP32_001'),
                        'direction': direction,
                        'calibrated_ax': accel_x,
                        'calibrated_ay': accel_y,
                        'calibrated_az': accel_z,
                        'confidence': confidence
                    })
            
            socketio.start_background_task(emit_sensor_update)
            socketio.start_background_task(emit_orientation)
            
            # 👟 Broadcast step counter update if steps were detected in this batch
            if total_steps_batch > 0:
                print(f"� Broadcasting step update: {step_count_global} total steps")
                broadcast_step_counter_update(step_count_global, 0)
                
                # 📊 Update step statistics immediately after detection
                update_step_stats_immediate(device_id=data.get('device_id', 'ESP32_001'), steps=total_steps_batch)
        except Exception as e:
            print(f'Warning: SocketIO broadcast failed: {e}')
        
        return jsonify({'status': 'success', 'message': 'Data received and orientation computed'}), 200
    
    except Exception as e:
        print(f'❌ Sensor data error: {e}')
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/api/orientation-data', methods=['POST'])
def receive_orientation_data():
    """Receive calibrated orientation/direction data from ESP32"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data received'}), 400
        
        # Extract orientation data
        direction = data.get('direction', 'UNKNOWN')
        calibrated_ax = data.get('calibrated_ax', 0.0)
        calibrated_ay = data.get('calibrated_ay', 0.0) 
        calibrated_az = data.get('calibrated_az', 0.0)
        confidence = data.get('confidence', 0.0)
        device_id = data.get('device_id', 'ESP32_001')
        
        print(f'🧭 Direction: {direction} | CAL_AX: {calibrated_ax:.3f} CAL_AY: {calibrated_ay:.3f} CAL_AZ: {calibrated_az:.3f} | Conf: {confidence:.1f}%')
        
        # Store orientation data in database
        with db_lock:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    # Update latest sensor record with orientation data
                    cursor.execute('''
                        UPDATE sensor_readings 
                        SET device_orientation = ?, orientation_confidence = ?, 
                            calibrated_ax = ?, calibrated_ay = ?, calibrated_az = ?
                        WHERE id = (SELECT MAX(id) FROM sensor_readings WHERE device_id = ?)
                    ''', (direction, confidence, calibrated_ax, calibrated_ay, calibrated_az, device_id))
                    
                    if cursor.rowcount == 0:
                        # If no sensor record exists, create one with orientation data only
                        cursor.execute('''
                            INSERT INTO sensor_readings (device_id, device_orientation, orientation_confidence,
                                                        calibrated_ax, calibrated_ay, calibrated_az, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (device_id, direction, confidence, calibrated_ax, calibrated_ay, calibrated_az))
                    
                    conn.commit()
                    print(f"✅ Stored orientation data for {device_id}")
                
                except sqlite3.Error as e:
                    print(f"❌ Database error: {e}")
                    return jsonify({'status': 'error', 'message': 'Database storage failed'}), 500
                finally:
                    conn.close()
        
        # Broadcast orientation update to connected clients
        try:
            def emit_orientation_update():
                with app.app_context():
                    socketio.emit('orientation_update', {
                        'timestamp': datetime.now().isoformat(),
                        'device_id': device_id,
                        'direction': direction,
                        'calibrated_ax': calibrated_ax,
                        'calibrated_ay': calibrated_ay,
                        'calibrated_az': calibrated_az,
                        'confidence': confidence
                    })
            socketio.start_background_task(emit_orientation_update)
        except Exception as e:
            print(f'Warning: SocketIO orientation broadcast failed: {e}')
        
        return jsonify({'status': 'success', 'message': 'Orientation data received'}), 200
    
    except Exception as e:
        print(f'❌ Orientation data error: {e}')
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ❌ DISABLED: /api/camera-upload endpoint removed
# Only using /upload (binary) endpoint for image uploads


@app.route('/upload-audio', methods=['POST'])
def upload_audio_data():
    """Receive audio data from ESP32"""
    try:
        data = request.get_json()
        if not data or 'audio_data' not in data:
            return jsonify({'status': 'error', 'message': 'No audio data'}), 400
        
        audio_base64 = data['audio_data']
        
        # Decode from base64 to binary
        audio_binary = base64.b64decode(audio_base64)
        
        # 🔒 CRITICAL FIX: Use improved database connection
        with db_lock:
            conn = get_db_connection()
            if not conn:
                return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500
            cursor = conn.cursor()
            
            # Add audio_data column if it doesn't exist
            cursor.execute("PRAGMA table_info(sensor_readings)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'audio_data' not in columns:
                cursor.execute('ALTER TABLE sensor_readings ADD COLUMN audio_data BLOB')
            
            # Find the latest sensor record and update it with audio
            cursor.execute('''
                UPDATE sensor_readings 
                SET audio_data = ?
                WHERE id = (SELECT MAX(id) FROM sensor_readings WHERE accel_x IS NOT NULL)
            ''', (audio_binary,))
            
            if cursor.rowcount == 0:
                # If no sensor record exists, create one with audio only
                cursor.execute('''
                    INSERT INTO sensor_readings (audio_data)
                    VALUES (?)
                ''', (audio_binary,))
            
            conn.commit()
            conn.close()
        
        # Broadcast to connected clients (non-blocking, smaller payload)
        def emit_audio_update():
            with app.app_context():
                socketio.emit('audio_update', {
                    'audio_data': audio_base64[:100] + '...',  # Preview only
                    'size': len(audio_binary)
                })
        socketio.start_background_task(emit_audio_update)
        
        print(f'✅ Audio received and linked to latest sensor record: {len(audio_binary)} bytes ({len(audio_base64)} base64 chars)')
        return jsonify({'status': 'success', 'message': 'Audio uploaded'}), 200
    except Exception as e:
        print(f'❌ Error uploading audio: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_binary_image():
    """
    Receive binary image from ESP32, save it to the database as BLOB, and also save as a file.
    Serve images to frontend using /api/image/<id> (database BLOB).
    """
    try:
        image_data = request.get_data()
        if not image_data:
            return 'ERROR', 400
        # Save image as file (for backup/debug)
        import time
        uploads_dir = os.path.join(os.getcwd(), 'uploads', 'images')
        os.makedirs(uploads_dir, exist_ok=True)
        filename = f"esp32_{int(time.time())}.jpg"
        filepath = os.path.join(uploads_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)
        print(f"✅ Image also saved locally: {filename} ({len(image_data)} bytes)")
        # Save image binary to database (camera_image BLOB)
        with db_lock:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sensor_readings (device_id, camera_image, timestamp)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', ('ESP32_CAM', image_data))
                image_id = cursor.lastrowid
                conn.commit()
                conn.close()
        print(f"✅ Image saved to database (id={image_id}, {len(image_data)} bytes)")
        return jsonify({'status': 'success', 'image_id': image_id, 'image_url': f'/api/image/{image_id}'}), 200
    except Exception as e:
        print(f"❌ Error uploading image: {e}")
        return 'ERROR', 500


@app.route('/api/latest', methods=['GET'])
def get_latest_data():
    """Get latest sensor readings (excluding binary data for JSON compatibility)"""
    try:
        limit = request.args.get('limit', 20, type=int)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if audio_data column exists
        cursor.execute("PRAGMA table_info(sensor_readings)")
        columns = [column[1] for column in cursor.fetchall()]
        has_audio_column = 'audio_data' in columns
        
        # Select data based on available columns
        if has_audio_column:
            cursor.execute('''
                SELECT id, timestamp, accel_x, accel_y, accel_z, 
                       gyro_x, gyro_y, gyro_z, mic_level, sound_data, image_filename,
                       CASE WHEN camera_image IS NOT NULL THEN 1 ELSE 0 END as has_image,
                       CASE WHEN audio_data IS NOT NULL THEN 1 ELSE 0 END as has_audio
                FROM sensor_readings 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT id, timestamp, accel_x, accel_y, accel_z, 
                       gyro_x, gyro_y, gyro_z, mic_level, sound_data, image_filename,
                       CASE WHEN camera_image IS NOT NULL THEN 1 ELSE 0 END as has_image,
                       0 as has_audio
                FROM sensor_readings 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        records = []
        
        for row in rows:
            record = dict(row)
            # Convert timestamp to string for JSON compatibility
            if record.get('timestamp'):
                record['timestamp'] = str(record['timestamp'])
            records.append(record)
        
        conn.close()
        
        return jsonify({
            'success': True,
            'records': records,
            'count': len(records)
        }), 200
    except Exception as e:
        print(f'Error fetching data: {e}')
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/latest-image', methods=['GET'])
def get_latest_image():
    """Get the latest image filename, URL, and AI caption"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT image_filename, ai_caption FROM sensor_readings 
            WHERE image_filename IS NOT NULL 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            image_filename = result[0]
            ai_caption = result[1] if result[1] else "Waiting for AI analysis..."
            image_url = f'/uploads/images/{image_filename}'
            return jsonify({
                'success': True,
                'image_url': image_url,
                'filename': image_filename,
                'ai_caption': ai_caption
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'No images found'
            }), 404
    except Exception as e:
        print(f'Error fetching latest image: {e}')
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM sensor_readings')
        total_records = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(accel_x), AVG(accel_y), AVG(accel_z) FROM sensor_readings')
        accel_avg = cursor.fetchone()
        
        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM sensor_readings')
        date_range = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            'total_records': total_records,
            'accel_average': {
                'x': accel_avg[0],
                'y': accel_avg[1],
                'z': accel_avg[2]
            },
            'date_range': {
                'start': date_range[0],
                'end': date_range[1]
            }
        }), 200
    except Exception as e:
        print(f'Error fetching statistics: {e}')
        return jsonify({'error': str(e)}), 400

@app.route('/api/export', methods=['GET'])
def export_data():
    """Export sensor data as JSON (excludes binary data)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Export only JSON-serializable data
        cursor.execute('''
            SELECT id, timestamp, accel_x, accel_y, accel_z, 
                   gyro_x, gyro_y, gyro_z, mic_level, sound_data,
                   CASE WHEN camera_image IS NOT NULL THEN 1 ELSE 0 END as has_image,
                   CASE WHEN audio_data IS NOT NULL THEN 1 ELSE 0 END as has_audio
            FROM sensor_readings 
            ORDER BY timestamp
        ''')
        
        rows = cursor.fetchall()
        records = []
        
        for row in rows:
            record = dict(row)
            # Convert timestamp to string for JSON compatibility
            if record.get('timestamp'):
                record['timestamp'] = str(record['timestamp'])
            records.append(record)
        
        conn.close()
        
        response = app.response_class(
            response=json.dumps(records, indent=2),
            status=200,
            mimetype='application/json'
        )
        response.headers['Content-Disposition'] = f'attachment;filename=sensor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        return response
    except Exception as e:
        print(f'Error exporting data: {e}')
        return jsonify({'error': str(e)}), 400

@app.route('/api/clear', methods=['POST'])
def clear_database():
    """Clear all data from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sensor_readings')
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Database cleared successfully'}), 200
    except Exception as e:
        print(f'Error clearing database: {e}')
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': os.path.exists(DB_PATH)
    }), 200

@app.route('/api/events', methods=['GET'])
def get_important_events():
    """Get important events for ESP32 device"""
    try:
        device_id = request.args.get('device_id', 'ESP32_001')
        
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get unsent important events for this device
            cursor.execute('''
                SELECT id, event_type, message, created_at
                FROM important_events 
                WHERE device_id = ? AND is_sent = 0
                ORDER BY created_at DESC
                LIMIT 10
            ''', (device_id,))
            
            events = cursor.fetchall()
            conn.close()
        
        if events:
            events_list = []
            for event in events:
                event_id, event_type, message, created_at = event
                events_list.append({
                    "id": event_id,
                    "event_type": event_type, 
                    "message": message,
                    "created_at": created_at,
                    "device_id": device_id
                })
            
            return jsonify({
                "status": "success",
                "events": events_list,
                "count": len(events_list),
                "message": f"Found {len(events_list)} important event(s)"
            }), 200
        else:
            return jsonify({
                "status": "success", 
                "events": [],
                "count": 0,
                "message": "No new important events"
            }), 200
            
    except Exception as e:
        print(f'❌ Error getting events: {e}')
        return jsonify({
            "status": "error",
            "message": "Failed to fetch events",
            "error": str(e)
        }), 500

@app.route('/api/device/event/received', methods=['POST'])
def mark_event_received():
    """Mark event as received by ESP32"""
    try:
        data = request.get_json()
        if not data or 'event_id' not in data:
            return jsonify({
                "status": "error",
                "message": "event_id is required"
            }), 400
            
        event_id = data['event_id']
        device_id = data.get('device_id', 'ESP32_001')
        
        with db_lock:
            conn = get_db_connection()
            if not conn:
                return jsonify({
                    "status": "error",
                    "message": "Database connection failed"
                }), 500
            cursor = conn.cursor()
            
            # Mark event as sent/received
            cursor.execute('''
                UPDATE important_events 
                SET is_sent = 1
                WHERE id = ? AND device_id = ?
            ''', (event_id, device_id))
            
            if cursor.rowcount > 0:
                conn.commit()
                print(f'✅ Event {event_id} marked as received by {device_id}')
                result = {
                    "status": "success",
                    "message": f"Event {event_id} marked as received",
                    "event_id": event_id
                }
            else:
                result = {
                    "status": "error", 
                    "message": "Event not found or already processed",
                    "event_id": event_id
                }
            
            conn.close()
            return jsonify(result), 200 if result["status"] == "success" else 404
            
    except Exception as e:
        print(f'❌ Error marking event received: {e}')
        return jsonify({
            "status": "error",
            "message": "Failed to update event status",
            "error": str(e)
        }), 500

# ==================== Database Functions ====================

def update_step_stats_immediate(device_id, steps):
    """Immediately update step statistics when steps are detected"""
    try:
        with db_lock:
            conn = get_db_connection()
            if not conn:
                return
            
            cursor = conn.cursor()
            today = datetime.now().date()
            
            # Check if today's stats exist
            cursor.execute('''
                SELECT total_steps, peak_steps FROM step_statistics
                WHERE device_id = ? AND date_recorded = ?
            ''', (device_id, today))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing stats
                current_total, current_peak = existing
                new_total = (current_total or 0) + steps
                new_peak = max(current_peak or 0, steps)
                
                cursor.execute('''
                    UPDATE step_statistics
                    SET total_steps = ?, peak_steps = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE device_id = ? AND date_recorded = ?
                ''', (new_total, new_peak, device_id, today))
            else:
                # Create new stats record
                cursor.execute('''
                    INSERT INTO step_statistics 
                    (device_id, date_recorded, total_steps, peak_steps, activity_level)
                    VALUES (?, ?, ?, ?, ?)
                ''', (device_id, today, steps, steps, 'LOW'))
            
            conn.commit()
            conn.close()
    
    except Exception as e:
        print(f"❌ Error updating immediate stats: {e}")

def store_sensor_data(data):
    """Store sensor data with thread-safe database access, orientation computation, step counting, and event detection"""
    with db_lock:
        conn = get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Store sensor data including orientation and steps (now computed on server)
            cursor.execute('''
                INSERT INTO sensor_readings 
                (device_id, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mic_level,
                 device_orientation, orientation_confidence, calibrated_ax, calibrated_ay, calibrated_az, step_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('device_id', 'ESP32_001'),
                data.get('accel_x', 0),
                data.get('accel_y', 0),
                data.get('accel_z', 0),
                data.get('gyro_x', 0),
                data.get('gyro_y', 0),
                data.get('gyro_z', 0),
                data.get('mic_level', 0),
                data.get('device_orientation', 'UNKNOWN'),
                data.get('orientation_confidence', 0),
                data.get('calibrated_ax', 0),
                data.get('calibrated_ay', 0),
                data.get('calibrated_az', 0),
                data.get('step_count', 0)
            ))
            
            # 🚨 EVENT DETECTION LOGIC
            device_id = data.get('device_id', 'ESP32_001')
            
            # Check for high sound event (mic_level > 80)
            mic_level = data.get('mic_level', 0)
            if mic_level > 80:
                cursor.execute('''
                    INSERT INTO important_events (device_id, event_type, message, is_sent)
                    VALUES (?, ?, ?, ?)
                ''', (device_id, 'high_sound', f'High sound detected: {mic_level:.1f} dB', 0))
                print(f'🚨 HIGH SOUND EVENT: {mic_level:.1f} dB from {device_id}')
            
            # Check for sudden motion (high acceleration change)
            accel_x = data.get('accel_x', 0)
            accel_y = data.get('accel_y', 0) 
            accel_z = data.get('accel_z', 0)
            
            if accel_x and accel_y and accel_z:
                total_accel = (accel_x**2 + accel_y**2 + accel_z**2)**0.5
                
                # Get previous acceleration for comparison
                cursor.execute('''
                    SELECT accel_x, accel_y, accel_z FROM sensor_readings 
                    WHERE device_id = ? AND accel_x IS NOT NULL 
                    ORDER BY id DESC LIMIT 1 OFFSET 1
                ''', (device_id,))
                prev_reading = cursor.fetchone()
                
                if prev_reading:
                    prev_x, prev_y, prev_z = prev_reading
                    prev_total = (prev_x**2 + prev_y**2 + prev_z**2)**0.5
                    accel_change = abs(total_accel - prev_total)
                    
                    # Sudden motion detected (change > 5 m/s²)
                    if accel_change > 5.0:
                        cursor.execute('''
                            INSERT INTO important_events (device_id, event_type, message, is_sent)
                            VALUES (?, ?, ?, ?)
                        ''', (device_id, 'sudden_motion', f'Sudden motion detected: {accel_change:.2f} m/s² change', 0))
                        print(f'🚨 MOTION EVENT: {accel_change:.2f} m/s² change from {device_id}')
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            print(f'❌ Database error storing sensor data: {e}')
            return False
        except Exception as e:
            print(f'❌ Error storing sensor data: {e}')
            return False
        finally:
            conn.close()

# ==================== OLED DISPLAY ANIMATION CONTROL ====================

@app.route('/api/oled-display/get', methods=['GET'])
def get_oled_display():
    """ESP32 polls this endpoint to get what animation to display on OLED
    
    Reads from database table oled_display_state.
    Generic endpoint that can return any animation type for future extensibility.
    """
    try:
        device_id = request.args.get('device_id', 'ESP32_001')
        
        with db_lock:
            conn = get_db_connection()
            if not conn:
                return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500
            
            cursor = conn.cursor()
            cursor.execute('''
                SELECT animation_type, animation_id, animation_name, updated_at
                FROM oled_display_state
                WHERE device_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            ''', (device_id,))
            
            result = cursor.fetchone()
            conn.close()
        
        if result:
            animation_type, animation_id, animation_name, updated_at = result
            print(f'📡 OLED state retrieved from DB: {animation_name} (ID: {animation_id})')
            return jsonify({
                'status': 'success',
                'animation_type': animation_type,
                'animation_id': animation_id,
                'animation_name': animation_name,
                'updated_at': updated_at,
                'message': f'Current OLED animation: {animation_name}'
            }), 200
        else:
            # Fallback if no record exists
            return jsonify({
                'status': 'success',
                'animation_type': 'pet',
                'animation_id': 1,
                'animation_name': 'CHILD',
                'message': 'Current OLED animation: CHILD (default)'
            }), 200
    
    except Exception as e:
        print(f'❌ Error getting OLED display: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/oled-display/set', methods=['POST'])
def set_oled_display():
    """Web UI sends POST request to set OLED display animation
    
    Updates oled_display_state table in database.
    Generic endpoint that can accept any animation type for future extensibility.
    """
    try:
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        animation_id = data.get('animation_id')
        animation_type = data.get('animation_type', 'pet')  # Default to pet type
        device_id = data.get('device_id', 'ESP32_001')
        
        # Validate animation_id value
        if animation_id not in [0, 1, 2, 3]:
            return jsonify({'status': 'error', 'message': 'Invalid animation_id. Must be 0-3'}), 400
        
        animation_map = {
            0: "INFANT",
            1: "CHILD", 
            2: "ADULT",
            3: "OLD"
        }
        
        animation_name = animation_map[animation_id]
        
        # Update database with new state
        with db_lock:
            conn = get_db_connection()
            if not conn:
                return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500
            
            try:
                cursor = conn.cursor()
                
                # Update or insert OLED state
                cursor.execute('''
                    UPDATE oled_display_state
                    SET animation_type = ?, animation_id = ?, animation_name = ?, 
                        updated_at = CURRENT_TIMESTAMP, updated_by = 'web_ui'
                    WHERE device_id = ?
                ''', (animation_type, animation_id, animation_name, device_id))
                
                if cursor.rowcount == 0:
                    # Insert if not exists
                    cursor.execute('''
                        INSERT INTO oled_display_state
                        (device_id, animation_type, animation_id, animation_name, updated_by)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (device_id, animation_type, animation_id, animation_name, 'web_ui'))
                
                conn.commit()
                print(f'✅ OLED state updated in database: {animation_id} ({animation_name})')
                print(f'   Device: {device_id} | Type: {animation_type}')
            except sqlite3.Error as e:
                print(f'❌ Database error: {e}')
                return jsonify({'status': 'error', 'message': 'Database update failed'}), 500
            finally:
                conn.close()
        
        # Broadcast animation change to all connected web clients (real-time)
        def emit_oled_change():
            with app.app_context():
                socketio.emit('oled_display_changed', {
                    'animation_type': animation_type,
                    'animation_id': animation_id,
                    'animation_name': animation_name,
                    'device_id': device_id,
                    'timestamp': datetime.now().isoformat()
                })
        
        socketio.start_background_task(emit_oled_change)
        
        return jsonify({
            'status': 'success',
            'animation_type': animation_type,
            'animation_id': animation_id,
            'animation_name': animation_name,
            'device_id': device_id,
            'message': f'OLED display set to: {animation_name}'
        }), 200
        
    except Exception as e:
        print(f'❌ Error setting OLED display: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ================= STEP COUNTER ENDPOINTS =================

@app.route('/api/step-counter/get', methods=['GET'])
def get_step_counter():
    """Get current step counter from server
    
    Returns total steps detected by server-side accelerometer analysis
    """
    try:
        device_id = request.args.get('device_id', 'ESP32_001')
        
        # Get total steps from global counter
        total_steps = step_count_global
        
        # Optional: Get daily steps from database
        with db_lock:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT SUM(step_count) as daily_steps
                    FROM sensor_readings
                    WHERE device_id = ? AND DATE(timestamp) = DATE('now')
                ''', (device_id,))
                result = cursor.fetchone()
                daily_steps = result[0] if result and result[0] else 0
                conn.close()
            else:
                daily_steps = 0
        
        return jsonify({
            'status': 'success',
            'device_id': device_id,
            'total_steps': total_steps,
            'daily_steps': daily_steps or 0,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f'❌ Error getting step counter: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/step-counter/reset', methods=['POST'])
def reset_step_counter():
    """Reset step counter
    
    Resets the global step counter to 0 (fresh session)
    """
    try:
        global step_count_global
        device_id = request.args.get('device_id', 'ESP32_001')
        old_count = step_count_global
        
        # Reset counter
        step_count_global = 0
        
        # Clear the acceleration history for clean slate
        accel_history.clear()
        
        print(f'🔄 Step counter reset: {old_count} → 0')
        
        # Broadcast reset to all clients
        broadcast_step_counter_update(0, 0)
        
        return jsonify({
            'status': 'success',
            'device_id': device_id,
            'reset_from': old_count,
            'new_count': 0,
            'message': 'Step counter reset to 0',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f'❌ Error resetting step counter: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/step-counter/stats', methods=['GET'])
def get_step_stats():
    """Get detailed step counter statistics with daily aggregation and trends
    
    Returns:
    - Daily statistics (total, peak, activity level)
    - Recent batch-level details
    - Comparison with previous data
    - Activity trends
    """
    try:
        device_id = request.args.get('device_id', 'ESP32_001')
        days = request.args.get('days', 7, type=int)  # Last N days
        
        with db_lock:
            conn = get_db_connection()
            if not conn:
                return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500
            
            try:
                cursor = conn.cursor()
                
                # Get daily aggregated statistics
                cursor.execute('''
                    SELECT 
                        date_recorded,
                        total_steps,
                        peak_steps,
                        avg_step_interval,
                        activity_level,
                        updated_at
                    FROM step_statistics
                    WHERE device_id = ? AND date_recorded >= DATE('now', '-' || ? || ' days')
                    ORDER BY date_recorded DESC
                ''', (device_id, days))
                
                daily_stats = [{
                    'date': str(row[0]),
                    'total_steps': row[1],
                    'peak_steps': row[2],
                    'avg_step_interval': round(row[3], 2),
                    'activity_level': row[4],
                    'updated_at': str(row[5])
                } for row in cursor.fetchall()]
                
                # Get today's detailed batch data
                today = datetime.now().date()
                cursor.execute('''
                    SELECT 
                        timestamp,
                        step_count,
                        accel_x, accel_y, accel_z,
                        SUM(step_count) OVER (ORDER BY timestamp) as cumulative_steps
                    FROM sensor_readings
                    WHERE device_id = ? AND DATE(timestamp) = ?
                    ORDER BY timestamp DESC
                    LIMIT 20
                ''', (device_id, today))
                
                batch_details = [{
                    'timestamp': str(row[0]),
                    'steps_in_batch': row[1],
                    'accel': [round(row[2], 3), round(row[3], 3), round(row[4], 3)],
                    'cumulative': row[5]
                } for row in cursor.fetchall()]
                
                # Calculate trends
                cursor.execute('''
                    SELECT 
                        total_steps,
                        activity_level
                    FROM step_statistics
                    WHERE device_id = ? AND date_recorded >= DATE('now', '-7 days')
                    ORDER BY date_recorded ASC
                ''', (device_id,))
                
                weekly_data = cursor.fetchall()
                trend = None
                if len(weekly_data) >= 2:
                    last_week = sum([row[0] or 0 for row in weekly_data])
                    
                    # Compare with previous week
                    cursor.execute('''
                        SELECT SUM(total_steps)
                        FROM step_statistics
                        WHERE device_id = ? 
                        AND date_recorded >= DATE('now', '-14 days')
                        AND date_recorded < DATE('now', '-7 days')
                    ''', (device_id,))
                    
                    prev_week_result = cursor.fetchone()
                    prev_week = prev_week_result[0] or 0
                    
                    if prev_week > 0:
                        trend_percent = ((last_week - prev_week) / prev_week) * 100
                        trend = {
                            'last_week': last_week,
                            'previous_week': prev_week,
                            'change_percent': round(trend_percent, 1),
                            'direction': 'up' if trend_percent > 0 else 'down' if trend_percent < 0 else 'stable'
                        }
                
                return jsonify({
                    'status': 'success',
                    'device_id': device_id,
                    'current_total': step_count_global,
                    'today': str(today),
                    'daily_statistics': daily_stats,
                    'today_details': batch_details,
                    'trend': trend,
                    'summary': {
                        'total_days_tracked': len(daily_stats),
                        'avg_daily_steps': round(sum([s['total_steps'] for s in daily_stats]) / len(daily_stats), 1) if daily_stats else 0,
                        'max_daily_steps': max([s['total_steps'] for s in daily_stats]) if daily_stats else 0,
                        'total_batches_today': len(batch_details)
                    },
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except sqlite3.Error as e:
                print(f'❌ Database error: {e}')
                return jsonify({'status': 'error', 'message': 'Database query failed'}), 500
            finally:
                conn.close()
    
    except Exception as e:
        print(f'❌ Error getting step stats: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f'Internal server error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(400)
def bad_request(error):
    print(f'Bad request: {error}')
    return jsonify({'error': 'Bad request'}), 400

# Handle WebSocket errors gracefully
@socketio.on_error()
def error_handler(e):
    print(f'SocketIO error: {e}')

# Handle connection errors
@socketio.on_error_default
def default_error_handler(e):
    print(f'SocketIO default error: {e}')

# ==================== Main ====================

if __name__ == '__main__':
    print('🚀 Starting ESP32 Dashboard Server...')
    print('📊 Dashboard: http://192.168.61.252:5000')  # Updated IP
    print('🔌 WebSocket: ws://192.168.61.252:5000/socket.io/')
    print('📡 Endpoints:')
    print('   • POST /api/sensor-data (JSON, ~146 bytes)')
    print('   • POST /upload (Binary, ~1-3KB)')  
    print('   • POST /upload-audio (JSON, ~32KB+)')
    print('')
    
    # Initialize database before starting server
    if not init_database():
        print('❌ Database initialization failed. Exiting.')
        exit(1)
    
    try:
        # Run the app with stability-focused configuration
        socketio.run(app, 
            host='0.0.0.0', 
            port=5000, 
            debug=False,  # Disable debug to prevent reloading
            allow_unsafe_werkzeug=True,
            use_reloader=False,  # Prevent duplicate processes
            log_output=False  # Reduce logging overhead
        )
    except KeyboardInterrupt:
        print('\n🛑 Server stopped by user')
    except Exception as e:
        print(f'❌ Server error: {e}')
        print('💡 Check if port 5000 is available and try restarting')
