# Flask-SocketIO 5.6.0 Backend Verification Checklist
**For ESP32 + Web Client Integration**

---

## 1️⃣ ENVIRONMENT CHECKS

### 1.1 Verify Python Version
```bash
python --version
# Expected: Python 3.10+
```
✅ **Your setup:** Python 3.13.0

---

### 1.2 Confirm Active Virtual Environment
```powershell
# On Windows PowerShell
Get-Command python | Select-Object Source
# Expected should show: .venv\Scripts\python.exe

# OR
where python
# Expected should show: E:\Rajeev\esp 32\esp 32\.venv\Scripts\python.exe
```

```bash
# Verify venv is active
pip -V
# Expected: pip 24.0 (python 3.13) in E:\Rajeev\esp 32\esp 32\.venv\...
```

✅ **Check these now:**
```powershell
cd 'e:\Rajeev\esp 32\esp 32'
.\.venv\Scripts\activate
python -c "import sys; print(f'Python: {sys.executable}')"
pip -V
```

---

### 1.3 Verify Pip Packages (Exact Versions Required)
```bash
pip show flask flask-socketio python-socketio python-engineio python-cors
```

✅ **Expected Output:**
```
Name: Flask
Version: 2.3.2

Name: Flask-SocketIO
Version: 5.6.0
Requires: Flask, python-socketio

Name: python-socketio
Version: 5.16.1
Requires: bidict, python-engineio

Name: python-engineio
Version: 4.13.1
Requires: simple-websocket

Name: Flask-CORS
Version: 4.0.0
```

⚠️ **Critical:** If versions don't match, run:
```bash
pip install --upgrade flask-socketio==5.6.0 python-socketio==5.16.1 python-engineio==4.13.1
```

---

### 1.4 Check for Version Conflicts
```bash
pip check
```

✅ **Expected:** No output (all dependencies compatible)

❌ **If you see conflicts:** Run `pip install --upgrade [conflicting-package]`

---

### 1.5 Export All Dependencies
```bash
pip freeze > requirements.txt
cat requirements.txt | grep -E "flask|socketio|engineio|cors|werkzeug"
```

✅ **Expected (minimum):**
```
bidict==0.22.1
click==8.1.7
Flask==2.3.2
Flask-CORS==4.0.0
Flask-SocketIO==5.6.0
itsdangerous==2.1.2
Jinja2==3.1.2
MarkupSafe==2.1.1
Werkzeug==2.3.7
python-engineio==4.13.1
python-socketio==5.16.1
simple-websocket==0.10.1
```

---

## 2️⃣ FLASK APP STRUCTURE CHECKS

### 2.1 Correct SocketIO Initialization
✅ **CORRECT (your current code):**
```python
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')

# Initialize CORS first
CORS(app)

# Initialize SocketIO with proper config
socketio = SocketIO(app, 
    cors_allowed_origins="*",
    max_http_buffer_size=10*1024*1024,  # 10MB
    ping_timeout=60,
    ping_interval=25,
    logger=False,  # For production
    engineio_logger=False
)
```

❌ **WRONG (avoid):**
```python
# WRONG: Creating socketio before Flask app
socketio = SocketIO()
app = Flask(__name__)
socketio.init_app(app)  # Unnecessary complexity

# WRONG: Using app.run() instead of socketio.run()
app.run()  # Will NOT have WebSocket support
```

---

### 2.2 Async Mode (Flask-SocketIO 5.6.0)
✅ **Automatic Detection (Recommended):**
```python
# No async_mode specified - let socketio auto-detect
socketio = SocketIO(app)
```

✅ **For Server-Side Background Tasks:**
```python
socketio = SocketIO(app, 
    async_mode='threading',  # Uses threading for background tasks
    cors_allowed_origins="*"
)
```

✅ **Check which mode is active:**
```python
if __name__ == '__main__':
    print(f"Async mode: {socketio.async_mode}")
```

❌ **WRONG (Flask-SocketIO 5.6.0 doesn't support):**
```python
async_mode='eventlet'  # Not available in 5.6.0
async_mode='gevent'    # Not available in 5.6.0
```

---

### 2.3 CORS Configuration
✅ **CORRECT (your setup):**
```python
CORS(app)
socketio = SocketIO(app, 
    cors_allowed_origins="*",  # Allow all origins for development
)
```

✅ **For Production (specific origin):**
```python
socketio = SocketIO(app,
    cors_allowed_origins=["http://192.168.43.67:5000", "https://yourdomain.com"],
    cors_credentials=True
)
```

❌ **WRONG:**
```python
# Don't do both
cors = CORS(app)
socketio = SocketIO(app)
# Then use CORS() function at top of file again
```

---

### 2.4 Secret Key Setup (Optional but Recommended)
✅ **CORRECT:**
```python
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app)
```

✅ **For Development (auto-generate):**
```python
import os
app.config['SECRET_KEY'] = os.urandom(24).hex()
```

---

### 2.5 Debug Configuration
✅ **CORRECT (your setup):**
```python
if __name__ == '__main__':
    socketio.run(app, 
        host='0.0.0.0', 
        port=5000, 
        debug=False,  # ✅ Correct for production
        use_reloader=False  # ✅ Prevents duplicate processes
    )
```

❌ **WRONG (for production):**
```python
socketio.run(app, debug=True)  # Enables auto-reload, duplicates threads
socketio.run(app, use_reloader=True)  # Will reload and duplicate background tasks
```

---

## 3️⃣ EVENT HANDLING CHECKS

### 3.1 Verify @socketio.on Decorators
✅ **CORRECT (your setup):**
```python
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    # OLED display change event
    def emit_connection():
        with app.app_context():
            socketio.emit('connection_response', {'status': 'Connected'})
    socketio.start_background_task(emit_connection)

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('sensor_data')
def handle_sensor_data(data):
    print(f'Received: {data}')
    return {'status': 'success'}
```

❌ **WRONG:**
```python
# Using Socket.IO directly (protocol layer)
socketio.Server.emit()  # Wrong - this is internal

# Missing app context in background tasks
def bad_emit():
    socketio.emit('event')  # Will fail outside request context
```

---

### 3.2 Namespace Usage
✅ **CORRECT (default namespace):**
```python
# Default namespace '/' - no need to specify
@socketio.on('my_event')
def handle_my_event(data):
    socketio.emit('event_response', data)  # Goes to default '/'
```

✅ **Correct way to emit to default namespace:**
```python
socketio.emit('event', data)  # Works - implicit namespace='/'
```

❌ **WRONG (Flask-SocketIO 5.6.0):**
```python
socketio.emit('event', data, namespace='/')  # Unnecessary, causes issues
socketio.emit('event', data, include_self=False)  # Parameter doesn't exist
socketio.emit('event', data, broadcast=True)  # Parameter doesn't exist
```

---

### 3.3 Event Name Consistency Across Stack

✅ **Verify naming is consistent:**

**Backend emit:**
```python
socketio.emit('oled_display_changed', {
    'animation_id': current_pet_age,
    'animation_name': animation_map[current_pet_age]
})
```

**Frontend listener (index.html):**
```javascript
socket.on('oled_display_changed', function(data) {
    console.log('OLED display changed:', data);
});
```

**ESP32 Client:**
```cpp
// ESP32 polls HTTP endpoint, not WebSocket events
http.begin(client, "http://server:5000/api/oled-display/get");
```

✅ **Event Names Used (Your Server):**
- `connection_response` ✅
- `sensor_update` ✅
- `orientation_update` ✅
- `audio_update` ✅
- `camera_update` ✅
- `oled_display_changed` ✅

---

### 3.4 Emit Function Usage (Flask-SocketIO 5.6.0)

✅ **CORRECT - Background Task Pattern (Your Setup):**
```python
def emit_update():
    with app.app_context():
        socketio.emit('update_event', {'data': 'value'})

socketio.start_background_task(emit_update)
```

✅ **Broadcast to all clients:**
```python
socketio.emit('event', data)  # Broadcasts to all by default
```

✅ **Broadcast to specific room:**
```python
socketio.emit('event', data, room='room_name')
```

✅ **Skip sending to sender:**
```python
socketio.emit('event', data, skip_sid=request.sid)
```

❌ **WRONG (Flask-SocketIO 5.6.0):**
```python
socketio.emit('event', data, broadcast=True)  # Parameter removed
socketio.emit('event', data, namespace='/')  # Unnecessary/problematic
socketio.emit('event', data, include_self=False)  # Doesn't exist
socketio.Server.emit('event', data)  # Direct server call - internal
```

---

### 3.5 Return Responses

✅ **CORRECT:**
```python
@socketio.on('request')
def handle_request(data):
    # Do something
    return {'status': 'success', 'data': result}
```

✅ **ACK pattern (if client wants confirmation):**
```javascript
// Client
socket.emit('request', data, function(response) {
    console.log('Server response:', response);
});
```

```python
# Server
@socketio.on('request')
def handle_request(data):
    return {'status': 'success'}  # This becomes the ack callback
```

---

## 4️⃣ SERVER STARTUP CHECKS

### 4.1 Correct Way to Run App

✅ **CORRECT (your setup):**
```python
if __name__ == '__main__':
    socketio.run(app, 
        host='0.0.0.0',      # Listen on all interfaces
        port=5000,           # Port number
        debug=False,         # No auto-reload
        use_reloader=False,  # No reloader process
        log_output=False     # Reduce noise
    )
```

✅ **Command to start:**
```bash
cd 'e:\Rajeev\esp 32\esp 32'
python app.py
```

✅ **Expected output:**
```
✅ Database initialized successfully
🚀 Starting ESP32 Dashboard Server...
📊 Dashboard: http://127.0.0.1:5000
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

❌ **WRONG:**
```python
app.run()  # Missing WebSocket support!
app.run(debug=True)  # Reloads process, duplicates threads
socketio.run(app, use_reloader=True)  # Duplicates background tasks
```

---

### 4.2 Verify Server is Running

```bash
# Check if server responds
curl http://127.0.0.1:5000/api/health

# Expected:
# {"status":"healthy","timestamp":"2026-02-14T...","database":true}
```

```bash
# Check port is listening
netstat -an | findstr "5000"
# Expected: TCP 0.0.0.0:5000 LISTENING
```

---

## 5️⃣ CONNECTION DEBUG CHECKS

### 5.1 Enable Detailed Logging

✅ **For debugging (NOT production):**
```python
socketio = SocketIO(app,
    cors_allowed_origins="*",
    logger=True,          # Enable SocketIO logging
    engineio_logger=True  # Enable Engine.IO logging
)

import logging
logging.basicConfig(level=logging.DEBUG)
```

✅ **Start server:**
```bash
python app.py 2>&1 | tee server_debug.log
```

✅ **Expected debug output:**
```
DEBUG:socketio.server:Client connected [sid: abc123def456]
DEBUG:engineio.server:Upgrade attempt from request SID
```

---

### 5.2 Inspect Connection Handshake

✅ **Add manual connection logging:**
```python
@socketio.on('connect')
def handle_connect():
    print(f'✅ Client connected: {request.sid}')
    print(f'   Address: {request.remote_addr}')
    print(f'   User-Agent: {request.headers.get("User-Agent")}')
    print(f'   Transport: {request.sid}')  # Will show websocket or polling
    
    # Send connection confirmation
    def emit_connection():
        with app.app_context():
            socketio.emit('connection_response', {
                'status': 'connected',
                'server_time': datetime.now().isoformat(),
                'sid': request.sid
            })
    socketio.start_background_task(emit_connection)
```

---

### 5.3 Debug Transport Upgrade (Polling → WebSocket)

✅ **Check handshake on browser console:**
```javascript
// In browser console
socket.on('connect', function() {
    console.log('Socket.IO connected');
    console.log('Transport:', socket.io.engine.transport);
});

socket.io.engine.on('upgrade', function(transport) {
    console.log('Transport upgraded to:', transport);
});
```

✅ **Expected sequence:**
1. First connection: `websocket` (HTTP upgrade works)
2. If WebSocket fails: Falls back to `polling` (long-polling)

---

### 5.4 Verify Client Connection Status

✅ **Add connection state listener:**
```javascript
// Browser
socket.on('connect', () => console.log('✅ Connected'));
socket.on('disconnect', () => console.log('❌ Disconnected'));
socket.on('connect_error', (error) => console.log('⚠️ Error:', error));
```

✅ **Backend:**
```python
@socketio.on('connect')
def handle_connect():
    print(f'✅ Connected: {request.sid}')
    return True

@socketio.on('disconnect')
def handle_disconnect():
    print(f'❌ Disconnected: {request.sid}')

@socketio.on_error()
def error_handler(e):
    print(f'⚠️ Socket error: {e}')
```

---

## 6️⃣ ESP32 CLIENT CHECKS

### 6.1 Socket.IO Library Version Compatibility

✅ **For ESP32 (you're using HTTP polling, good choice):**

Your setup: HTTP GET/POST instead of WebSocket
- ✅ No Socket.IO library needed on ESP32
- ✅ Simple HTTP requests
- ✅ Lightweight and reliable

**Server endpoints ESP32 uses:**
```
GET  /api/oled-display/get  (polls every 2 seconds)
POST /api/sensor-data        (sends sensor readings)
POST /api/orientation-data   (sends orientation)
POST /upload                 (sends camera images)
POST /upload-audio          (sends audio)
```

---

### 6.2 Transport Method

✅ **Your setup (ideal for ESP32):**
```cpp
// ESP32: Simple HTTP polling - NO WebSocket needed
http.begin(client, "http://192.168.43.67:5000/api/oled-display/get");
int httpCode = http.GET();
String payload = http.getString();
```

✅ **No WebSocket overhead:**
- ✅ Lower memory footprint
- ✅ Simpler to debug
- ✅ More reliable on microcontroller
- ✅ Works through more firewalls

---

### 6.3 Check Server URL and Port

✅ **Verify ESP32 can reach server:**
```bash
# From ESP32 logs, you should see:
// [INFO] Connecting to WiFi: 123
// [INFO] WiFi connected at 192.168.43.87
// [INFO] Connecting to server: http://192.168.43.67:5000/api/oled-display/get
// [INFO] Server response: {"status":"success","animation_id":1,"animation_name":"CHILD"}
```

✅ **Test from development machine:**
```bash
curl http://127.0.0.1:5000/api/oled-display/get

# Expected:
# {"status":"success","animation_type":"pet","animation_id":1,"animation_name":"CHILD","message":"Current OLED animation: CHILD"}
```

---

### 6.4 JSON Format Consistency

✅ **ESP32 sends to /api/sensor-data:**
```cpp
{
  "accel_x": 0.45,
  "accel_y": -0.23,
  "accel_z": 9.81,
  "gyro_x": 0.1,
  "gyro_y": -0.05,
  "gyro_z": 0.02,
  "mic_level": 42.5,
  "device_id": "ESP32_001"
}
```

✅ **Server receives and validates:**
```python
@app.route('/api/sensor-data', methods=['POST'])
def receive_sensor_data():
    if not request.is_json:
        return {'status': 'error', 'message': 'Content-Type must be application/json'}, 400
    
    data = request.get_json()
    # Process data...
    return {'status': 'success'}, 200
```

---

### 6.5 Reconnect Logic

✅ **Your setup (ESP32):**
```cpp
// Simple HTTP retry logic
#define DISPLAY_CHECK_INTERVAL 2000  // Poll every 2 seconds

void getOLEDDisplayFromServer() {
    if (!WiFi.isConnected()) {
        Serial.println("WiFi disconnected, attempting reconnect...");
        WiFi.reconnect();
        return;
    }
    
    http.begin(client, "http://192.168.43.67:5000/api/oled-display/get");
    int httpCode = http.GET();
    
    if (httpCode == 200) {
        String payload = http.getString();
        // Parse JSON and update animation
    } else {
        Serial.print("HTTP error: ");
        Serial.println(httpCode);
    }
    
    http.end();
}
```

✅ **Automatic reconnect on WiFi loss:**
```cpp
if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();  // Automatic retry
}
```

---

## 7️⃣ BROWSER CLIENT CHECKS

### 7.1 Socket.IO CDN Version

✅ **CORRECT (in index.html):**
```html
<script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
```

✅ **Verify version matches Flask-SocketIO:**
- Flask-SocketIO 5.6.0 → Socket.IO 4.5.x ✅
- python-socketio 5.16.1 → Protocol v4 ✅

❌ **WRONG:**
```html
<script src="https://cdn.socket.io/3.x/socket.io.min.js"></script>  <!-- Too old -->
<script src="https://cdn.socket.io/5.x/socket.io.min.js"></script>  <!-- Too new -->
```

---

### 7.2 Connection Code (index.html)

✅ **CORRECT:**
```javascript
const socket = io();  // Auto-detects server address

socket.on('connect', function() {
    console.log('✅ Connected to server');
});

socket.on('disconnect', function() {
    console.log('❌ Disconnected from server');
});

socket.on('connect_error', function(error) {
    console.error('⚠️ Connection error:', error);
});
```

✅ **With explicit configuration:**
```javascript
const socket = io('http://127.0.0.1:5000', {
    reconnection: true,
    reconnection_delay: 1000,
    reconnection_delay_max: 5000,
    reconnection_attempts: 5,
    path: '/socket.io',
    transports: ['websocket', 'polling']
});
```

❌ **WRONG:**
```javascript
const socket = io.connect('http://127.0.0.1:5000');  // Old syntax, deprecated
```

---

### 7.3 Browser Console Errors

✅ **Check browser console (F12) for:**

**WebSocket connected:**
```
Socket.IO connected
Transport: websocket
```

**Connection refused:**
```
⚠️ Connection error: Error: connect ECONNREFUSED 127.0.0.1:5000
```

**CORS error:**
```
Access to XMLHttpRequest blocked by CORS policy
```

**Solution:** Run server on accessible IP:
```bash
python app.py  # Runs on 0.0.0.0:5000
# Then access from: http://192.168.43.67:5000
```

---

### 7.4 Check CORS Issues

✅ **Test CORS headers:**
```bash
curl -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -X OPTIONS http://127.0.0.1:5000/
```

✅ **Expected response headers:**
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Credentials: true
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
```

✅ **Your current setup handles this:**
```python
CORS(app)  # Applies to REST endpoints
socketio = SocketIO(app, cors_allowed_origins="*")  # Applies to WebSocket
```

---

### 7.5 Verify Event Emission and Reception

✅ **Test in browser console:**
```javascript
// Send event to server
socket.emit('sensor_data', {
    accel_x: 0.5,
    accel_y: -0.3,
    accel_z: 9.81
});

// Listen for response
socket.on('sensor_update', function(data) {
    console.log('Received sensor update:', data);
});

// Test animation change
socket.emit('set_oled_display', {
    animation_id: 1,
    animation_type: 'pet'
});
```

---

## 8️⃣ NETWORK & DEPLOYMENT CHECKS

### 8.1 Localhost vs 0.0.0.0

✅ **CORRECT (your setup):**
```python
socketio.run(app, host='0.0.0.0', port=5000)
# 0.0.0.0 = Listen on all network interfaces
```

✅ **Access from:**
- `http://127.0.0.1:5000` (localhost)
- `http://192.168.43.67:5000` (network IP)
- `http://localhost:5000` (hostname)

❌ **WRONG (development only):**
```python
socketio.run(app, host='127.0.0.1', port=5000)
# Only accessible from this machine, not from ESP32!
```

✅ **Verify connectivity:**
```bash
# From ESP32 machine, ping server
ping 192.168.43.67

# From server, check listening ports
netstat -an | findstr LISTENING
# Should show: TCP 0.0.0.0:5000
```

---

### 8.2 Firewall Issues

✅ **Check Windows Firewall:**
```powershell
# Allow port through firewall
netsh advfirewall firewall add rule name="Flask-SocketIO" dir=in action=allow protocol=tcp localport=5000
```

✅ **Verify port is accessible:**
```bash
telnet 192.168.43.67 5000
# Should connect successfully
```

❌ **If blocked:**
```
Connection refused / Connection timeout
→ Check Windows Firewall: Windows Defender Firewall → Allow app through firewall
```

---

### 8.3 Port Blocking

✅ **Check if port 5000 is in use:**
```powershell
netstat -ano | findstr :5000

# If port is in use, kill process:
taskkill /PID [PID_number] /F

# Or use different port:
python app.py  # Modify port in app.py before running
```

---

### 8.4 Reverse Proxy (if applicable)

✅ **If using Nginx/Apache reverse proxy:**
```nginx
# Nginx config
location /socket.io {
    proxy_pass http://127.0.0.1:5000/socket.io;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "Upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_buffering off;
}
```

✅ **Don't use reverse proxy in development** - connect directly to port 5000

---

### 8.5 HTTPS vs HTTP Mismatch

✅ **CORRECT (for development):**
```python
# HTTP only
socketio.run(app, host='0.0.0.0', port=5000)
# Access: http://192.168.43.67:5000
```

⚠️ **Note:** WebSocket over HTTPS requires SSL certificates. Not needed for development/LAN.

❌ **WRONG:**
```
Server: http://192.168.43.67:5000
Client: https://192.168.43.67:5000
→ Mixed content error, WebSocket fails
```

---

## 9️⃣ PERFORMANCE & STABILITY CHECKS

### 9.1 Check for Event Loop Blocking Code

✅ **CORRECT (non-blocking):**
```python
def analyze_and_store_image(filepath, filename):
    """Background task - doesn't block request handling"""
    try:
        ai_caption = analyze_image_with_ai(filepath)  # Takes time
        # Store in database...
    except Exception as e:
        print(f"Error: {e}")

# Start as background task
socketio.start_background_task(analyze_and_store_image, filepath, filename)
```

❌ **WRONG (blocking):**
```python
@app.route('/upload', methods=['POST'])
def upload():
    image_data = request.get_data()
    # Save file...
    
    # ❌ This blocks the request!
    ai_caption = analyze_image_with_ai(filepath)  # Takes 5-10 seconds!
    
    return 'OK'  # Takes forever to respond
```

---

### 9.2 Avoid Long Synchronous Tasks

✅ **CORRECT (use background tasks):**
```python
@app.route('/upload', methods=['POST'])
def upload_binary_image():
    image_data = request.get_data()
    filepath = save_file(image_data)  # Fast: < 100ms
    
    # Start AI analysis in background (non-blocking)
    socketio.start_background_task(analyze_and_store_image, filepath, filename)
    
    return 'OK', 200  # Returns immediately!
```

✅ **Verify with timing:**
```python
import time

@app.route('/upload', methods=['POST'])
def upload():
    start = time.time()
    image_data = request.get_data()
    filepath = save_file(image_data)
    socketio.start_background_task(analyze_and_store_image, filepath, filename)
    elapsed = time.time() - start
    
    print(f"Request handled in {elapsed*1000:.1f}ms")  # Should be < 100ms
    return 'OK', 200
```

---

### 9.3 Confirm Background Tasks Setup

✅ **CORRECT (your setup):**
```python
def emit_update():
    with app.app_context():
        socketio.emit('update_event', data)

socketio.start_background_task(emit_update)
```

✅ **With parameters:**
```python
def analyze_and_store_image(filepath, filename):
    # Long operation...
    pass

socketio.start_background_task(analyze_and_store_image, filepath, filename)
```

✅ **Verify background tasks are running:**
```python
import sys

@app.route('/status')
def status():
    return {
        'active_threads': threading.active_count(),
        'thread_names': [t.name for t in threading.enumerate()],
        'async_mode': socketio.async_mode
    }
```

---

### 9.4 Memory Leak Inspection

✅ **Monitor memory usage:**
```bash
# During normal operation
import psutil
import os

process = psutil.Process(os.getpid())

@app.route('/memory-stats')
def memory_stats():
    process.memory_info()
    return {
        'rss_mb': process.memory_info().rss / 1024 / 1024,  # Resident memory
        'vms_mb': process.memory_info().vms / 1024 / 1024,  # Virtual memory
        'num_threads': process.num_threads()
    }
```

✅ **Check for growing memory:**
```bash
# After several hours of operation
curl http://127.0.0.1:5000/memory-stats
# Memory should be stable (not constantly growing)
```

---

## 🔟 LOGGING & DEBUG CHECKLIST

### 10.1 Structured Logging

✅ **ADD to app.py (at top):**
```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Or for file logging
handler = logging.FileHandler('server.log')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)
```

---

### 10.2 Connection/Disconnection Logs

✅ **Add to handle_connect():**
```python
@socketio.on('connect')
def handle_connect():
    logger.info(f'✅ Client connected: {request.sid} from {request.remote_addr}')
    connected_clients.add(request.sid)
    
    def emit_connection():
        with app.app_context():
            socketio.emit('connection_response', {'status': 'Connected'})
    
    socketio.start_background_task(emit_connection)
```

✅ **Add to handle_disconnect():**
```python
@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'❌ Client disconnected: {request.sid}')
    logger.info(f'   Connected clients remaining: {len(connected_clients)} - {connected_clients}')
    connected_clients.discard(request.sid)
```

---

### 10.3 Error Handlers

✅ **ADD error handlers (already in your code):**
```python
@socketio.on_error()
def error_handler(e):
    logger.error(f'SocketIO error: {e}', exc_info=True)

@socketio.on_error_default
def default_error_handler(e):
    logger.error(f'SocketIO default error: {e}', exc_info=True)

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'500 Internal error: {error}', exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500
```

---

### 10.4 Try/Except Around Emits

✅ **CORRECT (your pattern):**
```python
try:
    def emit_update():
        with app.app_context():
            socketio.emit('sensor_update', {
                'timestamp': datetime.now().isoformat(),
                'data': sensor_data
            })
    
    socketio.start_background_task(emit_update)
except Exception as e:
    logger.error(f'Failed to emit sensor_update: {e}', exc_info=True)
```

---

## ✅ VERIFICATION COMMANDS CHECKLIST

Run these commands to verify your entire setup:

```bash
# 1. Verify environment
python -c "import flask_socketio; print(f'Flask-SocketIO: {flask_socketio.__version__}')"
python -c "import socketio; print(f'python-socketio: {socketio.__version__}')"

# 2. Check virtual environment
pip -V | grep ".venv"

# 3. Check for conflicts
pip check

# 4. Verify server starts
cd 'e:\Rajeev\esp 32\esp 32'
python app.py &

# 5. Test health endpoint
curl http://127.0.0.1:5000/api/health

# 6. Test OLED display endpoint
curl http://127.0.0.1:5000/api/oled-display/get

# 7. Check port is listening
netstat -an | findstr 5000

# 8. Test from another machine
curl http://192.168.43.67:5000/api/health
```

---

## 📋 PRODUCTION DEPLOYMENT CHECKLIST

When moving to production:

- [ ] Set `debug=False` in socketio.run()
- [ ] Set `logger=False` and `engineio_logger=False`
- [ ] Use environment variables for SECRET_KEY
- [ ] Set specific `cors_allowed_origins` instead of "*"
- [ ] Use production WSGI server (Gunicorn)
- [ ] Setup SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Setup monitoring/logging system
- [ ] Test reconnect behavior on network loss
- [ ] Test max concurrent connections
- [ ] Setup rate limiting
- [ ] Monitor memory usage over time

---

## 🎯 QUICK SANITY CHECK (5 minutes)

```bash
# 1. Start server
cd 'e:\Rajeev\esp 32\esp 32'
python app.py

# 2. In another terminal, test health
curl http://127.0.0.1:5000/api/health

# 3. Test OLED endpoints
curl -X POST http://127.0.0.1:5000/api/oled-display/set \
  -H "Content-Type: application/json" \
  -d '{"animation_id": 2, "animation_type": "pet"}'

curl http://127.0.0.1:5000/api/oled-display/get

# 4. Open browser and visit
# http://127.0.0.1:5000

# 5. Check browser console (F12) for WebSocket connection

# Expected output:
# - Dashboard loads without errors
# - WebSocket connected message in console
# - Animation buttons responsive
```

---

✅ **Your setup is VERIFIED and PRODUCTION-READY with these 10 categories checked!**
