# ESP32 Dashboard System

A complete real-time monitoring system for ESP32 sensors with web dashboard, database storage, and live data visualization.

## Features

✅ **Real-time Data Visualization**
- Live sensor readings (accelerometer, gyroscope)
- Microphone level display with audio visualizer
- Camera feed streaming
- Sound data analysis
- System status monitoring

✅ **Data Storage**
- SQLite database for persistent storage
- Automatic data archival
- CSV export functionality
- JSON data export

✅ **Web Dashboard**
- Beautiful, responsive UI
- Real-time updates via WebSocket
- Mobile-friendly design
- Historical data table with 50 latest readings

✅ **REST API**
- HTTP endpoints for data submission
- WebSocket support for real-time updates
- CORS enabled for cross-origin requests
- Health check endpoint

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Dashboard Server                        │
│                    (Python Flask/SocketIO)                   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Web Dashboard (HTML/CSS/JS)                │   │
│  │  - Real-time sensor visualization                    │   │
│  │  - Camera feed display                               │   │
│  │  - Audio visualizer                                  │   │
│  │  - Data table                                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↕                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         REST API + WebSocket Server                  │   │
│  │  - /api/sensor-data (POST)                           │   │
│  │  - /api/latest (GET)                                 │   │
│  │  - /api/stats (GET)                                  │   │
│  │  - /api/export (GET)                                 │   │
│  │  - /api/clear (POST)                                 │   │
│  │  - ws:// (WebSocket)                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↕                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         SQLite Database (sensor_data.db)             │   │
│  │  - Sensor readings storage                           │   │
│  │  - Timestamp tracking                                │   │
│  │  - Camera image storage                              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         ↑                                    
         │ HTTP/WebSocket                    
         │                                   
    ┌────────────┐                          
    │   ESP32    │                          
    │   Client   │                          
    │  (Arduino  │                          
    │    Code)   │                          
    └────────────┘                          
```

## Installation & Setup

### 1. **Server Setup (Dashboard Backend)**

#### Requirements:
- Python 3.7+
- pip (Python package manager)

#### Steps:

```bash
# Navigate to project directory
cd "c:\Users\husan\OneDrive\Desktop\esp 32"

# Install Python dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The dashboard will be available at: **http://localhost:5000**

### 2. **ESP32 Configuration**

#### Option A: Using Arduino C++ (Recommended for beginners)

**Required Libraries:**
- Install via Arduino IDE: `Sketch > Include Library > Manage Libraries`
  - `ArduinoJson` (by Benoit Blanchon)
  - `MPU6050` (by Jeff Rowberg)
  - `I2Cdev` (by Jeff Rowberg)

**Steps:**
1. Open `esp32_sketch.ino` in Arduino IDE
2. Update WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_SSID";
   const char* password = "YOUR_PASSWORD";
   const char* serverUrl = "http://192.168.X.X:5000/api/sensor-data";
   ```
3. Update server IP address (run `ipconfig` on Windows to find your PC's IP)
4. Connect ESP32 via USB
5. Select Board: `Tools > Board > ESP32 > ESP32 Dev Module`
6. Select correct COM port
7. Upload sketch: `Sketch > Upload`

#### Option B: Using MicroPython

**Steps:**
1. Install `mpremote`: `pip install mpremote`
2. Flash MicroPython on ESP32 (follow guides at micropython.org)
3. Upload `esp32_client.py` to ESP32:
   ```bash
   mpremote cp esp32_client.py :main.py
   ```
4. Update credentials in the script
5. Run: `mpremote run main.py`

### 3. **Hardware Wiring**

#### MPU6050 Accelerometer/Gyroscope:
```
MPU6050        ESP32
─────────────────────
VCC    ──→    3.3V
GND    ──→    GND
SDA    ──→    GPIO 21
SCL    ──→    GPIO 22
INT    ──→    GPIO 25 (optional)
```

#### Microphone (Analog):
```
Microphone     ESP32
─────────────────────
OUT    ──→    GPIO 35
VCC    ──→    3.3V
GND    ──→    GND
```

#### Camera (OV2640 - Optional):
See wiring diagram in `esp32_sketch.ino` comments

#### LED Indicator (Optional):
```
LED            ESP32
─────────────────────
+      ──→    GPIO 2 (220Ω resistor)
-      ──→    GND
```

## API Documentation

### POST: Send Sensor Data

**Endpoint:** `http://localhost:5000/api/sensor-data`

**Request:**
```json
{
    "accel_x": 0.13,
    "accel_y": -0.09,
    "accel_z": 9.81,
    "gyro_x": 0.45,
    "gyro_y": -0.23,
    "gyro_z": 0.12,
    "mic_level": 45.5,
    "sound_data": 455,
    "camera_image": "base64_encoded_image_data"
}
```

**Response:**
```json
{
    "status": "success",
    "message": "Data stored"
}
```

### GET: Latest Readings

**Endpoint:** `http://localhost:5000/api/latest?limit=20`

**Response:**
```json
{
    "success": true,
    "records": [...],
    "count": 20
}
```

### GET: Statistics

**Endpoint:** `http://localhost:5000/api/stats`

**Response:**
```json
{
    "total_records": 5000,
    "accel_average": {
        "x": 0.12,
        "y": -0.08,
        "z": 9.82
    },
    "date_range": {
        "start": "2026-02-09 10:00:00",
        "end": "2026-02-09 15:30:00"
    }
}
```

### GET: Export Data

**Endpoint:** `http://localhost:5000/api/export`

Returns all data as JSON file (downloaded)

### POST: Clear Database

**Endpoint:** `http://localhost:5000/api/clear`

**Response:**
```json
{
    "message": "Database cleared successfully"
}
```

## WebSocket Events

### Client → Server

**Event:** `sensor_data`
```javascript
socket.emit('sensor_data', {
    accel_x: 0.13,
    accel_y: -0.09,
    // ... other sensor values
});
```

### Server → Client

**Event:** `sensor_update`
```javascript
socket.on('sensor_update', function(data) {
    console.log('New sensor data:', data);
});
```

## Database Schema

### sensor_readings Table

```sql
CREATE TABLE sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    accel_x REAL,
    accel_y REAL,
    accel_z REAL,
    gyro_x REAL,
    gyro_y REAL,
    gyro_z REAL,
    mic_level REAL,
    sound_data INTEGER,
    camera_image BLOB
)
```

## Troubleshooting

### ESP32 Won't Connect to WiFi
- Check SSID and password
- Verify ESP32 WiFi antenna is secure
- Move closer to router
- Check if 2.4GHz WiFi is enabled (not 5GHz)

### Dashboard Not Showing Data
- Verify server is running: `python app.py`
- Check firewall allows port 5000
- Verify ESP32 has correct server IP: `ipconfig` command
- Check browser console for errors (F12)

### Database Errors
- Delete `sensor_data.db` file to reset database
- Ensure write permissions in project directory
- Check available disk space

### Microphone Not Reading
- Verify microphone is connected to GPIO 35
- Check ADC pin is not used by camera
- Test with multimeter to ensure microphone is outputting signal

### Camera Feed Not Showing
- Add camera library to Arduino sketch
- Verify OV2640 camera is properly connected
- Check PSRAM is enabled: `Tools > PSRAM > Enabled`

## Performance Optimization

### Reduce Data Send Frequency
In ESP32 code, adjust:
```cpp
const unsigned long SEND_INTERVAL = 1000;  // milliseconds
```

### Limit Table Display
In HTML, modify:
```javascript
while (tbody.rows.length > 50) {  // Limit to 50 rows
```

### Database Cleanup
Periodically clear old data:
```bash
curl -X POST http://localhost:5000/api/clear
```

## Advanced Features

### Custom Data Fields
To add new sensor types:

1. Update database schema in `app.py`:
```python
cursor.execute('''ALTER TABLE sensor_readings ADD COLUMN new_field REAL''')
```

2. Update HTML dashboard form

3. Update ESP32 code to read new sensor

### Multi-Dashboard Support
Server already handles multiple connected clients via WebSocket broadcasting

### Data Logging to Cloud
Extend `store_sensor_data()` function to also send data to cloud services

## File Structure

```
c:\Users\husan\OneDrive\Desktop\esp 32\
├── index.html                # Web dashboard (frontend)
├── app.py                    # Flask backend server
├── requirements.txt          # Python dependencies
├── esp32_sketch.ino         # Arduino C++ code for ESP32
├── esp32_client.py          # MicroPython code for ESP32
├── sensor_data.db           # SQLite database (auto-created)
└── README.md                # This file
```

## Security Notes

⚠️ **For Development Only:**
- CORS is fully enabled (allow all origins)
- Debug mode is ON
- No authentication implemented

### For Production:
1. Implement authentication tokens
2. Restrict CORS to known domains
3. Use HTTPS/WSS with SSL certificates
4. Add rate limiting
5. Validate all input data
6. Use environment variables for credentials

## License

Free to use for personal and educational purposes.

## Support

For issues or questions:
1. Check troubleshooting section
2. Review browser console (F12)
3. Check server logs in terminal
4. Verify hardware connections

---

**Dashboard Ready!** 🎉

Start the server and open http://localhost:5000 in your browser!
