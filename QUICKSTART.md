# ESP32 Dashboard - Quick Start Guide

## TL;DR - Get Running in 5 Minutes

### Step 1: Install Python Dependencies
```powershell
cd "c:\Users\husan\OneDrive\Desktop\esp 32"
pip install -r requirements.txt
```

### Step 2: Start the Server
```powershell
python app.py
```
✓ Dashboard will be available at: **http://localhost:5000**

### Step 3: Test with Sample Data (Without ESP32)
Open **another PowerShell window**:
```powershell
cd "c:\Users\husan\OneDrive\Desktop\esp 32"
python test_generator.py
```

You should now see data flowing into your dashboard!

---

## Full Setup (With Real ESP32)

### Using Arduino C++ Code (Easier)

#### 1. Install Arduino IDE
Download from: https://www.arduino.cc/en/software

#### 2. Install Required Libraries
In Arduino IDE:
1. Go to: `Sketch > Include Library > Manage Libraries`
2. Search and install:
   - `ArduinoJson` (by Benoit Blanchon)
   - `MPU6050` (by Jeff Rowberg)

#### 3. Configure ESP32 Code
1. Open `esp32_sketch.ino` in Arduino IDE
2. Update WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_SSID";
   const char* password = "YOUR_PASSWORD";
   ```
3. Update dashboard server IP:
   - Run in PowerShell: `ipconfig`
   - Find "IPv4 Address" (looks like 192.168.X.X)
   ```cpp
   const char* serverUrl = "http://YOUR_IP:5000/api/sensor-data";
   ```

#### 4. Connect ESP32 & Upload
1. Connect ESP32 to computer via USB
2. Select board: `Tools > Board > ESP32 > ESP32 Dev Module`
3. Select COM port: `Tools > Port > COM3` (or your port)
4. Click Upload button or press `Ctrl+U`

#### 5. Verify Connection
1. Open Serial Monitor: `Tools > Serial Monitor` (Baud: 115200)
2. Look for "WiFi connected" and "Data sent successfully" messages

### Using MicroPython (Alternative)

1. Install mpremote: `pip install mpremote`
2. Flash MicroPython on ESP32 (see: https://micropython.org/download/esp32/)
3. Update WiFi credentials in `esp32_client.py`
4. Upload: `mpremote cp esp32_client.py :main.py`
5. Run: `mpremote run main.py`

---

## Hardware Connections

### Essential Components:
- ESP32 Dev Board
- MPU6050 (Accelerometer + Gyroscope)
- USB Cable
- Jumper wires

### Wiring (Simple):

```
MPU6050          ESP32
=============================
VCC      ----    3.3V
GND      ----    GND
SDA      ----    GPIO 21
SCL      ----    GPIO 22
```

### Optional:
- **Microphone**: Connect to GPIO 35
- **Camera (OV2640)**: See detailed wiring in `esp32_sketch.ino`
- **LED**: Connect to GPIO 2 for status indicator

---

## Dashboard Features

### Real-time Displays:
- 📊 Accelerometer readings (X, Y, Z axis)
- 🌀 Gyroscope values
- 🎤 Microphone levels with audio visualization
- 📷 Camera feed (if connected)
- 📈 Latest 50 sensor readings table
- 🔌 Connection status

### Controls:
- **Start/Stop Recording**: Save data buffer
- **Download Data**: Export as CSV
- **Clear Database**: Reset all stored data

---

## Testing Without Hardware

Use the test data generator to verify everything works:

```powershell
python test_generator.py

```

This sends realistic simulated sensor data to your dashboard. Perfect for:
- Testing the web interface
- Verifying database storage
- Checking API endpoints
- Development without hardware

---

## Accessing Your Dashboard

### Local Network:
```
http://localhost:5000
```

### From Another Computer:
```
http://YOUR_COMPUTER_IP:5000
```

To find your PC's IP:
```powershell
ipconfig
# Look for "IPv4 Address" starting with 192.168 or 10.0
```

---

## Common Issues & Fixes

### ❌ "ModuleNotFoundError: No module named 'flask'"
**Solution:**
```powershell
pip install -r requirements.txt
```

### ❌ "Port 5000 already in use"
**Solution:**
```powershell
# Change port in app.py line: socketio.run(app, host='0.0.0.0', port=5001)
# Or kill the process:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### ❌ ESP32 won't connect to WiFi
- Check SSID and password are correct
- Verify ESP32 antenna is connected
- Make sure router broadcasts 2.4GHz (not just 5GHz)

### ❌ Dashboard shows "Disconnected"
- Check server is running: `python app.py`
- Check firewall allows port 5000
- Verify ESP32 and PC are on same network

### ❌ No data appearing on dashboard
- Check test generator: `python test_generator.py`
- Verify browser console (F12) for errors
- Check browser isn't blocking WebSocket

---

## File Descriptions

| File | Purpose |
|------|---------|
| `index.html` | Web dashboard UI |
| `app.py` | Python Flask backend server |
| `esp32_sketch.ino` | Arduino C++ code for ESP32 |
| `esp32_client.py` | MicroPython code for ESP32 |
| `requirements.txt` | Python dependencies |
| `config.py` | Configuration settings |
| `test_generator.py` | Simulated data generator |
| `sensor_data.db` | SQLite database (created automatically) |

---

## API Quick Reference

### Send Data to Dashboard:
```python
import requests
data = {
    "accel_x": 0.13,
    "accel_y": -0.09,
    "accel_z": 9.81,
    "gyro_x": 0.45,
    "gyro_y": -0.23,
    "gyro_z": 0.12,
    "mic_level": 45.5,
    "sound_data": 455
}
requests.post("http://localhost:5000/api/sensor-data", json=data)
```

### Get Latest Data:
```python
response = requests.get("http://localhost:5000/api/latest?limit=50")
data = response.json()
```

### Get Statistics:
```python
response = requests.get("http://localhost:5000/api/stats")
stats = response.json()
```

### Download All Data:
```
http://localhost:5000/api/export
```

### Clear Database:
```python
requests.post("http://localhost:5000/api/clear")
```

---

## Next Steps

1. ✅ Get test data working first
2. ✅ Wire up MPU6050 (accelerometer + gyroscope)
3. ✅ Upload Arduino code to ESP32
4. ✅ Connect microphone (optional)
5. ✅ Add camera module (optional)

---

## Need Help?

1. Check browser console: Press `F12` → Console tab
2. Check server logs: Look at PowerShell output where you ran `python app.py`
3. Review README.md for detailed documentation
4. Check hardware wiring matches the diagrams

---

## Performance Tips

- **Reduce send frequency**: Modify `SEND_INTERVAL` in ESP32 code
- **Clean old data**: Run `/api/clear` periodically
- **Optimize database**: Delete records older than needed
- **Use WiFi over BLE**: Faster data transfer

---

## Security Reminder

⚠️ This is a **development setup**. For production:
1. Add authentication
2. Use HTTPS/WSS
3. Set `debug=False` in app.py
4. Restrict CORS origins
5. Use strong passwords
6. Keep software updated

---

**Happy monitoring!** 🚀

Start with test data, then integrate your real hardware.
