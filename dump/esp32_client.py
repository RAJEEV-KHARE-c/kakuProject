"""
ESP32 Sensor Data Sender - MicroPython Version
This code reads sensors from ESP32 and sends data to the dashboard

Required Libraries:
- mpu6050 (for accelerometer/gyroscope)
- camera module
- microphone input

Make sure to install: mpremote install mpu6050
"""

import network
import socket
import json
import time
from machine import Pin, I2C, ADC
import urequests

# WiFi credentials
SSID = "YOUR_SSID"
PASSWORD = "YOUR_PASSWORD"
SERVER_IP = "192.168.X.X"  # Change to your server IP
SERVER_PORT = 5000

# Sensor data storage
sensor_data = {
    'accel_x': 0.0,
    'accel_y': 0.0,
    'accel_z': 0.0,
    'gyro_x': 0.0,
    'gyro_y': 0.0,
    'gyro_z': 0.0,
    'mic_level': 0,
    'sound_data': 0,
    'camera_image': None
}

class ESP32Dashboard:
    def __init__(self):
        self.wifi_connected = False
        self.server_url = f"http://{SERVER_IP}:{SERVER_PORT}"
        self.connect_wifi()
        self.init_sensors()
    
    def connect_wifi(self):
        """Connect to WiFi network"""
        print("Connecting to WiFi...")
        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)
        wlan.connect(SSID, PASSWORD)
        
        # Wait for connection
        max_wait = 10
        while max_wait > 0:
            if wlan.isconnected():
                print(f"WiFi connected! IP: {wlan.ifconfig()[0]}")
                self.wifi_connected = True
                return
            max_wait -= 1
            time.sleep(1)
        
        print("WiFi connection failed!")
    
    def init_sensors(self):
        """Initialize MPU6050 and other sensors"""
        try:
            # I2C for MPU6050 (accelerometer/gyroscope)
            i2c = I2C(1, scl=Pin(22), sda=Pin(21), freq=400000)
            
            # Initialize ADC for microphone
            self.mic_adc = ADC(Pin(35))  # GPIO35 for analog input
            self.mic_adc.atten(ADC.ATTN_11DB)
            
            print("Sensors initialized successfully")
        except Exception as e:
            print(f"Error initializing sensors: {e}")
    
    def read_accelerometer(self):
        """Read accelerometer data (simulated)"""
        # In real implementation, read from MPU6050
        # For now, return simulated data
        return {
            'accel_x': 0.13,
            'accel_y': -0.09,
            'accel_z': 9.81
        }
    
    def read_gyroscope(self):
        """Read gyroscope data (simulated)"""
        # In real implementation, read from MPU6050
        return {
            'gyro_x': 0.45,
            'gyro_y': -0.23,
            'gyro_z': 0.12
        }
    
    def read_microphone(self):
        """Read microphone level"""
        try:
            # Read ADC value and convert to dB (simplified)
            adc_value = self.mic_adc.read()
            db_level = (adc_value / 4095.0) * 100
            return db_level
        except:
            return 0
    
    def read_camera(self):
        """Capture camera image (if camera module is connected)"""
        # This requires camera module - returns None if not available
        try:
            # Example with OV2640 camera module
            # import camera
            # camera.init(0, format=camera.JPEG, framesize=camera.FRAME_VGA)
            # buf = camera.capture()
            # return buf
            return None
        except:
            return None
    
    def collect_sensor_data(self):
        """Collect all sensor data"""
        data = {}
        
        # Read accelerometer
        accel = self.read_accelerometer()
        data.update(accel)
        
        # Read gyroscope
        gyro = self.read_gyroscope()
        data.update(gyro)
        
        # Read microphone
        data['mic_level'] = self.read_microphone()
        data['sound_data'] = int(data['mic_level'] * 10)
        
        # Read camera
        camera_img = self.read_camera()
        if camera_img:
            import ubinascii
            data['camera_image'] = ubinascii.b2a_base64(camera_img).decode('utf-8')
        
        return data
    
    def send_data(self, data):
        """Send sensor data to dashboard server"""
        if not self.wifi_connected:
            print("WiFi not connected, skipping data send")
            return False
        
        try:
            url = f"{self.server_url}/api/sensor-data"
            headers = {'Content-Type': 'application/json'}
            
            # Send data via HTTP POST
            response = urequests.post(url, 
                                     data=json.dumps(data),
                                     headers=headers,
                                     timeout=5)
            
            if response.status_code == 200:
                print("Data sent successfully")
                return True
            else:
                print(f"Server error: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"Error sending data: {e}")
            return False
    
    def run(self):
        """Main loop - collect and send data"""
        loop_count = 0
        
        while True:
            try:
                print(f"\n--- Loop {loop_count} ---")
                
                # Collect sensor data
                sensor_data = self.collect_sensor_data()
                print(f"Accel: X={sensor_data['accel_x']}, Y={sensor_data['accel_y']}, Z={sensor_data['accel_z']}")
                print(f"Gyro: X={sensor_data['gyro_x']}, Y={sensor_data['gyro_y']}, Z={sensor_data['gyro_z']}")
                print(f"Microphone: {sensor_data['mic_level']:.1f} dB")
                
                # Send data to server
                self.send_data(sensor_data)
                
                loop_count += 1
                
                # Send data every 1 second (adjust as needed)
                time.sleep(1)
            
            except KeyboardInterrupt:
                print("Stopping...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(2)

# Start the dashboard client
if __name__ == '__main__':
    dashboard = ESP32Dashboard()
    dashboard.run()
