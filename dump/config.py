"""
ESP32 Dashboard Configuration File
Modify these settings as needed
"""

# Server Configuration
SERVER_CONFIG = {
    'host': '0.0.0.0',           # Listen on all interfaces
    'port': 5000,                 # Flask server port
    'debug': True,                # Debug mode (set to False in production)
    'threaded': True,             # Enable threading
}

# Database Configuration
DATABASE_CONFIG = {
    'path': 'sensor_data.db',     # SQLite database file
    'type': 'sqlite',             # Database type
    'max_records': 100000,        # Maximum records before cleanup
    'auto_cleanup': True,         # Auto-delete old data
    'cleanup_days': 30,           # Keep data for this many days
}

# WiFi Configuration (for ESP32)
WIFI_CONFIG = {
    'ssid': '123',
    'password': 'KUNAL 26',
    'retry_count': 10,
    'retry_delay': 1,
}

# Sensor Configuration
SENSOR_CONFIG = {
    'accel_enabled': True,
    'gyro_enabled': True,
    'mic_enabled': True,
    'camera_enabled': False,      # Enable if camera module is connected
    'send_interval': 1000,        # Send data every N milliseconds
    'sample_rate': 100,           # Hz
}

# Camera Configuration
CAMERA_CONFIG = {
    'resolution': 'VGA',          # QQVGA, HQVGA, QVGA, CIF, VGA, SVGA, XGA
    'quality': 10,                # 0-63 (higher = better quality, slower)
    'brightness': 0,              # -2 to 2
    'contrast': 0,                # -2 to 2
    'saturation': 0,              # -2 to 2
}

# ESP32 Hardware Pins
HARDWARE_PINS = {
    'i2c_sda': 21,               # MPU6050 SDA
    'i2c_scl': 22,               # MPU6050 SCL
    'mic_adc': 35,               # Microphone ADC pin
    'led_status': 2,             # Status LED
    'camera_enabled': False,     # Set to True if using camera
}

# Data Limits
DATA_LIMITS = {
    'max_accel': 50,             # m/s² - maximum expected acceleration
    'max_gyro': 360,             # deg/s - maximum expected rotation
    'min_mic': 0,                # dB minimum
    'max_mic': 120,              # dB maximum
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',             # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'dashboard.log',
    'max_size': 10485760,        # 10MB
    'backup_count': 5,
}

# CORS Settings
CORS_CONFIG = {
    'origins': ['*'],            # Allow all origins (change for production)
    'methods': ['GET', 'POST'],
    'allow_headers': ['*'],
}

# API Rate Limiting
RATE_LIMIT_CONFIG = {
    'enabled': False,            # Enable rate limiting (production)
    'requests_per_minute': 1000,
    'requests_per_hour': 100000,
}

# Export Settings
EXPORT_CONFIG = {
    'formats': ['json', 'csv'],
    'max_records': 50000,
    'compression': 'gzip',
}

# MQTT Settings (optional - for cloud integration)
MQTT_CONFIG = {
    'enabled': False,
    'broker': 'mqtt.example.com',
    'port': 1883,
    'topic': 'esp32/sensors',
    'username': '',
    'password': '',
}

# InfluxDB Settings (optional - for time-series storage)
INFLUXDB_CONFIG = {
    'enabled': False,
    'host': 'localhost',
    'port': 8086,
    'database': 'esp32_sensors',
    'username': '',
    'password': '',
}

# Alert Settings
ALERT_CONFIG = {
    'enabled': True,
    'email': 'your_email@example.com',
    'thresholds': {
        'accel_x_max': 20,
        'accel_y_max': 20,
        'accel_z_min': 8,
        'accel_z_max': 11,
        'mic_level_max': 100,
    }
}

# Frontend Settings
FRONTEND_CONFIG = {
    'theme': 'dark',             # 'light' or 'dark'
    'update_interval': 1000,     # ms - how often to update display
    'max_table_rows': 50,        # Maximum rows in data table
    'chart_history': 300,        # Seconds of history to show in charts
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'gzip_compression': True,
    'cache_static_files': True,
    'cache_duration': 3600,      # seconds
    'use_cdn': False,
}

# Security Settings
SECURITY_CONFIG = {
    'enable_auth': False,        # Enable authentication
    'secret_key': 'change_this_secret_key_in_production',
    'session_timeout': 3600,     # seconds
    'require_https': False,      # Require HTTPS (production)
    'api_key_required': False,   # Require API key for endpoints
}
