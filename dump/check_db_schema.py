import sqlite3
import os

DB_PATH = 'sensor_data.db'

print("Checking database...")
if not os.path.exists(DB_PATH):
    print("❌ Database file does not exist!")
    exit()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check if sensor_readings table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sensor_readings'")
table_exists = cursor.fetchone()

if not table_exists:
    print("❌ sensor_readings table does not exist!")
else:
    print("✅ sensor_readings table exists")
    
    # Get table schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sensor_readings'")
    schema = cursor.fetchone()[0]
    print("\nTable schema:")
    print(schema)
    
    # Check if camera_image column exists
    cursor.execute("PRAGMA table_info(sensor_readings)")
    columns = cursor.fetchall()
    
    print("\nColumns:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    has_camera_image = any(col[1] == 'camera_image' for col in columns)
    if has_camera_image:
        print("✅ camera_image column exists")
    else:
        print("❌ camera_image column is missing!")

conn.close()