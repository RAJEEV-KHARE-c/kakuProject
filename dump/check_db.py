#!/usr/bin/env python3
import sqlite3
import os

print("📊 ESP32 DATABASE STRUCTURE")
print("=" * 50)

db_path = "sensor_data.db"

if not os.path.exists(db_path):
    print("❌ Database file not found!")
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("PRAGMA table_info(sensor_readings)")
    columns = cursor.fetchall()
    
    print("\n🗃️  TABLE: sensor_readings")
    print("-" * 30)
    print(f"{'Column':<15} {'Type':<10} {'Constraints'}")
    print("-" * 45)
    
    for col in columns:
        cid, name, data_type, not_null, default_val, pk = col
        
        constraints = []
        if pk:
            constraints.append("PRIMARY KEY")
        if not_null:
            constraints.append("NOT NULL")
        if default_val:
            constraints.append(f"DEFAULT {default_val}")
        
        constraint_str = " ".join(constraints)
        print(f"{name:<15} {data_type:<10} {constraint_str}")
    
    # Count records
    cursor.execute("SELECT COUNT(*) FROM sensor_readings")
    total_records = cursor.fetchone()[0]
    
    print(f"\n📈 CURRENT DATA:")
    print(f"   Total Records: {total_records}")
    
    if total_records > 0:
        # Get latest record
        cursor.execute("""
            SELECT timestamp, accel_x, accel_y, accel_z, mic_level, 
                   CASE WHEN camera_image IS NOT NULL THEN 'YES' ELSE 'NO' END as has_image,
                   CASE WHEN audio_data IS NOT NULL THEN 'YES' ELSE 'NO' END as has_audio
            FROM sensor_readings 
            ORDER BY id DESC LIMIT 1
        """)
        latest = cursor.fetchone()
        
        if latest:
            print(f"   Latest Entry: {latest[0]}")
            print(f"   Acceleration: X={latest[1]}, Y={latest[2]}, Z={latest[3]}")
            print(f"   Microphone: {latest[4]} dB")
            print(f"   Has Image: {latest[5]}")
            print(f"   Has Audio: {latest[6]}")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")