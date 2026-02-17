#!/usr/bin/env python3
"""
Database Migration Script: Optimize for Performance
- Remove BLOB columns from sensor_readings
- Add device_id column
- Create separate images table for media files
- Preserve existing sensor data
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "sensor_data.db"
BACKUP_PATH = f"sensor_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

def migrate_database():
    print("🔄 Starting Database Migration...")
    print("=" * 50)
    
    if not os.path.exists(DB_PATH):
        print("❌ Database not found!")
        return False
    
    # Create backup
    print("📋 Creating backup...")
    os.system(f'copy "{DB_PATH}" "{BACKUP_PATH}"')
    print(f"✅ Backup created: {BACKUP_PATH}")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check existing data count
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        existing_count = cursor.fetchone()[0]
        print(f"📊 Existing records: {existing_count}")
        
        # Step 1: Create new optimized sensor_readings table
        print("\n🗃️  Creating optimized sensor_readings table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT DEFAULT 'ESP32_001',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                accel_x REAL,
                accel_y REAL,
                accel_z REAL,
                gyro_x REAL,
                gyro_y REAL,
                gyro_z REAL,
                mic_level REAL
            )
        ''')
        
        # Step 2: Copy data (excluding BLOBs)
        print("📤 Migrating sensor data...")
        cursor.execute('''
            INSERT INTO sensor_readings_new 
            (id, timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mic_level)
            SELECT id, timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mic_level
            FROM sensor_readings
        ''')
        
        # Step 3: Create images table
        print("🖼️  Creating images table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT DEFAULT 'ESP32_001',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT NOT NULL,
                is_important BOOLEAN DEFAULT 0,
                file_size INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Step 4: Drop old table and rename new one
        print("🔄 Replacing old table...")
        cursor.execute("DROP TABLE sensor_readings")
        cursor.execute("ALTER TABLE sensor_readings_new RENAME TO sensor_readings")
        
        # Verify migration
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        new_count = cursor.fetchone()[0]
        
        conn.commit()
        conn.close()
        
        print(f"✅ Migration completed!")
        print(f"   Records migrated: {new_count}/{existing_count}")
        print(f"   Images table created")
        print(f"   BLOB columns removed")
        print(f"   Device ID added")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print(f"🔄 Restoring backup...")
        os.system(f'copy "{BACKUP_PATH}" "{DB_PATH}"')
        return False

if __name__ == "__main__":
    migrate_database()