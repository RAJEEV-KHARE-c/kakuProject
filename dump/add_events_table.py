#!/usr/bin/env python3
"""
Add important_events table to database
""" 

import sqlite3
import os

DB_PATH = "sensor_data.db"

def add_important_events_table():
    print("🚀 Adding important_events table...")
    print("=" * 40)
    
    if not os.path.exists(DB_PATH):
        print("❌ Database not found!")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create important_events table
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
        
        conn.commit()
        
        # Verify table creation
        cursor.execute("PRAGMA table_info(important_events)")
        columns = cursor.fetchall()
        
        print("✅ important_events table created!")
        print("\n📋 Table Structure:")
        print("-" * 30)
        for col in columns:
            cid, name, data_type, not_null, default_val, pk = col
            constraints = []
            if pk: constraints.append("PRIMARY KEY") 
            if not_null: constraints.append("NOT NULL")
            if default_val: constraints.append(f"DEFAULT {default_val}")
            
            constraint_str = " ".join(constraints)
            print(f"  {name:<12} {data_type:<10} {constraint_str}")
        
        # List all tables  
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"\n🗃️  All Tables: {tables}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    add_important_events_table()