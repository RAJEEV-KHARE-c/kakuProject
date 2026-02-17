import sqlite3

conn = sqlite3.connect('sensor_data.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]
print('📋 Database Tables:', tables)

# Check images table structure
cursor.execute("PRAGMA table_info(images)")
cols = cursor.fetchall()
print('\n🖼️ Images Table Structure:')
for col in cols:
    print(f'  {col[1]:<15} {col[2]:<10}')

conn.close()