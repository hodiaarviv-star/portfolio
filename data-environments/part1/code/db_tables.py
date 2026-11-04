import os
import getpass
import sqlite3

results_dir = os.path.join(
    '/home',
    getpass.getuser(),
    'task1',
    'results'
)

conn = sqlite3.connect(os.path.join(results_dir, 'monitor.db'))
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS files")
cursor.execute("DROP TABLE IF EXISTS calculations")

cursor.execute("""
CREATE TABLE files (
    name TEXT,
    format TEXT,
    scale REAL,
    compression TEXT,
    rows INTEGER,
    columns INTEGER,
    write_duration REAL
)
""")

cursor.execute("""
CREATE TABLE calculations (
    name TEXT,
    scale REAL,
    action TEXT,
    duration REAL
)
""")

conn.commit()
conn.close()

print("DONE")