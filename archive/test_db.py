import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("POSTGRES_URI")
print(f"Connecting to: {uri}")

try:
    conn = psycopg2.connect(uri)
    cur = conn.cursor()
    
    # Test Coldroom Query
    cur.execute("SELECT name FROM asset WHERE name LIKE 'Cold Room %' LIMIT 5")
    names = cur.fetchall()
    print(f"Cold Room Names: {names}")
    
    # Test Tank Query
    cur.execute("SELECT name FROM asset WHERE name LIKE 'physical_%' LIMIT 5")
    tank_names = cur.fetchall()
    print(f"Tank Names: {tank_names}")
    
    # Test Data points
    cur.execute("SELECT count(*) FROM asset_datapoint d JOIN asset a ON d.entity_id = a.id WHERE a.name LIKE 'physical_%' AND d.timestamp >= NOW() - INTERVAL '30 days'")
    count = cur.fetchone()[0]
    print(f"Datapoints for Tanks in last 30 days: {count}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
