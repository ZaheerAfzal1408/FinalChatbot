import os, psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('POSTGRES_URI'), cursor_factory=RealDictCursor)
with conn.cursor() as cur:
    # Check Physical01
    cur.execute("SELECT name, value, timestamp FROM asset_datapoint d JOIN asset a ON d.entity_id = a.id WHERE a.name = 'Physical01' ORDER BY d.timestamp DESC LIMIT 1")
    p1 = cur.fetchone()
    print(f"Physical01: {p1['value']} at {p1['timestamp']}")
    
    # Check Physical04
    cur.execute("SELECT name, value, timestamp FROM asset_datapoint d JOIN asset a ON d.entity_id = a.id WHERE a.name = 'Physical04' ORDER BY d.timestamp DESC LIMIT 1")
    p4 = cur.fetchone()
    print(f"Physical04: {p4['value']} at {p4['timestamp']}")
conn.close()
