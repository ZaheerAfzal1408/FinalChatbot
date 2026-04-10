import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
POSTGRES_URI = os.getenv("POSTGRES_URI")

def verify_data():
    conn = None
    try:
        conn = psycopg2.connect(POSTGRES_URI, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            print("--- Database Connected ---")
            
            # 1. Search for ANY assets that look like Coldrooms or Tanks
            search_query = "SELECT id, name FROM asset WHERE name ILIKE '%Physical%' OR name ILIKE '%Cold%'"
            cur.execute(search_query)
            assets = cur.fetchall()
            print(f"Found {len(assets)} assets matching 'Physical' or 'Cold'.")
            if assets:
                for a in assets[:10]:
                    print(f"ID: {a['id']}, Name: {a['name']}")
                    
            # 2. Check for data points for these assets in the last 30 days
            if assets:
                ids = [a['id'] for a in assets]
                data_query = """
                    SELECT entity_id, count(*) 
                    FROM asset_datapoint 
                    WHERE entity_id = ANY(%s) 
                    AND timestamp >= NOW() - INTERVAL '30 days'
                    GROUP BY entity_id
                """
                cur.execute(data_query, (ids,))
                dist = cur.fetchall()
                print(f"\nDistribuition of data in last 30 days: {dist}")
                
                # 3. Check for ALL TIME data points if 30 days is empty
                if not dist:
                    print("\nNo data in last 30 days. Checking ALL TIME...")
                    cur.execute("SELECT entity_id, count(*) FROM asset_datapoint WHERE entity_id = ANY(%s) GROUP BY entity_id LIMIT 10", (ids,))
                    all_time = cur.fetchall()
                    print(f"All-time distribution: {all_time}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    verify_data()
