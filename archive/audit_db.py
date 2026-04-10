import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
POSTGRES_URI = os.getenv("POSTGRES_URI")

def audit():
    conn = None
    try:
        conn = psycopg2.connect(POSTGRES_URI, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            # 1. Broadly find any assets related to Physical or Cold
            cur.execute("SELECT id, name FROM asset WHERE name ILIKE '%Physical%' OR name ILIKE '%Cold%'")
            assets = cur.fetchall()
            print(f"Total relevant assets: {len(assets)}")
            
            for asset in assets:
                # 2. For each asset, find the latest data point and all available attributes
                print(f"\nChecking Asset: {asset['name']} ({asset['id']})")
                cur.execute("""
                    SELECT attribute_name, MAX(timestamp) as last_ts, count(*) as count
                    FROM asset_datapoint 
                    WHERE entity_id = %s 
                    GROUP BY attribute_name
                """, (asset['id'],))
                attrs = cur.fetchall()
                for attr in attrs:
                    print(f" - Attr: {attr['attribute_name']}, Last TS: {attr['last_ts']}, Count: {attr['count']}")
                    
                    # 3. Sample one data point to see the JSON structure
                    cur.execute("""
                        SELECT value FROM asset_datapoint 
                        WHERE entity_id = %s AND attribute_name = %s 
                        ORDER BY timestamp DESC LIMIT 1
                    """, (asset['id'], attr['attribute_name']))
                    sample = cur.fetchone()
                    print(f"   Sample JSON: {sample['value']}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    audit()
