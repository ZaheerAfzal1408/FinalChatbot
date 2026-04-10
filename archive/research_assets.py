import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
POSTGRES_URI = os.getenv("POSTGRES_URI")

def research():
    try:
        conn = psycopg2.connect(POSTGRES_URI, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            # 1. Search for all assets
            cur.execute("SELECT id, name FROM asset ORDER BY name")
            assets = cur.fetchall()
            print("\n--- ALL ASSETS ---")
            for a in assets:
                print(f"ID: {a['id']}, Name: {a['name']}")
                
            # 2. Search for specifically mentioned IDs from n8n
            n8n_ids = [
                '2TV1OLqZSv5JwIEHomeRR1', '6UXHYoFE9as4nc25HA72AN', '5UwgGOBQ8VqdVZO37E1ci8', 
                '2H7Gj65gRcLS1V6OTq72Hk', '7BKD81c3z8gKpGZt5bIc87', '2pd9BV69qYHsnoyiAAqJmX', 
                '50Y5udYfIWJWLEbduASYwT', '6tUKaWyipyL8MfjaTntrSH', '6UrAIw3WfeBmCn7yYgHGKR', 
                '64G6XlehtAyuhFP6U5V9iB', '7gmxORynTjM8Lmp4DDtZSP', '4la3XSG9sUisBdczxxjYLv'
            ]
            cur.execute("SELECT id, name FROM asset WHERE id = ANY(%s)", (n8n_ids,))
            matches = cur.fetchall()
            print("\n--- N8N ID MATCHES ---")
            for m in matches:
                print(f"ID: {m['id']}, Name: {m['name']}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    research()
