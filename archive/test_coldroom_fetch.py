import logging
from database.coldroom_db import fetch_coldroom_data
from database.db_utils import execute_query

logging.basicConfig(level=logging.INFO)

def test():
    print("Testing fetch_coldroom_data...")
    df = fetch_coldroom_data(days=7)
    if df is not None:
        print(f"Fetched {len(df)} rows.")
        print("Unique Coldroom Names in DF:", df['coldroom_name'].unique())
    else:
        print("No data returned by fetch_coldroom_data.")

    print("\nChecking assets that contain 'Temperature' in value column...")
    query = "SELECT a.name, count(*) FROM asset_datapoint d JOIN asset a ON d.entity_id = a.id WHERE d.value::jsonb ? 'Temperature' AND d.timestamp >= (SELECT MAX(timestamp) FROM asset_datapoint) - INTERVAL '7 days' GROUP BY a.name"
    res = execute_query(query)
    print("Assets with Temperature data (last 7 days):", res)

if __name__ == "__main__":
    test()
