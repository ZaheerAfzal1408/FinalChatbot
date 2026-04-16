import os
import sys
import pandas as pd
import logging
from database.db_utils import execute_query

# Mappings of Hash IDs to Asset Names (Populated dynamically)
COLDROOM_MAPPINGS = {}
TANK_MAPPINGS = {}
SMOKE_MAPPINGS = {} # {id: name}
ASSET_CACHE = {} # Unified cache for name-based lookup
ZONE_CACHE = {}  # Cache for hierarchical zone lookups {id: zone_path}

logger = logging.getLogger(__name__)

def load_dynamic_mappings():
    """Fetches coldroom and tank assets from DB instead of hardcoding."""
    global COLDROOM_MAPPINGS, TANK_MAPPINGS
    
    # 1. Fetch ColdRooms (Naming: ColdRoom1, ColdRoom2, etc.)
    cold_query = "SELECT id, name FROM asset WHERE name ILIKE 'ColdRoom%'"
    try:
        cold_results = execute_query(cold_query)
        COLDROOM_MAPPINGS.clear()
        COLDROOM_MAPPINGS.update({r['id']: r['name'].lower() for r in cold_results})
    except Exception as e:
        logger.error(f"Error fetching ColdRoom mappings: {e}")
    
    # 2. Fetch Tanks (Naming: Physical01, Physical02, etc.)
    tank_query = "SELECT id, name FROM asset WHERE name ILIKE 'Physical%'"
    try:
        tank_results = execute_query(tank_query)
        TANK_MAPPINGS.clear()
        TANK_MAPPINGS.update({r['id']: r['name'] for r in tank_results})
    except Exception as e:
        logger.error(f"Error fetching Tank mappings: {e}")
    
    # 3. Fetch Smoke Alarm Hierarchy (Recursive)
    smoke_query = """
        WITH RECURSIVE hierarchy AS (
            SELECT id, name, parent_id, NULL::text as zone_path, 0 as depth
            FROM asset WHERE name = 'Smoke Alarm System'
            UNION ALL
            SELECT a.id, a.name, a.parent_id,
                   CASE WHEN h.depth = 0 THEN a.name ELSE h.zone_path || '/' || a.name END,
                   h.depth + 1
            FROM asset a JOIN hierarchy h ON a.parent_id = h.id
        )
        SELECT id, name, zone_path FROM hierarchy 
        WHERE depth > 0 
        AND id NOT IN (SELECT DISTINCT parent_id FROM asset WHERE parent_id IS NOT NULL)
        AND name NOT IN ('ControlNode', 'GSM', 'MCB001', 'zone1Main')
    """
    try:
        smoke_results = execute_query(smoke_query)
        SMOKE_MAPPINGS.clear()
        ZONE_CACHE.clear()
        for r in smoke_results:
            full_path = r["zone_path"] or "General"
            if '/' in full_path:
                parts = full_path.split('/')
                zone_name = "/".join(parts[:-1]) 
                asset_name = parts[-1] 
            else:
                zone_name = "General"
                asset_name = r["name"]
            
            SMOKE_MAPPINGS[r["id"]] = asset_name
            ZONE_CACHE[r["id"]] = zone_name
    except Exception as e:
        logger.error(f"Error fetching Smoke mappings: {e}")

    # 4. Populate Unified ASSET_CACHE
    ASSET_CACHE.clear()
    ASSET_CACHE.update(COLDROOM_MAPPINGS)
    ASSET_CACHE.update(TANK_MAPPINGS)
    ASSET_CACHE.update(SMOKE_MAPPINGS)
    
    logger.info(f"Mappings Loaded: {len(COLDROOM_MAPPINGS)} ColdRooms, {len(TANK_MAPPINGS)} Tanks, {len(SMOKE_MAPPINGS)} Smoke Alarms.")


def get_asset_name(asset_id):
    """Returns the user-friendly name for a given hash ID."""
    if not ASSET_CACHE: load_dynamic_mappings()
    return ASSET_CACHE.get(asset_id, asset_id)

def get_asset_zone(asset_id):
    """Returns the zone hierarchy (for smoke alarms)."""
    if not ZONE_CACHE: load_dynamic_mappings()
    return ZONE_CACHE.get(asset_id, "Unknown")

def get_asset_id(name):
    """Returns the hash ID for a given name."""
    if not ASSET_CACHE: load_dynamic_mappings()
    search_name = name.lower().strip().replace(" ", "")
    
    # 1. Exact Match via Unified Cache
    for aid, n in ASSET_CACHE.items():
        if n.lower().replace(" ", "") == search_name:
            return aid
            
    # 2. Fuzzy/Alias Matching for Tanks
    if 'tank' in search_name:
        for aid, n in TANK_MAPPINGS.items():
            db_name_clean = n.lower().replace(" ", "")
            if 'physical' in db_name_clean:
                import re
                u_match = re.search(r'tank(\d+)', search_name)
                db_match = re.search(r'physical(\d+)', db_name_clean)
                if u_match and db_match and int(u_match.group(1)) == int(db_match.group(1)):
                    return aid
                
    return None


# --- Industrial Data Fetching Layer ---

def fetch_coldroom_data(asset_id: str, days: int = 7):
    """ Dedicated Coldroom Fetcher: Retrieves temperature/humidity metrics. """
    query = f"""
        SELECT 
            (d.value->>'Temperature')::float AS temperature, 
            (d.value->>'Humidity')::float AS humidity,
            (d.value->>'Timestamp')::timestamp AS sensor_timestamp
        FROM asset_datapoint d
        WHERE d.entity_id = %s
        AND d.timestamp >= NOW() - INTERVAL '{days} days'
        AND d.attribute_name = 'data'
        ORDER BY d.timestamp ASC
    """
    try:
        results = execute_query(query, (asset_id,))
        if not results: return None
        
        df = pd.DataFrame(results)
        df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
        logger.info(f"Industrial Fetch: Successfully retrieved {len(df)} coldroom data rows for {asset_id}.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch coldroom data for {asset_id}: {e}")
        return None

def fetch_tank_data(asset_id: str, hours: int = 3):
    """ Dedicated Tank Fetcher: Retrieves level metrics for refinery assets. """
    query = f"""
        SELECT 
            (d.value->>'TankOilLevelInFeet001')::float AS level_feet,
            (d.value->>'Timestamp')::timestamp AS sensor_timestamp
        FROM asset_datapoint d
        WHERE d.entity_id = %s
        AND d.timestamp >= NOW() - INTERVAL '{hours} hours'
        AND d.attribute_name = 'data'
        ORDER BY d.timestamp ASC
    """
    try:
        results = execute_query(query, (asset_id,))
        if not results: return None
        
        df = pd.DataFrame(results)
        df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
        logger.info(f"Industrial Fetch: Successfully retrieved {len(df)} tank data rows for {asset_id}.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch tank data for {asset_id}: {e}")
        return None

def fetch_smoke_data(asset_id: str, days: int = 30):
    """ Retrieves smoke alarm metrics (temp, humi, battery). """
    query = f"""
        SELECT 
            d.attribute_name AS sensor_node,
            (d.value->>'temp')::float AS temp, 
            (d.value->>'humi')::float AS humi,
            (d.value->>'bat_voltage')::float AS bat_voltage,
            (d.value->>'bat_percent')::float AS bat_percent,
            (d.value->>'warn')::text AS warn,
            d.timestamp AS sensor_timestamp
        FROM asset_datapoint d
        WHERE d.entity_id = %s
        AND d.timestamp >= NOW() - INTERVAL '{days} days'
        AND d.attribute_name ILIKE 'sensor%%'
        ORDER BY d.timestamp ASC
    """
    try:
        results = execute_query(query, (asset_id,))
        if not results: return None
        df = pd.DataFrame(results)
        df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
        logger.info(f"Industrial Fetch: Retrieved {len(df)} smoke rows for {asset_id}.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch smoke data: {e}")
        return None
