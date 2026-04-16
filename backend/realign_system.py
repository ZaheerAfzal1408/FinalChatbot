import os
import sys
import logging
import shutil

# Local imports from unified backend structure
import core.asset_mapping as am
import specialists.tools_industrial as ti
import specialists.tools_smoke as ts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealignSystem")

def main():
    logger.info("Starting Industrial System Realignment (Unified Backend)...")
    
    # 1. Targeting unified models folder
    backend_models = ti.MODEL_BASE_DIR 
    logger.info(f"Targeting models folder: {backend_models}")
    
    # 2. Fresh Fetch/Mapping
    am.load_dynamic_mappings()
    
    # 3. Force Retrain All ColdRooms
    logger.info("--- Retraining ColdRooms ---")
    for aid, name in am.COLDROOM_MAPPINGS.items():
        logger.info(f"Realignment: Processing {name}...")
        dest = os.path.join(backend_models, "coldroom", ti.slugify(name))
        if os.path.exists(dest): shutil.rmtree(dest)
        res = ti.analyze_coldroom(name)
        if isinstance(res, dict):
            logger.info(f"SUCCESS: {name} retrained.")
        else:
            logger.error(f"FAILURE: {name} -> {res}")

    # 4. Force Retrain All Tanks
    logger.info("--- Retraining Refinery Tanks ---")
    for aid, name in am.TANK_MAPPINGS.items():
        logger.info(f"Realignment: Processing {name}...")
        import re
        tank_num_match = re.search(r'\d+', name)
        tank_num = int(tank_num_match.group()) if tank_num_match else 0
        group_name = "tanks_1_to_6" if 1 <= tank_num <= 6 else "tanks_7_to_13"
        dest = os.path.join(backend_models, group_name, f"tank{tank_num}")
        if os.path.exists(dest): shutil.rmtree(dest)
        res = ti.analyze_tank(name)
        if isinstance(res, dict):
            logger.info(f"SUCCESS: {name} retrained.")
        else:
            logger.error(f"FAILURE: {name} -> {res}")

    # 5. Force Retrain All Smoke Alarms
    logger.info("--- Retraining Smoke Alarm System ---")
    for aid, name in am.SMOKE_MAPPINGS.items():
        logger.info(f"Realignment: Processing Safety Sensor: {name}...")
        zone = am.get_asset_zone(aid)
        slug = ti.slugify(name)
        dest = os.path.join(ts.SMOKE_MODEL_DIR, zone, slug)
        
        if os.path.exists(dest): shutil.rmtree(dest)
        res = ts.analyze_smoke_incident(name, force_retrain=True)
        if isinstance(res, list): # Smoke reports come as a list of sensors
            logger.info(f"SUCCESS: {name} ({zone}) network retrained.")
        else:
            logger.error(f"FAILURE: {name} -> {res}")

    logger.info("System Realignment Complete.")

if __name__ == "__main__":
    main()
