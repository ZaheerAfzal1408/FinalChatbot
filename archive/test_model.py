import os
import logging
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from database.coldroom_db import fetch_coldroom_data
from database.tank_db import fetch_tank_data
import json
import re

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIME_STEPS = 30
MAX_FT = 25.0
SPIKE_THRESHOLD = 1.5

TANK_MAX_LEVELS = {
    'physical_07': 15.0,
    'physical_08': 25.0,
    'physical_09': 25.0,
    'physical_11': 50.0,
    'physical_12': 50.0,
    'physical_13': 50.0
}

def slugify(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def normalize_tank_id(tank_id):
    """Aligns Tank IDs with the TANK_MAX_LEVELS keys (physical_01, etc.)."""
    return tank_id.lower().replace(' ', '_')

def process_coldrooms():
    logger.info("Evaluating Coldrooms...")
    df = fetch_coldroom_data(days=1)
    if df is None:
        logger.info("No coldroom data found.")
        return []

    results = []
    for name, group in df.groupby('coldroom_name'):
        slug = slugify(name)
        model_path = f"models/coldroom/{slug}/model.h5"
        scaler_path = f"models/coldroom/{slug}/scaler.joblib"
        threshold_path = f"models/coldroom/{slug}/threshold.joblib"

        if not os.path.exists(model_path):
            continue

        # Load artifacts
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        threshold = joblib.load(threshold_path)

        group = group.sort_values('sensor_timestamp')
        
        # Feature Engineering (Matching User's snippet)
        group["temp_diff"] = group["temperature"].diff().fillna(0)
        group["hum_diff"] = group["humidity"].diff().fillna(0)
        group["rolling_mean"] = group["temperature"].rolling(12).mean().fillna(group["temperature"].mean())
        group["rolling_std"] = group["temperature"].rolling(12).std().fillna(0)

        features = ["temperature", "humidity", "temp_diff", "rolling_mean", "rolling_std"]
        X_scaled = scaler.transform(group[features])

        if len(X_scaled) < TIME_STEPS:
            results.append({"coldroom_name": name, "status": "Buffering"})
            continue

        # Inference
        sequences = [X_scaled[i : i + TIME_STEPS] for i in range(len(X_scaled) - TIME_STEPS + 1)]
        X_room = np.array(sequences)
        X_pred = model.predict(X_room, verbose=0)
        mse_per_sequence = np.mean(np.power(X_room - X_pred, 2), axis=(1, 2))

        anomaly_list = []
        for i, mse in enumerate(mse_per_sequence):
            actual_row = group.iloc[i + TIME_STEPS - 1]
            temp = float(actual_row["temperature"])
            hum = float(actual_row["humidity"])
            temp_diff = abs(actual_row["temp_diff"])
            hum_diff = abs(actual_row["hum_diff"])
            ts = actual_row["sensor_timestamp"].strftime("%Y-%m-%d %H:%M:%S")

            # Detection logic
            if temp == 0 and hum == 0:
                anomaly_list.append({"timestamp": ts, "type": "sensor_fault", "temp": temp})
            elif temp_diff > 3 or hum_diff > 10:
                anomaly_list.append({"timestamp": ts, "type": "sudden_spike", "temp": temp})
            elif mse > threshold:
                if temp_diff < 0.5 and hum_diff < 2: continue # Ignore drift
                anomaly_list.append({"timestamp": ts, "type": "model_anomaly", "temp": temp, "mse": float(mse)})

        results.append({
            "coldroom_name": name,
            "anomaly": anomaly_status,
            "status": "Anomaly Detected" if anomaly_list else "Normal",
            "anomaly_count": len(anomaly_list),
            "latest_temp": float(group["temperature"].iloc[-1]),
            "anomalies": anomaly_list[-3:], # Show last 3
        })
    return results

def process_tanks():
    logger.info("Evaluating Tanks...")
    df = fetch_tank_data(days=1)
    if df is None:
        logger.info("No tank data found.")
        return []

    results = []
    
    # Identify which group the tank belongs to for folder structure
    # My folder structure was models/tanks_1_to_6/tankX/
    for tank_id, group in df.groupby('tank_id'):
        try:
            tank_num = int(re.search(r'\d+', tank_id).group())
        except: continue

        parent_folder = "tanks_1_to_6" if 1 <= tank_num <= 6 else "tanks_7_to_13"
        asset_folder = f"tank{tank_num}"
        base_path = f"models/{parent_folder}/{asset_folder}"
        
        model_path = f"{base_path}/model.h5"
        scaler_path = f"{base_path}/scaler.joblib"
        config_path = f"{base_path}/config.joblib"

        if not os.path.exists(model_path):
            continue

        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        config = joblib.load(config_path)

        threshold = config["threshold"]
        features_list = config["features"]
        max_ft = TANK_MAX_LEVELS.get(normalize_tank_id(tank_id), MAX_FT)

        group = group.sort_values("sensor_timestamp").copy()
        
        # Feature Engineering (Matching User's snippet)
        group["headspace"] = max_ft - group["level_feet"]
        group["fill_pct"] = (group["level_feet"] / max_ft) * 100
        group["roc"] = group["level_feet"].diff()
        group["roc_abs"] = group["roc"].abs()
        group["accel"] = group["roc"].diff()
        group["roll_mean"] = group["level_feet"].rolling(60, min_periods=1).mean()
        group["roll_std"] = group["level_feet"].rolling(60, min_periods=1).std().fillna(0)
        group["roll_range"] = group["level_feet"].rolling(60, min_periods=1).max() - group["level_feet"].rolling(60, min_periods=1).min()
        group["dev_from_mean"] = group["level_feet"] - group["roll_mean"]
        
        mu = group["level_feet"].mean()
        sig = group["level_feet"].std() + 1e-9
        group["z_score"] = (group["level_feet"] - mu) / sig
        
        group["hour"] = group["sensor_timestamp"].dt.hour
        group["minute"] = group["sensor_timestamp"].dt.minute
        group["day_of_week"] = group["sensor_timestamp"].dt.dayofweek
        group["is_night"] = ((group["hour"] >= 22) | (group["hour"] <= 5)).astype(int)
        group = group.fillna(0)

        scaled = scaler.transform(group[features_list])

        if len(scaled) < TIME_STEPS:
            results.append({"tank_id": tank_id, "status": "Buffering"})
            continue

        sequences = [scaled[i : i + TIME_STEPS] for i in range(len(scaled) - TIME_STEPS + 1)]
        X_tank = np.array(sequences)
        X_pred = model.predict(X_tank, verbose=0)
        mae_per_sequence = np.mean(np.abs(X_tank - X_pred), axis=(1, 2))

        anomaly_list = []
        for i, mae in enumerate(mae_per_sequence):
            row = group.iloc[i + TIME_STEPS - 1]
            level = float(row["level_feet"])
            roc_abs = float(row["roc_abs"])
            ts = row["sensor_timestamp"].strftime("%Y-%m-%d %H:%M:%S")

            if level == 0.0:
                anomaly_list.append({"timestamp": ts, "type": "sensor_fault", "level": level})
            elif level < 0 or level > (max_ft + 5):
                anomaly_list.append({"timestamp": ts, "type": "impossible_value", "level": level})
            elif roc_abs > SPIKE_THRESHOLD:
                anomaly_list.append({"timestamp": ts, "type": "sudden_spike", "level": level})
            elif mae > threshold:
                if mae > threshold * 10:
                    anomaly_list.append({"timestamp": ts, "type": "critical_anomaly", "level": level, "mae": float(mae)})
                elif roc_abs >= 0.05:
                    anomaly_list.append({"timestamp": ts, "type": "model_anomaly", "level": level, "mae": float(mae)})

        results.append({
            "tank_id": tank_id,
            "status": "Anomaly Detected" if anomaly_list else "Normal",
            "anomaly_count": len(anomaly_list),
            "latest_level": float(group["level_feet"].iloc[-1]),
            "anomalies": anomaly_list[-3:]
        })
    return results

def main():
    print("--- ANOMALY DETECTION TEST REPORT ---")
    cr_results = process_coldrooms()
    tank_results = process_tanks()

    print("\n[COLDROOMS]")
    for r in cr_results:
        print(f"{r['coldroom_name']}: {r['status']} | Count: {r['anomaly_count']} | Temp: {r['latest_temp']:.2f}")

    print("\n[REFINERY TANKS]")
    for r in tank_results:
        if 'status' in r:
            print(f"{r['tank_id']}: {r['status']} | Count: {r.get('anomaly_count', 0)} | Level: {r.get('latest_level', 0):.2f}")

if __name__ == "__main__":
    main()
