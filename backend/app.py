import os
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
from database.coldroom_db import fetch_coldroom_data
from database.tank_db import fetch_tank_data
from train.coldroom.train_coldroom import train_coldroom
from train.tanks_1_6.train_tanks_1_6 import train_tank_1_6
from train.tanks_7_13.train_tanks_7_13 import train_tank_7_13
from status_evaluator import evaluate_coldroom_status, evaluate_tank_status
import re
import schedule
import time
import threading

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(DATA_DIR, "reports")

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIME_STEPS = 30
MAX_FT_DEFAULT = 25.0
SPIKE_THRESHOLD = 1.5

TANK_MAX_LEVELS = {
    'physical_07': 15.0,
    'physical_11': 50.0,
    'physical_12': 50.0,
    'physical_13': 50.0
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def slugify(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def is_peak_hour(dt):
    # Peak Hours: 10:00 to 18:00
    return 10 <= dt.hour <= 18

def get_time_flags(dt):
    is_night = (dt.hour >= 22 or dt.hour <= 5)
    is_weekend = (dt.weekday() >= 5)
    is_peak = is_peak_hour(dt)
    return is_night, is_weekend, is_peak

def process_coldrooms(df):
    logger.info("Processing Coldrooms (Inference)...")
    final_results = []
    if df is None: return final_results

    for name, group in df.groupby('coldroom_name'):
        slug = slugify(name)
        model_dir = os.path.join(MODEL_DIR, "coldroom", slug)
        csv_path = os.path.join(DATA_DIR, "coldroom", slug, "data.csv")
        
        # 1. Train Model
        print(f"Now training the model for {name}...")
        ensure_dir(f"data/coldroom/{slug}")
        ensure_dir(model_dir)
        group.to_csv(csv_path, index=False)
        stats = train_coldroom(csv_path, model_dir)
        
        if not stats: 
            print(f"Failed to train model for {name}")
            continue
        
        # Load Artifacts back for prediction
        model = load_model(os.path.join(model_dir, 'model.h5'), compile=False)
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        threshold = stats['threshold']
        
        # 2. Prediction on full dataset (Predict)
        group = group.sort_values('sensor_timestamp').copy()
        group = group.dropna(subset=['temperature', 'humidity'])
        
        if len(group) < TIME_STEPS:
            print(f"{name} has insufficient valid data for sequence creation.")
            final_results.append({
                "coldroom_name": name,
                "anomaly": 0,
                "status": "Buffering",
                "level": "Normal",
                "anomalies": []
            })
            continue
            
        # Feature Engineering (Matching User Snippet)
        group["temp_diff"] = group["temperature"].diff()
        group["hum_diff"] = group["humidity"].diff()
        group["rolling_mean"] = group["temperature"].rolling(12).mean()
        group["rolling_std"] = group["temperature"].rolling(12).std()
        group = group.fillna(0)
        
        features = ["temperature", "humidity", "temp_diff", "rolling_mean", "rolling_std"]
        X_scaled = scaler.transform(group[features].values)
        
        # Sequence Creation
        sequences = [
            X_scaled[i : i + TIME_STEPS]
            for i in range(len(X_scaled) - TIME_STEPS + 1)
        ]
        X_room = np.array(sequences)
        
        # Model Prediction
        X_pred = model.predict(X_room, verbose=0)
        mse_per_sequence = np.mean(np.power(X_room - X_pred, 2), axis=(1, 2))
        
        anomaly_list = []
        
        # Iterate through all sequences to detect historical anomalies
        for i, mse in enumerate(mse_per_sequence):
            actual_row = group.iloc[i + TIME_STEPS - 1]
            temp = float(actual_row["temperature"])
            hum = float(actual_row["humidity"])
            temp_diff = abs(actual_row["temp_diff"])
            hum_diff = abs(actual_row["hum_diff"])
            ts = actual_row["sensor_timestamp"]

            # SENSOR FAULT
            if temp == 0 and hum == 0:
                anomaly_list.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature": temp,
                    "humidity": hum,
                    "type": "sensor_fault"
                })
                continue

            # SUDDEN SPIKE
            if temp_diff > 3 or hum_diff > 10:
                anomaly_list.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature": temp,
                    "humidity": hum,
                    "type": "sudden_spike"
                })
                continue

            # LSTM ANOMALY
            if mse > threshold:
                # continuous trend ignore logic from user snippet
                if temp_diff < 0.5 and hum_diff < 2:
                    continue
                
                anomaly_list.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature": temp,
                    "humidity": hum,
                    "type": "model_anomaly",
                    "mse": float(mse)
                })

        # CLEAN OUTPUT
        anomaly_list = sorted(anomaly_list, key=lambda x: x["timestamp"], reverse=True)
        anomaly_status = 1 if len(anomaly_list) > 0 else 0
        highest_error = float(np.max(mse_per_sequence)) if len(mse_per_sequence) > 0 else 0
        
        # Evaluation for dashboard severity (using latest context)
        latest_row = group.iloc[-1]
        latest_ts = latest_row["sensor_timestamp"]
        is_night, is_weekend, is_peak = get_time_flags(latest_ts)
        latest_mse = mse_per_sequence[-1] if len(mse_per_sequence) > 0 else 0
        
        intense, level = evaluate_coldroom_status(
            anomaly_status, float(latest_row["temperature"]), is_night, is_weekend, is_peak, highest_error, threshold
        )
        
        # Console output
        status_text = "Anomaly Detected" if anomaly_status else "Normal"
        print(f"Model Output for {name}: MSE={latest_mse:.6f}, Threshold={threshold:.6f}")
        print(f"Evaluation: Status={status_text}, Level={level}, Anomalies={len(anomaly_list)}")
        if len(anomaly_list) > 0:
            print(f"Recent Anomalies: {anomaly_list[:2]}") # show top 2
        print("-" * 20)

        final_results.append({
            "name": name,
            "anomaly": anomaly_status,
            "status": status_text,
            "level": level,
            "intense": intense,
            "threshold": round(float(threshold), 6),
            "reconstruction_error": round(highest_error, 6),
            "anomaly_count": len(anomaly_list),
            "latest_temp": float(latest_row["temperature"]),
            "anomalies": anomaly_list
        })
    return final_results

def process_tanks(df):
    logger.info("Processing Tanks (Inference)...")
    results = []
    if df is None: return results

    for tank_id, group in df.groupby('tank_id'):
        try:
            tank_num = int(re.search(r'\d+', tank_id).group())
        except: continue
        
        # Folder Mapping
        if 1 <= tank_num <= 6:
            parent_model = "tanks_1_to_6"
            train_func = train_tank_1_6
        else:
            parent_model = "tanks_7_to_13"
            train_func = train_tank_7_13
            
        model_dir = os.path.join(MODEL_DIR, parent_model, f"tank{tank_num}")
        csv_path = os.path.join(DATA_DIR, "tanks", parent_model, f"tank{tank_num}", "data.csv")
        
        # 1. Train Model
        print(f"Now training the model for {tank_id}...")
        ensure_dir(os.path.dirname(csv_path))
        ensure_dir(model_dir)
        group.to_csv(csv_path, index=False)
        stats = train_func(csv_path, model_dir)
        
        if not stats: 
            print(f"Failed to train model for {tank_id}")
            continue
        
        # Load Artifacts
        model = load_model(os.path.join(model_dir, 'model.h5'), compile=False)
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        config = joblib.load(os.path.join(model_dir, 'config.joblib'))
        
        threshold = config["threshold"]
        features_list = config["features"]
        max_ft = TANK_MAX_LEVELS.get(tank_id.lower().replace(' ', '_'), MAX_FT_DEFAULT)
        
        # 2. Prediction (Predict)
        group = group.sort_values("sensor_timestamp").copy()
        group = group.dropna(subset=['level_feet'])
        
        if len(group) < TIME_STEPS:
            print(f"{tank_id} has insufficient data ({len(group)}/{TIME_STEPS}) for window analysis.")
            results.append({
                "name": tank_id,
                "anomaly": 0,
                "status": "Buffering",
                "level": "Normal",
                "intense": 0,
                "threshold": round(float(threshold), 6),
                "reconstruction_error": 0.0,
                "anomaly_count": 0,
                "anomalies": []
            })
            continue
            
        # Feature Engineering (Exactly matching training notebook)
        group["headspace"] = max_ft - group["level_feet"]
        group["fill_pct"] = (group["level_feet"] / max_ft) * 100
        group["roc"] = group["level_feet"].diff()
        group["roc_abs"] = group["roc"].abs()
        group["accel"] = group["roc"].diff()
        group["roll_mean"] = group["level_feet"].rolling(60, min_periods=1).mean()
        group["roll_std"] = group["level_feet"].rolling(60, min_periods=1).std().fillna(0)
        group["roll_range"] = (
            group["level_feet"].rolling(60, min_periods=1).max() - 
            group["level_feet"].rolling(60, min_periods=1).min()
        )
        group["dev_from_mean"] = group["level_feet"] - group["roll_mean"]
        
        mu = group["level_feet"].mean()
        sig = group["level_feet"].std() + 1e-9
        group["z_score"] = (group["level_feet"] - mu) / sig
        
        group["hour"] = group["sensor_timestamp"].dt.hour
        group["minute"] = group["sensor_timestamp"].dt.minute
        group["day_of_week"] = group["sensor_timestamp"].dt.dayofweek
        group["is_night"] = ((group["hour"] >= 22) | (group["hour"] <= 5)).astype(int)
        
        group = group.fillna(0)
        
        # Scale and Sequence
        X_scaled = scaler.transform(group[features_list].values)
        sequences = [
            X_scaled[i : i + TIME_STEPS]
            for i in range(len(X_scaled) - TIME_STEPS + 1)
        ]
        X_tank = np.array(sequences)
        
        # Model Prediction
        X_pred = model.predict(X_tank, verbose=0)
        mae_per_sequence = np.mean(np.abs(X_tank - X_pred), axis=(1, 2))
        
        # Comprehensive Anomaly Detection
        anomaly_list = []
        for i, mae in enumerate(mae_per_sequence):
            row = group.iloc[i + TIME_STEPS - 1]
            level_val = float(row["level_feet"])
            roc_abs = float(row["roc_abs"])
            ts_point = row["sensor_timestamp"]
            
            # 1. Sensor Fault
            if level_val == 0.0:
                anomaly_list.append({
                    "timestamp": ts_point.strftime("%Y-%m-%d %H:%M:%S"),
                    "level_feet": level_val,
                    "mse": round(float(mae), 6),
                    "type": "sensor_fault"
                })
                continue
            
            # 2. Impossible Value
            if level_val < 0 or level_val > 30: # 30 is the safety cap from snippet
                anomaly_list.append({
                    "timestamp": ts_point.strftime("%Y-%m-%d %H:%M:%S"),
                    "level_feet": level_val,
                    "mse": round(float(mae), 6),
                    "type": "impossible_value"
                })
                continue
                
            # 3. Sudden Spike
            if roc_abs > SPIKE_THRESHOLD:
                anomaly_list.append({
                    "timestamp": ts_point.strftime("%Y-%m-%d %H:%M:%S"),
                    "level_feet": level_val,
                    "mse": round(float(mae), 6),
                    "type": "sudden_spike"
                })
                continue
                
            # 4. LSTM Anomalies
            if mae > threshold:
                if mae > threshold * 10:
                    anomaly_list.append({
                        "timestamp": ts_point.strftime("%Y-%m-%d %H:%M:%S"),
                        "level_feet": level_val,
                        "mse": round(float(mae), 6),
                        "type": "critical_anomaly"
                    })
                elif roc_abs >= 0.05: # Ignore slow drift
                    anomaly_list.append({
                        "timestamp": ts_point.strftime("%Y-%m-%d %H:%M:%S"),
                        "level_feet": level_val,
                        "mse": round(float(mae), 6),
                        "type": "model_anomaly"
                    })

        # Cleanup and evaluation
        anomaly_list = sorted(anomaly_list, key=lambda x: x["timestamp"], reverse=True)
        anomaly_detected = len(anomaly_list) > 0
        
        ts_latest = group["sensor_timestamp"].iloc[-1]
        is_night_latest = ((ts_latest.hour >= 22) | (ts_latest.hour <= 5))
        
        intense, level_status = evaluate_tank_status(anomaly_detected, anomaly_list, is_night_latest)
        
        highest_error = float(np.max(mae_per_sequence)) if len(mae_per_sequence) > 0 else 0.0
        curr_level = float(group["level_feet"].iloc[-1])

        # Show Output
        status_text = "Anomaly" if anomaly_detected else "Normal"
        print(f"Model Output for {tank_id}: MaxMAE={highest_error:.6f}, Threshold={threshold:.6f}")
        print(f"Evaluation: Status={status_text}, Level={level_status}, Intensity={intense}")
        if anomaly_detected:
            print(f"Detected Anomalies: {len(anomaly_list)} events found in window.")
            if len(anomaly_list) > 0:
                print(f"Latest Anomalies: {anomaly_list[:2]}")
        print("-" * 20)

        results.append({
            "name": tank_id,
            "anomaly": 1 if anomaly_detected else 0,
            "intense": intense,
            "level": level_status,
            "latest_level": curr_level,
            "mae": highest_error,
            "threshold": threshold,
            "anomaly_count": len(anomaly_list),
            "anomalies": anomaly_list
        })
    return results

def save_pipeline_results(cr_results, tank_results):
    """
    Saves the results of the pipeline to CSV files in the reports directory.
    Appends data if file exists.
    """
    ensure_dir(REPORT_DIR)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save Coldroom Results
    if cr_results:
        cr_df = pd.DataFrame(cr_results)
        cr_df['report_timestamp'] = now
        cr_path = os.path.join(REPORT_DIR, "coldroom_reports.csv")
        # Handle 'anomalies' list column (flatten for CSV)
        if 'anomalies' in cr_df.columns:
            cr_df['anomalies'] = cr_df['anomalies'].apply(lambda x: str(x))
        
        mode = 'a' if os.path.exists(cr_path) else 'w'
        header = not os.path.exists(cr_path)
        cr_df.to_csv(cr_path, mode=mode, header=header, index=False)
        logger.info(f"Report: Coldroom results saved to {cr_path}")

    # Save Tank Results
    if tank_results:
        tank_df = pd.DataFrame(tank_results)
        tank_df['report_timestamp'] = now
        tank_path = os.path.join(REPORT_DIR, "tank_reports.csv")
        # Handle 'anomalies' list column
        if 'anomalies' in tank_df.columns:
            tank_df['anomalies'] = tank_df['anomalies'].apply(lambda x: str(x))

        mode = 'a' if os.path.exists(tank_path) else 'w'
        header = not os.path.exists(tank_path)
        tank_df.to_csv(tank_path, mode=mode, header=header, index=False)
        logger.info(f"Report: Tank results saved to {tank_path}")

def main():
    logger.info("Executing Primary Anomaly Pipeline (Fetch -> Train -> Show Result)...")
    
    # FETCH
    logger.info("Fetching 7-day data...")
    cr_data = fetch_coldroom_data(days=7)
    tank_data = fetch_tank_data(days=7)

    # DATA RETRIEVAL REPORT (PRE-TRAINING)
    if cr_data is not None:
        for name, group in cr_data.groupby('coldroom_name'):
            print(f"{name} data retrieved: {len(group)} rows")
    
    if tank_data is not None:
        for tank_id, group in tank_data.groupby('tank_id'):
            print(f"{tank_id} data retrieved: {len(group)} rows")
            
    # TRAIN & PREDICT
    cr_results = process_coldrooms(cr_data)
    tank_results = process_tanks(tank_data)
    
    # SHOW RESULT
    print("\n" + "="*60)
    print(" ANOMALY DETECTION PIPELINE SUMMARY REPORT ")
    print("="*60)
    
    print("\nCOLDROOMS:")
    print(f"{'Asset Name':<25} | {'Temp':<8} | {'Status':<15} | {'Level':<10}")
    print("-"*60)
    for r in cr_results:
        print(f"{r['name']:<25} | {r.get('latest_temp', 0):<8.2f} | {'Anomaly' if r['anomaly'] else 'Normal':<15} | {r['level']:<10}")
        
    print("\nREFINERY TANKS:")
    print(f"{'Asset Name':<25} | {'Level':<8} | {'Status':<15} | {'Level':<10}")
    print("-"*60)
    for r in tank_results:
        print(f"{r['name']:<25} | {r.get('latest_level', 0):<8.2f} | {'Anomaly' if r['anomaly'] else 'Normal':<15} | {r['level']:<10}")
        
    print("\n" + "="*60)
    
    # SAVE TO CSV
    save_pipeline_results(cr_results, tank_results)
    
    logger.info("Pipeline Execution Complete.")

def run_scheduler():
    """Background loop for the 3-hour automation."""
    logger.info("Scheduler initialized: Running every 3 hours.")
    while True:
        schedule.run_pending()
        time.sleep(10) # Check every 10 seconds

if __name__ == "__main__":
    # 1. Run once immediately on startup
    main()
    
    # 2. Schedule for every 3 hours
    schedule.every(3).hours.do(main)
    
    # 3. Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # 4. Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Industrial Pipeline: Shutdown requested.")
