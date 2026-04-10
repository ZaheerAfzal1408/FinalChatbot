import pandas as pd
import numpy as np
import os
import logging
from ..train_utils import create_sequences, build_lstm_autoencoder, save_artifact
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

def train_tank_1_6(data_path, model_dir):
    """
    Trains/Retrains Tank LSTM Autoencoder for groups 1-6.
    Uses corrected level logic: current_level = MaxFt - raw_sensor.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None
    
    try:
        df = pd.read_csv(data_path)
        df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
        df = df.sort_values('sensor_timestamp')
        df = df.dropna(subset=['level_feet'])
        if df.empty:
            logger.warning(f"No valid data (NaNs dropped) for {data_path}")
            return None
        
        MAX_FT = 25.0 # Default for 1-6
        if 'tank3' in data_path: MAX_FT = 25.0 # Explicitly set if needed
        
        # ─── CORRECTED LEVEL LOGIC ───
        # Based on user request: actual_level = MaxFt - raw_sensor_value
        df['actual_level'] = MAX_FT - df['level_feet']
        
        # Feature Engineering based on ACTUAL Level
        df['fill_pct'] = (df['actual_level'] / MAX_FT) * 100
        df['roc'] = df['actual_level'].diff().fillna(0)
        df['roc_abs'] = df['roc'].abs()
        df['accel'] = df['roc'].diff().fillna(0)
        df['roll_mean'] = df['actual_level'].rolling(60, min_periods=1).mean().fillna(df['actual_level'].mean())
        df['roll_std'] = df['actual_level'].rolling(60, min_periods=1).std().fillna(0)
        df['roll_range'] = df['actual_level'].rolling(60, min_periods=1).max() - df['actual_level'].rolling(60, min_periods=1).min()
        df['dev_from_mean'] = df['actual_level'] - df['roll_mean']
        
        mu = df['actual_level'].mean()
        sig = df['actual_level'].std() + 1e-9
        df['z_score'] = (df['actual_level'] - mu) / sig
        
        # Time Features
        df['hour'] = df['sensor_timestamp'].dt.hour
        df['minute'] = df['sensor_timestamp'].dt.minute
        df['day_of_week'] = df['sensor_timestamp'].dt.dayofweek
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        df = df.fillna(0)
        
        features = ["actual_level", "fill_pct", "roc", "roc_abs", "accel", "roll_mean", "roll_std", "roll_range", "dev_from_mean", "z_score", "hour", "minute", "day_of_week", "is_night"]
        X = df[features].values
        
        # Scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Sequencing
        TIME_STEPS = 30
        X_seq = create_sequences(X_scaled, TIME_STEPS)
        
        if len(X_seq) == 0:
            logger.warning(f"Not enough data for sequence: {data_path}")
            return None
        
        # Build & Train Model (Boosting parameters for stability)
        model = build_lstm_autoencoder((TIME_STEPS, X_scaled.shape[1]))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(
            X_seq, X_seq, 
            epochs=40, 
            batch_size=64, 
            validation_split=0.1, 
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate reconstruction error
        reconstructions = model.predict(X_seq)
        mse = np.mean(np.power(X_seq - reconstructions, 2), axis=(1, 2))
        
        # Threshold: 95th percentile for robustness
        threshold = np.percentile(mse, 95)
        
        # Save Artifacts with standardized naming
        save_artifact(model, os.path.join(model_dir, 'model.h5'))
        save_artifact(scaler, os.path.join(model_dir, 'scalar.pkl'))
        
        config = {
            "threshold": float(threshold),
            "features": features,
            "seq_len": TIME_STEPS,
            "max_ft": MAX_FT,
            "asset_type": "tank"
        }
        save_artifact(config, os.path.join(model_dir, 'config.pkl'))
        
        logger.info(f"Tank 1-6 Training complete for {data_path}. Threshold: {threshold:.4f}")
        
        return {
            "mse": float(np.mean(mse)),
            "threshold": float(threshold),
            "last_mse": float(mse[-1])
        }
    except Exception as e:
        logger.error(f"Tank 1-6 Training failed: {e}")
        return None
