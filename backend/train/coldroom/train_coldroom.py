import pandas as pd
import numpy as np
import os
import logging
from ..train_utils import create_sequences, build_lstm_autoencoder, save_artifact
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

def train_coldroom(data_path, model_dir):
    """
    Trains/Retrains Coldroom LSTM Autoencoder on the 7-day window.
    Standardized to use config.pkl and scalar.pkl.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None
    
    try:
        df = pd.read_csv(data_path)
        df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
        df = df.sort_values('sensor_timestamp')
        
        # Drop NaNs in core features
        df = df.dropna(subset=['temperature', 'humidity'])
        if df.empty:
            logger.warning(f"No valid data (NaNs dropped) for {data_path}")
            return None
        
        # Feature Engineering (Matching Notebook Logic)
        df['temp_diff'] = df['temperature'].diff().fillna(0)
        df['rolling_mean'] = df['temperature'].rolling(window=12).mean().fillna(df['temperature'].mean())
        df['rolling_std'] = df['temperature'].rolling(window=12).std().fillna(0)
        
        features = ['temperature', 'humidity', 'temp_diff', 'rolling_mean', 'rolling_std']
        X = df[features].values
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Sequencing
        TIME_STEPS = 30
        X_seq = create_sequences(X_scaled, TIME_STEPS)
        
        if len(X_seq) == 0:
            logger.warning(f"Not enough data for sequence creation: {data_path}")
            return None
        
        # Build & Train Model
        model = build_lstm_autoencoder((TIME_STEPS, X_scaled.shape[1]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(
            X_seq, X_seq, 
            epochs=40, 
            batch_size=32, 
            validation_split=0.1, 
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate reconstruction error
        reconstructions = model.predict(X_seq)
        mse = np.mean(np.power(X_seq - reconstructions, 2), axis=(1, 2))
        
        # Threshold: 95th percentile
        threshold = np.percentile(mse, 95)
        
        # Save Artifacts with standardized naming
        save_artifact(model, os.path.join(model_dir, 'model.h5'))
        save_artifact(scaler, os.path.join(model_dir, 'scalar.pkl'))
        
        config = {
            "threshold": float(threshold),
            "features": features,
            "seq_len": TIME_STEPS,
            "asset_type": "coldroom"
        }
        save_artifact(config, os.path.join(model_dir, 'config.pkl'))
        
        logger.info(f"ColdRoom Training complete. config.pkl updated in {model_dir}")
        
        return {
            "mse": float(np.mean(mse)),
            "threshold": float(threshold),
            "last_mse": float(mse[-1])
        }
    except Exception as e:
        logger.error(f"Training failed for {data_path}: {e}")
        return None
