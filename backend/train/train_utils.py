import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import pickle
import logging
import os

logger = logging.getLogger(__name__)

def create_sequences(data, time_steps=30):
    """
    Creates overlapping sequences for LSTM training.
    """
    xs = []
    for i in range(len(data) - time_steps):
        xs.append(data[i:(i + time_steps)])
    return np.array(xs)

def build_lstm_autoencoder(input_shape):
    """
    Standard LSTM Autoencoder Architecture aligned with Model_on_9_ColdRooms.ipynb.
    """
    inputs = Input(shape=input_shape)
    # Encoder
    L1 = LSTM(64, activation='relu', return_sequences=True)(inputs)
    L2 = LSTM(32, activation='relu', return_sequences=False)(L1)
    # Decoder
    L3 = RepeatVector(input_shape[0])(L2)
    L4 = LSTM(32, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(64, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(input_shape[1]))(L5)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def save_artifact(obj, path):
    """
    Saves a model (.h5) or artifact (.pkl) to the specified path.
    Uses pickle for .pkl as per user request.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith('.h5') or path.endswith('.keras'):
        obj.save(path)
    elif path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    else:
        joblib.dump(obj, path)

def load_artifact(path):
    """
    Loads a model or scaler from the specified path.
    """
    if not os.path.exists(path):
        return None
    if path.endswith('.h5') or path.endswith('.keras'):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            return None
    elif path.endswith('.pkl'):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle from {path}: {e}")
            return None
    else:
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading joblib from {path}: {e}")
            return None
