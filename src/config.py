"""
Configuration file for Karachi AQI Predictor
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Streamlit for cloud deployment
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def get_config(key, default=None):
    """Get config from either Streamlit secrets (cloud) or env (local)"""
    # Try Streamlit secrets first (for Streamlit Cloud)
    if HAS_STREAMLIT:
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except (AttributeError, FileNotFoundError):
            pass
    # Fallback to environment variable
    return os.getenv(key, default)

# MongoDB Configuration
MONGODB_URI = get_config('MONGODB_URI')
print(f"DEBUG: My URI is {MONGODB_URI}")
MONGODB_DATABASE = get_config('MONGODB_DATABASE', 'aqi_karachi')

# Location Configuration
CITY_NAME = get_config('CITY_NAME', 'Karachi')
LATITUDE = float(get_config('LATITUDE', 24.8608))
LONGITUDE = float(get_config('LONGITUDE', 67.0104))

# OpenWeather API Configuration (for AQI data)
OPENWEATHER_API_KEY = get_config('OPENWEATHER_API_KEY')

# Model Configuration
PREDICTION_HORIZON = int(get_config('PREDICTION_HORIZON', 72))  # Hours to predict
TRAIN_TEST_SPLIT = float(get_config('TRAIN_TEST_SPLIT', 0.2))
RANDOM_STATE = 42

# Feature Engineering Configuration
# Only use lags >= 24h to avoid data leakage (predicting 72h ahead)
LAG_FEATURES = [24, 48, 72]  # Removed: 1, 2, 3, 6, 12 (data leakage)
ROLLING_WINDOWS = [72]  # Only 72h rolling windows (matches clean dataset)

# MongoDB Collections
COLLECTION_RAW_DATA = 'raw_data'
COLLECTION_FEATURES = 'features'
COLLECTION_PREDICTIONS = 'predictions'
COLLECTION_MODELS = 'models'

# Target variable
TARGET_VARIABLE = 'pm2_5'

# Features to use for training (we'll populate this after data analysis)
FEATURE_COLUMNS = None  # Will be set dynamically

print("✅ Configuration loaded successfully")