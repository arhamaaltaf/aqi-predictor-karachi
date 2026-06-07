"""
Feature Pipeline for Karachi AQI Predictor

This script:
1. Fetches latest weather + AQI data from Open-Meteo APIs
2. Engineers features (time, lag, rolling, derived)
3. Stores features in MongoDB

Run:
- Manually: python -m src.feature_pipeline
- Automated: Via GitHub Actions every hour
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.database import MongoDBHandler
from src.config import LATITUDE, LONGITUDE, LAG_FEATURES, ROLLING_WINDOWS, OPENWEATHER_API_KEY
import time


class FeaturePipeline:
    """Handle feature engineering and storage"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.db = MongoDBHandler()
        self.lat = LATITUDE
        self.lon = LONGITUDE
    
    # ========================================
    # DATA FETCHING
    # ========================================
    
    def fetch_weather_data(self, days=1):
        """Fetch weather data from Open-Meteo"""
        print(f"🌤️  Fetching weather data for last {days} day(s)...")
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'hourly': [
                'temperature_2m',
                'relative_humidity_2m',
                'wind_speed_10m',
                'wind_direction_10m',
                'cloud_cover'
            ],
            'timezone': 'auto',
            'past_days': days
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame({
                'datetime': data['hourly']['time'],
                'temperature': data['hourly']['temperature_2m'],
                'humidity': data['hourly']['relative_humidity_2m'],
                'wind_speed': data['hourly']['wind_speed_10m'],
                'wind_direction': data['hourly']['wind_direction_10m'],
                'cloud_cover': data['hourly']['cloud_cover']
            })
            
            print(f"   ✅ Fetched {len(df)} weather records")
            return df
            
        except Exception as e:
            print(f"   ❌ Error fetching weather data: {e}")
            return None
    
    def fetch_aqi_data(self, days=1):
        """Fetch AQI data from OpenWeather API (matches training dataset source)"""
        print(f"💨 Fetching AQI data from OpenWeather for last {days} day(s)...")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'start': start_timestamp,
            'end': end_timestamp,
            'appid': OPENWEATHER_API_KEY
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract hourly records
            records = []
            for hour_data in data.get('list', []):
                record = {
                    'datetime': datetime.fromtimestamp(hour_data['dt']).strftime('%Y-%m-%dT%H:%M'),
                    'aqi': hour_data['main']['aqi'],  # Direct AQI (1-5)
                    'co': hour_data['components']['co'],
                    'no': hour_data['components'].get('no', 0),
                    'no2': hour_data['components']['no2'],
                    'o3': hour_data['components']['o3'],
                    'so2': hour_data['components']['so2'],
                    'pm2_5': hour_data['components']['pm2_5'],
                    'pm10': hour_data['components']['pm10'],
                    'nh3': hour_data['components'].get('nh3', 0)
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            
            print(f"   ✅ Fetched {len(df)} AQI records from OpenWeather")
            print(f"   📊 AQI distribution: {df['aqi'].value_counts().sort_index().to_dict()}")
            return df
            
        except Exception as e:
            print(f"   ❌ Error fetching AQI data from OpenWeather: {e}")
            return None
    
    # ========================================
    # FEATURE ENGINEERING
    # ========================================
    
    def create_time_features(self, df):
        """Create time-based features (only cyclical + weekend to match 30-feature dataset)"""
        print("🕐 Creating time features...")
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Create intermediate features (will drop later)
        hour = df['datetime'].dt.hour
        day_of_week = df['datetime'].dt.dayofweek
        month = df['datetime'].dt.month
        
        # Keep only cyclical features + is_weekend
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        df['is_weekend'] = day_of_week.isin([5, 6]).astype(int)
        
        print("   ✅ Created 7 time features (cyclical + weekend)")
        return df
    
    def create_lag_features(self, df):
        """Create lag features (only AQI lags to match 30-feature dataset)"""
        print("⏮️  Creating lag features...")
        
        # AQI lag features only (24h, 48h, 72h)
        for lag in LAG_FEATURES:
            df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)
        
        print(f"   ✅ Created {len(LAG_FEATURES)} AQI lag features")
        return df
    
    def create_rolling_features(self, df):
        """Create rolling window features (using AQI as in training data)"""
        print("📊 Creating rolling features...")
        
        # AQI rolling features (matches training dataset)
        for window in ROLLING_WINDOWS:
            df[f'aqi_rolling_mean_{window}h'] = df['aqi'].rolling(window).mean()
            df[f'aqi_rolling_std_{window}h'] = df['aqi'].rolling(window).std()
            df[f'aqi_rolling_min_{window}h'] = df['aqi'].rolling(window).min()
            df[f'aqi_rolling_max_{window}h'] = df['aqi'].rolling(window).max()
        
        print(f"   ✅ Created {len(ROLLING_WINDOWS) * 4} rolling features")
        return df
    
    def create_derived_features(self, df):
        """Create derived features (matches 30-feature dataset)"""
        print("🔧 Creating derived features...")
        
        # AQI rate of change (only 24h)
        df['aqi_change_24h'] = df['aqi'].diff(24)
        
        # Interactions (matches 30-feature dataset)
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['wind_pollution'] = df['wind_speed'] * df['aqi']
        
        # Note: AQI column already exists from OpenWeather API (direct values 1-5)
        
        print("   ✅ Created 3 derived features")
        return df
    
    # ========================================
    # PIPELINE EXECUTION
    # ========================================
    
    def run_hourly_update(self):
        """
        Run hourly feature update
        
        This is called by GitHub Actions every hour to:
        1. Fetch latest 1 day of data (gets the newest hour)
        2. Get last 72 hours from MongoDB (for lag calculations)
        3. Combine and engineer features
        4. Store only NEW records (typically 1 per hour = 24/day)
        """
        print("\n" + "="*60)
        print("🔄 HOURLY FEATURE UPDATE")
        print("="*60)
        print(f"Time: {datetime.now()}")
        
        try:
            # Fetch latest 1 day - API returns hourly data, we need the newest hour
            # We load last 72h from MongoDB to calculate lag features
            weather_df = self.fetch_weather_data(days=1)
            aqi_df = self.fetch_aqi_data(days=1)
            
            if weather_df is None or aqi_df is None:
                print("❌ Failed to fetch data. Aborting update.")
                return False
            
            # Combine
            print("🔗 Combining datasets...")
            df_new = pd.merge(weather_df, aqi_df, on='datetime', how='inner')
            df_new['datetime'] = pd.to_datetime(df_new['datetime'])
            df_new = df_new.sort_values('datetime').reset_index(drop=True)
            
            # Get ONLY last 72 hours from MongoDB (optimization: don't load ALL data)
            print("📥 Loading last 72 hours from MongoDB...")
            df_existing = self.db.get_latest_features(n_hours=72)
            
            if not df_existing.empty:
                # Find what's the latest timestamp in DB
                latest_db_time = pd.to_datetime(df_existing['datetime'].max())
                print(f"   Latest DB timestamp: {latest_db_time}")
                
                # Combine with existing (for lag/rolling calculations)
                print("🔗 Combining with existing features...")
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['datetime'])
                df_combined = df_combined.sort_values('datetime').reset_index(drop=True)
            else:
                print("ℹ️  No existing features found. Using new data only.")
                df_combined = df_new
                latest_db_time = None
            
            # Engineer features
            df_combined = self.create_time_features(df_combined)
            df_combined = self.create_lag_features(df_combined)
            df_combined = self.create_rolling_features(df_combined)
            df_combined = self.create_derived_features(df_combined)
            
            # Find NEW records only (after latest DB timestamp)
            if latest_db_time is not None:
                df_new_only = df_combined[df_combined['datetime'] > latest_db_time]
                print(f"🔍 Found {len(df_new_only)} records newer than {latest_db_time}")
            else:
                df_new_only = df_combined
                print(f"🔍 No existing data - will store all {len(df_new_only)} records")
            
            # Remove NaN from new records (due to lag features)
            print(f"🧹 Cleaning NaN values from {len(df_new_only)} new records...")
            original_new_count = len(df_new_only)
            df_new_only = df_new_only.dropna()
            nan_removed = original_new_count - len(df_new_only)
            
            if nan_removed > 0:
                print(f"   ⚠️  Removed {nan_removed} records with NaN (insufficient lag history)")
                print(f"   ✅ {len(df_new_only)} clean records ready to store")
            
            if len(df_new_only) > 0:
                print(f"\n💾 Storing {len(df_new_only)} NEW records to MongoDB...")
                print(f"   Expected: ~1 record/hour = ~24 records/day")
                
                # Append new records to MongoDB
                records = df_new_only.to_dict('records')
                for record in records:
                    if not isinstance(record['datetime'], datetime):
                        record['datetime'] = pd.to_datetime(record['datetime'])
                
                self.db.features.insert_many(records)
                print(f"✅ Successfully stored {len(df_new_only)} new records")
                
                # Show what was added
                if len(df_new_only) <= 5:
                    print(f"\n📋 Added records:")
                    for _, row in df_new_only.iterrows():
                        print(f"   • {row['datetime']} - AQI: {row['aqi']}")
            else:
                print("\nℹ️  No new records to store. Database is up to date.")
            
            print("\n" + "="*60)
            print("✅ HOURLY UPDATE COMPLETE!")
            print(f"   Records added this run: {len(df_new_only) if len(df_new_only) > 0 else 0}")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR in hourly update: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_backfill(self, days=180):
        """
        Run backfill for historical data
        
        Use this ONCE to populate MongoDB with historical data
        """
        print("\n" + "="*60)
        print(f"🔄 BACKFILL: {days} DAYS OF HISTORICAL DATA")
        print("="*60)
        
        try:
            # Fetch historical data
            weather_df = self.fetch_weather_data(days=days)
            aqi_df = self.fetch_aqi_data(days=days)
            
            if weather_df is None or aqi_df is None:
                print("❌ Failed to fetch data. Aborting backfill.")
                return False
            
            # Combine
            print("🔗 Combining datasets...")
            df = pd.merge(weather_df, aqi_df, on='datetime', how='inner')
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Engineer features
            df = self.create_time_features(df)
            df = self.create_lag_features(df)
            df = self.create_rolling_features(df)
            df = self.create_derived_features(df)
            
            # Clean
            print("🧹 Cleaning NaN values...")
            df_clean = df.dropna()
            
            print(f"\n📊 Backfill Summary:")
            print(f"   Original records: {len(df):,}")
            print(f"   After cleaning: {len(df_clean):,}")
            print(f"   Date range: {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")
            
            # Store in MongoDB
            print(f"\n💾 Storing {len(df_clean)} records in MongoDB...")
            self.db.insert_features(df_clean)
            
            print("\n" + "="*60)
            print("✅ BACKFILL COMPLETE!")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR in backfill: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close(self):
        """Close database connection"""
        self.db.close()


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    import sys
    
    pipeline = FeaturePipeline()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        # Run backfill
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 180
        success = pipeline.run_backfill(days=days)
    else:
        # Run hourly update (default)
        success = pipeline.run_hourly_update()
    
    pipeline.close()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)