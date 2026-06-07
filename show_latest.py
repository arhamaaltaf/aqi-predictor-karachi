"""
Show the absolute latest record in MongoDB
"""
from src.database import MongoDBHandler
import json
from datetime import datetime

print("🔍 Fetching the latest record from MongoDB...\n")

db = MongoDBHandler()

# Get all features and sort by datetime
all_features = db.get_features(limit=None)

if all_features is not None and len(all_features) > 0:
    # Sort descending by datetime
    all_features = all_features.sort_values('datetime', ascending=False)
    
    # Get the very latest record
    latest = all_features.iloc[0].to_dict()
    
    print("=" * 60)
    print("📅 LATEST RECORD IN DATABASE")
    print("=" * 60)
    
    print(f"\n🕐 Datetime: {latest['datetime']}")
    print(f"\n🌡️  Weather:")
    print(f"   Temperature: {latest.get('temperature', 'N/A')}°C")
    print(f"   Humidity: {latest.get('humidity', 'N/A')}%")
    print(f"   Wind Speed: {latest.get('wind_speed', 'N/A')} km/h")
    print(f"   Precipitation: {latest.get('precipitation', 'N/A')} mm")
    
    print(f"\n💨 Air Quality:")
    print(f"   PM2.5: {latest.get('pm2_5', 'N/A')}")
    print(f"   PM10: {latest.get('pm10', 'N/A')}")
    print(f"   AQI Bucket: {latest.get('aqi_bucket', 'N/A')}")
    
    # Calculate how long ago
    record_time = latest['datetime']
    if isinstance(record_time, str):
        record_time = datetime.fromisoformat(record_time)
    
    time_ago = datetime.now() - record_time
    hours_ago = time_ago.total_seconds() / 3600
    minutes_ago = time_ago.total_seconds() / 60
    
    print(f"\n⏰ Last updated: {hours_ago:.1f} hours ago ({minutes_ago:.0f} minutes)")
    
    print("\n" + "=" * 60)
    print(f"📊 Total records in database: {len(all_features):,}")
    print("=" * 60)
    
else:
    print("❌ No data found in database")