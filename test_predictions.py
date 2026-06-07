"""
Test the prediction system end-to-end
"""
from src.model_registry import ModelRegistry
from src.database import MongoDBHandler
import pandas as pd

print("="*70)
print("🔮 TESTING PREDICTIONS")
print("="*70)

# Load best model
print("\n📥 Loading best model...")
registry = ModelRegistry()
best_model_name, best_model = registry.get_best_model()

if best_model is None:
    print("❌ No best model found!")
    exit(1)

print(f"✅ Loaded: {best_model_name}")

# Get latest features from MongoDB
print("\n📥 Getting latest features from MongoDB...")
db = MongoDBHandler()
features_df = db.get_features(limit=10)

if features_df.empty:
    print("❌ No features found in MongoDB!")
    exit(1)

print(f"✅ Got {len(features_df)} records")
print(f"   Latest: {features_df['datetime'].max()}")

# Prepare features for prediction
feature_cols = registry.feature_columns
scaler = registry.scaler
label_encoder = registry.label_encoder

# Select feature columns
X = features_df[feature_cols].tail(5)  # Get last 5 records
X_scaled = scaler.transform(X)

# Make predictions
print("\n🔮 Making predictions...")
predictions_encoded = best_model.predict(X_scaled)
predictions = label_encoder.inverse_transform(predictions_encoded)

# Show results
print("\n📊 PREDICTION RESULTS (72h ahead):")
print("="*70)

results_df = features_df.tail(5)[['datetime', 'aqi', 'aqi_category', 'temperature', 'humidity', 'pm2_5']].copy()
results_df['predicted_aqi'] = predictions

# Map predictions to categories
aqi_mapping = {
    1: 'Good',
    2: 'Fair',
    3: 'Moderate',
    4: 'Poor',
    5: 'Very Poor'
}
results_df['predicted_category'] = results_df['predicted_aqi'].map(aqi_mapping)

print(results_df[['datetime', 'aqi', 'aqi_category', 'predicted_aqi', 'predicted_category']].to_string(index=False))

print("\n✅ Prediction system working!")
print("\n💡 Note: These are predictions for AQI 72 hours in the future")
print("   Actual AQI shown is current value, predicted is for 3 days ahead")

db.close()