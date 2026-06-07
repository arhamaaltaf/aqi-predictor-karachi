"""
Model Registry for Karachi AQI Predictor

This module:
1. Manages multiple trained models
2. Selects best model based on metrics
3. Makes predictions for different time horizons (24h, 48h, 72h)
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.database import MongoDBHandler


class ModelRegistry:
    """Manage model registry and predictions"""
    
    def __init__(self):
        """Initialize registry"""
        self.db = MongoDBHandler()
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.models = {}
        self.model_metadata = {}
        
        # Load shared artifacts
        self._load_shared_artifacts()
    
    def _load_shared_artifacts(self):
        """Load scaler, label encoder, and feature columns"""
        try:
            self.scaler = joblib.load('models/scaler.joblib')
            self.label_encoder = joblib.load('models/label_encoder.joblib')
            self.feature_columns = joblib.load('models/feature_columns.joblib')
            print("✅ Loaded scaler, label encoder, and feature columns")
        except Exception as e:
            print(f"⚠️  Warning: Could not load shared artifacts: {e}")
    
    def load_all_models(self):
        """Load all models from registry"""
        print("\n📦 Loading all models from registry...")
        
        # Get all model metadata from MongoDB
        cursor = self.db.models.find().sort('created_at', -1)
        model_records = list(cursor)
        
        if not model_records:
            print("❌ No models found in registry!")
            return False
        
        # Group by model name and get latest version
        model_dict = {}
        for record in model_records:
            model_name = record['model_name']
            if model_name not in model_dict:
                model_dict[model_name] = record
        
        # Load each model
        for model_name, metadata in model_dict.items():
            try:
                model_path = metadata['model_path']
                self.models[model_name] = joblib.load(model_path)
                self.model_metadata[model_name] = metadata
                
                is_best = "🥇 BEST" if metadata.get('is_best', False) else ""
                print(f"   ✅ {model_name:15} - Accuracy: {metadata['metrics']['test_accuracy']:.3f} {is_best}")
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {e}")
        
        print(f"\n✅ Loaded {len(self.models)} models")
        return True
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.models:
            self.load_all_models()
        
        # Find best model from metadata
        best_model_name = None
        best_accuracy = 0
        
        for model_name, metadata in self.model_metadata.items():
            if metadata.get('is_best', False):
                best_model_name = model_name
                break
            
            # Fallback: use highest accuracy
            accuracy = metadata['metrics']['test_accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
        
        if best_model_name:
            print(f"\n🥇 Best Model: {best_model_name}")
            print(f"   Accuracy:  {self.model_metadata[best_model_name]['metrics']['test_accuracy']:.3f}")
            print(f"   Precision: {self.model_metadata[best_model_name]['metrics']['precision']:.3f}")
            return best_model_name, self.models[best_model_name]
        
        return None, None
    
    def get_model(self, model_name):
        """Get a specific model by name"""
        if not self.models:
            self.load_all_models()
        
        if model_name in self.models:
            return self.models[model_name]
        
        print(f"❌ Model '{model_name}' not found in registry")
        return None
    
    def list_models(self):
        """List all available models with their metrics"""
        if not self.models:
            self.load_all_models()
        
        print("\n" + "="*60)
        print("📊 MODEL REGISTRY")
        print("="*60)
        
        # Sort by accuracy (highest first)
        sorted_models = sorted(
            self.model_metadata.items(),
            key=lambda x: x[1]['metrics']['test_accuracy'],
            reverse=True
        )
        
        for model_name, metadata in sorted_models:
            is_best = "🥇" if metadata.get('is_best', False) else "  "
            metrics = metadata['metrics']
            print(f"{is_best} {model_name:15} | Accuracy: {metrics['test_accuracy']:.3f} | Precision: {metrics['precision']:.3f} | F1: {metrics['f1_score']:.3f}")
        
        print("="*60)
    
    def prepare_input_features(self, input_data):
        """Prepare input data for prediction"""
        # Ensure input has all required features
        if isinstance(input_data, pd.DataFrame):
            # Select only the features used in training
            X = input_data[self.feature_columns]
        else:
            raise ValueError("Input must be a pandas DataFrame")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict_multi_horizon(self, model_name=None):
        """
        Make predictions for multiple time horizons:
        - 24h ahead (using data from 72h ago)
        - 48h ahead (using data from 48h ago)
        - 72h ahead (using current data)
        
        Returns: dict with predictions for each horizon
        """
        print("\n" + "="*60)
        print("🔮 MULTI-HORIZON PREDICTIONS")
        print("="*60)
        
        # Get model to use
        if model_name:
            model = self.get_model(model_name)
            if not model:
                return None
            print(f"Using model: {model_name}")
        else:
            model_name, model = self.get_best_model()
            if not model:
                return None
        
        # Get latest data from MongoDB
        df_latest = self.db.get_latest_features(n_hours=100)
        
        if df_latest.empty:
            print("❌ No data available for predictions")
            return None
        
        # Sort by datetime
        df_latest = df_latest.sort_values('datetime').reset_index(drop=True)
        
        predictions = {}
        
        # 1. Predict 24h ahead (using data from 72h ago)
        if len(df_latest) >= 72:
            input_72h_ago = df_latest.iloc[-72:-71]  # Data from 72h ago
            if not input_72h_ago.empty:
                X_72h = self.prepare_input_features(input_72h_ago)
                pred_24h_encoded = model.predict(X_72h)[0]
                pred_24h = self.label_encoder.inverse_transform([pred_24h_encoded])[0]  # Decode to original AQI
                pred_time_24h = pd.to_datetime(input_72h_ago['datetime'].iloc[0]) + timedelta(hours=72)
                
                predictions['24h_ahead'] = {
                    'prediction': float(pred_24h),
                    'prediction_time': pred_time_24h,
                    'input_time': input_72h_ago['datetime'].iloc[0],
                    'description': 'Tomorrow AQI (using 3 days ago data)'
                }
                print(f"\n📍 24h Ahead (Tomorrow)")
                print(f"   Input from: {input_72h_ago['datetime'].iloc[0]}")
                print(f"   Predicting: {pred_time_24h}")
                print(f"   AQI: {pred_24h:.1f}")
        
        # 2. Predict 48h ahead (using data from 48h ago)
        if len(df_latest) >= 48:
            input_48h_ago = df_latest.iloc[-48:-47]  # Data from 48h ago
            if not input_48h_ago.empty:
                X_48h = self.prepare_input_features(input_48h_ago)
                pred_48h_encoded = model.predict(X_48h)[0]
                pred_48h = self.label_encoder.inverse_transform([pred_48h_encoded])[0]  # Decode to original AQI
                pred_time_48h = pd.to_datetime(input_48h_ago['datetime'].iloc[0]) + timedelta(hours=72)
                
                predictions['48h_ahead'] = {
                    'prediction': float(pred_48h),
                    'prediction_time': pred_time_48h,
                    'input_time': input_48h_ago['datetime'].iloc[0],
                    'description': 'Day after tomorrow AQI (using yesterday data)'
                }
                print(f"\n📍 48h Ahead (Day After Tomorrow)")
                print(f"   Input from: {input_48h_ago['datetime'].iloc[0]}")
                print(f"   Predicting: {pred_time_48h}")
                print(f"   AQI: {pred_48h:.1f}")
        
        # 3. Predict 72h ahead (using current/latest data)
        input_current = df_latest.iloc[-1:]  # Latest data
        if not input_current.empty:
            X_current = self.prepare_input_features(input_current)
            pred_72h_encoded = model.predict(X_current)[0]
            pred_72h = self.label_encoder.inverse_transform([pred_72h_encoded])[0]  # Decode to original AQI
            pred_time_72h = pd.to_datetime(input_current['datetime'].iloc[0]) + timedelta(hours=72)
            
            predictions['72h_ahead'] = {
                'prediction': float(pred_72h),
                'prediction_time': pred_time_72h,
                'input_time': input_current['datetime'].iloc[0],
                'description': '3 days ahead AQI (using today data)'
            }
            print(f"\n📍 72h Ahead (3 Days Later)")
            print(f"   Input from: {input_current['datetime'].iloc[0]}")
            print(f"   Predicting: {pred_time_72h}")
            print(f"   AQI: {pred_72h:.1f}")
        
        print("\n" + "="*60)
        print(f"✅ Generated {len(predictions)} predictions using {model_name}")
        print("="*60)
        
        # Add model info
        predictions['model_used'] = model_name
        predictions['model_metrics'] = self.model_metadata[model_name]['metrics']
        
        return predictions
    
    def close(self):
        """Close database connection"""
        self.db.close()


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    import sys
    
    registry = ModelRegistry()
    
    # Check command
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            # List all models
            registry.list_models()
        
        elif command == "predict":
            # Make multi-horizon predictions
            model_name = sys.argv[2] if len(sys.argv) > 2 else None
            predictions = registry.predict_multi_horizon(model_name=model_name)
            
            if predictions:
                print("\n📊 PREDICTION SUMMARY:")
                for horizon, pred_data in predictions.items():
                    if horizon not in ['model_used', 'model_metrics']:
                        print(f"\n{pred_data['description']}")
                        print(f"   AQI: {pred_data['prediction']:.1f}")
                        print(f"   Time: {pred_data['prediction_time']}")
    else:
        # Default: list models
        registry.list_models()
    
    registry.close()