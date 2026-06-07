"""
Training Pipeline for Karachi AQI Predictor (Regression Version)

This script:
1. Loads features from MongoDB
2. Prepares training data
3. Trains multiple ML regression models
4. Evaluates and selects best model (lowest RMSE)
5. Saves model and metadata

Run:
- Manually: python -m src.training_pipeline
- Automated: Via GitHub Actions daily
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Regression Metrics & Models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.database import MongoDBHandler
from src.config import TARGET_VARIABLE, TRAIN_TEST_SPLIT, RANDOM_STATE, PREDICTION_HORIZON

class TrainingPipeline:
    """Handle model training and evaluation"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.db = MongoDBHandler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        # Note: LabelEncoder removed for regression
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    
    def load_data(self):
        """Load features from MongoDB"""
        print("📥 Loading training data from MongoDB...")
        
        df = self.db.get_features()
        
        if df.empty:
            raise ValueError("No data found in MongoDB! Run feature pipeline first.")
        
        print(f"✅ Loaded {len(df):,} records")
        print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    
    def prepare_training_data(self, df):
        """
        Prepare data for training
        
        Creates target variable: Continuous AQI value N hours in the future
        """
        print(f"\n🎯 Preparing training data (predicting {PREDICTION_HORIZON}h ahead)...")
        
        # Create target: Continuous value N hours in the future
        df = df.sort_values('datetime').reset_index(drop=True)
        df['target'] = df[TARGET_VARIABLE].shift(-PREDICTION_HORIZON)
        
        # Remove rows with NaN target
        df_train = df.dropna(subset=['target'])
        
        print(f"   Original: {len(df):,} records")
        print(f"   After creating target: {len(df_train):,} records")
        
        # Separate features and target FIRST (before dropna)
        # Exclude datetime, target, and target variable
        exclude_cols = ['datetime', 'target', TARGET_VARIABLE]
        feature_cols = [col for col in df_train.columns if col not in exclude_cols]
        
        # Remove rows with NaN in FEATURE columns only
        original_count = len(df_train)
        df_train = df_train.dropna(subset=feature_cols)
        nan_removed = original_count - len(df_train)
        
        if nan_removed > 0:
            print(f"   ⚠️  Removed {nan_removed:,} records with NaN features (insufficient history)")
        
        print(f"   Final training records: {len(df_train):,}")
        
        # Extract features and target from cleaned data
        X = df_train[feature_cols]
        y = df_train['target']
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target: {TARGET_VARIABLE} (shifted by {PREDICTION_HORIZON}h)")
        
        return X, y, feature_cols
    
    def split_data(self, X, y, test_size=None, random_state=None):
        """Split data into train and test sets"""
        # Use provided parameters or defaults from config
        test_size = test_size if test_size is not None else TRAIN_TEST_SPLIT
        random_state = random_state if random_state is not None else RANDOM_STATE
        
        print(f"\n📊 Splitting data (test size: {test_size*100}%)...")
        
        # Note: stratify removed as it is not applicable for continuous regression targets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Testing samples: {len(X_test):,}")
        
        # Scale features
        print("   Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    # ========================================
    # MODEL TRAINING
    # ========================================
    
    def define_models(self):
        """Define all regression models to train"""
        print("\n🤖 Defining regression models...")
        
        self.models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_split=15,
                min_samples_leaf=5,
                max_features=0.5,
                bootstrap=True,
                max_samples=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            
            'XGBoost': XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.6,
                gamma=0.2,
                reg_alpha=1.0,
                reg_lambda=3.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric='rmse'
            ),
            
            'LightGBM': LGBMRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1,
                min_child_samples=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                force_col_wise=True,
                verbose=-1
            ),
        }
        
        print(f"   ✅ Defined {len(self.models)} regression models")
        print(f"   Models: {', '.join(self.models.keys())}")
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("🏋️  TRAINING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n📍 Training {name}...")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Evaluate - Regression Metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Cross-validation (5-fold) using neg_root_mean_squared_error
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
                cv_rmse = -cv_scores.mean()  # Convert back to positive RMSE
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                    'cv_rmse': cv_rmse,
                    'predictions': y_test_pred
                }
                
                print(f"   Train RMSE: {train_rmse:.3f}")
                print(f"   Test RMSE:  {test_rmse:.3f}")
                print(f"   Test MAE:   {test_mae:.3f}")
                print(f"   Test R²:    {test_r2:.3f}")
                print(f"   CV RMSE:    {cv_rmse:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
    
    def select_best_model(self):
        """Select best model based on lowest test RMSE"""
        print("\n" + "="*60)
        print("🏆 MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        comparison = []
        for name, result in self.results.items():
            comparison.append({
                'Model': name,
                'RMSE': result['test_rmse'],
                'MAE': result['test_mae'],
                'R2 Score': result['test_r2'],
                'CV RMSE': result['cv_rmse']
            })
        
        df_comparison = pd.DataFrame(comparison)
        # Sort by RMSE ascending (lower is better)
        df_comparison = df_comparison.sort_values('RMSE', ascending=True)
        
        print("\n" + df_comparison.to_string(index=False))
        
        # Select best model (lowest RMSE)
        self.best_model_name = df_comparison.iloc[0]['Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n🥇 Best Model: {self.best_model_name}")
        print(f"   RMSE:     {self.results[self.best_model_name]['test_rmse']:.3f}")
        print(f"   MAE:      {self.results[self.best_model_name]['test_mae']:.3f}")
        print(f"   R² Score: {self.results[self.best_model_name]['test_r2']:.3f}")
        
        return df_comparison
    
    # ========================================
    # MODEL SAVING
    # ========================================
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)
    
    def save_all_models(self, feature_cols):
        """Save all trained models to registry"""
        print("\n💾 Saving all models to registry...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save scaler (shared by all models)
        scaler_path = 'models/scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        print(f"   ✅ Scaler saved: {scaler_path}")
        
        # Save feature columns (shared by all models)
        feature_path = 'models/feature_columns.joblib'
        joblib.dump(feature_cols, feature_path)
        print(f"   ✅ Feature columns saved: {feature_path}")
        
        # Clear old model metadata from MongoDB
        self.db.models.delete_many({})
        
        # Save each model
        for model_name, result in self.results.items():
            # Save model file
            model_path = f'models/{model_name.lower()}_model.joblib'
            joblib.dump(result['model'], model_path)
            print(f"   ✅ {model_name} saved: {model_path}")
            
            # Prepare metadata - Regression Metrics
            metrics = {
                'test_rmse': float(result['test_rmse']),
                'train_rmse': float(result['train_rmse']),
                'test_mae': float(result['test_mae']),
                'test_r2': float(result['test_r2']),
                'cv_rmse': float(result['cv_rmse'])
            }
            
            # Convert params to JSON-serializable format
            params = result['model'].get_params()
            params = self._convert_to_serializable(params)
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(result['model'], 'feature_importances_'):
                importance_dict = {str(k): float(v) for k, v in zip(feature_cols, result['model'].feature_importances_)}
                # Top 20 features
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
                feature_importance = dict(sorted_importance)
            
            # Mark if this is the best model
            is_best = (model_name == self.best_model_name)
            
            # Save to MongoDB registry
            record = {
                'model_name': model_name,
                'model_path': model_path,
                'is_best': is_best,
                'created_at': datetime.now(),
                'metrics': metrics,
                'params': params,
                'feature_importance': feature_importance,
                'prediction_horizon': PREDICTION_HORIZON
            }
            
            self.db.models.insert_one(record)
            print(f"      {'🥇' if is_best else '  '} Metadata saved to MongoDB")
        
        print(f"\n✅ Model Registry Updated: {len(self.results)} models saved")
    
    # ========================================
    # PIPELINE EXECUTION
    # ========================================
    
    def run(self):
        """Run complete training pipeline"""
        print("\n" + "="*60)
        print("🚀 TRAINING PIPELINE START")
        print("="*60)
        print(f"Time: {datetime.now()}\n")
        
        try:
            # Load data
            df = self.load_data()
            
            # Prepare training data
            X, y, feature_cols = self.prepare_training_data(df)
            
            # Split data (Note: stratify=y is removed here since y is continuous)
            X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=0.2, random_state=42)
            
            # Define models
            self.define_models()
            
            # Train and evaluate
            self.train_and_evaluate(X_train, X_test, y_train, y_test)
            
            # Select best model
            comparison = self.select_best_model()
            
            # Save all models to registry
            self.save_all_models(feature_cols)
            
            print("\n" + "="*60)
            print("✅ TRAINING PIPELINE COMPLETE!")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR in training pipeline: {e}")
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
    pipeline = TrainingPipeline()
    success = pipeline.run()
    pipeline.close()
    
    import sys
    sys.exit(0 if success else 1)