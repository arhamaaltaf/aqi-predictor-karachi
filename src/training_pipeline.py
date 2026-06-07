"""
Training Pipeline for Karachi AQI Predictor

This script:
1. Loads features from MongoDB
2. Prepares training data
3. Trains multiple ML models
4. Evaluates and selects best model
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Classification Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
        self.label_encoder = LabelEncoder()  # For converting AQI classes (2,3,4,5) to (0,1,2,3)
    
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
        
        Creates target variable: PM2.5 value N hours in the future
        """
        print(f"\n🎯 Preparing training data (predicting {PREDICTION_HORIZON}h ahead)...")
        
        # Create target: PM2.5 value N hours in the future
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
        
        # Encode labels for XGBoost (2,3,4,5 → 0,1,2,3)
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target: {TARGET_VARIABLE} (shifted by {PREDICTION_HORIZON}h)")
        print(f"   Classes: {self.label_encoder.classes_} → {list(range(len(self.label_encoder.classes_)))}")
        
        return X, y_encoded, feature_cols
    
    def split_data(self, X, y, test_size=None, random_state=None, stratify=None):
        """Split data into train and test sets"""
        # Use provided parameters or defaults from config
        test_size = test_size if test_size is not None else TRAIN_TEST_SPLIT
        random_state = random_state if random_state is not None else RANDOM_STATE
        
        print(f"\n📊 Splitting data (test size: {test_size*100}%)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=True if stratify is not None else False  # Shuffle if stratifying
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
        """Define all models to train"""
        print("\n🤖 Defining classification models...")
        
        # ─────────────────────────────────────────────
        #  RANDOM FOREST - CONSERVATIVE PARAMETERS
        # ─────────────────────────────────────────────
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,           # Moderate number of trees
                max_depth=4,                # Reduced from 5 to prevent overfitting
                min_samples_split=15,       # Increased from 10 (more conservative splits)
                min_samples_leaf=5,         # Increased from 1 (require more samples per leaf)
                max_features=0.5,           # Reduced from 0.7 (use 50% of features)
                bootstrap=True,             # Changed to True (use bootstrap for variance reduction)
                criterion='gini',           # Changed to gini (often more robust)
                max_samples=0.8,            # Use 80% of data per tree
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            
            # ─────────────────────────────────────────────
            #  XGBOOST - CONSERVATIVE PARAMETERS
            # ─────────────────────────────────────────────
            'XGBoost': XGBClassifier(
                n_estimators=200,           # Reduced from 500 to prevent overfitting
                max_depth=4,                # HEAVILY reduced from 11 (was way too deep)
                learning_rate=0.05,         # Keep slow learning rate
                subsample=0.7,              # Reduced from 0.8 (use less data per tree)
                colsample_bytree=0.6,       # Slightly increased from 0.5
                min_child_weight=5,         # Increased from 1 (more samples per leaf)
                gamma=0.2,                  # Increased from 0.1 (higher split threshold)
                reg_alpha=1.0,              # Increased L1 regularization from 0.01
                reg_lambda=3.0,             # Increased L2 regularization from 0.5
                max_delta_step=0,           # Reset to 0 (default)
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            
            # ─────────────────────────────────────────────
            #  LIGHTGBM - CONSERVATIVE PARAMETERS
            # ─────────────────────────────────────────────
            'LightGBM': LGBMClassifier(
                n_estimators=200,           # Reduced from 500
                max_depth=4,                # Slightly increased from 3 but conservative
                learning_rate=0.1,         # Reduced from 0.1 (slower learning)
                num_leaves=15,              # Heavily reduced from 63 (was too complex)
                subsample= 0.8,              # Reduced from 0.9
                colsample_bytree=0.8,       # Reduced from 1.0
                reg_alpha= 0.5,              # Increased L1 from 0.001
                reg_lambda= 1,             # Increased L2 from 0.01
                min_child_samples=10,       # Increased from 20
                # min_split_gain=0.0,         # Increased from 0.0 (require gain to split)
                # max_bin= 511,                # Reduced from 511
                # path_smooth=0.0,            # Reset to default
                # extra_trees=True,          # Disabled (was True)
                # class_weight='balanced',    # Keep balanced for imbalanced classes
                random_state=RANDOM_STATE,
                n_jobs=-1,
                force_col_wise=True,
                verbose=-1
            ),

        }
        
        print(f"   ✅ Defined {len(self.models)} classification models")
        print(f"   Models: {', '.join(self.models.keys())}")
        print(f"   📝 Using conservative hyperparameters to prevent overfitting")
    
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
                
                # Evaluate - Classification Metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                
                # Cross-validation (5-fold)
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )
                cv_accuracy = cv_scores.mean()
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'cv_accuracy': cv_accuracy,
                    'predictions': y_test_pred
                }
                
                print(f"   Train Accuracy: {train_accuracy:.3f}")
                print(f"   Test Accuracy:  {test_accuracy:.3f}")
                print(f"   Precision:      {test_precision:.3f}")
                print(f"   Recall:         {test_recall:.3f}")
                print(f"   F1 Score:       {test_f1:.3f}")
                print(f"   CV Accuracy:    {cv_accuracy:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
    
    def select_best_model(self):
        """Select best model based on test accuracy"""
        print("\n" + "="*60)
        print("🏆 MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        comparison = []
        for name, result in self.results.items():
            comparison.append({
                'Model': name,
                'Accuracy': result['test_accuracy'],
                'Precision': result['test_precision'],
                'Recall': result['test_recall'],
                'F1 Score': result['test_f1'],
                'CV Accuracy': result['cv_accuracy']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('Accuracy', ascending=False)
        
        print("\n" + df_comparison.to_string(index=False))
        
        # Select best model (highest accuracy)
        self.best_model_name = df_comparison.iloc[0]['Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n🥇 Best Model: {self.best_model_name}")
        print(f"   Accuracy:  {self.results[self.best_model_name]['test_accuracy']:.3f}")
        print(f"   Precision: {self.results[self.best_model_name]['test_precision']:.3f}")
        print(f"   Recall:    {self.results[self.best_model_name]['test_recall']:.3f}")
        print(f"   F1 Score:  {self.results[self.best_model_name]['test_f1']:.3f}")
        
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
        
        # Save label encoder (shared by all models)
        label_encoder_path = 'models/label_encoder.joblib'
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"   ✅ Label encoder saved: {label_encoder_path}")
        
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
            
            # Prepare metadata - Classification Metrics
            metrics = {
                'test_accuracy': float(result['test_accuracy']),
                'train_accuracy': float(result['train_accuracy']),
                'precision': float(result['test_precision']),
                'recall': float(result['test_recall']),
                'f1_score': float(result['test_f1']),
                'cv_accuracy': float(result['cv_accuracy'])
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
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=0.2, random_state=42, stratify=y)
            
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