"""
MongoDB Database Handler for Karachi AQI Predictor
"""
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import certifi


from src.config import MONGODB_URI, MONGODB_DATABASE
from src.config import (
    COLLECTION_RAW_DATA, 
    COLLECTION_FEATURES, 
    COLLECTION_PREDICTIONS,
    COLLECTION_MODELS
)

class MongoDBHandler:
    """Handle all MongoDB operations"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        print("🔌 Connecting to MongoDB...")
        
        self.client = MongoClient(
            MONGODB_URI,
            tlsCAFile=certifi.where()
        )
        self.db = self.client[MONGODB_DATABASE]
        
        # Collections
        self.raw_data = self.db[COLLECTION_RAW_DATA]
        self.features = self.db[COLLECTION_FEATURES]
        self.predictions = self.db[COLLECTION_PREDICTIONS]
        self.models = self.db[COLLECTION_MODELS]
        
        # Create indexes for better performance
        self._create_indexes()
        
        print(f"✅ Connected to MongoDB database: {MONGODB_DATABASE}")
    
    def _create_indexes(self):
        """Create indexes on collections"""
        try:
            self.raw_data.create_index([('datetime', -1)])
            self.features.create_index([('datetime', -1)])
            self.predictions.create_index([('created_at', -1)])
            print("✅ Database indexes created")
        except Exception as e:
            print(f"⚠️  Index creation warning: {e}")
    
    # ========================================
    # RAW DATA OPERATIONS
    # ========================================
    
    def insert_raw_data(self, df):
        """Insert raw data into MongoDB"""
        print(f"💾 Inserting {len(df)} raw data records...")
        
        records = df.to_dict('records')
        
        # Convert datetime to proper format
        for record in records:
            if 'datetime' in record and not isinstance(record['datetime'], datetime):
                record['datetime'] = pd.to_datetime(record['datetime'])
        
        if records:
            # Delete existing data to avoid duplicates
            self.raw_data.delete_many({})
            self.raw_data.insert_many(records)
            print(f"✅ Inserted {len(records)} raw data records")
    
    def get_raw_data(self, limit=None):
        """Get raw data from MongoDB"""
        print("📥 Loading raw data from MongoDB...")
        
        query = self.raw_data.find().sort('datetime', -1)
        
        if limit:
            query = query.limit(limit)
        
        df = pd.DataFrame(list(query))
        
        if not df.empty and '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        print(f"✅ Loaded {len(df)} raw data records")
        return df
    
    # ========================================
    # FEATURE OPERATIONS
    # ========================================
    
    def insert_features(self, df):
        """Insert engineered features into MongoDB"""
        print(f"💾 Inserting {len(df)} feature records...")
        
        records = df.to_dict('records')
        
        # Convert datetime
        for record in records:
            if 'datetime' in record and not isinstance(record['datetime'], datetime):
                record['datetime'] = pd.to_datetime(record['datetime'])
        
        if records:
            # Delete existing and insert new
            self.features.delete_many({})
            self.features.insert_many(records)
            print(f"✅ Inserted {len(records)} feature records")
    
    def get_features(self, limit=None, start_date=None, end_date=None):
        """Get features from MongoDB"""
        print("📥 Loading features from MongoDB...")
        
        # Build query
        query = {}
        if start_date and end_date:
            query['datetime'] = {'$gte': start_date, '$lte': end_date}
        
        cursor = self.features.find(query).sort('datetime', -1)
        
        if limit:
            cursor = cursor.limit(limit)
        
        df = pd.DataFrame(list(cursor))
        
        if not df.empty and '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        print(f"✅ Loaded {len(df)} feature records")
        return df
    
    def get_latest_features(self, n_hours=72):
        """Get latest N hours of features"""
        print(f"📥 Loading latest {n_hours} hours of features...")
        
        df = pd.DataFrame(list(
            self.features.find().sort('datetime', -1).limit(n_hours)
        ))
        
        if not df.empty and '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        print(f"✅ Loaded {len(df)} latest feature records")
        return df
    
    # ========================================
    # PREDICTION OPERATIONS
    # ========================================
    
    def save_predictions(self, predictions_df, model_name, metrics=None):
        """Save model predictions"""
        print(f"💾 Saving predictions for {model_name}...")
        
        record = {
            'model_name': model_name,
            'created_at': datetime.now(),
            'predictions': predictions_df.to_dict('records'),
            'metrics': metrics
        }
        
        self.predictions.insert_one(record)
        print(f"✅ Saved {len(predictions_df)} predictions")
    
    def get_latest_predictions(self, model_name=None):
        """Get latest predictions"""
        print("📥 Loading latest predictions...")
        
        query = {'model_name': model_name} if model_name else {}
        
        result = self.predictions.find_one(
            query,
            sort=[('created_at', -1)]
        )
        
        if result:
            df = pd.DataFrame(result['predictions'])
            print(f"✅ Loaded {len(df)} predictions")
            return df
        else:
            print("⚠️  No predictions found")
            return pd.DataFrame()
    
    # ========================================
    # MODEL METADATA OPERATIONS
    # ========================================
    
    def save_model_metadata(self, model_name, metrics, params, feature_importance=None):
        """Save model metadata"""
        print(f"💾 Saving metadata for {model_name}...")
        
        record = {
            'model_name': model_name,
            'created_at': datetime.now(),
            'metrics': metrics,
            'params': params,
            'feature_importance': feature_importance
        }
        
        self.models.insert_one(record)
        print(f"✅ Saved model metadata")
    
    def get_model_metadata(self, model_name):
        """Get model metadata"""
        print(f"📥 Loading metadata for {model_name}...")
        
        result = self.models.find_one(
            {'model_name': model_name},
            sort=[('created_at', -1)]
        )
        
        if result and '_id' in result:
            del result['_id']
        
        return result
    
    # ========================================
    # UTILITY OPERATIONS
    # ========================================
    
    def get_collection_stats(self):
        """Get statistics about all collections"""
        print("\n" + "="*60)
        print("📊 DATABASE STATISTICS")
        print("="*60)
        
        collections = {
            'Raw Data': self.raw_data,
            'Features': self.features,
            'Predictions': self.predictions,
            'Models': self.models
        }
        
        for name, collection in collections.items():
            count = collection.count_documents({})
            print(f"   {name:20s}: {count:,} records")
        
        print("="*60 + "\n")
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        print("✅ MongoDB connection closed")


# Test connection
if __name__ == "__main__":
    db = MongoDBHandler()
    db.get_collection_stats()
    db.close()