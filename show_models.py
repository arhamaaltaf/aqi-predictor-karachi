"""
Show all trained models and their performance from MongoDB
"""
from src.database import MongoDBHandler
import pandas as pd
from datetime import datetime

print("📊 Fetching model registry from MongoDB...\n")

db = MongoDBHandler()

# Get all models from MongoDB
models_cursor = db.models.find().sort('trained_at', -1)
models_list = list(models_cursor)

if models_list:
    print("=" * 80)
    print("🤖 MODEL REGISTRY - PERFORMANCE METRICS")
    print("=" * 80)
    
    for i, model in enumerate(models_list, 1):
        print(f"\n{'='*80}")
        print(f"Model #{i}: {model.get('model_name', 'Unknown')}")
        print(f"{'='*80}")
        
        print(f"\n📅 Training Date: {model.get('created_at', 'Unknown')}")
        
        if model.get('is_best', False):
            print("🏆 ** BEST MODEL **")
        
        # Get metrics from nested structure
        metrics = model.get('metrics', {})
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Test Accuracy:  {metrics.get('test_accuracy', 0):.3f}")
        print(f"   Train Accuracy: {metrics.get('train_accuracy', 0):.3f}")
        print(f"   CV Accuracy:    {metrics.get('cv_accuracy', 0):.3f}")
        print(f"   Precision:      {metrics.get('precision', 0):.3f}")
        print(f"   Recall:         {metrics.get('recall', 0):.3f}")
        print(f"   F1 Score:       {metrics.get('f1_score', 0):.3f}")
        
        params = model.get('params', {})
        if params:
            print(f"\n⚙️  Hyperparameters:")
            for key, value in params.items():
                print(f"   {key}: {value}")
        
        print(f"\n💾 Model File: {model.get('model_path', 'N/A')}")
    
    print("\n" + "=" * 80)
    print(f"📈 Total Models in Registry: {len(models_list)}")
    print("=" * 80)
    
    # Summary table
    print("\n📊 PERFORMANCE SUMMARY TABLE:\n")
    df = pd.DataFrame([{
        'Model': m.get('model_name', 'Unknown'),
        'Test Acc': f"{m.get('metrics', {}).get('test_accuracy', 0):.3f}",
        'Train Acc': f"{m.get('metrics', {}).get('train_accuracy', 0):.3f}",
        'CV Acc': f"{m.get('metrics', {}).get('cv_accuracy', 0):.3f}",
        'F1 Score': f"{m.get('metrics', {}).get('f1_score', 0):.3f}",
        'Best': '🏆' if m.get('is_best', False) else ''
    } for m in models_list])
    
    print(df.to_string(index=False))
    
else:
    print("❌ No models found in MongoDB registry")
    print("Run: python -m src.training_pipeline")

db.close()