"""
Example usage and demo script for the apartment rent prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from visualizer import ModelVisualizer

def demo_prediction_pipeline():
    """Demonstrate the complete prediction pipeline."""
    print("🏠 Apartment Rent Prediction Demo")
    print("=" * 40)
    
    # 1. Load or create sample data
    print("\\n1. Loading sample data...")
    collector = DataCollector()
    df = collector.load_data()
    print(f"   ✓ Loaded {len(df)} apartment records")
    print(f"   ✓ Average rent: ${df['rent'].mean():.2f}")
    print(f"   ✓ Rent range: ${df['rent'].min():.2f} - ${df['rent'].max():.2f}")
    
    # 2. Feature engineering
    print("\\n2. Preparing features...")
    engineer = FeatureEngineer()
    X, y = engineer.prepare_features(df)
    print(f"   ✓ Created {X.shape[1]} features")
    print(f"   ✓ Feature examples: {list(X.columns[:5])}")
    
    # 3. Train a simple model (just Random Forest for demo)
    print("\\n3. Training model...")
    X_train, X_test, y_train, y_test = engineer.split_data(X, y)
    
    trainer = ModelTrainer()
    # Train only Random Forest for quick demo
    rf_model = trainer.models['random_forest']
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   ✓ Model trained!")
    print(f"   ✓ Test R²: {r2:.4f}")
    print(f"   ✓ Test RMSE: ${rmse:.2f}")
    print(f"   ✓ Test MAE: ${mae:.2f}")
    
    # 4. Demo single prediction
    print("\\n4. Making sample predictions...")
    
    # Create sample apartments
    sample_apartments = [
        {
            'bedrooms': 2, 'bathrooms': 1.5, 'square_feet': 1000, 'floor': 5,
            'building_age': 10, 'latitude': 40.7831, 'longitude': -73.9712,
            'has_elevator': True, 'has_doorman': False, 'has_gym': True,
            'has_parking': True, 'pet_friendly': True, 'neighborhood_crime_rate': 3.5,
            'distance_to_subway': 0.3, 'distance_to_downtown': 5, 'nearby_restaurants': 20,
            'nearby_schools_rating': 8.5, 'building_type': 'apartment', 'heating_type': 'central'
        },
        {
            'bedrooms': 1, 'bathrooms': 1, 'square_feet': 600, 'floor': 2,
            'building_age': 25, 'latitude': 40.7589, 'longitude': -73.9851,
            'has_elevator': False, 'has_doorman': False, 'has_gym': False,
            'has_parking': False, 'pet_friendly': False, 'neighborhood_crime_rate': 6.0,
            'distance_to_subway': 0.8, 'distance_to_downtown': 12, 'nearby_restaurants': 8,
            'nearby_schools_rating': 6.5, 'building_type': 'apartment', 'heating_type': 'radiator'
        },
        {
            'bedrooms': 3, 'bathrooms': 2.5, 'square_feet': 1500, 'floor': 15,
            'building_age': 5, 'latitude': 40.7505, 'longitude': -73.9934,
            'has_elevator': True, 'has_doorman': True, 'has_gym': True,
            'has_parking': True, 'pet_friendly': True, 'neighborhood_crime_rate': 2.0,
            'distance_to_subway': 0.1, 'distance_to_downtown': 3, 'nearby_restaurants': 35,
            'nearby_schools_rating': 9.2, 'building_type': 'condo', 'heating_type': 'central'
        }
    ]
    
    for i, apt in enumerate(sample_apartments, 1):
        # Prepare features for this apartment
        apt_df = pd.DataFrame([apt])
        apt_features, _ = engineer.prepare_features(apt_df, fit_scalers=False)
        
        # Make prediction
        predicted_rent = rf_model.predict(apt_features)[0]
        
        print(f"\\n   Apartment {i}:")
        print(f"   • {apt['bedrooms']} bed, {apt['bathrooms']} bath, {apt['square_feet']} sqft")
        print(f"   • Floor {apt['floor']}, {apt['building_age']} years old")
        print(f"   • Building type: {apt['building_type']}")
        print(f"   • Distance to subway: {apt['distance_to_subway']} miles")
        print(f"   • Amenities: Elevator({apt['has_elevator']}), Doorman({apt['has_doorman']}), Gym({apt['has_gym']})")
        print(f"   🔮 Predicted rent: ${predicted_rent:.2f}/month")
    
    # 5. Feature importance
    print("\\n5. Top features influencing rent:")
    if hasattr(rf_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': engineer.get_feature_importance_names(),
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in importance_df.head(10).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.4f}")
    
    print("\\n🎉 Demo completed!")
    print("\\nNext steps:")
    print("• Run 'python train.py' for full model training and evaluation")
    print("• Run 'python src/api.py' to start the prediction API server")
    print("• Check the visualizations folder for detailed analysis plots")

def demo_cost_prediction():
    """Demo for predicting both rent and operational costs."""
    print("\\n" + "=" * 40)
    print("💰 Cost Prediction Extension Demo")
    print("=" * 40)
    
    # Create sample data with additional cost features
    np.random.seed(42)
    n_samples = 100
    
    # Basic apartment features
    apartments = pd.DataFrame({
        'bedrooms': np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.4, 0.2]),
        'bathrooms': np.random.choice([1, 1.5, 2], n_samples, p=[0.4, 0.3, 0.3]),
        'square_feet': np.random.normal(900, 300, n_samples).clip(400, 2000),
        'building_age': np.random.randint(0, 40, n_samples),
        'has_central_ac': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'has_dishwasher': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'insulation_quality': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    })
    
    # Calculate estimated costs
    apartments['base_rent'] = (
        apartments['bedrooms'] * 400 +
        apartments['bathrooms'] * 200 +
        apartments['square_feet'] * 2.0 +
        apartments['has_central_ac'] * 100 +
        apartments['has_dishwasher'] * 50 +
        np.random.normal(1000, 200, n_samples)
    ).clip(800, 4000)
    
    # Fixed costs (monthly)
    apartments['utilities_fixed'] = (
        apartments['square_feet'] * 0.15 +  # Base utility cost
        (6 - apartments['insulation_quality']) * 20 +  # Insulation impact
        apartments['has_central_ac'] * 30 +
        np.random.normal(50, 20, n_samples)
    ).clip(80, 300)
    
    apartments['insurance'] = (
        apartments['base_rent'] * 0.05 +  # Insurance based on rent
        np.random.normal(25, 10, n_samples)
    ).clip(15, 100)
    
    # Variable costs (usage-dependent)
    apartments['heating_cooling_variable'] = (
        apartments['square_feet'] * 0.08 +
        (6 - apartments['insulation_quality']) * 15 +
        np.random.normal(40, 25, n_samples)
    ).clip(20, 200)
    
    apartments['total_monthly_cost'] = (
        apartments['base_rent'] +
        apartments['utilities_fixed'] +
        apartments['insurance'] +
        apartments['heating_cooling_variable']
    )
    
    print(f"\\nGenerated cost analysis for {len(apartments)} apartments:")
    print(f"• Average base rent: ${apartments['base_rent'].mean():.2f}")
    print(f"• Average utilities: ${apartments['utilities_fixed'].mean():.2f}")
    print(f"• Average insurance: ${apartments['insurance'].mean():.2f}")
    print(f"• Average heating/cooling: ${apartments['heating_cooling_variable'].mean():.2f}")
    print(f"• Average total cost: ${apartments['total_monthly_cost'].mean():.2f}")
    
    # Cost breakdown analysis
    print("\\nCost breakdown by apartment size:")
    for bedrooms in [1, 2, 3]:
        subset = apartments[apartments['bedrooms'] == bedrooms]
        if len(subset) > 0:
            avg_total = subset['total_monthly_cost'].mean()
            avg_rent = subset['base_rent'].mean()
            avg_other = avg_total - avg_rent
            print(f"• {bedrooms} bedroom: ${avg_total:.0f}/month (${avg_rent:.0f} rent + ${avg_other:.0f} other costs)")
    
    print("\\n💡 This demonstrates how the model could be extended to predict:")
    print("• Total cost of ownership/renting")
    print("• Utility costs based on apartment features")
    print("• Seasonal cost variations")
    print("• Energy efficiency impact on costs")

if __name__ == "__main__":
    demo_prediction_pipeline()
    demo_cost_prediction()
