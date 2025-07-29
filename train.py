"""
Main training and evaluation script for apartment rent prediction model.
Run this script to train models, evaluate performance, and generate visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

# Import our custom modules
from src.data_collector import DataCollector
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.visualizer import ModelVisualizer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main training pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("visualizations").mkdir(exist_ok=True)
    
    logger.info("Starting apartment rent prediction model training...")
    
    # 1. Data Collection
    logger.info("Step 1: Collecting data...")
    collector = DataCollector()
    df = collector.collect_sample_data()
    logger.info(f"Collected {len(df)} apartment records")
    
    # 2. Feature Engineering
    logger.info("Step 2: Feature engineering...")
    engineer = FeatureEngineer()
    X, y = engineer.prepare_features(df)
    X_train, X_test, y_train, y_test = engineer.split_data(X, y)
    
    logger.info(f"Features prepared: {X.shape[1]} features, {len(X)} samples")
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # 3. Model Training
    logger.info("Step 3: Training models...")
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # 4. Model Evaluation
    logger.info("Step 4: Evaluating models...")
    comparison_df = trainer.create_model_comparison_report()
    print("\\nModel Performance Comparison:")
    print(comparison_df.round(4))
    
    # Get best model
    best_name, best_model, best_info = trainer.get_best_model()
    logger.info(f"Best model: {best_name} (Test R²: {best_info['test_metrics']['r2']:.4f})")
    
    # 5. Visualizations
    logger.info("Step 5: Creating visualizations...")
    visualizer = ModelVisualizer()
    
    # Data distribution plots
    visualizer.plot_data_distribution(df, "visualizations/data_distribution.png")
    
    # Model comparison plots
    visualizer.plot_model_comparison(comparison_df, "visualizations/model_comparison.png")
    
    # Best model predictions
    best_predictions = best_info['predictions']['test']
    visualizer.plot_predictions_vs_actual(
        y_test.values, best_predictions, best_name, 
        f"visualizations/{best_name}_predictions.png"
    )
    
    # Feature importance (if available)
    if best_name in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
        importance_df = trainer.get_feature_importance(best_name, engineer.get_feature_importance_names())
        visualizer.plot_feature_importance(
            importance_df, best_name, 
            save_path=f"visualizations/{best_name}_feature_importance.png"
        )
    
    # Price trends analysis
    visualizer.plot_price_trends(df, "visualizations/price_trends.png")
    
    # Interactive map (if running in Jupyter or with display capability)
    try:
        visualizer.create_interactive_rent_map(df, "visualizations/rent_map.html")
    except Exception as e:
        logger.warning(f"Could not create interactive map: {e}")
    
    # 6. Save the fitted feature engineer
    import joblib
    joblib.dump(engineer, "models/feature_engineer.pkl")
    logger.info("Saved fitted feature engineer to models/feature_engineer.pkl")
    
    # 7. Save results summary
    summary = {
        "best_model": best_name,
        "best_test_r2": best_info['test_metrics']['r2'],
        "best_test_rmse": best_info['test_metrics']['rmse'],
        "best_test_mae": best_info['test_metrics']['mae'],
        "features_count": len(engineer.get_feature_importance_names()),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    with open("models/training_summary.txt", "w") as f:
        f.write("Apartment Rent Prediction Model Training Summary\\n")
        f.write("=" * 50 + "\\n\\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\\n")
        f.write("\\n" + "=" * 50 + "\\n")
        f.write("Model Comparison:\\n")
        f.write(comparison_df.to_string())
    
    logger.info("Training completed successfully!")
    logger.info(f"Best model ({best_name}) saved to models/{best_name}.pkl")
    logger.info("Check the 'visualizations' folder for plots and analysis")
    
    return trainer, engineer, best_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train apartment rent prediction models")
    parser.add_argument("--data-file", help="Path to custom data file (CSV)")
    parser.add_argument("--quick", action="store_true", help="Quick training with fewer models")
    
    args = parser.parse_args()
    
    # Run training
    trainer, engineer, best_model_name = main()
    
    print(f"\\n🎉 Training completed! Best model: {best_model_name}")
    print("📊 Check the 'visualizations' folder for analysis plots")
    print("🤖 To start the API server, run: python src/api.py")
