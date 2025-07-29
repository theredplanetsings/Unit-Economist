import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Any
import pickle
import logging
from pathlib import Path
import joblib

class ModelTrainer:
    """
    Trains and evaluates multiple machine learning models for apartment rent prediction.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.model_scores = {}
        self.trained_models = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different regression models."""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            ),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train all models and evaluate their performance."""
        
        results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_metrics(y_test, y_pred_test)
                
                # Store results
                results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'predictions': {
                        'train': y_pred_train,
                        'test': y_pred_test
                    }
                }
                
                # Save the trained model
                self._save_model(model, name)
                
                self.logger.info(f"{name} - Test R²: {test_metrics['r2']:.4f}, "
                               f"Test RMSE: {test_metrics['rmse']:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.trained_models = results
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _save_model(self, model, name: str):
        """Save trained model to disk."""
        model_path = self.model_dir / f"{name}.pkl"
        joblib.dump(model, model_path)
        self.logger.info(f"Saved {name} model to {model_path}")
    
    def load_model(self, name: str):
        """Load a trained model from disk."""
        model_path = self.model_dir / f"{name}.pkl"
        if model_path.exists():
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model {name} not found at {model_path}")
    
    def get_best_model(self) -> Tuple[str, Any, Dict]:
        """Get the best performing model based on test R² score."""
        if not self.trained_models:
            raise ValueError("No models have been trained yet.")
        
        best_score = -np.inf
        best_model_name = None
        best_model_info = None
        
        for name, info in self.trained_models.items():
            if 'error' not in info:
                test_r2 = info['test_metrics']['r2']
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model_name = name
                    best_model_info = info
        
        return best_model_name, best_model_info['model'], best_model_info
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a specific model."""
        if model_name in self.trained_models:
            model = self.trained_models[model_name]['model']
            return model.predict(X)
        else:
            # Try to load from disk
            model = self.load_model(model_name)
            return model.predict(X)
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models.")
        
        model = self.trained_models[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning(f"Model {model_name} does not have feature importance.")
            return pd.DataFrame()
    
    def create_model_comparison_report(self) -> pd.DataFrame:
        """Create a comparison report of all trained models."""
        if not self.trained_models:
            raise ValueError("No models have been trained yet.")
        
        comparison_data = []
        
        for name, info in self.trained_models.items():
            if 'error' not in info:
                row = {
                    'model': name,
                    'train_r2': info['train_metrics']['r2'],
                    'test_r2': info['test_metrics']['r2'],
                    'train_rmse': info['train_metrics']['rmse'],
                    'test_rmse': info['test_metrics']['rmse'],
                    'train_mae': info['train_metrics']['mae'],
                    'test_mae': info['test_metrics']['mae'],
                    'overfit_score': info['train_metrics']['r2'] - info['test_metrics']['r2']
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('test_r2', ascending=False)
        return comparison_df

if __name__ == "__main__":
    # Test the model training pipeline
    from data_collector import DataCollector
    from feature_engineering import FeatureEngineer
    
    # Load and prepare data
    collector = DataCollector()
    df = collector.load_data()
    
    engineer = FeatureEngineer()
    X, y = engineer.prepare_features(df)
    X_train, X_test, y_train, y_test = engineer.split_data(X, y)
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Get best model
    best_name, best_model, best_info = trainer.get_best_model()
    print(f"Best model: {best_name}")
    print(f"Test R²: {best_info['test_metrics']['r2']:.4f}")
    
    # Create comparison report
    comparison = trainer.create_model_comparison_report()
    print("\nModel Comparison:")
    print(comparison)
