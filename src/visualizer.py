import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

class ModelVisualizer:
    """
    Creates visualizations for apartment rent prediction models.
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def plot_data_distribution(self, df: pd.DataFrame, save_path: str = None):
        """Plot distribution of key features and target variable."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Apartment Data Distribution Analysis', fontsize=16)
        
        # Rent distribution
        axes[0, 0].hist(df['rent'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Rent Distribution')
        axes[0, 0].set_xlabel('Rent ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Square feet vs rent
        axes[0, 1].scatter(df['square_feet'], df['rent'], alpha=0.5, color='coral')
        axes[0, 1].set_title('Square Feet vs Rent')
        axes[0, 1].set_xlabel('Square Feet')
        axes[0, 1].set_ylabel('Rent ($)')
        
        # Bedrooms distribution
        bedroom_counts = df['bedrooms'].value_counts().sort_index()
        axes[0, 2].bar(bedroom_counts.index, bedroom_counts.values, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Bedroom Distribution')
        axes[0, 2].set_xlabel('Number of Bedrooms')
        axes[0, 2].set_ylabel('Count')
        
        # Rent by number of bedrooms
        df.boxplot(column='rent', by='bedrooms', ax=axes[1, 0])
        axes[1, 0].set_title('Rent by Number of Bedrooms')
        axes[1, 0].set_xlabel('Number of Bedrooms')
        axes[1, 0].set_ylabel('Rent ($)')
        
        # Distance to subway vs rent
        axes[1, 1].scatter(df['distance_to_subway'], df['rent'], alpha=0.5, color='purple')
        axes[1, 1].set_title('Distance to Subway vs Rent')
        axes[1, 1].set_xlabel('Distance to Subway (miles)')
        axes[1, 1].set_ylabel('Rent ($)')
        
        # Correlation heatmap (subset of features)
        numeric_columns = ['rent', 'bedrooms', 'bathrooms', 'square_feet', 'floor', 
                          'distance_to_subway', 'distance_to_downtown', 'building_age']
        corr_matrix = df[numeric_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved data distribution plot to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_path: str = None):
        """Plot model performance comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # R² scores
        x_pos = np.arange(len(comparison_df))
        axes[0].bar(x_pos, comparison_df['test_r2'], alpha=0.7, color='skyblue', label='Test R²')
        axes[0].bar(x_pos, comparison_df['train_r2'], alpha=0.7, color='orange', label='Train R²')
        axes[0].set_title('R² Scores')
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('R² Score')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(comparison_df['model'], rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RMSE scores
        axes[1].bar(x_pos, comparison_df['test_rmse'], alpha=0.7, color='lightcoral', label='Test RMSE')
        axes[1].bar(x_pos, comparison_df['train_rmse'], alpha=0.7, color='lightgreen', label='Train RMSE')
        axes[1].set_title('RMSE Scores')
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('RMSE ($)')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(comparison_df['model'], rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Overfitting analysis
        axes[2].bar(x_pos, comparison_df['overfit_score'], alpha=0.7, color='purple')
        axes[2].set_title('Overfitting Analysis (Train R² - Test R²)')
        axes[2].set_xlabel('Models')
        axes[2].set_ylabel('Overfitting Score')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(comparison_df['model'], rotation=45)
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved model comparison plot to {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str, save_path: str = None):
        """Plot predictions vs actual values."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{model_name} - Predictions Analysis', fontsize=16)
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Rent ($)')
        axes[0].set_ylabel('Predicted Rent ($)')
        axes[0].set_title('Predicted vs Actual')
        axes[0].grid(True, alpha=0.3)
        
        # Add R² score to plot
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Rent ($)')
        axes[1].set_ylabel('Residuals ($)')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved predictions plot to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str, 
                               top_n: int = 15, save_path: str = None):
        """Plot feature importance for tree-based models."""
        if importance_df.empty:
            self.logger.warning(f"No feature importance data available for {model_name}")
            return
        
        # Take top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # Add importance values on bars
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def create_interactive_rent_map(self, df: pd.DataFrame, save_path: str = None):
        """Create interactive map showing rent distribution by location."""
        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color="rent",
            size="square_feet",
            hover_data=["bedrooms", "bathrooms", "building_type"],
            color_continuous_scale="Viridis",
            size_max=15,
            zoom=10,
            title="Apartment Rent Distribution Map"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            height=600,
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved interactive map to {save_path}")
        
        fig.show()
    
    def plot_price_trends(self, df: pd.DataFrame, save_path: str = None):
        """Plot rent trends by various features."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Rent Analysis by Features', fontsize=16)
        
        # Rent by building type
        building_type_stats = df.groupby('building_type')['rent'].agg(['mean', 'std']).reset_index()
        axes[0, 0].bar(building_type_stats['building_type'], building_type_stats['mean'], 
                      yerr=building_type_stats['std'], capsize=5, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Average Rent by Building Type')
        axes[0, 0].set_ylabel('Average Rent ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Rent by floor level
        floor_bins = pd.cut(df['floor'], bins=5)
        floor_stats = df.groupby(floor_bins)['rent'].mean()
        axes[0, 1].plot(range(len(floor_stats)), floor_stats.values, marker='o', color='red', linewidth=2)
        axes[0, 1].set_title('Average Rent by Floor Level')
        axes[0, 1].set_xlabel('Floor Range')
        axes[0, 1].set_ylabel('Average Rent ($)')
        axes[0, 1].set_xticks(range(len(floor_stats)))
        axes[0, 1].set_xticklabels([f"{int(interval.left)}-{int(interval.right)}" for interval in floor_stats.index])
        
        # Rent by building age
        age_bins = pd.cut(df['building_age'], bins=5)
        age_stats = df.groupby(age_bins)['rent'].mean()
        axes[1, 0].plot(range(len(age_stats)), age_stats.values, marker='s', color='green', linewidth=2)
        axes[1, 0].set_title('Average Rent by Building Age')
        axes[1, 0].set_xlabel('Age Range (years)')
        axes[1, 0].set_ylabel('Average Rent ($)')
        axes[1, 0].set_xticks(range(len(age_stats)))
        axes[1, 0].set_xticklabels([f"{int(interval.left)}-{int(interval.right)}" for interval in age_stats.index])
        
        # Rent distribution by amenities
        amenities = ['has_elevator', 'has_doorman', 'has_gym', 'has_parking', 'pet_friendly']
        amenity_impact = []
        
        for amenity in amenities:
            with_amenity = df[df[amenity] == 1]['rent'].mean()
            without_amenity = df[df[amenity] == 0]['rent'].mean()
            impact = with_amenity - without_amenity
            amenity_impact.append(impact)
        
        colors = ['green' if x > 0 else 'red' for x in amenity_impact]
        axes[1, 1].bar(amenities, amenity_impact, color=colors, alpha=0.7)
        axes[1, 1].set_title('Rent Impact of Amenities')
        axes[1, 1].set_ylabel('Rent Difference ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved price trends plot to {save_path}")
        
        plt.show()

if __name__ == "__main__":
    # Test visualization with sample data
    from data_collector import DataCollector
    
    collector = DataCollector()
    df = collector.load_data()
    
    visualizer = ModelVisualizer()
    visualizer.plot_data_distribution(df)
    visualizer.plot_price_trends(df)
