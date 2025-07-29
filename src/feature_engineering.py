import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import logging

class FeatureEngineer:
    """
    Handles feature engineering for apartment rent prediction.
    Includes feature creation, scaling, encoding, and selection.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'rent'
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        df_processed = df.copy()
        
        # Price per square foot (only if rent column exists)
        if 'rent' in df_processed.columns:
            df_processed['price_per_sqft'] = df_processed['rent'] / df_processed['square_feet']
        
        # Room ratios
        df_processed['bathroom_bedroom_ratio'] = df_processed['bathrooms'] / (df_processed['bedrooms'] + 1)
        df_processed['sqft_per_room'] = df_processed['square_feet'] / (df_processed['bedrooms'] + df_processed['bathrooms'])
        
        # Location features
        df_processed['manhattan_distance'] = abs(df_processed['latitude'] - 40.7831) + abs(df_processed['longitude'] + 73.9712)
        
        # Convenience score
        amenity_columns = ['has_elevator', 'has_doorman', 'has_gym', 'has_parking', 'pet_friendly']
        df_processed['amenity_score'] = df_processed[amenity_columns].sum(axis=1)
        
        # Age categories
        df_processed['building_age_category'] = pd.cut(
            df_processed['building_age'], 
            bins=[0, 5, 15, 30, 50], 
            labels=['new', 'modern', 'established', 'old']
        )
        
        # Floor categories
        df_processed['floor_category'] = pd.cut(
            df_processed['floor'],
            bins=[0, 3, 7, 15, 50],
            labels=['low', 'mid', 'high', 'penthouse']
        )
        
        # Neighbourhood desirability score
        df_processed['neighbourhood_score'] = (
            (10 - df_processed['neighbourhood_crime_rate']) * 0.3 +
            (5 - df_processed['distance_to_subway']) * 0.2 +
            (25 - df_processed['distance_to_downtown']) * 0.1 +
            (df_processed['nearby_restaurants'] / 30) * 0.2 +
            (df_processed['nearby_schools_rating'] / 10) * 0.2
        ).clip(0, 10)
        
        self.logger.info(f"Created features. Dataset shape: {df_processed.shape}")
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()
        
        categorical_columns = ['building_type', 'heating_type', 'building_age_category', 'floor_category']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame, fit_scalers: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete feature preparation pipeline."""
        
        # Create new features
        df_processed = self.create_features(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_processed)
        
        # Separate features and target
        if self.target_column in df_encoded.columns:
            y = df_encoded[self.target_column]
            X = df_encoded.drop(columns=[self.target_column])
        else:
            y = None
            X = df_encoded
        
        # Remove non-numeric columns for modeling
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]
        
        # Handle missing values
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        # Scale features
        if fit_scalers:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_numeric),
                columns=X_numeric.columns,
                index=X_numeric.index
            )
            self.feature_columns = X_numeric.columns.tolist()
        else:
            # When not fitting scalers, ensure we only use columns that were in training
            if self.feature_columns:
                # Keep only the features that were used in training
                missing_features = set(self.feature_columns) - set(X_numeric.columns)
                for feature in missing_features:
                    X_numeric[feature] = 0  # Add missing features with default value
                
                # Reorder columns to match training order
                X_numeric = X_numeric[self.feature_columns]
            
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_numeric),
                columns=X_numeric.columns,
                index=X_numeric.index
            )
        
        self.logger.info(f"Prepared {len(X_scaled.columns)} features for modeling")
        
        return X_scaled, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def get_feature_importance_names(self) -> List[str]:
        """Get the names of features used in modeling."""
        return self.feature_columns if self.feature_columns else []

if __name__ == "__main__":
    # Test the feature engineering pipeline
    from data_collector import DataCollector
    
    collector = DataCollector()
    df = collector.load_data()
    
    engineer = FeatureEngineer()
    X, y = engineer.prepare_features(df)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {engineer.get_feature_importance_names()}")
