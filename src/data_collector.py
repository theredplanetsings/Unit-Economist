import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

class DataCollector:
    """
    Handles data collection from various sources for apartment rent prediction.
    Sources include: Web scraping, APIs, local datasets, etc.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def collect_sample_data(self) -> pd.DataFrame:
        """
        Generate sample apartment data for initial development.
        In production, this would connect to real APIs or scrape websites.
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic apartment data
        data = {
            'bedrooms': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
            'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.3, 0.2, 0.3, 0.15, 0.05]),
            'square_feet': np.random.normal(900, 400, n_samples).clip(300, 3000),
            'floor': np.random.randint(1, 20, n_samples),
            'building_age': np.random.randint(0, 50, n_samples),
            'latitude': np.random.normal(40.7128, 0.1, n_samples),  # NYC area
            'longitude': np.random.normal(-74.0060, 0.1, n_samples),
            'has_elevator': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'has_doorman': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'has_gym': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'has_parking': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'pet_friendly': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'neighborhood_crime_rate': np.random.normal(5, 2, n_samples).clip(1, 10),
            'distance_to_subway': np.random.exponential(0.5, n_samples).clip(0.1, 5),
            'distance_to_downtown': np.random.normal(8, 4, n_samples).clip(1, 25),
            'nearby_restaurants': np.random.poisson(15, n_samples),
            'nearby_schools_rating': np.random.normal(7, 2, n_samples).clip(1, 10),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic rent prices based on features
        df['rent'] = (
            300 +  # base rent
            df['bedrooms'] * 400 +
            df['bathrooms'] * 200 +
            df['square_feet'] * 2.5 +
            df['floor'] * 10 +
            df['has_elevator'] * 100 +
            df['has_doorman'] * 200 +
            df['has_gym'] * 150 +
            df['has_parking'] * 180 +
            df['pet_friendly'] * 50 +
            (10 - df['neighborhood_crime_rate']) * 50 +
            (5 - df['distance_to_subway']) * 100 +
            (25 - df['distance_to_downtown']) * 20 +
            df['nearby_restaurants'] * 5 +
            df['nearby_schools_rating'] * 30 +
            np.random.normal(0, 200, n_samples)  # noise
        ).clip(800, 8000)
        
        # Add some categorical features
        df['building_type'] = np.random.choice(['apartment', 'condo', 'townhouse'], n_samples, p=[0.7, 0.2, 0.1])
        df['heating_type'] = np.random.choice(['central', 'radiator', 'baseboard'], n_samples, p=[0.6, 0.3, 0.1])
        
        # Save the dataset
        df.to_csv(self.data_dir / 'apartment_data.csv', index=False)
        self.logger.info(f"Generated {len(df)} sample apartment records")
        
        return df
    
    def load_data(self, filename: str = "apartment_data.csv") -> pd.DataFrame:
        """Load apartment data from file."""
        file_path = self.data_dir / filename
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            self.logger.warning(f"File {filename} not found. Generating sample data.")
            return self.collect_sample_data()

if __name__ == "__main__":
    collector = DataCollector()
    df = collector.collect_sample_data()
    print(f"Generated dataset with shape: {df.shape}")
    print(f"Average rent: ${df['rent'].mean():.2f}")
    print(f"Rent range: ${df['rent'].min():.2f} - ${df['rent'].max():.2f}")
