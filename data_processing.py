# data_processing.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import os

class AirQualityDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.processed_data = None
        self.feature_columns = []

    def load_data(self, filepath='data/air_quality_raw.csv'):
        """Load raw data from CSV"""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            print("Raw data not found. Generating synthetic data...")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic air quality data"""
        print("Generating synthetic air quality data...")
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        data = {
            'date': dates,
            'pm25': np.random.normal(25, 8, n_samples),
            'pm10': np.random.normal(45, 12, n_samples),
            'no2': np.random.normal(30, 10, n_samples),
            'o3': np.random.normal(0.06, 0.02, n_samples),
            'city': ['City_A'] * n_samples,
            'population_density': np.random.randint(100, 5000, n_samples),
            'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples)
        }
        
        return pd.DataFrame(data)

    def clean_data(self, df):
        """Basic data cleaning"""
        print("Cleaning data...")
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Remove extreme outliers
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[col] >= (q1 - 1.5*iqr)) & (df[col] <= (q3 + 1.5*iqr))]
            
        return df

    def calculate_aqi(self, df):
        """Calculate Air Quality Index"""
        print("Calculating AQI...")
        # Simplified AQI calculation based on PM2.5
        df['aqi'] = df['pm25'] * 2  # Simplified conversion
        df['health_risk'] = pd.cut(df['aqi'],
                                 bins=[0, 50, 100, 150, 200, 300, 500],
                                 labels=['Good', 'Moderate', 'Unhealthy for SG', 
                                         'Unhealthy', 'Very Unhealthy', 'Hazardous'])
        return df

    def save_data(self, df, filepath='data/air_quality_processed.csv'):
        """Save processed data"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def run_pipeline(self):
        """Run complete data processing pipeline"""
        print("Starting data processing pipeline...")
        
        # Step 1: Load data
        raw_data = self.load_data()
        
        # Step 2: Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # Step 3: Calculate AQI
        processed_data = self.calculate_aqi(cleaned_data)
        
        # Step 4: Save processed data
        self.save_data(processed_data)
        
        self.processed_data = processed_data
        return processed_data

if __name__ == "__main__":
    processor = AirQualityDataProcessor()
    processor.run_pipeline()
