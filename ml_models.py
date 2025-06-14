# ml_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class AirQualityModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = ['pm25', 'pm10', 'no2', 'o3', 'population_density']
        self.target = 'aqi'

    def load_data(self, filepath='data/air_quality_processed.csv'):
        """Load processed data"""
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape}")
        return df

    def prepare_data(self, df):
        """Prepare features and target"""
        X = df[self.features]
        y = df[self.target]
        return X, y

    def train_model(self, X, y):
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained. MAE: {mae:.2f}, R2: {r2:.2f}")
        return X_test, y_test

    def save_model(self, filepath='models/air_quality_model.joblib'):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def run_pipeline(self):
        """Run complete ML pipeline"""
        # Load data
        df = self.load_data()
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Train model
        X_test, y_test = self.train_model(X, y)
        
        # Save model
        self.save_model()
        
        return X_test, y_test

if __name__ == "__main__":
    model = AirQualityModel()
    model.run_pipeline()
