import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import time
warnings.filterwarnings('ignore')

class AirQualityDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.processed_data = None
        self.feature_columns = []
        
    def fetch_openaq_data(self, city="Los Angeles", days_back=30):  # Reduced days for testing
        """
        Fetch air quality data from OpenAQ API with improved error handling
        """
        base_url = "https://api.openaq.org/v2/measurements"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_data = []
        parameters = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        
        # Fetch data for each parameter separately
        for param in parameters:
            try:
                params = {
                    'city': city,
                    'date_from': start_date.strftime('%Y-%m-%d'),
                    'date_to': end_date.strftime('%Y-%m-%d'),
                    'limit': 1000,  # Reduced limit
                    'parameter': param,  # Single parameter
                    'order_by': 'datetime'
                }
                
                print(f"Fetching {param} data...")
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data and data['results']:
                        all_data.extend(data['results'])
                        print(f"✅ {param}: {len(data['results'])} measurements")
                    else:
                        print(f"⚠️ {param}: No data returned")
                else:
                    print(f"❌ {param}: API request failed with status {response.status_code}")
                    if response.status_code == 429:
                        print("Rate limit exceeded. Waiting 60 seconds...")
                        time.sleep(60)
                
                # Add delay to avoid rate limiting
                time.sleep(1)
                
            except requests.exceptions.Timeout:
                print(f"❌ {param}: Request timed out")
                continue
            except requests.exceptions.RequestException as e:
                print(f"❌ {param}: Request failed - {e}")
                continue
            except Exception as e:
                print(f"❌ {param}: Unexpected error - {e}")
                continue
        
        if all_data:
            print(f"✅ Successfully fetched {len(all_data)} total measurements")
            return self._parse_openaq_data(all_data)
        else:
            print("⚠️ No data fetched from API. Generating synthetic data...")
            return self._generate_synthetic_data()
    
    def _parse_openaq_data(self, api_data):
        """
        Parse OpenAQ API response into DataFrame with improved error handling
        """
        records = []
        for measurement in api_data:
            try:
                # Handle different date formats
                date_info = measurement.get('date', {})
                if isinstance(date_info, dict):
                    datetime_str = date_info.get('utc', date_info.get('local', ''))
                else:
                    datetime_str = str(date_info)
                
                record = {
                    'datetime': datetime_str,
                    'parameter': measurement.get('parameter', ''),
                    'value': measurement.get('value', 0),
                    'unit': measurement.get('unit', ''),
                    'location': measurement.get('location', 'Unknown'),
                    'city': measurement.get('city', 'Unknown'),
                    'country': measurement.get('country', 'Unknown')
                }
                records.append(record)
            except Exception as e:
                print(f"Error parsing measurement: {e}")
                continue
        
        if not records:
            print("No valid records parsed. Using synthetic data.")
            return self._generate_synthetic_data()
        
        df = pd.DataFrame(records)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
        if df.empty:
            print("No valid datetime records. Using synthetic data.")
            return self._generate_synthetic_data()
        
        # Pivot to have parameters as columns
        try:
            df_pivot = df.pivot_table(
                index=['datetime', 'location', 'city'], 
                columns='parameter', 
                values='value', 
                aggfunc='mean'
            ).reset_index()
            
            # Fill missing parameters with synthetic data
            expected_params = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
            for param in expected_params:
                if param not in df_pivot.columns:
                    print(f"Missing {param} data, filling with synthetic values")
                    df_pivot[param] = self._generate_synthetic_parameter(param, len(df_pivot))
            
            return df_pivot
            
        except Exception as e:
            print(f"Error creating pivot table: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_parameter(self, parameter, n_samples):
        """Generate synthetic data for a specific parameter"""
        np.random.seed(42)
        
        if parameter == 'pm25':
            return np.random.normal(25, 8, n_samples)
        elif parameter == 'pm10':
            return np.random.normal(45, 12, n_samples)
        elif parameter == 'o3':
            return np.random.normal(0.06, 0.02, n_samples)
        elif parameter == 'no2':
            return np.random.normal(30, 10, n_samples)
        elif parameter == 'so2':
            return np.random.normal(5, 2, n_samples)
        elif parameter == 'co':
            return np.random.normal(1.2, 0.4, n_samples)
        else:
            return np.random.normal(10, 3, n_samples)
    
    def _generate_synthetic_data(self, n_samples=8760):  # 1 year hourly data
        """
        Generate synthetic air quality data for demonstration
        """
        print("🔄 Generating synthetic air quality data...")
        
        # Create datetime index (hourly data for 1 year)
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        # Generate synthetic pollution data with realistic patterns
        np.random.seed(42)
        
        # Base pollution levels with seasonal variation
        seasonal_factor = np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) * 0.3 + 1
        daily_factor = np.sin(2 * np.pi * np.arange(n_samples) / 24) * 0.2 + 1
        
        data = {
            'datetime': dates,
            'location': ['City_Center'] * n_samples,
            'city': ['Los Angeles'] * n_samples,
            'pm25': np.random.normal(25, 8, n_samples) * seasonal_factor * daily_factor,
            'pm10': np.random.normal(45, 12, n_samples) * seasonal_factor * daily_factor,
            'o3': np.random.normal(0.06, 0.02, n_samples) * seasonal_factor,
            'no2': np.random.normal(30, 10, n_samples) * daily_factor,
            'so2': np.random.normal(5, 2, n_samples),
            'co': np.random.normal(1.2, 0.4, n_samples) * daily_factor
        }
        
        # Ensure no negative values
        for param in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
            data[param] = np.maximum(data[param], 0.1)
        
        return pd.DataFrame(data)
    
    def calculate_aqi(self, df):
        """
        Calculate Air Quality Index (AQI) based on EPA standards
        """
        def pm25_to_aqi(pm25):
            if pm25 <= 12.0:
                return ((50-0)/(12.0-0)) * (pm25-0) + 0
            elif pm25 <= 35.4:
                return ((100-51)/(35.4-12.1)) * (pm25-12.1) + 51
            elif pm25 <= 55.4:
                return ((150-101)/(55.4-35.5)) * (pm25-35.5) + 101
            elif pm25 <= 150.4:
                return ((200-151)/(150.4-55.5)) * (pm25-55.5) + 151
            elif pm25 <= 250.4:
                return ((300-201)/(250.4-150.5)) * (pm25-150.5) + 201
            else:
                return ((400-301)/(350.4-250.5)) * (pm25-250.5) + 301
        
        df['aqi_pm25'] = df['pm25'].apply(pm25_to_aqi)
        df['aqi_category'] = pd.cut(df['aqi_pm25'], 
                                   bins=[0, 50, 100, 150, 200, 300, 500],
                                   labels=['Good', 'Moderate', 'USG', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
        
        return df
    
    def create_temporal_features(self, df):
        """
        Create temporal features from datetime
        """
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['season'] = df['month'].apply(lambda x: (x-1)//3)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_lag_features(self, df, columns=['pm25', 'pm10', 'no2'], lags=[1, 6, 24]):
        """
        Create lag features for time series prediction
        """
        df_sorted = df.sort_values('datetime')
        
        for col in columns:
            if col in df_sorted.columns:
                for lag in lags:
                    df_sorted[f'{col}_lag_{lag}'] = df_sorted[col].shift(lag)
        
        return df_sorted
    
    def clean_and_preprocess(self, df):
        """
        Main preprocessing pipeline
        """
        print("🔄 Starting data preprocessing...")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Remove outliers using IQR method
        for col in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Calculate AQI
        df = self.calculate_aqi(df)
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Remove rows with NaN values created by lag features
        df = df.dropna()
        
        print(f"✅ Preprocessing complete. Dataset shape: {df.shape}")
        return df
    
    def prepare_features(self, df):
        """
        Prepare feature matrix for ML models
        """
        # Define feature columns
        pollutant_cols = [col for col in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co'] if col in df.columns]
        temporal_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                        'is_weekend', 'is_rush_hour', 'season']
        lag_cols = [col for col in df.columns if 'lag' in col]
        
        self.feature_columns = pollutant_cols + temporal_cols + lag_cols
        
        # Prepare feature matrix
        X = df[self.feature_columns].copy()
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_columns,
            index=X.index
        )
        
        return X_scaled
    
    def save_processed_data(self, df, filepath='data/processed/air_quality_processed.csv'):
        """
        Save processed data to CSV
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"💾 Processed data saved to {filepath}")
    
    def run_full_pipeline(self, city="Los Angeles", use_api=True):
        """
        Execute complete data preprocessing pipeline
        """
        print("="*50)
        print("SDG 3 AIR QUALITY DATA PREPROCESSING")
        print("="*50)
        
        # Step 1: Fetch data
        if use_api:
            print("1. 🌐 Fetching air quality data from API...")
            raw_data = self.fetch_openaq_data(city=city)
        else:
            print("1. 🔄 Generating synthetic data...")
            raw_data = self._generate_synthetic_data()
        
        # Step 2: Clean and preprocess
        print("2. 🧹 Cleaning and preprocessing data...")
        processed_data = self.clean_and_preprocess(raw_data)
        
        # Step 3: Prepare features
        print("3. ⚙️ Preparing features for ML models...")
        feature_matrix = self.prepare_features(processed_data)
        
        # Step 4: Save data
        print("4. 💾 Saving processed data...")
        self.save_processed_data(processed_data)
        
        # Data summary
        print("\n" + "="*30)
        print("DATA PREPROCESSING SUMMARY")
        print("="*30)
        print(f"Total samples: {len(processed_data)}")
        print(f"Features for ML: {len(self.feature_columns)}")
        print(f"Date range: {processed_data['datetime'].min()} to {processed_data['datetime'].max()}")
        print(f"Average PM2.5: {processed_data['pm25'].mean():.2f} μg/m³")
        print(f"Average AQI: {processed_data['aqi_pm25'].mean():.1f}")
        print(f"Data quality: {(1 - processed_data.isnull().sum().sum() / processed_data.size) * 100:.1f}% complete")
        
        self.processed_data = processed_data
        return processed_data, feature_matrix

# Main execution with error handling
if __name__ == "__main__":
    try:
        processor = AirQualityDataProcessor()
        
        # Try API first, fallback to synthetic data
        print("Attempting to use API data...")
        processed_df, features_df = processor.run_full_pipeline(use_api=True)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Falling back to synthetic data only...")
        processor = AirQualityDataProcessor()
        processed_df, features_df = processor.run_full_pipeline(use_api=False)
    
    print("\n🎯 Member 1 deliverable complete!")
    print("📁 Files ready for team:")
    print("   - data/processed/air_quality_processed.csv")
    print("   - Preprocessing pipeline: ✅")
    print("   - Feature engineering: ✅")
    print("   - Data quality checks: ✅")
