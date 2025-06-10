import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class AirQualityMLModels:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_processed_data(self, filepath='data/processed/air_quality_processed.csv'):
        """
        Load preprocessed data from CSV
        """
        try:
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"Data loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            print("Processed data not found. Generating synthetic data...")
            return self._generate_model_ready_data()
    
    def _generate_model_ready_data(self):
        """
        Generate synthetic model-ready data if processed data unavailable
        """
        np.random.seed(42)
        n_samples = 5000
        
        # Generate features
        data = {
            'datetime': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'pm25': np.random.normal(25, 8, n_samples),
            'pm10': np.random.normal(45, 12, n_samples),
            'o3': np.random.normal(0.06, 0.02, n_samples),
            'no2': np.random.normal(30, 10, n_samples),
            'so2': np.random.normal(5, 2, n_samples),
            'co': np.random.normal(1.2, 0.4, n_samples),
            'hour_sin': np.random.uniform(-1, 1, n_samples),
            'hour_cos': np.random.uniform(-1, 1, n_samples),
            'month_sin': np.random.uniform(-1, 1, n_samples),
            'month_cos': np.random.uniform(-1, 1, n_samples),
            'is_weekend': np.random.binomial(1, 0.3, n_samples),
            'is_rush_hour': np.random.binomial(1, 0.2, n_samples),
            'season': np.random.randint(0, 4, n_samples)
        }
        
        # Add lag features
        for param in ['pm25', 'pm10', 'no2']:
            for lag in [1, 6, 24]:
                data[f'{param}_lag_{lag}'] = np.roll(data[param], lag)
        
        df = pd.DataFrame(data)
        
        # Calculate AQI as target
        df['aqi_pm25'] = df['pm25'] * 2 + np.random.normal(0, 5, n_samples)
        df['aqi_pm25'] = np.clip(df['aqi_pm25'], 0, 500)
        
        return df
    
    def prepare_data_for_modeling(self, df, target_column='aqi_pm25'):
        """
        Prepare features and target for ML modeling
        """
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['datetime', 'location', 'city', 'aqi_pm25', 'aqi_category']]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        print(f"Features prepared: {X.shape}")
        print(f"Target prepared: {y.shape}")
        
        return X, y, feature_cols
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.1):
        """
        Split data into train, validation, and test sets
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # Second split: separate train and validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, shuffle=True
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def initialize_models(self):
        """
        Initialize all ML models with default parameters
        """
        self.models = {
            'linear_regression': LinearRegression(),
            
            'ridge_regression': Ridge(alpha=1.0, random_state=42),
            
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
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
                random_state=42,
                eval_metric='rmse'
            ),
            
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train all models and evaluate on validation set
        """
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_pred_train)
                val_metrics = self._calculate_metrics(y_val, y_pred_val)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'predictions_val': y_pred_val
                }
                
                print(f"  Training R²: {train_metrics['r2']:.4f}")
                print(f"  Validation R²: {val_metrics['r2']:.4f}")
                print(f"  Validation MAE: {val_metrics['mae']:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics
        """
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning for best models
        """
        print("\nPerforming hyperparameter tuning...")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8]
            }
        }
        
        best_models = {}
        
        for model_name, param_grid in param_grids.items():
            if model_name in self.models:
                print(f"Tuning {model_name}...")
                
                # Use TimeSeriesSplit for time series data
                tscv = TimeSeriesSplit(n_splits=3)
                
                grid_search = GridSearchCV(
                    self.models[model_name],
                    param_grid,
                    cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                best_models[model_name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': -grid_search.best_score_
                }
                
                print(f"  Best MAE: {-grid_search.best_score_:.4f}")
                print(f"  Best params: {grid_search.best_params_}")
        
        return best_models
    
    def select_best_model(self):
        """
        Select the best performing model based on validation metrics
        """
        if not self.results:
            print("No models trained yet!")
            return None
        
        # Rank models by validation R² score
        model_scores = {}
        for name, result in self.results.items():
            model_scores[name] = result['val_metrics']['r2']
        
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = {
            'name': best_model_name,
            'model': self.results[best_model_name]['model'],
            'metrics': self.results[best_model_name]['val_metrics']
        }
        
        print(f"\nBest model: {best_model_name}")
        print(f"Validation R²: {self.best_model['metrics']['r2']:.4f}")
        print(f"Validation MAE: {self.best_model['metrics']['mae']:.4f}")
        
        return self.best_model
    
    def evaluate_on_test_set(self, X_test, y_test):
        """
        Evaluate best model on test set
        """
        if not self.best_model:
            print("No best model selected!")
            return None
        
        model = self.best_model['model']
        y_pred_test = model.predict(X_test)
        
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        print("\n" + "="*40)
        print("FINAL MODEL EVALUATION ON TEST SET")
        print("="*40)
        print(f"Model: {self.best_model['name']}")
        print(f"Test R²: {test_metrics['r2']:.4f}")
        print(f"Test MAE: {test_metrics['mae']:.4f}")
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        
        return test_metrics, y_pred_test
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from best model (if available)
        """
        if not self.best_model:
            return None
        
        model = self.best_model['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))
            
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_models(self, model_dir='models/'):
        """
        Save trained models to disk
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save best model
        if self.best_model:
            model_path = f"{model_dir}best_model_{self.best_model['name']}.joblib"
            joblib.dump(self.best_model['model'], model_path)
            print(f"Best model saved: {model_path}")
        
        # Save all models
        for name, result in self.results.items():
            model_path = f"{model_dir}{name}_model.joblib"
            joblib.dump(result['model'], model_path)
        
        print(f"All models saved to {model_dir}")
    
    def predict_air_quality(self, features, return_risk_level=True):
        """
        Make air quality predictions and return risk levels
        """
        if not self.best_model:
            print("No trained model available!")
            return None
        
        predictions = self.best_model['model'].predict(features)
        
        if return_risk_level:
            risk_levels = []
            for pred in predictions:
                if pred <= 50:
                    risk_levels.append("Good")
                elif pred <= 100:
                    risk_levels.append("Moderate")
                elif pred <= 150:
                    risk_levels.append("Unhealthy for Sensitive Groups")
                elif pred <= 200:
                    risk_levels.append("Unhealthy")
                elif pred <= 300:
                    risk_levels.append("Very Unhealthy")
                else:
                    risk_levels.append("Hazardous")
            
            return predictions, risk_levels
        
        return predictions
    
    def generate_model_summary(self):
        """
        Generate comprehensive model performance summary
        """
        summary = {
            'total_models_trained': len(self.results),
            'best_model_name': self.best_model['name'] if self.best_model else None,
            'best_model_performance': self.best_model['metrics'] if self.best_model else None,
            'all_model_performance': {}
        }
        
        for name, result in self.results.items():
            summary['all_model_performance'][name] = result['val_metrics']
        
        return summary
    
    def run_complete_ml_pipeline(self, data_path='data/processed/air_quality_processed.csv'):
        """
        Execute complete ML modeling pipeline
        """
        print("="*60)
        print("SDG 3 AIR QUALITY PREDICTION - ML MODELING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        print("1. Loading processed data...")
        df = self.load_processed_data(data_path)
        
        # Step 2: Prepare data
        print("2. Preparing data for modeling...")
        X, y, feature_names = self.prepare_data_for_modeling(df)
        
        # Step 3: Split data
        print("3. Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 4: Initialize and train models
        print("4. Initializing models...")
        self.initialize_models()
        
        print("5. Training models...")
        self.train_models(X_train, y_train, X_val, y_val)
        
        # Step 5: Hyperparameter tuning
        print("6. Performing hyperparameter tuning...")
        best_tuned_models = self.hyperparameter_tuning(X_train, y_train)
        
        # Update results with tuned models
        for name, tuned_result in best_tuned_models.items():
            y_pred_val = tuned_result['model'].predict(X_val)
            val_metrics = self._calculate_metrics(y_val, y_pred_val)
            
            self.results[f'{name}_tuned'] = {
                'model': tuned_result['model'],
                'val_metrics': val_metrics,
                'best_params': tuned_result['best_params']
            }
        
        # Step 6: Select best model
        print("7. Selecting best model...")
        self.select_best_model()
        
        # Step 7: Final evaluation
        print("8. Final evaluation on test set...")
        test_metrics, y_pred_test = self.evaluate_on_test_set(X_test, y_test)
        
        # Step 8: Feature importance
        print("9. Analyzing feature importance...")
        self.get_feature_importance(feature_names)
        
        # Step 9: Save models
        print("10. Saving models...")
        self.save_models()
        
        # Generate summary
        summary = self.generate_model_summary()
        
        print("\n" + "="*40)
        print("ML PIPELINE SUMMARY")
        print("="*40)
        print(f"✅ Models trained: {summary['total_models_trained']}")
        print(f"🏆 Best model: {summary['best_model_name']}")
        print(f"📊 Best R² score: {summary['best_model_performance']['r2']:.4f}")
        print(f"📈 Best MAE: {summary['best_model_performance']['mae']:.4f}")
        print("💾 All models saved to /models/ directory")
        
        return {
            'best_model': self.best_model,
            'test_metrics': test_metrics,
            'predictions': y_pred_test,
            'summary': summary,
            'feature_names': feature_names
        }

# Additional utility functions for the team
class ModelEvaluator:
    """
    Additional evaluation utilities for the ML models
    """
    
    @staticmethod
    def create_prediction_intervals(model, X, confidence=0.95):
        """
        Create prediction intervals for uncertainty quantification
        """
        if hasattr(model, 'estimators_'):  # For ensemble methods
            predictions = np.array([tree.predict(X) for tree in model.estimators_])
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Calculate confidence intervals
            alpha = 1 - confidence
            lower = mean_pred - 1.96 * std_pred
            upper = mean_pred + 1.96 * std_pred
            
            return mean_pred, lower, upper
        else:
            return None, None, None
    
    @staticmethod
    def calculate_health_risk_metrics(y_true, y_pred):
        """
        Calculate health-specific risk metrics
        """
        # Define AQI risk thresholds
        unhealthy_threshold = 100
        
        # True and predicted unhealthy days
        true_unhealthy = (y_true > unhealthy_threshold).astype(int)
        pred_unhealthy = (y_pred > unhealthy_threshold).astype(int)
        
        # Calculate health-related metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        health_metrics = {
            'unhealthy_day_accuracy': accuracy_score(true_unhealthy, pred_unhealthy),
            'unhealthy_day_precision': precision_score(true_unhealthy, pred_unhealthy, zero_division=0),
            'unhealthy_day_recall': recall_score(true_unhealthy, pred_unhealthy, zero_division=0),
            'unhealthy_day_f1': f1_score(true_unhealthy, pred_unhealthy, zero_division=0)
        }
        
        return health_metrics

# Main execution
if __name__ == "__main__":
    # Initialize ML modeling system
    ml_system = AirQualityMLModels()
    
    # Run complete pipeline
    results = ml_system.run_complete_ml_pipeline()
    
    print("\n🎯 Member 2 deliverable complete!")
    print("🤖 ML Models ready:")
    print("   - Multiple algorithms trained: ✅")
    print("   - Hyperparameter tuning: ✅") 
    print("   - Model evaluation: ✅")
    print("   - Best model selection: ✅")
    print("   - Feature importance analysis: ✅")
    print("   - Models saved for deployment: ✅")
    print("   - Ready for integration with health analysis!")
    
    # Display final performance
    if results['best_model']:
        print(f"\n🏆 Champion Model: {results['best_model']['name']}")
        print(f"📊 Final Test R²: {results['test_metrics']['r2']:.4f}")
        print(f"📈 Final Test MAE: {results['test_metrics']['mae']:.4f} AQI points")

