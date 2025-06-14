# model_evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def load_data_and_model(self, 
                          data_path='data/air_quality_processed.csv',
                          model_path='models/air_quality_model.joblib'):
        """Load test data and trained model"""
        self.data = pd.read_csv(data_path)
        self.model = joblib.load(model_path)
        print("Data and model loaded successfully")

    def evaluate_model(self):
        """Evaluate model performance"""
        X = self.data[['pm25', 'pm10', 'no2', 'o3', 'population_density']]
        y = self.data['aqi']
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = {
            'MAE': mean_absolute_error(y, y_pred),
            'R2': r2_score(y, y_pred)
        }
        
        # Store results
        self.results = metrics
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Actual vs Predicted AQI')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_results/actual_vs_predicted.png')
        plt.show()
        
        return metrics

    def generate_report(self):
        """Generate evaluation report"""
        report = {
            'model_type': 'RandomForestRegressor',
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'metrics': self.results,
            'health_impact': {
                'high_risk_accuracy': 'N/A'  # Would be calculated in real implementation
            }
        }
        
        print("\nModel Evaluation Report:")
        print("="*40)
        for key, value in report.items():
            print(f"{key}: {value}")
        
        return report

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        self.load_data_and_model()
        self.evaluate_model()
        self.generate_report()

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
