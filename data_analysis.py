# data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AirQualityAnalyzer:
    # def __init__(self):
    #     self.data = None

    def load_data(self, filepath='data/air_quality.csv'):
        """Load processed data"""
        self.data = pd.read_csv(filepath)
        print(f"Data loaded: {self.data.shape}")
        return self.data

    def basic_stats(self):
        """Show basic statistics"""
        print("\nBasic Statistics:")
        print(self.data.describe())

    def plot_pollution_trends(self):
        """Plot pollution trends"""
        plt.figure(figsize=(12, 6))
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
            
        # Plot PM2.5 over time
        plt.plot(self.data['date'], self.data['pm25'], label='PM2.5')
        plt.plot(self.data['date'], self.data['pm10'], label='PM10')
        plt.title('Air Pollution Over Time')
        plt.xlabel('Date')
        plt.ylabel('Concentration (µg/m³)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_results/pollution_trends.png')
        plt.show()

    def plot_health_risk_distribution(self):
        """Plot health risk distribution"""
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.data, x='health_risk')
        plt.title('Health Risk Level Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analysis_results/health_risk_dist.png')
        plt.show()

    def run_analysis(self):
        """Run complete analysis"""
        self.load_data()
        self.basic_stats()
        self.plot_pollution_trends()
        self.plot_health_risk_distribution()

if __name__ == "__main__":
    analyzer = AirQualityAnalyzer()
    analyzer.run_analysis()
