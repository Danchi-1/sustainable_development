# ethics_sdg_integration.py
import pandas as pd
import matplotlib.pyplot as plt

class EthicsSDGAnalyzer:
    def __init__(self):
        self.data = None
        self.bias_report = {}

    def load_data(self, filepath='data/air_quality_processed.csv'):
        """Load processed data"""
        self.data = pd.read_csv(filepath)
        print(f"Data loaded: {self.data.shape}")
        return self.data

    def analyze_income_disparity(self):
        """Analyze pollution disparity by income level"""
        if 'income_level' not in self.data.columns:
            print("Income level data not available")
            return None

        result = self.data.groupby('income_level')['pm25'].mean().sort_values()
        disparity_ratio = result.max() / result.min()

        self.bias_report['income_disparity'] = {
            'disparity_ratio': disparity_ratio,
            'most_affected': result.idxmax(),
            'least_affected': result.idxmin()
        }

        # Visualization
        result.plot(kind='bar', title='Average PM2.5 by Income Level')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.tight_layout()
        plt.savefig('ethics_results/income_disparity.png')
        plt.show()

        return self.bias_report

    def check_sdg_alignment(self):
        """Check alignment with Sustainable Development Goals"""
        sdg_alignment = {
            'SDG3': {
                'target': '3.9 Reduce deaths from air pollution',
                'metrics': {
                    'high_risk_days': (self.data['health_risk'].isin(['Unhealthy', 'Very Unhealthy', 'Hazardous'])).mean(),
                    'exceeds_who_limits': (self.data['pm25'] > 25).mean()
                }
            },
            'SDG10': {
                'target': '10.2 Reduce inequalities',
                'metrics': {
                    'income_disparity_ratio': self.bias_report.get('income_disparity', {}).get('disparity_ratio', 'Not calculated')
                }
            }
        }
        
        print("\nSDG Alignment Report:")
        for goal, details in sdg_alignment.items():
            print(f"\n{goal}: {details['target']}")
            for metric, value in details['metrics'].items():
                print(f"  {metric}: {value}")

        return sdg_alignment

    def run_analysis(self):
        """Run complete ethics and SDG analysis"""
        self.load_data()
        self.analyze_income_disparity()
        self.check_sdg_alignment()

if __name__ == "__main__":
    analyzer = EthicsSDGAnalyzer()
    analyzer.run_analysis()
