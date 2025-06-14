# ethics_sdg_integration.py
import pandas as pd
import matplotlib.pyplot as plt
import os

class EthicsSDGAnalyzer:
    def __init__(self):
        self.data = None
        self.bias_report = {
            'demographic_bias': {},
            'prediction_fairness': {},
            'environmental_justice': {}
        }

    def load_data(self, filepath='data/air_quality_processed.csv'):
        """Load processed data with error handling"""
        try:
            self.data = pd.read_csv(filepath)
            if self.data.empty:
                raise ValueError("Loaded an empty dataframe")
            print(f"✅ Data loaded successfully. Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.data = None
            return False

    def detect_demographic_bias(self):
        """Detect bias across demographic groups with safeguards"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None

        required_columns = {'income_level', 'pm25'}
        if not required_columns.issubset(self.data.columns):
            print(f"Missing required columns. Need: {required_columns}")
            return None

        try:
            income_pollution = self.data.groupby('income_level')['pm25'].mean()
            disparity_ratio = income_pollution.max() / income_pollution.min()

            self.bias_report['demographic_bias']['income'] = {
                'disparity_ratio': disparity_ratio,
                'most_affected': income_pollution.idxmax(),
                'least_affected': income_pollution.idxmin(),
                'group_stats': income_pollution.to_dict()
            }

            # Visualization
            os.makedirs('ethics_results', exist_ok=True)
            plt.figure(figsize=(10, 6))
            income_pollution.sort_values().plot(kind='bar', color='skyblue')
            plt.title('Average PM2.5 Exposure by Income Level')
            plt.ylabel('PM2.5 (µg/m³)')
            plt.axhline(y=25, color='r', linestyle='--', label='WHO Guideline')
            plt.legend()
            plt.tight_layout()
            plt.savefig('ethics_results/income_pollution_disparity.png')
            plt.close()
            
            print(f"Disparity Ratio: {disparity_ratio:.2f}")
            return self.bias_report

        except Exception as e:
            print(f"Error in bias detection: {e}")
            return None

    def generate_ethical_recommendations(self):
        """Generate recommendations based on bias findings"""
        if not self.bias_report.get('demographic_bias'):
            return ["Insufficient data for recommendations"]

        recommendations = []
        income_bias = self.bias_report['demographic_bias'].get('income')
        
        if income_bias and income_bias['disparity_ratio'] > 1.5:
            rec = (
                f"Address income disparities: {income_bias['most_affected']} income groups "
                f"face {income_bias['disparity_ratio']:.1f}x higher pollution exposure "
                f"than {income_bias['least_affected']} groups"
            )
            recommendations.append(rec)

        # Add universal recommendations
        recommendations.extend([
            "Implement regular environmental justice audits",
            "Prioritize pollution control in high-exposure areas",
            "Ensure transparency in environmental data reporting"
        ])

        print("\nEthical Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        return recommendations

    def save_results(self):
        """Save all results to files"""
        os.makedirs('ethics_results', exist_ok=True)
        
        try:
            # Save bias report
            pd.DataFrame(self.bias_report['demographic_bias']).to_csv(
                'ethics_results/demographic_bias.csv'
            )
            
            # Save recommendations
            with open('ethics_results/recommendations.txt', 'w') as f:
                f.write("\n".join(self.generate_ethical_recommendations()))
                
            print("✅ Results saved to ethics_results/ directory")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def run_analysis(self):
        """Complete analysis pipeline"""
        if not self.load_data():
            return

        print("\n=== Demographic Bias Analysis ===")
        self.detect_demographic_bias()
        
        print("\n=== Ethical Recommendations ===")
        self.generate_ethical_recommendations()
        
        self.save_results()

if __name__ == "__main__":
    analyzer = EthicsSDGAnalyzer()
    analyzer.run_analysis()
