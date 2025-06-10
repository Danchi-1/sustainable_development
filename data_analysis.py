import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class AirPollutionAnalyzer:
    def __init__(self):
        """Initialize the Air Pollution Data Analyzer"""
        self.data = None
        self.predictions = None
        
    def load_data(self, data_path):
        """Load preprocessed data"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def perform_eda(self):
        """Comprehensive Exploratory Data Analysis"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        print("=== EXPLORATORY DATA ANALYSIS ===")
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Missing Values:\n{self.data.isnull().sum()}")
        print(f"Data Types:\n{self.data.dtypes}")
        print(f"Statistical Summary:\n{self.data.describe()}")
        
        # Create comprehensive visualizations
        self.create_pollution_overview()
        self.analyze_correlations()
        self.health_impact_analysis()
        self.temporal_analysis()
        
    def create_pollution_overview(self):
        """Create overview visualizations of pollution data"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Air Pollution Overview Analysis', fontsize=16, fontweight='bold')
        
        # PM2.5 Distribution
        if 'pm25' in self.data.columns:
            axes[0,0].hist(self.data['pm25'].dropna(), bins=30, alpha=0.7, color='red', edgecolor='black')
            axes[0,0].set_title('PM2.5 Concentration Distribution')
            axes[0,0].set_xlabel('PM2.5 (µg/m³)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].axvline(self.data['pm25'].mean(), color='darkred', linestyle='--', 
                             label=f'Mean: {self.data["pm25"].mean():.2f}')
            axes[0,0].legend()
        
        # Air Quality Index Distribution
        if 'AQI' in self.data.columns:
            axes[0,1].hist(self.data['AQI'].dropna(), bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[0,1].set_title('Air Quality Index Distribution')
            axes[0,1].set_xlabel('AQI')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].axvline(self.data['AQI'].mean(), color='darkorange', linestyle='--',
                             label=f'Mean: {self.data["AQI"].mean():.2f}')
            axes[0,1].legend()
        
        # Health Risk Categories
        if 'health_risk' in self.data.columns:
            risk_counts = self.data['health_risk'].value_counts()
            axes[1,0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                         colors=['green', 'yellow', 'orange', 'red'])
            axes[1,0].set_title('Health Risk Distribution')
        
        # Box plot for pollution levels by season (if available)
        if 'season' in self.data.columns and 'pm25' in self.data.columns:
            sns.boxplot(data=self.data, x='season', y='pm25', ax=axes[1,1])
            axes[1,1].set_title('PM2.5 Levels by Season')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('pollution_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_correlations(self):
        """Analyze correlations between pollution and health indicators"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, fmt='.2f')
        plt.title('Correlation Matrix: Pollution vs Health Indicators', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identify strongest correlations with health outcomes
        if 'respiratory_cases' in self.data.columns:
            health_corr = correlation_matrix['respiratory_cases'].abs().sort_values(ascending=False)
            print("\n=== STRONGEST CORRELATIONS WITH RESPIRATORY CASES ===")
            for var, corr in health_corr.head(5).items():
                if var != 'respiratory_cases':
                    print(f"{var}: {corr:.3f}")
                    
    def health_impact_analysis(self):
        """Analyze the impact of pollution on health outcomes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Pollution Impact on Health Analysis', fontsize=16, fontweight='bold')
        
        # PM2.5 vs Respiratory Cases
        if 'pm25' in self.data.columns and 'respiratory_cases' in self.data.columns:
            axes[0,0].scatter(self.data['pm25'], self.data['respiratory_cases'], alpha=0.6, color='red')
            z = np.polyfit(self.data['pm25'].dropna(), 
                          self.data['respiratory_cases'].dropna(), 1)
            p = np.poly1d(z)
            axes[0,0].plot(self.data['pm25'], p(self.data['pm25']), "r--", alpha=0.8)
            axes[0,0].set_xlabel('PM2.5 Concentration (µg/m³)')
            axes[0,0].set_ylabel('Respiratory Cases')
            axes[0,0].set_title('PM2.5 vs Respiratory Health Cases')
            
            # Calculate and display correlation
            corr = self.data['pm25'].corr(self.data['respiratory_cases'])
            axes[0,0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                          transform=axes[0,0].transAxes, fontsize=12,
                          bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # AQI Categories vs Health Risk
        if 'AQI_category' in self.data.columns and 'health_risk' in self.data.columns:
            crosstab = pd.crosstab(self.data['AQI_category'], self.data['health_risk'])
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Reds', ax=axes[0,1])
            axes[0,1].set_title('AQI Categories vs Health Risk')
            axes[0,1].set_xlabel('Health Risk Level')
            axes[0,1].set_ylabel('AQI Category')
        
        # Monthly respiratory cases trend
        if 'date' in self.data.columns and 'respiratory_cases' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            monthly_cases = self.data.groupby(self.data['date'].dt.to_period('M'))['respiratory_cases'].sum()
            axes[1,0].plot(monthly_cases.index.astype(str), monthly_cases.values, 
                          marker='o', linewidth=2, markersize=6, color='darkblue')
            axes[1,0].set_title('Monthly Respiratory Cases Trend')
            axes[1,0].set_xlabel('Month')
            axes[1,0].set_ylabel('Total Cases')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Population density vs pollution impact
        if 'population_density' in self.data.columns and 'pm25' in self.data.columns:
            axes[1,1].scatter(self.data['population_density'], self.data['pm25'], 
                             alpha=0.6, color='purple')
            axes[1,1].set_xlabel('Population Density')
            axes[1,1].set_ylabel('PM2.5 Concentration')
            axes[1,1].set_title('Population Density vs Air Pollution')
        
        plt.tight_layout()
        plt.savefig('health_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def temporal_analysis(self):
        """Analyze temporal patterns in pollution data"""
        if 'date' not in self.data.columns:
            print("No date column found for temporal analysis")
            return
            
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['hour'] = self.data['date'].dt.hour
        self.data['day_of_week'] = self.data['date'].dt.day_name()
        self.data['month'] = self.data['date'].dt.month
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Patterns in Air Pollution', fontsize=16, fontweight='bold')
        
        # Hourly pollution patterns
        if 'pm25' in self.data.columns:
            hourly_pm = self.data.groupby('hour')['pm25'].mean()
            axes[0,0].plot(hourly_pm.index, hourly_pm.values, marker='o', linewidth=2, color='red')
            axes[0,0].set_title('Average PM2.5 by Hour of Day')
            axes[0,0].set_xlabel('Hour')
            axes[0,0].set_ylabel('PM2.5 (µg/m³)')
            axes[0,0].grid(True, alpha=0.3)
        
        # Weekly pollution patterns
        if 'AQI' in self.data.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_aqi = self.data.groupby('day_of_week')['AQI'].mean().reindex(day_order)
            axes[0,1].bar(range(len(weekly_aqi)), weekly_aqi.values, color='orange', alpha=0.7)
            axes[0,1].set_title('Average AQI by Day of Week')
            axes[0,1].set_xlabel('Day of Week')
            axes[0,1].set_ylabel('AQI')
            axes[0,1].set_xticks(range(len(day_order)))
            axes[0,1].set_xticklabels(day_order, rotation=45)
        
        # Monthly trends
        if 'pm25' in self.data.columns:
            monthly_pm = self.data.groupby('month')['pm25'].mean()
            axes[1,0].plot(monthly_pm.index, monthly_pm.values, marker='s', linewidth=2, color='green')
            axes[1,0].set_title('Average PM2.5 by Month')
            axes[1,0].set_xlabel('Month')
            axes[1,0].set_ylabel('PM2.5 (µg/m³)')
            axes[1,0].set_xticks(range(1, 13))
            axes[1,0].grid(True, alpha=0.3)
        
        # Seasonal health impact
        if 'season' in self.data.columns and 'respiratory_cases' in self.data.columns:
            seasonal_health = self.data.groupby('season')['respiratory_cases'].sum()
            axes[1,1].pie(seasonal_health.values, labels=seasonal_health.index, autopct='%1.1f%%',
                         colors=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
            axes[1,1].set_title('Respiratory Cases by Season')
        
        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def evaluate_predictions(self, predictions, true_values):
        """Evaluate model predictions and create performance visualizations"""
        self.predictions = predictions
        
        # Calculate evaluation metrics
        accuracy = (predictions == true_values).mean()
        report = classification_report(true_values, predictions, output_dict=True)
        
        print("=== MODEL PERFORMANCE EVALUATION ===")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nDetailed Classification Report:")
        print(classification_report(true_values, predictions))
        
        # Confusion Matrix Visualization
        cm = confusion_matrix(true_values, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'],
                   yticklabels=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'])
        plt.title('Confusion Matrix - Health Risk Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Health Risk')
        plt.ylabel('Actual Health Risk')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        insights = []
        
        if self.data is not None:
            # Key statistics
            if 'pm25' in self.data.columns:
                avg_pm25 = self.data['pm25'].mean()
                max_pm25 = self.data['pm25'].max()
                insights.append(f"Average PM2.5 concentration: {avg_pm25:.2f} µg/m³")
                insights.append(f"Maximum PM2.5 recorded: {max_pm25:.2f} µg/m³")
                
                # WHO guideline comparison (WHO guideline: 15 µg/m³ annual mean)
                exceeding_who = (self.data['pm25'] > 15).sum()
                total_records = len(self.data)
                pct_exceeding = (exceeding_who / total_records) * 100
                insights.append(f"{pct_exceeding:.1f}% of measurements exceed WHO guidelines")
            
            if 'respiratory_cases' in self.data.columns:
                total_cases = self.data['respiratory_cases'].sum()
                avg_cases = self.data['respiratory_cases'].mean()
                insights.append(f"Total respiratory cases analyzed: {total_cases:,}")
                insights.append(f"Average cases per measurement period: {avg_cases:.1f}")
            
            # Health risk distribution
            if 'health_risk' in self.data.columns:
                risk_dist = self.data['health_risk'].value_counts(normalize=True) * 100
                for risk, pct in risk_dist.items():
                    insights.append(f"{risk} health risk: {pct:.1f}% of observations")
        
        print("\n=== KEY INSIGHTS SUMMARY ===")
        for insight in insights:
            print(f"• {insight}")
            
        return insights
    
    def save_analysis_results(self, output_dir='analysis_results'):
        """Save all analysis results and visualizations"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save insights to text file
        insights = self.generate_insights_report()
        with open(f'{output_dir}/analysis_insights.txt', 'w') as f:
            f.write("AIR POLLUTION HEALTH IMPACT ANALYSIS\n")
            f.write("="*50 + "\n\n")
            for insight in insights:
                f.write(f"• {insight}\n")
        
        print(f"Analysis results saved to {output_dir}/")

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AirPollutionAnalyzer()
    
    # Load sample data (replace with actual data path)
    # analyzer.load_data('preprocessed_air_pollution_data.csv')
    
    # Perform comprehensive analysis
    # analyzer.perform_eda()
    # insights = analyzer.generate_insights_report()
    # analyzer.save_analysis_results()
    
    print("Member 3 Analysis Module Ready!")
    print("Usage:")
    print("1. analyzer = AirPollutionAnalyzer()")
    print("2. analyzer.load_data('your_data.csv')")
    print("3. analyzer.perform_eda()")
    print("4. analyzer.generate_insights_report()")

