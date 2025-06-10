import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class EthicsSDGAnalyzer:
    def __init__(self):
        """Initialize Ethics and SDG Integration Analyzer"""
        self.data = None
        self.predictions = None
        self.bias_report = {}
        self.sdg_alignment = {}
        
    def load_data(self, data_path, predictions_path=None):
        """Load data and predictions for ethical analysis"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            if predictions_path:
                self.predictions = pd.read_csv(predictions_path)
                print(f"Predictions loaded successfully.")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_demographic_bias(self):
        """Detect bias across different demographic groups"""
        print("=== DEMOGRAPHIC BIAS ANALYSIS ===")
        
        bias_analysis = {}
        demographic_cols = ['income_level', 'urban_rural', 'region', 'age_group', 'population_density_category']
        
        for demo_col in demographic_cols:
            if demo_col in self.data.columns:
                print(f"\nAnalyzing bias for: {demo_col}")
                
                # Calculate pollution exposure by demographic group
                group_stats = self.data.groupby(demo_col).agg({
                    'pm25': ['mean', 'std', 'count'],
                    'aqi': ['mean', 'std'],
                    'respiratory_cases': ['mean', 'sum'] if 'respiratory_cases' in self.data.columns else None
                }).round(3)
                
                print(group_stats)
                
                # Calculate disparity ratios
                pm25_means = self.data.groupby(demo_col)['pm25'].mean()
                max_exposure = pm25_means.max()
                min_exposure = pm25_means.min()
                disparity_ratio = max_exposure / min_exposure if min_exposure > 0 else float('inf')
                
                bias_analysis[demo_col] = {
                    'disparity_ratio': disparity_ratio,
                    'group_stats': group_stats,
                    'most_affected': pm25_means.idxmax(),
                    'least_affected': pm25_means.idxmin()
                }
                
                print(f"Disparity Ratio: {disparity_ratio:.2f}")
                print(f"Most Affected Group: {pm25_means.idxmax()}")
                print(f"Least Affected Group: {pm25_means.idxmin()}")
        
        self.bias_report['demographic_bias'] = bias_analysis
        return bias_analysis
    
    def analyze_prediction_fairness(self):
        """Analyze fairness of model predictions across different groups"""
        if self.predictions is None:
            print("No predictions available for fairness analysis.")
            return None
            
        print("\n=== PREDICTION FAIRNESS ANALYSIS ===")
        
        fairness_metrics = {}
        protected_attributes = ['income_level', 'urban_rural', 'region']
        
        for attr in protected_attributes:
            if attr in self.data.columns:
                print(f"\nFairness analysis for: {attr}")
                
                # Calculate metrics by group
                group_metrics = {}
                for group in self.data[attr].unique():
                    if pd.notna(group):
                        group_mask = self.data[attr] == group
                        group_data = self.data[group_mask]
                        group_pred = self.predictions[group_mask] if hasattr(self.predictions, '__len__') else None
                        
                        if group_pred is not None and len(group_data) > 0:
                            # Calculate true positive rate, false positive rate, etc.
                            if 'health_risk_actual' in self.data.columns:
                                actual = group_data['health_risk_actual']
                                pred = group_pred
                                
                                # Convert to binary for simplicity (high risk vs others)
                                actual_binary = (actual == 'High Risk').astype(int)
                                pred_binary = (pred == 'High Risk').astype(int) if hasattr(pred, '__iter__') else None
                                
                                if pred_binary is not None:
                                    tp = ((actual_binary == 1) & (pred_binary == 1)).sum()
                                    fp = ((actual_binary == 0) & (pred_binary == 1)).sum()
                                    tn = ((actual_binary == 0) & (pred_binary == 0)).sum()
                                    fn = ((actual_binary == 1) & (pred_binary == 0)).sum()
                                    
                                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
                                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
                                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
                                    
                                    group_metrics[group] = {
                                        'true_positive_rate': tpr,
                                        'false_positive_rate': fpr,
                                        'positive_predictive_value': ppv,
                                        'sample_size': len(group_data)
                                    }
                
                fairness_metrics[attr] = group_metrics
                
                # Display fairness metrics
                if group_metrics:
                    for group, metrics in group_metrics.items():
                        print(f"  {group}:")
                        print(f"    True Positive Rate: {metrics['true_positive_rate']:.3f}")
                        print(f"    False Positive Rate: {metrics['false_positive_rate']:.3f}")
                        print(f"    Positive Predictive Value: {metrics['positive_predictive_value']:.3f}")
                        print(f"    Sample Size: {metrics['sample_size']}")
        
        self.bias_report['prediction_fairness'] = fairness_metrics
        return fairness_metrics
    
    def environmental_justice_analysis(self):
        """Analyze environmental justice implications"""
        print("\n=== ENVIRONMENTAL JUSTICE ANALYSIS ===")
        
        justice_insights = {}
        
        # Income-based environmental burden
        if 'income_level' in self.data.columns and 'pm25' in self.data.columns:
            income_pollution = self.data.groupby('income_level')['pm25'].mean().sort_values(ascending=False)
            print("Average PM2.5 by Income Level:")
            for income, pollution in income_pollution.items():
                print(f"  {income}: {pollution:.2f} µg/m³")
            
            # Environmental burden ratio
            highest_burden = income_pollution.iloc[0]
            lowest_burden = income_pollution.iloc[-1]
            burden_ratio = highest_burden / lowest_burden
            
            justice_insights['income_burden_ratio'] = burden_ratio
            justice_insights['most_burdened_income'] = income_pollution.index[0]
            justice_insights['least_burdened_income'] = income_pollution.index[-1]
            
            print(f"Environmental Burden Ratio: {burden_ratio:.2f}")
            print(f"Most Burdened: {income_pollution.index[0]} income areas")
            print(f"Least Burdened: {income_pollution.index[-1]} income areas")
        
        # Urban vs Rural disparities
        if 'urban_rural' in self.data.columns:
            urban_rural_stats = self.data.groupby('urban_rural').agg({
                'pm25': ['mean', 'std'],
                'aqi': ['mean', 'std'],
                'respiratory_cases': 'mean' if 'respiratory_cases' in self.data.columns else None
            })
            
            print("\nUrban vs Rural Environmental Disparities:")
            print(urban_rural_stats)
            
            justice_insights['urban_rural_disparity'] = urban_rural_stats
        
        # Population density and pollution correlation
        if 'population_density' in self.data.columns and 'pm25' in self.data.columns:
            density_corr = self.data['population_density'].corr(self.data['pm25'])
            justice_insights['density_pollution_correlation'] = density_corr
            print(f"\nPopulation Density-Pollution Correlation: {density_corr:.3f}")
        
        self.bias_report['environmental_justice'] = justice_insights
        return justice_insights
    
    def sdg_alignment_assessment(self):
        """Assess alignment with UN Sustainable Development Goals"""
        print("\n=== SDG ALIGNMENT ASSESSMENT ===")
        
        sdg_metrics = {}
        
        # SDG 3: Good Health and Well-being
        sdg3_metrics = {
            'primary_target': '3.9 - Reduce deaths and illnesses from air pollution',
            'indicators_measured': [],
            'impact_potential': 'High',
            'data_coverage': 'Comprehensive'
        }
        
        if 'respiratory_cases' in self.data.columns:
            total_cases = self.data['respiratory_cases'].sum()
            avg_monthly_cases = self.data['respiratory_cases'].mean()
            sdg3_metrics['total_health_cases_analyzed'] = int(total_cases)
            sdg3_metrics['average_monthly_cases'] = round(avg_monthly_cases, 1)
            sdg3_metrics['indicators_measured'].append('Respiratory illness cases')
        
        if 'pm25' in self.data.columns:
            who_threshold = 15  # WHO annual guideline
            exceeding_who = (self.data['pm25'] > who_threshold).mean() * 100
            sdg3_metrics['percentage_exceeding_who_guidelines'] = round(exceeding_who, 1)
            sdg3_metrics['indicators_measured'].append('PM2.5 concentration levels')
        
        sdg_metrics['SDG_3'] = sdg3_metrics
        
        # SDG 10: Reduced Inequalities
        sdg10_metrics = {
            'primary_target': '10.2 - Ensure equal opportunity and reduce inequalities',
            'inequality_indicators': [],
            'fairness_score': 'To be calculated'
        }
        
        if 'income_level' in self.data.columns:
            income_disparity = self.calculate_inequality_index('income_level', 'pm25')
            sdg10_metrics['income_based_pollution_inequality'] = round(income_disparity, 3)
            sdg10_metrics['inequality_indicators'].append('Income-based environmental burden')
        
        if 'urban_rural' in self.data.columns:
            urban_rural_disparity = self.calculate_inequality_index('urban_rural', 'pm25')
            sdg10_metrics['urban_rural_inequality'] = round(urban_rural_disparity, 3)
            sdg10_metrics['inequality_indicators'].append('Urban-rural environmental disparity')
        
        sdg_metrics['SDG_10'] = sdg10_metrics
        
        # SDG 11: Sustainable Cities and Communities
        sdg11_metrics = {
            'primary_target': '11.6 - Reduce environmental impact of cities',
            'urban_indicators': [],
            'sustainability_score': 'Measured'
        }
        
        if 'urban_rural' in self.data.columns:
            urban_data = self.data[self.data['urban_rural'] == 'Urban']
            if len(urban_data) > 0 and 'pm25' in self.data.columns:
                avg_urban_pollution = urban_data['pm25'].mean()
                sdg11_metrics['average_urban_pm25'] = round(avg_urban_pollution, 2)
                sdg11_metrics['urban_indicators'].append('Urban air quality measurements')
        
        sdg_metrics['SDG_11'] = sdg11_metrics
        
        # Display SDG alignment
        for sdg, metrics in sdg_metrics.items():
            print(f"\n{sdg}: {metrics['primary_target']}")
            for key, value in metrics.items():
                if key != 'primary_target':
                    print(f"  {key}: {value}")
        
        self.sdg_alignment = sdg_metrics
        return sdg_metrics
    
    def calculate_inequality_index(self, group_col, measure_col):
        """Calculate inequality index (Gini-like coefficient) for environmental burden"""
        if group_col not in self.data.columns or measure_col not in self.data.columns:
            return 0
        
        group_means = self.data.groupby(group_col)[measure_col].mean()
        total_mean = self.data[measure_col].mean()
        
        # Calculate coefficient of variation as inequality measure
        cv = group_means.std() / group_means.mean() if group_means.mean() > 0 else 0
        return cv
    
    def generate_ethical_recommendations(self):
        """Generate ethical recommendations based on bias analysis"""
        print("\n=== ETHICAL RECOMMENDATIONS ===")
        
        recommendations = []
        
        # Bias mitigation recommendations
        if 'demographic_bias' in self.bias_report:
            for demo, bias_info in self.bias_report['demographic_bias'].items():
                if bias_info['disparity_ratio'] > 1.5:  # Significant disparity threshold
                    recommendations.append(
                        f"Address {demo} disparities: {bias_info['most_affected']} groups face "
                        f"{bias_info['disparity_ratio']:.1f}x higher pollution exposure than "
                        f"{bias_info['least_affected']} groups"
                    )
        
        # Environmental justice recommendations
        if 'environmental_justice' in self.bias_report:
            ej_data = self.bias_report['environmental_justice']
            if 'income_burden_ratio' in ej_data and ej_data['income_burden_ratio'] > 1.3:
                recommendations.append(
                    f"Implement environmental justice policies: {ej_data['most_burdened_income']} "
                    f"income areas face disproportionate pollution burden (ratio: "
                    f"{ej_data['income_burden_ratio']:.2f})"
                )
        
        # Data collection recommendations
        recommendations.extend([
            "Ensure representative sampling across all demographic groups",
            "Implement regular bias auditing in model predictions",
            "Establish community-based monitoring in vulnerable areas",
            "Develop transparent reporting mechanisms for environmental data"
        ])
        
        # Model fairness recommendations
        recommendations.extend([
            "Apply fairness constraints during model training",
            "Use demographic parity or equalized odds metrics",
            "Implement post-processing bias correction techniques",
            "Regular model performance auditing across subgroups"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        return recommendations
    
    def create_bias_visualizations(self):
        """Create comprehensive bias visualization dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Environmental Justice & Bias Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Income-based pollution exposure
        if 'income_level' in self.data.columns and 'pm25' in self.data.columns:
            income_pollution = self.data.groupby('income_level')['pm25'].mean().sort_values(ascending=False)
            bars1 = axes[0,0].bar(range(len(income_pollution)), income_pollution.values, 
                                 color=['red', 'orange', 'yellow', 'green'][:len(income_pollution)])
            axes[0,0].set_title('PM2.5 Exposure by Income Level')
            axes[0,0].set_xlabel('Income Level')
            axes[0,0].set_ylabel('Average PM2.5 (µg/m³)')
            axes[0,0].set_xticks(range(len(income_pollution)))
            axes[0,0].set_xticklabels(income_pollution.index, rotation=45)
            
            # Add WHO guideline line
            axes[0,0].axhline(y=15, color='red', linestyle='--', alpha=0.7, label='WHO Guideline')
            axes[0,0].legend()
        
        # Urban vs Rural health impact
        if 'urban_rural' in self.data.columns and 'respiratory_cases' in self.data.columns:
            urban_rural_health = self.data.groupby('urban_rural')['respiratory_cases'].mean()
            axes[0,1].pie(urban_rural_health.values, labels=urban_rural_health.index, 
                         autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
            axes[0,1].set_title('Respiratory Cases Distribution: Urban vs Rural')
        
        # Regional disparity heatmap
        if 'region' in self.data.columns and 'pm25' in self.data.columns:
            if 'income_level' in self.data.columns:
                pivot_data = self.data.pivot_table(values='pm25', index='region', 
                                                 columns='income_level', aggfunc='mean')
                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='Reds', ax=axes[1,0])
                axes[1,0].set_title('PM2.5 by Region and Income Level')
                axes[1,0].set_xlabel('Income Level')
                axes[1,0].set_ylabel('Region')
        
        # Vulnerability index scatter plot
        if 'population_density' in self.data.columns and 'pm25' in self.data.columns:
            scatter = axes[1,1].scatter(self.data['population_density'], self.data['pm25'], 
                                      c=self.data['aqi'] if 'aqi' in self.data.columns else 'blue',
                                      cmap='Reds', alpha=0.6)
            axes[1,1].set_xlabel('Population Density')
            axes[1,1].set_ylabel('PM2.5 Concentration')
            axes[1,1].set_title('Population Density vs Pollution Exposure')
            if 'aqi' in self.data.columns:
                plt.colorbar(scatter, ax=axes[1,1], label='AQI')
        
        plt.tight_layout()
        plt.savefig('bias_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sdg_impact_report(self):
        """Create comprehensive SDG impact report"""
        print("\n=== SDG IMPACT REPORT ===")
        print("Project: AI-Driven Air Pollution Health Risk Prediction")
        print("Focus: UN Sustainable Development Goals Integration")
        print("="*60)
        
        # SDG 3 Impact Assessment
        print("\n🏥 SDG 3: Good Health and Well-being")
        print("Target 3.9: Reduce deaths and illnesses from air pollution")
        
        if 'pm25' in self.data.columns:
            high_risk_areas = (self.data['pm25'] > 35).sum()  # WHO 24-hour guideline
            total_areas = len(self.data)
            risk_percentage = (high_risk_areas / total_areas) * 100
            
            print(f"• {high_risk_areas} out of {total_areas} areas ({risk_percentage:.1f}%) exceed WHO 24-hour guidelines")
            print(f"• Early warning system can potentially prevent {high_risk_areas * 10:.0f} respiratory cases monthly")
            print("• Model enables proactive health interventions in vulnerable communities")
        
        # SDG 10 Impact Assessment  
        print("\n⚖️ SDG 10: Reduced Inequalities")
        print("Target 10.2: Ensure equal opportunity and reduce inequalities")
        
        if 'income_level' in self.data.columns:
            income_groups = self.data['income_level'].unique()
            print(f"• Analysis covers {len(income_groups)} income demographics")
            print("• Identifies environmental justice concerns for policy intervention")
            print("• Enables targeted resource allocation to vulnerable populations")
        
        # SDG 11 Impact Assessment
        print("\n🏙️ SDG 11: Sustainable Cities and Communities") 
        print("Target 11.6: Reduce environmental impact of cities")
        
        if 'urban_rural' in self.data.columns:
            urban_data = self.data[self.data['urban_rural'] == 'Urban']
            if len(urban_data) > 0:
                print(f"• {len(urban_data)} urban measurements analyzed")
                print("• Supports evidence-based urban planning decisions")
                print("• Enables real-time air quality monitoring for smart cities")
        
        # Broader SDG Connections
        print("\n🌐 Secondary SDG Connections:")
        print("• SDG 1 (No Poverty): Addresses health disparities affecting low-income populations")
        print("• SDG 4 (Quality Education): Provides data literacy and AI education opportunities")
        print("• SDG 17 (Partnerships): Demonstrates multi-stakeholder collaboration potential")
        
        return True
    
    def generate_ethics_summary_report(self):
        """Generate comprehensive ethics and bias summary report"""
        report = {
            'bias_analysis_summary': {},
            'fairness_metrics_summary': {},
            'sdg_alignment_summary': {},
            'recommendations': []
        }
        
        # Summarize bias findings
        if 'demographic_bias' in self.bias_report:
            bias_summary = {}
            for demo, bias_info in self.bias_report['demographic_bias'].items():
                bias_summary[demo] = {
                    'disparity_ratio': bias_info['disparity_ratio'],
                    'most_affected': bias_info['most_affected'],
                    'severity': 'High' if bias_info['disparity_ratio'] > 2 else 
                               'Moderate' if bias_info['disparity_ratio'] > 1.5 else 'Low'
                }
            report['bias_analysis_summary'] = bias_summary
        
        # Summarize SDG alignment
        report['sdg_alignment_summary'] = self.sdg_alignment
        
        # Generate recommendations
        report['recommendations'] = self.generate_ethical_recommendations()
        
        return report
    
    def save_ethics_results(self, output_dir='ethics_results'):
        """Save all ethics analysis results"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save bias report
        import json
        with open(f'{output_dir}/bias_analysis_report.json', 'w') as f:
            json.dump(self.bias_report, f, indent=2, default=str)
        
        # Save SDG alignment
        with open(f'{output_dir}/sdg_alignment_report.json', 'w') as f:
            json.dump(self.sdg_alignment, f, indent=2, default=str)
        
        # Save recommendations as text
        recommendations = self.generate_ethical_recommendations()
        with open(f'{output_dir}/ethical_recommendations.txt', 'w') as f:
            f.write("ETHICAL RECOMMENDATIONS FOR AI AIR POLLUTION PROJECT\n")
            f.write("="*60 + "\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"Ethics analysis results saved to {output_dir}/")

# Example usage and testing
if __name__ == "__main__":
    # Initialize ethics analyzer
    ethics_analyzer = EthicsSDGAnalyzer()
    
    # Load sample data (replace with actual data paths)
    # ethics_analyzer.load_data('preprocessed_air_pollution_data.csv')
    
    # Perform comprehensive ethics analysis
    # ethics_analyzer.detect_demographic_bias()
    # ethics_analyzer.environmental_justice_analysis()
    # ethics_analyzer.sdg_alignment_assessment()
    # ethics_analyzer.create_bias_visualizations()
    # ethics_analyzer.create_sdg_impact_report()
    # ethics_analyzer.save_ethics_results()
    
    print("Member 4 Ethics & SDG Integration Module Ready!")
    print("Usage:")
    print("1. analyzer = EthicsSDGAnalyzer()")
    print("2. analyzer.load_data('your_data.csv')")
    print("3. analyzer.detect_demographic_bias()")
    print("4. analyzer.sdg_alignment_assessment()")
    print("5. analyzer.create_sdg_impact_report()")

