import numpy as np

import pandas as pd

from sklearn.metrics import (

    mean_squared_error, mean_absolute_error, r2_score,

    classification_report, confusion_matrix, accuracy_score,

    precision_recall_fscore_support, roc_auc_score, roc_curve

)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')



class ModelEvaluator:

    """

    Comprehensive model evaluation system for air pollution prediction models

    supporting both regression and classification tasks

    """

    

    def __init__(self):

        self.evaluation_results = {}

        self.model_comparisons = {}

        

    def evaluate_regression_model(self, model_name, y_true, y_pred, model_params=None):

        """

        Evaluate regression model performance for continuous air pollution predictions

        

        Args:

            model_name (str): Name of the model being evaluated

            y_true (array): True values

            y_pred (array): Predicted values

            model_params (dict): Model parameters for documentation

        

        Returns:

            dict: Comprehensive evaluation metrics

        """

        print(f"\n{'='*50}")

        print(f"EVALUATING REGRESSION MODEL: {model_name}")

        print(f"{'='*50}")

        

        # Calculate regression metrics

        mse = mean_squared_error(y_true, y_pred)

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(y_true, y_pred)

        r2 = r2_score(y_true, y_pred)

        

        # Calculate additional metrics

        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        residuals = y_true - y_pred

        

        metrics = {

            'model_name': model_name,

            'mse': mse,

            'rmse': rmse,

            'mae': mae,

            'r2_score': r2,

            'mape': mape,

            'residuals_mean': np.mean(residuals),

            'residuals_std': np.std(residuals),

            'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            'model_params': model_params

        }

        

        # Print metrics

        print(f"Mean Squared Error (MSE): {mse:.4f}")

        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        print(f"Mean Absolute Error (MAE): {mae:.4f}")

        print(f"R² Score: {r2:.4f}")

        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

        

        # Health impact interpretation

        self._interpret_regression_results(metrics)

        

        # Store results

        self.evaluation_results[model_name] = metrics

        

        return metrics

    

    def evaluate_classification_model(self, model_name, y_true, y_pred, y_pred_proba=None, model_params=None):

        """

        Evaluate classification model performance for air quality categories

        

        Args:

            model_name (str): Name of the model being evaluated

            y_true (array): True class labels

            y_pred (array): Predicted class labels

            y_pred_proba (array): Predicted probabilities (optional)

            model_params (dict): Model parameters for documentation

        

        Returns:

            dict: Comprehensive evaluation metrics

        """

        print(f"\n{'='*50}")

        print(f"EVALUATING CLASSIFICATION MODEL: {model_name}")

        print(f"{'='*50}")

        

        # Calculate classification metrics

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        

        metrics = {

            'model_name': model_name,

            'accuracy': accuracy,

            'precision': precision,

            'recall': recall,

            'f1_score': f1,

            'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            'model_params': model_params

        }

        

        # Add AUC if probabilities are provided

        if y_pred_proba is not None:

            try:

                if len(np.unique(y_true)) == 2:  # Binary classification

                    auc = roc_auc_score(y_true, y_pred_proba[:, 1])

                    metrics['auc_roc'] = auc

                    print(f"AUC-ROC: {auc:.4f}")

            except:

                pass

        

        # Print metrics

        print(f"Accuracy: {accuracy:.4f}")

        print(f"Precision: {precision:.4f}")

        print(f"Recall: {recall:.4f}")

        print(f"F1-Score: {f1:.4f}")

        

        # Detailed classification report

        print("\nDetailed Classification Report:")

        print(classification_report(y_true, y_pred))

        

        # Health impact interpretation

        self._interpret_classification_results(metrics)

        

        # Store results

        self.evaluation_results[model_name] = metrics

        

        return metrics

    

    def cross_validation_evaluation(self, model, X, y, cv_folds=5, model_name="Model"):

        """

        Perform cross-validation evaluation

        

        Args:

            model: Trained model object

            X (array): Feature data

            y (array): Target data

            cv_folds (int): Number of cross-validation folds

            model_name (str): Name of the model

        

        Returns:

            dict: Cross-validation results

        """

        from sklearn.model_selection import cross_val_score

        

        print(f"\n{'='*50}")

        print(f"CROSS-VALIDATION EVALUATION: {model_name}")

        print(f"{'='*50}")

        

        # Determine scoring metric based on problem type

        try:

            # Try regression scoring

            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

            scoring_type = "R² Score"

        except:

            # Fall back to classification scoring

            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')

            scoring_type = "Accuracy"

        

        cv_results = {

            'model_name': model_name,

            'cv_scores': cv_scores,

            'mean_cv_score': np.mean(cv_scores),

            'std_cv_score': np.std(cv_scores),

            'scoring_metric': scoring_type,

            'cv_folds': cv_folds

        }

        

        print(f"{scoring_type} Scores: {cv_scores}")

        print(f"Mean {scoring_type}: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

        

        return cv_results

    

    def compare_models(self, model_results):

        """

        Compare multiple models and rank them by performance

        

        Args:

            model_results (dict): Dictionary of model evaluation results

        

        Returns:

            pd.DataFrame: Comparison table

        """

        print(f"\n{'='*50}")

        print("MODEL PERFORMANCE COMPARISON")

        print(f"{'='*50}")

        

        comparison_data = []

        

        for model_name, results in model_results.items():

            if 'r2_score' in results:  # Regression model

                comparison_data.append({

                    'Model': model_name,

                    'Type': 'Regression',

                    'Primary Metric': results['r2_score'],

                    'Secondary Metric': results['rmse'],

                    'Metric Names': 'R² / RMSE'

                })

            elif 'accuracy' in results:  # Classification model

                comparison_data.append({

                    'Model': model_name,

                    'Type': 'Classification',

                    'Primary Metric': results['accuracy'],

                    'Secondary Metric': results['f1_score'],

                    'Metric Names': 'Accuracy / F1'

                })

        

        comparison_df = pd.DataFrame(comparison_data)

        

        if not comparison_df.empty:

            # Sort by primary metric (higher is better for both R² and Accuracy)

            comparison_df = comparison_df.sort_values('Primary Metric', ascending=False)

            print("\nModel Rankings:")

            print(comparison_df.to_string(index=False))

            

            # Recommend best model

            best_model = comparison_df.iloc[0]

            print(f"\n🏆 RECOMMENDED MODEL: {best_model['Model']}")

            print(f"   Type: {best_model['Type']}")

            print(f"   {best_model['Metric Names']}: {best_model['Primary Metric']:.4f} / {best_model['Secondary Metric']:.4f}")

        
        return comparison_df

    def create_evaluation_visualizations(self, model_name, y_true, y_pred, save_plots=True):

        """
        Create comprehensive evaluation visualizations
        """

        print(f"\n📊 Creating evaluation visualizations for {model_name}...")

        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation: {model_name}\nSDG 3: Air Pollution Health Impact Prediction', 
                     fontsize=16, fontweight='bold')

        
        # Determine if this is regression or classification
        is_regression = len(np.unique(y_true)) > 10

        
        if is_regression:
            # Regression visualizations

            # 1. Actual vs Predicted
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='steelblue')
            axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                           'r--', lw=2, label='Perfect Prediction')
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Actual vs Predicted Values')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            
            # 2. Residuals plot
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            axes[0, 1].grid(True, alpha=0.3)

            
            # 3. Residuals distribution
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Residuals')
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            axes[1, 0].grid(True, alpha=0.3)

            
            # 4. Error metrics visualization
            metrics = ['MSE', 'RMSE', 'MAE', 'R²']
            values = [
                mean_squared_error(y_true, y_pred),
                np.sqrt(mean_squared_error(y_true, y_pred)),
                mean_absolute_error(y_true, y_pred),
                r2_score(y_true, y_pred)
            ]

            bars = axes[1, 1].bar(metrics, values, color=['red', 'orange', 'yellow', 'green'])
            axes[1, 1].set_title('Model Performance Metrics')
            axes[1, 1].set_ylabel('Metric Value')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')

        else:
            # Classification visualizations

            # 1. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')

            # 2. Class distribution
            unique, counts = np.unique(y_true, return_counts=True)
            axes[0, 1].bar(unique, counts, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Class Distribution (Actual)')
            axes[0, 1].set_xlabel('Air Quality Category')
            axes[0, 1].set_ylabel('Count')

            # 3. Prediction distribution
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            axes[1, 0].bar(unique_pred, counts_pred, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 0].set_title('Class Distribution (Predicted)')
            axes[1, 0].set_xlabel('Air Quality Category')
            axes[1, 0].set_ylabel('Count')

            # 4. Performance metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            values = [accuracy_score(y_true, y_pred), precision, recall, f1]

            bars = axes[1, 1].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
            axes[1, 1].set_title('Model Performance Metrics')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 1)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        
        if save_plots:
            plt.savefig(f'{model_name}_evaluation_plots.png', dpi=300, bbox_inches='tight')
            print(f"📈 Plots saved as '{model_name}_evaluation_plots.png'")

        
        plt.show()

    
    def _interpret_regression_results(self, metrics):
        """Interpret regression results in health context"""

        print(f"\n_HELTH IMPACT INTERPRETATION:")

        
        r2 = metrics['r2_score']
        rmse = metrics['rmse']

        
        if r2 > 0.8:
            print("✅ EXCELLENT: Model can reliably predict air pollution levels")
            print("   → High confidence for early warning systems")
            print("   → Suitable for public health policy decisions")
        elif r2 > 0.6:
            print("✅ GOOD: Model shows strong predictive capability")
            print("   → Useful for general air quality monitoring")
            print("   → Can support health advisory systems")
        elif r2 > 0.4:
            print("⚠️  MODERATE: Model has limited but useful predictive power")
            print("   → Requires additional features or data")
            print("   → Use with caution for health recommendations")
        else:
            print("❌ POOR: Model performance needs significant improvement")
            print("   → Not suitable for health-critical applications")
            print("   → Consider different algorithms or more data")

        
        print(f"\n💡 Prediction accuracy: ±{rmse:.2f} pollution units on average")
        print(f"📊 Model explains {r2*100:.1f}% of air pollution variation")

    
    def _interpret_classification_results(self, metrics):
        """Interpret classification results in health context"""

        print(f"\n_HEALTH IMPACT INTERPRETATION:")

        
        accuracy = metrics['accuracy']
        f1 = metrics['f1_score']

        
        if accuracy > 0.9:
            print("✅ EXCELLENT: Highly reliable air quality classification")
            print("   → Safe for automated health alert systems")
            print("   → Can guide immediate health recommendations")
        elif accuracy > 0.8:
            print("✅ GOOD: Reliable air quality prediction")
            print("   → Suitable for general public health advisories")
            print("   → Useful for respiratory health planning")
        elif accuracy > 0.7:
            print("⚠️  MODERATE: Decent classification performance")
            print("   → Requires human oversight for health decisions")
            print("   → Good for supplementary health information")
        else:
            print("❌ POOR: Classification needs improvement")
            print("   → Not recommended for health-critical decisions")
            print("   → Consider ensemble methods or more features")

        
        print(f"\n💡 Correctly classifies {accuracy*100:.1f}% of air quality levels")
        print(f"📊 Balanced performance score (F1): {f1:.3f}")

    
    def generate_evaluation_report(self, output_file="model_evaluation_report.txt"):
        """
        Generate comprehensive evaluation report for SDG 3 project
        """

        print(f"\n📋 Generating comprehensive evaluation report...")

        
        report_content = []
        report_content.append("="*70)
        report_content.append("SDG 3: AIR POLLUTION PREDICTION - MODEL EVALUATION REPORT")
        report_content.append("="*70)
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")

        
        report_content.append("🎯 PROJECT OBJECTIVE:")
        report_content.append("Develop AI models to predict air pollution levels and support")
        report_content.append("respiratory health protection under UN SDG 3 (Good Health & Well-being)")
        report_content.append("")

        
        report_content.append("📊 MODEL EVALUATION SUMMARY:")
        report_content.append("-" * 40)

        
        if self.evaluation_results:
            for model_name, results in self.evaluation_results.items():
                report_content.append(f"\n🤖 MODEL: {model_name}")
                report_content.append(f"   Evaluation Time: {results['evaluation_time']}")

                
                if 'r2_score' in results:
                    report_content.append(f"   Type: Regression Model")
                    report_content.append(f"   R² Score: {results['r2_score']:.4f}")
                    report_content.append(f"   RMSE: {results['rmse']:.4f}")
                    report_content.append(f"   MAE: {results['mae']:.4f}")
                    report_content.append(f"   MAPE: {results['mape']:.2f}%")
                elif 'accuracy' in results:
                    report_content.append(f"   Type: Classification Model")
                    report_content.append(f"   Accuracy: {results['accuracy']:.4f}")
                    report_content.append(f"   Precision: {results['precision']:.4f}")
                    report_content.append(f"   Recall: {results['recall']:.4f}")
                    report_content.append(f"   F1-Score: {results['f1_score']:.4f}")

        report_content.append("\n_HEALTH IMPACT ASSESSMENT:")
        report_content.append("-" * 40)
        report_content.append("• Models support early warning systems for air quality")
        report_content.append("• Predictions can guide respiratory health advisories")
        report_content.append("• AI-driven insights help reduce pollution-related illnesses")
        report_content.append("• Contributes to SDG 3 target 3.9: reduce deaths from pollution")

        
        report_content.append("\nSDG ALIGNMENT:")
        report_content.append("-" * 40)
        report_content.append("• Primary: SDG 3 (Good Health and Well-being)")
        report_content.append("• Secondary: SDG 11 (Sustainable Cities)")
        report_content.append("• Secondary: SDG 13 (Climate Action)")

        
        report_content.append("\n⚖️ ETHICAL CONSIDERATIONS:")
        report_content.append("-" * 40)
        report_content.append("• Ensure model fairness across different geographic regions")
        report_content.append("• Address potential bias in historical pollution data")
        report_content.append("• Maintain transparency in health risk communication")
        report_content.append("• Protect privacy of location-based health data")

        
        report_content.append("\n🚀 RECOMMENDATIONS:")
        report_content.append("-" * 40)
        report_content.append("• Deploy models with continuous monitoring and updates")
        report_content.append("• Integrate with public health alert systems")
        report_content.append("• Expand dataset with real-time sensor data")
        report_content.append("• Collaborate with health authorities for validation")

        
        report_content.append("\n" + "="*70)

        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_content))

        
        print(f"📄 Report saved as '{output_file}'")

        
        # Also print to console
        for line in report_content:
            print(line)


# Example usage and testing functions
def demo_evaluation_system():
    """
    Demonstrate the evaluation system with sample data
    """
    print("🚀 DEMO: SDG 3 Air Pollution Model Evaluation System")
    print("="*60)

    
    # Initialize evaluator
    evaluator = ModelEvaluator()

    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000

    
    # Sample regression data (air pollution levels)
    y_true_reg = np.random.normal(50, 20, n_samples)  # True pollution levels
    y_pred_reg = y_true_reg + np.random.normal(0, 5, n_samples)  # Predicted with some error

    # Sample classification data (air quality categories)
    y_true_class = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])  # 0: Good, 1: Moderate, 2: Poor
    y_pred_class = y_true_class.copy()
    # Add some prediction errors
    error_indices = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
    y_pred_class[error_indices] = np.random.choice([0, 1, 2], len(error_indices))

    # Evaluate regression model
    reg_results = evaluator.evaluate_regression_model(
        "Random Forest Regressor",
        y_true_reg,
        y_pred_reg,
        {"n_estimators": 100, "max_depth": 10}
    )

    # Evaluate classification model
    class_results = evaluator.evaluate_classification_model(
        "SVM Classifier",
        y_true_class,
        y_pred_class,
        model_params={"kernel": "rbf", "C": 1.0}
    )

    # Compare models
    all_results = {
        "Random Forest Regressor": reg_results,
        "SVM Classifier": class_results
    }

    comparison = evaluator.compare_models(all_results)

    # Create visualizations
    evaluator.create_evaluation_visualizations("Random Forest Regressor", y_true_reg, y_pred_reg, save_plots=False)
    evaluator.create_evaluation_visualizations("SVM Classifier", y_true_class, y_pred_class, save_plots=False)

    # Generate final report
    evaluator.generate_evaluation_report()

    return evaluator


if __name__ == "__main__":
    # Run demonstration
    demo_evaluator = demo_evaluation_system()

    print("\n✅ Member 5 Task Completed!")
    print("📋 Deliverables Ready:")
    print("   • Model evaluation framework")
    print("   • Performance metrics calculation")
    print("   • Health impact interpretation")
    print("   • Comprehensive visualizations")
    print("   • Final evaluation report")
    print("   • SDG alignment assessment")

    print(f"\n⏰ Timeline Status: Day 6 Complete")
    print("🎯 Ready for final integration and submission!")```
