import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    roc_auc_score
)
from pathlib import Path

# Configuration
class ModelMonitor:
    def __init__(self, data_path='monitoring_data.csv'):
        self.data_path = data_path
        self.training_metrics = {
            'accuracy': 0.5646,
            'recall': 0.9355,
            'precision': 0.1162,
            'f1': 0.2068,
            'roc_auc': 0.8475
        }
        self.alert_thresholds = {
            'performance_drop': 0.15,
            'drift_significance': 0.01
        }
        
        # Load training data distributions
        self.training_distributions = self._load_training_distributions('training_data.csv')

    def _load_training_distributions(self, training_path):
        """Load distributions from training CSV"""
        try:
            train_df = pd.read_csv(training_path)
            return {
                'age': train_df['age'],
                'avg_glucose_level': train_df['avg_glucose_level'],
                'bmi': train_df['bmi'],
                'hypertension': train_df['hypertension'].astype(int),
                'heart_disease': train_df['heart_disease'].astype(int)
            }
        except Exception as e:
            print(f"Warning: Could not load training data - drift detection limited. Error: {str(e)}")
            return None

    def load_monitoring_data(self):
        """Load current monitoring data from CSV"""
        try:
            df = pd.read_csv(self.data_path)
            
            # Ensure required columns exist
            required_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 
                           'heart_disease', 'predicted_y', 'true_y']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
                
            return df
        except Exception as e:
            print(f"Error loading monitoring data: {str(e)}")
            return None

    def calculate_performance(self, df):
        """Calculate current performance metrics"""
        return {
            'accuracy': accuracy_score(df['true_y'], df['predicted_y']),
            'recall': recall_score(df['true_y'], df['predicted_y']),
            'precision': precision_score(df['true_y'], df['predicted_y']),
            'f1': f1_score(df['true_y'], df['predicted_y']),
            'roc_auc': roc_auc_score(df['true_y'], df['predicted_y']),
            'sample_size': len(df)
        }

    def check_data_drift(self, df):
        """Check feature distributions vs training data"""
        if self.training_distributions is None:
            return {}
            
        drift_results = {}
        
        for feature in self.training_distributions.keys():
            if feature not in df.columns:
                continue
                
            prod_data = df[feature].dropna().values
            if len(prod_data) < 30:
                continue
                
            if feature in ['hypertension', 'heart_disease']:
                # Binary feature check
                train_prop = np.mean(self.training_distributions[feature])
                prod_prop = np.mean(prod_data)
                drift_results[feature] = {
                    'type': 'proportion',
                    'training': train_prop,
                    'production': prod_prop,
                    'change': prod_prop - train_prop,
                    'threshold': 0.1
                }
            else:
                # Continuous feature - K-S test
                ks_result = ks_2samp(self.training_distributions[feature], prod_data)
                drift_results[feature] = {
                    'type': 'ks_test',
                    'D_statistic': ks_result.statistic,
                    'p_value': ks_result.pvalue,
                    'training_mean': np.mean(self.training_distributions[feature]),
                    'production_mean': np.mean(prod_data)
                }
                
                # Plot if significant drift
                if ks_result.pvalue < self.alert_thresholds['drift_significance']:
                    self._plot_distribution(feature, prod_data)
        
        return drift_results

    def _plot_distribution(self, feature, prod_data):
        """Generate comparison plot"""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.training_distributions[feature], label='Training', color='blue')
        sns.kdeplot(prod_data, label='Production', color='red')
        plt.title(f'Distribution Drift: {feature}')
        plt.xlabel(feature)
        plt.legend()
        
        os.makedirs('monitoring_plots', exist_ok=True)
        plt.savefig(f'monitoring_plots/{feature}_drift.png')
        plt.close()

    def generate_report(self):
        """Generate monitoring report"""
        df = self.load_monitoring_data()
        if df is None:
            return {'status': 'error', 'message': 'Failed to load data'}
            
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'performance': self.calculate_performance(df),
            'data_drift': self.check_data_drift(df),
            'alerts': []
        }
        
        # Check for performance drops
        for metric, value in report['performance'].items():
            if metric in self.training_metrics:
                drop = (self.training_metrics[metric] - value) / self.training_metrics[metric]
                if drop > self.alert_thresholds['performance_drop']:
                    report['alerts'].append(
                        f"Performance drop in {metric}: {drop:.1%} "
                        f"(from {self.training_metrics[metric]:.3f} to {value:.3f})"
                    )
        
        # Check for significant drift
        for feature, result in report['data_drift'].items():
            if result['type'] == 'ks_test' and result['p_value'] < self.alert_thresholds['drift_significance']:
                report['alerts'].append(
                    f"Data drift in {feature}: "
                    f"D={result['D_statistic']:.3f}, p={result['p_value']:.4f}"
                )
            elif result['type'] == 'proportion' and abs(result['change']) > result['threshold']:
                report['alerts'].append(
                    f"Proportion change in {feature}: "
                    f"{result['change']:.1%} (from {result['training']:.3f} to {result['production']:.3f})"
                )
        
        return report

    def save_report(self, report):
        """Save report to JSON file"""
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"reports/monitoring_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")
        return filename

if __name__ == "__main__":
    print("Starting Model Monitoring...")
    
    # Initialize with your CSV file
    monitor = ModelMonitor(data_path='monitoring_data.csv')
    
    # Generate and save report
    report = monitor.generate_report()
    monitor.save_report(report)
    
    # Print summary
    print("\n=== Monitoring Summary ===")
    print(f"Status: {report['status'].upper()}")
    print(f"Sample Size: {report['performance']['sample_size']}")
    
    print("\nPerformance Metrics:")
    for metric, value in report['performance'].items():
        if metric != 'sample_size':
            print(f"{metric.capitalize()}: {value:.4f}")
    
    if report['alerts']:
        print("\n🚨 ALERTS:")
        for alert in report['alerts']:
            print(f"- {alert}")
    else:
        print("\nNo alerts detected")
    
    print("\nMonitoring complete")
