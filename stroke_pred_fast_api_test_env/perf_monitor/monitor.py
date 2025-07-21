import mysql.connector
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    roc_auc_score
)
import joblib
from pathlib import Path

# Load environment variables
load_dotenv()

class ModelMonitor:
    def __init__(self):
        # Database configuration (as specified)
        self.db_config = {
            'host': os.getenv("DB_HOST"),
            'port': os.getenv("DB_PORT"),
            'database': os.getenv("DB_NAME"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD"),
            'connect_timeout': 5  # Added timeout
        }
        
        # Model performance benchmarks from your training
        self.training_metrics = {
            'accuracy': 0.5646,
            'recall': 0.9355,
            'precision': 0.1162,
            'f1': 0.2068,
            'roc_auc': 0.8475
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'performance_drop': 0.15,  # 15% relative drop
            'z_score_threshold': 3.0,
            'volume_change': 0.3,
            'drift_significance': 0.01
        }
        
        # Initialize training data distributions
        self.training_distributions = None
        self._load_training_data()

    def _load_training_data(self):
        """Load training data distributions from local CSV"""
        try:
            train_df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
            self.training_distributions = {
                'age': train_df['age'],
                'avg_glucose_level': train_df['avg_glucose_level'],
                'bmi': train_df['bmi'],
                'hypertension': train_df['hypertension'].astype(int),
                'heart_disease': train_df['heart_disease'].astype(int)
            }
        except Exception as e:
            print(f"Warning: Could not load training data - drift detection limited. Error: {str(e)}")

    def _get_db_connection(self):
        """Get database connection with error handling"""
        try:
            return mysql.connector.connect(**self.db_config)
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return None

    def get_recent_predictions(self, days=7):
        """Retrieve predictions from the last N days"""
        query = """
        SELECT * FROM predictions 
        WHERE timestamp >= %s
        ORDER BY timestamp DESC
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = self._get_db_connection()
        if conn is None:
            return []
            
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(query, (cutoff_date,))
                results = cursor.fetchall()
                for row in results:
                    row['contributing_factors'] = self._safe_json_load(row.get('contributing_factors'))
                    row['prediction_data'] = self._safe_json_load(row.get('prediction_data'))
                return results
        except Exception as e:
            print(f"Error fetching predictions: {str(e)}")
            return []
        finally:
            if conn.is_connected():
                conn.close()

    def _safe_json_load(self, json_str):
        """Safely parse JSON strings"""
        if not json_str:
            return {}
        try:
            return json.loads(json_str)
        except:
            return {}

    def get_validated_predictions(self, days=7):
        """Retrieve predictions with ground truth outcomes"""
        query = """
        SELECT p.id, p.probability, t.actual_outcome 
        FROM predictions p
        JOIN true_outcomes t ON p.id = t.prediction_id
        WHERE p.timestamp >= %s
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = self._get_db_connection()
        if conn is None:
            return []
            
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(query, (cutoff_date,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error fetching validated predictions: {str(e)}")
            return []
        finally:
            if conn.is_connected():
                conn.close()

    def calculate_performance_metrics(self, validated_data):
        """Calculate current performance metrics"""
        if not validated_data:
            return {}
            
        df = pd.DataFrame(validated_data)
        df['predicted_label'] = (df['probability'] >= 0.3).astype(int)  # Using your threshold
        
        return {
            'accuracy': accuracy_score(df['actual_outcome'], df['predicted_label']),
            'recall': recall_score(df['actual_outcome'], df['predicted_label']),
            'precision': precision_score(df['actual_outcome'], df['predicted_label']),
            'f1': f1_score(df['actual_outcome'], df['predicted_label']),
            'roc_auc': roc_auc_score(df['actual_outcome'], df['probability']),
            'sample_size': len(df)
        }

    def check_data_drift(self, predictions):
        """Check for statistical drift using K-S test"""
        if not predictions or not self.training_distributions:
            return {}
            
        pred_df = pd.DataFrame([p['prediction_data'] for p in predictions])
        drift_results = {}
        
        for feature in self.training_distributions.keys():
            if feature not in pred_df.columns:
                continue
                
            prod_data = pred_df[feature].dropna().values
            if len(prod_data) < 30:  # Minimum samples
                continue
                
            # Handle binary features differently
            if feature in ['hypertension', 'heart_disease']:
                drift_results[feature] = self._check_binary_feature_drift(
                    feature, 
                    prod_data.astype(int)
                )
            else:
                # K-S test for continuous features
                ks_result = ks_2samp(
                    self.training_distributions[feature], 
                    prod_data
                )
                drift_results[feature] = {
                    'test': 'ks_2samp',
                    'D_statistic': ks_result.statistic,
                    'p_value': ks_result.pvalue,
                    'drift_detected': ks_result.pvalue < self.alert_thresholds['drift_significance'],
                    'training_mean': float(np.mean(self.training_distributions[feature])),
                    'production_mean': float(np.mean(prod_data)),
                    'training_std': float(np.std(self.training_distributions[feature])),
                    'production_std': float(np.std(prod_data))
                }
                
                # Plot distributions if drift detected
                if drift_results[feature]['drift_detected']:
                    self._plot_feature_distributions(feature, prod_data)
        
        return drift_results

    def _check_binary_feature_drift(self, feature, production_data):
        """Check drift for binary features using proportion test"""
        train_mean = np.mean(self.training_distributions[feature])
        prod_mean = np.mean(production_data)
        
        # Simple threshold-based detection for binary features
        threshold = 0.1  # 10% change in proportion
        drift_detected = abs(train_mean - prod_mean) > threshold
        
        return {
            'test': 'proportion_threshold',
            'training_proportion': train_mean,
            'production_proportion': prod_mean,
            'drift_detected': drift_detected,
            'threshold': threshold
        }

    def _plot_feature_distributions(self, feature, production_data):
        """Generate comparison plots"""
        plt.figure(figsize=(10, 6))
        
        if feature in ['hypertension', 'heart_disease']:
            # Plot proportions for binary features
            train_prop = np.mean(self.training_distributions[feature])
            prod_prop = np.mean(production_data)
            
            plt.bar(['Training', 'Production'], [train_prop, prod_prop])
            plt.ylabel('Proportion')
            plt.title(f'{feature} Prevalence Comparison')
        else:
            # KDE plots for continuous features
            sns.kdeplot(self.training_distributions[feature], 
                       label='Training', color='blue', linestyle='--')
            sns.kdeplot(production_data, 
                       label='Production', color='red')
            plt.ylabel('Density')
            plt.title(f'{feature} Distribution Comparison')
        
        plt.xlabel(feature)
        plt.legend()
        
        os.makedirs('monitoring_plots', exist_ok=True)
        plt.savefig(f'monitoring_plots/{feature}_comparison.png')
        plt.close()

    def detect_anomalies(self, current_metrics):
        """Detect performance anomalies"""
        if not current_metrics:
            return []
            
        alerts = []
        
        # Check each metric against training benchmarks
        for metric, current_value in current_metrics.items():
            if metric == 'sample_size':
                continue
                
            if metric in self.training_metrics:
                relative_diff = (self.training_metrics[metric] - current_value) / self.training_metrics[metric]
                if relative_diff > self.alert_thresholds['performance_drop']:
                    alerts.append(
                        f"Performance drop in {metric}: "
                        f"{relative_diff:.1%} decrease "
                        f"(from {self.training_metrics[metric]:.3f} to {current_value:.3f})"
                    )
        
        return alerts

    def get_prediction_counts(self, days=14):
        """Get daily prediction volumes"""
        query = """
        SELECT DATE(timestamp) as date, COUNT(*) as count 
        FROM predictions
        WHERE timestamp >= %s
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = self._get_db_connection()
        if conn is None:
            return []
            
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(query, (cutoff_date,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error fetching prediction counts: {str(e)}")
            return []
        finally:
            if conn.is_connected():
                conn.close()

    def generate_report(self, days=7):
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'time_period': f"Last {days} days",
            'status': 'success',
            'performance_metrics': None,
            'data_drift': None,
            'volume_trend': None,
            'alerts': [],
            'warnings': []
        }
        
        # 1. Get recent predictions
        predictions = self.get_recent_predictions(days)
        if not predictions:
            report['warnings'].append("No predictions found in time period")
            report['status'] = 'partial'
            return report
        
        # 2. Performance metrics
        validated_data = self.get_validated_predictions(days)
        if validated_data:
            current_metrics = self.calculate_performance_metrics(validated_data)
            report['performance_metrics'] = {
                'current': current_metrics,
                'training': self.training_metrics
            }
            report['alerts'].extend(self.detect_anomalies(current_metrics))
        else:
            report['warnings'].append("No validated predictions available")
        
        # 3. Data drift analysis
        if self.training_distributions:
            report['data_drift'] = self.check_data_drift(predictions)
            for feature, result in report['data_drift'].items():
                if result.get('drift_detected', False):
                    report['alerts'].append(
                        f"Data drift in {feature}: "
                        f"D={result.get('D_statistic', 0):.3f}, "
                        f"p={result.get('p_value', 0):.4f}"
                    )
        else:
            report['warnings'].append("Training data not available for drift detection")
        
        # 4. Volume analysis
        counts = self.get_prediction_counts(days*2)
        if counts:
            report['volume_trend'] = {
                'current': counts[-1]['count'],
                'previous': counts[-2]['count'] if len(counts) > 1 else None,
                'trend': 'increasing' if len(counts) > 1 and counts[-1]['count'] > counts[-2]['count'] else 'stable'
            }
            
            # Check for volume anomalies
            if len(counts) > 7:  # Only check if we have enough history
                z_scores = stats.zscore([x['count'] for x in counts])
                if abs(z_scores[-1]) > self.alert_thresholds['z_score_threshold']:
                    report['alerts'].append(
                        f"Prediction volume anomaly: "
                        f"z-score = {z_scores[-1]:.2f}"
                    )
        else:
            report['warnings'].append("No prediction volume data available")
        
        # Update status if we have warnings but no alerts
        if report['warnings'] and not report['alerts']:
            report['status'] = 'partial'
        
        return report

    def send_alert(self, message):
        """Send email alert if configured"""
        if not all(os.getenv(k) for k in ['SMTP_SERVER', 'SMTP_PORT', 'ALERT_EMAIL_FROM', 'ALERT_EMAIL_TO']):
            print("Email alert configuration incomplete - skipping alert")
            return
            
        try:
            msg = MIMEText(message)
            msg['Subject'] = 'Stroke Model Monitoring Alert'
            msg['From'] = os.getenv("ALERT_EMAIL_FROM")
            msg['To'] = os.getenv("ALERT_EMAIL_TO")
            
            with smtplib.SMTP(
                os.getenv("SMTP_SERVER"), 
                int(os.getenv("SMTP_PORT"))
            ) as server:
                if os.getenv("SMTP_USERNAME"):
                    server.login(
                        os.getenv("SMTP_USERNAME"), 
                        os.getenv("SMTP_PASSWORD")
                    )
                server.send_message(msg)
            print("Alert email sent successfully")
        except Exception as e:
            print(f"Failed to send alert email: {str(e)}")

def save_report(report):
    """Save report to file with timestamp"""
    os.makedirs('reports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"reports/stroke_monitor_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to {filename}")
    return filename

if __name__ == "__main__":
    print("Starting Stroke Prediction Model Monitoring...")
    
    monitor = ModelMonitor()
    report = monitor.generate_report(days=7)
    
    # Save and display report
    report_file = save_report(report)
    print("\nMonitoring Report Summary:")
    print(f"Status: {report['status'].upper()}")
    print(f"Period: {report['time_period']}")
    
    if report['performance_metrics']:
        print("\nPerformance Metrics:")
        for metric, values in report['performance_metrics']['current'].items():
            if metric in report['performance_metrics']['training']:
                train_val = report['performance_metrics']['training'][metric]
                print(f"{metric.capitalize()}: {values:.4f} (Training: {train_val:.4f})")
    
    if report['alerts']:
        print("\nALERTS:")
        for alert in report['alerts']:
            print(f"⚠️ {alert}")
        
        # Send alerts if any
        # monitor.send_alert("Stroke Model Alerts:\n\n" + "\n".join(report['alerts']))
    
    if report['warnings']:
        print("\nWarnings:")
        for warning in report['warnings']:
            print(f"⚠️ {warning}")
    
    print("\nMonitoring complete")
