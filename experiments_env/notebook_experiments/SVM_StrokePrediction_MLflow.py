#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Install required packages
#!pip install imbalanced-learn mlflow


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                           roc_auc_score, recall_score, f1_score,
                           precision_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import mlflow
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# get_ipython().run_line_magic('cd', '/home/jovyan/')


# In[3]:


# Data Loading and Preprocessing
df = pd.read_csv("work/notebook_experiments/healthcare-dataset-stroke-data.csv")


# In[4]:


# Data Cleaning
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = df[df['gender'] != 'Other']

# Feature Selection
selected_features = ['gender', 'age', 'hypertension', 'heart_disease',
                    'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
df = df[selected_features]

# Feature Engineering
df_fe = df.copy()
# 1. Age Grouping
df_fe['age_group'] = pd.cut(df_fe['age'],
                           bins=[0, 50, 80, 120],
                           labels=['Young', 'Middle-aged', 'Senior'])

# 2. BMI Categories
df_fe['bmi_category'] = pd.cut(df_fe['bmi'],
                              bins=[0, 18.5, 25, 30, 35, 40, 100],
                              labels=['Underweight', 'Normal', 'Overweight',
                                     'Obese I', 'Obese II', 'Obese III'])

# 3. Glucose Categories
df_fe['glucose_category'] = pd.cut(df_fe['avg_glucose_level'],
                                  bins=[0, 70, 100, 126, 200, 300],
                                  labels=['Low', 'Normal', 'Prediabetic',
                                         'Diabetic', 'Severe'])


# In[5]:


# Identify feature types
categorical_cols = ['gender', 'smoking_status', 'age_group',
                   'bmi_category', 'glucose_category']
numerical_cols = [col for col in df_fe.columns
                 if col not in categorical_cols + ['stroke']]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


# In[6]:


# Train-test split
X = df_fe.drop(columns=['stroke'])
y = df_fe['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights
class_weights = {1: len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                0: 1.0}  # Inverse ratio for minority class


# In[9]:


# MLflow Experiment
mlflow.set_experiment("Stroke_Prediction_SVM")
mlflow.set_tracking_uri("http://103.6.171.147:5000")

# MLflow Experiment -- bug workaround
mlflow.set_experiment("Stroke_Prediction_SVM")
mlflow.set_tracking_uri("http://103.6.171.147:5000")

# In[10]:


with mlflow.start_run(run_name="Stroke_Prediction_SVM_v1"):
    # SVM Pipeline
    svm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(
            kernel='rbf',
            class_weight=class_weights,
            probability=True,  # Enable probability estimates
            random_state=42,
            gamma='scale',
            C=1.0
        ))
    ])

    mlflow.set_tag("mlflow.user", "Thato")

    # Train model
    svm_pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = svm_pipeline.predict(X_test)
    y_proba = svm_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba)
    }

    # Log parameters
    mlflow.log_params({
        'kernel': 'rbf',
        'class_weight': class_weights,
        'gamma': 'scale',
        'C': 1.0
    })

    # Log metrics
    mlflow.log_metrics(metrics)

    # Save model
    joblib.dump(svm_pipeline, "svm_model.pkl")
    mlflow.sklearn.log_model(svm_pipeline, "svm_model")

    # Results
    print("\n--- SVM Classifier ---")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")

    # Feature Coefficients (for linear kernel)
    try:
        if svm_pipeline.named_steps['classifier'].kernel == 'linear':
            coefficients = svm_pipeline.named_steps['classifier'].coef_[0]
            feature_names = (numerical_cols +
                           list(svm_pipeline.named_steps['preprocessor']
                               .named_transformers_['cat']
                               .named_steps['encoder']
                               .get_feature_names_out(categorical_cols)))

            plt.figure(figsize=(10, 6))
            plt.barh(feature_names, coefficients)
            plt.title("SVM Feature Coefficients")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()
    except Exception as e:
        print(f"Feature coefficients not available for RBF kernel: {str(e)}")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Stroke', 'Stroke'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # PR Curve
    from sklearn.metrics import PrecisionRecallDisplay
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, name="SVM")
    plt.title("Precision-Recall Curve")
    plt.savefig("pr_curve.png")
    mlflow.log_artifact("pr_curve.png")
    plt.close()


# In[ ]:




