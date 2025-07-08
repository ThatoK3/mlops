#!/usr/bin/env python
# coding: utf-8

# 

# In[6]:


# Install required packages
#s!pip install imbalanced-learn mlflow xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                           roc_auc_score, recall_score, f1_score,
                           precision_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")


# In[7]:


# get_ipython().run_line_magic('cd', '/home/jovyan/')


# In[8]:


# Data Loading and Preprocessing
df = pd.read_csv("work/notebook_experiments/healthcare-dataset-stroke-data.csv")

# Data Cleaning
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = df[df['gender'] != 'Other']

# Feature Selection
selected_features = ['gender', 'age', 'hypertension', 'heart_disease',
                    'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
df = df[selected_features]


# In[9]:


# Feature Engineering
df_fe = df.copy()
# 1. Age Grouping
age_bins = [0, 50, 80, 120]
age_labels = ['Young adult', 'Middle-aged', 'Very old']
df_fe['age_group'] = pd.cut(df_fe['age'], bins=age_bins, labels=age_labels, right=False)

# 2. BMI Categories
bmi_bins = [0, 18.5, 25, 30, 35, 40, 100]
bmi_labels = ['Underweight', 'Healthy Weight', 'Overweight', 'Class 1 Obesity', 'Class 2 Obesity', 'Class 3 Obesity']
df_fe['bmi_category'] = pd.cut(df_fe['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)

# 3. Interaction Feature
df_fe['age_hypertension'] = df_fe['age'] * df_fe['hypertension']

# 4. Glucose Level Binning
glucose_bins = [0, 70, 85, 100, 110, 126, 140, 300]
glucose_labels = ['Hypoglycemia', 'Low Normal', 'Normal', 'Elevated', 'Pre-diabetic', 'Borderline Diabetic', 'Diabetic']
df_fe['glucose_category'] = pd.cut(df_fe['avg_glucose_level'], bins=glucose_bins, labels=glucose_labels, right=False)


# In[10]:


# Defining categorical and numerical columns
categorical_cols = ['gender', 'smoking_status','age_group', 'bmi_category', 'glucose_category']
numerical_cols = [col for col in df_fe.columns if col not in categorical_cols + ['stroke']]


# In[11]:


# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


# In[12]:


# Train-test split
X = df_fe.drop(columns=['stroke'])
y = df_fe['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)


# In[13]:


# Train-test split
X = df_fe.drop(columns=['stroke'])
y = df_fe['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)


# In[15]:


# MLflow Experiment
mlflow.set_experiment("Stroke_Prediction_XGBoost")
mlflow.set_tracking_uri("http://103.6.171.147:5000")

# MLflow Experiment- bug workaround
mlflow.set_experiment("Stroke_Prediction_XGBoost")
mlflow.set_tracking_uri("http://103.6.171.147:5000")

mlflow.xgboost.autolog()

# In[16]:


with mlflow.start_run(run_name="Stroke_Prediction_XGBoost_v1"):
    # Simplified approach - start with basic XGBoost before grid search
    xgb_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            use_label_encoder=False,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        ))
    ])

    mlflow.set_tag("mlflow.user", "Thato")

    # First verify the pipeline works without grid search
    xgb_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_pipeline.predict(X_test)
    y_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    })

    # Log parameters
    mlflow.log_params({
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1
    })

    # Save model
    joblib.dump(xgb_pipeline, "XGBoost_Model.pkl")
    mlflow.sklearn.log_model(xgb_pipeline, "xgboost_model")

    # Print results
    print("\n--- XGBoost Classifier ---")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    # Feature Importance
    try:
        xgb_model = xgb_pipeline.named_steps['classifier']
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(xgb_model, ax=ax)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
    except Exception as e:
        print(f"Could not generate feature importance: {str(e)}")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()


# In[ ]:




