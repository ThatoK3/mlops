FROM jupyter/base-notebook

# Switch to root to install OS packages
USER root

# Install pip and apt packages
RUN pip install --no-cache-dir \
        imbalanced-learn \
        mlflow \
        seaborn \
        lightgbm \
        xgboost \
    && apt-get update \
    && apt-get install -y curl git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the default user
# USER $NB_UID

