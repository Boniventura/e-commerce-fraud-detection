#|--------------------------------------------------------------|
#|                          Requirements                        |
#|--------------------------------------------------------------|

import logging
import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

import src.common as common

#|--------------------------------------------------------------|
#|                          Macors                              |
#|--------------------------------------------------------------|

MODELS_PATH = "models/"
PLOTS_PATH = "plots/ModelEvaluation/"
METRICS_PATH = "metrics/"

#|--------------------------------------------------------------|
#|                          Main part                           |
#|--------------------------------------------------------------|

logger = logging.getLogger(__name__)

## SECTION 1: DATAPREPARATION
class DataPreparation:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = self.dataset.drop(columns=['is_fraud'])
        self.y = self.dataset['is_fraud']
        self.scaler = StandardScaler()
        self.feature_names = list(self.X.columns)

    def _train_test_split(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y)

        train_test_dict = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test}

        return train_test_dict

    def _normalization(self, train_test_dict):
        X_train_scaled = self.scaler.fit_transform(train_test_dict["X_train"])
        X_test_scaled = self.scaler.transform(train_test_dict["X_test"])

        train_test_scaled_dict = {
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled}

        return train_test_scaled_dict

    def _perform_pca(self, n_components = 5):
        pca = PCA(n_components = n_components)
        X_pca = pca.fit_transform(self.X)

        return X_pca

    def run_data_preparing(self, test_size=0.2, random_state=42):
        train_test_dict = self._train_test_split(test_size=test_size, random_state=random_state)
        train_test_scaled_dict = self._normalization(train_test_dict)
        X_pca = self._perform_pca(n_components = 5)

        fraud_count = (train_test_dict["y_train"] == 1).sum()
        non_fraud_count = (train_test_dict["y_train"] == 0).sum()
        scale_pos_weight = non_fraud_count / fraud_count if fraud_count > 0 else 1

        logger.info(f"\nClass distribution - Non-Fraud: {non_fraud_count}, Fraud: {fraud_count}")
        logger.info(f"\nScale pos weight (for imbalanced data): {scale_pos_weight:.2f}")

        prepared_data = {
            "X_train": train_test_dict["X_train"],
            "X_test": train_test_dict["X_test"],
            "y_train": train_test_dict["y_train"],
            "y_test": train_test_dict["y_test"],
            "X_pca": X_pca,
            "scale_pos_weight": scale_pos_weight,
            "feature_names": self.feature_names,
            "scaler": self.scaler}

        return prepared_data


## SECTION 2A: XGBOOST MODEL TRAINING
class XGBoostModelTraining:
    def __init__(self, prepared_data):
        self.X_train = prepared_data["X_train"]
        self.X_test = prepared_data["X_test"]
        self.y_train = prepared_data["y_train"]
        self.y_test = prepared_data["y_test"]
        self.scale_pos_weight = prepared_data["scale_pos_weight"]
        self.feature_names = prepared_data["feature_names"]
        self.pca_n_components = 5
        self.model = None

    def train_xgboost(self, params=None):
        if params is None:
            params = {'n_estimators': 10,
                    'max_depth': 2,
                    'learning_rate': 0.1,
                    'scale_pos_weight': self.scale_pos_weight,
                    'random_state': 42,
                    'eval_metric': 'logloss'}
        
        logger.info(f"Training XGBoost with parameters: {params}")

        self.pipeline = make_pipeline(StandardScaler(), PCA(n_components = self.pca_n_components), XGBClassifier(**params))
        self.pipeline.fit(self.X_train, self.y_train)

        train_score = self.pipeline.score(self.X_train, self.y_train)
        logger.info(f"XGBoost model training completed.. Train accuracy: {train_score:.4f}")

        self.model = self.pipeline
        return self.pipeline

    def save_model(self, filename="xgboost_model.pkl"):
        common.check_if_path_exists(MODELS_PATH)
        filepath = os.path.join(MODELS_PATH, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filename="xgboost_model.pkl"):
        filepath = os.path.join(MODELS_PATH, filename)
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return self.model

    def get_model(self):
        return self.model


## SECTION 2B: LOGISTIC REGRESSION MODEL TRAINING (BENCHMARK MODEL)
class LogisticRegressionBenchmark:
    def __init__(self, prepared_data):
        self.X_train = prepared_data["X_train"]
        self.X_test = prepared_data["X_test"]
        self.y_train = prepared_data["y_train"]
        self.y_test = prepared_data["y_test"]
        self.feature_names = prepared_data["feature_names"]
        self.model = None
        self.pca_n_components = 5

    def train(self, max_iter=1000, class_weight='balanced'):
        logger.info("Training Logistic Regression benchmark model...")
        
        self.model = make_pipeline(
            PCA(n_components=self.pca_n_components, random_state=42),
            LogisticRegression(max_iter=max_iter, class_weight=class_weight,
                            random_state=42, penalty='l2'))

        self.model.fit(self.X_train, self.y_train)

        pca_fitted = self.model.named_steps['pca'] 
        n_components_used = pca_fitted.n_components_

        train_accuracy = self.model.score(self.X_train, self.y_train)

        logger.info(f"PCA reduced dimensions from {self.X_train.shape[1]} to {n_components_used}.")
        logger.info(f"Logistic Regression training completed. Train accuracy: {train_accuracy:.4f}")

        return self.model

    def evaluate(self):
        logger.info("Evaluating Logistic Regression benchmark mdel...")

        y_pred = self.model.predict(self.X_test)
        y_pred_probability = self.model.predict_proba(self.X_test)[:, 1]

        metrics = {'accuracy': accuracy_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_probability)}

        logger.info("LOGISTIC REGRESSION BENCHMARK RESULTS")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

        report = classification_report(self.y_test, y_pred)

        logger.info("\nClassification Report:\n" + report)
        metrics['classification_report'] = classification_report(self.y_test, y_pred, output_dict=True)

        return metrics

    def get_model(self):
        return self.model

    def save_model(self, filename="logistic_regression_benchmark.pkl"):
        common.check_if_path_exists(MODELS_PATH)
        filepath = os.path.join(MODELS_PATH, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_model(), f)
        logger.info(f"Benchmark model saved to {filepath}")
        return filepath