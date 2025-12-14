# -- coding: utf-8 --
import os
import pandas as pd
import joblib

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("skill-manual-tuning-logreg")

params_list = [
    {"C": 0.01, "solver": "lbfgs"},
    {"C": 0.1, "solver": "lbfgs"},
    {"C": 1.0, "solver": "lbfgs"},
]

for params in params_list:
    with mlflow.start_run():

        df = pd.read_csv("Predictive_Maintenance_Preproces.csv")

        X = df.drop(columns=['Target', 'Failure Type'])
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        best_f1 = 0
        best_model = None

        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            **params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n=== Training Result ===")
        print(f"Params      : {params}")
        print(f"Accuracy    : {acc:.4f}")
        print(f"Precision   : {prec:.4f}")
        print(f"Recall      : {rec:.4f}")
        print(f"F1-score    : {f1:.4f}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_y_pred = y_pred


mlflow.log_metric("best_f1_score", best_f1)

os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/best_logreg_model.pkl"
joblib.dump(best_model, model_path)
mlflow.log_artifact(model_path)

cm = confusion_matrix(y_test, best_y_pred)
print("Confusion Matrix:")
print(cm)

print("Prediction distribution:",
      np.unique(best_y_pred, return_counts=True))
