# -- coding: utf-8 --
import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("Predictive_Maintenance_Preproces.csv")

X = df.drop(columns=["Target", "Failure Type"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)

mlflow.sklearn.log_model(model, artifact_path="model")

os.makedirs("artifacts", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
np.savetxt("artifacts/confusion_matrix.txt", cm, fmt="%d")

with open("artifacts/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

mlflow.log_artifacts("artifacts")

print("Training & logging finished successfully")
