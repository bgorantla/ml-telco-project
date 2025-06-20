#train.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess_df


df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = preprocess_df(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

model = LogisticRegression(max_iter = 5000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

import mlflow
import joblib
import os

with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")
