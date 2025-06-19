#train.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess.py import preprocess_df

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)