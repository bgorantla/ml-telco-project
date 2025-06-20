import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
	df.drop("customerID", axis = 1, inplace = True)

	df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
	df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
	df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
	df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})
	df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
	df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
	df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})
	df["PhoneService"] = df["PhoneService"].map({"Yes": 1, "No": 0})

	binary_service_map = {
    'Yes': 1,
    'No': 0,
    'No phone service': -1,
    'No internet service': -1
	}

	mapped_columns = [
    	'MultipleLines',
    	'OnlineSecurity',
    	'OnlineBackup',
	    'DeviceProtection',
	    'TechSupport',
	    'StreamingTV',
	    'StreamingMovies'
	]

	# Apply the custom mapping
	for col in mapped_columns:
	    df[col] = df[col].map(binary_service_map)

	# One-hot encode 'InternetService' and 'Contract'
	df = pd.get_dummies(df, columns=['InternetService', 'Contract'], drop_first=True)
	df = pd.get_dummies(df, columns=['PaymentMethod'], drop_first=False)

	df.dropna(inplace=True)

	return df