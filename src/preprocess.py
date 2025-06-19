import pandas as pd

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
	df.drop("customerID", axis = 1, inplace = True)

	df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
	df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
	df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})

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

	return df