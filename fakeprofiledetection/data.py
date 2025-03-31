import pandas as pd

# Load the preprocessed dataset
df = pd.read_csv("feature_engineered_dataset.csv")

# Extract feature names (excluding target column if present)
feature_names = df.drop(columns=["Is Fake Profile"]).columns.tolist()

print(feature_names)  # These are the features used during training
