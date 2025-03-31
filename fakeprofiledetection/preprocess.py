from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
# Load the preprocessed dataset
df = pd.read_csv("preprocessed_dataset.csv")

# Convert text in 'Processed Bio' into numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=100)  # Convert top 100 words into features
bio_features = tfidf.fit_transform(df["Processed Bio"]).toarray()

# Convert TF-IDF array into a DataFrame
bio_features_df = pd.DataFrame(bio_features, columns=[f"word_{i}" for i in range(bio_features.shape[1])])

# Merge TF-IDF features with original dataset (excluding 'Processed Bio' column)
df = pd.concat([df.drop(columns=["Processed Bio"]), bio_features_df], axis=1)

# Create new engagement ratio features
df["Likes per Post"] = df["Number of Likes"] / (df["Number of Posts"] + 1)
df["Comments per Post"] = df["Number of Comments"] / (df["Number of Posts"] + 1)
df["Friends per Post"] = df["Number of Friends"] / (df["Number of Posts"] + 1)

# Replace infinite values with 0
df.replace([np.inf, -np.inf], 0, inplace=True)

# Save the feature-engineered dataset
df.to_csv("feature_engineered_dataset.csv", index=False)

print("\n✅ Feature Engineering Completed! Saved as 'feature_engineered_dataset.csv'.")
print("\nFirst 5 Processed Rows:")
print(df.head())

