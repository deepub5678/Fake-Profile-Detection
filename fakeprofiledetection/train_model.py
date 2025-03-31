import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the feature-engineered dataset
df = pd.read_csv("feature_engineered_dataset.csv")

# Define features (X) and target (y)
X = df.drop(columns=["Is Fake Profile"])  # Features
y = df["Is Fake Profile"]  # Target variable

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Training Completed! Accuracy: {accuracy:.2f}\n")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open("fake_profile_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("\n✅ Model Saved as 'fake_profile_model.pkl'. Ready for Deployment!")
