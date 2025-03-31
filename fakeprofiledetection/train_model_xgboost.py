import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the feature-engineered dataset
df = pd.read_csv("feature_engineered_dataset.csv")

# Define features (X) and target (y)
X = df.drop(columns=["Is Fake Profile"])
y = df["Is Fake Profile"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Define hyperparameters for tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

# Perform Grid Search
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Improved Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(best_model, "improved_fake_profile_model.pkl")

print("\n✅ Model Saved as 'improved_fake_profile_model.pkl'. Ready for Deployment!")
import joblib

# Save the trained XGBoost model
with open("fake_profile_xgb.pkl", "wb") as model_file:
    joblib.dump(model, model_file)

print("\n✅ XGBoost Model Saved as 'fake_profile_xgb.pkl'. Ready for Deployment!")

