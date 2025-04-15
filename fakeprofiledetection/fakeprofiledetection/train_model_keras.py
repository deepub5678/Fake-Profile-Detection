import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
df = pd.read_csv("feature_engineered_dataset.csv")  # Adjust path if needed
df["Friends per Post"] = (df["Friends per Post"] > 10).astype(int)


# 2. Feature/Label Split
X = df.drop("Friends per Post", axis=1)
y = df["Friends per Post"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build Keras Model
model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # No activation
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Use mean squared error

# 6. Train the Model
history = model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 7. Predict on Test Set (Fixed Here)
y_pred_probs = model.predict(X_test_scaled)                          # Get probabilities
y_pred_classes = (y_pred_probs > 0.5).astype("int32").flatten()     # Convert to 0 or 1
y_test = y_test.values.flatten()                                    # Ensure shape match

# 8. Evaluate
accuracy = accuracy_score(y_test, y_pred_classes)

# 9. Print Sample Results
print("y_test shape:", y_test.shape)
print("First 5 y_test values:", y_test[:5])
print("y_pred_classes shape:", y_pred_classes.shape)
print("First 5 predicted values:", y_pred_classes[:5])
print(f"Accuracy: {accuracy * 100:.2f}%")

# 10. Save Model and Scaler
model.save("fake_profile_keras_model.h5")
joblib.dump(scaler, "scaler.pkl")
