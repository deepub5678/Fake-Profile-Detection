from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load trained XGBoost model
model = joblib.load("fake_profile_xgb.pkl")  # Load new model

# Define feature names (Replace with actual feature names from your dataset)
feature_columns = ['Number of Friends', 'Account Type', 'Number of Posts', 'Number of Likes', 'Number of Comments', 'Sentiment Score', 'Account Age (Days)', 'Days Since Last Active', 'Has Profile Picture', 'word_0', 'word_1', 'word_2', 'word_3', 'word_4', 'word_5', 'word_6', 'word_7', 'word_8', 'word_9', 'word_10', 'word_11', 'word_12', 'word_13', 'word_14', 'word_15', 'word_16', 'word_17', 'word_18', 'word_19', 'word_20', 'word_21', 'word_22', 'word_23', 'word_24', 'word_25', 'word_26', 'word_27', 'word_28', 'word_29', 'word_30', 'word_31', 'word_32', 'word_33', 'word_34', 'word_35', 'word_36', 'word_37', 'word_38', 'word_39', 'word_40', 'word_41', 'word_42', 'word_43', 'word_44', 'word_45', 'word_46', 'word_47', 'word_48', 'word_49', 'word_50', 'word_51', 'word_52', 'word_53', 'word_54', 'word_55', 'word_56', 'word_57', 'word_58', 'word_59', 'word_60', 'word_61', 'word_62', 'word_63', 'word_64', 'word_65', 'word_66', 'word_67', 'word_68', 'word_69', 'word_70', 'word_71', 'word_72', 'word_73', 'word_74', 'word_75', 'word_76', 'word_77', 'word_78', 'word_79', 'word_80', 'word_81', 'word_82', 'word_83', 'word_84', 'word_85', 'word_86', 'word_87', 'word_88', 'word_89', 'word_90', 'word_91', 'word_92', 'word_93', 'word_94', 'word_95', 'word_96', 'word_97', 'word_98', 'word_99', 'Likes per Post', 'Comments per Post', 'Friends per Post']

# API Key for authentication (Set your own key)
API_KEY = "your_secret_api_key"

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check API key
        api_key = request.headers.get("x-api-key")
        if api_key != API_KEY:
            return jsonify({"error": "Unauthorized access"}), 401

        data = request.get_json()
        
        # Convert input JSON to DataFrame
        input_data = pd.DataFrame([data])
-
        # Ensure feature names match those used during training
        input_data = input_data[feature_columns]  # Select only relevant features

        # Make prediction
        prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of fake profile
        prediction = int(prediction_prob >= 0.5)  # Threshold-based classification

        return jsonify({
            "is_fake_profile": prediction,
            "confidence": round(prediction_prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
