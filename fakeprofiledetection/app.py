from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
import pickle
import pandas as pd
import sqlite3
from datetime import datetime
import logging
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure key in production

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for login
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    features TEXT,
                    is_fake INTEGER,
                    detection_date TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_users INTEGER,
                    fake_profiles INTEGER,
                    active_users INTEGER,
                    success_rate REAL,
                    update_date TIMESTAMP)''')
    conn.commit()
    conn.close()

# Load Pickle model
try:
    with open("fake_profile_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    logger.info("Pickle model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Pickle model: {e}")
    model = None

# Load Keras model and tokenizer
try:
    keras_model = load_model('model/fake_profile_keras_model.h5')
    keras_tokenizer = joblib.load('model/tokenizer.pkl')
    logger.info("Keras model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Keras model or tokenizer: {e}")
    keras_model = None
    keras_tokenizer = None

# ----------------------- API ROUTES ----------------------- #

@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT * FROM stats ORDER BY update_date DESC LIMIT 1")
    stats = c.fetchone()
    c.execute("""
        SELECT strftime('%Y-%m', detection_date) AS month,
               COUNT(*) AS count
        FROM detections
        WHERE is_fake = 1
        GROUP BY month
        ORDER BY month
        LIMIT 12
    """)
    trends = c.fetchall()
    conn.close()
    
    if stats:
        return jsonify({
            'total_users': stats[1],
            'fake_profiles': stats[2],
            'active_users': stats[3],
            'success_rate': stats[4],
            'trend_labels': [t[0] for t in trends],
            'trend_data': [t[1] for t in trends]
        })
    return jsonify({'error': 'No stats available'}), 404

@app.route('/api/recent-detections', methods=['GET'])
@login_required
def recent_detections():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY detection_date DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    
    return jsonify([
        {
            'id': row[0],
            'username': row[1],
            'is_fake': bool(row[3]),
            'detection_date': row[4]
        } for row in rows
    ])

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        logger.info(f"Prediction request: {data}")
        if not all(k in data for k in ['username', 'features']):
            return jsonify({"error": "Missing required fields"}), 400

        features_df = pd.DataFrame([data['features']])
        prediction = int(model.predict(features_df)[0])
        confidence = float(np.max(model.predict_proba(features_df)[0]))

        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        c.execute("""
            INSERT INTO detections (username, features, is_fake, detection_date)
            VALUES (?, ?, ?, ?)
        """, (data['username'], str(data['features']), prediction, datetime.now()))
        conn.commit()
        conn.close()

        return jsonify({
            "prediction": "Fake Profile" if prediction else "Real Profile",
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------------- AUTH ROUTES ----------------------- #

@app.route('/login', methods=['POST'])
def login():
    user = User(1)
    login_user(user)
    return jsonify({"success": True})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ----------------------- FRONTEND ROUTES ----------------------- #

@app.route('/', methods=['GET'])
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    if model is None:
        return "Model not loaded"

    try:
        username = request.form['username']
        features = {
            "Number of Friends": int(request.form["Number_of_Friends"]),
            "Account Type": int(request.form["Account_Type"]),
            "Number of Posts": int(request.form["Number_of_Posts"]),
            "Number of Likes": int(request.form["Number_of_Likes"]),
            "Number of Comments": int(request.form["Number_of_Comments"]),
            "Sentiment Score": int(request.form["Sentiment_Score"]),
            "Account Age (Days)": int(request.form["Account_Age_Days"]),
            "Days Since Last Active": int(request.form["Days_Since_Last_Active"]),
            "Has Profile Picture": int(request.form["Has_Profile_Picture"]),
            "Likes per Post": int(request.form["Number_of_Likes"]) / (int(request.form["Number_of_Posts"]) + 1),
            "Comments per Post": int(request.form["Number_of_Comments"]) / (int(request.form["Number_of_Posts"]) + 1),
            "Friends per Post": int(request.form["Number_of_Friends"]) / (int(request.form["Number_of_Posts"]) + 1),
            "Post Frequency": int(request.form["Number_of_Posts"]) / (int(request.form["Account_Age_Days"]) + 1),
            "Like Ratio": int(request.form["Number_of_Likes"]) / (int(request.form["Number_of_Friends"]) + 1),
            "Bio Length": 30  # You can modify this if your form supports bio input
        }

        df = pd.DataFrame([features])
        prediction = int(model.predict(df)[0])
        confidence = float(np.max(model.predict_proba(df)[0]))

        return render_template("result.html", username=username,
                               prediction="Fake Profile" if prediction else "Real Profile",
                               confidence=round(confidence * 100, 2))
    except Exception as e:
        logger.error(f"Form submission error: {e}")
        return f"Error: {str(e)}", 500

# ----------------------- APP START ----------------------- #

if __name__ == '__main__':
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
