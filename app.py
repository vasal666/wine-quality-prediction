from flask import Flask, render_template, request, session, make_response, redirect, url_for
from datetime import datetime
from xhtml2pdf import pisa
from io import BytesIO
import joblib
import numpy as np
import json
import os
import sqlite3
import bcrypt
from functools import wraps

# Initialize app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = 3600  # 1 hour session lifetime

# Database setup
def init_db():
    with sqlite3.connect('users.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

# Initialize database
init_db()

# Constants
MODEL_FILE = 'wine_quality_model.pkl'
SCALER_FILE = 'scaler.pkl'
FEATURES_FILE = 'features.json'
MAX_HISTORY = 10

# Load model and scaler
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Load features
with open(FEATURES_FILE) as f:
    feature_data = json.load(f)
    ORIGINAL_FEATURES = feature_data["original_features"]
    ENGINEERED_FEATURES = feature_data["engineered_features"]
    EXPECTED_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES

# Feature Engineering
def compute_engineered_features(data):
    try:
        acid_balance = float(data['fixed acidity']) - float(data['volatile acidity'])
        sulfur_ratio = float(data['free sulfur dioxide']) / (float(data['total sulfur dioxide']) + 1e-5)
        alcohol_to_acid = float(data['alcohol']) / (float(data['fixed acidity']) + 1e-5)
        wine_type = 0
    except Exception as e:
        raise ValueError(f"Error computing engineered features: {e}")

    return {
        'acid_balance': acid_balance,
        'sulfur_ratio': sulfur_ratio,
        'alcohol_to_acid': alcohol_to_acid,
        'wine_type': wine_type
    }

# Input Validation
def validate_inputs(data):
    missing = set(ORIGINAL_FEATURES) - set(data.keys())
    if missing:
        raise ValueError(f"Missing input fields: {', '.join(missing)}")

    for feature in ORIGINAL_FEATURES:
        try:
            value = float(data[feature])
            if value < 0:
                raise ValueError(f"Negative value not allowed for {feature}")
        except ValueError:
            raise ValueError(f"Invalid value for {feature}: must be a number")

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Intro page
@app.route('/')
def intro():
    return render_template('intro.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, password FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            
            if user and bcrypt.checkpw(password, user[1].encode('utf-8')):
                session['user_id'] = user[0]
                session.permanent = True
                session['history'] = session.get('history', [])  # Initialize history if not exists
                app.logger.info(f"User {username} logged in, history: {session['history']}")
                return redirect(url_for('home'))
            else:
                return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

# Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        confirm_password = request.form['confirm_password'].encode('utf-8')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
        
        try:
            with sqlite3.connect('users.db') as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                             (username, hashed_password.decode('utf-8')))
                conn.commit()
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists')
    
    return render_template('register.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('history', None)  # Clear history on logout
    return redirect(url_for('intro'))

# Main prediction page
@app.route('/predictor')
@login_required
def home():
    app.logger.info(f"Rendering predictor page, history: {session.get('history', [])}")
    return render_template('index.html',
                           features=EXPECTED_FEATURES,
                           history=session.get('history', []))

# Prediction logic
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        form_data = request.form.to_dict()
        validate_inputs(form_data)

        engineered = compute_engineered_features(form_data)
        full_input = {**form_data, **engineered}
        feature_vector = np.array([[float(full_input[feat]) for feat in EXPECTED_FEATURES]])

        X_scaled = scaler.transform(feature_vector)
        prediction = int(model.predict(X_scaled)[0])

        result = {
            'score': prediction,
            'quality': f"{prediction}/10",
            'message': "Excellent!" if prediction >= 7 else "Good" if prediction >= 5 else "Average"
        }

        # Store prediction in session history
        history = session.get('history', [])
        history.append({
            'input': form_data,
            'prediction': result,
            'timestamp': datetime.now().strftime("Predicted on %B %d, %Y at %I:%M %p")
        })
        session['history'] = history[-MAX_HISTORY:]
        app.logger.info(f"Prediction stored, new history: {session['history']}")

        return render_template('index.html',
                               prediction=result,
                               original_values=form_data,
                               features=ORIGINAL_FEATURES,
                               history=session['history'])

    except ValueError as e:
        app.logger.error(f"Validation error: {str(e)}")
        return render_template('index.html',
                               error=str(e),
                               original_values=request.form.to_dict(),
                               features=ORIGINAL_FEATURES,
                               history=session.get('history', []))
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html',
                               error="An unexpected error occurred.",
                               features=ORIGINAL_FEATURES,
                               history=session.get('history', []))

# PDF download route
@app.route('/download_pdf/<int:history_index>')
@login_required
def download_pdf(history_index):
    history = session.get('history', [])
    app.logger.info(f"Downloading PDF for history index {history_index}, history: {history}")
    if history_index >= len(history):
        return "Invalid history index", 404

    record = history[history_index]
    pdf_html = render_template('pdf_template.html', record=record, current_year=datetime.now().year)

    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(pdf_html, dest=pdf_buffer)

    if pisa_status.err:
        app.logger.error("Error generating PDF")
        return "Error generating PDF", 500

    pdf_buffer.seek(0)
    response = make_response(pdf_buffer.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=wine_prediction_{history_index}.pdf'
    return response

# Run app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)