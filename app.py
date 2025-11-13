from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from datetime import datetime
import random
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# CSV file to store results
RESULTS_FILE = 'results.csv'
if not os.path.exists(RESULTS_FILE):
    pd.DataFrame(columns=['filename', 'prediction', 'timestamp']).to_csv(RESULTS_FILE, index=False)

# ----------------------------
# FAKE TRAINING PROGRESS LOGS
# ----------------------------
training_logs = []
is_training = False


def fake_training():
    """Simulates a model training process."""
    global is_training, training_logs
    is_training = True
    training_logs.clear()

    logs = [
        "Initializing model...",
        "Loading dataset...",
        "Epoch 1/10 - loss: 0.68 - accuracy: 0.75",
        "Epoch 2/10 - loss: 0.45 - accuracy: 0.82",
        "Epoch 3/10 - loss: 0.38 - accuracy: 0.87",
        "Epoch 4/10 - loss: 0.32 - accuracy: 0.89",
        "Epoch 5/10 - loss: 0.29 - accuracy: 0.91",
        "Model validation in progress...",
        "Saving trained model...",
        "✅ Model trained and saved successfully!"
    ]

    for log in logs:
        training_logs.append(log)
        time.sleep(1.2)

    is_training = False


# ----------------------------
# ROUTES
# ----------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Simulated prediction
        prediction = random.choice(['Normal', 'Pneumonia', 'COVID-19'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save result
        df = pd.read_csv(RESULTS_FILE)
        new_row = pd.DataFrame([[filename, prediction, timestamp]], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(RESULTS_FILE, index=False)

        return render_template('result.html', filename=filename, prediction=prediction)

    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    df = pd.read_csv(RESULTS_FILE)
    return render_template('dashboard.html', tables=df.to_dict(orient='records'))


@app.route('/training')
def training():
    global is_training
    if not is_training:
        threading.Thread(target=fake_training, daemon=True).start()
    return render_template('training.html', logs=training_logs, is_training=is_training)


# ----------------------------
# RUN APP SAFELY
# ----------------------------
if __name__ == '__main__':
    # disable debug auto-reloader (avoids socket conflict)
    app.run(host='127.0.0.1', port=8085, debug=True)