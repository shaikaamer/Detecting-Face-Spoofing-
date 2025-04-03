from flask import Flask, render_template, request, url_for
import numpy as np
import tensorflow as tf
import cv2
import os
import csv

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__, static_folder="uploads")

# -----------------------------
# Importing the Models
# -----------------------------
AUTOENCODER_PATH = "autoencoder.keras"
CLASSIFIER_PATH = "anomaly_classifier.keras"

# Check the model Existing
if not os.path.exists(AUTOENCODER_PATH) or not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError("Required model file(s) missing!")

# Loading the trained models
autoencoder = tf.keras.models.load_model(AUTOENCODER_PATH, compile=False)
classifier = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)

# -----------------------------
# File Upload Configuration
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Logging Function
# -----------------------------
# Logs each prediction to a CSV file 
def log_prediction(image_name, mse, score, decision):
    with open("predictions_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([image_name, round(mse, 5), round(score, 4), decision])

# -----------------------------
# Core Inference Logic
# -----------------------------
# Processes uploaded image through both models and makes decision
def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150)) / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)          # Shape: (1, 150, 150, 3)

    # Autoencoder reconstruction and MSE
    reconstructed = autoencoder.predict(img)
    mse_error = np.mean(np.square(img - reconstructed))

    # Classifier prediction (probability of spoof)
    prediction = classifier.predict(img)[0][0]

    # -----------------------------
    # Decision Logic
    # -----------------------------
    if mse_error > 0.01:
        if prediction > 0.6:
            anomaly_result = "Anomalous-Fake/Defective"
            result_dir = "results/negative"
        else:
            anomaly_result = "Normal"
            result_dir = "results/positive"
    else:
        if prediction > 0.4:
            anomaly_result = "Anomalous-Fake/Defective"
            result_dir = "results/negative"
        else:
            anomaly_result = "Normal"
            result_dir = "results/positive"

    # Log the result for later analysis
    os.makedirs(result_dir, exist_ok=True)  # Ensure the directory exists

    # Save a copy of the image to the result folder
    result_image_path = os.path.join(result_dir, os.path.basename(image_path))
    cv2.imwrite(result_image_path, cv2.imread(image_path))

    log_prediction(os.path.basename(image_path), mse_error, prediction, anomaly_result)
    return anomaly_result, mse_error, prediction

# -----------------------------
# Flask Web Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Save uploaded file to disk
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Process image and get prediction results
            result, mse_error, prediction = process_image(filepath)
            rule = f"MSE = {mse_error:.5f}, Classifier = {prediction:.4f}"

            # Render results to user
            return render_template("index.html",
                                   result=result,
                                   mse_error=round(mse_error, 5),
                                   prediction=round(prediction, 4),
                                   rule=rule,
                                   image=file.filename)
    return render_template("index.html")

# -----------------------------
# App Starting Point
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
