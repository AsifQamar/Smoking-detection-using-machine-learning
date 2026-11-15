import os
import base64
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'smoke_detection_model.keras'))
SMOKING_THRESHOLD = 0.70

def predict_frame(frame):
    """Predict whether a given OpenCV frame shows smoking or not."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (64, 64)) / 255.0
    prob = float(model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0])
    is_smoking = prob >= SMOKING_THRESHOLD
    conf = round((prob if is_smoking else 1 - prob) * 100, 1)
    return ("Smoking", conf) if is_smoking else ("Not Smoking", conf)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    label, conf = predict_frame(img)
    return jsonify({'label': label, 'confidence': conf})

@app.route('/predict_video', methods=['POST'])
def predict_video():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No video uploaded'}), 400
    path = os.path.join(BASE_DIR, 'temp_video.mp4')
    file.save(path)
    cap = cv2.VideoCapture(path)
    smoke_count = 0
    total = 0
    while cap.isOpened() and total < 30:
        ret, frame = cap.read()
        if not ret:
            break
        lbl, _ = predict_frame(frame)
        if lbl == "Smoking":
            smoke_count += 1
        total += 1
    cap.release()
    if os.path.exists(path):
        os.remove(path)
    is_smoke = smoke_count > (total * 0.25)
    return jsonify({
        'label': 'Smoking' if is_smoke else 'Not Smoking',
        'confidence': round((smoke_count / total * 100) if total else 0, 1)
    })

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    data = request.json.get('image', '')
    if not data:
        return jsonify({'error': 'No image data'}), 400
    encoded = data.split(',')[1] if ',' in data else data
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    label, conf = predict_frame(frame)
    return jsonify({'label': label, 'confidence': conf})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

