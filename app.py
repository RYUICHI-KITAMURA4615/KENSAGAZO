from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(filepath)

    # 陰性・陽性の判定処理（ここをAIモデルに置き換え可能）
    result = judge_image(filepath)

    return jsonify({'result': result})

def judge_image(image_path):
    # ※仮の判定（明るさによって判定）
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return '陽性' if mean_intensity > 100 else '陰性'

if __name__ == '__main__':
    app.run(debug=True)
