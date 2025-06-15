from flask import Flask, render_template, request, jsonify
import os, cv2, numpy as np, pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
CSV_FILE = 'result.csv'
MODEL_PATH = 'model/model.h5'
IMAGE_SIZE = (224, 224)
class_names = ['陰性', '陽性', '不明']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filename = datetime.now().strftime('%Y%m%d_%H%M%S.jpg')
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))[0]
    label = class_names[np.argmax(pred)]
    confidence = float(np.max(pred))

    result = {'ファイル名': filename, '判定結果': label, '確信度': round(confidence * 100, 2)}
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = df.append(result, ignore_index=True)
    else:
        df = pd.DataFrame([result])
    df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
