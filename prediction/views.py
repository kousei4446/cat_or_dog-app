from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import json
import os
from io import BytesIO
import base64
import gdown

MODEL_PATH = "model.h5"
CLASS_PATH = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'class_names.json')

URL = "https://drive.google.com/uc?id=13_SbB1iLc227VY77iEvQgRdA_HDTfAr_"
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(URL, MODEL_PATH, quiet=False)
    else:
        print("Model already downloaded.")

# アプリケーションの起動時にモデルをダウンロード
download_model()

model = load_model(MODEL_PATH)
with open(CLASS_PATH, 'r') as f:
    class_names = json.load(f)

def predict(request):
    form = ImageUploadForm(request.POST or None, request.FILES or None)
    context = {'form': form}

    if request.method == 'POST' and form.is_valid():
        # アップロード画像をメモリ上で読み込む
        img_file = form.cleaned_data['image']
        
        # InMemoryUploadedFile を BytesIO に変換
        img_bytes = BytesIO(img_file.read())

        # 画像をロード（BytesIO を使う）
        img = load_img(img_bytes, target_size=(32, 32))  # CIFAR-10 は 32x32
        x = img_to_array(img) 
        x = np.expand_dims(x, axis=0)  # (1,32,32,3)

        # 推論
        logits = model.predict(x)[0]
        preds = tf.nn.softmax(logits).numpy()  # 確率に変換
        top_idx = np.argmax(preds)
        prediction = class_names[top_idx]
        confidence = preds[top_idx]

        # base64 エンコードしてテンプレートに渡す（プレビュー用）
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        img_data = f"data:image/png;base64,{img_str}"

        context.update({
            'prediction': prediction,
            'confidence': f"{confidence*100:.1f}",
            'img_data': img_data,
        })

    return render(request, 'home.html', context)
