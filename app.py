from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load tokenizer dan model
tokenizer = joblib.load("tokenizer.joblib")
model = load_model("model_sentimen_lstm.h5")

# Label mapping
label_map = {0: "negatif", 1: "netral", 2: "positif"}

# Fungsi bersihkan teks
def bersihkan_teks(teks):
    teks = teks.lower()
    teks = re.sub(r"http\S+|www\S+|https\S+", "", teks)
    teks = re.sub(r"[^a-zA-Z0-9\s]", "", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks

@app.route("/")
def index():
    return render_template("index.html", hasil=None, komentar=None)

@app.route("/klasifikasi", methods=["POST"])
def klasifikasi():
    komentar = request.form.get("komentar")

    if not komentar:
        return render_template("index.html", hasil="Komentar kosong!", komentar=komentar)

    komentar_bersih = bersihkan_teks(komentar)
    sequence = tokenizer.texts_to_sequences([komentar_bersih])
    padded = pad_sequences(sequence, maxlen=50, padding="post", truncating="post")

    prediction = model.predict(padded)
    label = np.argmax(prediction)
    hasil = label_map[label]

    return render_template("index.html", hasil=hasil, komentar=komentar)

if __name__ == "__main__":
    app.run(debug=True)
