from flask import Flask, render_template, request, jsonify
import xgboost as xgb
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load TF-IDF vectorizer
with open("models/tfidf_vectorizer_fixed.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load XGBoost models
models = []
for i in range(6):
    model = xgb.Booster()
    model.load_model(f"models/xgb_booster_label_{i}.json")
    models.append(model)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sequence = request.form["sequence"]
    features = vectorizer.transform([sequence])
    dmatrix = xgb.DMatrix(features)

    predictions = []
    for model in models:
        pred = model.predict(dmatrix)
        predictions.append(int(pred[0] >= 0.5))

    return jsonify({"input": sequence, "predictions": predictions})

if __name__ == "__main__":
    app.run(debug=True)
