from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define the absolute path for the models folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_folder = os.path.join(project_root, "models")

# Load the model
model_path = os.path.join(models_folder, "crop_health_model.pkl")
try:
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Calculate soil_fertility_index
    data["soil_fertility_index"] = (
        data["nitrogen"] + data["phosphorus"] + data["potassium"]
    )

    # Remove the individual soil components as they weren't used in training
    del data["nitrogen"]
    del data["phosphorus"]
    del data["potassium"]

    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    app.run(debug=True)

