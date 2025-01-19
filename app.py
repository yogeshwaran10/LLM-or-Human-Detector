import torch
import numpy as np
import os
import logging
from flask import Flask, render_template, request, jsonify
from embedding_generation import get_embedding  # Assuming this is your embedding generation script
import traceback
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define the CombinedModel class (as it was defined in your training script)
class CombinedModel(nn.Module):
    def __init__(self, input_dim):
        super(CombinedModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # Output layer for binary classification
        )

    def forward(self, combined_input):
        logits = self.network(combined_input)
        return logits

# Set default data type to float (recommended by PyTorch)
torch.set_default_dtype(torch.float)

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model globally, so it happens when the app starts
model = None

# Debugging: Check model loading
def load_model():
    global model
    try:
        model = CombinedModel(input_dim=797)  # Replace 797 with the actual input dimension from your training

        if not os.path.exists('best_model_combination.pth'):
            print("Model file 'best_model_combination.pth' not found in the current directory.")
            model = None
        else:
            print("Model file found. Attempting to load...")
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using device: {device}")
            else:
                device = torch.device("cpu")
                print(f"Using device: {device}")

            try:
                model.load_state_dict(torch.load('best_model_combination.pth', map_location=device))
                model.eval()  # Set the model to evaluation mode
                print("Model loaded successfully.")
            except Exception as load_error:
                print(f"Error while loading model: {load_error}")
                traceback.print_exc()
                model = None
    except Exception as e:
        print(f"Error during model initialization: {e}")
        traceback.print_exc()
        model = None

# Load the model as soon as the app starts
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    print("Entered classify_text function.")  # Debugging entry point

    try:
        text = request.form.get('text', '').strip()
        print(f"Input received: {text}")  # Debug input

        if not text:
            print("No text input provided.")
            return jsonify({'error': 'Please enter some text to classify'}), 400

        # Debug: Print before embedding generation
        print("Generating embedding for input text...")
        embedding = get_embedding(text)
        print(f"Generated embedding: {embedding}")

        # Debug: Print before converting to tensor
        print("Converting embedding to tensor...")
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0)
        print(f"Tensor shape: {embedding_tensor.shape}")

        # Debug: Ensure model is loaded
        if model is None:
            print("Model not loaded.")
            return jsonify({'error': 'Model failed to load'}), 500

        # Debug: Predicting with the model
        print("Running model prediction...")
        with torch.no_grad():
            outputs = model(embedding_tensor)
            print(f"Model outputs (logits): {outputs}")

        # Debug: Calculate probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()
        print(f"Class probabilities: {probabilities}")

        # Debug: Predicted class
        predicted_class = torch.argmax(probabilities).item()
        print(f"Predicted class: {predicted_class}")

        result = {
            'prediction': 'LLM-Generated' if predicted_class == 1 else 'Human-Written',
            'confidence': f"{probabilities[predicted_class] * 100:.2f}%"
        }
        print(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"Error in classify_text function: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    # The `main` function is not required for gunicorn deployment, so it's omitted.
    # Instead, the app is ready when gunicorn starts it.
    # app.run(host='0.0.0.0', port=8080, debug=True)  # This is only for local testing, gunicorn will handle it in prod
    app.run(debug=True)