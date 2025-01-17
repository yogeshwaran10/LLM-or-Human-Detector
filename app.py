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

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        text = request.form.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Please enter some text to classify'}), 400

        print(f"Raw input text: {text}")  # Debugging: Print the raw input text
        # Generate embedding using your optimized script
        embedding = get_embedding(text)
        print(f"Generated embedding: {embedding}")  # Debugging the embedding

        # Convert to torch tensor
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0)
        print(f"Embedding tensor shape: {embedding_tensor.shape}")  # Debugging

        # Ensure the model is loaded correctly
        if model is None:
            return jsonify({'error': 'Model failed to load'}), 500

        # Get prediction
        with torch.no_grad():
            outputs = model(embedding_tensor)
            print(f"Model output (logits): {outputs}")  # Debugging model output

        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()  # Shape: [2]
        print(f"Class probabilities: {probabilities}")  # Debugging probabilities

        # Get the predicted class (index of highest probability)
        predicted_class = torch.argmax(probabilities).item()  # This gives you 0 or 1
        print(f"Predicted class: {predicted_class}")

        # Prepare the result with class label and confidence
        result = {
            'prediction': 'LLM-Generated' if predicted_class == 1 else 'Human-Written',
            'confidence': f"{probabilities[predicted_class] * 100:.2f}%"  # Confidence in percentage
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()  # Print full stack trace for debugging
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    # Load the pre-trained model
    try:
        # Initialize the model with the correct input dimension
        model = CombinedModel(input_dim=797)  # Replace 797 with the actual input dimension from your training

        # Check if the model file exists
        if not os.path.exists('best_model_combination.pth'):
            print("Model file not found.")
            model = None
        else:
            # Load the saved state_dict into the model instance
            if torch.cuda.is_available():
                device = torch.device("cuda") 
                print(device)
            else:
                device = torch.device("cpu")
                print(device)

            model.load_state_dict(torch.load('best_model_combination.pth', map_location=device))
            model.eval()  # Set the model to evaluation mode
            print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    # Run the Flask app
    port = int(os.getenv("PORT", 5000))  # Render will provide the port
    app.run(debug=True, host='0.0.0.0', port=port)
