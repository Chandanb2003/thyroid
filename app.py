from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import io
import math
import uuid
from datetime import datetime, timedelta
import requests
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class_names = ['Benign', 'Malignant', 'Normal']

class ThyroidNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base = models.resnet18(pretrained=False)
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base(x)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ThyroidNet()
try:
    model.load_state_dict(torch.load("thyroid_model.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

chat_sessions = {}
SESSION_EXPIRY = 60



def safe_float(value):
    if value is None:
        return 0.0
    if math.isnan(value):
        return 0.0
    if math.isinf(value) and value > 0:
        return 1.0e100
    if math.isinf(value) and value < 0:
        return -1.0e100
    return float(value)

def get_groq_response(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1024
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error communicating with Groq API: {str(e)}")
        return f"Unable to generate response. Please try again later. Error: {str(e)}"

def get_groq_explanation(prediction_data):
    prediction = prediction_data["prediction"]
    confidence = prediction_data["confidence_scores"][prediction]
    confidence_text = ", ".join([f"{k}: {v:.2f}%" for k, v in prediction_data["confidence_scores"].items()])
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful medical AI assistant explaining the results of a thyroid CT image classification.
The image was classified as "{prediction}" with {confidence:.2f}% confidence.
The confidence scores for all classes are: {confidence_text}.

Provide a clear, user-friendly explanation about these findings. Avoid technical jargon when possible or explain it when used.
Remember that this is an AI classification and should be reviewed by healthcare professionals.
Always emphasize that these results are not a diagnosis and clinical correlation is essential.
Keep your answer concise (maximum 4 paragraphs), accurate, and empathetic. People may be concerned about these results.

For context:
- Benign: Indicates a non-cancerous growth in the thyroid
- Malignant: Indicates potential thyroid cancer
- Normal: Indicates no significant abnormality detected in the thyroid
"""
        },
        {
            "role": "user",
            "content": "Please explain these thyroid CT scan findings in simple terms and what they might mean."
        }
    ]
    return get_groq_response(messages)

def get_session(session_id):
    if session_id not in chat_sessions:
        return None
    session = chat_sessions[session_id]
    session['last_accessed'] = datetime.now()
    return session

def cleanup_expired_sessions():
    current_time = datetime.now()
    expired_sessions = []
    for session_id, session in chat_sessions.items():
        if current_time - session['last_accessed'] > timedelta(minutes=SESSION_EXPIRY):
            expired_sessions.append(session_id)
    for session_id in expired_sessions:
        del chat_sessions[session_id]
    if expired_sessions:
        print(f"Cleaned up {len(expired_sessions)} expired sessions")

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Thyroid CT Image Classifier API with Chatbot is running. Use /predict endpoint to classify images."})

@app.route("/health", methods=["GET"])
def health_check():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"detail": "Model not loaded"}), 503
    if 'file' not in request.files:
        return jsonify({"detail": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"detail": "Invalid image format. Please upload a PNG or JPG image."}), 400
    try:
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence_scores = {class_names[i]: safe_float(float(probabilities[i]) * 100) for i in range(len(class_names))}
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]
        prediction_data = {
            "prediction": prediction,
            "confidence_scores": confidence_scores
        }
        explanation = get_groq_explanation(prediction_data)
        session_id = str(uuid.uuid4())
        chat_history = [
            {
                "role": "system",
                "content": f"""You are a helpful medical AI assistant specialized in thyroid CT scan interpretation.
An image was analyzed and classified as "{prediction}" with {confidence_scores[prediction]:.2f}% confidence.
The confidence scores for all classes were:
{', '.join([f"{k}: {v:.2f}%" for k, v in confidence_scores.items()])}

You should help answer questions about this classification in a clear, accurate, and empathetic way.
Always remember to emphasize that these are AI-based interpretations and a healthcare professional should be consulted for proper diagnosis.

For context:
- Benign: Indicates a non-cancerous growth in the thyroid
- Malignant: Indicates potential thyroid cancer
- Normal: Indicates no significant abnormality detected in the thyroid
"""
            },
            {
                "role": "assistant",
                "content": explanation
            }
        ]
        chat_sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "image_info": prediction_data,
            "chat_history": chat_history
        }
        cleanup_expired_sessions()
        return jsonify({
            "session_id": session_id,
            "prediction": prediction,
            "confidence_scores": confidence_scores,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"detail": f"Prediction error: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    session_id = data.get("session_id")
    message = data.get("message")
    session = get_session(session_id)
    if session is None:
        return jsonify({"detail": "Session not found"}), 404
    try:
        session["chat_history"].append({"role": "user", "content": message})
        groq_messages = [{"role": msg["role"], "content": msg["content"]} for msg in session["chat_history"]]
        response_text = get_groq_response(groq_messages)
        session["chat_history"].append({"role": "assistant", "content": response_text})
        if len(session["chat_history"]) > 22:
            session["chat_history"] = [session["chat_history"][0]] + session["chat_history"][-21:]
        return jsonify({
            "session_id": session_id,
            "response": response_text
        })
    except Exception as e:
        return jsonify({"detail": f"Chat error: {str(e)}"}), 500

@app.route("/sessions/<session_id>", methods=["GET"])
def get_session_info(session_id):
    session = get_session(session_id)
    if session is None:
        return jsonify({"detail": "Session not found"}), 404
    try:
        return jsonify({
            "session_id": session["session_id"],
            "created_at": session["created_at"].isoformat(),
            "last_accessed": session["last_accessed"].isoformat(),
            "image_classification": session["image_info"]["prediction"],
            "confidence": session["image_info"]["confidence_scores"][session["image_info"]["prediction"]],
            "message_count": len(session["chat_history"]) - 1
        })
    except Exception as e:
        return jsonify({"detail": f"Error retrieving session: {str(e)}"}), 500

# @app.route("/transcribe", methods=["POST"])
# def transcribe():
#     if 'file' not in request.files:
#         return jsonify({"detail": "No audio file uploaded"}), 400
#     file = request.files['file']
#     audio_data = file.read()
#     transcription = transcribe_audio(audio_data)
#     return jsonify({"transcription": transcription})

# @app.before_first_request
# def before_first_request_func():
#     cleanup_expired_sessions()

# @app.teardown_appcontext
# def shutdown_session(exception=None):
#     chat_sessions.clear()
#     print("Cleared all sessions on shutdown")

if __name__ == "__main__":
    cleanup_expired_sessions()
    app.run(host="0.0.0.0", port=5000, debug=True)
