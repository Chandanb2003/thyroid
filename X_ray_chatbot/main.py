from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import pipeline, BeitFeatureExtractor, SwinModel, BeitModel, WhisperProcessor, WhisperForConditionalGeneration
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import io
import base64
import requests
from io import BytesIO

import os
from typing import List, Dict, Any, Optional

import soundfile as sf
import uuid
from datetime import datetime


app = Flask(__name__)
CORS(app)


# Data classes for chat history (not using Pydantic)
class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

# Session storage for analysis results and chat history
class AnalysisSession:
    def __init__(self, image_b64: str, predictions: list, explanation: str):
        self.image_b64 = image_b64
        self.predictions = predictions
        self.explanation = explanation
        self.chat_history = [
            ChatMessage(role="system", content=f"You are a medical imaging expert specializing in X-ray analysis. The user has uploaded an X-ray image. Here is the analysis: {explanation}")
        ]
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()

# Global storage for sessions
analysis_sessions = {}


# Global variables for models
detection_models = None
custom_components = None
custom_model = None


# Whisper model and processor for audio transcription
global processor, stt_model, device
processor = None
stt_model = None
device = "cpu"

# Load AI models
def load_detection_models():
    return {
        "BoneEye": pipeline("object-detection", model="D3STRON/bone-fracture-detr"),
        "BoneGuardian": pipeline("image-classification", model="Heem2/bone-fracture-detection-using-xray"),
        "XRayMaster": pipeline("image-classification", 
            model="nandodeomkar/autotrain-fracture-detection-using-google-vit-base-patch-16-54382127388")
    }

# Load custom model components
def load_custom_model_components():
    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Feature Extractors
    beit_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224")
    swin_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    beit_model = BeitModel.from_pretrained("microsoft/beit-large-patch16-224")
    
    # MedViT (Simulating with ResNet since MedViT is not in HuggingFace yet)
    medvit_model = resnet50(pretrained=True)
    
    return {
        "transform": transform,
        "beit_extractor": beit_extractor,
        "swin_model": swin_model,
        "beit_model": beit_model,
        "medvit_model": medvit_model
    }

# Custom model class
class MultiModelFractureDetector(nn.Module):
    def __init__(self):
        super(MultiModelFractureDetector, self).__init__()
        
        # Extract feature dimensions
        self.swin_features = 768
        self.beit_features = 1024
        self.medvit_features = 2048  # ResNet50 last layer
        
        # Fusion Layer
        self.fc = nn.Linear(self.swin_features + self.beit_features + self.medvit_features, 512)
        self.classifier = nn.Linear(512, 2)  # Fracture or No Fracture

    def forward(self, swin_out, beit_out, medvit_out):
        # Ensure dimensions match
        if len(medvit_out.shape) > 2:
            # Flatten if necessary
            medvit_out = medvit_out.view(medvit_out.size(0), -1)
            
        # Make sure dimensions match expected feature sizes
        if medvit_out.shape[1] != self.medvit_features:
            # Use adaptive pooling if dimensions don't match
            medvit_out = medvit_out[:, :self.medvit_features]
        
        # Concatenate features
        fused_features = torch.cat((swin_out, beit_out, medvit_out), dim=1)
        fused_features = self.fc(fused_features)
        output = self.classifier(fused_features)
        return output

# Helper functions
def translate_label(label):
    translations = {
        "fracture": "Bone Fracture",
        "no fracture": "No Bone Fracture",
        "normal": "Normal",
        "abnormal": "Abnormal",
        "F1": "Bone Fracture",
        "NF": "No Bone Fracture"
    }
    return translations.get(label.lower(), label)

def create_heatmap_overlay(image, box, score):
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x1, y1 = box['xmin'], box['ymin']
    x2, y2 = box['xmax'], box['ymax']

    if score > 0.8:
        fill_color = (255, 0, 0, 100)  # Red for high confidence
        border_color = (255, 0, 0, 255)
    elif score > 0.6:
        fill_color = (255, 165, 0, 100)  # Orange for medium confidence
        border_color = (255, 165, 0, 255)
    else:
        fill_color = (255, 255, 0, 100)  # Yellow for low confidence
        border_color = (255, 255, 0, 255)

    draw.rectangle([x1, y1, x2, y2], fill=fill_color)
    draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)

    return overlay

def draw_boxes(image, predictions):
    result_image = image.copy().convert('RGBA')

    for pred in predictions:
        box = pred['box']
        score = pred['score']

        overlay = create_heatmap_overlay(image, box, score)
        result_image = Image.alpha_composite(result_image, overlay)

        draw = ImageDraw.Draw(result_image)
        temperature = 36.5 + (score * 2.5)
        label = f"{translate_label(pred['label'])} ({score:.1%} • {temperature:.1f}°C)"

        text_bbox = draw.textbbox((box['xmin'], box['ymin'] - 20), label)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 180))

        draw.text(
            (box['xmin'], box['ymin'] - 20),
            label,
            fill=(255, 255, 255, 255)
        )

    return result_image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def extract_features(image, components):
    """Process image with custom model components"""
    # Process Image for Swin & BEiT
    inputs = components["beit_extractor"](images=image, return_tensors="pt")
    swin_out = components["swin_model"](**inputs).last_hidden_state.mean(dim=1)  # Swin features
    beit_out = components["beit_model"](**inputs).last_hidden_state.mean(dim=1)  # BEiT features
    
    # Process Image for MedViT (ResNet50)
    image_tensor = components["transform"](image).unsqueeze(0)
    features_extractor = torch.nn.Sequential(*list(components["medvit_model"].children())[:-1])
    medvit_out = features_extractor(image_tensor).squeeze(-1).squeeze(-1)
    
    # Make sure all tensors have the same batch size
    batch_size = swin_out.size(0)
    if medvit_out.size(0) != batch_size:
        medvit_out = medvit_out.expand(batch_size, -1)
    
    return swin_out, beit_out, medvit_out

def get_xray_explanation(predictions):
    """Get explanation from Groq LLM API"""
    # Get API key from environment
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    if not GROQ_API_KEY:
        return "API key for Groq not configured. Please set the GROQ_API_KEY environment variable."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a detailed prompt for the LLM
    detection_results = ""
    for i, pred in enumerate(predictions):
        detection_results += f"Detection {i+1}: {translate_label(pred.get('label', ''))} with confidence {pred.get('score', 0):.1%}\n"
    
    # Construct the prompt with just the detection results (no image)
    prompt = f"""
    You are a medical imaging expert specializing in X-ray analysis.
    I've analyzed an X-ray image with AI bone fracture detection.

    Please provide a simple and brief explanation based on these detection results:
    {detection_results}

    If no detections are present, explain what this means medically.

    In your explanation, please include:
    1. What the detection results suggest about potential fractures
    2. What body part might be shown based on the context
    3. What type of fracture might be present, if applicable
    4. Recommended medical follow-up actions
    
    Provide your analysis in a clear, structured format suitable for medical professionals.
    """
    
    # Prepare the request payload - use text-only format
    payload = {
        "model": "llama3-70b-8192",  # Using Llama 3, but you can use any Groq-supported model
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error from Groq API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Failed to get explanation: {str(e)}"

def generate_contextual_response(message: str, session_id: str = None):
    """Generate a context-aware response for the chat"""
    if not session_id or session_id not in analysis_sessions:
        # No specific X-ray context, provide a general response
        return "I don't have any X-ray analysis to reference. Please upload an X-ray image first, and I'll be able to discuss the findings in detail."
    
    # Get the session with X-ray analysis data
    session = analysis_sessions[session_id]
    session.last_accessed = datetime.now()  # Update last accessed time
    
    # Extract relevant data
    predictions = session.predictions
    explanation = session.explanation
    chat_history = session.chat_history
    
    # Simple message preprocessing
    message_lower = message.lower()
    
    # Check for follow-up indicators
    is_follow_up = False
    follow_up_indicators = ["also", "what about", "and", "but", "how about", "additionally", 
                          "more", "another", "again", "why", "still", "further", "too"]
    
    if any(indicator in message_lower for indicator in follow_up_indicators):
        is_follow_up = True
    
    # Check for short queries which are often follow-ups
    if len(message.split()) < 4 and len([m for m in chat_history if m.role == "user"]) > 0:
        is_follow_up = True
    
    # Check for pronoun references which suggest follow-ups
    if any(ref in message_lower for ref in ["it", "this", "that", "these", "those", "they", "them"]):
        is_follow_up = True
    
    # Create context for the LLM
    context_prompt = ""
    
    if is_follow_up:
        context_prompt = "Based on our previous conversation about the X-ray analysis, "
    
    # Get response from Groq LLM API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    if not GROQ_API_KEY:
        return "API key for Groq not configured. Please set the GROQ_API_KEY environment variable."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Format chat history for LLM
    formatted_history = []
    
    # First add system prompt with analysis details
    formatted_history.append({
        "role": "system",
        "content": f"""You are a medical imaging expert specializing in X-ray analysis.
The user has uploaded an X-ray image that has been analyzed for bone fractures.

Here are the detection results:
{', '.join([f"{p.get('label', 'Unknown')} ({p.get('score', 0)*100:.1f}%)" for p in predictions])}

AI explanation of findings:
{explanation}

Answer the user's questions about these findings. If the user asks questions not related to the X-ray analysis, politely guide them back to discussing the X-ray results.
Keep your answers concise, medically accurate, and focused on the X-ray findings.
Always remember previous parts of the conversation for context."""
    })
    
    # Add user/assistant conversation history excluding the system message
    for msg in chat_history[1:]:
        formatted_history.append({"role": msg.role, "content": msg.content})
    
    # Add current message
    formatted_history.append({"role": "user", "content": message})
    
    # Prepare the request payload
    payload = {
        "model": "llama3-70b-8192",
        "messages": formatted_history,
        "temperature": 0.3,
        "max_tokens": 800
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error from Groq API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Failed to generate response: {str(e)}"

def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio data using Whisper model."""
    try:
        # Read audio file using soundfile
        with io.BytesIO(audio_data) as audio_buffer:
            audio_array, sample_rate = sf.read(audio_buffer)
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            # Convert to float32
            audio_array = audio_array.astype(np.float32)
            # Process audio
            input_features = processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features.to(device)
            # Generate transcription
            predicted_ids = stt_model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return transcription[0]
    except Exception as e:
        return f"Error processing audio: {str(e)}"


# Model loading moved to main block for Windows compatibility

# Root endpoint to provide API info
@app.route("/", methods=["GET"])
def get_root():
    return jsonify({
        "name": "X-ray Fracture Analysis API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/analyze/", "method": "POST", "description": "Analyze X-ray image for fractures"},
            {"path": "/chat/", "method": "POST", "description": "Chat about X-ray analysis results"},
            {"path": "/models/", "method": "GET", "description": "Get info about available models"},
            {"path": "/health/", "method": "GET", "description": "Check API health status"}
        ],
        "documentation": "(see OpenAPI docs for FastAPI, not available in Flask)"
    })

# Analysis endpoint
@app.route("/analyze/", methods=["POST"])
def analyze_xray():
    file = request.files.get("file")
    model_choice = request.form.get("model_choice", "pretrained")
    confidence_threshold = float(request.form.get("confidence_threshold", 0.6))
    if not file:
        return jsonify({"detail": "No file uploaded"}), 400
    if not file.content_type.startswith("image/"):
        return jsonify({"detail": "Uploaded file is not an image"}), 400
    try:
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        if model_choice == "pretrained":
            predictions_watcher = detection_models["BoneGuardian"](image)
            predictions_master = detection_models["XRayMaster"](image)
            predictions_locator = detection_models["BoneEye"](image)
            filtered_preds = [p for p in predictions_locator if p['score'] >= confidence_threshold]
        else:
            swin_out, beit_out, medvit_out = extract_features(image, custom_components)
            result = custom_model(swin_out, beit_out, medvit_out)
            probabilities = torch.nn.functional.softmax(result, dim=1)
            pred_value = probabilities[0][1].item()
            filtered_preds = []
            if pred_value >= confidence_threshold:
                filtered_preds = [{
                    'score': pred_value,
                    'label': 'fracture',
                    'box': {
                        'xmin': 10,
                        'ymin': 10,
                        'xmax': image.width - 10,
                        'ymax': image.height - 10
                    }
                }]
        if filtered_preds:
            result_image = draw_boxes(image, filtered_preds)
        else:
            result_image = image.copy()
        result_image_b64 = image_to_base64(result_image)
        explanation = get_xray_explanation(filtered_preds if filtered_preds else [])
        session_id = str(uuid.uuid4())
        analysis_sessions[session_id] = AnalysisSession(
            image_b64=result_image_b64,
            predictions=filtered_preds,
            explanation=explanation
        )
        return jsonify({
            "result_image": result_image_b64,
            "predictions": filtered_preds,
            "explanation": explanation,
            "session_id": session_id
        })
    except Exception as e:
        return jsonify({"detail": f"Error processing image: {str(e)}"}), 500

# Chat endpoint
@app.route("/chat/", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")
    session_id = data.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        analysis_sessions[session_id] = AnalysisSession(
            image_b64="",
            predictions=[],
            explanation="No X-ray has been analyzed yet."
        )
    if session_id not in analysis_sessions:
        return jsonify({"response": "Session not found. Please upload an X-ray image to start a new session.", "session_id": ""})
    session = analysis_sessions[session_id]
    session.chat_history.append(ChatMessage(role="user", content=message))
    response_text = generate_contextual_response(message, session_id)
    session.chat_history.append(ChatMessage(role="assistant", content=response_text))
    if len(session.chat_history) > 21:
        session.chat_history = [session.chat_history[0]] + session.chat_history[-20:]
    return jsonify({"response": response_text, "session_id": session_id})

# Model details endpoint
@app.route("/models/", methods=["GET"])
def get_models():
    return jsonify({
        "available_models": [
            {
                "id": "pretrained",
                "name": "Pre-trained Models",
                "description": "Uses ensemble of pre-trained bone fracture detection models"
            },
            {
                "id": "custom",
                "name": "Custom MultiModel Detector",
                "description": "Uses a custom fusion model combining multiple feature extractors"
            }
        ]
    })

# Health check endpoint
@app.route("/health/", methods=["GET"])
def health_check():
    if detection_models and custom_components and custom_model:
        return jsonify({"status": "healthy", "models_loaded": True})
    return jsonify({"status": "unhealthy", "models_loaded": False})

# Audio transcription endpoint
@app.route("/transcribe-audio/<session_id>", methods=["POST"])
def transcribe_audio_endpoint(session_id):
    audio_file = request.files.get("audio_file")
    if not audio_file or not audio_file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
        return jsonify({"detail": "Only WAV, MP3, OGG, and FLAC files are supported"}), 400
    try:
        audio_data = audio_file.read()
        transcribed_text = transcribe_audio(audio_data)
        if session_id in analysis_sessions:
            findings = analysis_sessions[session_id].predictions
            response = get_xray_explanation([{"label": "user_question", "score": 1.0, "content": transcribed_text}])
        else:
            response = "Please upload an X-ray image first to discuss its findings."
        return jsonify({
            "transcribed_text": transcribed_text,
            "response": response
        })
    except Exception as e:
        return jsonify({"detail": f"Error processing audio: {str(e)}"}), 500

# Run the application with Flask if script is executed directly
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    with app.app_context():
        try:
            detection_models = load_detection_models()
            custom_components = load_custom_model_components()
            custom_model = MultiModelFractureDetector()
            # Load Whisper model and processor for audio transcription
            processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            stt_model = stt_model.to(device)
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    try:
        print("Starting Flask API server...")
        app.run(host="127.0.0.1", port=8080, debug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        print("Trying alternative port...")
        try:
            app.run(host="127.0.0.1", port=8000, debug=True)
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            print("Please check if another server is running on these ports (8080 or 8000)")
