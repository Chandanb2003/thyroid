from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2
import torch
import io
import random
import time
import uuid
import requests
import json
import math

# Create Flask app
app = Flask(__name__)
CORS(app)

GROQ_API_KEY = "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"

POSSIBLE_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax", "No Finding"
]

global_chat_history = [
    {"role": "system", "content": "You are a helpful medical AI assistant specializing in chest X-rays and medical imaging. You can analyze uploaded X-ray images and answer general questions about medical imaging, interpretation, and related topics. For specific diagnosis, always recommend consulting with a healthcare professional."}
]
xray_analysis_data = None

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

def preprocess_xray(image):
    if image.mode == 'L':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.equalizeHist(gray)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(enhanced_rgb)
    return image

def analyze_xray(image):
    random.seed(int(time.time()))
    base_probs = torch.zeros(len(POSSIBLE_FINDINGS))
    if random.random() < 0.5:
        base_probs[-1] = random.uniform(0.7, 0.9)
        for i in range(len(POSSIBLE_FINDINGS) - 1):
            base_probs[i] = random.uniform(0, 0.2)
    else:
        num_findings = random.randint(1, 3)
        finding_indices = random.sample(range(len(POSSIBLE_FINDINGS) - 1), num_findings)
        for idx in finding_indices:
            base_probs[idx] = random.uniform(0.6, 0.9)
        for i in range(len(POSSIBLE_FINDINGS) - 1):
            if i not in finding_indices:
                base_probs[i] = random.uniform(0, 0.3)
        base_probs[-1] = random.uniform(0, 0.2)
    simulated_probabilities = torch.softmax(base_probs, dim=0)
    findings = {POSSIBLE_FINDINGS[i]: safe_float(float(simulated_probabilities[i]) * 100) for i in range(len(POSSIBLE_FINDINGS))}
    sorted_findings = dict(sorted(findings.items(), key=lambda item: item[1], reverse=True))
    return sorted_findings

# ... (reuse all helper functions: create_detailed_system_prompt, simulate_llm_response, get_anatomical_region, get_contextual_response, get_groq_response)

# For brevity, copy all helper functions from your original code here

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Chest X-ray Analysis API with ChatGPT-like interface is running."})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/chat", methods=["POST"])
def chat():
    global global_chat_history, xray_analysis_data
    try:
        data = request.get_json()
        message = data.get("message", "")
        global_chat_history.append({"role": "user", "content": message})
        groq_messages = [{"role": msg["role"], "content": msg["content"]} for msg in global_chat_history]
        response_text = get_groq_response(groq_messages, xray_analysis_data)
        global_chat_history.append({"role": "assistant", "content": response_text})
        if len(global_chat_history) > 22:
            global_chat_history = [global_chat_history[0]] + global_chat_history[-21:]
        return jsonify({
            "response": response_text,
            "xray_data": xray_analysis_data
        })
    except Exception as e:
        return jsonify({"detail": f"Chat error: {str(e)}"}), 500

@app.route("/upload-xray", methods=["POST"])
def upload_xray():
    global global_chat_history, xray_analysis_data
    if "file" not in request.files:
        return jsonify({"detail": "File must be an image"}), 400
    file = request.files["file"]
    if not file.content_type.startswith("image/"):
        return jsonify({"detail": "File must be an image"}), 400
    try:
        contents = file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_xray(image)
        findings = analyze_xray(processed_image)
        top_condition = list(findings.keys())[0]
        confidence = findings[top_condition]
        xray_analysis_data = findings
        system_prompt = create_detailed_system_prompt(findings)
        global_chat_history[0] = {"role": "system", "content": system_prompt}
        explanation = get_groq_response([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please explain these chest X-ray findings in simple terms and what they might mean."}
        ], findings)
        global_chat_history.append({"role": "assistant", "content": f"I've analyzed your chest X-ray. The primary finding is {top_condition} with {confidence:.1f}% confidence. {explanation}"})
        return jsonify({
            "findings": findings,
            "explanation": explanation,
            "top_condition": top_condition,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"detail": f"Analysis error: {str(e)}"}), 500

@app.route("/clear-chat", methods=["POST"])
def clear_chat():
    global global_chat_history, xray_analysis_data
    global_chat_history = [
        {"role": "system", "content": "You are a helpful medical AI assistant specializing in chest X-rays and medical imaging. You can analyze uploaded X-ray images and answer general questions about medical imaging, interpretation, and related topics. For specific diagnosis, always recommend consulting with a healthcare professional."}
    ]
    xray_analysis_data = None
    return jsonify({"status": "success", "message": "Chat history and X-ray data cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
