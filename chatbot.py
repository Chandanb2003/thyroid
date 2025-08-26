import os
import tempfile
import faiss
import numpy as np
import hashlib
import uvicorn
import uuid
from typing import TypedDict, Dict, Any, List, Optional
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from transformers import pipeline
import soundfile as sf
import io
import torch

app = Flask(__name__)
CORS(app)

os.environ["GROQ_API_KEY"] = "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"
GROQ_MODEL = "llama3-70b-8192"

device = "cuda" if torch.cuda.is_available() else "cpu"
stt_model = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device
)

sessions = {}

class AgentState(TypedDict):
    messages: List[BaseMessage]
    sentiment: str
    task_type: str
    pdf_store: object

def get_or_create_session(session_id: str) -> Dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "conversation_history": [],
            "pdf_store": initialize_pdf_vector_store(),
            "has_uploaded_pdf": False
        }
    return sessions[session_id]

def simple_text_to_vector(text: str, dimension: int = 100) -> np.ndarray:
    hash_object = hashlib.md5(text.encode())
    hash_hex = hash_object.hexdigest()
    vector = np.zeros(dimension)
    for i in range(min(dimension, len(hash_hex))):
        vector[i] = int(hash_hex[i], 16) / 16.0
    return vector

def initialize_pdf_vector_store():
    dimension = 100
    index = faiss.IndexFlatL2(dimension)
    return {
        "index": index,
        "documents": [],
        "embeddings": []
    }

def add_pdf_to_vector_store(pdf_path, pdf_store):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for doc in docs:
        embedding = simple_text_to_vector(doc.page_content)
        pdf_store['index'].add(np.array([embedding]))
        pdf_store['documents'].append(doc)
        pdf_store['embeddings'].append(embedding)
    return pdf_store

def similarity_search_pdf(query, pdf_store, top_k=3):
    query_embedding = simple_text_to_vector(query)
    D, I = pdf_store['index'].search(np.array([query_embedding]), top_k)
    similar_docs = [pdf_store['documents'][i] for i in I[0]]
    return " ".join([doc.page_content for doc in similar_docs])

def create_memory_enhanced_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful healthcare assistant. Consider the entire conversation history when responding."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])

def sentiment_analyzer(state: AgentState) -> Dict[str, Any]:
    try:
        llm = ChatGroq(model=GROQ_MODEL)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the sentiment of the following message. Classify it as 'positive', 'negative', or 'neutral'."),
            ("human", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()
        last_message = state['messages'][-1].content
        sentiment = chain.invoke({"input": last_message})
        return {
            "sentiment": sentiment,
            "task_type": None,
            "messages": state['messages']
        }
    except Exception as e:
        print(f"Sentiment Analysis Error: {e}")
        return {
            "sentiment": "neutral",
            "task_type": None,
            "messages": state['messages']
        }

def task_router(state: AgentState) -> Dict[str, Any]:
    session_id = getattr(state, 'session_id', None)
    task_type = "general_agent"
    if session_id and sessions[session_id].get('has_uploaded_pdf'):
        task_type = "pdf_agent"
    elif any(word in state['messages'][-1].content.lower() for word in ["symptom", "disease", "condition", "medical"]):
        task_type = "medical_query_agent"
    return {
        "task_type": task_type,
        "sentiment": state['sentiment'],
        "messages": state['messages']
    }

def create_memory_aware_agent(agent_type: str):
    def agent_func(state: AgentState) -> Dict[str, Any]:
        try:
            llm = ChatGroq(model=GROQ_MODEL)
            prompt = create_memory_enhanced_prompt()
            session_id = getattr(state, 'session_id', None)
            if agent_type == "medical":
                system_message = "You are a helpful medical information assistant. Provide clear, general medical information. IMPORTANT: Always advise consulting a doctor for specific medical concerns."
            elif agent_type == "pdf" and session_id:
                pdf_context = similarity_search_pdf(
                    state['messages'][-1].content, 
                    sessions[session_id]['pdf_store']
                )
                system_message = f"You are an expert at analyzing medical PDFs. Use the following context to answer the query: {pdf_context}. If the information suggests a serious condition, strongly advise seeing a doctor."
            else:
                system_message = "You are a helpful assistant providing general health and wellness information."
            modified_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{input}")
            ])
            chain = modified_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "messages": state['messages'],
                "input": state['messages'][-1].content
            })
            if agent_type == "medical" and state.get('sentiment', '') == 'negative':
                response += "\n\nI sense you're worried. Please remember that while I can provide general information, it's crucial to consult a healthcare professional for personalized medical advice."
            if agent_type == "pdf" and any(word in response.lower() for word in ["serious", "critical", "urgent", "immediate attention"]):
                response += "\n\n⚠️ IMPORTANT: The information suggests a potentially serious medical condition. Please consult a healthcare professional immediately."
            new_messages = state['messages'] + [AIMessage(content=response)]
            return {
                "messages": new_messages,
                "sentiment": state['sentiment'],
                "task_type": None
            }
        except Exception as e:
            print(f"{agent_type.capitalize()} Agent Error: {e}")
            error_message = f"I'm sorry, but I encountered an error processing your {agent_type} query."
            new_messages = state['messages'] + [AIMessage(content=error_message)]
            return {
                "messages": new_messages,
                "sentiment": state['sentiment'],
                "task_type": None
            }
    return agent_func

workflow = StateGraph(AgentState)
workflow.add_node("sentiment_analyzer", sentiment_analyzer)
workflow.add_node("task_router", task_router)
workflow.add_node("medical_query_agent", create_memory_aware_agent("medical"))
workflow.add_node("pdf_agent", create_memory_aware_agent("pdf"))
workflow.add_node("general_agent", create_memory_aware_agent("general"))
workflow.set_entry_point("sentiment_analyzer")
workflow.add_edge("sentiment_analyzer", "task_router")
workflow.add_conditional_edges(
    "task_router",
    lambda state: state["task_type"],
    {
        "medical_query_agent": "medical_query_agent",
        "pdf_agent": "pdf_agent",
        "general_agent": "general_agent"
    }
)
workflow.add_edge("medical_query_agent", END)
workflow.add_edge("pdf_agent", END)
workflow.add_edge("general_agent", END)
graph_app = workflow.compile()

@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Healthcare Chatbot API is running"})

@app.route("/sessions/", methods=["POST"])
def create_session():
    session_id = str(uuid.uuid4())
    get_or_create_session(session_id)
    return jsonify({"session_id": session_id, "message": "Session created successfully"})

@app.route("/chat/", methods=["POST"])
def chat():
    data = request.get_json()
    session_id = data.get("session_id")
    user_message = data.get("message")
    if not session_id or not user_message:
        return jsonify({"error": "session_id and message are required"}), 400
    session = get_or_create_session(session_id)
    session["conversation_history"].append({"role": "user", "content": user_message})
    try:
        messages = [
            HumanMessage(content=msg["content"]) 
            if msg["role"] == "user" 
            else AIMessage(content=msg["content"]) 
            for msg in session["conversation_history"]
        ]
        inputs = {
            "messages": messages, 
            "sentiment": None, 
            "task_type": None,
            "session_id": session_id
        }
        final_response = graph_app.invoke(inputs)
        bot_response = final_response['messages'][-1].content
        session["conversation_history"].append({"role": "assistant", "content": bot_response})
        return jsonify({"session_id": session_id, "response": bot_response})
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        session["conversation_history"].append({"role": "assistant", "content": error_msg})
        return jsonify({"error": error_msg}), 500

@app.route("/upload-pdf/<session_id>", methods=["POST"])
def upload_pdf(session_id):
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are accepted"}), 400
    session = get_or_create_session(session_id)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            file.save(tmp_file)
            tmp_file_path = tmp_file.name
        session["pdf_store"] = add_pdf_to_vector_store(tmp_file_path, session["pdf_store"])
        session["has_uploaded_pdf"] = True
        os.unlink(tmp_file_path)
        return jsonify({"session_id": session_id, "message": "PDF uploaded and indexed successfully"})
    except Exception as e:
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

@app.route("/history/<session_id>", methods=["GET"])
def get_conversation_history(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"session_id": session_id, "history": sessions[session_id]["conversation_history"]})

@app.route("/config/model", methods=["PUT"])
def update_model_config():
    data = request.get_json()
    model = data.get("model")
    global GROQ_MODEL
    valid_models = ["llama3-8b-8192", "llama3-70b-8192", "gemma-7b-it"]
    if model not in valid_models:
        return jsonify({"error": f"Invalid model. Choose from: {', '.join(valid_models)}"}), 400
    GROQ_MODEL = model
    return jsonify({"message": f"Model updated to {GROQ_MODEL}"})

@app.route("/transcribe-audio/<session_id>", methods=["POST"])
def transcribe_audio(session_id):
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file part"}), 400
    audio_file = request.files["audio_file"]
    if not audio_file.filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
        return jsonify({"error": "Only WAV, MP3, OGG, and FLAC files are supported"}), 400
    session = get_or_create_session(session_id)
    try:
        audio_data = audio_file.read()
        with io.BytesIO(audio_data) as audio_buffer:
            audio_array, sample_rate = sf.read(audio_buffer)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)
            result = stt_model({"raw": audio_array, "sampling_rate": sample_rate})
            transcribed_text = result["text"]
            session["conversation_history"].append({"role": "user", "content": transcribed_text})
            messages = [
                HumanMessage(content=msg["content"]) 
                if msg["role"] == "user" 
                else AIMessage(content=msg["content"]) 
                for msg in session["conversation_history"]
            ]
            inputs = {
                "messages": messages, 
                "sentiment": None, 
                "task_type": None,
                "session_id": session_id
            }
            final_response = graph_app.invoke(inputs)
            bot_response = final_response['messages'][-1].content
            session["conversation_history"].append({"role": "assistant", "content": bot_response})
            return jsonify({
                "session_id": session_id,
                "transcribed_text": transcribed_text,
                "response": bot_response
            })
    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
