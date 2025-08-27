from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

CORS_ORIGINS = ["*"]  # Change to your frontend domain(s) in production
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

# Load dataset
try:
    df = pd.read_csv(r"D:\nerveo\creation of api\archive\AyurGenixAI_Dataset.csv")
except FileNotFoundError:
    raise FileNotFoundError("AyurGenixAI_Dataset.csv not found. Please ensure the file is in the correct directory.")


# Select required columns and create a copy to avoid SettingWithCopyWarning
selected_columns = [
    'Disease', 'Hindi Name', 'Symptoms', 'Dietary Habits', 'Age Group',
    'Gender', 'Herbal/Alternative Remedies', 'Ayurvedic Herbs',
    'Formulation', 'Yoga & Physical Therapy', 'Patient Recommendations'
]
df_filtered = df[selected_columns].copy()

# Combine into text
def combine_fields(row):
    return "\n".join([
        f"Disease: {row['Disease']}",
        f"Hindi Name: {row['Hindi Name']}",
        f"Symptoms: {row['Symptoms']}",
        f"Dietary Habits: {row['Dietary Habits']}",
        f"Age Group: {row['Age Group']}",
        f"Gender: {row['Gender']}",
        f"Herbal Remedies: {row['Herbal/Alternative Remedies']}",
        f"Ayurvedic Herbs: {row['Ayurvedic Herbs']}",
        f"Formulation: {row['Formulation']}",
        f"Yoga & Physical Therapy: {row['Yoga & Physical Therapy']}",
        f"Patient Recommendations: {row['Patient Recommendations']}"
    ])

df_filtered.loc[:, "combined_text"] = df_filtered.apply(combine_fields, axis=1)
docs = [Document(page_content=text) for text in df_filtered["combined_text"]]

# Text splitting
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# Embedding model (requires `pip install fastembed`)
embedding_model = FastEmbedEmbeddings()

# Load or create vectorstore
persist_dir = "./ayurveda_chroma"
try:
    if os.path.exists(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        vectorstore = Chroma.from_documents(split_docs, embedding=embedding_model, persist_directory=persist_dir)
        vectorstore.persist()
except Exception as e:
    raise Exception(f"Failed to initialize vectorstore: {str(e)}")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LLM setup
if "GROQ_API_KEY" not in os.environ:
    raise EnvironmentError("GROQ_API_KEY environment variable not set. Please set it before running the application.")

llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)

# Prompt template
custom_prompt = PromptTemplate.from_template("""
You are an expert Ayurvedic doctor. Based on the symptoms described by the user, retrieve relevant Ayurvedic remedies and instructions from the context provided below. Suggest only safe and traditional remedies. Always include:

- Remedy name(s)
- Ingredients (if available)
- How to prepare and take them (dosage, timing, precautions)

If the symptoms are not present in the document or the user's query is unrelated to the document, answer based on your own knowledge.

If the symptoms seem serious, politely recommend consulting a certified Ayurvedic practitioner.

Context:
{context}

User Symptoms:
{question}
""")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)


# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# List all diseases
@app.route('/diseases', methods=['GET'])
def list_diseases():
    try:
        diseases = df_filtered['Disease'].dropna().unique().tolist()
        return jsonify({"diseases": diseases})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch diseases: {str(e)}"}), 500

# Get detailed info for a specific disease
@app.route('/remedy_info', methods=['GET'])
def remedy_info():
    disease = request.args.get('disease', '').strip()
    if not disease:
        return jsonify({"error": "Please provide a disease name as a query parameter."}), 400
    try:
        row = df_filtered[df_filtered['Disease'].str.lower() == disease.lower()]
        if row.empty:
            return jsonify({"error": f"No information found for disease: {disease}"}), 404
        info = row.iloc[0].to_dict()
        return jsonify({"remedy_info": info})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch remedy info: {str(e)}"}), 500

# Main remedy endpoint (improved error handling)
@app.route('/get_remedy', methods=['POST'])
def get_remedy():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400
    try:
        data = request.get_json()
        query = data.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return jsonify({"error": "Please provide symptom details as a non-empty string in 'query'."}), 400
        response = rag_chain.invoke(query)
        return jsonify({
            "query": query,
            "advice": response["result"],
            "sources": [doc.page_content[:200] for doc in response["source_documents"]]
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)